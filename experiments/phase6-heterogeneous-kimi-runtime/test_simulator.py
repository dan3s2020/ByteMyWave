import importlib.util
import sys
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "heterogeneous_kimi_sim",
    HERE / "simulate_heterogeneous_kimi.py",
)
sim = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = sim
SPEC.loader.exec_module(sim)


class TestHeterogeneousKimiSimulator(unittest.TestCase):
    def setUp(self):
        self.hw = sim.Hardware()
        self.k25 = sim.MODELS["k2.5"]
        self.k3 = sim.MODELS["k3"]
        self.q4 = sim.FORMATS["tw_q4_g32_f32s"]
        self.q2 = sim.FORMATS["q2_g64_f16s"]

    def test_k25_routed_parameter_derivation(self):
        self.assertAlmostEqual(self.k25.routed_active_b, 21.13929216, places=8)
        self.assertAlmostEqual(self.k25.non_routed_active_b, 10.86070784, places=8)

    def test_k25_current_q4_stream_all_is_point_six_tps(self):
        r = sim.stream_case(self.k25, self.q4, self.hw, False)
        self.assertAlmostEqual(r["stream_gb_per_token"], 20.0, places=8)
        self.assertAlmostEqual(r["bandwidth_ceiling_tps"], 0.6, places=8)
        self.assertAlmostEqual(r["combined_roofline_tps"], 0.6, places=8)

    def test_k25_static_residency_q4(self):
        r = sim.stream_case(self.k25, self.q4, self.hw, True)
        self.assertAlmostEqual(r["stream_gb_per_token"], 13.2120576, places=6)
        self.assertAlmostEqual(
            r["bandwidth_ceiling_tps"], 12.0 / 13.2120576, places=8
        )

    def test_k25_q2_static_residency(self):
        r = sim.stream_case(self.k25, self.q2, self.hw, True)
        self.assertAlmostEqual(r["stream_gb_per_token"], 5.94542592, places=6)
        self.assertAlmostEqual(
            r["bandwidth_ceiling_tps"], 12.0 / 5.94542592, places=8
        )

    def test_k25_four_socket_q4_thresholds(self):
        r5 = sim.cpu_expert_requirement(self.k25, self.q4, self.hw, 5.0)
        r10 = sim.cpu_expert_requirement(self.k25, self.q4, self.hw, 10.0)
        self.assertAlmostEqual(
            r5["required_mem_bw_gbps_per_socket"], 16.515072, places=6
        )
        self.assertAlmostEqual(
            r10["required_mem_bw_gbps_per_socket"], 33.030144, places=6
        )
        self.assertAlmostEqual(
            r5["required_gflops_per_socket"], 52.8482304, places=6
        )
        self.assertAlmostEqual(
            r10["required_gflops_per_socket"], 105.6964608, places=6
        )

    def test_four_sockets_halve_per_socket_work_vs_two(self):
        hw2 = sim.Hardware(cpu_sockets=2)
        r2 = sim.cpu_expert_requirement(self.k25, self.q4, hw2, 5.0)
        r4 = sim.cpu_expert_requirement(self.k25, self.q4, self.hw, 5.0)
        self.assertAlmostEqual(
            r2["required_mem_bw_gbps_per_socket"],
            2.0 * r4["required_mem_bw_gbps_per_socket"],
            places=8,
        )

    def test_k3_compression_alone_five_tps_requires_below_point_two_bits(self):
        bits = sim.compression_only_required_bits(self.k3, self.hw, 5.0)
        self.assertAlmostEqual(bits, 0.1846153846, places=8)
        self.assertLess(bits, 0.2)

    def test_k3_q4_capacity_in_two_tib(self):
        hw2tib = sim.Hardware(host_ram_gib=2048.0)
        storage_gb = sim.model_storage_gb(self.k3, self.q4)
        self.assertAlmostEqual(storage_gb, 1750.0, places=8)
        self.assertLess(storage_gb * 1e9, hw2tib.host_ram_gib * 1024**3)

    def test_hypothetical_full_vram_uses_vram_bandwidth_not_only_compute(self):
        r = sim.hypothetical_full_vram(self.k3, self.q4, self.hw)
        self.assertAlmostEqual(
            r["vram_bandwidth_ceiling_tps"], 360.0 / 65.0, places=8
        )
        self.assertGreater(
            r["compute_only_ceiling_tps"], r["vram_bandwidth_ceiling_tps"]
        )
        self.assertAlmostEqual(
            r["combined_roofline_tps"], 360.0 / 65.0, places=8
        )

    def test_activation_volume_is_mb_not_gb(self):
        r = sim.cpu_expert_requirement(self.k25, self.q4, self.hw, 5.0)
        self.assertAlmostEqual(
            r["activation_roundtrip_mb_per_token_upper_bound"], 6.88128, places=5
        )
        self.assertLess(r["activation_pcie_gbps_at_target"], 0.04)


if __name__ == "__main__":
    unittest.main()
