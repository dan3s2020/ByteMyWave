import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "simulate_r920_tensorwave.py"
SPEC = importlib.util.spec_from_file_location("tensorwave_r920_sim", MODULE_PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


class R920SimulationTests(unittest.TestCase):
    def test_q4_wire_density_matches_phase3(self):
        self.assertAlmostEqual(MOD.Q4_BYTES_PER_PARAM, 0.625, places=12)

    def test_default_crossover_matches_phase4_equation(self):
        value = MOD.crossover_m(
            bytes_per_param=0.625,
            resident_fraction=0.0,
            h2d_gbps=12.0,
            effective_tflops=10.0,
        )
        self.assertAlmostEqual(value, 260.4166666666667, places=9)

    def test_70b_q4_transfer_floor(self):
        cell = MOD.roofline(
            model_b=70.0,
            m=1,
            bytes_per_param=0.625,
            resident_fraction=0.0,
            h2d_gbps=12.0,
            effective_tflops=10.0,
        )
        self.assertAlmostEqual(cell.stream_gb_per_step, 43.75, places=9)
        self.assertAlmostEqual(cell.transfer_ms, 3645.8333333333335, places=9)

    def test_two_slot_ring_reproduces_expected_overlap_near_crossover(self):
        cell = MOD.ring_simulation(
            m=256,
            k=8192,
            n=256,
            tiles=32,
            h2d_gbps=12.0,
            effective_tflops=10.0,
            dequant_us_per_tile=0.0,
        )
        self.assertAlmostEqual(cell.q4_tile_mib, 1.25, places=9)
        self.assertAlmostEqual(cell.fixed_vram_mib, 10.75, places=9)
        self.assertAlmostEqual(cell.hidden_transfer_pct, 98.304, places=6)
        self.assertLess(cell.steady_starvation_pct, 2.0)

    def test_compute_bound_after_crossover(self):
        cell = MOD.ring_simulation(
            m=512,
            k=8192,
            n=256,
            tiles=32,
            h2d_gbps=12.0,
            effective_tflops=10.0,
            dequant_us_per_tile=0.0,
        )
        self.assertEqual(cell.hidden_transfer_pct, 100.0)
        self.assertEqual(cell.steady_starvation_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
