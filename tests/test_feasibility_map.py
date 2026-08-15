import importlib.util
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "build_feasibility_map.py"
SPEC = importlib.util.spec_from_file_location("tensorwave_feasibility", MODULE_PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


class FeasibilityMapTests(unittest.TestCase):
    def setUp(self):
        self.inputs = MOD.Inputs(
            pcie_gbps=12.0,
            effective_tflops=10.0,
            bytes_per_param=0.625,
            resident_fraction=0.0,
            active_fraction=1.0,
        )

    def test_crossover_formula(self):
        expected = 0.625 * 10.0e12 / (2.0 * 12.0e9)
        self.assertAlmostEqual(MOD.crossover_m(self.inputs), expected, places=9)
        self.assertAlmostEqual(expected, 260.4166666666667, places=6)

    def test_model_size_cancels_from_overlap_ratio(self):
        small = MOD.make_cell(7.0, 64, self.inputs)
        large = MOD.make_cell(70.0, 64, self.inputs)
        self.assertAlmostEqual(small.hidden_transfer_pct, large.hidden_transfer_pct, places=9)
        self.assertAlmostEqual(large.transfer_ms / small.transfer_ms, 10.0, places=9)
        self.assertAlmostEqual(large.compute_ms / small.compute_ms, 10.0, places=9)

    def test_residency_reduces_crossover(self):
        half_resident = MOD.Inputs(
            pcie_gbps=12.0,
            effective_tflops=10.0,
            bytes_per_param=0.625,
            resident_fraction=0.5,
            active_fraction=1.0,
        )
        self.assertAlmostEqual(
            MOD.crossover_m(half_resident),
            0.5 * MOD.crossover_m(self.inputs),
            places=9,
        )

    def test_active_fraction_changes_absolute_time_but_not_ideal_crossover(self):
        moe = MOD.Inputs(
            pcie_gbps=12.0,
            effective_tflops=10.0,
            bytes_per_param=0.625,
            resident_fraction=0.0,
            active_fraction=0.25,
        )
        dense = MOD.make_cell(70.0, 64, self.inputs)
        sparse = MOD.make_cell(70.0, 64, moe)
        self.assertAlmostEqual(MOD.crossover_m(moe), MOD.crossover_m(self.inputs), places=9)
        self.assertAlmostEqual(sparse.transfer_ms, dense.transfer_ms * 0.25, places=9)
        self.assertAlmostEqual(sparse.compute_ms, dense.compute_ms * 0.25, places=9)

    def test_q4_v1_70b_transfer_floor_at_15_gbps(self):
        inputs = MOD.Inputs(
            pcie_gbps=15.0,
            effective_tflops=10.0,
            bytes_per_param=0.625,
            resident_fraction=0.0,
            active_fraction=1.0,
        )
        cell = MOD.make_cell(70.0, 1, inputs)
        self.assertAlmostEqual(cell.stream_gb, 43.75, places=6)
        self.assertAlmostEqual(cell.transfer_ms, 2916.6666666666665, places=6)


if __name__ == "__main__":
    unittest.main()
