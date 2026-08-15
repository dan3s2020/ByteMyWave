import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "calibrate_feasibility_map.py"
SPEC = importlib.util.spec_from_file_location("tensorwave_calibration", MODULE_PATH)
MOD = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


class FeasibilityCalibrationTests(unittest.TestCase):
    def write_result(self, directory: Path, name: str, payload: dict) -> None:
        (directory / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_phase3_prefers_compressed_h2d_and_gemm_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.write_result(
                run,
                "m-64.json",
                {
                    "experiment": "phase3-q4",
                    "geometry": {"m": 64, "k": 1024, "n": 256, "tiles": 8},
                    "overlapped": {
                        "compressed_h2d_gbps": 11.5,
                        "source_equivalent_h2d_gbps": 36.8,
                        "gemm_ms": 4.0,
                        "compute_ms": 5.0,
                        "dequant_ms": 1.0,
                        "steady_starvation_pct": 12.0,
                        "steady_hidden_transfer_pct": 75.0,
                    },
                    "correctness_ok": True,
                },
            )
            payload = MOD.collect(run)
            self.assertAlmostEqual(payload["effective_h2d_gbps_median"], 11.5)
            expected_tflops = (2.0 * 64 * 1024 * 256 * 8) / (4.0 / 1000.0) / 1.0e12
            self.assertAlmostEqual(payload["effective_tflops_median"], expected_tflops)
            self.assertAlmostEqual(payload["samples"][0]["source_equivalent_h2d_gbps"], 36.8)

    def test_phase2_falls_back_to_h2d_and_compute(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.write_result(
                run,
                "m-128.json",
                {
                    "experiment": "phase2-real",
                    "geometry": {"m": 128, "k": 512, "n": 128, "tiles": 4},
                    "overlapped": {
                        "h2d_gbps": 10.25,
                        "compute_ms": 2.0,
                        "steady_starvation_pct": 4.0,
                        "steady_hidden_transfer_pct": 92.0,
                    },
                    "correctness_ok": True,
                },
            )
            payload = MOD.collect(run)
            self.assertAlmostEqual(payload["effective_h2d_gbps_median"], 10.25)
            self.assertEqual(payload["correct_h2d_samples"], 1)
            self.assertEqual(payload["correct_compute_samples"], 1)

    def test_incorrect_samples_do_not_calibrate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            self.write_result(
                run,
                "m-1.json",
                {
                    "geometry": {"m": 1, "k": 32, "n": 32, "tiles": 2},
                    "overlapped": {"h2d_gbps": 12.0, "compute_ms": 1.0},
                    "correctness_ok": False,
                },
            )
            with self.assertRaises(ValueError):
                MOD.collect(run)


if __name__ == "__main__":
    unittest.main()
