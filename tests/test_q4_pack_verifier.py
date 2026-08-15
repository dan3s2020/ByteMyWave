#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from quantize_q4_pack import GROUP_BYTES, GROUP_SIZE, quantize_values  # noqa: E402
from verify_q4_pack import verify  # noqa: E402


class Q4PackVerifierTest(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[dict, Path]:
        values0 = np.linspace(-1.0, 1.0, GROUP_SIZE, dtype=np.float32)
        values1 = np.linspace(-2.0, 2.0, GROUP_SIZE, dtype=np.float32)
        q0, _ = quantize_values(values0)
        q1, _ = quantize_values(values1)
        payload = q0 + q1

        pack_path = root / "weights-q4.pack"
        pack_path.write_bytes(payload)

        plan = {
            "schema": "tensorwave.q4-plan.v1",
            "quantization": {
                "name": "Q4_SYM_G32_F32S",
                "group_size": 32,
                "group_bytes": 20,
            },
            "geometry": {
                "tile_count": 2,
                "tile_elements": 32,
                "q4_tile_bytes": GROUP_BYTES,
                "q4_pack_bytes": len(payload),
                "source_tile_bytes": 64,
            },
            "tiles": [
                {
                    "tile_id": 0,
                    "q4_pack_offset": 0,
                    "q4_nbytes": GROUP_BYTES,
                    "q4_sha256": hashlib.sha256(q0).hexdigest(),
                },
                {
                    "tile_id": 1,
                    "q4_pack_offset": GROUP_BYTES,
                    "q4_nbytes": GROUP_BYTES,
                    "q4_sha256": hashlib.sha256(q1).hexdigest(),
                },
            ],
        }
        return plan, pack_path

    def test_valid_pack_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            plan, pack = self.make_fixture(Path(temp))
            verify(plan, pack)

    def test_single_byte_corruption_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            plan, pack = self.make_fixture(Path(temp))
            payload = bytearray(pack.read_bytes())
            payload[-1] ^= 0x01
            pack.write_bytes(payload)
            with self.assertRaises(ValueError):
                verify(plan, pack)

    def test_non_contiguous_offset_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            plan, pack = self.make_fixture(Path(temp))
            plan["tiles"][1]["q4_pack_offset"] += 1
            with self.assertRaises(ValueError):
                verify(plan, pack)


if __name__ == "__main__":
    unittest.main()
