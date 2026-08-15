#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from safetensors_atlas import build_atlas  # noqa: E402


def write_test_safetensors(path: Path) -> dict[str, bytes]:
    tensors = {
        "layer0.weight": bytes(range(0, 24)),
        "layer1.weight": bytes(range(24, 48)),
    }

    header = {
        "layer0.weight": {
            "dtype": "F16",
            "shape": [4, 3],
            "data_offsets": [0, 24],
        },
        "layer1.weight": {
            "dtype": "F16",
            "shape": [4, 3],
            "data_offsets": [24, 48],
        },
        "__metadata__": {"purpose": "TensorWave parser test"},
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_bytes)))
        handle.write(header_bytes)
        for payload in tensors.values():
            handle.write(payload)

    return tensors


class SafetensorsToolsTest(unittest.TestCase):
    def test_atlas_and_packer_preserve_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir = root / "model"
            output_dir = root / "prepared"
            model_dir.mkdir()

            shard = model_dir / "model-00001-of-00001.safetensors"
            tensors = write_test_safetensors(shard)

            atlas = build_atlas(model_dir)
            self.assertEqual(atlas["summary"]["shard_count"], 1)
            self.assertEqual(atlas["summary"]["tensor_count"], 2)
            self.assertEqual(atlas["summary"]["tensor_payload_bytes"], 48)
            self.assertEqual(atlas["summary"]["size_mismatch_count"], 0)

            records = {entry["name"]: entry for entry in atlas["tensors"]}
            self.assertEqual(records["layer0.weight"]["shape"], [4, 3])
            self.assertEqual(records["layer1.weight"]["shape"], [4, 3])
            self.assertEqual(records["layer0.weight"]["nbytes"], 24)
            self.assertEqual(records["layer1.weight"]["nbytes"], 24)

            subprocess.run(
                [
                    sys.executable,
                    str(TOOLS / "pack_stream_tiles.py"),
                    "--model-dir",
                    str(model_dir),
                    "--output-dir",
                    str(output_dir),
                    "--dtype",
                    "F16",
                    "--k",
                    "3",
                    "--tile-n",
                    "2",
                    "--max-tiles",
                    "4",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            packed = (output_dir / "weights.pack").read_bytes()
            expected = tensors["layer0.weight"] + tensors["layer1.weight"]
            self.assertEqual(packed, expected)

            plan = json.loads((output_dir / "execution-plan.json").read_text())
            self.assertEqual(plan["geometry"]["dtype"], "F16")
            self.assertEqual(plan["geometry"]["k"], 3)
            self.assertEqual(plan["geometry"]["n"], 2)
            self.assertEqual(plan["geometry"]["tile_count"], 4)
            self.assertEqual(plan["geometry"]["tile_bytes"], 12)
            self.assertEqual(plan["geometry"]["pack_bytes"], 48)

            for tile in plan["tiles"]:
                offset = tile["pack_offset"]
                nbytes = tile["nbytes"]
                payload = packed[offset : offset + nbytes]
                self.assertEqual(hashlib.sha256(payload).hexdigest(), tile["sha256"])

            self.assertEqual(plan["tiles"][0]["tensor_name"], "layer0.weight")
            self.assertEqual(plan["tiles"][0]["row_start"], 0)
            self.assertEqual(plan["tiles"][1]["row_start"], 2)
            self.assertEqual(plan["tiles"][2]["tensor_name"], "layer1.weight")
            self.assertEqual(plan["tiles"][3]["row_start"], 2)


if __name__ == "__main__":
    unittest.main()
