#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from build_runtime_schedule import build_schedule, validate_schedule  # noqa: E402


class RuntimeScheduleTest(unittest.TestCase):
    def make_plan(self, tile_count: int = 6, tile_bytes: int = 1024) -> dict:
        return {
            "schema": "tensorwave.execution-plan.v1",
            "geometry": {
                "tile_count": tile_count,
                "tile_bytes": tile_bytes,
                "pack_bytes": tile_count * tile_bytes,
            },
            "tiles": [
                {
                    "tile_id": i,
                    "pack_offset": i * tile_bytes,
                    "nbytes": tile_bytes,
                    "tensor_name": f"layer.{i}.weight",
                }
                for i in range(tile_count)
            ],
        }

    def test_two_slot_schedule_has_exact_reuse_guards(self) -> None:
        schedule = build_schedule(self.make_plan(), slots=2)
        validate_schedule(schedule)

        ops = schedule["operations"]
        self.assertEqual([op["slot"] for op in ops], [0, 1, 0, 1, 0, 1])
        self.assertEqual(ops[0]["copy"]["wait_for"], [])
        self.assertEqual(ops[1]["copy"]["wait_for"], [])
        self.assertEqual(ops[2]["copy"]["wait_for"], ["compute:0:done"])
        self.assertEqual(ops[3]["copy"]["wait_for"], ["compute:1:done"])
        self.assertEqual(ops[4]["copy"]["wait_for"], ["compute:2:done"])
        self.assertEqual(ops[5]["copy"]["wait_for"], ["compute:3:done"])

        for i, op in enumerate(ops):
            self.assertEqual(op["compute"]["wait_for"], [f"copy:{i}:done"])

    def test_three_slot_schedule_generalizes(self) -> None:
        schedule = build_schedule(self.make_plan(tile_count=7), slots=3)
        validate_schedule(schedule)

        ops = schedule["operations"]
        self.assertEqual([op["slot"] for op in ops], [0, 1, 2, 0, 1, 2, 0])
        self.assertEqual(ops[3]["copy"]["wait_for"], ["compute:0:done"])
        self.assertEqual(ops[6]["copy"]["wait_for"], ["compute:3:done"])

    def test_invalid_pack_offset_is_rejected(self) -> None:
        plan = self.make_plan()
        plan["tiles"][3]["pack_offset"] += 1
        with self.assertRaises(ValueError):
            build_schedule(plan, slots=2)

    def test_missing_reuse_dependency_is_rejected(self) -> None:
        schedule = build_schedule(self.make_plan(), slots=2)
        schedule["operations"][2]["copy"]["wait_for"] = []
        with self.assertRaises(ValueError):
            validate_schedule(schedule)


if __name__ == "__main__":
    unittest.main()
