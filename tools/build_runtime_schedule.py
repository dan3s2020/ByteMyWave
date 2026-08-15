#!/usr/bin/env python3
"""Compile an execution-plan into an explicit fixed-slot TensorWave schedule.

This is intentionally a static compiler, not a runtime predictor. Given an
ordered tile list and a fixed number of VRAM slots, it emits every copy/compute
resource dependency before inference begins. That makes the memory choreography
reviewable and later suitable for CUDA-Graph style capture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TensorWave static runtime schedule.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slots", type=int, default=2)
    return parser.parse_args()


def validate_plan(plan: dict[str, Any]) -> None:
    if plan.get("schema") != "tensorwave.execution-plan.v1":
        raise ValueError("unsupported/missing execution-plan schema")
    geometry = plan.get("geometry")
    tiles = plan.get("tiles")
    if not isinstance(geometry, dict) or not isinstance(tiles, list):
        raise ValueError("plan must contain geometry object and tiles array")
    if int(geometry.get("tile_count", -1)) != len(tiles):
        raise ValueError("geometry.tile_count does not match tiles array")

    expected_pack_offset = 0
    tile_bytes = int(geometry.get("tile_bytes", 0))
    if tile_bytes <= 0:
        raise ValueError("geometry.tile_bytes must be > 0")

    for index, tile in enumerate(tiles):
        if int(tile.get("tile_id", -1)) != index:
            raise ValueError(f"tile IDs must be contiguous; expected {index}")
        if int(tile.get("pack_offset", -1)) != expected_pack_offset:
            raise ValueError(f"tile {index}: pack offset is not contiguous")
        if int(tile.get("nbytes", -1)) != tile_bytes:
            raise ValueError(f"tile {index}: nbytes differs from geometry.tile_bytes")
        expected_pack_offset += tile_bytes

    if int(geometry.get("pack_bytes", -1)) != expected_pack_offset:
        raise ValueError("geometry.pack_bytes does not match tile ranges")


def build_schedule(plan: dict[str, Any], slots: int) -> dict[str, Any]:
    if slots < 2:
        raise ValueError("at least two VRAM slots are required for overlap")
    validate_plan(plan)

    tiles: list[dict[str, Any]] = plan["tiles"]
    operations: list[dict[str, Any]] = []

    for i, tile in enumerate(tiles):
        slot = i % slots
        previous_owner = i - slots if i >= slots else None

        copy_dependencies: list[str] = []
        if previous_owner is not None:
            # The slot cannot be overwritten until the previous GEMM that used
            # this physical VRAM address has completed.
            copy_dependencies.append(f"compute:{previous_owner}:done")

        copy_done = f"copy:{i}:done"
        compute_done = f"compute:{i}:done"

        operations.append(
            {
                "sequence": i,
                "tile_id": int(tile["tile_id"]),
                "slot": slot,
                "pack_offset": int(tile["pack_offset"]),
                "nbytes": int(tile["nbytes"]),
                "tensor_name": tile.get("tensor_name"),
                "copy": {
                    "stream": "h2d",
                    "wait_for": copy_dependencies,
                    "signal": copy_done,
                },
                "compute": {
                    "stream": "compute",
                    "wait_for": [copy_done],
                    "signal": compute_done,
                },
                "slot_previous_owner_tile": previous_owner,
            }
        )

    return {
        "schema": "tensorwave.runtime-schedule.v1",
        "source_plan_schema": plan["schema"],
        "slots": slots,
        "tile_count": len(tiles),
        "policy": {
            "slot_assignment": "tile_id % slots",
            "copy_order": "execution-plan order on one H2D stream",
            "compute_order": "execution-plan order on one compute stream",
            "slot_reuse_guard": "copy(i) waits compute(i-slots)",
            "compute_readiness_guard": "compute(i) waits copy(i)",
            "runtime_lookup_required": False,
        },
        "operations": operations,
    }


def validate_schedule(schedule: dict[str, Any]) -> None:
    slots = int(schedule["slots"])
    operations = schedule["operations"]
    current_owner: dict[int, int] = {}

    for i, operation in enumerate(operations):
        if int(operation["sequence"]) != i or int(operation["tile_id"]) != i:
            raise ValueError(f"schedule entry {i} is not in deterministic tile order")
        slot = int(operation["slot"])
        if slot < 0 or slot >= slots:
            raise ValueError(f"schedule entry {i} has invalid slot {slot}")
        if slot != i % slots:
            raise ValueError(f"schedule entry {i} violates slot-assignment policy")

        previous = current_owner.get(slot)
        waits = list(operation["copy"].get("wait_for", []))
        if previous is None:
            if waits:
                raise ValueError(f"tile {i}: first slot owner should not wait on old compute")
        else:
            required = f"compute:{previous}:done"
            if required not in waits:
                raise ValueError(
                    f"tile {i}: slot {slot} reuse lacks dependency on previous owner {previous}"
                )

        required_copy = f"copy:{i}:done"
        if required_copy not in operation["compute"].get("wait_for", []):
            raise ValueError(f"tile {i}: compute can start before its H2D copy is complete")

        current_owner[slot] = i


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    schedule = build_schedule(plan, args.slots)
    validate_schedule(schedule)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(schedule, indent=2) + "\n", encoding="utf-8")

    print("TensorWave static runtime schedule")
    print(f"  tiles:  {schedule['tile_count']}")
    print(f"  slots:  {schedule['slots']}")
    print(f"  output: {args.output}")
    print("  runtime tile lookup: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
