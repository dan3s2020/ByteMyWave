#!/usr/bin/env python3
"""Validate TensorWave Q4 plan geometry and every packed-tile checksum."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify a TensorWave Q4 pack.")
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--pack", type=Path, required=True)
    return parser.parse_args()


def verify(plan: dict, pack_path: Path) -> None:
    if plan.get("schema") != "tensorwave.q4-plan.v1":
        raise ValueError("unsupported/missing Q4 plan schema")

    quant = plan.get("quantization", {})
    geometry = plan.get("geometry", {})
    tiles = plan.get("tiles", [])

    if quant.get("name") != "Q4_SYM_G32_F32S":
        raise ValueError("unexpected Q4 format")
    if int(quant.get("group_size", 0)) != 32:
        raise ValueError("unexpected Q4 group size")
    if int(quant.get("group_bytes", 0)) != 20:
        raise ValueError("unexpected Q4 group byte size")

    tile_count = int(geometry.get("tile_count", -1))
    tile_elements = int(geometry.get("tile_elements", 0))
    q4_tile_bytes = int(geometry.get("q4_tile_bytes", 0))
    q4_pack_bytes = int(geometry.get("q4_pack_bytes", 0))
    source_tile_bytes = int(geometry.get("source_tile_bytes", 0))

    if tile_count <= 0 or tile_count != len(tiles):
        raise ValueError("Q4 tile count is invalid")
    if tile_elements <= 0 or tile_elements % 32:
        raise ValueError("tile_elements must be a positive multiple of 32")
    expected_q4_tile_bytes = (tile_elements // 32) * 20
    if q4_tile_bytes != expected_q4_tile_bytes:
        raise ValueError("q4_tile_bytes does not match Q4 group geometry")
    if source_tile_bytes != tile_elements * 2:
        raise ValueError("source_tile_bytes is inconsistent with a 16-bit source")
    if q4_pack_bytes != tile_count * q4_tile_bytes:
        raise ValueError("q4_pack_bytes does not match tile_count * q4_tile_bytes")

    actual_pack_bytes = pack_path.stat().st_size
    if actual_pack_bytes != q4_pack_bytes:
        raise ValueError(
            f"Q4 pack size mismatch: plan={q4_pack_bytes}, file={actual_pack_bytes}"
        )

    with pack_path.open("rb") as handle:
        expected_offset = 0
        for index, tile in enumerate(tiles):
            if int(tile.get("tile_id", -1)) != index:
                raise ValueError(f"tile IDs are not contiguous at index {index}")
            offset = int(tile.get("q4_pack_offset", -1))
            nbytes = int(tile.get("q4_nbytes", -1))
            if offset != expected_offset:
                raise ValueError(f"tile {index}: non-contiguous q4_pack_offset")
            if nbytes != q4_tile_bytes:
                raise ValueError(f"tile {index}: wrong q4_nbytes")

            handle.seek(offset)
            payload = handle.read(nbytes)
            if len(payload) != nbytes:
                raise ValueError(f"tile {index}: short read")
            actual_hash = hashlib.sha256(payload).hexdigest()
            expected_hash = str(tile.get("q4_sha256", ""))
            if actual_hash != expected_hash:
                raise ValueError(
                    f"tile {index}: SHA-256 mismatch; expected {expected_hash}, got {actual_hash}"
                )
            expected_offset += nbytes


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    verify(plan, args.pack.resolve())
    print("TensorWave Q4 pack verification: PASS")
    print(f"  plan: {args.plan.resolve()}")
    print(f"  pack: {args.pack.resolve()}")
    print(f"  tiles: {plan['geometry']['tile_count']}")
    print(f"  bytes: {plan['geometry']['q4_pack_bytes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
