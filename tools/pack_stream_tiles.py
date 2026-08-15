#!/usr/bin/env python3
"""Pack real 2D safetensors weights into a deterministic streaming tile file.

The packer never deserializes tensor values. For F16/BF16 tensors, rows are
already contiguous bytes in safetensors. TensorWave therefore copies exact
checkpoint bytes into a flat pack and emits an execution plan describing every
tile's provenance and checksum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from safetensors_atlas import build_atlas


SUPPORTED_DTYPES = {"F16": 2, "BF16": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed-shape TensorWave streaming pack from safetensors."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=["auto", "F16", "BF16"],
        default="auto",
        help="Source dtype. auto selects the group with the most usable tiles.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=0,
        help="Required input width. 0 selects automatically.",
    )
    parser.add_argument(
        "--tile-n",
        type=int,
        default=256,
        help="Rows/output channels per tile (default: 256).",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=64,
        help="Maximum number of real tiles to pack (default: 64).",
    )
    parser.add_argument(
        "--min-tiles",
        type=int,
        default=2,
        help="Fail if fewer than this many full tiles are available.",
    )
    return parser.parse_args()


def choose_group(
    tensors: list[dict[str, Any]], dtype_filter: str, k_filter: int, tile_n: int
) -> tuple[str, int, list[dict[str, Any]]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for tensor in tensors:
        shape = tensor.get("shape")
        dtype = tensor.get("dtype")
        if dtype not in SUPPORTED_DTYPES:
            continue
        if not isinstance(shape, list) or len(shape) != 2:
            continue
        rows, k = shape
        if not isinstance(rows, int) or not isinstance(k, int):
            continue
        if rows < tile_n or k <= 0:
            continue
        if dtype_filter != "auto" and dtype != dtype_filter:
            continue
        if k_filter > 0 and k != k_filter:
            continue
        if tensor.get("size_check") != "ok":
            continue
        groups[(dtype, k)].append(tensor)

    if not groups:
        raise ValueError(
            "no usable rank-2 F16/BF16 tensor group matches the requested filters"
        )

    def score(item: tuple[tuple[str, int], list[dict[str, Any]]]) -> tuple[int, int, int]:
        (dtype, k), members = item
        full_tiles = sum(int(t["shape"][0]) // tile_n for t in members)
        payload = sum(int(t["nbytes"]) for t in members)
        return full_tiles, payload, k

    (dtype, k), members = max(groups.items(), key=score)
    members = sorted(
        members,
        key=lambda t: (
            str(t["shard"]),
            int(t["tensor_offset_relative"]),
            str(t["name"]),
        ),
    )
    return dtype, k, members


def copy_exact_slice(
    source_path: Path,
    absolute_offset: int,
    nbytes: int,
    destination,
) -> str:
    hasher = hashlib.sha256()
    remaining = nbytes
    chunk_size = 8 * 1024 * 1024

    with source_path.open("rb") as source:
        source.seek(absolute_offset)
        while remaining:
            chunk = source.read(min(chunk_size, remaining))
            if not chunk:
                raise ValueError(
                    f"unexpected EOF while reading {source_path} at offset "
                    f"{absolute_offset + (nbytes - remaining)}"
                )
            destination.write(chunk)
            hasher.update(chunk)
            remaining -= len(chunk)

    return hasher.hexdigest()


def main() -> int:
    args = parse_args()
    if args.k < 0:
        raise ValueError("--k must be >= 0")
    if args.tile_n <= 0:
        raise ValueError("--tile-n must be > 0")
    if args.max_tiles <= 0:
        raise ValueError("--max-tiles must be > 0")
    if args.min_tiles < 2:
        raise ValueError("--min-tiles must be >= 2")

    model_dir = args.model_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    atlas = build_atlas(model_dir)
    atlas_path = output_dir / "weight-atlas.json"
    atlas_path.write_text(json.dumps(atlas, indent=2) + "\n", encoding="utf-8")

    dtype, k, tensors = choose_group(
        atlas["tensors"], args.dtype, args.k, args.tile_n
    )
    bytes_per_element = SUPPORTED_DTYPES[dtype]
    tile_bytes = args.tile_n * k * bytes_per_element

    pack_path = output_dir / "weights.pack"
    plan_path = output_dir / "execution-plan.json"
    plan_tiles: list[dict[str, Any]] = []
    pack_offset = 0

    with pack_path.open("wb") as pack:
        tile_id = 0
        for tensor in tensors:
            rows = int(tensor["shape"][0])
            full_tile_count = rows // args.tile_n
            tensor_payload_abs = int(tensor["tensor_offset_absolute"])
            source_path = model_dir / str(tensor["shard"])

            for local_tile in range(full_tile_count):
                if tile_id >= args.max_tiles:
                    break

                row_start = local_tile * args.tile_n
                row_end = row_start + args.tile_n
                source_offset = (
                    tensor_payload_abs + row_start * k * bytes_per_element
                )

                sha256 = copy_exact_slice(
                    source_path, source_offset, tile_bytes, pack
                )

                plan_tiles.append(
                    {
                        "tile_id": tile_id,
                        "tensor_name": tensor["name"],
                        "shard": tensor["shard"],
                        "dtype": dtype,
                        "source_shape": tensor["shape"],
                        "row_start": row_start,
                        "row_end": row_end,
                        "k": k,
                        "n": args.tile_n,
                        "source_offset_absolute": source_offset,
                        "pack_offset": pack_offset,
                        "nbytes": tile_bytes,
                        "sha256": sha256,
                    }
                )
                tile_id += 1
                pack_offset += tile_bytes

            if tile_id >= args.max_tiles:
                break

    if len(plan_tiles) < args.min_tiles:
        pack_path.unlink(missing_ok=True)
        raise ValueError(
            f"only {len(plan_tiles)} full tiles available; need at least {args.min_tiles}"
        )

    plan = {
        "schema": "tensorwave.execution-plan.v1",
        "source": {
            "model_dir": str(model_dir),
            "atlas": str(atlas_path),
            "selection_policy": (
                "rank-2 exact raw row slices; storage-order within shards; "
                "single dtype/K group"
            ),
            "note": (
                "This plan proves real-checkpoint byte streaming. Storage order is not "
                "yet claimed to equal model inference order; graph-derived ordering is "
                "a later phase."
            ),
        },
        "geometry": {
            "dtype": dtype,
            "bytes_per_element": bytes_per_element,
            "k": k,
            "n": args.tile_n,
            "tile_bytes": tile_bytes,
            "tile_count": len(plan_tiles),
            "pack_bytes": pack_offset,
        },
        "tiles": plan_tiles,
    }
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    print("TensorWave real-weight pack")
    print(f"  dtype:       {dtype}")
    print(f"  K:           {k}")
    print(f"  tile N:      {args.tile_n}")
    print(f"  tile bytes:  {tile_bytes:,}")
    print(f"  tiles:       {len(plan_tiles)}")
    print(f"  pack MiB:    {pack_offset / (1024**2):.2f}")
    print(f"  atlas:       {atlas_path}")
    print(f"  plan:        {plan_path}")
    print(f"  pack:        {pack_path}")
    print("\nSelected tensors:")

    seen: set[str] = set()
    for tile in plan_tiles:
        name = str(tile["tensor_name"])
        if name not in seen:
            seen.add(name)
            print(f"  {tile['dtype']:>4} {tile['source_shape']}  {name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
