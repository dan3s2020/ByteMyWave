#!/usr/bin/env python3
"""Build a deterministic TensorWave Weight Atlas from safetensors shards.

This tool intentionally uses only Python's standard library. It reads only the
safetensors headers; tensor payloads are not materialized into Python objects.
That lets us inventory very large checkpoints on machines with modest RAM.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable


DTYPE_BYTES: dict[str, int] = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}

MAX_HEADER_BYTES = 1 << 30  # defensive 1 GiB ceiling


@dataclass(frozen=True)
class TensorRecord:
    name: str
    shard: str
    dtype: str
    shape: list[int]
    rank: int
    numel: int
    nbytes: int
    bytes_per_element: int | None
    data_start_absolute: int
    tensor_offset_relative: int
    tensor_end_relative: int
    tensor_offset_absolute: int
    tensor_end_absolute: int
    size_check: str


def _product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid tensor dimension: {value!r}")
        result *= value
    return result


def read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: file is too small to be safetensors")
        (header_len,) = struct.unpack("<Q", raw)
        if header_len <= 1 or header_len > MAX_HEADER_BYTES:
            raise ValueError(f"{path}: unreasonable safetensors header length {header_len}")
        header_raw = handle.read(header_len)
        if len(header_raw) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")

    try:
        header = json.loads(header_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid safetensors JSON header: {exc}") from exc

    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header root is not an object")
    return 8 + header_len, header


def scan_shard(path: Path, model_dir: Path) -> tuple[dict[str, Any], list[TensorRecord]]:
    data_start, header = read_safetensors_header(path)
    file_size = path.stat().st_size
    relative_shard = path.relative_to(model_dir).as_posix()
    tensors: list[TensorRecord] = []

    for name, entry in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: tensor {name!r} metadata is not an object")

        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if not isinstance(dtype, str):
            raise ValueError(f"{path}: tensor {name!r} has invalid dtype")
        if not isinstance(shape, list):
            raise ValueError(f"{path}: tensor {name!r} has invalid shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(v, int) for v in offsets)
        ):
            raise ValueError(f"{path}: tensor {name!r} has invalid data_offsets")

        start_rel, end_rel = offsets
        if start_rel < 0 or end_rel < start_rel:
            raise ValueError(f"{path}: tensor {name!r} has invalid offset range")
        start_abs = data_start + start_rel
        end_abs = data_start + end_rel
        if end_abs > file_size:
            raise ValueError(f"{path}: tensor {name!r} extends beyond end of file")

        numel = _product(shape)
        actual_nbytes = end_rel - start_rel
        bytes_per_element = DTYPE_BYTES.get(dtype)
        if bytes_per_element is None:
            size_check = "unknown-dtype"
        else:
            expected_nbytes = numel * bytes_per_element
            size_check = "ok" if expected_nbytes == actual_nbytes else "mismatch"

        tensors.append(
            TensorRecord(
                name=name,
                shard=relative_shard,
                dtype=dtype,
                shape=shape,
                rank=len(shape),
                numel=numel,
                nbytes=actual_nbytes,
                bytes_per_element=bytes_per_element,
                data_start_absolute=data_start,
                tensor_offset_relative=start_rel,
                tensor_end_relative=end_rel,
                tensor_offset_absolute=start_abs,
                tensor_end_absolute=end_abs,
                size_check=size_check,
            )
        )

    shard_summary = {
        "path": relative_shard,
        "file_bytes": file_size,
        "data_start_absolute": data_start,
        "tensor_count": len(tensors),
        "metadata": header.get("__metadata__", {}),
    }
    return shard_summary, tensors


def build_atlas(model_dir: Path) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    if not model_dir.is_dir():
        raise ValueError(f"model directory does not exist: {model_dir}")

    shards = sorted(model_dir.rglob("*.safetensors"))
    if not shards:
        raise ValueError(f"no .safetensors files found under {model_dir}")

    shard_records: list[dict[str, Any]] = []
    tensor_records: list[TensorRecord] = []
    for shard in shards:
        shard_summary, tensors = scan_shard(shard, model_dir)
        shard_records.append(shard_summary)
        tensor_records.extend(tensors)

    tensor_records.sort(
        key=lambda t: (t.shard, t.tensor_offset_relative, t.name)
    )

    dtype_counts: dict[str, int] = {}
    dtype_bytes: dict[str, int] = {}
    rank_counts: dict[str, int] = {}
    for tensor in tensor_records:
        dtype_counts[tensor.dtype] = dtype_counts.get(tensor.dtype, 0) + 1
        dtype_bytes[tensor.dtype] = dtype_bytes.get(tensor.dtype, 0) + tensor.nbytes
        rank_key = str(tensor.rank)
        rank_counts[rank_key] = rank_counts.get(rank_key, 0) + 1

    total_tensor_bytes = sum(t.nbytes for t in tensor_records)
    invalid_size_count = sum(t.size_check == "mismatch" for t in tensor_records)

    return {
        "schema": "tensorwave.weight-atlas.v1",
        "model_dir": str(model_dir),
        "summary": {
            "shard_count": len(shard_records),
            "tensor_count": len(tensor_records),
            "tensor_payload_bytes": total_tensor_bytes,
            "tensor_payload_gib": total_tensor_bytes / (1024**3),
            "dtype_counts": dtype_counts,
            "dtype_bytes": dtype_bytes,
            "rank_counts": rank_counts,
            "size_mismatch_count": invalid_size_count,
        },
        "shards": shard_records,
        "tensors": [asdict(record) for record in tensor_records],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a TensorWave Weight Atlas without loading tensor payloads."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--print-top",
        type=int,
        default=12,
        help="Print the N largest tensors after writing the atlas (default: 12).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    atlas = build_atlas(args.model_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(atlas, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    summary = atlas["summary"]
    print("TensorWave Weight Atlas")
    print(f"  shards:       {summary['shard_count']}")
    print(f"  tensors:      {summary['tensor_count']}")
    print(f"  payload GiB:  {summary['tensor_payload_gib']:.3f}")
    print(f"  dtypes:       {summary['dtype_counts']}")
    print(f"  size errors:  {summary['size_mismatch_count']}")
    print(f"  output:       {args.output}")

    top_n = max(0, args.print_top)
    if top_n:
        largest = sorted(
            atlas["tensors"], key=lambda t: int(t["nbytes"]), reverse=True
        )[:top_n]
        print("\nLargest tensors:")
        for tensor in largest:
            gib = tensor["nbytes"] / (1024**3)
            print(
                f"  {gib:8.3f} GiB  {tensor['dtype']:>5}  "
                f"{str(tensor['shape']):>24}  {tensor['name']}"
            )

    if summary["size_mismatch_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
