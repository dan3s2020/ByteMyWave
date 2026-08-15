#!/usr/bin/env python3
"""Quantize a TensorWave F16/BF16 real-weight pack to a streamable Q4 format.

Q4 v1 is intentionally simple and inspectable:

    group = 32 weights
    bytes 0..3   : little-endian float32 scale
    bytes 4..19  : 16 bytes, two signed int4 values per byte

Signed int4 values use two's-complement nibbles. Quantization is symmetric with
q in [-7, 7], scale=max(abs(x))/7. Zero groups use scale=1 and all q=0.

The format is 20 bytes per 32 weights = 5 effective bits/weight, including
scales. Compared with F16/BF16 (64 bytes/group), H2D traffic is 31.25% of the
original bytes (3.2x smaller).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - user-facing environment guard
    raise SystemExit(
        "numpy is required for Q4 conversion. Install it with: python -m pip install numpy"
    ) from exc


GROUP_SIZE = 32
SCALE_BYTES = 4
PACKED_VALUE_BYTES = GROUP_SIZE // 2
GROUP_BYTES = SCALE_BYTES + PACKED_VALUE_BYTES
QMIN = -7
QMAX = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a TensorWave real-weight pack to Q4 v1."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--input-pack", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def decode_source(payload: bytes | memoryview, dtype: str) -> np.ndarray:
    if dtype == "F16":
        return np.frombuffer(payload, dtype="<f2").astype(np.float32)
    if dtype == "BF16":
        # BF16 is the high 16 bits of IEEE-754 float32. This conversion is exact
        # with respect to the source BF16 value and does not depend on NumPy
        # having a native bfloat16 dtype.
        u16 = np.frombuffer(payload, dtype="<u2")
        u32 = u16.astype(np.uint32) << np.uint32(16)
        return u32.view(np.float32)
    raise ValueError(f"unsupported source dtype for Q4 v1: {dtype}")


def quantize_values(values: np.ndarray) -> tuple[bytes, dict[str, float]]:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("values must be flat")
    if values.size == 0 or values.size % GROUP_SIZE != 0:
        raise ValueError(f"weight count must be a positive multiple of {GROUP_SIZE}")
    if not np.all(np.isfinite(values)):
        raise ValueError("source weights contain NaN or infinity")

    groups = values.reshape(-1, GROUP_SIZE)
    max_abs = np.max(np.abs(groups), axis=1)
    scales = np.where(max_abs > 0.0, max_abs / float(QMAX), 1.0).astype(np.float32)

    normalized = groups / scales[:, None]
    q = np.rint(normalized)
    q = np.clip(q, QMIN, QMAX).astype(np.int8)

    # Store signed int4 as two's-complement nibbles.
    nibble = np.bitwise_and(q.astype(np.int16), 0xF).astype(np.uint8)
    packed = np.bitwise_or(nibble[:, 0::2], np.left_shift(nibble[:, 1::2], 4))

    records = np.empty((groups.shape[0], GROUP_BYTES), dtype=np.uint8)
    records[:, :SCALE_BYTES] = scales.astype("<f4", copy=False).view(np.uint8).reshape(-1, 4)
    records[:, SCALE_BYTES:] = packed

    reconstructed = q.astype(np.float32) * scales[:, None]
    error = reconstructed - groups
    error_f64 = error.astype(np.float64)
    values_f64 = groups.astype(np.float64)

    squared_error = float(np.sum(error_f64 * error_f64, dtype=np.float64))
    squared_signal = float(np.sum(values_f64 * values_f64, dtype=np.float64))
    max_error = float(np.max(np.abs(error)))
    count = int(values.size)

    return records.tobytes(order="C"), {
        "count": float(count),
        "squared_error": squared_error,
        "squared_signal": squared_signal,
        "max_abs_error": max_error,
    }


def main() -> int:
    args = parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    if plan.get("schema") != "tensorwave.execution-plan.v1":
        raise ValueError("input plan must use tensorwave.execution-plan.v1")

    geometry = plan.get("geometry", {})
    tiles = plan.get("tiles", [])
    dtype = str(geometry.get("dtype", ""))
    if dtype not in {"F16", "BF16"}:
        raise ValueError("Q4 v1 input must be F16 or BF16")

    bytes_per_element = int(geometry.get("bytes_per_element", 0))
    if bytes_per_element != 2:
        raise ValueError("Q4 v1 expects a 2-byte source dtype")

    tile_count = int(geometry.get("tile_count", -1))
    original_tile_bytes = int(geometry.get("tile_bytes", 0))
    if tile_count != len(tiles) or tile_count <= 0 or original_tile_bytes <= 0:
        raise ValueError("invalid execution-plan geometry")
    if original_tile_bytes % bytes_per_element != 0:
        raise ValueError("source tile byte count is not element-aligned")

    tile_elements = original_tile_bytes // bytes_per_element
    if tile_elements % GROUP_SIZE != 0:
        raise ValueError(
            f"tile has {tile_elements} elements; Q4 v1 requires a multiple of {GROUP_SIZE}"
        )
    groups_per_tile = tile_elements // GROUP_SIZE
    quant_tile_bytes = groups_per_tile * GROUP_BYTES

    input_pack = args.input_pack.resolve()
    expected_input_bytes = tile_count * original_tile_bytes
    actual_input_bytes = input_pack.stat().st_size
    if actual_input_bytes != expected_input_bytes:
        raise ValueError(
            f"input pack has {actual_input_bytes} bytes; plan requires {expected_input_bytes}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pack = output_dir / "weights-q4.pack"
    output_plan = output_dir / "q4-plan.json"

    q4_tiles: list[dict[str, Any]] = []
    total_squared_error = 0.0
    total_squared_signal = 0.0
    global_max_error = 0.0
    total_count = 0

    with input_pack.open("rb") as src, output_pack.open("wb") as dst:
        q4_offset = 0
        for index, tile in enumerate(tiles):
            source_offset = int(tile["pack_offset"])
            if source_offset != index * original_tile_bytes:
                raise ValueError("Q4 v1 currently requires a contiguous uniform source pack")

            src.seek(source_offset)
            raw = src.read(original_tile_bytes)
            if len(raw) != original_tile_bytes:
                raise ValueError(f"short read for source tile {index}")

            values = decode_source(raw, dtype)
            qbytes, stats = quantize_values(values)
            if len(qbytes) != quant_tile_bytes:
                raise AssertionError("internal Q4 byte-size mismatch")
            dst.write(qbytes)

            q4_tiles.append(
                {
                    "tile_id": index,
                    "source_tensor_name": tile.get("tensor_name"),
                    "source_shard": tile.get("shard"),
                    "source_row_start": tile.get("row_start"),
                    "source_row_end": tile.get("row_end"),
                    "source_pack_offset": source_offset,
                    "source_nbytes": original_tile_bytes,
                    "q4_pack_offset": q4_offset,
                    "q4_nbytes": quant_tile_bytes,
                    "q4_sha256": hashlib.sha256(qbytes).hexdigest(),
                    "quant_max_abs_error": stats["max_abs_error"],
                }
            )

            q4_offset += quant_tile_bytes
            total_squared_error += stats["squared_error"]
            total_squared_signal += stats["squared_signal"]
            global_max_error = max(global_max_error, stats["max_abs_error"])
            total_count += int(stats["count"])

    rms_error = math.sqrt(total_squared_error / total_count) if total_count else 0.0
    signal_rms = math.sqrt(total_squared_signal / total_count) if total_count else 0.0
    if total_squared_error == 0.0:
        snr_db = float("inf") if total_squared_signal > 0.0 else 0.0
    elif total_squared_signal == 0.0:
        snr_db = float("-inf")
    else:
        snr_db = 10.0 * math.log10(total_squared_signal / total_squared_error)

    q4_pack_bytes = tile_count * quant_tile_bytes
    q4_plan = {
        "schema": "tensorwave.q4-plan.v1",
        "source_plan": str(args.plan.resolve()),
        "source_pack": str(input_pack),
        "quantization": {
            "name": "Q4_SYM_G32_F32S",
            "group_size": GROUP_SIZE,
            "qmin": QMIN,
            "qmax": QMAX,
            "scale_dtype": "F32",
            "scale_bytes": SCALE_BYTES,
            "packed_value_bytes_per_group": PACKED_VALUE_BYTES,
            "group_bytes": GROUP_BYTES,
            "effective_bits_per_weight": GROUP_BYTES * 8.0 / GROUP_SIZE,
            "nibble_encoding": "signed int4 two's-complement",
        },
        "geometry": {
            "source_dtype": dtype,
            "k": int(geometry["k"]),
            "n": int(geometry["n"]),
            "tile_count": tile_count,
            "tile_elements": tile_elements,
            "groups_per_tile": groups_per_tile,
            "source_tile_bytes": original_tile_bytes,
            "q4_tile_bytes": quant_tile_bytes,
            "source_pack_bytes": expected_input_bytes,
            "q4_pack_bytes": q4_pack_bytes,
            "q4_to_source_byte_ratio": q4_pack_bytes / expected_input_bytes,
            "source_to_q4_compression_x": expected_input_bytes / q4_pack_bytes,
        },
        "quality": {
            "weight_count": total_count,
            "rms_error": rms_error,
            "signal_rms": signal_rms,
            "max_abs_error": global_max_error,
            "snr_db": snr_db,
        },
        "tiles": q4_tiles,
    }
    output_plan.write_text(json.dumps(q4_plan, indent=2) + "\n", encoding="utf-8")

    print("TensorWave Q4 pack")
    print(f"  source dtype:        {dtype}")
    print(f"  tiles:               {tile_count}")
    print(f"  weights/tile:        {tile_elements:,}")
    print(f"  source MiB:          {expected_input_bytes / (1024**2):.2f}")
    print(f"  Q4 MiB:              {q4_pack_bytes / (1024**2):.2f}")
    print(f"  compression:         {expected_input_bytes / q4_pack_bytes:.3f}x")
    print(f"  effective bits/w:    {GROUP_BYTES * 8.0 / GROUP_SIZE:.3f}")
    print(f"  weight RMS error:    {rms_error:.8g}")
    print(f"  weight max error:    {global_max_error:.8g}")
    print(f"  weight SNR:          {snr_db:.3f} dB")
    print(f"  pack:                {output_pack}")
    print(f"  plan:                {output_plan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
