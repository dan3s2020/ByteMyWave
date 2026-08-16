"""Exact software reference for the first Transit bitplane datapath.

This module intentionally models the *proof format* used by the measured host
benchmarks: signed two's-complement INT4 weights and signed INT8 activations.
It is not an MXFP4/MXFP8 implementation.

The functions here are designed to generate golden vectors for FPGA simulation
and physical-tile validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

WEIGHT_COEFFS = np.asarray([1, 2, 4, -8], dtype=np.int64)
ACTIVATION_COEFFS = np.asarray([1, 2, 4, 8, 16, 32, 64, -128], dtype=np.int64)


@dataclass(frozen=True)
class BitplaneDotResult:
    value: int
    intersections: np.ndarray  # shape [4, 8], popcounts


def _require_1d(name: str, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape={x.shape}")
    return x


def validate_int4(weights: np.ndarray) -> np.ndarray:
    """Return weights as int8 after validating signed INT4 range [-8, 7]."""
    w = _require_1d("weights", np.asarray(weights))
    if np.any(w < -8) or np.any(w > 7):
        raise ValueError("weights contain values outside signed INT4 range [-8, 7]")
    return w.astype(np.int8, copy=False)


def validate_int8(activations: np.ndarray) -> np.ndarray:
    """Return activations as int8 after validating signed INT8 range."""
    a = _require_1d("activations", np.asarray(activations))
    if np.any(a < -128) or np.any(a > 127):
        raise ValueError("activations contain values outside signed INT8 range")
    return a.astype(np.int8, copy=False)


def int4_to_bitplanes(weights: np.ndarray) -> np.ndarray:
    """Convert signed INT4 values to unpacked bitplanes with shape [4, N]."""
    w = validate_int4(weights)
    codes = (w.astype(np.int16) & 0x0F).astype(np.uint8)
    return np.stack([((codes >> bit) & 1) for bit in range(4)], axis=0)


def int8_to_bitplanes(activations: np.ndarray) -> np.ndarray:
    """Convert signed INT8 values to unpacked bitplanes with shape [8, N]."""
    a = validate_int8(activations)
    codes = (a.astype(np.int16) & 0xFF).astype(np.uint8)
    return np.stack([((codes >> bit) & 1) for bit in range(8)], axis=0)


def pack_planes_little(planes: np.ndarray) -> bytes:
    """Pack [P, N] bitplanes as plane-major bytes, LSB-first within each byte."""
    p = np.asarray(planes, dtype=np.uint8)
    if p.ndim != 2:
        raise ValueError(f"planes must be [P,N], got shape={p.shape}")
    if p.shape[1] % 8:
        raise ValueError("number of elements must be divisible by 8 for packed output")
    if np.any((p != 0) & (p != 1)):
        raise ValueError("planes must contain only 0/1")
    return b"".join(np.packbits(row, bitorder="little").tobytes() for row in p)


def unpack_planes_little(blob: bytes, plane_count: int, element_count: int) -> np.ndarray:
    """Inverse of :func:`pack_planes_little`."""
    if element_count % 8:
        raise ValueError("element_count must be divisible by 8")
    bytes_per_plane = element_count // 8
    expected = bytes_per_plane * plane_count
    if len(blob) != expected:
        raise ValueError(f"expected {expected} bytes, got {len(blob)}")
    out = []
    for p in range(plane_count):
        start = p * bytes_per_plane
        raw = np.frombuffer(blob[start : start + bytes_per_plane], dtype=np.uint8)
        out.append(np.unpackbits(raw, bitorder="little")[:element_count])
    return np.stack(out, axis=0).astype(np.uint8, copy=False)


def reconstruct_int4(planes: np.ndarray) -> np.ndarray:
    """Reconstruct signed INT4 values from unpacked [4,N] bitplanes."""
    p = np.asarray(planes, dtype=np.int64)
    if p.ndim != 2 or p.shape[0] != 4:
        raise ValueError(f"expected [4,N] planes, got shape={p.shape}")
    return (WEIGHT_COEFFS[:, None] * p).sum(axis=0).astype(np.int8)


def reconstruct_int8(planes: np.ndarray) -> np.ndarray:
    """Reconstruct signed INT8 values from unpacked [8,N] bitplanes."""
    p = np.asarray(planes, dtype=np.int64)
    if p.ndim != 2 or p.shape[0] != 8:
        raise ValueError(f"expected [8,N] planes, got shape={p.shape}")
    return (ACTIVATION_COEFFS[:, None] * p).sum(axis=0).astype(np.int8)


def dot_scalar(weights: np.ndarray, activations: np.ndarray) -> int:
    """Golden signed-integer dot product."""
    w = validate_int4(weights)
    a = validate_int8(activations)
    if w.size != a.size:
        raise ValueError("weights and activations must have the same length")
    return int(np.dot(w.astype(np.int64), a.astype(np.int64)))


def dot_bitplanes(weights: np.ndarray, activations: np.ndarray) -> BitplaneDotResult:
    """Exact dot product through the 4×8 bitplane-intersection identity."""
    w = validate_int4(weights)
    a = validate_int8(activations)
    if w.size != a.size:
        raise ValueError("weights and activations must have the same length")

    wp = int4_to_bitplanes(w)
    ap = int8_to_bitplanes(a)

    # Broadcasting creates [4,8,N], then sums the logical intersections.
    intersections = np.logical_and(wp[:, None, :], ap[None, :, :]).sum(axis=2, dtype=np.int64)
    coeff = WEIGHT_COEFFS[:, None] * ACTIVATION_COEFFS[None, :]
    value = int(np.sum(intersections * coeff, dtype=np.int64))
    return BitplaneDotResult(value=value, intersections=intersections)


def matvec_bitplanes(weights: np.ndarray, activations: np.ndarray) -> np.ndarray:
    """Reference matrix-vector operation using the exact bitplane identity row-by-row."""
    w = np.asarray(weights)
    if w.ndim != 2:
        raise ValueError(f"weights must be [rows, cols], got shape={w.shape}")
    a = validate_int8(activations)
    if w.shape[1] != a.size:
        raise ValueError("matrix columns must equal activation length")
    if np.any(w < -8) or np.any(w > 7):
        raise ValueError("matrix contains values outside signed INT4 range [-8, 7]")

    out = np.empty(w.shape[0], dtype=np.int64)
    for row in range(w.shape[0]):
        out[row] = dot_bitplanes(w[row], a).value
    return out


def assert_exact(weights: np.ndarray, activations: np.ndarray) -> None:
    """Raise AssertionError unless scalar and bitplane implementations match exactly."""
    scalar = dot_scalar(weights, activations)
    bitplane = dot_bitplanes(weights, activations).value
    if scalar != bitplane:
        raise AssertionError(f"bitplane mismatch: scalar={scalar} bitplane={bitplane}")


def generate_golden_vector(length: int = 2048, seed: int = 1) -> tuple[np.ndarray, np.ndarray, int]:
    """Generate deterministic signed INT4/INT8 inputs and their exact golden dot."""
    if length <= 0:
        raise ValueError("length must be positive")
    rng = np.random.default_rng(seed)
    w = rng.integers(-8, 8, size=length, dtype=np.int8)
    a = rng.integers(-128, 128, size=length, dtype=np.int16).astype(np.int8)
    return w, a, dot_scalar(w, a)


if __name__ == "__main__":
    w, a, golden = generate_golden_vector()
    result = dot_bitplanes(w, a)
    assert result.value == golden
    print(f"length={w.size}")
    print(f"scalar={golden}")
    print(f"bitplane={result.value}")
    print("exact=True")
