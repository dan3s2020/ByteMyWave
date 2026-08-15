#!/usr/bin/env python3

from __future__ import annotations

import struct
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))

from quantize_q4_pack import (  # noqa: E402
    GROUP_BYTES,
    GROUP_SIZE,
    SCALE_BYTES,
    decode_source,
    quantize_values,
)


def unpack_q4(payload: bytes, element_count: int) -> np.ndarray:
    if element_count % GROUP_SIZE:
        raise ValueError("bad test element count")
    groups = element_count // GROUP_SIZE
    if len(payload) != groups * GROUP_BYTES:
        raise ValueError("bad Q4 payload size")

    result = np.empty(element_count, dtype=np.float32)
    for group in range(groups):
        base = group * GROUP_BYTES
        (scale,) = struct.unpack_from("<f", payload, base)
        for lane in range(GROUP_SIZE):
            packed = payload[base + SCALE_BYTES + lane // 2]
            nibble = (packed & 0x0F) if lane % 2 == 0 else ((packed >> 4) & 0x0F)
            q = nibble if nibble < 8 else nibble - 16
            result[group * GROUP_SIZE + lane] = q * scale
    return result


class Q4QuantizationTest(unittest.TestCase):
    def test_known_group_round_trips_with_bounded_error(self) -> None:
        values = np.linspace(-1.0, 1.0, GROUP_SIZE, dtype=np.float32)
        payload, stats = quantize_values(values)
        self.assertEqual(len(payload), GROUP_BYTES)

        reconstructed = unpack_q4(payload, GROUP_SIZE)
        scale = 1.0 / 7.0
        self.assertLessEqual(float(np.max(np.abs(reconstructed - values))), scale / 2 + 1e-6)
        self.assertAlmostEqual(stats["max_abs_error"], float(np.max(np.abs(reconstructed - values))), places=6)

    def test_zero_group_has_finite_scale_and_exact_zero(self) -> None:
        values = np.zeros(GROUP_SIZE, dtype=np.float32)
        payload, stats = quantize_values(values)
        reconstructed = unpack_q4(payload, GROUP_SIZE)
        self.assertTrue(np.array_equal(reconstructed, values))
        (scale,) = struct.unpack_from("<f", payload, 0)
        self.assertEqual(scale, 1.0)
        self.assertEqual(stats["squared_error"], 0.0)

    def test_twos_complement_nibbles_cover_negative_and_positive(self) -> None:
        values = np.zeros(GROUP_SIZE, dtype=np.float32)
        values[0] = -7.0
        values[1] = 7.0
        payload, _ = quantize_values(values)
        first_packed = payload[SCALE_BYTES]
        self.assertEqual(first_packed & 0x0F, 0x09)  # -7 in signed int4 two's complement
        self.assertEqual((first_packed >> 4) & 0x0F, 0x07)
        reconstructed = unpack_q4(payload, GROUP_SIZE)
        self.assertEqual(reconstructed[0], -7.0)
        self.assertEqual(reconstructed[1], 7.0)

    def test_fp16_and_bf16_decoders(self) -> None:
        fp16 = np.array([-1.5, 0.0, 2.25], dtype="<f2")
        decoded_fp16 = decode_source(fp16.tobytes(), "F16")
        np.testing.assert_allclose(decoded_fp16, [-1.5, 0.0, 2.25], rtol=0, atol=0)

        values = np.array([-1.5, 0.0, 2.25], dtype=np.float32)
        bits = values.view(np.uint32)
        bf16_bits = (bits >> np.uint32(16)).astype("<u2")
        decoded_bf16 = decode_source(bf16_bits.tobytes(), "BF16")
        np.testing.assert_allclose(decoded_bf16, values, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
