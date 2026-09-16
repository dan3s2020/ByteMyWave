#!/usr/bin/env python3
"""Reproducible roofline arithmetic for the Carrizo APU + DDR3 Kimi K3 track.

This tool intentionally reports analytical/screening quantities only. It does not
claim end-to-end Kimi K3 token throughput.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from typing import Dict, List


TOTAL_PARAMS = 2.8e12
ACTIVE_PARAMS = 104e9
CHECKPOINT_BYTES = 1.56e12
LAYERS = 93
DENSE_LAYERS = 1
SELECTED_EXPERTS = 16
LATENT_DIM = 3584
MOE_HIDDEN_DIM = 3072
MXFP4_GROUP_SIZE = 32
MXFP4_BITS_PER_WEIGHT = 4
MXFP4_SCALE_BYTES_PER_GROUP = 1
BF16_BYTES_PER_PARAM = 2


@dataclass(frozen=True)
class ByteModels:
    four_bit_lower_bound_gb: float
    checkpoint_average_screening_gb: float
    conservative_mixed_envelope_gb: float
    routed_active_params_b: float
    routed_mxfp4_bytes_gb: float
    remaining_active_params_b: float


def bytes_per_mxfp4_weight() -> float:
    packed_weight_bytes = MXFP4_GROUP_SIZE * MXFP4_BITS_PER_WEIGHT / 8.0
    return (packed_weight_bytes + MXFP4_SCALE_BYTES_PER_GROUP) / MXFP4_GROUP_SIZE


def build_byte_models() -> ByteModels:
    expert_weights = 3 * LATENT_DIM * MOE_HIDDEN_DIM
    moe_layers = LAYERS - DENSE_LAYERS
    routed_active = moe_layers * SELECTED_EXPERTS * expert_weights

    lower = ACTIVE_PARAMS * 0.5
    screening = ACTIVE_PARAMS * (CHECKPOINT_BYTES / TOTAL_PARAMS)
    routed_bytes = routed_active * bytes_per_mxfp4_weight()
    remaining_active = ACTIVE_PARAMS - routed_active
    conservative = routed_bytes + remaining_active * BF16_BYTES_PER_PARAM

    return ByteModels(
        four_bit_lower_bound_gb=lower / 1e9,
        checkpoint_average_screening_gb=screening / 1e9,
        conservative_mixed_envelope_gb=conservative / 1e9,
        routed_active_params_b=routed_active / 1e9,
        routed_mxfp4_bytes_gb=routed_bytes / 1e9,
        remaining_active_params_b=remaining_active / 1e9,
    )


def channel_payload_gbps(mtps: float) -> float:
    # DDR x64 channel: MT/s * 8 bytes/transfer, expressed in decimal GB/s.
    return mtps * 8.0 / 1000.0


def layer_budget_ms(target_tps: float) -> float:
    return 1000.0 / (target_tps * LAYERS)


def build_report(channels: int, mtps: float, target_tps: List[float]) -> Dict[str, object]:
    models = build_byte_models()
    one_channel = channel_payload_gbps(mtps)
    aggregate = channels * one_channel

    rooflines = {
        "four_bit_lower_bound": aggregate / models.four_bit_lower_bound_gb,
        "checkpoint_average_screening": aggregate / models.checkpoint_average_screening_gb,
        "conservative_mixed_envelope": aggregate / models.conservative_mixed_envelope_gb,
    }

    return {
        "evidence_class": "analytical_roofline_not_end_to_end_k3_tps",
        "inputs": {
            "channels": channels,
            "mtps": mtps,
            "layers": LAYERS,
            "active_params": ACTIVE_PARAMS,
            "checkpoint_bytes": CHECKPOINT_BYTES,
        },
        "derived_model": {
            "expert_weights_per_expert": 3 * LATENT_DIM * MOE_HIDDEN_DIM,
            "mxfp4_bytes_per_weight_including_one_uint8_scale_per_32": bytes_per_mxfp4_weight(),
            **asdict(models),
        },
        "memory_roofline": {
            "one_channel_gbps": one_channel,
            "aggregate_gbps": aggregate,
            "weight_path_tps_equivalent": rooflines,
        },
        "layer_budgets_ms": {str(tps): layer_budget_ms(tps) for tps in target_tps},
        "warnings": [
            "52 GB/token is an all-active-weights-at-four-bit lower bound, not exact K3 traffic.",
            "Checkpoint-average screening assumes the active subset has the full checkpoint's average byte density.",
            "The conservative mixed envelope pessimistically treats all non-routed active parameters as BF16; it is not exact.",
            "Nominal DDR payload does not include controller efficiency, compute stalls, routing, collectives, KDA/state, or software overhead.",
            "Installed channel bandwidth only contributes to one token if the active work is placed/sharded so those channels execute concurrently.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channels", type=int, default=320, help="Independent DDR channels (default: 320)")
    parser.add_argument("--mtps", type=float, default=1600.0, help="DDR transfer rate in MT/s (default: 1600)")
    parser.add_argument(
        "--target-tps",
        type=float,
        nargs="*",
        default=[1.0, 5.0, 10.0, 20.0, 30.0],
        help="Decode target rates for average layer-budget calculation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.channels <= 0:
        raise SystemExit("--channels must be > 0")
    if args.mtps <= 0:
        raise SystemExit("--mtps must be > 0")
    if any(tps <= 0 for tps in args.target_tps):
        raise SystemExit("all --target-tps values must be > 0")

    report = build_report(args.channels, args.mtps, args.target_tps)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
