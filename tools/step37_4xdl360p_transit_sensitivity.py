#!/usr/bin/env python3
"""Analytical sensitivity calculator for Step-3.7/3.5 on 4x dual-socket DL360p Gen8.

NOT a benchmark or prediction.  It exposes the assumptions behind the
2026-09-16 architecture note so measured server values can replace them later.
"""

from dataclasses import dataclass, asdict
import json

TOTAL_PARAMS = 196.81e9
Q4KS_FILE_BYTES = 111.5e9
ACTIVE_PARAMS = 11e9
HIDDEN = 4096
EXPERT_INTERMEDIATE = 1280
MOE_LAYERS = 42
TOP_K = 8
SOCKETS = 8
Q8K_BLOCK = 256
Q8K_BLOCK_BYTES = 292

EXPERT_PARAMS = 3 * HIDDEN * EXPERT_INTERMEDIATE
ROUTED_ACTIVE_PARAMS = EXPERT_PARAMS * MOE_LAYERS * TOP_K
SHARED_ACTIVE_PARAMS = EXPERT_PARAMS * MOE_LAYERS
AVG_BYTES_PER_PARAM = Q4KS_FILE_BYTES / TOTAL_PARAMS
ROUTED_BYTES_PER_TOKEN = ROUTED_ACTIVE_PARAMS * AVG_BYTES_PER_PARAM
ROUTED_BYTES_PER_SOCKET_TOKEN = ROUTED_BYTES_PER_TOKEN / SOCKETS
ACTIVE_BYTES_PER_TOKEN = ACTIVE_PARAMS * AVG_BYTES_PER_PARAM
Q8K_ACTIVATION_BYTES = HIDDEN // Q8K_BLOCK * Q8K_BLOCK_BYTES


@dataclass(frozen=True)
class Scenario:
    socket_expert_gb_s: float
    non_routed_critical_ms: float = 45.0
    future_local_kernel_speedup: float = 1.0
    mtp_wall_speedup: float = 1.0


def run(s: Scenario) -> dict:
    routed_ms = ROUTED_BYTES_PER_SOCKET_TOKEN / (s.socket_expert_gb_s * 1e9) * 1000
    routed_after_kernel_ms = routed_ms / s.future_local_kernel_speedup
    base_ms = routed_after_kernel_ms + s.non_routed_critical_ms
    tps_before_mtp = 1000.0 / base_ms
    tps = tps_before_mtp * s.mtp_wall_speedup
    return {
        **asdict(s),
        "routed_ms": routed_ms,
        "routed_after_kernel_ms": routed_after_kernel_ms,
        "token_ms_before_mtp": base_ms,
        "tok_s_before_mtp": tps_before_mtp,
        "tok_s": tps,
    }


def main() -> None:
    scenarios = []
    for bw in (5, 8, 10, 12, 15):
        scenarios.append(run(Scenario(bw)))
        scenarios.append(run(Scenario(bw, future_local_kernel_speedup=1.10)))
        scenarios.append(run(Scenario(bw, future_local_kernel_speedup=1.10, mtp_wall_speedup=1.15)))

    print(json.dumps({
        "schema": "bytemywave.step37-4xdl360p-transit-sensitivity.v1",
        "warning": "analytical sensitivity only; replace assumptions with measurements",
        "constants": {
            "total_params": TOTAL_PARAMS,
            "active_params": ACTIVE_PARAMS,
            "expert_params": EXPERT_PARAMS,
            "routed_active_params": ROUTED_ACTIVE_PARAMS,
            "shared_active_params": SHARED_ACTIVE_PARAMS,
            "avg_bytes_per_param_screen": AVG_BYTES_PER_PARAM,
            "active_bytes_per_token_screen": ACTIVE_BYTES_PER_TOKEN,
            "routed_bytes_per_token_screen": ROUTED_BYTES_PER_TOKEN,
            "routed_bytes_per_socket_token_screen": ROUTED_BYTES_PER_SOCKET_TOKEN,
            "q8k_activation_bytes": Q8K_ACTIVATION_BYTES,
        },
        "scenarios": scenarios,
    }, indent=2))


if __name__ == "__main__":
    main()
