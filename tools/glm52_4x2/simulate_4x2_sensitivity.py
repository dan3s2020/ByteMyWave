#!/usr/bin/env python3
"""Analytical sensitivity model for the ByteMyWave GLM-5.2 4x2 prototype.

This is deliberately not presented as a hardware benchmark.  It models the
critical path of one batch=1 decode token under the architecture selected for
four 4-socket servers:

* 75 MoE layers, top-8 routed experts;
* routed expert work is load-balanced across all 16 NUMA/socket domains;
* each selected expert may be tensor-sharded over two sockets;
* GPU-resident hot experts may replace the CPU critical path only when every
  selected expert needed by that layer is available on the GPU fast path;
* the always-on attention/shared/dense path is summarized by a measured-or-
  assumed per-token latency input;
* network/reduction latency is charged once per MoE layer;
* speculative/MTP benefit is a wall-clock multiplier, not tokens-per-forward.

The model exists to expose which physical measurements control the answer.
Replace the scenario assumptions with measured values as soon as the four
servers are available.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json


MOE_LAYERS = 75
SELECTED_EXPERTS = 8
HIDDEN_SIZE = 6144
MOE_INTERMEDIATE_SIZE = 2048
EXPERT_PARAMS = 3 * HIDDEN_SIZE * MOE_INTERMEDIATE_SIZE  # gate + up + down


@dataclass(frozen=True)
class Scenario:
    name: str
    cpu_gweights_per_s_per_socket: float
    sockets_per_expert: int
    socket_shard_efficiency: float
    gpu_hot_hit_rate: float
    gpu_cached_expert_ms: float
    always_on_path_ms_per_token: float
    network_reduce_ms_per_moe_layer: float
    mtp_wall_speedup: float


def simulate(s: Scenario) -> dict:
    # Two socket shards do not automatically give exactly 2x.  With E shards,
    # effective speedup is 1 + (E-1)*efficiency.
    shard_speedup = 1.0 + (s.sockets_per_expert - 1) * s.socket_shard_efficiency

    cpu_expert_ms = (
        EXPERT_PARAMS
        / (s.cpu_gweights_per_s_per_socket * 1e9)
        / shard_speedup
        * 1000.0
    )

    # With 8 routed experts, a GPU cache only removes the CPU critical path for
    # a layer if all eight required experts are ready on the GPU fast path.
    # Partial hits still help bandwidth/energy, but at least one CPU miss can
    # remain the layer's latency critical path.
    p_all_8_gpu_ready = s.gpu_hot_hit_rate ** SELECTED_EXPERTS

    routed_layer_ms = (
        p_all_8_gpu_ready * s.gpu_cached_expert_ms
        + (1.0 - p_all_8_gpu_ready) * cpu_expert_ms
    )

    routed_total_ms = MOE_LAYERS * routed_layer_ms
    network_total_ms = MOE_LAYERS * s.network_reduce_ms_per_moe_layer

    no_spec_token_ms = (
        s.always_on_path_ms_per_token
        + routed_total_ms
        + network_total_ms
    )
    no_spec_tok_s = 1000.0 / no_spec_token_ms
    optimized_tok_s = no_spec_tok_s * s.mtp_wall_speedup

    return {
        **asdict(s),
        "expert_params": EXPERT_PARAMS,
        "effective_socket_shard_speedup": shard_speedup,
        "cpu_expert_ms_per_moe_layer": cpu_expert_ms,
        "p_all_8_gpu_ready": p_all_8_gpu_ready,
        "routed_ms_per_moe_layer": routed_layer_ms,
        "routed_total_ms_per_token": routed_total_ms,
        "network_total_ms_per_token": network_total_ms,
        "no_spec_token_ms": no_spec_token_ms,
        "no_spec_tok_s": no_spec_tok_s,
        "optimized_tok_s": optimized_tok_s,
    }


SCENARIOS = [
    Scenario(
        name="floor_one_socket_per_expert",
        cpu_gweights_per_s_per_socket=10.0,
        sockets_per_expert=1,
        socket_shard_efficiency=0.0,
        gpu_hot_hit_rate=0.60,
        gpu_cached_expert_ms=0.30,
        always_on_path_ms_per_token=80.0,
        network_reduce_ms_per_moe_layer=0.50,
        mtp_wall_speedup=1.00,
    ),
    Scenario(
        name="pessimistic_two_socket_expert",
        cpu_gweights_per_s_per_socket=12.0,
        sockets_per_expert=2,
        socket_shard_efficiency=0.80,
        gpu_hot_hit_rate=0.65,
        gpu_cached_expert_ms=0.30,
        always_on_path_ms_per_token=70.0,
        network_reduce_ms_per_moe_layer=0.45,
        mtp_wall_speedup=1.05,
    ),
    Scenario(
        name="conservative",
        cpu_gweights_per_s_per_socket=18.0,
        sockets_per_expert=2,
        socket_shard_efficiency=0.80,
        gpu_hot_hit_rate=0.75,
        gpu_cached_expert_ms=0.30,
        always_on_path_ms_per_token=50.0,
        network_reduce_ms_per_moe_layer=0.30,
        mtp_wall_speedup=1.10,
    ),
    Scenario(
        name="central",
        cpu_gweights_per_s_per_socket=25.0,
        sockets_per_expert=2,
        socket_shard_efficiency=0.80,
        gpu_hot_hit_rate=0.82,
        gpu_cached_expert_ms=0.30,
        always_on_path_ms_per_token=35.0,
        network_reduce_ms_per_moe_layer=0.20,
        mtp_wall_speedup=1.18,
    ),
    Scenario(
        name="strong",
        cpu_gweights_per_s_per_socket=32.0,
        sockets_per_expert=2,
        socket_shard_efficiency=0.80,
        gpu_hot_hit_rate=0.88,
        gpu_cached_expert_ms=0.30,
        always_on_path_ms_per_token=28.0,
        network_reduce_ms_per_moe_layer=0.15,
        mtp_wall_speedup=1.25,
    ),
]


def main() -> None:
    results = [simulate(s) for s in SCENARIOS]
    print(json.dumps({"schema": "bytemywave.glm52-4x2-sensitivity.v1", "results": results}, indent=2))


if __name__ == "__main__":
    main()
