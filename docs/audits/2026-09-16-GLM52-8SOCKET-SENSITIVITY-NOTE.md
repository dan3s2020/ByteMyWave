# GLM-5.2 8-socket counterfactual sensitivity note — 2026-09-16

Status: **analytical sanity check only; not a benchmark and not a prediction**.

Purpose: prevent the GLM-5.2 branch's 16-socket R920 sensitivity envelope from being copied directly onto the current four dual-socket HP Gen8 cluster.

Source simulator audited read-only:

- branch: `research/glm52-4x2-optimized-runtime-2026-08-18`
- `tools/glm52_4x2/simulate_4x2_sensitivity.py`
- `docs/21-GLM52-4X2-SIMULATION-CORRECTION-2026-08-18.md`

The source simulator explicitly models four 4-socket servers / 16 CPU socket domains and its main scenarios use two CPU socket shards per selected expert. GLM-5.2 selects eight routed experts per MoE layer, so this mapping can activate:

```text
8 selected experts × 2 socket shards/expert = 16 socket domains
```

The current HP cluster has:

```text
4 servers × 2 CPU sockets/server = 8 CPU sockets total
```

A natural all-CPU top-8 mapping can therefore give one CPU socket to each selected expert simultaneously, but cannot give every selected expert two CPU socket shards simultaneously without serialization or replacing some critical-path experts with GPU-resident execution.

## Counterfactual calculation

This note preserves each source scenario's CPU Gweights/s assumption, GPU hot-hit rate, always-on path, network latency and MTP wall-clock multiplier. It changes only:

```text
sockets_per_expert: 2 -> 1
```

for the scenarios that originally used two socket shards.

Using the source simulator equation:

```text
EXPERT_PARAMS = 3 × 6144 × 2048 = 37,748,736 weights
cpu_expert_ms = EXPERT_PARAMS / socket_Gweights_per_s / shard_speedup
P(all 8 GPU ready) = hit_rate^8
routed_layer_ms = Pgpu × gpu_ms + (1-Pgpu) × cpu_expert_ms
no_spec_token_ms = always_on_ms + 75 × routed_layer_ms + 75 × network_ms
optimized_tok_s = 1000/no_spec_token_ms × mtp_wall_speedup
```

results become:

| Source scenario assumptions retained | 16-socket source result | 8-socket / 1-shard counterfactual |
|---|---:|---:|
| floor / one socket already | 2.52 tok/s | 2.52 tok/s |
| pessimistic | 4.54 tok/s | 3.15 tok/s |
| conservative | 7.17 tok/s | 5.09 tok/s |
| central | 11.28 tok/s | 8.16 tok/s |
| strong | 15.86 tok/s | 12.02 tok/s |

## Hypothetical impact of a future 1.11× GLM expert kernel

The current measured x8meta result is Q4_K×Q8_K and **does not directly apply** to the selected GLM Q3 build, whose routed experts are Q2_K/Q3_K. However, this sanity check asks a useful architectural question:

> If a future exact Q2_K/Q3_K row-interleaved/predecoded kernel reproduced the same 1.11× local CPU-expert speedup, how much would the uncalibrated 8-socket sensitivity model move?

Mechanically multiplying only each scenario's assumed CPU expert Gweights/s by `1.11` gives:

| 8-socket scenario | Before hypothetical expert kernel | After 1.11× CPU-expert rate | Full-model delta |
|---|---:|---:|---:|
| pessimistic | 3.15 tok/s | 3.38 tok/s | +7.3% |
| conservative | 5.09 tok/s | 5.44 tok/s | +6.9% |
| central | 8.16 tok/s | 8.69 tok/s | +6.6% |
| strong | 12.02 tok/s | 12.71 tok/s | +5.7% |

This is an **Amdahl-effect illustration**, not a prediction. It deliberately leaves GPU hit rate, always-on work, network latency and MTP unchanged. It shows why even a real +11% expert-kernel improvement should not be reported as +11% model tok/s.

## Interpretation boundary

These are **not expected HP DL360p tok/s**.

The source scenarios assume per-socket expert rates of 10/12/18/25/32 Gweights/s plus specific GPU cache-hit, always-on, network and MTP values. None of those values has yet been calibrated on the current E5-2680 v2 servers and current GPU/network inventory.

This counterfactual proves only one point:

> the 16-socket GLM sensitivity numbers contain a material benefit from two-socket-per-expert same-layer parallelism and must be reduced/recalibrated for an 8-socket cluster.

It also does not include the Q4_K x8meta multiplier as a current implementation result. The public GLM Q3_K_M layout audited by the source branch uses Q2_K/Q3_K routed experts, so the current Q4_K×Q8_K kernel is not directly applicable.

The next meaningful GLM CPU measurement on the HP hardware is one real Q2_K/Q3_K expert in NUMA-local memory, reported as Gweights/s per socket, followed by a one-socket-per-expert eight-socket critical-path calibration.