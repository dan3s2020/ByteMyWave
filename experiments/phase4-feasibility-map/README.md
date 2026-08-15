# Phase 4 — Feasibility Map: experiments that can falsify TensorWave

## Purpose

Phase 1–3 build mechanisms. Phase 4 asks a different question:

> **where does the mechanism actually work?**

The result must be an operating envelope, not a single benchmark number.

The map axes are:

```text
M / activation rows / batch / prefill reuse
tile geometry K,N
wire bytes per parameter
measured H2D GB/s
measured effective GEMM TFLOP/s
dequant time
persistent residency/cache fraction
active parameter fraction (dense vs MoE)
```

The primary output is the measured crossover between transfer-bound and compute-bound execution.

---

# Experiment E1 — Measure the real M crossover

## Hypothesis

For a fixed real tensor family and tile geometry, increasing `M` increases compute per transferred tile without increasing weight H2D bytes. Therefore a crossover should appear where:

```text
T_compute(tile) ~= T_H2D(next tile)
```

and starvation should collapse.

## Sweep

Recommended first sweep:

```text
M = 1,4,16,64,128,256,512,1024,2048
TileN = 256
MaxTiles = 32 or 64
```

Run on Phase 3 Q4 because it is the intended compressed-wire path.

## Measure

Per M:

```text
physical H2D GB/s
H2D ms
dequant ms
GEMM ms
steady starvation ms
steady starvation %
hidden transfer %
wall time
correctness
```

## Success condition

The map is supported if measured starvation falls as `M` increases and a reproducible crossover region exists.

Strong TensorWave support for a geometry remains:

```text
correctness_ok = true
steady_starvation_pct <= 10%
steady_hidden_transfer_pct >= 80%
```

## Failure condition

If starvation remains high even when GEMM time substantially exceeds measured H2D time, then the runtime is not achieving real concurrency and the scheduling/stream/event assumptions are wrong.

---

# Experiment E2 — Tile-size crossover

## Hypothesis

There is a tile size large enough to make H2D efficient but small enough to fit the fixed VRAM ring and preserve useful overlap.

Very small tiles can lose to:

```text
launch overhead
PCIe transaction overhead
event/synchronization overhead
poor GEMM efficiency
```

Very large tiles can lose to:

```text
VRAM footprint
long copy bursts
reduced scheduling flexibility
activation/workspace pressure
```

## Sweep

```text
TileN = 64,128,256,512
M = 1,16,64,128,256,512,1024
same K/tensor family when possible
same MaxTiles
```

## Output

For each tile geometry find:

```text
minimum starvation
M at first <=10% starvation
H2D GB/s
effective GEMM TFLOP/s
fixed VRAM bytes
```

The best tile geometry is not necessarily the one with the highest H2D GB/s. It is the one with the best end-to-end starvation/wall-time tradeoff.

---

# Experiment E3 — 16-bit wire vs Q4 wire

## Hypothesis

Keeping weights compressed across RAM and PCIe should reduce exposed H2D enough to move the crossover to smaller `M`, even after paying GPU dequantization.

## Method

Use the same real checkpoint, same K, same TileN, same tile count and same M sweep.

Compare:

```text
Phase 2:
F16/BF16 RAM -> F16/BF16 H2D -> GEMM

Phase 3:
Q4 RAM -> Q4 H2D -> GPU dequant -> FP16 GEMM
```

## Required comparisons

```text
physical bytes transferred
H2D time
starvation
wall time
GEMM time
Q4 dequant time
correctness
source-equivalent feed rate
```

## Strong result

Q4 should materially lower starvation or wall time at the same M/tensor geometry.

## Negative result

If Q4 H2D shrinks 3.2x but wall/starvation barely improves, then another bottleneck dominates. Candidates:

```text
dequant kernel
global-memory write/read of full FP16 tile
GEMM inefficiency
synchronization
lack of real copy/compute overlap
```

That result directly motivates Phase 4 fused decode/MMA.

---

# Experiment E4 — Dense LLM decode floor

This experiment initially uses the calibrated map rather than claiming a complete LLM runtime.

## Question

Given measured effective H2D and compute, what is the physical lower bound if every dense active weight must be streamed once per token?

For model parameters `P`:

```text
stream_bytes = P * bytes_per_param * (1-resident_fraction)
T_transfer = stream_bytes / measured_H2D
T_compute = 2 * P * M / measured_effective_FLOPS
T_ideal = max(T_transfer, T_compute)
```

Evaluate at:

```text
P = 7B,13B,33B,70B,120B
M = 1,4,16,64,128,256,512,1024,2048
```

## Interpretation

`M=1` approximates the dense single-sequence decode reuse problem.

Large `M` approximates batching/prefill reuse.

This experiment tells us when a full graph integration is worth attempting.

---

# Experiment E5 — MoE active-fraction map

## Hypothesis

MoE is attractive because only active experts need to cross PCIe if routing is known early enough.

Generate maps for:

```text
active_fraction = 1.0, 0.50, 0.25, 0.125, 0.0625
```

The ideal compute/transfer crossover is unchanged if active compute and active streamed bytes shrink proportionally, but **absolute step latency** drops sharply.

Then model cache residency:

```text
resident_fraction = 0,0.25,0.50,0.75
```

This approximates a hot-expert cache.

A later real-MoE runtime must measure:

```text
expert cache hit rate
unique active experts per batch
router lead time
expert prefetch miss rate
```

---

# Experiment E6 — Prediction vs measurement

The analytical map is useful only if it predicts the measured crossover.

For each real run:

1. calibrate measured `H2D GB/s`;
2. estimate effective GEMM TFLOP/s;
3. compute predicted `M_cross`;
4. identify measured first M with <=10% starvation;
5. compare predicted vs measured.

If prediction error is large, extend the model with measured terms:

```text
T_dequant
T_launch/event
copy startup/latency
non-overlappable kernels
activation traffic
```

Do not tune equations to make TensorWave look good. The map should expose failures.

---

# First hardware run

Fastest useful run on the target 4 GB NVIDIA GPU:

```powershell
# from research/feasibility-map-v1
.\scripts\run-feasibility-experiments.ps1 `
  -ModelDir "D:\models\some-safetensors-model" `
  -TileN @(128,256) `
  -M @(1,4,16,64,128,256,512,1024) `
  -MaxTiles 32
```

For a very quick smoke run:

```powershell
.\scripts\run-feasibility-experiments.ps1 `
  -ModelDir "D:\models\some-safetensors-model" `
  -TileN @(256) `
  -M @(1,16,64,256,512) `
  -MaxTiles 16
```

The runner should create one top-level Phase-4 directory containing pointers/calibrations/maps for the underlying Phase-2/Phase-3 runs.

---

# What would prove the project useful?

Not merely “it runs.”

Evidence should show at least one practically relevant region where:

```text
correctness passes
compressed H2D is real
fixed VRAM remains bounded
starvation <=10%
map prediction roughly matches measurement
wall time is materially better than sequential/offload baseline
```

For LLMs, the likely useful regions are expected to be:

```text
prefill
batched serving
MoE / sparse active experts
```

Dense interactive batch=1 decode is intentionally treated as the hardest adversarial case.
