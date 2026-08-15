# ADR 0004 — Calibrated Feasibility Map

Status: Accepted for Phase 4

Date: 2026-08-15

## Context

TensorWave Phase 1–3 prove mechanisms but do not answer the practical question:

> which workloads and model geometries can hide host-to-device weight transfer well enough to make a tiny-VRAM GPU useful?

A single benchmark point is insufficient. Dense decode, prefill, batch serving and MoE can have radically different weight reuse.

## Decision

TensorWave will maintain a **calibrated operating-envelope map** rather than classify a model only by parameter count or whether it fits in RAM.

The first-order dense-linear crossover is:

```text
M_cross = b * (1-r) * F / (2 * BW)
```

where:

```text
M   = activation rows reusing each streamed weight tile
b   = physical wire bytes per active parameter
r   = persistently resident weight fraction
F   = measured effective GEMM FLOP/s
BW  = measured physical H2D bytes/s
```

The map uses **measured effective** `F` and `BW` whenever available, not GPU peak specifications.

## Why parameter count cancels

For a dense streamed model:

```text
T_compute  ~= 2 * P * M / F
T_transfer ~= P * b * (1-r) / BW
```

Equating the two cancels `P`.

Therefore total parameter count controls absolute time and RAM requirements but not the first-order overlap crossover when both compute and streamed bytes scale linearly with active parameters.

## Required caveat

This is a roofline-style model. It omits or simplifies:

```text
dequantization
attention/KV traffic
launch/event overhead
non-overlappable kernels
PCIe startup effects
graph irregularity
MoE routing misses
cache misses
activation/workspace constraints
```

Those terms must be added only when measurements show they are required.

## Validation rule

The analytical map is not accepted as evidence by itself.

For each tested geometry, compare predicted crossover against measured:

```text
steady_starvation_pct
steady_hidden_transfer_pct
physical H2D GB/s
GEMM/dequant time
wall time
```

If prediction and measurement diverge, preserve the failure and improve the model. Do not tune inputs merely to produce a favorable result.

## Consequences

Positive:

- gives a falsifiable target before full model integration;
- explains why batch/prefill can work while dense batch=1 decode fails;
- separates physical bandwidth limits from implementation overhead;
- allows measured hardware to be compared without assuming peak TFLOPS;
- gives MoE/cache experiments a common framework.

Negative:

- first-order map can overestimate performance;
- effective compute depends on K/N/tile shape, so one TFLOPS value may not represent all shapes;
- measured maps may need one calibration per tile geometry.

## Implementation

```text
tools/build_feasibility_map.py
tools/calibrate_feasibility_map.py
tools/aggregate_feasibility_runs.py
scripts/run-feasibility-experiments.ps1
experiments/phase4-feasibility-map/README.md
maps/default/FEASIBILITY-MAP.md
```
