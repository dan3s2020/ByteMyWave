# TensorWave Maps

This directory separates **analytical maps** from **measured maps**.

## `default/`

Committed roofline-style baseline using explicit assumptions. It is useful for reasoning and for checking equations, but it is not evidence that a target GPU behaves that way.

Current committed baseline:

- H2D: 12 GB/s
- effective dense-linear compute: 10 TFLOP/s
- Q4-v1 wire density: 0.625 bytes/parameter
- 0% persistent weight residency
- dense active fraction 100%

See [`default/FEASIBILITY-MAP.md`](default/FEASIBILITY-MAP.md).

## Measured maps

Measured maps should normally live under `runs/phase4-feasibility-*` and are **not automatically committed**, because raw benchmark runs can be machine-specific and large.

Flow:

```text
Phase-3 real Q4 run
        |
        v
calibrate_feasibility_map.py
        |
        v
measured H2D GB/s + measured effective GEMM TFLOP/s
        |
        v
build_feasibility_map.py
        |
        +-- feasibility-map.json
        +-- feasibility-map.csv
        +-- FEASIBILITY-MAP.md
        +-- feasibility-map.svg
```

## Key equation

For ideal dense streaming:

```text
M_cross = bytes_per_param * (1-resident_fraction) * effective_FLOPS
          ---------------------------------------------------------
                         2 * H2D_bandwidth
```

At `M << M_cross`, transfer dominates.

At `M ~ M_cross`, the runtime is balanced.

At `M > M_cross`, the next weight transfer can theoretically be hidden under current compute.

Real measured starvation supersedes the analytical prediction.
