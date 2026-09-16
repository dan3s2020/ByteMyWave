# Transit V16 real Qwen laptop result — 2026-09-16

Tensor: `blk.0.ffn_gate.weight`

GGUF Q4_K tensor geometry used by the GEMV path:

- GEMV: 9,216 output rows × 2,560 input columns
- logical weights: 23,592,960
- Q4_K storage: 12.656 MiB
- build target: Native (`-march=native -mtune=native`)
- compiled path: AVX2 late-scale x8/x16 + F16C/FMA where available
- AVX-VNNI: enabled (`vpdpbusd` production path)

## Correctness

All V15/V16 paths matched the scalar Q4_K × Q8_K operator to floating-point accumulation-order noise only:

- relative L2: `2.139e-07`
- cosine: `1.000000000`
- max absolute difference: `4.172325134e-07`

This is not a codebook or approximate-weight path. The underlying Q4_K weights are unchanged; differences are from accumulation order / floating-point rounding.

## Single-thread autotune

| Kernel | ms/GEMV | logical GOP/s | vs V15 |
|---|---:|---:|---:|
| V15 x8meta | 0.738 | 31.96 | 1.000x |
| V16 x8 PF0 | 0.630 | 37.43 | 1.171x |
| V16 x8 PF1 | 0.646 | 36.53 | 1.143x |
| V16 x8 PF2 | 0.716 | 32.97 | 1.032x |
| V16 x8 PF4 | 0.621 | 37.97 | 1.188x |
| V16 x16 PF0 | 0.602 | 39.20 | 1.226x |
| V16 x16 PF1 | 0.790 | 29.88 | 0.935x |
| V16 x16 PF2 | 0.710 | 33.25 | 1.040x |
| V16 x16 PF4 | 0.594 | 39.70 | 1.242x |
| V16 VNNI PF0 | **0.520** | **45.36** | **1.419x** |
| V16 VNNI PF1 | 0.801 | 29.44 | 0.921x |
| V16 VNNI PF2 | 0.536 | 44.05 | 1.378x |
| V16 VNNI PF4 | 0.568 | 41.55 | 1.300x |

Best single-thread V16 result: **0.520 ms/GEMV**, **45.36 logical GOP/s**, **1.419× over the in-binary V15 x8meta control**.

V16 production layout overhead: **+5.556% RAM** relative to raw Q4_K storage.

For context, the earlier standalone V15 run on the same laptop reported `packed SIMD control = 1.254 ms/GEMV` and `x8meta = 0.894 ms/GEMV`. Cross-run timings are affected by binary/layout/build/thermal/cache differences and should not be treated as a strict same-binary ratio, but the new V16 single-thread result is 0.520 ms.

## Thread scaling

Persistent-work simulation using x16 PF2:

| Threads | ms/GEMV | logical GOP/s | vs V15 single |
|---:|---:|---:|---:|
| 1 | 0.657 | 35.92 | 1.12x |
| 2 | 0.311 | 75.77 | 2.37x |
| 4 | 0.160 | 147.30 | 4.61x |
| 8 | **0.109** | **215.93** | **6.76x** |

These multi-thread figures are a persistent-work/operator scaling test, not a full-model token/s claim. Full Qwen decode may become limited by memory bandwidth, synchronization, other operators, scheduling, cache reuse and model-wide tensor streaming.

## Interpretation

The single-thread 0.520 ms result exceeds the stated 0.6–0.7 ms target. The key winning ingredients are:

- x16 fused row scheduling;
- predecoded metadata / runtime-friendly RAM layout;
- late scaling;
- AVX-VNNI `vpdpbusd`;
- prefetch autotuning;
- a small memory expansion instead of lossy weight replacement.

The next production gate should be integration into the actual llama.cpp CPU path, then full-model `llama-bench` / decode testing with the same model. The best laptop single-thread production choice from this run is `V16 VNNI PF0`; the thread-scaling path currently benchmarks x16 PF2 and should be updated to benchmark VNNI + affinity-aware workers as a separate next optimization.
