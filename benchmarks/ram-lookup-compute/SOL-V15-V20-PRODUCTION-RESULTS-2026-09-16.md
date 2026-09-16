# Transit Sol kernel lineage V15 → V20 — measured production results (2026-09-16)

Status: **measured on the project laptop using the real Qwen Q4_K tensor**.

Provenance / collision-avoidance note: this file records only the kernel lineage produced in the current Transit Sol conversation (`V15` through `V20`). It deliberately does **not** merge, overwrite, or reinterpret the other agents' native-control files (`NATIVE-Q4K-Q8K-*`, V11, exact-LUT/PQ, etc.). Those remain separate evidence tracks.

## Fixed workload

```text
model blob      : qwen3.5:4b-q4_K_M local Ollama GGUF blob
tensor          : blk.0.ffn_gate.weight
GGUF ne order   : 2560 x 9216
real GEMV       : 9216 output rows x 2560 input columns
logical weights : 23,592,960
Q4_K bytes      : 13,271,040 = 12.656 MiB
activation      : Q8_K
platform        : Intel Core i5-12500H, Windows PowerShell, GCC/MinGW
```

All accepted V15-V20 paths preserve the original Q4_K weights. No learned codebook or lossy weight replacement is used.

## V15 — exact native x8 / x8meta

Real user run, 15 iterations.

### Native laptop build

```text
scalar exact            2.279 ms   10.35 logical GOP/s
packed SIMD control     1.254 ms   18.81 logical GOP/s
llama-x8 fused          1.303 ms   18.11 logical GOP/s
x8meta fused            0.894 ms   26.39 logical GOP/s

llama-x8 / packed       0.963x
x8meta / packed         1.403x
x8meta RAM overhead     +2.78%
```

Correctness in that run:

```text
packed SIMD   rel-L2=0  cosine=1.000000000  max_abs=0
llama x8      rel-L2=0  cosine=1.000000000  max_abs=0
x8meta        rel-L2=0  cosine=1.000000000  max_abs=0
```

### Ivy-compatible ISA build, still executed on the laptop

```text
scalar exact            3.320 ms    7.11 logical GOP/s
packed SIMD control     1.331 ms   17.72 logical GOP/s
llama-x8 fused          1.295 ms   18.22 logical GOP/s
x8meta fused            1.238 ms   19.07 logical GOP/s

llama-x8 / packed       1.028x
x8meta / packed         1.076x
x8meta RAM overhead     +2.78%
```

Boundary: this is only an Ivy-ISA proxy build; it is **not** a measurement on the HP DL360p Gen8 Xeons.

## V16 — x16, late-scale, AVX-VNNI, prefetch autotune

Runtime layout expansion: `1.05556x Q4_K` (+5.556%).

Correctness versus the scalar Q4_K×Q8_K operator:

```text
relative L2             2.139e-07
cosine                  1.000000000
max absolute difference 4.172325134e-07
```

The non-zero values are floating-point accumulation-order noise; the quantized weights are unchanged.

Single-thread user result:

| Kernel | ms/GEMV | logical GOP/s | vs in-binary V15 |
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
| **V16 VNNI PF0** | **0.520** | **45.36** | **1.419x** |
| V16 VNNI PF1 | 0.801 | 29.44 | 0.921x |
| V16 VNNI PF2 | 0.536 | 44.05 | 1.378x |
| V16 VNNI PF4 | 0.568 | 41.55 | 1.300x |

Persistent-work x16 PF2 scaling:

```text
1 thread   0.657 ms    35.92 GOP/s
2 threads  0.311 ms    75.77 GOP/s
4 threads  0.160 ms   147.30 GOP/s
8 threads  0.109 ms   215.93 GOP/s
```

## V17 — activation prepack + persistent VNNI worker pool

Real user run.

Single-thread:

| Kernel | ms/GEMV | logical GOP/s | vs V15 |
|---|---:|---:|---:|
| V17 x8 VNNI PF0 | 0.789 | 29.89 | 1.028x |
| V17 x8 VNNI PF1 | 0.688 | 34.28 | 1.179x |
| V17 x8 VNNI PF2 | 0.637 | 37.06 | 1.275x |
| V17 x8 VNNI PF4 | 0.576 | 40.95 | 1.409x |
| V17 x16 VNNI PF0 | 0.565 | 41.74 | 1.436x |
| V17 x16 VNNI PF1 | 0.535 | 44.08 | 1.517x |
| V17 x16 VNNI PF2 | 0.551 | 42.80 | 1.472x |
| **V17 x16 VNNI PF4** | **0.514** | **45.94** | **1.580x** |

Persistent worker autotune, dispatch included:

```text
1 thread   PF0   0.545 ms    43.25 GOP/s
2 threads  PF4   0.243 ms    97.13 GOP/s
4 threads  PF0   0.151 ms   156.66 GOP/s
8 threads  PF4   0.124 ms   189.65 GOP/s
```

## V18 — hybrid-aware Windows scheduler/autotune

V18 fixed the worker-placement search and tested dynamic/static scheduling, CPU-set placement, chunk size, and prefetch.

Important measurement-boundary discovery: the best hot-cache result is much faster than plausible DRAM streaming of the 13+ MiB runtime layout, so it must be treated as a hot-cache/operator number rather than full-model memory performance.

V17 persistent numbers when rerun inside the V18 LTO binary:

```text
1T   0.550 ms
2T   0.320 ms
4T   0.116 ms
8T   0.081 ms
16T  0.118 ms
```

V18 detected 11 CPU sets because this version accidentally excluded Windows CPU sets marked parked at that instant.

Best V18:

```text
V18 BEST        0.064 ms
throughput      370.96 logical GOP/s
configuration   11 threads, dynamic-physical, PF0, chunk 4
V18 / V15       12.78x
```

This `0.064 ms` result is **hot/cache-resident**. It must not be converted into a DDR or full-model token/s claim.

## V19 — 521 MiB streaming/cold-weight gate

V19 rotates GEMVs through different copies of the runtime weight layout to force a large working set.

```text
runtime/copy            13.359 MiB
streaming copies        39
working set             521.016 MiB
```

Correctness:

```text
rel-L2=0
cosine=1.000000000
max_abs=0
```

Measured streaming autotune:

```text
1T unpinned   hot 0.714 ms   stream 0.779 ms   18.0 GB/s
1T physical   hot 0.665 ms   stream 0.770 ms   18.2 GB/s
2T unpinned   hot 0.477 ms   stream 0.470 ms   29.8 GB/s
2T physical   hot 0.340 ms   stream 0.431 ms   32.5 GB/s
4T unpinned   hot 0.111 ms   stream 0.323 ms   43.4 GB/s
4T physical   hot 0.116 ms   stream 0.323 ms   43.4 GB/s
```

Best:

```text
V19 STREAMING BEST      0.323 ms
configuration           4T physical PF4 CH8
effective weight BW     43.38 GB/s
logical throughput      73.07 GOP/s
same-config hot         0.116 ms
streaming/hot penalty   2.78x
working set             521.02 MiB
```

V19 still had the CPU-set bug: parked CPU sets were excluded, so only 5 logical CPU sets were exposed in that run.

## V20 — production streaming, full CPU-set enumeration, ZERO vs COMPACT

V20 fixes the parked-CPU-set mistake, resets affinity between tests, and compares two exact layouts.

```text
ZERO
  runtime/copy          12.656 MiB
  overhead              0.00%
  d/dmin                FP16
  scales/mins           exact packed/transposed metadata

COMPACT
  runtime/copy          13.008 MiB
  overhead              +2.78%
  d/dmin                FP16
  scales/mins           pre-unpacked
```

Correctness:

```text
ZERO     rel-L2=0  cosine=1.000000000  max_abs=0
COMPACT  rel-L2=0  cosine=1.000000000  max_abs=0
```

Windows CPU-set snapshot:

```text
logical CPU sets total  16
parked at snapshot      11
threads tested          up to 16
physical-first plan     0 2 4 6 8 9 10 11 12 13 14 15 1 3 5 7
```

ZERO stream (518.91 MiB):

```text
1T   0.759 ms   17.5 GB/s
2T   0.405 ms   32.8 GB/s
4T   0.296 ms   44.8 GB/s
6T   0.279 ms   47.6 GB/s
8T   0.270 ms   49.2 GB/s
10T  0.270 ms   49.2 GB/s
12T  0.266 ms   49.9 GB/s
16T  0.244 ms   54.3 GB/s
```

COMPACT stream (520.31 MiB):

```text
1T   0.769 ms   17.7 GB/s
2T   0.428 ms   31.8 GB/s
4T   0.293 ms   46.6 GB/s
6T   0.287 ms   47.6 GB/s
8T   0.284 ms   48.1 GB/s
10T  0.274 ms   49.8 GB/s
12T  0.228 ms   60.0 GB/s
16T  0.228 ms   60.0 GB/s
```

Best measured V20:

```text
V20 STREAMING BEST      0.228 ms
layout                  COMPACT
configuration           12T physical static PF0 T0 CH4
effective weight BW     59.95 GB/s
logical throughput      103.71 GOP/s
runtime overhead        +2.78%
```

Relative to V19's corrected large-working-set objective, V20 improves `0.323 → 0.228 ms` (1.417x higher iteration rate) and measured runtime-layout bandwidth `43.38 → 59.95 GB/s` (1.382x).

## Conclusions from this lineage

### 1. The successful lever is exact runtime layout, not lossy weight substitution

The codebook/PQ branch was abandoned. V15-V20 keep the real Q4_K values and optimize how the CPU sees them.

### 2. Predecoded metadata is worth a small RAM expansion

V20's `COMPACT +2.78%` layout beats the storage-neutral `ZERO` layout under streaming. On this laptop, saving decode work is worth more than the extra 2.78% of streamed bytes.

### 3. AVX-VNNI matters in hot compute, but DRAM becomes the real limiter

V16/V17 show strong single-thread VNNI gains. V18 shows enormous hot-cache multi-thread rates. V19/V20 show that once the working set is hundreds of MiB, throughput collapses toward the memory subsystem limit.

### 4. V18 is not a full-model speed claim

The `0.064 ms / 370.96 GOP/s` result is retained because it is a valid hot-cache operator measurement. It is not used as a DDR or token/s claim.

### 5. V20 is the strongest current host-memory result in this lineage

```text
real Qwen Q4_K tensor
exact Q4_K×Q8_K arithmetic
~520 MiB rotating working set
12 physical-core-first workers
0.228 ms/GEMV
59.95 GB/s runtime-layout bandwidth
103.71 logical GOP/s
+2.78% runtime weight storage
```

### 6. The next optimization target is bandwidth efficiency, not more arithmetic tricks

At ~60 GB/s streaming, further speedup must come from reaching a larger fraction of sustainable raw read bandwidth, reducing bytes transferred per exact Q4_K GEMV, lowering exact metadata-decode cost without increasing streamed bytes, and improving page/TLB behavior and memory-level parallelism.

### 7. No token/s claim yet

These are real tensor/operator measurements. They are not end-to-end Qwen decode token/s. The production gate remains integration into the actual llama.cpp/Ollama CPU path and full-model measurement.

## Current decision

Continue the exact-Q4_K line. Do **not** return to global learned codebooks for this host CPU path. V21 should attack the V20 streaming ceiling with a zero-overhead faster metadata representation, raw-read ceiling measurement, and tighter physical-core/static bandwidth scheduling.
