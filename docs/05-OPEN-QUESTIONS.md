# 05 — Open Questions / Things That Must Be Demonstrated

This file separates **ideas discussed** from **claims that have actually been proven inside ByteMyWave**.

At repository initialization, none of the following has yet been demonstrated by code in this repository.

## A. Can a 4 GB GPU remain busy while a much larger model lives in RAM?

Required measurements:

```text
compute_time(tile)
H2D_time(tile)
overlap_time(tile)
uncovered_transfer_time(tile)
GPU_starvation_time
GPU_utilization
```

The first proof should use a single real Transformer/DiT block or representative matrix path before attempting the complete H3 pipeline.

## B. What tile size is optimal?

Too large:

- consumes too much VRAM;
- reduces buffering flexibility.

Too small:

- launch overhead dominates;
- PCIe transaction overhead grows;
- GEMM shapes become inefficient;
- GPU utilization can collapse.

Need to sweep tile sizes experimentally.

## C. How much data can truly be transferred while compute is active?

Theoretical PCIe bandwidth is not enough.

Need real measurements for:

- pageable host memory;
- pinned host memory;
- different transfer sizes;
- one copy engine vs overlapping streams;
- simultaneous kernels;
- laptop GPU platform vs desktop/server platform;
- PCIe generation and actual lane count.

## D. Is ordinary DDR3 sufficient when PCIe is the tighter bottleneck?

The discussion proposed that cheap multi-channel DDR3 ECC may be useful if:

```text
RAM bandwidth > sustained H2D PCIe requirement
```

This must be measured on real old Xeon/server platforms. Latency, NUMA placement and chipset topology may matter as much as headline bandwidth.

## E. Can MiniMax H3 be tiled below whole-layer granularity efficiently?

Need to inspect actual H3 tensor shapes and runtime implementation.

Questions:

- which projections can be split along output dimension?
- which can be split along input/K dimension with partial accumulation?
- what temporary activations must remain resident?
- which operations impose full-tensor dependencies?
- how much VRAM is irreducibly required for latent/attention/workspace?

## F. Which H3 parameters can remain resident and which should stream?

Need profiling-based classes:

```text
HOT      -> permanently resident in VRAM
WARM     -> likely to be reused soon / cache if room
STREAM   -> deterministic H2D just before use
CPU      -> execute or retain on host if transfer is pointless
SKIP     -> reusable cached result where mathematically valid
```

## G. How much can H3 be quantized without unacceptable quality loss?

Need comparisons across Q8/Q6/Q5/Q4/Q3/Q2 or other available formats.

Measure both:

- visual/audio quality;
- runtime performance.

## H. Can dequantization be fused enough that expanded weights never occupy significant VRAM?

Need to determine whether existing kernels already provide suitable fused quantized GEMM paths.

Only write custom kernels if measurement shows a concrete bottleneck.

## I. Can execution be compiled into a static transfer schedule?

For deterministic paths, build an execution trace in advance.

Need to determine which parts of H3 vary dynamically based on:

- input dimensions;
- guidance configuration;
- denoising step count;
- caching decisions;
- conditionals inside runtime/framework.

A static plan may have multiple precompiled variants rather than one universal sequence.

## J. How much host CPU involvement can be removed?

Target hot path:

```text
prebuilt offsets + async copy enqueue + dependency events + kernel launch
```

Need to measure whether Python/framework overhead is relevant enough to justify a C/C++ runtime or CUDA Graph-style capture.

## K. Does zero-copy host access help anywhere?

Potentially useful for one-shot or very cold tensors, but discrete GPU access to host RAM can be much slower than VRAM.

Need benchmark rather than assumption.

Compare:

```text
copy compressed tile -> VRAM -> compute
vs
GPU directly reads mapped pinned host data
```

## L. Can model structure provide real compression beyond quantization?

Research questions:

- cross-layer low-rank commonality;
- shared bases;
- vector/codebook quantization;
- predictable/reconstructable tiles;
- semantic similarity graph usefulness.

This is exploratory and must not delay the basic streaming proof.

## M. Can missing/corrupt tiles be reconstructed?

Two separate goals:

1. exact recovery using parity/erasure coding;
2. approximate recovery using structural neighbors/shared basis.

These should be tested independently.

## N. What constitutes success for MiniMax H3?

Not merely avoiding OOM.

Suggested success ladder:

1. representative block executes correctly with 4 GB VRAM cap;
2. transfer overlaps compute measurably;
3. GPU starvation is low enough to justify architecture;
4. complete H3 graph runs under strict VRAM cap;
5. output is valid;
6. generation time is useful compared with conventional CPU offload;
7. architecture generalizes to another large model.

## First proposed experiment from the conversation

Use the existing 4 GB NVIDIA GPU as the test platform and build a microbenchmark around a real H3-like block.

Measure only three fundamental values first:

```text
compute_time(tile)
H2D_time(tile)
unhidden_transfer_time(tile)
```

Those numbers determine whether deeper implementation work is justified.
