# 05 — Open Questions / Things That Must Be Demonstrated

This file separates **ideas discussed** from **claims that have actually been proven inside TensorWave**.

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

---

# Kimi K3 distributed-cluster open questions

The following questions apply to the DDR2/DDR3/DDR4 cluster track and are deliberately kept separate from the original RAM→GPU questions.

## K3-A. What is the exact per-token compressed-weight traffic?

The 104B activated-parameter figure and MXFP4 headline quantization give a useful lower-bound model, but `config.json` shows that not all paths use the same 4-bit representation.

Need `tw-k3-inspect` to calculate the exact byte ownership and, where possible, the exact selected-weight bytes for:

```text
attention/KDA/MLA
router
shared experts
routed experts
norms
lm_head
vision path
scale/quant metadata
```

The throughput model currently uses ~58 GB/token only as a screening estimate.

## K3-B. Can old CPUs execute the released MXFP4 format efficiently enough?

Need real kernels and measurements on the exact purchased CPU.

Required ISA paths may include:

```text
scalar reference
SSE2
SSE4.x
AVX
AVX2
AVX-512 on modern route
```

The proof is not “CPU can mathematically decode MXFP4”; the proof is measured compressed-weight throughput on real K3 tensor shapes.

## K3-C. How much single-token bandwidth can expert parallelism actually aggregate?

K3 selects 16 experts/token, but real router popularity may be skewed.

Measure:

```text
selected expert histogram
selected experts per physical node
slowest expert owner per layer
expert queue depth
load imbalance
benefit of round-robin vs frequency-aware placement
benefit/cost of replicating hot experts
```

## K3-D. What should remain layer-local vs expert-sharded?

Need to compare:

- attention/local state replicated vs sharded;
- shared experts local vs remote;
- routed experts sharded globally vs within stage groups;
- tensor parallelism inside a stage;
- pure pipeline stages;
- hybrid stage + expert groups.

The fastest topology may differ between 5 DDR2 nodes, 3 R920 nodes and 10 dual-EPYC boards.

## K3-E. Is 10 GbE sufficient for the first prototype?

Do not answer from link bandwidth alone.

Need real K3-like repeated expert dispatch benchmark measuring:

```text
payload bytes/layer
collective latency/layer
p50/p95/p99
CPU pack/unpack overhead
head-of-line blocking
```

40 GbE / InfiniBand / faster fabrics should be tested only if 10 GbE latency or bandwidth is demonstrated to be the limiter.

## K3-F. Can a 1.56 TB full model remain entirely DRAM-resident?

Capacity must include more than raw checkpoint bytes:

- OS and daemons;
- local runtime buffers;
- activation buffers;
- KDA/KV state;
- transport buffers;
- allocator fragmentation;
- model manifest/metadata.

Configurations that merely equal 1.56 TB nominal are rejected for full-RAM production testing.

## K3-G. Is SSD tiering useful at all for interactive decode?

If aggregate RAM is below checkpoint size, a cold-expert SSD cache is technically possible.

Need to measure:

```text
expert miss rate
SSD random read latency
prefetch hit rate
cache residency distribution
stall time/token
```

The current procurement preference is enough DRAM to avoid SSD weight faults during steady-state decode.

## K3-H. Can distributed output match the reference implementation?

Required correctness ladder:

```text
P0 tensor inventory
P1 primitive numerical tests
P2 one real K3 layer
P3 two-process split
P4 full 93-layer forward
P5 multi-token generation
P6 OpenAI-compatible endpoint
```

Any performance result before this ladder passes is a kernel/network experiment, not a demonstrated K3 inference result.

## K3-I. Can the cluster reach the initial >=1 decoded token/s target?

Current screening model:

```text
~58 GB active compressed-weight traffic/token (rough estimate)
~208 GFLOP/token linear-work estimate
```

Initial procurement screening threshold with margin:

```text
>= 72.5 GB/s effective critical-path parallel weight stream
>= 312 GFLOP/s effective useful kernel throughput
```

This still does **not** prove 1 token/s because serial layer work and network synchronization must fit inside the same one-second budget.

The actual proof is:

```text
measured full-model T_token <= 1.0 s
```

for a defined batch/context/prompt after warm-up.

## K3-J. Which hardware route wins after power and bandwidth are included?

Compare at least:

```text
HP DL785 G6 / Sun X4640 / X4600 M2 DDR2
Dell R920 DDR3
Supermicro H12DGQ-NT6 + EPYC DDR4
```

Metrics:

```text
acquisition RON
usable TB
measured DRAM GB/s
measured expert kernel GB/s
network latency
wall power
real K3 tok/s
RON / tok/s
watts / tok/s
```

The cheapest chassis is not automatically the cheapest inference system.
