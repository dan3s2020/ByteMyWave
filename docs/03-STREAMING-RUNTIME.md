# 03 — Streaming Runtime

## Core execution model

The GPU does not independently decide which part of a neural network it wants next. The host runtime knows the computation graph and launches operations in a known order.

For a simplified Transformer block:

```text
input X
  |
  +--> X * Wq -> Q
  +--> X * Wk -> K
  +--> X * Wv -> V
  |
  v
attention
  |
  v
* Wo
  |
  v
residual / normalization
  |
  v
FFN projections
  |
  v
next block
```

Because this execution order is known, ByteMyWave proposes **scheduled prefetch**, not speculative guessing.

## Fundamental pipeline

While the GPU computes tile N, the transfer engine should load tile N+1.

```text
TIME ------------------------------------------------->

GPU:
[ COMPUTE N ][ COMPUTE N+1 ][ COMPUTE N+2 ][ COMPUTE N+3 ]

H2D DMA:
             [ LOAD N+1 ]    [ LOAD N+2 ]    [ LOAD N+3 ]
```

If transfer finishes before the current compute finishes, the transfer is effectively hidden from wall-clock time.

## Fixed VRAM slots

VRAM should not be managed as repeated allocate/free operations for each model fragment.

Instead, reserve fixed working regions for the lifetime of inference.

Illustrative 4 GB layout:

```text
+--------------------------------+
| activations / latent / outputs |
| workspace                      |
+--------------------------------+
| weight slot A                  |
+--------------------------------+
| weight slot B                  |
+--------------------------------+
| optional prefetch slot C       |
+--------------------------------+
```

Exact sizes must be measured and tuned. The important property is that the addresses remain stable.

## Double/triple buffering

Example double buffer sequence:

```text
Initial:
RAM -> slot A : tile 1841
RAM -> slot B : tile 1842

Then:
GPU computes slot A / tile 1841
DMA fills/reuses next safe slot with tile 1843

GPU computes slot B / tile 1842
DMA fills slot A with tile 1843

GPU computes slot A / tile 1843
DMA fills slot B with tile 1844
```

The scheduler must never overwrite a slot before the kernel consuming it has completed.

## Precompiled execution plan

Before inference, create a flat ordered schedule:

```text
EXECUTION_PLAN =
  1841
  1842
  1843
  1844
  ...
```

Each entry points to metadata already known from the Weight Atlas:

```text
execution_index
host offset
compressed byte count
VRAM destination slot class
operation/kernel type
input/output region IDs
quantization format
dependency event IDs
```

The goal is to reduce runtime coordination overhead to pointer arithmetic and enqueue operations.

## Pinned host memory

Efficient asynchronous host-to-device DMA normally needs page-locked/pinned host memory.

Two possible strategies were discussed:

### A. Pin the relevant model region

If practical, page-lock the host regions needed during the active execution window.

### B. Pinned staging ring

Keep the complete model in ordinary large RAM, but maintain a smaller pinned ring ahead of the GPU:

```text
Large normal RAM model store
        |
        v
Pinned host ring: P0 / P1 / P2 / ...
        |
        v
PCIe DMA
        |
        v
VRAM slots: V0 / V1 / ...
```

This adds a RAM-to-RAM staging operation, so it should only be used if pinning model storage directly is impractical or harmful.

## Multiple operations ahead

The runtime does not have to prepare only N+1. Because model order is known, it can maintain a queue several operations ahead:

```text
GPU now:          Q17
ready in VRAM:    K17
currently DMA:    V17
ready in host:    O17
next host tile:   FFN-UP17
then:             FFN-DOWN17
then:             Q18
```

Queue depth should be adaptive to measured compute and transfer durations.

## Sub-layer streaming

If a complete projection matrix is too large for the desired VRAM footprint, split it into mathematically compatible tiles.

Instead of:

```text
load complete 800 MB Wq
compute
```

use:

```text
load Wq tile 0 -> compute
while computing: load tile 1
load Wq tile 1 -> compute
while computing: load tile 2
...
```

The runtime must choose tile shapes that preserve efficient GEMM behavior. Very small transfers reduce VRAM needs but may destroy throughput through launch overhead and poor matrix kernel efficiency.

## GPU-side dequantization

Weights should remain compressed across PCIe.

Desired path:

```text
RAM Q4/Q3/Q2
   |
   | compressed H2D DMA
   v
VRAM compressed tile
   |
   v
GPU loads tiny fragments
   |
   v
dequantize in registers/shared memory
   |
   v
matrix computation
```

Avoid:

```text
RAM compressed
 -> CPU decompress to FP16
 -> transfer expanded data over PCIe
```

because that destroys much of the bandwidth advantage.

## Events and synchronization

Dependencies should be represented through GPU/driver events rather than blocking CPU polling wherever possible.

Conceptually:

```text
copy tile N complete
      |
      v
READY_N event
      |
      v
kernel N may execute
```

Similarly, a slot may only be overwritten after a completion event confirms that the previous consumer kernel no longer needs it.

## Main performance metric

Do not judge the architecture only by `H2D_time`.

Measure:

```text
compute_time
H2D_time
transfer_compute_overlap
uncovered_H2D_time
GPU_starvation_time
GPU_utilization
```

A transfer taking 15 ms is not necessarily a 15 ms penalty. If 12 ms overlap with useful GPU work, the exposed penalty is approximately 3 ms.

The project therefore focuses on **unhidden transfer time**.
