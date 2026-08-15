# R920 + RTX 3060 — TensorWave simulation result

> **Simulation, not measurement.** The timing inputs below are analytical assumptions intended to be replaced by Phase-4 calibration on the real machine.

## Configuration

- server: Dell PowerEdge R920, 4 sockets
- CPU profile: Intel Xeon E7-4890 v2 × 4 = 60 cores / 120 threads
- RAM: 1024 GiB total, 256 GiB/socket if balanced
- GPU profile: GeForce RTX 3060 12GB
- assumed effective H2D: 12.00 GB/s per GPU
- assumed effective dense-linear compute: 10.00 TFLOP/s per GPU
- model: generic dense reference 70B; **not a claim about MiniMax H3 parameter count**
- TensorWave wire format: Q4_SYM_G32_F32S = 0.625 B/parameter
- Q4 host/wire model size: 43.75 GB
- simulated persistent cache: 0 GiB/GPU
- Phase-3 tile geometry: K=8192, N=256, tiles=32
- dequant assumption: 0 µs/tile, preserving the Phase-4 lower-bound semantics

Predicted dense-linear crossover: **M ≈ 260.4**.

## 70B dense roofline

| M | stream GB/step | H2D ms | compute ms | hidden % | starvation lower bound % | ideal rows/s | regime |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 1 | 43.750 | 3645.83 | 14.00 | 0.38 | 99.62 | 0.27 | TRANSFER_BOUND |
| 4 | 43.750 | 3645.83 | 56.00 | 1.54 | 98.46 | 1.10 | TRANSFER_BOUND |
| 16 | 43.750 | 3645.83 | 224.00 | 6.14 | 93.86 | 4.39 | TRANSFER_BOUND |
| 64 | 43.750 | 3645.83 | 896.00 | 24.58 | 75.42 | 17.55 | TRANSFER_BOUND |
| 128 | 43.750 | 3645.83 | 1792.00 | 49.15 | 50.85 | 35.11 | TRANSFER_BOUND |
| 256 | 43.750 | 3645.83 | 3584.00 | **98.30** | **1.70** | 70.22 | NEAR_BALANCED |
| 512 | 43.750 | 3645.83 | 7168.00 | 100.00 | 0.00 | 71.43 | COMPUTE_BOUND |
| 1024 | 43.750 | 3645.83 | 14336.00 | 100.00 | 0.00 | 71.43 | COMPUTE_BOUND |
| 2048 | 43.750 | 3645.83 | 28672.00 | 100.00 | 0.00 | 71.43 | COMPUTE_BOUND |

The result is deliberately hostile to dense single-sequence decode: **capacity is solved by the R920 RAM, but bandwidth is not**. Large `M`—prefill or batched reuse—changes the arithmetic intensity enough for overlap to become plausible.

## Phase-3 two-slot ring simulation

This follows the existing CUDA ownership schedule:

```text
slot(i) = i % 2
copy(i) waits compute(i-2)
compute(i) waits copy(i)
```

| M | Q4 tile MiB | fixed VRAM MiB | copy ms/tile | compute ms/tile | 32-tile wall ms | starvation % | hidden % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.250 | 6.517 | 0.1092 | 0.0004 | 3.496 | 99.60 | 0.38 |
| 4 | 1.250 | 6.566 | 0.1092 | 0.0017 | 3.497 | 98.42 | 1.54 |
| 16 | 1.250 | 6.766 | 0.1092 | 0.0067 | 3.502 | 93.67 | 6.14 |
| 64 | 1.250 | 7.563 | 0.1092 | 0.0268 | 3.522 | 74.83 | 24.58 |
| 128 | 1.250 | 8.625 | 0.1092 | 0.0537 | 3.549 | 50.05 | 49.15 |
| 256 | 1.250 | **10.750** | 0.1092 | 0.1074 | 3.603 | **1.64** | **98.30** |
| 512 | 1.250 | 15.000 | 0.1092 | 0.2147 | 6.981 | 0.00 | 100.00 |
| 1024 | 1.250 | 23.500 | 0.1092 | 0.4295 | 13.853 | 0.00 | 100.00 |
| 2048 | 1.250 | 40.500 | 0.1092 | 0.8590 | 27.597 | 0.00 | 100.00 |

### Important consequence

For the representative K/N geometry, the transient Phase-3 ring uses only **10.75 MiB at M=256** and **15 MiB at M=512**. A 12 GiB RTX 3060 therefore has vastly more VRAM than the current ring requires.

The best use of that extra VRAM is likely not an enormous transient tile. A stronger next experiment is:

```text
small fixed stream ring
+
persistent compressed hot-weight / hot-expert cache
```

## 8 GiB cache sensitivity

If a later runtime safely allocates 8 GiB of the RTX 3060 for persistent compressed weights:

```text
70B Q4 wire model = 43.75 GB
8 GiB ~= 8.59 GB
resident fraction ~= 19.63%
streamed dense bytes ~= 35.16 GB
M_cross ~= 209.3 instead of 260.4
```

This is a material shift in the operating boundary. It gives a concrete reason to prefer a 12 GB 3060 over a 4 GB card even when the transient ring fits easily in both.

## Multi-GPU scaling envelopes

The safe/current-code interpretation is independent workers. The ideal-shard column is intentionally optimistic and **not implemented**; it excludes activation exchange, collectives and synchronization.

| GPUs | M | replicated aggregate rows/s | ideal equal-shard step ms, no collectives |
|---:|---:|---:|---:|
| 1 | 1 | 0.27 | 3645.83 |
| 1 | 64 | 17.55 | 3645.83 |
| 1 | 256 | 70.22 | 3645.83 |
| 1 | 512 | 71.43 | 7168.00 |
| 2 | 1 | 0.55 | 1822.92 |
| 2 | 64 | 35.11 | 1822.92 |
| 2 | 256 | 140.43 | 1822.92 |
| 2 | 512 | 142.86 | 3584.00 |
| 3 | 1 | 0.82 | 1215.28 |
| 3 | 64 | 52.66 | 1215.28 |
| 3 | 256 | 210.65 | 1215.28 |
| 3 | 512 | 214.29 | 2389.33 |

Replication improves aggregate throughput but does **not** solve the latency of one dense M=1 request. One-model multi-GPU sharding requires a new runtime layer.

## NUMA / memory-controller interpretation

With E7-4890 v2's published 85 GB/s maximum memory bandwidth:

```text
one assumed 12 GB/s GPU feed = 14.1% of one CPU's published maximum
two assumed 12 GB/s feeds     = 28.2%
```

This suggests theoretical local-memory headroom, but it is not proof. The real R920 must measure local and remote pinned H2D separately. Cross-socket QPI traffic is exactly the kind of hidden penalty that a non-NUMA-aware allocator could introduce.

## Observations

1. **1 TiB solves capacity, not dense streaming bandwidth.**
2. **NUMA locality should become a first-class TensorWave scheduling property**, not an afterthought.
3. **The 12 GiB RTX 3060 has enough spare VRAM to make persistent Q4 caching a high-priority optimization.**
4. **Current TensorWave code naturally scales as independent per-GPU workers**, not as tensor parallelism.
5. **The R920's six electrical x16 links are not six guaranteed internal RTX 3060 positions.** Physical length, dual-slot width, 8-pin power and airflow are separate limits.
6. **M≈256 is the first near-balanced reference point under the assumed 12 GB/s / 10 TFLOP/s calibration.**
7. **M>=512 is compute-bound in the idealized model**, which reinforces prefill/batched serving as stronger regimes than dense batch=1 decode.

## Proposals generated from the simulation

### P0 — hardware gate

Validate one RTX 3060 in the actual chassis: fit, auxiliary power, thermals, firmware enumeration, negotiated PCIe width/speed and sustained CUDA load.

### P1 — NUMA-aware Weight Atlas/runtime

Extend execution metadata with:

```text
host NUMA node
preferred GPU
PCI bus/root
host allocation node
measured local H2D
measured remote H2D
```

### P2 — persistent compressed cache

Add Q4 cache metrics:

```text
cache hit rate
bytes avoided
resident GB
evictions
starvation with/without cache
```

### P3 — per-GPU workers

Create one copy/compute/ring state per GPU, bound to its local NUMA domain, before attempting distributed model execution.

### P4 — topology-aware sharding later

For a real multi-GPU model, evaluate in this order:

```text
MoE expert ownership
pipeline/layer-range ownership
fine-grained tensor parallelism
```

The first two can reduce dependence on frequent cross-GPU collectives, which is attractive on separate PCIe roots and RTX 3060s without NVLink.

## What would falsify the R920 hypothesis?

The platform should be reconsidered if real measurements show:

- poor/unstable local pinned H2D;
- severe NUMA/QPI collapse with concurrent GPU traffic;
- inability to power/cool the desired GPU configuration safely;
- no real copy/compute concurrency despite correct CUDA events/streams;
- measured crossover much worse than a Phase-4-calibrated prediction;
- host memory becoming the dominant bottleneck at the intended GPU count.

The committed simulator is meant to be replaced by measurement, not defended against measurement.
