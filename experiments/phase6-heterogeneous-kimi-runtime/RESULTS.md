# Phase 6 derived results

All numbers below are analytical ceilings/requirements unless explicitly labeled otherwise.

## Kimi K2.5 — 1 TiB host RAM + 1x RTX 3060 12 GiB

Inputs:

```text
32B official active params/token
21.13929216B derived routed active params/token
10.86070784B derived non-routed planning remainder
12 GB/s assumed effective H2D
10 TFLOP/s assumed effective GPU dense-linear compute
```

### One-GPU streaming ceilings

| Strategy | Format | Fresh H2D GB/token | Transfer ms/token | Ceiling tok/s |
|---|---|---:|---:|---:|
| stream all active | TensorWave Q4 0.625 B/w | 20.000 | 1666.7 | **0.600** |
| static non-routed residency | TensorWave Q4 | 13.212 | 1101.0 | **0.908** |
| stream all active | Q2 G64/F16 scale 0.28125 B/w | 9.000 | 750.0 | **1.333** |
| static non-routed residency | Q2 G64/F16 scale | 5.945 | 495.5 | **2.018** |
| stream all active | ideal 1-bit stress test | 4.000 | 333.3 | **3.000** |
| static non-routed residency | ideal 1-bit stress test | 2.642 | 220.2 | **4.541** |

Conclusion: even ideal 1-bit routed streaming remains below 5 tok/s through one 12 GB/s host feed. The 5–10 tok/s target therefore requires weight traffic avoidance, not only a smaller encoding.

### Four-socket CPU routed-expert requirements

Current TensorWave Q4 routed bytes/token:

```text
13.2120576 GB/token
```

| Target | Required selected-weight BW/socket | Logical expert compute/socket | Selected weights/socket | Raw aggregate cycle budget |
|---:|---:|---:|---:|---:|
| **5 tok/s** | **16.515 GB/s** | **52.848 GFLOP/s** | **26.424 Gweights/s** | **1.589 cycles/weight** |
| **10 tok/s** | **33.030 GB/s** | **105.696 GFLOP/s** | **52.848 Gweights/s** | **0.795 cycles/weight** |

Intel's published 85 GB/s/socket makes the memory-side fractions:

```text
5 tok/s  -> 19.43% of published maximum
10 tok/s -> 38.86% of published maximum
```

This does not prove the AVX-only low-bit kernel can meet the compute/decode requirement.

### Two sockets vs four

| Sockets | Target | Q4 GB/s/socket | GFLOP/s/socket |
|---:|---:|---:|---:|
| 2 | 5 tok/s | 33.030 | 105.696 |
| 2 | 10 tok/s | 66.060 | 211.393 |
| 4 | 5 tok/s | 16.515 | 52.848 |
| 4 | 10 tok/s | 33.030 | 105.696 |

For CPU expert execution, four sockets exactly halve the per-socket requirement versus two.

### Q2 CPU expert memory requirement

Candidate Q2 G64 + FP16 scale = 0.28125 B/weight.

```text
routed bytes/token = 5.94542592 GB
```

Four sockets:

```text
5 tok/s  -> 7.432 GB/s/socket
10 tok/s -> 14.864 GB/s/socket
```

The logical matrix arithmetic remains; Q2 mainly reduces memory traffic and adds unpack/dequant work.

### Activation volume

Conservative upper-bound volume for four sockets:

```text
6.88128 MB/token
```

At 5 tok/s:

```text
0.0344 GB/s
```

Therefore **activation volume is not the problem**. Serial CPU/GPU handoff latency across 60 MoE layers is the problem to benchmark.

---

# Kimi K3 — 2 TiB host RAM + 1x RTX 3060 12 GiB

Inputs:

```text
104B official active params/token
97.240743936B derived routed active params/token
6.759256064B derived non-routed planning remainder
```

Current TensorWave Q4 full-model host size:

```text
2.8T * 0.625 B = 1.75 TB decimal
```

2 TiB host RAM passes the capacity-only gate.

### One-GPU streaming ceilings

| Strategy | Format | Fresh H2D GB/token | Ceiling tok/s |
|---|---|---:|---:|
| stream all | TensorWave Q4 | 65.000 | **0.185** |
| static non-routed residency | TensorWave Q4 | 60.775 | **0.197** |
| stream all | Q2 G64/F16 | 29.250 | **0.410** |
| static non-routed residency | Q2 G64/F16 | 27.349 | **0.439** |
| stream all | ideal 1-bit | 13.000 | **0.923** |
| static non-routed residency | ideal 1-bit | 12.155 | **0.987** |

This mathematically confirms the earlier qualitative statement: `2 TiB RAM + 1x3060` is below 1 tok/s for K3 under one-feed weight streaming, even at the idealized 1-bit stress-test level.

### Compression-only impossibility bound

If every active K3 weight must traverse one 12 GB/s link:

```text
5 tok/s  -> <= 0.1846 bit/active-weight
10 tok/s -> <= 0.0923 bit/active-weight
```

This is why K3 needs bytes avoided/reused or independent parallel links.

### CPU routed-expert thresholds for K3

Four sockets, current Q4:

```text
5 tok/s:
75.969 GB/s/socket
243.102 GFLOP/s/socket

10 tok/s:
151.939 GB/s/socket
486.204 GFLOP/s/socket
```

The 10 tok/s memory requirement alone exceeds Intel's published 85 GB/s/socket maximum. K3 therefore cannot simply reuse the K2.5 four-CPU plan at the same target rate; substantial GPU expert ownership and/or multiple GPUs are required.

---

# K3 ideal multi-GPU routed-shard sensitivity

These rows assume:

```text
- derived non-routed set handled separately/resident
- routed expert bytes perfectly split across independent 12 GB/s H2D feeds
- no collective/reduction/attention/routing/synchronization cost
```

They are **ceilings, not current TensorWave capability**.

## Current TensorWave Q4 routed bytes = 60.775 GB/pass

| GPUs | Target passes/s | output tok/s @ 2.61 accepted | output tok/s @ 4.73 accepted |
|---:|---:|---:|---:|
| 1 | 0.197 | 0.515 | 0.934 |
| 2 | 0.395 | 1.031 | 1.868 |
| 3 | 0.592 | 1.546 | 2.802 |
| 4 | 0.790 | 2.061 | 3.736 |
| 6 | 1.185 | 3.092 | **5.604** |

## Candidate Q3 G64/F16 routed bytes = 39.504 GB/pass

| GPUs | Target passes/s | output tok/s @ 2.61 | output tok/s @ 4.73 |
|---:|---:|---:|---:|
| 1 | 0.304 | 0.793 | 1.437 |
| 2 | 0.608 | 1.586 | 2.874 |
| 3 | 0.911 | 2.378 | 4.310 |
| 4 | 1.215 | 3.171 | **5.747** |
| 6 | 1.823 | 4.757 | **8.621** |

## Candidate Q2 G64/F16 routed bytes = 27.349 GB/pass

| GPUs | Target passes/s | output tok/s @ 2.61 | output tok/s @ 4.73 |
|---:|---:|---:|---:|
| 1 | 0.439 | 1.145 | 2.075 |
| 2 | 0.878 | 2.290 | 4.151 |
| 3 | 1.316 | 3.436 | **6.226** |
| 4 | 1.755 | 4.581 | **8.302** |
| 6 | 2.633 | **6.871** | **12.452** |

The DSpark accepted-token factors come from vLLM's external K3 results and are included only as sensitivity values. They are not guaranteed on TensorWave workloads.

---

# Full-residency thought experiment correction

The earlier compute-only thought experiment for a hypothetical RTX 3060 with enough VRAM for all K3 weights gave about 48 tok/s from 10 effective TFLOP/s:

```text
10 TFLOP/s / (2 * 104B FLOP/token) = ~48.1 tok/s
```

But the resident active weights still have to be read from VRAM.

With current TensorWave Q4:

```text
65 GB/token active reads
360 GB/s VRAM-bandwidth input
=> 360/65 = **5.54 tok/s**
```

Thus:

```text
combined simple resident roofline
= min(48.1 compute, 5.54 VRAM bandwidth)
= **5.54 tok/s**
```

This replaces the incomplete compute-only conclusion.

---

# Go/no-go measurements on the physical R920

Before claiming 5 or 10 tok/s, measure:

```text
1. local NUMA compressed-weight read bandwidth/socket
2. low-bit expert GEMV selected weights/s/socket
3. unpack/dequant cost
4. p50/p95/p99 CPU<->GPU handoff per MoE layer
5. remote-vs-local NUMA penalty
6. real RTX 3060 H2D bandwidth
7. real P2P support/bandwidth for each GPU topology
8. resident-set VRAM usage including KV/workspace
9. quality loss for Q3/Q2/mixed formats
```

Pass criterion for the first serious K2.5 5 tok/s attempt:

```text
>= 26.424 Gweights/s/socket on 4 sockets
>= 16.515 GB/s/socket of selected Q4 expert reads
with total serial handoff + GPU resident path + attention/state staying within the 200 ms/token budget
```
