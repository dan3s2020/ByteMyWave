# Three-R920 Kimi K2.5 cluster

Date: 2026-08-15

Status: **target hardware configuration / analytical design; not yet measured on physical hardware**

This document freezes the current three-node R920 design for TensorWave Phase 6. It is intentionally separate from the single-node baseline and from future-platform notes.

## Objective

Run Kimi K2.5 as a single distributed inference engine using three low-cost Dell PowerEdge R920 nodes.

The design does **not** assume that one R920 independently produces a fixed number of tokens/s and that three nodes simply multiply that number. One token is a distributed dependency chain. All three nodes cooperate on the same token and synchronize at each MoE layer.

Primary planning envelope for a single Kimi K2.5 decode stream:

```text
minimum planning estimate: ~6 tok/s
target region:             ~10-12 tok/s
upper planning estimate:   ~15 tok/s
```

These are analytical planning values only. The physical `bench_cpu_expert_q4.cpp` result on an E7-4890 v2 socket is the acceptance gate that will replace the estimate.

## Hardware topology

Target cluster:

```text
3 x Dell PowerEdge R920

per node:
  4 x Intel Xeon E7-4890 v2
  60 cores / 120 threads
  DDR3 NUMA-local RAM

cluster total:
  12 CPU sockets
  180 cores / 360 threads
  48 memory channels total
```

Published processor memory-bandwidth maximum is 85 GB/s/socket. Therefore the purely published, non-achievable-as-an-assumption aggregate envelope is:

```text
12 x 85 GB/s = 1.02 TB/s
```

TensorWave must never treat 1.02 TB/s as measured cluster bandwidth. Local/remote NUMA placement, DIMM population, memory-controller efficiency, QPI traffic and the low-bit kernel determine the real value.

### GPU plan

Initial configuration uses only one head GPU:

```text
R920 #0: 1 x RTX 3060 12 GB
R920 #1: no GPU required initially
R920 #2: no GPU required initially
```

The RTX 3060 is the head accelerator for the always-active/non-routed path, attention/router work and other GPU-resident operations where appropriate.

The current K2.5 planning split is:

```text
official active parameters/token:     32B
routed active parameters/token:       21.13929216B
planning non-routed active remainder: 10.86070784B
```

At TensorWave Q4 G32/F32S (`0.625 B/weight`), the planning non-routed weight volume is:

```text
10.86070784B x 0.625 B
= 6.7879424 GB decimal
~= 6.32 GiB
```

This fits inside the Phase-6 8 GiB weight budget on a 12 GiB RTX 3060 when 4 GiB is reserved for runtime/KV/workspace. Exact checkpoint census remains mandatory before runtime placement is frozen.

Additional GTX 1050 Ti / GTX 1060 / similar GPU workers are **optional expansion**, not part of the initial three-R920 target. They are added only if measured CPU-expert throughput is below the target gate.

## Model storage and RAM

TensorWave Q4 uses:

```text
20 bytes / 32 weights
= 0.625 bytes/weight
```

A nominal 1T-parameter checkpoint therefore needs approximately:

```text
1T x 0.625 B ~= 625 GB decimal
```

With three-way sharding:

```text
625 GB / 3 ~= 208.3 GB decimal per node
```

Therefore three R920 nodes do **not** require 1 TiB each merely to store one K2.5 Q4 checkpoint.

Practical planning tiers:

```text
256 GiB/node -> minimum interesting cluster target; limited headroom
512 GiB/node -> preferred cost/capacity target
1 TiB/node   -> useful if already available, but not required for K2.5 Q4 capacity
```

The exact minimum depends on the real checkpoint tensor census, duplicated runtime state, KV cache, pinned buffers, OS overhead and the chosen shard layout.

## Distributed expert layout

Do not assign one complete expert to one server. K2.5 routes only eight experts per token, so whole-expert placement can leave many sockets idle.

Instead, shard **every routed expert across all three R920 nodes**.

For a standard routed SwiGLU expert:

```text
hidden = 7168
expert intermediate = 2048

Gate: 7168 -> 2048
Up:   7168 -> 2048
Down: 2048 -> 7168
```

Partition the 2048 intermediate dimension approximately:

```text
node 0: 683 channels
node 1: 683 channels
node 2: 682 channels
```

Each node owns the corresponding Gate/Up rows and matching Down input columns for every routed expert.

Per selected expert:

```text
1. Head/router determines selected expert IDs.
2. Hidden activation is available to all three shards.
3. Each node computes its Gate/Up slice.
4. Each node computes SiLU(gate) * up locally.
5. Each node applies its Down slice and produces a full-size partial output.
6. Partial outputs are reduced/summed.
7. The completed expert contribution continues to the next dependent operation.
```

This keeps the large expert weights in NUMA-local RAM and distributes work for every selected expert across the full cluster.

## CPU-expert throughput gates

K2.5 routed active work per token:

```text
21.13929216 Gweights/token
```

With 12 sockets, the required average selected-weight processing rate is:

```text
Gweights/s/socket = 21.13929216 * tokens_per_second / 12
```

Acceptance table:

| Final target | Required average routed rate/socket |
|---:|---:|
| 6 tok/s | 10.570 Gweights/s |
| 8 tok/s | 14.093 Gweights/s |
| 10 tok/s | 17.616 Gweights/s |
| 12 tok/s | 21.139 Gweights/s |
| 15 tok/s | 26.424 Gweights/s |

These gates are much more useful than guessing from CPU generation.

The physical benchmark decides the cluster target directly:

```text
<10.57 Gweights/s/socket
  -> below current 6 tok/s planning floor; GPU workers or a different format/kernel are required

~17.62 Gweights/s/socket
  -> 10 tok/s routed-compute gate

~21.14 Gweights/s/socket
  -> 12 tok/s routed-compute gate

~26.42 Gweights/s/socket
  -> 15 tok/s routed-compute gate
```

These are routed-work gates only. Final decode speed also includes GPU/non-routed work, synchronization, network latency, routing, reductions and runtime overhead.

## Network

The cluster must move activations and reductions, **not expert weights**.

K2.5 hidden vector in BF16:

```text
7168 x 2 bytes = 14,336 bytes ~= 14 KiB
```

For a head-centric three-node implementation, a conservative per-MoE-layer network volume is approximately:

```text
head -> two remote nodes: 2 x 14,336 B
remote partials -> head:  2 x 14,336 B
--------------------------------------
~= 57.3 KB/layer
```

Across 60 MoE layers:

```text
~= 3.44 MB/token
```

Even at 15 tok/s this is only about 51.6 MB/s of payload at the head before protocol overhead. Therefore raw link bandwidth is not the central problem.

**Serial latency is the central network problem.**

There are approximately 60 dependent MoE synchronization points per token. Example sensitivity:

```text
100 us/layer ->  6 ms/token
250 us/layer -> 15 ms/token
500 us/layer -> 30 ms/token
1 ms/layer   -> 60 ms/token
```

Target network direction:

```text
preferred: 25/40 GbE or InfiniBand/RDMA-class low-latency interconnect
acceptable for early functional proof: 10 GbE
```

A dual-port NIC in the head can directly connect the two worker R920 nodes for an initial three-node proof, avoiding a switch. A low-latency switch is cleaner once collective/reduction topology becomes more complex.

## NUMA placement

Each R920 has four CPU sockets. TensorWave must treat all twelve sockets as distinct memory/compute domains.

Rules:

```text
expert shard memory -> allocated on owning socket
CPU threads         -> pinned to owning socket
pinned network/GPU buffers -> allocated near the relevant PCIe root
avoid remote-QPI weight reads in the steady expert path
```

The Weight Atlas/runtime schedule must eventually carry at least:

```text
node_id
socket_id
expert_id
expert_shard_id
weight_offset
compression_type
execution_engine
network_reduction_group
```

## Runtime roles

### R920 #0 — head + CPU shard

```text
RTX 3060 12GB
attention / router / always-active path where resident
4 CPU sockets also compute their 1/3 expert shards
network coordinator / reduction endpoint initially
```

### R920 #1 — CPU expert worker

```text
4 CPU sockets
NUMA-local expert shards
receives compact activations / expert IDs
returns partial expert outputs
```

### R920 #2 — CPU expert worker

Same role as R920 #1.

The head node must not waste its four CPUs. All 12 sockets participate in expert work.

## Token/s planning envelope

Current planning range for a **single Kimi K2.5 stream**:

```text
~6 tok/s  -> conservative lower planning point
~10-12    -> main target region
~15 tok/s -> upper planning target
```

This range is intentionally not multiplied from any previous single-R920 tok/s estimate.

It is based on the exact 12-socket routed-work gates above plus the architectural requirement that weights remain local and only compact activations/reductions cross the network.

The range is invalidated and replaced as soon as these physical measurements exist:

1. `bench_cpu_expert_q4.cpp` Gweights/s per E7-4890 v2 socket.
2. Local vs remote NUMA memory bandwidth on all nodes.
3. p50/p95/p99 activation round-trip/reduction latency between nodes.
4. Real RTX 3060 non-routed path latency.
5. Full K2.5 checkpoint tensor census and exact static-residency fit.

## Optional GPU-worker expansion

Do not buy worker GPUs before the CPU gate is measured.

If CPU throughput is insufficient, each R920 exposes multiple PCIe 3.0 x16 links and can add small NVIDIA workers. A 4 GiB card is sufficient for a streaming expert-worker ring in the current design; it does not need to hold the model.

Possible path:

```text
NUMA-local RAM
  |-- CPU computes some expert tiles directly
  `-- PCIe -> cheap NVIDIA worker computes other expert tiles

CPU and GPU expert work execute concurrently.
```

The purpose of worker GPUs is to purchase additional independent host-memory-to-compute paths cheaply. They are not mandatory in the base three-R920 architecture.

## Cost model

Planning price supplied during the hardware discussion:

```text
~1,000 lei / R920-class node
3 nodes ~= 3,000 lei chassis/platform class
```

Treat this as a local used-market planning assumption, not a permanent market price.

Total build cost must be evaluated separately for:

```text
3 x R920
12 x desired E7-4890 v2 CPUs if not included
RAM population
1 x RTX 3060 12GB head GPU
3 x low-latency NICs
DAC/fiber/switch as required
power/cooling
```

The primary economic metric remains:

```text
complete-system lei / measured decode tok/s
```

## Initial bring-up order

```text
1. Validate one R920 with 4 x E7-4890 v2.
2. Run the AVX-only Q4 expert benchmark per NUMA socket.
3. Measure local/remote memory bandwidth.
4. Add second and third R920 with the minimum required RAM population.
5. Install low-latency NICs and measure one-way/round-trip activation latency.
6. Implement three-way expert intermediate-dimension sharding.
7. Implement partial-output reduction.
8. Add RTX 3060 head/non-routed path.
9. Run full K2.5 single-stream acceptance.
10. Add cheap GPU expert workers only if measured CPU gates justify them.
```

## Decision

The current low-cost TensorWave scale-out target is:

```text
3 x Dell PowerEdge R920
12 x E7-4890 v2 sockets total
one RTX 3060 12GB head GPU initially
K2.5 routed experts sharded across all three nodes
weights remain in NUMA-local RAM
network carries activations and partial reductions only
no mandatory worker-GPU fleet
```

Success is not defined by the number of servers. It is defined by measured single-stream Kimi K2.5 decode throughput and `lei / tok/s`.
