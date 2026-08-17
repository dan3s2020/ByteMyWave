# Validation gates — before scaling to ten R920 nodes

Date: **2026-08-17**

The complete cluster is a research build, not a guaranteed 15-25 tok/s appliance. This file defines the measurements that must pass before buying the next procurement tranche.

## Gate 0 — inventory acceptance

For every R920 received:

```text
all required memory risers present
four CPUs + heatsinks present if included in purchase
all fans and proprietary PCIe hardware present
PSUs matched and healthy
iDRAC works
no critical SEL errors
```

Reject/return nodes whose missing proprietary parts erase the liquidation-price advantage.

## Gate 1 — RAM/NUMA baseline

One node, 16 x 16GB RDIMM, four CPUs.

Measure:

```text
all 256GB detected
16 native channels populated as intended
local NUMA sequential read bandwidth/socket
remote NUMA penalty
memory bandwidth with all four sockets active
ECC/iDRAC error-free stress test
```

Do not project Intel published max bandwidth into K3 performance. Record actual sustained values.

## Gate 2 — one powered x16 riser

Install one GTX1060 worker through the selected x16->x16 powered riser.

Verify:

```text
PCIe Gen3 negotiated
x16 width negotiated
stable under sustained transfer + compute
no AER/link resets
riser/cable/connector temperature acceptable
```

Pinned H2D target:

```text
planning target: ~12GB/s
minimum procurement gate: >=10GB/s sustained
```

If the riser negotiates x8/x4/x1 or is unstable, do not buy the remaining 54.

## Gate 3 — five workers simultaneously

Populate RTX3060 + five GTX1060 workers, but first test the five workers as expert feeds.

Measure all five H2D streams concurrently using NUMA-local pinned buffers.

Record:

```text
GB/s per worker
aggregate GB/s
CPU/socket memory traffic
PCIe link counters/errors
p50/p95 transfer latency
power and thermals
```

Planning expectation:

```text
5 * 12GB/s = 60GB/s nominal aggregate
```

Hard decision rule:

- **GO:** each feed remains near the single-card result and aggregate is high enough to justify multi-feed scaling.
- **NO-GO:** five feeds collapse so badly that a second/third/etc. worker adds little aggregate service.

The exact go threshold should be tied to measured K3 expert service, but a node that cannot sustain roughly the `>=50GB/s` class aggregate under realistic concurrency is already materially below the design premise.

## Gate 4 — GTX1060 routed-expert kernel

Implement the actual K3 routed expert shape:

```text
latent input 3584
Gate/Up hidden 3072
Down output 3584
```

Worker path:

```text
compressed local DDR3 shard
-> async pinned H2D
-> packed low-bit decode on Pascal
-> fused/near-fused GEMV/GEMM
-> compact result
```

Required measurements:

```text
Gweights/s
GB/s compressed payload consumed
kernel time per expert/tile
dequant/unpack overhead
H2D-compute overlap efficiency
M=1
M>1 batched rows for speculative verification experiments
```

Do not use theoretical CUDA-core FLOPS as the result.

## Gate 5 — E7 AVX1 expert kernel

Use the corrected K3 shape, not the obsolete 7168-based expert derivation.

Measure one process per NUMA socket with local memory.

Record:

```text
Gweights/s/socket
compressed GB/s/socket
cycles/weight
scaling 1 -> 2 -> 4 sockets
performance while GPU worker feeds are simultaneously active
```

Sensitivity model for the final ten-node cluster:

```text
c = measured Gweights/s/socket
C_expert = 1129.41 + 40*c Gweights/s
routed ceiling = C_expert / 48.62037
```

Examples are sensitivity points only:

```text
c=0  -> 23.23 routed tok-eq/s
c=10 -> 31.46
c=20 -> 39.68
c=30 -> 47.91
```

## Gate 6 — one RTX3060 fixed-path shard

The ten RTX3060s cannot be treated as one 120GB GPU. Implement a real shard of the K3 non-routed/fixed path.

Planning direction:

```text
mixed/8-bit fixed-path representation
KDA/MLA shard
router shard
latent-MoE projections
shared/dense/lm-head placement as selected
```

Measure:

```text
resident bytes
workspace bytes
KV/state headroom
sustained low-batch bandwidth
kernel latency
quantization quality impact
```

The planning remainder from rounded 104B active parameters is ~55.38B active non-routed weights. At one byte/weight and perfect balance this is ~5.54GB weight payload per RTX3060, leaving capacity for runtime; this is a planning split, not an exact checkpoint census.

## Gate 7 — two-node FDR fabric

Build two complete nodes and connect through the SX6036/FDR fabric.

Measure host-buffer transport first:

```text
one-way latency
round-trip latency
bandwidth
p50/p95/p99
CPU overhead
NUMA locality
```

Then measure while all five local worker H2D feeds are active.

The important question is interference:

```text
does NIC traffic reduce local DDR->GPU service materially?
```

## Gate 8 — real distributed K3 layer prototype

Before a full model runtime, implement one representative dependent layer path:

```text
fixed-path/router shard
-> dispatch route metadata + latent activation
-> routed expert work across two nodes
-> partial reduction
-> continuation
```

Measure full per-layer critical-path latency.

K3 has 92 MoE layers. Useful reference budgets:

```text
15 tok/s -> 66.7ms/token -> 725us/MoE layer average total budget
20 tok/s -> 50.0ms/token -> 543us/layer
25 tok/s -> 40.0ms/token -> 435us/layer
35 tok/s -> 28.6ms/token -> 311us/layer
```

These budgets include compute, dispatch, reduction and runtime overhead, not merely network RTT.

## Gate 9 — one-request scaling 1 -> 2 nodes

Run the same single-request decode prototype on one node and two nodes.

Decision rule:

```text
ideal direction: close to ~2x
acceptable prototype evidence: substantial latency reduction, e.g. ~1.6x+ class
bad sign: ~1.0-1.2x where most added hardware is idle or synchronization-bound
```

Do not confuse requests/second from serving two independent prompts with single-request tokens/second.

If two nodes do not materially improve the one-request critical path, **stop the ten-node purchase** and fix the runtime/topology first.

## Gate 10 — 4-node checkpoint

Only after 1->2 scaling works, expand to four.

Validate:

```text
collective latency trend
routing imbalance
memory-bandwidth interference
switch behavior
scheduler CPU overhead
one-request tok/s improvement
```

If the scaling curve flattens at four nodes, do not assume ten will restore linearity.

## Gate 11 — ten-node full build

Only after prior gates pass.

Measure and publish separately:

```text
raw worker H2D aggregate
CPU expert throughput
fixed-path shard throughput
network per-layer latency
normal K3 decode tok/s
prefill separately
power at wall
thermals
speculative decoding separately
```

Never derive the benchmark by multiplying the routed-expert roofline by an assumed efficiency.

## Speculative decoding gate

Define:

```text
A = accepted output tokens / verification block
U = unique routed-expert weight traffic in one verification block
    / one normal-token routed traffic
```

Use measured `A/U`, not acceptance `A` alone.

A speculative block can touch many more unique experts than one normal token, so target-pass cost is not free on a DDR/PCIe-streamed expert fabric.

## Procurement tranches

Recommended spend discipline:

```text
TRANCHE A:
1 complete node
6 powered x16 risers
1 worker/fixed-path software prototype

TRANCHE B:
second node
SX6036 + 2 DACs
prove one-request scaling

TRANCHE C:
nodes 3-4
prove scaling curve

TRANCHE D:
remaining nodes 5-10 + remaining bulk GPUs/risers/RAM
```

A project budget of ~100k-120k lei is not authorization to skip these gates.