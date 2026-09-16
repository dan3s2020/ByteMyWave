# POC Validation and Procurement Gates

## Decision rule

This track is **not** authorized for fleet procurement merely because the nominal DDR arithmetic looks attractive.

The purchase ladder is:

```text
Gate 0 -> exact model accounting
Gate 1 -> 1 node memory path
Gate 2 -> 1 node K3-format expert kernel
Gate 3 -> 1 node non-routed/KDA path
Gate 4 -> 2 node communication
Gate 5 -> 4/8 node scaling
Gate 6 -> complete real K3 layer
Gate 7 -> capacity + system extrapolation
Fleet purchase decision
```

A failed gate means either optimize/retest or reject that board/runtime/topology. Do not bypass a failed gate by multiplying more nodes into the spreadsheet.

---

# Gate 0 — exact K3 active-byte inventory

## Goal

Replace the current 52 / ~58 / ~136.6 GB/token models with executable tensor-level accounting for the exact K3 revision we intend to run.

## Required output

For every decode-active tensor/component:

```text
name
layer
operation class
shape
storage/quantization format
weight bytes
scale/metadata bytes
activation frequency
routing frequency
bytes required per token
reuse/cache assumption
```

At minimum separate:

```text
routed experts
shared experts
latent/down/up projections
attention/KDA/Gated MLA
router
embeddings/output head
normalization/residual
state/KV traffic
```

## PASS

- total checkpoint accounting matches the downloaded revision;
- active-path accounting is generated from tensor/config metadata rather than hand-entered totals;
- the tool emits component-by-component bytes/token;
- assumptions about caching/reuse are explicit.

## FAIL

Any fleet token/s estimate still depends only on `104B * 0.5 byte` or a single average bytes/parameter number.

---

# Gate 1 — one-node GPU-visible DDR bandwidth

## Goal

Prove that Radeon R6 can actually consume the two DDR channels through the shared-memory path at a useful sustained rate.

## Setup

```text
1 Carrizo node
matched dual-channel DIMMs
Linux + amdgpu + RADV
persistent large buffer in system memory
Vulkan compute shader reading data
30+ minute stability run
```

Measure both CPU memory bandwidth and GPU-visible read bandwidth, but use **GPU-visible sustained read** for the decision.

## Metrics

```text
actual DDR MT/s
nominal dual-channel GB/s
sustained GPU read GB/s
percent of nominal
p50/p95 iteration time
CPU utilization
wall watts
temperature/errors
```

## Initial engineering thresholds

These are project decision thresholds, not vendor guarantees:

```text
>= 60% of nominal sustained GPU-visible payload  -> PASS
40-60%                                           -> INVESTIGATE/OPTIMIZE
< 40%                                            -> NO-GO for that board/runtime path
```

Example at DDR3-1600 dual channel:

```text
nominal       25.6 GB/s
60% threshold 15.36 GB/s
40% threshold 10.24 GB/s
```

The test must use enough data to escape cache-only behavior.

---

# Gate 2 — K3-shaped MXFP4 expert kernel

## Goal

Prove that the old GFX8 iGPU can turn memory bandwidth into useful expert arithmetic rather than becoming compute/dequant-bound.

## Required kernel

Use K3-shaped dimensions and packed representation:

```text
latent dimension       3584
expert hidden          3072
MXFP4-style groups     32 weights
uint8 scale metadata
matrix-vector decode path
```

The benchmark should include:

```text
packed read
scale load
unpack/dequant
multiply
accumulate
output write
```

Do not benchmark an FP32 toy matrix and call it an MXFP4 result.

## Correctness

Compare against a software reference on deterministic vectors. Record maximum/mean error appropriate to the exact representation used.

## Performance metric

```text
useful packed-weight GB/s
output vectors/s
percent of Gate-1 pure-read bandwidth
CPU overhead
wall watts
```

## Initial threshold

```text
>= 50% of measured Gate-1 pure-read bandwidth -> PASS to distributed work
30-50%                                        -> OPTIMIZE first
< 30%                                         -> likely NO-GO for this compute engine
```

Reason: if unpack/arithmetic already throws away most local DDR bandwidth on one node, adding hundreds of channels will not recover it.

---

# Gate 3 — non-routed and KDA representative kernels

## Goal

Prevent a false success where routed-expert GEMV is fast but the serial non-expert path dominates token latency.

## Benchmark at least

```text
representative high-precision linear projection
shared-expert operation
normalization/residual operation
KDA recurrent-state update shaped from official K3 code/config
representative Gated MLA operation/state movement
```

## PASS condition

For every target token rate being considered, no unavoidable single-node serial component may already consume the entire average per-layer budget.

Average layer budgets:

```text
10 tok/s -> ~1.075 ms/layer
20 tok/s -> ~0.538 ms/layer
30 tok/s -> ~0.358 ms/layer
```

A result can still PASS with slower individual kernels if they are demonstrably parallelizable/fusable and the complete measured layer fits the target budget. The complete-layer measurement in Gate 6 supersedes microbench estimates.

---

# Gate 4 — two-node communication path

## Goal

Measure the communication path at K3-like message sizes before buying a switch/fleet.

## Synthetic cadence

Run repeated exchanges for at least:

```text
7 KiB
14 KiB
28 KiB
```

and a 93-round barrier/collective loop.

Record:

```text
one-way latency
ping-pong latency
bandwidth
collective latency
p50/p95/p99
jitter
CPU utilization
packet/error counters
```

## Target-relative threshold

Communication p95 for the representative collective should consume no more than roughly **20% of the average target layer budget** before scaling further.

Examples:

```text
10 tok/s budget ~= 1.075 ms/layer -> comm p95 target <= ~215 us
20 tok/s budget ~= 0.538 ms/layer -> comm p95 target <= ~108 us
```

These are screening thresholds, not claims that the final layer uses exactly one collective.

If a topology exceeds the whole target layer budget on communication alone, it is a NO-GO for that target regardless of local DDR speed.

---

# Gate 5 — 4-node then 8-node scaling

## Goal

Determine whether tensor-sharded work converts additional nodes into lower single-operation latency.

## Test

Use the same matrix/vector operation and collective structure at:

```text
1 node
2 nodes
4 nodes
8 nodes
```

Keep problem definition and precision fixed. Shard the active matrix rather than giving every node independent unrelated work.

## Report

```text
operation latency
local useful GB/s per node
aggregate useful GB/s
collective latency
parallel efficiency
p95/p99
CPU utilization/node
network bytes/node
wall watts/node and total
```

Define speedup:

```text
S_N = T_1 / T_N
parallel_efficiency_N = S_N / N
```

## Initial decision thresholds at 8 nodes

```text
>= 60% parallel efficiency -> strong PASS
40-60%                     -> CONDITIONAL; optimize topology/runtime
< 40%                      -> NO-GO for extrapolating this topology to 160 nodes
```

Also require monotonic latency improvement. A nominally high efficiency calculation that hides severe p95/p99 jitter does not pass.

---

# Gate 6 — one complete K3 layer with real tensors

## Goal

Stop extrapolating from isolated GEMV and networking tests.

Implement one real layer path using tensors from the selected official K3 revision, including the layer's actual relevant pieces:

```text
input hidden state
attention/KDA/Gated-MLA work as applicable
latent projections
router
16 selected experts
shared expert path
expert combine/reduction
residual/norm
output hidden state
```

Use real tensor formats and real routing semantics.

## Correctness

Compare against the official/reference implementation for deterministic inputs within a documented numerical tolerance.

## Performance

Record a breakdown:

```text
local DDR time
local compute/dequant time
network/collective time
CPU/runtime time
barrier/idle/straggler time
complete layer latency
```

## PASS

A complete layer must fit the intended target's per-layer budget with headroom, or measured evidence must show why later fusion/overlap changes the critical path.

A layer that only meets the target by omitting shared/attention/state work does not pass.

---

# Gate 7 — system capacity, placement and 160-node extrapolation

## Goal

Use measured small-cluster data to decide whether a fleet purchase has a defensible expected outcome.

The extrapolation must use:

```text
exact Gate-0 bytes/token
Gate-1 measured GPU-visible bandwidth
Gate-2/3 kernel efficiency
Gate-4 measured communication latency
Gate-5 scaling curve
Gate-6 complete layer breakdown
measured wall power
real board/NIC/RAM delivered cost
real switch topology/bisection
```

It must also include:

```text
checkpoint capacity + placement slack
expert hotness/replication strategy
node/switch failure strategy
spare units
boot/model-load mechanism
cooling
PSU conversion
cabling
physical rack/shelf space
```

## Required fleet prediction format

Do not output a single optimistic number. Output at least:

```text
p50 expected decode tok/s
p95/p99 inter-token latency estimate
sensitivity to 20% slower memory
sensitivity to 2x collective latency
power at wall
hardware cost
network cost
cost/tok/s
W/tok/s
```

Every derived number must point to a measured small-cluster input.

---

# Fleet GO / NO-GO

## GO to 1-4 prototype nodes

Current status: **GO**.

The architecture has enough physical/software plausibility to justify a small POC.

## GO to 8 nodes

Only after Gates 0-4 pass.

## GO to bulk/fleet purchase

Only after Gates 0-6 pass and Gate 7 predicts an outcome worth the delivered cost and power.

## Automatic NO-GO triggers

Any of the following blocks fleet procurement until resolved:

```text
GPU-visible DDR <40% nominal on chosen board
K3-shaped expert kernel <30% of pure-read bandwidth
non-routed/KDA path dominates target layer budget with no demonstrated partition/fusion path
network collective alone consumes target layer budget
8-node parallel efficiency <40%
real K3 layer numerically incorrect
real K3 layer latency does not scale with additional useful memory-compute nodes
insufficient RAM capacity/placement slack
fleet NIC/switch cost erases the cheap-node advantage
measured wall power makes the architecture economically inferior to alternatives
```

## Procurement discipline

At every stage buy only enough units to answer the next unresolved measurement:

```text
1 -> memory/kernel
2 -> link
4 -> collective
8 -> scaling
then decide fleet
```

This prevents “cheap unit price” from turning into an expensive architecture assumption.