# Phase 7 — Ten-R920 procurement and build plan

Date: **2026-08-17**

Status: **procurement/build branch; analytical design, not a measured Kimi K3 benchmark**

This directory freezes the procurement and physical-build plan for the capstone cheap-R920 architecture derived in Phase 6.

Base branch: `research/heterogeneous-moe-kimi-v1`

Canonical topology:

```text
10 x Dell PowerEdge R920

per node:
    4 x Xeon E7-4890 v2 class sockets
    16 x 16GB DDR3-1600 ECC RDIMM = 256GB/node
    1 x RTX 3060 12GB fixed-path shard
    5 x GTX 1060 6GB-class routed-expert workers
    1 x ConnectX-3 FDR56 PCIe x8 NIC
    1 x SATA boot SSD
    1 x external 1200W-class GPU PSU
    6 x true x16->x16 powered PCIe risers

cluster totals:
    10 x R920
    40 x E7 sockets
    160 x 16GB DIMMs = 2.56TB nominal DDR3
    10 x RTX 3060 12GB = 120GB aggregate fixed-path VRAM
    50 x GTX 1060 6GB = 300GB aggregate worker VRAM
    10 x FDR56 NICs
    1 x 36-port FDR56 switch
```

The 120GB RTX-3060 pool is **not unified VRAM**. K3 fixed-path tensors must be explicitly sharded. The worker pool accelerates one request only if routed-expert work is sharded so many workers participate in each dependent layer.

## Frozen K3 arithmetic

Older Phase-6 K3 notes incorrectly used hidden size 7168 directly inside routed experts. K3 Stable LatentMoE projects `7168 -> 3584`, routed experts operate at `3584 -> 3072 -> 3584`, then the result is projected back.

Therefore:

```text
one routed expert = 3 * 3584 * 3072
                  = 33,030,144 weights

92 MoE layers * 16 selected experts/token
= 48,620,371,968 routed weights/token
= 48.620371968B
```

Native MXFP4 routed payload planning model:

```text
32 x 4-bit weights = 16 bytes
+ 1 byte scale      = 17 bytes/group
17/32               = 0.53125 B/weight

48.620371968B * 0.53125
= 25.829572608 GB routed payload/token
```

At the retained planning assumption of 12GB/s sustained compressed H2D per clean PCIe Gen3 x16 worker feed:

```text
50 workers * 12GB/s = 600GB/s
600 / 25.8296 = 23.23 routed-token-equivalents/s
```

This **23.23 is an internal routed-expert roofline, not K3 output tok/s**.

Current engineering labels, before physical validation:

```text
first useful end-to-end target:  ~15-25 output tok/s
stretch:                         ~25-35 output tok/s
speculative stretch:             ~30-45 only after measured A/U
100 tok/s:                       not credible for this architecture
```

## Budget status

The old `~55k lei` number covered only the major compute/RAM pieces. Once true x16 powered risers, external GPU PSUs, low-latency network, rack, storage, rails and power distribution are included, the known subtotal is closer to the `~86k-92k lei` class before riser RFQ, transport/import, HVAC and building electrical work.

A **100k lei cap is therefore a tight project budget, not a guaranteed finished-install price**. A safer maximum planning envelope is about **120k lei**, while procurement should remain staged.

## Procurement rule

Do **not** buy ten nodes first.

Build in this order:

```text
1 node -> validate local hardware and kernels
2 nodes -> validate single-request scale-out over FDR
4 nodes -> validate scaling trend
10 nodes -> only after previous gates pass
```

If two nodes do not materially reduce one-request latency versus one node, stop. Aggregate multi-request throughput is not the project objective.

## Files

- `BOM.md` — exact quantities, current purchase targets, source links, known vs RFQ pricing.
- `ASSEMBLY.md` — physical slot allocation, RAM population, external GPU mounting/power and network assembly.
- `VALIDATION.md` — mandatory go/no-go measurements before scaling procurement.
- Phase-6 canonical performance rationale remains in `../phase6-heterogeneous-kimi-runtime/TEN-R920-K3-CLUSTER.md`.

## Safety / facility note

The complete cluster has a planning wall-load envelope of roughly **16-20kW under sustained heavy load**, before a physical measurement exists. Cooling must remove approximately the same heat load. Building mains, three-phase distribution, breakers, grounding and HVAC must be designed/installed by qualified professionals. This repository specifies load requirements; it does not prescribe mains wiring work.