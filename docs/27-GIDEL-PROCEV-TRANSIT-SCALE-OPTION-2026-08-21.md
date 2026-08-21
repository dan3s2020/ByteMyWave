# 27 — GIDEL ProceV as a Full Transit Scale Option — 2026-08-21

This document records the GIDEL ProceV path not merely as a surplus-price candidate, but as a concrete Transit architecture option for scaling Kimi K3-like sparse-MoE inference with local DDR3 + FPGA compute.

## 1. Board primitive

Working board family:

- GIDEL ProceV Rev.2 / Rev.3
- observed model labels: `PROCE VD8-BM` and `PROCE VD8-BXSM`
- Stratix V GS FPGA (`5SGSD8` family on documented variants)
- PCIe Gen3 x8 host interface
- **2 independent DDR3 ECC SO-DIMM banks per card**
- documented configuration: **2 x 8 GB = 16 GB DDR3/card**
- each bank: **72-bit physical bus = 64 data + 8 ECC**
- DDR3-1600
- documented sustained bandwidth assumption: **~9.6 GB/s per bank, ~19.2 GB/s/card**
- current observed four-card lot: **~US$199.99 total (~US$50/card before shipping)**

Logical primitive:

```text
R920 / PCIe fabric
        |
    PCIe Gen3 x8
        |
   GIDEL ProceV
      Stratix V
      /       \
 DDR3 ch0   DDR3 ch1
   8 GB       8 GB
```

Therefore one physical ProceV card contributes:

```text
16 GB local DDR3
2 independent DDR3 channels
1 programmable FPGA compute engine
~19.2 GB/s documented sustained aggregate DDR bandwidth
```

## 2. Four-card logical Transit tile

Four ProceV cards map naturally to the existing logical `8 DDR3 channels/tile` Transit concept:

```text
Logical Transit tile
├── ProceV #0 -> 2 DDR3 channels / 16 GB
├── ProceV #1 -> 2 DDR3 channels / 16 GB
├── ProceV #2 -> 2 DDR3 channels / 16 GB
└── ProceV #3 -> 2 DDR3 channels / 16 GB

TOTAL
8 independent DDR3 channels
64 GB local DDR3
4 Stratix V compute engines
~76.8 GB/s documented sustained aggregate DDR bandwidth
~US$200 at the currently observed four-card lot price
```

This is a logical tile. Unless a local PCIe switch is added, it exposes four PCIe endpoints rather than one.

## 3. K3 / Transit sizing assumptions

The current ByteMyWave sizing model uses:

```text
active parameters per token ~= 104 billion
simple 4-bit-equivalent lower bound ~= 0.5 byte/weight
active weight payload ~= 52 GB/token
```

Performance language must remain precise:

- `bandwidth / 52 GB` is a **weight-path tok/s equivalent**;
- it is not measured end-to-end Kimi K3 generation speed;
- full end-to-end speed still depends on arithmetic format, routing, attention/state, reductions, PCIe scheduling and real physical measurements.

## 4. Minimum-capacity build

Using ~1.56 TB as the working complete-checkpoint storage target:

```text
1 ProceV = 16 GB
1560 GB / 16 GB = 97.5 cards
```

Therefore the first integer configuration that reaches capacity is:

```text
98 ProceV cards
= 1,568 GB local DDR3
= 196 independent DDR3 channels
= 98 Stratix V FPGAs
```

Using the documented ~19.2 GB/s/card sustained-memory figure:

```text
98 x 19.2 GB/s = 1,881.6 GB/s
1,881.6 / 52 ~= 36.18 weight-path tok/s equivalent
```

So the **capacity-first configuration** is approximately:

```text
98 cards
~1.57 TB local DDR3
196 independent channels
~1.88 TB/s documented aggregate DDR bandwidth
~36.2 weight-path tok/s equivalent
```

At ~US$50/card, board-only cost would be approximately:

```text
98 x US$50 ~= US$4,900
```

before PCIe switching/fanout, power, cooling, shipping, taxes and host infrastructure.

## 5. ~100 weight-path tok/s build

The current Transit target requires approximately:

```text
52 GB/token x 100 token/s = 5,200 GB/s
```

At ~19.2 GB/s/card:

```text
5,200 / 19.2 = 270.83
```

Therefore round upward to:

```text
271 ProceV cards
```

This gives:

```text
271 x 16 GB = 4,336 GB DDR3
271 x 2 = 542 independent DDR3 channels
271 x 19.2 GB/s = 5,203.2 GB/s
5,203.2 / 52 ~= 100.06 weight-path tok/s equivalent
```

So the **bandwidth-first ~100 tok/s configuration** is:

```text
271 ProceV cards
~4.34 TB local DDR3
542 independent DDR3 channels
271 Stratix V compute engines
~5.20 TB/s documented aggregate DDR bandwidth
~100.1 weight-path tok/s equivalent
```

At ~US$50/card, board-only cost is approximately:

```text
271 x US$50 ~= US$13,550
```

Again this excludes switching/fanout, power, cooling, shipping and taxes.

## 6. Why the 271-card build has much more memory than strictly required

The capacity requirement and bandwidth requirement are different constraints.

Capacity alone is reached around 98 cards.

The ~100 weight-path tok/s target forces ~271 cards because the limiting term is aggregate sustained memory bandwidth, not storage capacity.

Thus the 271-card configuration has substantial excess capacity (~4.34 TB vs ~1.56 TB working checkpoint target). That excess could later be used for:

- expert replication to reduce contention;
- alternate quantized representations;
- local metadata/scales;
- KV/state experiments where appropriate;
- redundancy and spare shards;
- placement that minimizes cross-domain traffic.

It does not by itself increase the 100 tok/s estimate unless it is converted into more useful parallel weight-read bandwidth.

## 7. GPU role in this architecture

A Transit system built around ProceV does **not require conventional GPUs for the main resident-weight expert path**.

The intended hot path is:

```text
resident weights in ProceV DDR3
        |
local Stratix V compute
        |
reduced result over PCIe
```

The R920 remains host/orchestrator and can perform:

- model atlas / placement management;
- routing and scheduling;
- PCIe command submission;
- result collection/reduction;
- telemetry and recovery;
- checkpoint staging;
- host-side state as appropriate.

However, keeping **1-2 auxiliary GPUs** in the R920 can still be useful for work that is not initially moved into FPGA logic, for example:

- attention / KV-related kernels;
- sampling;
- selected dense transforms;
- numerical reference/debug;
- fallback kernels while the FPGA path is incomplete.

Therefore the recommended architectural interpretation is:

```text
R920
├── CPU / RAM / NVMe: orchestration, routing, staging, state
├── optional 1-2 GPUs: auxiliary dense/attention/debug work
└── PCIe switch fabric
      ├── ProceV #0
      ├── ProceV #1
      ├── ...
      └── ProceV #N
```

The GPUs are auxiliary, not where the full K3 weight set must reside.

## 8. Practical acquisition decision

Do **not** jump directly to 98 or 271 cards.

The current practical purchase target is the four-card lot because it forms one complete logical 8-channel Transit group:

```text
4 cards
64 GB local DDR3
8 independent DDR3 channels
~76.8 GB/s documented sustained DDR bandwidth
~1.48 weight-path tok/s equivalent
```

Required physical validation before any bulk purchase:

1. confirm exact FPGA revision and installed DIMMs on all cards;
2. prove both DDR3 banks train independently;
3. measure simultaneous sustained read bandwidth from both banks;
4. preserve/recover factory flash before custom programming;
5. prove custom Quartus bitstream and reliable reprogramming;
6. prove PCIe enumeration and DMA in the Dell R920;
7. port a Transit local integer/bitplane or native-MXFP test kernel;
8. compare FPGA output against the host reference;
9. measure DDR-stall vs compute-active cycles;
10. test reduced-lane / PCIe-switch / fanout behavior;
11. scale 1 -> 2 -> 4 cards before planning a multi-dozen-card fabric.

## 9. Current conclusion

GIDEL ProceV is now a serious Transit architecture option because one cheap card combines:

```text
PCIe Gen3 x8
+ programmable Stratix V
+ 16 GB local DDR3
+ 2 real independent DDR3 channels
```

The key scale points are:

```text
4 cards   -> 64 GB, 8 channels, ~1.48 weight-path tok/s
98 cards  -> ~1.57 TB, 196 channels, ~36.2 weight-path tok/s
271 cards -> ~4.34 TB, 542 channels, ~100.1 weight-path tok/s
```

These are architecture-sizing/weight-path numbers, not measured end-to-end Kimi K3 throughput.

The next action is therefore not bulk procurement. It is to acquire/test one four-card group and measure the real dual-DDR bandwidth + local Transit compute path on the R920.
