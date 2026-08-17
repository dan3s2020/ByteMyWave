# 14 — Memory-Controller Trade Study: CPU vs FPGA vs Minimal Logic

Date: 2026-08-17

This document captures the design discussion that followed the DDR3 tile architecture work. The core question was narrowed from "what powerful processor should run Kimi K3?" to a much more specific hardware question:

> What is the cheapest and most power-efficient way to expose enough DRAM bandwidth to a very small local arithmetic engine?

The important conclusion is that the arithmetic itself is not the difficult hardware problem. The difficult hardware problem is the memory interface: DDR PHY, training, timing, command generation, signal integrity and packaging.

This document intentionally distinguishes **working project numbers** from procurement facts. Any current market price or exact component availability must be re-verified before purchase.

---

## 1. What Transit actually computes near memory

The heavy routed-expert path is dominated by matrix-vector products / dot products.

At the primitive level the arithmetic is:

```text
sum += weight * activation
```

For the signed INT4 × INT8 proof kernel already implemented in this repository, the multiplication was decomposed exactly into bitplane operations:

```text
AND
  -> POPCOUNT
  -> fixed shifts / signed coefficients
  -> accumulate
```

The host proof produced exact integer equality against the scalar reference. Therefore, for the proof format, the arithmetic hardware does not need a large general-purpose CPU, large caches, speculative execution or a sophisticated instruction scheduler.

The K3 production path remains more complicated because published routed-expert formats are MXFP4/MXFP8 rather than the proof kernel's plain signed INT4/INT8. Even so, the dominant operation is still low-bit multiply/accumulate with scaling/decoding around it.

### Consequence

Do not choose a CPU or FPGA because Transit needs "a lot of compute" in the conventional sense.

Choose the device because it can:

1. terminate and control the required DRAM interfaces;
2. sustain enough weight-read bandwidth;
3. perform the small arithmetic pipeline without becoming the bottleneck;
4. return reduced results over the host interconnect.

---

## 2. Why a CPU looked attractive even though the arithmetic is simple

A very cheap used DDR3 CPU can bundle all of the expensive hard parts into one piece of silicon:

- DDR3 PHYs;
- memory controllers;
- training/calibration logic;
- command generation;
- clocking;
- cache/interconnect;
- general-purpose compute.

A working example discussed in the project is a used CPU in roughly the following class:

```text
~50 lei
~70 W class
4 independent DDR3 memory channels
```

The exact CPU model must remain attached to the purchasing record once selected; the numbers above are the working candidate class from the conversation, not a newly verified SKU in this document.

If such a CPU can keep four DDR3 channels near their sustainable read bandwidth while executing the Transit kernel, then the general-purpose compute is effectively "free" because the CPU was bought primarily for its four working DDR3 interfaces.

For a 4-channel / 70 W-class part:

```text
70 W / 4 = 17.5 W CPU TDP per channel
```

TDP is not equal to measured wall power and must not be treated as such. The relevant number is measured watts while running the real streaming kernel.

---

## 3. Why "CPU only reads, external gates calculate" is usually worse

The idea considered was:

```text
DDR3 -> CPU memory controller -> CPU reads only -> external AND/popcount gates
```

This sounds attractive because the CPU would not spend cycles on arithmetic. The problem is the data path.

Once a CPU memory controller has captured the DDR3 data, the bytes enter the CPU's internal uncore/cache/register hierarchy. There is no generic external pin that exposes each completed memory read directly to discrete gates.

To send the same stream to an external arithmetic circuit, the system would normally need another interface:

```text
DDR3
  -> CPU memory controller
  -> CPU/un-core/cache
  -> PCIe or another external bus
  -> external gates
```

That adds another high-speed transfer and often consumes more power/bandwidth than simply executing a few AND/POPCNT/ADD operations locally.

### Required experiment

Before removing compute from a cheap CPU tile, benchmark:

```text
A. maximum streaming read / memcpy bandwidth
B. streaming read + Transit AND/POPCNT/accumulate bandwidth
```

If B remains close to A, compute offload is pointless.

If B collapses far below A, then external/local dedicated compute becomes justified.

This is a measurement question, not an architectural assumption.

---

## 4. Why discrete transistors can do the arithmetic but not economically replace the DDR interface

At the logical level, yes: AND gates, counters, adder trees and accumulators are all built from transistors.

A conceptual arithmetic path can be built as:

```text
weight bits
   -> AND gates
   -> population-count tree
   -> accumulator
```

The issue is not that the operation is complicated. The issue is that **discrete transistors on a PCB are a terrible medium for implementing a GHz-class DDR3 receiver/controller**.

A full DDR3 interface must handle, among other things:

- CK / CK#;
- command/address lines;
- bank addressing;
- DQ data lines;
- DQS strobes;
- termination;
- initialization;
- refresh;
- read/write timing;
- write leveling;
- read gate training;
- DQ/DQS alignment;
- clock-domain handling;
- signal-integrity constraints.

A conventional x64 DDR3 channel exposes dozens of high-speed data/strobe pins plus command/address pins. Recreating the PHY, capture registers, timing adjustment and state machine from individual transistors would rapidly become thousands to tens of thousands of discrete devices and would be physically much worse than doing the same thing inside silicon.

### Decision

**No discrete-transistor DDR3 controller.**

Discrete/minimal logic may remain conceptually useful after the data has already been captured, but the DDR interface must come from existing silicon.

---

## 5. The minimum silicon function we actually need

The minimum useful silicon block is:

```text
DDR3 PHY
   +
DDR3 controller
   +
small local arithmetic interface
```

or, if PHY and controller are split:

```text
DDR3 DIMM
   -> DDR3 PHY
   -> controller / command state machine
   -> user logic
   -> AND / popcount / accumulate
```

The project searched conceptually for a cheap commodity "DDR3 reader x64" IC that could expose a clean user-side data stream without a CPU or FPGA.

### Current finding

No cheap, generic, standalone DDR3 controller/PHY IC in the desired commodity form has been established yet.

Parts that look close are often only:

- registered-DIMM command/address buffers;
- scalable memory buffers tied to proprietary host-side links;
- DRAM interface IP intended to be instantiated inside an FPGA/ASIC;
- SoCs or CPUs with integrated DDR controllers.

A memory buffer is not automatically a standalone controller.

### Intel C104 example discussed

The Intel C104 class of scalable-memory buffer is interesting because it sits between an Intel host memory link and DDR3 channels and has relatively low power compared with a full CPU. However, it is not a generic independently programmable DDR3 reader: its host side is tied to Intel's memory architecture/protocol.

Therefore it is not currently a drop-in replacement for the CPU.

---

## 6. CPU vs FPGA: what the comparison really is

The wrong comparison is:

```text
CPU compute vs FPGA compute
```

For Transit, the more useful comparison is:

```text
cost + watts of obtaining N working DDR interfaces
```

### CPU

Advantages:

- DDR PHY/controller already solved;
- very cheap on the used enterprise market;
- software development is easy;
- memory training/refresh/protocol are handled;
- compute is already present.

Disadvantages:

- general-purpose cores/cache/uncore consume power;
- often only 3–4 DDR3 channels per socket in commodity x86 generations;
- physical module becomes closer to a computer than a tiny arithmetic controller.

### FPGA

Advantages:

- compute pipeline can be exactly Transit-specific;
- local reductions can be done before PCIe;
- no need for general-purpose cores;
- user logic can sit directly behind the DDR controller.

Disadvantages:

- sufficient DDR-capable I/O packages are expensive;
- DDR PHYs consume I/O banks/pins and impose board constraints;
- retail FPGA pricing can dominate the entire module BOM;
- a "cheap FPGA" may not have enough pins or memory interfaces for multiple x64 channels;
- implementation, timing closure and PCB signal integrity are harder.

### Important pin-count observation

A full x64 DDR3 interface consumes a large number of physical I/O pins. Four independent x64 channels can require several hundred pins before accounting for PCIe, clocks, power, control and miscellaneous I/O.

This is why a small low-cost FPGA that has plenty of LUTs can still be unusable: **logic capacity is not the limiting resource; package I/O is.**

This also explains why the AND/popcount engine can be tiny while the overall FPGA must still be large/expensive.

---

## 7. FPGA candidates discussed

### Lattice ECP5

ECP5 is attractive because it is relatively inexpensive and has open tooling/community support, but small packages do not have enough I/O for several independent full-width DDR3 channels.

The project previously identified this as a first-order constraint in `12-HARDWARE-CANDIDATES.md`.

A larger package may provide more I/O, but the FPGA cost then rises and can lose badly against a ~50 lei used CPU that already exposes four DDR3 controllers.

### Gowin GW2A-55 class

A high-I/O Gowin part was identified as a possible Chinese low-cost direction because some packages expose substantially more GPIO than small ECP5 packages and Gowin provides DDR3 interface IP.

Important status:

- physically interesting;
- exact package/bank pin mapping must be validated;
- simultaneous count of independent full x64 DDR3 interfaces must be proven in the vendor tools;
- current factory pricing must be requested directly;
- do not assume "enough total GPIO" means the DDR interfaces can be placed legally across banks.

### Decision

Do not buy FPGA parts in quantity until price/channel and real routable DDR channel count are known.

---

## 8. Custom ASIC discussion and decision

A custom Transit ASIC could theoretically contain exactly:

```text
DDR PHYs
+ small DDR controller state machines
+ MXFP/bitplane decode
+ AND/popcount or MAC lanes
+ accumulators
+ host link
```

At high production volume this could beat both CPUs and FPGAs in cost and watts per channel.

However, a custom chip introduces:

- design/tape-out/NRE cost;
- mask/shuttle cost;
- DDR PHY licensing/verification complexity;
- packaging;
- bring-up risk;
- minimum economic volume.

### User decision

**No custom chip.**

Transit should proceed using existing silicon plus a custom PCB/module. ASIC is explicitly out of scope for the current hardware phase.

---

## 9. Wider custom DDR buses: important correction

A custom board is not required to mimic a CPU's exact notion of "four memory channels" if raw DDR chips rather than standard DIMMs are used.

In principle, multiple x8 DDR3 devices can be operated in lockstep to form a wider data interface. For example:

```text
32 × DDR3 x8 devices -> 256 data bits per transfer
64 × DDR3 x8 devices -> 512 data bits per transfer
```

At DDR3-1600 the ideal payload arithmetic is:

```text
256 bits = 32 bytes
32 B × 1.6 GT/s = 51.2 GB/s

512 bits = 64 bytes
64 B × 1.6 GT/s = 102.4 GB/s
```

Those are equivalent in raw bus width to four or eight conventional x64 DDR3 channels respectively.

However this does **not** eliminate the need for a PHY/controller. It actually makes the I/O/package/signal-integrity problem larger. A device still has to drive/capture all those DQ/DQS lines correctly.

Therefore "one 512-bit DDR3 interface" is not a free way to avoid controller silicon. It is useful as a conceptual description of the bandwidth target, not yet a cheap implementation.

---

## 10. DDR3 vs DDR4 vs DDR5 bandwidth arithmetic

The project uses the working K3 routed-weight roofline from `10-KIMI-K3-TARGET.md`:

```text
~104 billion active weights/token
× 0.5 byte/weight at Q4-equivalent packing
≈ 52 GB/token
```

For a 100 token/s weight-path equivalent target:

```text
52 GB/token × 100 token/s ≈ 5.2 TB/s
```

This is a weight-path roofline, not an end-to-end K3 tok/s guarantee.

### Ideal payload bandwidth per conventional x64 channel

```text
DDR3-1600  = 12.8 GB/s
DDR3-1866  = 14.93 GB/s
DDR3-2133  = 17.07 GB/s
DDR4-2400  = 19.2 GB/s
DDR4-2666  = 21.33 GB/s
DDR4-2933  = 23.46 GB/s
DDR4-3200  = 25.6 GB/s
DDR5-4800  = 38.4 GB/s per 64-bit-equivalent DIMM aggregate
DDR5-5600  = 44.8 GB/s
DDR5-6400  = 51.2 GB/s
DDR5-8000  = 64.0 GB/s
```

DDR5 DIMMs expose two independent 32-bit subchannels, but the 64-bit-equivalent aggregate bandwidth above already accounts for the total data width and must not be doubled again.

### Ideal channel count for 5.2 TB/s

Using `5200 GB/s / per-channel bandwidth`:

```text
DDR3-1600  ~407 channels
DDR3-1866  ~349 channels
DDR3-2133  ~305 channels

DDR4-2400  ~271 channels
DDR4-2666  ~244 channels
DDR4-2933  ~222 channels
DDR4-3200  ~204 channels

DDR5-4800  ~136 channels
DDR5-5600  ~117 channels
DDR5-6400  ~102 channels
DDR5-8000  ~82 channels
```

This explains where the existing Transit target of roughly 304 DDR3 channels came from: it approximately matches the ideal DDR3-2133 payload needed for the 5.2 TB/s weight-path roofline.

### 80% sustained example

Real hardware will not sustain 100% theoretical bus payload. At 80% sustained efficiency:

```text
DDR4-3200 effective: 25.6 × 0.8 = 20.48 GB/s
5200 / 20.48 ≈ 254 channels

DDR5-6400 effective: 51.2 × 0.8 = 40.96 GB/s
5200 / 40.96 ≈ 127 channels
```

These are still only weight-path equivalents.

---

## 11. Channel count is not board count

The project repeatedly confused or mixed channel count with physical board count during exploration. Keep them separate.

For an example 127-channel target:

```text
4 channels/module  -> 32 modules
8 channels/module  -> 16 modules
12 channels/module -> 11 modules
16 channels/module -> 8 modules
```

For an example ~320-channel DDR3 target:

```text
4 channels/module  -> 80 modules
8 channels/module  -> 40 modules
12 channels/module -> 27 modules
16 channels/module -> 20 modules
```

The physical module count is set by **channels per memory-controller complex**, not directly by the model.

---

## 12. Why DDR4/DDR5 are not automatically cheaper even though they reduce channels

Faster DRAM reduces required channel count, which can reduce:

- number of controllers;
- number of CPUs/FPGAs;
- PCB count;
- power supplies;
- interconnect endpoints;
- cooling overhead.

But it can increase:

- price per DIMM;
- platform/controller price;
- PCB complexity;
- power-delivery complexity;
- minimum memory capacity purchased per channel.

The discussion identified DDR5 as a particularly strong example: a much smaller number of channels can hit the bandwidth roofline, but large DDR5 modules can cost thousands of lei each. If the module cost is in the 2,000–4,000 lei range, populating ~100+ channels immediately moves memory cost into hundreds of thousands of lei.

Therefore Transit must **not optimize channel count alone**.

The correct procurement metric is closer to:

```text
lei / sustained TB/s
```

with secondary metrics:

```text
watts / sustained TB/s
lei / usable TB of capacity
watts / channel
lei / controller channel
modules / system
```

DDR3 remains attractive because surplus capacity and old controller silicon can be extremely cheap even though more channels are required.

---

## 13. CPU generations and DDR3 channel-count lesson

The project considered whether one old CPU could provide dramatically more than four DDR3 channels.

Important caution:

- count **real independent data channels**, not DIMM slots;
- server memory risers and buffer chips can create many DIMM sockets without creating the same number of independent CPU channels;
- some enterprise architectures insert memory buffers between CPU and DIMMs, so the visible DIMM-side topology differs from the CPU-side link topology.

This is particularly relevant to Intel Xeon E7/R920-era memory-buffer architectures and IBM POWER/Centaur designs.

Do not use the number of physical DIMM sockets as the channel count.

Any candidate server/CPU must be mapped as:

```text
CPU/socket
  -> host-side memory links
  -> buffer chips (if any)
  -> independent DDR3 data channels
  -> DIMM sockets per channel
```

before it enters the Transit bandwidth model.

---

## 14. What would make an FPGA actually win

An FPGA should replace a cheap CPU only if the complete module wins materially on at least one of these:

1. more useful DDR channels per chip/package;
2. much lower measured watts per sustained GB/s;
3. significantly better kernel throughput at the same DRAM bandwidth;
4. much lower endpoint/module count;
5. required functionality that the CPU cannot implement without losing bandwidth.

If an FPGA costs several times more than a used CPU and exposes no more DDR channels, then its elegant arithmetic pipeline is economically irrelevant.

The arithmetic is cheap. The memory interfaces are what we are buying.

---

## 15. What would make a CPU actually win

A cheap CPU tile wins if:

```text
measured Transit kernel bandwidth
≈ measured raw memory bandwidth
```

while cost and power per channel remain acceptable.

In that case:

- the CPU already contains the required DDR PHY/controller;
- the simple arithmetic does not materially reduce bandwidth;
- no external FPGA/gate fabric is needed;
- development becomes software rather than board-level high-speed logic.

A CPU in this role should be viewed as a **cheap DDR-controller complex with free programmable arithmetic**, not as the central K3 processor.

---

## 16. Procurement architecture after this discussion

The current design request to a module manufacturer should **not** start with:

> Build us a CPU board.

or:

> Build us an FPGA board.

It should start with measurable requirements:

```text
Transit memory-compute module

- DDR generation: DDR3 first unless a better total-cost point is proven
- N independent full-bandwidth memory interfaces
- sustained sequential/model-shaped read bandwidth target
- resident weights
- local low-bit arithmetic / reduction
- activation input + reduced result output
- minimum BOM cost
- minimum watts per sustained GB/s
- no requirement to stream active weights through a central host
```

Then request multiple implementation BOMs only if the factory can supply them:

```text
A. used/embedded CPU controller complex
B. low-cost FPGA controller complex
C. any existing standalone DDR controller/SoC solution they already manufacture
```

The manufacturer should report:

```text
exact controller/CPU/FPGA part number
number of real independent DDR channels
supported DDR speed per channel
measured/estimated sustained GB/s
power per module
controller silicon cost
PCB/assembly cost
DIMM or raw-chip topology
host interconnect
minimum order quantity
availability of parts
```

---

## 17. Immediate experiments that decide the architecture

### Experiment A — CPU cost of arithmetic

On the exact cheap 4-channel DDR3 CPU candidate:

1. measure NUMA-local sequential read bandwidth;
2. run the same stream with the Transit bitplane kernel;
3. measure GB/s, Gweights/s and wall power;
4. compare raw-read vs compute-read throughput.

Acceptance question:

> Does the bitplane arithmetic prevent the CPU from saturating its own memory channels?

If no, keep CPU compute.

If yes, investigate dedicated compute.

### Experiment B — exact DDR3 channel map

For each server/CPU candidate:

- identify memory-controller datasheet;
- identify buffers;
- count true independent x64 channels;
- ignore DIMM-slot count until topology is known.

### Experiment C — factory RFQ

Ask the module manufacturer for real cost quotes for:

- 4-channel DDR3 CPU module;
- 8-channel DDR3 implementation if available;
- high-I/O FPGA implementation only if it can beat CPU cost/channel or W/channel.

### Experiment D — total-system cost by DRAM generation

For DDR3, DDR4 and DDR5 calculate from actual sourced parts:

```text
memory cost
+ controller/module cost
+ power/cooling cost
+ interconnect cost
```

and normalize to:

```text
lei / sustained TB/s
watts / sustained TB/s
```

Do not choose DDR4/DDR5 merely because channel count is smaller.

---

## 18. Current decisions

As of this branch:

1. **Arithmetic is not the reason to buy a large CPU or FPGA.**
2. **DDR PHY/controller availability is the hard hardware requirement.**
3. **Discrete-transistor arithmetic is possible in principle, but discrete-transistor DDR3 control is rejected.**
4. **Using a CPU only to read DDR3 and exporting all data to external gates is not attractive unless measurement proves the CPU arithmetic is a serious bottleneck.**
5. **A cheap used CPU may currently be the cheapest way to purchase multiple working DDR3 PHY/controller channels.**
6. **FPGA is justified only if complete cost/channel or W/GB/s wins, not because its compute architecture is prettier.**
7. **No custom ASIC/custom chip in the current phase.**
8. **No assumption that a cheap standalone DDR3 reader/controller exists until a specific purchasable part is identified and its host interface is understood.**
9. **DDR4 and DDR5 reduce channel count but must be judged by total lei/TB/s, not by channel count.**
10. **DDR3 remains the baseline because surplus memory/controller economics may dominate despite the larger channel count.**
11. **The next decisive evidence is a measured 4-channel DDR3 CPU benchmark and a real manufacturer BOM.**

---

## 19. One-sentence architecture summary

Transit does not need hundreds of powerful processors; it needs enough cheap DRAM interfaces to create the required aggregate weight bandwidth, with the smallest possible arithmetic engine placed immediately behind each interface.
