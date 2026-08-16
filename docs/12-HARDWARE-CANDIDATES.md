# 12 — Hardware Candidates and Decisions

This file records the hardware classes examined for Transit GPU and the current decision on each. It is intentionally conservative: interesting hardware is not called plug-and-play unless the complete path is known.

## 1. Dell PowerEdge R920

Role: **host/orchestrator candidate**.

Why it remains attractive:

- inexpensive used enterprise platform;
- four-socket NUMA;
- up to 96 DIMM sockets through its native memory riser system;
- very large DDR3 capacity;
- multiple PCIe expansion slots;
- good Linux/server ecosystem;
- can hold a very large checkpoint in host RAM if sufficiently populated.

What it is not:

- a motherboard with 300 directly attachable independent DDR3 channels;
- a 5.2 TB/s memory system;
- evidence that every visible DIMM/riser path behaves as one full independent x64 DDR3 channel to the CPU.

Current decision:

> Use one R920 as host if the price/configuration is favorable. Measure its real NUMA-local memory bandwidth. Do not solve the project by buying many complete R920s.

## 2. Native R920 memory risers

The R920's own memory risers are valuable because they are the supported way to populate its internal RAM capacity.

They should be used as designed by Dell for host memory.

They are **not** assumed to be generic detachable eight-channel Transit tiles.

Current decision:

- populate host RAM as needed;
- use host RAM for checkpoint staging/cache/KV/runtime;
- do not count native risers as a substitute for external local-compute tiles without measuring and understanding the SMB/NUMA topology.

## 3. IBM POWER7 eight-slot DDR3 memory riser

Observed physical value:

- eight DDR3 DIMM sockets;
- professionally routed high-speed board;
- active memory-buffer components;
- available as cheap enterprise surplus in some markets.

Important limitation:

> It is not a passive `one DDR3 slot -> eight independent slots` expander.

The board was designed around IBM's POWER memory interface and buffer chips. A compatible host interface or reverse-engineered programmable front end is required.

Current decision:

- interesting research target if exact buffer/protocol documentation appears;
- not the immediate plug-and-play solution for an R920;
- do not buy hundreds before one is proven controllable.

## 4. Generic server memory risers/backplanes

General lesson:

- many cheap enterprise memory boards exist;
- some have active buffer chips;
- some are DDR2 FB-DIMM, not DDR3;
- physical connector compatibility says nothing about protocol compatibility.

Example rejected from the visual search: HP ProLiant ML370 G5 memory boards are DDR2 FB-DIMM generation hardware and do not solve the desired DDR3 path.

Current decision:

> Search by exact part number, memory generation, buffer IC and protocol. No anonymous riser is treated as useful until those are known.

## 5. PCIe mining risers

Role: **cheap physical PCIe extenders**.

What they do:

- take a PCIe link, often x1;
- extend it over a cable;
- provide a powered mechanical x16 socket.

What they do not do:

- create x16 electrical bandwidth from x1;
- create extra PCIe lanes;
- act as DDR3 controllers;
- directly connect a memory riser to a host.

Why Transit can still use them:

- tile weights stay local;
- local DDR bandwidth does not traverse the mining riser;
- only activation/command/result traffic crosses PCIe.

Current decision:

> Good for laboratory fan-out if link stability and power quality pass sustained DMA tests. Replace with cleaner switch/backplane wiring later if needed.

## 6. PCIe switches / fan-out boards

Role: **make one R920 host many Transit endpoints**.

A proper PCIe switch has one/few upstream ports and multiple downstream ports. It does not magically multiply upstream bandwidth, but that is acceptable if the downstream tiles mostly process local weights.

Requirements:

- transparent enumeration;
- enough bus numbers and MMIO windows;
- sufficient aggregate activation/result bandwidth;
- Linux support;
- stable link training with chosen risers/cables;
- known switch chip/documentation where possible.

Current decision:

> Required at scale. Prefer a small number of large fan-out domains over a chain of fragile mystery multipliers.

## 7. YPCB-00338-1P1

Current role: **best laboratory Transit tile candidate found so far**.

Known useful properties from the public reverse-engineering ecosystem:

- Xilinx Kintex-7 XC7K480T-class FPGA;
- PCIe edge interface;
- two local DDR3 memory channels/banks;
- public Vivado board definitions and pin constraints;
- public system-test work involving DDR3 and PCIe/XDMA;
- LiteX board support exists;
- surplus pricing can be far below normal FPGA development-board pricing.

Why this matters:

```text
PCIe endpoint
+ programmable logic
+ local DDR3
+ public community documentation
```

is exactly the primitive needed to move the project from CPU theory to physical local-memory compute.

Limitations:

- only two DDR3 channels/banks, not eight;
- local DDR3 is soldered, not loose DIMMs;
- capacity is laboratory scale;
- board variants with similar names/layouts may not be compatible with the same reverse-engineered project;
- it is not 'plug in and K3 runs' — the FPGA bitstream/runtime/kernel still must be built.

Current decision:

> Buy/test one exact known-compatible board revision before any bulk order. Use it to prove Transit endpoint + DDR + kernel. Do not confuse prototype suitability with final scale suitability.

Public projects already identified during research:

- `TiferKing/ypcb_00338_1p1_hack`
- LiteX board definitions for `ypcb_00338_1p1`

## 8. Other cheap FPGA development boards

Boards with Artix-7/Zynq/ECP5 and soldered DDR3 are common enough to prototype controller code, but many are poor final Transit choices because:

- memory capacity is small;
- only one DDR channel exists;
- PCIe endpoint capability may be absent;
- high-pin-count packages are expensive;
- retail dev-board markup destroys cost/channel.

Current decision:

- useful only when a specific surplus board has exceptional price/capability;
- do not generalize 'FPGA board' into a cost-effective 8-channel tile.

## 9. ECP5-45 TQFP144 idea

A previous idea considered a low-cost Lattice ECP5-45 in a TQFP144 package.

Important correction:

- the package has roughly ~98 usable I/O, not enough to directly terminate many independent full x64 DDR3 channels;
- the package choice also lacks the high-speed serial resources expected from larger BGA variants.

Current decision:

> Do not use small-pin-count ECP5 packages as the basis for an eight-x64-channel tile. DDR PHY pin count is a first-order constraint.

## 10. Open-source DDR3 controller IP

### LiteDRAM

Useful because it provides an open configurable DRAM controller/PHY ecosystem across several FPGA families and is integrated with LiteX.

Transit relevance:

- good software/RTL starting point;
- can integrate PCIe/LitePCIe and local logic;
- board support may already exist for candidate FPGA cards.

### UberDDR3

Useful as another open DDR3 controller implementation, with features aimed at a practical controller rather than a purely educational design.

Transit relevance:

- controller RTL cost is zero;
- may be useful for custom board work or comparison.

### ultraembedded/core_ddr3_controller

Compact/open, but examples intentionally run at relatively low DDR clock rates compared with the Transit bandwidth target.

Transit relevance:

- good learning/reference core;
- not automatically suitable for a >100 GB/s eight-channel tile.

Current decision on open DDR IP:

> Reuse open PHY/controller infrastructure wherever possible, but remember that RTL cost is not the hardware cost. Pins, package, PCB, power and signal integrity dominate multi-channel DDR3 hardware.

## 11. SoftMC / DRAM Bender

Role: **research infrastructure for low-level DRAM control**.

These projects matter because they enable arbitrary/low-level memory commands and timing experimentation on FPGA platforms.

Potential Transit use later:

- RowClone-like copy behavior;
- unusual activation sequences;
- bitwise bulk operations;
- characterization of specific DDR3 chips/DIMMs;
- experimentation with compute-near/in-DRAM primitives.

Current decision:

> Future acceleration research. First make a standard-compliant tile fast and correct.

## 12. ComputeDRAM / PULSAR direction

Research literature demonstrates that commodity DRAM can sometimes perform useful bulk operations when driven outside conservative standard command sequences.

Relevance:

- fixed weight bitplanes may be a good match for bulk row operations;
- local DRAM operations could reduce external bus traffic further.

Missing piece:

- efficient local popcount/reduction/MAC-equivalent path still needs engineering;
- standard commodity DIMMs do not expose a ready-made complete inference primitive.

Current decision:

> Keep as a possible Transit V2/V3 tile enhancement, not a prerequisite for Transit V1.

## 13. Old server blades / motherboards as memory-controller tiles

Examples considered conceptually include old dual-socket DDR3 blades/motherboards with many DIMM slots.

Advantages:

- very cheap per DIMM slot in liquidation markets;
- CPU memory controllers already work;
- no DDR PHY design required.

Disadvantages:

- each 'tile' becomes a complete computer;
- power/cooling/network cost;
- limited local custom bitplane logic;
- communication typically Ethernet/PCIe rather than a simple endpoint;
- risks recreating the many-server solution that Transit is trying to avoid.

Current decision:

- useful as a fallback or temporary distributed-memory experiment;
- not preferred final architecture unless pricing becomes absurdly favorable.

## 14. LRDIMM/RDIMM and memory-buffer ideas

Important distinctions:

- RDIMM registers command/address to reduce electrical load but data still follows the memory channel architecture;
- LRDIMM introduces more buffering and is not interchangeable with arbitrary raw-controller assumptions;
- enterprise memory buffer chips often speak proprietary host-side protocols.

Current decision:

> A memory-buffer chip is valuable only if its host protocol, training and initialization are understood well enough to control it.

## 15. Flash/NAND/NOR hardware

Parts and concepts explored included SLC NAND packages, NOR/flash arrays and external programmers.

Current decision:

- commodity flash remains useful as storage;
- do not assume normal NAND interfaces expose analog compute-in-memory behavior;
- do not buy high-density BGA NAND for Transit compute until a concrete controllable mechanism exists;
- the project no longer accepts a huge discrete-memory BOM as the final path.

## 16. Optical/holographic concepts

Optical fixed transforms and holographic storage/compute were discussed because inference weights are static.

Current decision:

- intellectually relevant long-term research;
- not the near-term implementation path because precision, reconfiguration, interfacing and fabrication dominate.

## 17. Current hardware shopping rule

Before buying more than one unit of any surplus candidate, require:

```text
exact part/revision identified
photos match documented board
programmable device identified
PCIe endpoint path known
local memory topology known
power rails known
documentation / constraints / pinout available
one-unit programming success
one-unit DDR test success
one-unit sustained PCIe DMA success
one-unit Transit kernel success
```

Only then discuss bulk quantity.

## 18. Missing final hardware piece

The missing piece is now precisely defined:

```text
cheap Transit tile
  = one host-facing endpoint
  + ~8 independent DDR3 channels
  + enough capacity for useful expert placement
  + local programmable bitwise/reduction compute
  + documented/recoverable toolchain
  + low enough used/BOM cost to replicate ~38 times
```

Finding or constructing that tile is the current hardware search objective.
