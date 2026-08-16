# 15 — Hardware Search Matrix

This document consolidates the hardware pieces found or discussed across the TensorWave -> Transit exploration and records **what each piece actually does**, whether it is useful now, and what exact missing object we are still searching for.

The purpose is to stop repeating the same hardware search and to prevent attractive-looking surplus parts from being promoted without a complete control path.

## 1. Current architecture families

Transit currently has three hardware families worth preserving:

### A. DDR3 custom/local-compute tile

```text
R920/head
  -> PCIe switch/fanout
    -> Transit endpoint
      -> multiple local DDR3 channels
      -> FPGA/local compute
```

Best performance direction if the final cheap multi-channel tile can be found/built.

### B. Whole-server DDR2 memory-compute node

```text
head
  -> 10/25/40 GbE or InfiniBand
    -> old 8-socket server
      -> local DDR2
      -> local CPU compute
```

Best immediate path for exploiting extremely cheap complete enterprise servers without custom DDR PHY work.

### C. Hybrid

```text
fast DDR3/FPGA tiles -> hot experts
old DDR2 servers     -> cold experts / overflow / replicas / staging
```

Likely the most robust long-term architecture if both surplus classes remain cheap.

## 2. Consolidated part/candidate table

| Candidate | What it really is | Transit role | Status |
|---|---|---|---|
| Dell PowerEdge R920 | 4-socket DDR3 enterprise server | host/orchestrator, staging, router, reducer | keep |
| R920 native memory risers | OEM internal R920 memory path | populate host RAM only | keep in native role |
| IBM POWER7 8-slot DDR3 memory riser | active enterprise memory board with 8 DIMM sockets | potential pre-routed memory PCB | research; protocol/front-end missing |
| HP ProLiant ML370 G5 memory board | DDR2 FB-DIMM-generation memory board | teardown/reverse-engineering candidate | not a DDR3 solution |
| PCIe mining riser | powered physical PCIe extender, often x1 | cheap lab fanout for active endpoints | useful after DMA validation |
| PCIe switch/fanout board | true PCIe fanout | attach many active Transit endpoints to one host | required for tile scale |
| YPCB-00338-1P1 | Kintex-7 PCIe card with local DDR3 | first physical FPGA Transit proof | best lab candidate found |
| ECP5-45 TQFP144 | small-pin-count FPGA | proposed cheap DDR controller | rejected for 8 x64 DDR channels |
| LiteDRAM | open DDR controller/PHY ecosystem | FPGA DDR implementation base | useful |
| UberDDR3 | open DDR3 controller | alternative controller reference | useful |
| ultraembedded/core_ddr3_controller | compact open DDR3 controller | learning/reference | useful, bandwidth must be validated |
| SoftMC / DRAM Bender | FPGA low-level DRAM research platforms | future in-DRAM experiments | later |
| HP ProLiant DL785 G5 | 8-socket DDR2 server | complete memory-compute node | high-priority if extremely cheap |
| HP ProLiant DL785 G6 | 8-socket DDR2 server generation | complete memory-compute node | high-priority if extremely cheap |
| Sun Fire X4640 | 8 CPU modules, 64 DDR2 DIMMs | complete memory-compute node | high-priority if extremely cheap |
| BEE2/BEE3-class multi-FPGA systems | old research/emulation hardware class | possible dense FPGA+memory platform | search lead only; no purchase-ready unit proven |
| DDR1-era servers/boards | older low-density memory infrastructure | extreme-scrap experiment | low priority unless effectively free |

## 3. Dell PowerEdge R920

Already documented in detail elsewhere.

Keep it because it can centralize:

- model metadata;
- tokenizer/generation loop;
- routing;
- placement;
- reduction;
- checkpoint staging;
- NVMe;
- PCIe switching;
- network aggregation.

Do not misuse it as proof that 96 DIMM sockets equal 96 independent DDR3 channels.

## 4. R920 native memory risers

Correct interpretation:

```text
R920 CPU/SMB/NUMA memory subsystem
  -> OEM memory riser
    -> DIMMs
```

They are not generic external Transit tiles.

Use them only as Dell intended unless the complete electrical/protocol path is independently reverse engineered.

## 5. IBM POWER7 eight-slot DDR3 riser

Why it was exciting:

- eight DIMM sockets already routed professionally;
- memory buffer silicon already populated;
- enterprise signal integrity already solved on the board;
- used/surplus pricing can be extremely low.

Why it is not plug-and-play:

- it is not a passive one-channel-to-eight-channel multiplier;
- the host side was designed for IBM POWER memory infrastructure;
- buffer initialization/training/protocol are the real missing pieces.

What would make it useful:

```text
exact IBM part number
+ buffer IC identification
+ datasheet or reverse-engineered protocol
+ connector pinout
+ power rails
+ clock requirements
+ FPGA/bridge capable of speaking the host-side protocol
```

Until those are known, the board is a promising professionally routed PCB, not a usable memory tile.

## 6. HP ML370 G5 and similar server memory boards

The visual search found server memory boards with many sockets.

Important lesson:

- some are DDR2 FB-DIMM rather than DDR3;
- active buffer chips can hide proprietary or generation-specific interfaces;
- connector shape is irrelevant without protocol knowledge.

For the old-server DDR2 path, these boards can still be interesting as **donor/reverse-engineering objects**, but a complete working server is usually a much easier way to inherit the controller.

## 7. PCIe mining risers

A key correction in the project was realizing that an x1 mining riser can still be useful.

It is not useful for moving active weights.

It **is** useful when the endpoint computes locally:

```text
x1 PCIe
  -> command + activation
  -> local DDR read + compute
  <- result
```

Required one-unit tests:

- correct enumeration;
- negotiated link generation;
- sustained host-to-device DMA;
- sustained device-to-host DMA;
- simultaneous DMA on multiple risers;
- PCIe error counters;
- power stability;
- cable/EMI behavior.

## 8. PCIe switch/fanout

This is the proper way to host dozens of active FPGA endpoints from one machine.

Search target:

```text
used PCIe Gen2/Gen3 switch board or backplane
8–16+ downstream ports/domain
transparent Linux enumeration
known switch silicon
standard power input
cabled or slot-based downstream connectivity
```

Do not confuse USB-style mining-riser cables with USB protocol; on common mining risers the cable is simply a cheap physical transport for PCIe signals.

## 9. YPCB-00338-1P1

This remains the best **lab** Transit FPGA card discovered so far.

Useful properties already documented in the project:

- Xilinx Kintex-7 XC7K480T-class FPGA;
- PCIe edge connector;
- two local DDR3 banks/channels;
- public reverse engineering;
- public pin constraints/board definitions;
- LiteX ecosystem support;
- existing DDR/PCIe bring-up work in the community.

Use it to prove:

```text
PCIe command
-> local DDR resident weights
-> FPGA compute
-> local reduction
-> result over PCIe
```

Do not buy hundreds. It has only two DDR3 channels and small fixed local memory relative to the final K3 fabric.

## 10. ECP5 small-package idea

The idea was to use a very cheap FPGA as a DDR controller per group of DIMMs.

The failure mode was physical I/O count.

An 8-channel x64 DDR3 tile requires hundreds of high-speed I/O signals plus clocks/control. A low-pin-count TQFP device cannot terminate this no matter how small the controller RTL is.

Decision:

> Controller gate count is not the problem. DDR PHY pins/package/PCB are the problem.

## 11. Open DDR controller software/RTL

### LiteDRAM

Preferred ecosystem starting point when supported by the selected FPGA/board.

### UberDDR3

Useful independent open implementation/reference.

### ultraembedded/core_ddr3_controller

Useful for understanding a compact controller and for lower-speed experiments.

### SoftMC / DRAM Bender

Useful when Transit reaches the point of intentionally issuing unusual DRAM commands or testing compute-near/in-DRAM behavior.

None of these projects eliminate the physical requirement for enough FPGA I/O banks and a correct PCB.

## 12. HP ProLiant DL785 G5/G6

Why these became important later:

Instead of reverse engineering a memory riser, buy the **entire machine around the riser/controller** if the market considers it worthless.

Desired listing characteristics:

```text
complete chassis
all/most CPU-memory cells present
8 CPUs preferred
64 DIMM-socket-class configuration
working BIOS/iLO
PSUs included
PCIe risers/cages included
shipping cost sane
seller willing to provide internal photos
```

Memory is optional if DDR2 DIMMs can be sourced separately at scrap pricing.

Do not value a listing by chassis price alone. Missing CPU cells, memory trays, proprietary PSU modules or PCIe cages can destroy the economics.

## 13. Sun Fire X4640

This is one of the cleanest known DDR2 server candidates because the memory topology is physically modular:

```text
up to 8 CPU modules
8 DDR2 DIMM slots/module
64 DIMMs total
```

Desired listing characteristics:

```text
8 CPU modules installed
all CPU modules same/supported revision where possible
DIMM ejectors/sockets intact
4 PSUs included
PCIe cage intact
ILOM works
bootable Linux-compatible configuration
```

The onboard 1 GbE ports are sufficient for bring-up, but the target cluster should budget for faster PCIe NICs.

## 14. BEE2/BEE3 search lead

BEE2/BEE3-class old FPGA emulation/research platforms were investigated because they combine:

- multiple FPGAs;
- external DRAM;
- professionally manufactured high-speed PCBs;
- a hardware class that may be obsolete commercially.

No exact purchase-ready candidate was promoted because availability, board revision, toolchain and memory topology were not yet sufficiently pinned down.

Keep the class as a search keyword, not as an assumed BOM item.

## 15. DDR1 direction

DDR1 was considered only because the project is willing to exploit hardware with near-zero market value.

Reasons it remains low priority:

- much lower capacity per DIMM;
- much lower memory bandwidth;
- older CPU ISA and weaker compute;
- even worse power/space economics;
- more machines required for a 1.56 TB-class checkpoint.

Promote a DDR1 candidate only if the complete server/memory infrastructure is essentially free and unusually dense.

## 16. Exact object we still need for the DDR3 tile path

The ideal missing surplus object is:

```text
ONE active board
  host interface: PCIe Gen2/Gen3 or faster
  local compute: FPGA/ASIC sufficiently programmable
  memory: 4–8 independent DDR3 x64 channels
  capacity: DIMM sockets or >=64–128 GB useful local RAM
  local reduction
  documented/reverse-engineered boot/programming path
  low used price
```

Best-case version:

```text
PCIe endpoint
+ FPGA
+ 8 DDR3 DIMM channels
+ known pinout/toolchain
```

If this exact object appears cheaply, it can collapse the 38-tile design from a custom-hardware project into a system-integration project.

## 17. Exact object we still need for the server path

The whole-server search should prioritize **price per measured local memory bandwidth**, not just price per DIMM slot.

Target server characteristics:

```text
>=48 DIMM sockets preferred
DDR2 RDIMM/FB-DIMM only if memory is extremely cheap
>=4 NUMA sockets; 8 preferred
all CPU/memory modules included
Linux boots without proprietary external infrastructure
at least one usable PCIe x8-class slot for fast NIC
remote management preferred
complete PSUs/fans
standard rack power practical
```

For every listing record:

```text
asking price
shipping
CPU count/model
DIMM slot count
installed memory
maximum supported DIMM size
memory type/speed
number of PCIe slots
NIC options
PSU count/rating
weight
seller test status
missing modules
```

## 18. Exact network hardware we need

For one-server proof:

- onboard 1 GbE is acceptable.

For multi-server proof:

```text
10 GbE minimum useful target
25/40 GbE or InfiniBand preferred if similarly cheap
```

Shopping target:

```text
10/25/40 GbE or QDR/FDR InfiniBand NICs
one per compute server
one/two high-bandwidth NICs in the head
matching switch
DAC cables where possible
Linux-supported drivers
```

Avoid exotic interconnects whose cable/transceiver/licensing cost exceeds the servers.

## 19. Exact memory we need

Before buying DIMMs in quantity, determine the winner server's population rules.

Search priority:

```text
8 GB DDR2 registered ECC modules
matching ranks/part numbers per OEM population rules
large lots from one seller
known-good pulls
```

Why 8 GB is attractive:

```text
64 slots × 8 GB = 512 GB/server
```

But if 4 GB DIMMs are dramatically cheaper:

```text
64 × 4 GB = 256 GB/server
7 servers ≈ 1.79 TB raw capacity
```

so density should be optimized against total lot price, shipping, power and required server count.

## 20. Procurement gates

No bulk order until one-unit proof passes.

### Server gate

```text
boots reliably
all intended NUMA nodes visible
all intended DIMM slots usable
measured local memory bandwidth acceptable
Transit kernel throughput measured
fast NIC works
wall power measured
```

### FPGA/riser gate

```text
exact part/revision identified
programming path known
DDR trains
PCIe enumerates
sustained DMA works
Transit local compute works
```

### Memory lot gate

```text
exact DIMM part number
OEM compatibility verified
rank/population rules verified
small sample passes memtest
price/GB beats modern alternatives after shipping
```

## 21. Search decision rule

The next purchase should maximize:

```text
useful_score =
  measured_or_credible_memory_bandwidth
× usable_capacity
× controllability
/ (purchase_cost + missing_parts_cost + network_cost + power_penalty)
```

A server with 64 sockets but missing CPU modules is worse than a 48-socket complete machine. A cheap riser with no controllable protocol is worse than a slightly more expensive complete server. A powerful FPGA board with proprietary programming is worse than a slower board with public constraints if we cannot make the first one run.

The project should continue to prefer **manufactured complexity the market has already depreciated to near zero**.