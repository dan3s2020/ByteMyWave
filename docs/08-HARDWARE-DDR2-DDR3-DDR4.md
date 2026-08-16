# 08 — Hardware Survey: DDR2, DDR3 and DDR4

## Scope

This file records the hardware paths investigated for the distributed Kimi K3 track.

The goal is **not** to collect vintage servers. The goal is to minimize:

```text
cost per usable DIMM slot
cost per usable GB
cost per unit of sustainable memory bandwidth
cost per unit of useful CPU compute
```

while still obtaining a machine that can boot Linux, expose its RAM to local compute, accept an adequate network adapter, and participate in a distributed K3 runtime.

Market prices in this file are **leads observed during research on 2026-08-16**, not guaranteed inventory. Official vendor specifications and marketplace observations are deliberately separated.

---

# A. Non-negotiable compatibility rules

## DDR2 is not one interchangeable thing

Old servers may use:

- DDR2 ECC Registered / RDIMM;
- DDR2 Fully Buffered DIMM / FB-DIMM;
- different speeds, ranks and voltage rules.

A cheap DDR2 module is useless if it is the wrong electrical architecture for the chosen server.

Examples from the investigated platforms:

- **HP DL785 G6**: Registered DDR2;
- **Sun Fire X4640 / X4600 M2**: server DDR2 on CPU memory modules;
- **HP ML370 G5**: **FB-DIMM**, not ordinary Registered DDR2;
- Dell PowerEdge R900 and several Intel 5000-series platforms: FB-DIMM class.

Before buying hundreds of DIMMs, the exact label/part number must be matched to the exact server manual.

## DDR3 R920 rule

Dell PowerEdge R920 accepts **DDR3 ECC RDIMM or LRDIMM**, not desktop UDIMM. Dell documents 96 physical 240-pin memory sockets, organized as eight 12-DIMM memory risers, two risers per processor.

Official source:

- https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/technical-specifications?guid=guid-4980896f-cc3e-4d43-9f84-ef8f013b7404&lang=en-us
- https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/system-memory?guid=guid-a94c7f4a-512e-44cb-b142-ce638a9304ff&lang=en-us

## Passive adapters do not create memory channels

A passive PCIe/mining riser or DIMM mechanical extender cannot turn an R920's 96 supported sockets into 300 independently addressable CPU DIMMs.

The CPU/chipset memory controllers, SMI-2 links, memory buffers, firmware topology and electrical loading define the addressable DIMM topology. More than the platform-supported topology requires an actual active memory-expansion architecture, not a passive connector multiplier.

This correction is important because early brainstorming considered large numbers of risers/controllers. Those ideas remain useful as exploration, but **the procurement plan must use real supported memory slots or separate compute nodes**.

---

# B. DDR2 candidates

## B1. HP ProLiant DL785 G6 — primary 64-DIMM DDR2 target

### Verified useful properties

HPE documentation identifies Registered DDR2 memory options for DL785 G6, including an **8 × 8 GB = 64 GB registered PC2-5300 kit**, and the DL785 G6 architecture uses processor-memory cells around its multi-socket Opteron design.

The historical QuickSpecs used during this project research specify a maximum configuration of:

```text
64 DIMMs × 8 GB = 512 GB
```

for the fully populated eight-processor/eight-memory-cell configuration.

HPE option-parts source:

- https://support.hpe.com/hpesc/public/docDisplay?docId=c01883384&docLocale=en_US

Useful HPE memory option numbers from that official parts page:

- `408855-B21` — 16 GB Registered PC2-5300 kit, 2 × 8 GB;
- `495605-B21` — 64 GB Registered PC2-5300 kit, 8 × 8 GB;
- `497767-B21` — 8 GB Registered PC2-6400 kit, 2 × 4 GB.

### Part-number leads collected during market research

These are procurement search keys and must be checked against the exact HPE spare-parts guide before ordering:

- DL785 G6 system/chassis family: `AM437A`, `AM438A`, `AM439A`;
- processor/memory cell or board lead: `588797-001`;
- DL785 G5 family leads: `AM422A`, `AM423A`, `AM424A`, `AM427A`, `AM428A`, `AM429A`, `AM430A`, `AM431A`;
- G5 processor/memory cell leads: `491104-001`, `AH233-2109D`, `AH233-60005`.

### G5 caution

Do **not** automatically treat every DL785 G5 as equivalent to a fully populated G6. Exact board revision, processor-memory-cell count, supported DIMM size and firmware must be verified. The G6 64 × 8 GB configuration is the better documented procurement target.

### Capacity arithmetic

```text
5 servers × 64 slots = 320 DIMM slots
320 × 8 GB = 2.56 TB nominal RAM
```

This is enough nominal capacity to hold the ~1.56 TB K3 checkpoint plus substantial runtime headroom.

---

## B2. Sun Fire X4640 — verified 64-DIMM DDR2 target

Oracle documents:

- 4U chassis;
- up to 8 CPU modules;
- each CPU module contains **8 DIMM slots**;
- maximum **64 DIMMs / 512 GB**;
- 8 PCI expansion slots.

Official source:

- https://docs.oracle.com/cd/E19273-01/html/835-0781/sfx46sm.gixmp.html

Capacity:

```text
5 × X4640
= 5 × 64 DIMM
= 320 DIMM slots
= 2.56 TB with 8 GB DIMMs
```

### Part-number search leads

Collected during market research:

- motherboard: `511-1387`;
- 8-DIMM CPU/memory board: `511-1461`;
- CPU module leads: `541-4146`, `541-4147`;
- chassis subassembly: `350-1476`;
- base chassis: `599-3661`;
- option leads: `X8486A`, `X8487A`.

These part numbers are useful for searching Alibaba/e-waste suppliers, but actual stock must be requested as a **complete 8-module machine or complete required subassemblies**. A motherboard alone at legacy-spare pricing is not useful.

### Marketplace lesson

A user-supplied Alibaba screenshot showed a `Sun Fire X4640 server motherboard 511-1387-0` at approximately **5,808–7,720 RON**. That is a legacy-replacement price and is economically irrelevant to TensorWave. It demonstrated why search must target complete scrap/barebone systems, dismantlers, and obscure FRU numbers rather than retail spare parts.

---

## B3. Sun Fire X4600 M2 — verified 64-DIMM option only with the right CPU modules

Oracle documents that X4600 M2 can use 8-DIMM CPU modules and explicitly states the maximum configuration:

```text
8 CPU modules
× 8 DIMM sockets per module
× 8 GB per DIMM
= 64 DIMMs
= 512 GB
```

but the required module is specifically the **`501-7817` split-plane CPU module**.

Official source:

- https://docs.oracle.com/cd/E19121-01/sf.x4600/819-4342-18/html/z40007e81010242.html

This means an arbitrary X4600/X4600 M2 listing is **not sufficient**. The seller must confirm the 8-DIMM split-plane CPU modules.

Capacity with five correctly populated machines:

```text
5 × 512 GB = 2.56 TB
```

---

## B4. IBM System x3850 M2 — 32-DIMM fallback

IBM documents the x3850 M2 as supporting up to **32 DDR2 DIMM slots per chassis**.

Official source:

- https://www.ibm.com/support/pages/overview-ibm-system-x3850-m2-type-7141-7144

IBM states up to 128 GB per chassis in the historical configuration described there; this makes it less attractive than a 64 × 8 GB DL785 G6/X4640 for capacity, but it remains a useful very-cheap-node candidate if complete servers appear as scrap.

Previously collected part-number leads include:

- motherboard/I/O board: `44E4485`, `43W8671`;
- memory card lead: `44E4252` / `43W8672`.

Verify all FRUs against the exact machine type before ordering.

---

## B5. HP ProLiant DL585 G5 — 32-slot-class candidate

This machine remains interesting because it is a conventional x86-64 four-socket Opteron server from the right low-value era and uses DDR2 Registered memory.

HPE still provides the retired DL585 G5 QuickSpecs reference:

- https://www.hpe.com/psnow/doc/c04282646

Part-number leads collected:

- rack CTO chassis: `455349-B21`;
- processor/memory drawer: `454592-001`;
- complete-system family leads: `448188-421`, `534498-001`, `534499-001`, `534500-001`.

A marketplace search surfaced `454592-001` at an unusually low apparent unit price. **That is not a bootable server by itself.** A purchase must include the system I/O board, processor/memory drawer, power infrastructure, CPUs, fans/cabling and any mandatory chassis electronics.

For procurement, treat the 32-slot/256-GB-class configuration as **pending exact QuickSpecs revision verification** before bulk DIMM purchase.

---

## B6. HP ProLiant ML370 G5 — cheap market evidence, but not our 8-GB-RDIMM solution

A user-supplied OLX listing showed a complete **HP ProLiant ML370 G5 at 250 RON**. This is valuable evidence that G5-era enterprise servers can reach true e-waste pricing in Romania.

However, the platform uses **PC2-5300 Fully Buffered DIMM (FB-DIMM)** and its official configuration is materially smaller than we initially assumed. The earlier brainstorming statement that it could simply take 16 × 8 GB must not be used for procurement.

The relevant historical HPE QuickSpecs are available here:

- https://www.hpe.com/psnow/doc/c04282493

Project research on those QuickSpecs gives:

```text
2 memory boards × 8 slots = 16 slots
maximum = 64 GB = 16 × 4 GB
```

So ML370 G5 is useful as **price evidence**, not as the preferred K3 capacity node.

---

## B7. Other DDR2 systems encountered

These are retained so future searches do not repeat already-rejected paths without a new reason.

| Platform | Research status | Why it is not primary |
|---|---|---|
| Dell PowerEdge R905 | 32-slot-class DDR2 Registered candidate | useful only if e-waste cheap; lower density than 64-slot targets |
| Dell PowerEdge R900 | DDR2 FB-DIMM | wrong DIMM class if our bulk lot is Registered DDR2 |
| HP DL580 G5 | FB-DIMM generation | availability exists, but memory type/capacity must beat better targets |
| HP BL680c G5 | blade, DDR2 FB-DIMM | needs blade enclosure/power/fabric; user saw ~500 RON listing |
| Dell PowerEdge 1950 | DDR2 FB-DIMM-era | too few slots for observed ~600 RON price |
| Dell CS24-SC | DDR2-era | observed ~800 RON and low DIMM density |
| Intel S5000VSA systems | FB-DIMM-era | low density / poor cost per slot in observed listings |
| Intel SR1530SH / S3200SHL | DDR2-era | too few slots for observed ~700 RON price |
| Supermicro H8QMi-2 | multi-socket DDR2 board candidate | Alibaba listing price observed around hundreds of USD, not e-waste price |

The correct response to a bad retail price is not to force the architecture around it; it is to continue searching dismantlers and bulk scrap.

---

# C. DDR3 candidates

## C1. Dell PowerEdge R920 — primary high-density DDR3 target

This is the most important DDR3 machine found so far.

Dell officially specifies:

- **96 × 240-pin memory sockets**;
- DDR3 ECC **RDIMM and LRDIMM**;
- 1066 / 1333 / 1600 MT/s depending on configuration;
- up to 6 TB with supported DIMMs;
- eight memory risers;
- 12 DIMM sockets per riser;
- four channels per riser;
- two memory risers per processor, four processor groups total.

Official sources:

- https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/technical-specifications?guid=guid-4980896f-cc3e-4d43-9f84-ef8f013b7404&lang=en-us
- https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/system-memory?guid=guid-a94c7f4a-512e-44cb-b142-ce638a9304ff&lang=en-us

Dell also states that population should be balanced across processors/channels for best performance.

### 8 GB DIMM arithmetic

```text
1 × R920 = 96 × 8 GB = 768 GB nominal
2 × R920 = 1.536 TB nominal
3 × R920 = 2.304 TB nominal
4 × R920 = 3.072 TB nominal
```

Two machines are approximately the same order as the 1.56 TB checkpoint and provide effectively no safe runtime/headroom margin. Three are a much more realistic full-RAM capacity target. If the goal is physically to use **300 × 8 GB DIMMs**, three R920s expose only 288 slots, so four servers are required for slot count.

### Critical R920 correction

The R920 has **96 real supported sockets**. The project must not model mining risers or arbitrary PCIe-to-DIMM adapters as another 200 memory channels. Anything beyond the documented topology needs an active coherent memory-expansion architecture, or it must be implemented as independent distributed nodes.

### Supplier request for the bulk DIMM lot

For R920, the useful request is:

```text
8 GB DDR3 ECC RDIMM
server memory
exact part number and chip vendor
rank: e.g. 1Rx4 / 2Rx4 as applicable
voltage
speed grade, preferably compatible with R920
300 pcs
```

Do not order ECC UDIMM merely because the listing says “ECC”.

---

## C2. IBM System x3850 X5 + MAX5 — real vendor-designed memory expansion

IBM's X5 platform is interesting because it demonstrates the difference between a real memory expansion architecture and a passive riser hack.

IBM documents:

- `59Y6265` — **MAX5 32-DIMM Expansion Module**;
- `59Y6267` — eX5 MAX5 to x3850 X5 QPI cable kit (`40K6750` replacement/FRU reference in IBM documentation);
- DDR3 RDIMM options including 8, 16 and 32 GB classes;
- specific firmware and memory-population restrictions for MAX5.

Official sources:

- https://www.ibm.com/support/pages/memory-accessories-ibm-system-x3850-x5-x3950-x5
- https://www.ibm.com/support/pages/x3850-x5-max5-memory-restrictions-ibm-system-x3850-x5-type-7145
- https://www.ibm.com/support/pages/x3850-x5-minimum-code-level-required-attach-max-5-ibm-system-x3850-x5-7145-7146

This path is worth searching by FRU because old complete MAX5 modules may be ignored by normal buyers. Exact total DIMM topology for any candidate machine must be verified by machine type and installed memory expansion cards before purchase.

---

## C3. Other DDR3 marketplace observations

User-supplied OLX screenshots showed:

- HP DL380 G7, 2 × Xeon X5660, around **450 RON**, sold without RAM/HDD;
- Dell R610 around **500 RON**, with the listing incorrectly calling its RAM DDR2 — the R610 generation is DDR3;
- Supermicro X8DTU-F 2U around **600 RON**, no RAM.

These establish that complete DDR3 servers can be cheap, but the priority remains **DIMM density + bandwidth + total node cost**, not merely a low chassis price.

---

# D. DDR4 candidate

## D1. Supermicro H12DGQ-NT6 — potentially transformative if the Alibaba price is real

A user-supplied Alibaba screenshot showed:

- model: **H12DGQ-NT6**;
- supplier displayed: **Shenzhen All True Tech Electronic Co., Ltd.**;
- apparent price about **303.21 RON each at MOQ 10**;
- lower displayed tier prices at 100+ and 1000+ units.

This price is a **marketplace lead only** and is suspiciously low for the board. It must be treated as unverified until the seller proves actual working inventory and a complete bootable power/cabling solution.

### Official board specifications

Supermicro documents:

- dual Socket SP3;
- AMD EPYC 7002/7003 support;
- **32 DDR4 DIMM slots**;
- up to **8 TB ECC Registered DDR4-3200**;
- proprietary 17.32 × 14.29 inch form factor;
- board optimized for the **AS-4124GQ-TNMI** GPU server family;
- extensive PCIe 4.0 SlimSAS connectivity.

Official source:

- https://www.supermicro.com/en/products/motherboard/h12dgq-nt6
- https://www.supermicro.com/en/products/system/gpu/4u/as-4124gq-tnmi

### Why the proprietary form factor matters

Do not assume this is a cheap E-ATX motherboard that can be dropped onto a table with an ATX PSU.

Before purchase, obtain exact answers for:

- required power-distribution board;
- PSU model(s);
- motherboard power harnesses;
- power-on/front-panel interface;
- CPU heatsinks and mounting hardware;
- BMC/BIOS state;
- whether boards POST with standard EPYC 7002/7003 CPUs;
- whether any components/connectors are missing from dismantling;
- whether a stripped AS-4124GQ-TNMI chassis can be supplied cheaply.

### Best memory population concept

Each EPYC 7002/7003 socket exposes **8 memory channels**. A dual-socket board therefore exposes 16 channels.

An attractive initial population is:

```text
16 DIMMs per board
= 8 DIMMs per CPU
= one DIMM per memory channel
```

Across 10 boards:

```text
10 boards × 16 DIMMs = 160 DIMMs
160 × 16 GB = 2.56 TB nominal
```

This is a much cleaner K3-capacity target than 300 tiny modules.

With 32 GB DIMMs:

```text
160 × 32 GB = 5.12 TB nominal
```

### Memory-bandwidth ceiling

AMD documents the EPYC 7262, one inexpensive 7002-series candidate, as:

- 8 cores / 16 threads;
- dual-socket capable;
- 8 DDR4 channels;
- DDR4-3200;
- **204.8 GB/s theoretical per-socket memory bandwidth**.

Official AMD source:

- https://www.amd.com/en/support/downloads/drivers.html/processors/epyc/epyc-7002-series/amd-epyc-7262.html

Thus a dual-socket board has a **409.6 GB/s theoretical DRAM interface ceiling**, and ten boards have a 4.096 TB/s sum of socket-level theoretical ceilings.

That number is **not K3 throughput**. Real sustained K3 MXFP4 decode bandwidth can be far lower because of access pattern, CPU kernel efficiency, NUMA, quantization unpack, cache behavior and network synchronization. It is useful only to show why the DDR4 architecture has much more physical bandwidth headroom than DDR2-era machines.

---

# E. Capacity comparison for the K3 checkpoint

The official Hugging Face repository currently reports approximately **1.56 TB** for Kimi K3.

Nominal procurement arithmetic:

| Configuration | Nominal aggregate RAM | Result |
|---|---:|---|
| 160 × 8 GB | 1.28 TB | insufficient for full checkpoint in RAM |
| 200 × 8 GB | 1.60 TB | checkpoint barely fits nominally; insufficient operational margin |
| 240 × 8 GB | 1.92 TB | plausible minimum with some headroom |
| 300 × 8 GB | 2.40 TB | healthy capacity target |
| 320 × 8 GB | 2.56 TB | healthy capacity target |
| 160 × 16 GB | 2.56 TB | preferred DDR4 target |
| 160 × 32 GB | 5.12 TB | large headroom |

The table uses vendor-style nominal GB arithmetic for procurement. Exact usable bytes differ because DIMMs are binary-capacity devices, firmware reserves memory, and checkpoint/runtime allocations are not perfectly packed. **Never buy a configuration whose arithmetic only matches the checkpoint to within a few percent.**

K3 checkpoint source:

- https://huggingface.co/moonshotai/Kimi-K3/tree/main

---

# F. Candidate cluster configurations

## DDR2, 64-slot route

```text
5 × HP DL785 G6 @ 64 × 8 GB
= 320 DIMMs
= 2.56 TB nominal
```

or:

```text
5 × Sun Fire X4640 @ 64 × 8 GB
= 2.56 TB
```

or:

```text
5 × Sun Fire X4600 M2
with 8 × 501-7817 modules each
= 2.56 TB
```

This route minimizes node count but uses extremely old CPUs and memory controllers. It should be purchased only if the chassis price is genuinely e-waste level.

## DDR3 R920 route

```text
3 × R920 × 96 × 8 GB = 2.304 TB
```

This is enough capacity for K3 plus useful headroom, using 288 of the 8 GB DIMMs.

If we already own or insist on using 300 modules:

```text
4 × R920 = 384 available DIMM slots
300 × 8 GB = 2.40 TB populated
```

## DDR4 H12DGQ route

```text
10 × H12DGQ-NT6
2 CPUs/board
16 × 16 GB/board
= 160 DIMMs
= 2.56 TB
```

This uses one DIMM per memory channel and is currently the most technically attractive route **if and only if the board/chassis/power price lead survives verification**.

---

# G. Procurement ranking as of 2026-08-16

The ranking is conditional on actual quotes:

1. **H12DGQ-NT6 + cheap EPYC 7002 + 160 × 16 GB DDR4 RDIMM**, if the ~303 RON board listing is genuine and required proprietary infrastructure is cheap.
2. **Dell R920 + bulk 8 GB DDR3 ECC RDIMM**, if complete machines can be obtained cheaply; 96 sockets/server is excellent density.
3. **HP DL785 G6 / Sun X4640 / X4600 M2** only at true e-waste pricing, because the compute and power efficiency are poor even though 64 DDR2 slots are attractive.
4. 32-slot DDR2 machines only when their price per usable slot beats the above significantly.
5. 16-slot FB-DIMM boxes are price evidence and fallback capacity, not first-choice K3 nodes.

The final ranking must use measured **RON per useful GB/s**, not only RON per chassis.

---

# H. Seller verification checklist

For every candidate bulk purchase request:

```text
[ ] exact server / motherboard model
[ ] exact board revision / FRU / part number
[ ] actual photos of the units being sold
[ ] tested POST status
[ ] CPU count and exact CPU models
[ ] DIMM slot count physically present
[ ] exact supported DIMM type: RDIMM / LRDIMM / FB-DIMM
[ ] BIOS/BMC not password-locked or otherwise unusable
[ ] all memory risers / processor-memory cells included
[ ] all required power distribution boards and cables included
[ ] heatsinks and fans included
[ ] NIC slots usable
[ ] no proprietary missing enclosure dependency
[ ] EXW price
[ ] packed weight/dimensions
[ ] DDP Romania quote
[ ] return/DOA terms
```

For H12DGQ-NT6 specifically, ask whether the quoted value is **the full motherboard price, not a deposit/placeholder**, and request a single board for validation before a ten-board order if commercially possible.
