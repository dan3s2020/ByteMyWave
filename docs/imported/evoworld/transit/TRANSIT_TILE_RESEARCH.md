# Transit Tile Research

_Last updated: 2026-08-23_

## Goal

Build a very cheap modular **Transit memory-compute tile** that can connect to the server cluster through PCIe fan-out and provide:

- PCIe endpoint connectivity to the host/server;
- local DDR3/DDR3L bandwidth;
- a programmable FPGA/SoC/controller path;
- cheap expansion through enterprise-surplus hardware;
- compatibility with external PCIe switch/backplane/riser topologies;
- enough documentation, pinout/JTAG information, or open-source support to make custom firmware realistic.

This is **not** intended to behave automatically as normal system RAM. The expected model is:

`host RAM / GPU -> PCIe -> Transit FPGA/controller -> local DDR3 -> compute/cache/reduction -> PCIe -> host/GPU`

For many tiles, the local memories form a distributed memory-compute pool rather than one coherent DIMM-style RAM space.

---

## Current cluster assumption

The current target is **4 servers**, each with roughly **2-3 usable PCIe/GPU slots**. Therefore, 100 Transit tiles cannot be attached directly. The scalable topology is:

`server PCIe uplink -> PCIe switch fabric -> many Transit tiles`

A practical target discussed is around **20-25 Transit tiles per server**, leaving the main GPU slots available for GPUs where possible.

For ~100 tiles total:

- ~24-25 tiles/server;
- 4 servers;
- one or more PCIe switch fabrics per server;
- external power and forced airflow for the FPGA cards.

Important: all downstream cards behind a PCIe switch share the bandwidth of the upstream link(s). 24 endpoints do not create 24 independent host-bandwidth links.

---

# Strong Transit-tile candidates

## 1. Microsoft / HP Azure Storey Peak / Catapult v2

### Exact models / part numbers

- Microsoft / HP **X930613-001**
- HP **861309-001**
- also appears under related OEM identifiers such as **0DX1C5** in resale listings

### Hardware

- FPGA: **Intel/Altera Stratix V GS 5SGSKF40I3LNAC**
- local memory: about **4 GB DDR3L ECC**
- DRAM implementation: 9x SK Hynix H5TC4G83BFR devices
- memory bus: **72-bit total = 64-bit data + 8-bit ECC**
- PCIe connector: physical x16
- FPGA contains two PCIe hard-IP x8 paths; practical designs can use one x8 endpoint when host bifurcation is unavailable
- 2x QSFP+ networking also present
- onboard FT232H / JTAG access

### Documentation / open source

Very strong compared with most enterprise-surplus cards.

Community work includes:

- `ruurdk/storey-peak` reverse-engineering repository;
- BOM and component identification;
- Quartus QSF / pin mapping work;
- factory firmware dumps;
- Linux and Windows JTAG work;
- PCIe endpoint experimentation;
- PCIe DMA examples;
- DDR3 access work;
- reported host-RAM <-> PCIe DMA <-> onboard DDR3 tests;
- experiments using one x8 link on Dell PowerEdge hosts without requiring x8/x8 bifurcation.

### Prices observed during research

Prices moved substantially during the search:

- ~$51-62 at older/current generic resale listings;
- $19.95 surplus listing with large stock;
- ~$17-18 delivered European listing observed;
- ~$11.99 single-unit eBay listing;
- volume break observed around **$9.59 each for 4+**.

Prices and stock must be rechecked before purchase.

### Why it is currently the best ultra-cheap candidate

At roughly $10-20 per card, it combines:

- 4 GB real local DDR3L;
- a large reprogrammable Stratix V FPGA;
- PCIe endpoint capability;
- a standard PCIe card form factor;
- unusually good reverse-engineering/community documentation.

Approximate density:

- 1 card -> ~4 GB local DDR3
- 10 cards -> ~40 GB local DDR3
- 100 cards -> ~400 GB distributed local DDR3

### Limitation

There are **no DIMM/SODIMM sockets**. The 4 GB is soldered to the board. You cannot add normal DDR3 sticks to Storey Peak.

---

## 2. Napatech NTMAINB1E2 / NT40E2 family

### Exact identifiers seen

- **NTMAINB1E2**
- **810-00026-02**
- NT40E2 family / NT40E2-4 variants

### Hardware / known properties

- PCIe x8
- FPGA-based architecture
- onboard DDR3 confirmed for the family
- exact memory capacity/topology for every NTMAINB1E2 resale variant still needs board-level confirmation before bulk purchase

### Documentation / reverse engineering

Useful community information exists:

- FPGA JTAG header identified;
- community pinout reported for J26:
  - TCK = pin 1
  - TDO = pin 3
  - VREF = pin 4
  - TMS = pin 5
  - TDI = pin 7
  - GND = pins 2/8
- experiments reported using the card on externally powered/mining-style PCIe risers.

### Price observed

- approximately **$15.99** in one surplus listing, materially below an earlier ~$24.95 listing.

### Transit assessment

Promising because it is cheap and appears reprogrammable, but it remains **higher risk than Storey Peak** until the exact FPGA and DDR3 capacity/bus topology are confirmed for the exact part number being bought.

---

## 3. Napatech NT40E3-4-PTP

### Hardware

- FPGA: **Xilinx Virtex-7 XC7V330T** reported by open-source board support
- memory: **4 GB DDR3-1866**
- memory bus: **64-bit**
- PCIe: **Gen3 x8**
- PCIe controller implemented in/through the FPGA design

### Documentation / open source

Better than the NT40E2 situation:

- official Napatech hardware/firmware documentation;
- open-source board target/support in the Taxi ecosystem;
- community reports of obtaining direct FPGA programming access;
- reported modification involving two resistors to bypass/redirect the JTAG path;
- custom PCIe example design reportedly programmed successfully on real hardware.

### Price observed

- around **$39.98 + shipping** in a good listing;
- other current European listings can be substantially more expensive.

### Transit assessment

Technically very strong: PCIe Gen3 x8 + fast 64-bit DDR3 + Virtex-7 + growing community support. Buyable if found again in the ~$15-25 range; otherwise Storey Peak wins on price.

---

## 4. Solarflare ApplicationOnload Engine SFA6902F

### Hardware

- FPGA: **Altera/Intel Stratix V GX A5**
- PCIe: **Gen2 x8**
- memory: **4x DDR3 SODIMM sockets**
- documentation/literature indicates up to **16 GB per socket**, potentially **64 GB total**
- 2x 10GbE networking

### Price observed

- around **$59.99 used / Best Offer** in a listing examined.

### Why it is unusual and interesting

Unlike Storey Peak, this card can potentially accept removable DDR3 SODIMMs. Therefore a single tile could offer far greater capacity.

Potential theoretical configuration:

`1 card + 4x16 GB DDR3 SODIMM -> up to 64 GB local RAM`

### Main risk

The FPGA-development flow is substantially more proprietary and less documented. Community comments indicate that converting it into a generic custom FPGA accelerator is not trivial.

### Transit assessment

Excellent memory density if FPGA/JTAG/DDR-controller access can be solved. Not currently as safe a prototype choice as Storey Peak.

---

## 5. GiDEL ProceV / ProceVD8-BXSM

### Exact models observed

- **ProceV Rev.2 PROCE VD8-BM**
- **ProceV Rev.3 PROCE VD8-BXSM**

### Hardware

- FPGA: **Stratix V GS 5SGSD8**
- memory: **2 independent DDR3 ECC SODIMM banks**
- typical documented population: 4 or 8 GB per bank
- maximum discussed configuration: **2x8 GB = 16 GB**
- bus: **72-bit per bank**
- memory type/speed: DDR3-1600
- documented sustained memory bandwidth around **19.2 GB/s total**
- PCIe: **Gen3 x8**
- DMA: up to 32 DMA channels documented
- JTAG available
- Quartus + GiDEL ProcDeveloper/ProcWizard software ecosystem

### Prices observed

- ~**$74.99 + shipping** for a 16 GB Rev.2 card
- lot of 4 Rev.3 cards around **$259.99**, or roughly **$65/card** before shipping

### Transit assessment

Probably the strongest **16 GB high-density programmable Transit tile** found so far because the board was designed explicitly around host PCIe <-> DMA <-> FPGA <-> DDR3 movement. More expensive per tile than Storey Peak but far more RAM and two independent memory channels.

---

## 6. Nallatech/BittWare PCIe-385N D5

### Exact identifiers

- **PCIe-385N D5**
- **NT1D1-0473-V0502**
- **NT101-0473**

### Hardware

- FPGA: **Altera Stratix V GS D5**
- memory: **8 GB DDR3 total**
- topology: **2 independent 4 GB banks**
- bus: **72-bit per bank**
- PCIe: x8; hardware supports Gen3 direct to FPGA, with some BSP/OpenCL configurations documented at Gen2
- networking: 2x SFP+

### Software/documentation

- commercial FPGA accelerator board rather than an opaque appliance;
- Intel/Altera OpenCL SDK usage documented in literature;
- host PCIe <-> FPGA <-> onboard DDR3 flow is an intended use case.

### Price observed

- around **$104.95 used** for NT1D1-0473-V0502.

### Transit assessment

Technically excellent and much easier to treat as a conventional FPGA accelerator, but currently too expensive to compete with Storey Peak for large quantities.

---

## 7. PMC/Microsemi Flashtec NV-1616

### Exact model

- **Flashtec NV-1616**
- part number **TCA-00364-08-D**

### Hardware

- local DRAM: **16 GB**
- PCIe: **Gen3 x8**
- controller: PMC/Microsemi Flashtec NV1600 family
- supports NVMe-style access and a Direct Memory Interface in the original platform
- flash backup / supercapacitor architecture intended for persistent memory behavior

### Prices observed

- **EUR 34.90** in a German listing with several units available;
- ~$39.99 each in a larger surplus quantity listing.

### Transit assessment

Very interesting as a **dense PCIe memory tile** but not currently proven as a programmable compute tile. Firmware/controller development access is proprietary and old development libraries are difficult to obtain.

Use case if software access is recovered:

`host -> PCIe Gen3 x8 -> Flashtec direct-memory interface -> 16 GB DRAM`

Do not count it as an FPGA compute tile unless the controller/firmware path is genuinely opened.

---

## 8. PMC/Microsemi NV-1604

- same broad Flashtec/NVRAM family;
- around **4 GB DDR3**;
- surplus prices observed around **$12.99-15.59**;
- very attractive as a PCIe memory appliance;
- currently not proven to expose a reprogrammable controller suitable for our custom compute path.

---

## 9. Microsoft Catapult v1 / Mt Granite

Technically very attractive:

- Stratix V GS family;
- approximately **8 GB DDR3-1333**;
- two 4 GB banks reported;
- PCIe Gen3 x8 architecture.

The problem is availability: no sufficiently cheap, clearly identified current listing was found during the research cycle. Keep it on the watch list.

---

## 10. IBM 98Y2610

### Hardware

- Cyclone IV GX **EP4CGX22BF14**
- PCIe connectivity
- schematics and community/open-source reverse engineering exist

### Why it was rejected as the main Transit tile

It lacks enough external DDR3 to solve the memory side of Transit without another memory board. Interesting as a cheap PCIe FPGA/controller board, but not a complete memory-compute tile by itself.

---

# PCIe fan-out research

## Why ordinary mining risers are not enough

A USB-cable mining riser normally provides only a PCIe x1 electrical path. It is useful for physical separation and power experiments but destroys most of the bandwidth available from an x8/x16 FPGA card.

For Transit, use a real PCIe switch fabric or x8/x16 cabled PCIe where bandwidth matters.

---

## Desired architecture for ~20-25 FPGA tiles/server

Conceptually:

```text
SERVER
  |
  +-- GPU(s) remain in primary GPU slots
  |
  +-- PCIe uplink x8/x16
        |
        +-- PCIe switch fabric
              |-- FPGA tile 1
              |-- FPGA tile 2
              |-- ...
              `-- FPGA tile 20-25
```

For 4 servers and ~100 FPGA tiles, target about 24-25 tiles/server.

The switch may be implemented as multiple smaller switch cards/backplanes. Cascading is possible, but every level adds shared-bandwidth and topology/compatibility considerations.

---

# PCIe switch candidates

## One Stop Systems OSS-BP-452 / PEX8796 architecture

### Hardware

- switch ASIC: **PLX/Broadcom PEX8796**
- PCIe Gen3
- **96 lanes**
- documented configuration:
  - x16 host/target side
  - **8 downstream x8 links**
  - remaining lanes available/unallocated depending on implementation

### Why it is architecturally ideal

It is almost exactly the fan-out we want for Storey Peak: real x8 downstream links rather than x1 mining-riser links.

### Problem

New OSS backplanes are extremely expensive (around thousands of dollars; one price observed around $2,398), so only surplus/decommissioned units are relevant.

Target surplus price should be roughly **under $100-150/backplane**, ideally much lower.

Do not buy new OSS hardware for this project.

---

## OSS/Magma ExpressBox 16

- external PCIe expansion chassis concept;
- turns a host PCIe connection into many external slots;
- useful architectural reference;
- complete units are generally too expensive unless found as datacenter surplus, broken/incomplete chassis, or bare backplanes.

Search terms worth monitoring:

- OSS 452
- OSS-BP-452
- Magma ExpressBox 16
- OSS expansion backplane
- PEX8796 backplane
- PEX8749 backplane
- PEX8733 expansion
- Netstor PCIe expansion
- Cyclone Microsystems PCIe backplane

---

## Oracle/Sun 7096186 / 7064634

### What was initially attractive

- very cheap surplus listing around **$19.99**;
- uses a **PLX PEX8749**-class PCIe Gen3 switch architecture;
- marketed by Oracle as an 8-port PCIe/NVMe switch;
- four physical SFF-8643 connectors.

### Critical correction

This should **NOT** currently be purchased in quantity for Storey Peak fan-out.

Why:

1. the downstream interfaces were designed for NVMe/U.2 storage topology, not generic x8 FPGA cards;
2. four SFF-8643 connectors do not equal eight ready-to-use x8 PCIe slots;
3. SFF-8643 NVMe cabling normally exposes x4-style paths;
4. community reports exist where generic downstream device detection was problematic on non-Oracle hosts;
5. Storey Peak's PCIe topology involves x8 links, so we cannot assume that attaching only an arbitrary x4 lane group will enumerate/use the FPGA correctly.

### Current recommendation

At most, buy **one** as a low-cost experiment if desired. Do **not** order 12 until one complete chain has successfully enumerated a Storey Peak card.

Test chain concept:

```text
server x8/x16 slot
    -> Oracle 7064634/7096186
    -> SFF-8643 cable/breakout
    -> powered PCIe adapter/riser
    -> Storey Peak
```

Required proof before scaling:

- switch enumerates on our server;
- downstream bridge is visible;
- Storey Peak enumerates behind it;
- FPGA/JTAG remains functional;
- stable PCIe link width/speed is measured;
- DMA test passes;
- onboard DDR3 test passes;
- sustained transfer test passes without AER errors.

---

## Cheap bifurcation boards are not PCIe switches

Adapters such as `PCIe x16 -> 2x x8` may cost only tens of dollars, but they generally rely on **host bifurcation**. They do not create independent downstream PCIe topology like a PEX switch.

Use them only when the server BIOS/CPU/root-port explicitly supports the required bifurcation mode.

---

# Proposed prototype path

Do not jump directly to 100 cards.

## Phase 1 - One direct Storey Peak

Use:

- 1x Storey Peak X930613-001;
- one server PCIe slot;
- suitable external power/riser only if needed mechanically;
- strong forced airflow.

Validate:

1. PCIe enumeration;
2. Quartus/JTAG access;
3. known-good open-source bitstream;
4. onboard DDR3 read/write test;
5. PCIe DMA host <-> DDR3;
6. sustained transfer rate;
7. temperatures and power draw.

## Phase 2 - One real PCIe switch + 2-4 tiles

Validate:

- switch enumeration;
- multiple simultaneous endpoint enumeration;
- IOMMU / BAR allocation;
- PCIe reset behavior;
- simultaneous DMA;
- bandwidth sharing;
- AER/error stability.

## Phase 3 - 8 tiles/server

Use a known-good x8-capable switch/backplane. Validate BIOS resource allocation and sustained aggregate throughput.

## Phase 4 - 20-25 tiles/server

Only after Phase 2/3 is proven. Expect one or more switched fabrics per server rather than 20 direct server slots.

## Phase 5 - ~100 tiles / 4 servers

Target roughly 24-25 FPGA cards per server with external chassis/racks, separate power, and forced airflow.

---

# Power and cooling

Storey Peak and similar SmartNIC/FPGA cards were designed for server airflow. When mounted externally:

- provide forced airflow directly through/over heatsinks;
- do not rely on passive room airflow;
- use a separate PSU or appropriately rated server PSU for large card banks;
- ensure powered risers/backplanes provide correct slot power rails;
- common ground between host and external PCIe power domains is mandatory;
- avoid low-quality mining risers/cables for high-speed x8 Gen3 links.

---

# What 100 Storey Peak cards would actually provide

Approximately:

- **100 Stratix V GS FPGAs**;
- **~400 GB aggregate local DDR3L**;
- many independent local memory channels;
- distributed compute/control capability;
- shared host-facing PCIe bandwidth through the switch fabrics.

This is **not** equivalent to adding 400 GB of ordinary DIMM RAM to the operating system.

The software/runtime must explicitly shard data/work across the FPGA tiles.

---

# LLM / tokens-per-second reality

The cards should not be purchased under the assumption that 100 FPGA cards automatically provide 1000+ tokens/s.

Potential useful roles include:

- weight/cache staging;
- quantized matrix kernels implemented in FPGA fabric;
- reductions;
- compression/decompression;
- preprocessing/postprocessing;
- KV/cache experiments;
- many independent inference workers for small workloads;
- offloading memory-bound operators from GPUs/CPUs.

Actual LLM throughput will depend on:

- DDR bandwidth per card;
- FPGA clock and DSP utilization;
- quantization;
- PCIe topology;
- how often data crosses the host uplink;
- batching;
- model partitioning;
- synchronization overhead;
- quality of custom RTL/OpenCL/HLS kernels.

A 100-card array should be treated as a research accelerator fabric, not assumed to behave like 100 GPUs.

---

# Current ranking

## Ultra-cheap programmable tile

**#1 Storey Peak X930613-001**

Reason: cheapest combination found of usable local DDR3 + large FPGA + PCIe + strong reverse engineering.

## Higher-density programmable tile

**#1 GiDEL ProceVD8-BXSM**

Reason: 16 GB potential, dual 72-bit DDR3 banks, Gen3 x8, DMA-focused architecture and proper FPGA-development tooling.

## High-capacity socketed-memory experiment

**Solarflare SFA6902F**

Reason: four DDR3 SODIMM sockets and potentially 64 GB/card, but custom FPGA access is the major unresolved risk.

## Dense non-FPGA memory tile

**Flashtec NV-1616**

Reason: 16 GB + Gen3 x8 at low surplus price, but not yet a programmable compute tile.

---

# Purchase rules

1. Do not bulk-buy opaque enterprise switch cards simply because the PEX chip is attractive.
2. Verify exact downstream connector lane width and endpoint compatibility.
3. Prefer documented x8 downstream links for Storey Peak.
4. Buy 1-2 units first and validate PCIe enumeration + DMA + DDR before scaling.
5. Prefer enterprise surplus with public schematics, pinouts, Quartus/Vivado board files, JTAG access, or community reverse engineering.
6. Keep Storey Peak target price around $10-20/card.
7. Keep switched fan-out target well below 400 RON per switch/backplane, ideally under $50-100 in surplus.
8. Preserve GPU slots: ideally consume one host uplink per external FPGA fabric rather than one host slot per FPGA.

---

# Immediate next research target

Find a **sub-400 RON PCIe Gen3 switch/backplane** with:

- one host x8/x16 upstream;
- at least 4, preferably 8+, downstream ports;
- downstream links that can expose **real x8 PCIe** to generic endpoint cards;
- PEX8749/PEX8796/PEX8733 or equivalent;
- standard PCIe slots or documented cabled-PCIe connectors;
- Linux/server compatibility;
- cheap surplus availability in multiples.

Once that exact switch is found and its pinout/cabling is verified, produce a complete BOM for one server and then multiply it across all four servers.