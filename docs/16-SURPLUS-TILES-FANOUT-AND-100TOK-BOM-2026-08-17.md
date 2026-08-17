# 16 — Transit Surplus Tiles, PCIe Fanout and 100 tok/s BOM — 2026-08-17

This document consolidates the surplus-hardware search performed on 2026-08-16/17 for the Transit GPU / ByteMyWave near-memory Kimi K3 architecture.

It is intentionally conservative about evidence levels. Marketplace prices and stock are snapshots and can change quickly. A listing title is not treated as proof of board population, firmware state or PCIe compatibility. Analytical weight-path throughput is not reported as measured end-to-end Kimi K3 throughput.

---

## 1. Current Transit target

Transit keeps large model weights resident next to programmable compute and uses the R920 primarily as host/orchestrator:

```text
Dell R920
  |
  | commands / activations / reduced results
  v
PCIe switch / fanout fabric
  |
  +-- memory-compute tile 0
  +-- memory-compute tile 1
  +-- ...
  +-- memory-compute tile N

Each tile:
PCIe endpoint
  |
programmable FPGA / GPU / NPU / SoC
  |
local DDR3 channels
  |
resident weight shard
```

The current ByteMyWave K3 sizing lower bound is:

```text
active parameters per token  ~= 104 billion
4-bit-equivalent bytes/weight = 0.5 byte
active weight payload/token   ~= 52 GB/token
```

Therefore a 100 weight-path tok/s equivalent requires approximately:

```text
52 GB/token * 100 token/s = 5.2 TB/s local aggregate weight bandwidth
```

The original architecture therefore used:

```text
38 logical tiles * 8 DDR3-2133 x64 channels = 304 independent channels
304 * 17.07 GB/s nominal ~= 5.19 TB/s
5.19 TB/s / 52 GB ~= 99.8 weight-path tok/s equivalent
```

This remains an analytical sizing target, not an end-to-end Kimi K3 benchmark.

---

## 2. Key discovery: Microsoft / HP Storey Peak

### Exact identifiers

Observed identifiers include:

- Microsoft / HP Azure Storey Peak
- `X930613-001`
- HP `861309-001`
- PCB family `DAT6MTHUEB0 Rev B`

### Hardware

```text
FPGA: Intel/Altera Stratix V GS
      5SGSKF40I3LNAC

Local memory:
  4 GB DDR3/DDR3L ECC
  9 x x8 DDR3 devices
  72-bit physical bus
  64 data + 8 ECC

PCIe:
  x16 physical edge
  FPGA exposes two PCIe Gen3 x8 hard-IP paths

Other I/O:
  2 x QSFP+
  USB/FTDI/JTAG path
```

### Documentation / reverse engineering

Storey Peak is unusually attractive because public reverse engineering has already reached the pieces Transit needs:

- FPGA identification;
- DDR3 device and pin mapping;
- DDR3 controller bring-up;
- ECC-width memory test;
- PCIe endpoint/DMA designs;
- Linux host-side access;
- JTAG / FTDI programming path;
- board-level constraints/reference work.

The public Storey Peak work reports DDR3 operating at 1600 MT/s and a real memory-checker read result of approximately **9.662 GB/s per board**.

This measured value is used below instead of the ideal DDR3-1600 x64 payload of 12.8 GB/s.

### Observed prices during the search

Prices moved significantly during the hunt. Observed snapshots included approximately:

- US$11.99 single-card listing;
- US$9.59/card multi-buy tier in one listing;
- US$18.39/card multi-buy tier in a later listing;
- US$19.95/card from another surplus supplier with 100+ units observed;
- EUR ~14.6–15/card from European surplus listings;
- Alibaba listings were much worse, around US$59–65/card.

Conclusion: **buy Storey Peak from Western datacenter-surplus sellers, not Alibaba, unless Alibaba pricing changes radically.**

### Transit fit

Storey Peak is currently the strongest proven low-cost micro-tile:

```text
1 board
= 1 independent local 64-bit DDR3 data channel (+ ECC)
= 4 GB resident DDR3
= 1 large programmable Stratix V FPGA
= PCIe endpoint
= ~9.662 GB/s measured local DDR read
```

Its weakness is density: one board gives only one independent DDR3 channel. Reaching hundreds of channels therefore means hundreds of physical boards.

---

## 3. Storey Peak scaling calculations

Using the measured per-board read result:

```text
B_board = 9.662 GB/s
payload = 52 GB/token
```

Weight-path roofline for N boards:

```text
tok/s_weight_path = N * 9.662 / 52
```

Representative points:

| Storey Peak boards | Local DDR capacity | Independent DDR channels | Aggregate measured-read equivalent | Weight-path tok/s equivalent |
|---:|---:|---:|---:|---:|
| 1 | 4 GB | 1 | 9.662 GB/s | 0.186 |
| 4 | 16 GB | 4 | 38.648 GB/s | 0.743 |
| 10 | 40 GB | 10 | 96.62 GB/s | 1.858 |
| 40 | 160 GB | 40 | 386.48 GB/s | 7.43 |
| 304 | 1.216 TB | 304 | 2.937 TB/s | 56.5 |
| 390 | 1.560 TB | 390 | 3.768 TB/s | 72.5 |
| 539 | 2.156 TB | 539 | 5.208 TB/s | 100.2 |

The 390-board point was used as a simple capacity reference for a ~1.56 TB K3 checkpoint footprint. The exact resident representation may differ after Transit-native conversion, metadata/layout changes, replication and non-expert state placement.

### Important interpretation

The ~100.2 number for 539 boards means only:

> local active-weight bandwidth divided by the 52 GB/token analytical lower bound.

It does **not** establish:

- complete MXFP4/MXFP8 correctness;
- full K3 numerical equivalence;
- attention/KDA throughput;
- router overhead;
- cross-tile reductions;
- KV/state cost;
- end-to-end generation at 100 token/s.

---

## 4. Current best two-channel FPGA candidates

### 4.1 YPCB-00338-1P1 / YZCA-00338-104

Observed configuration:

```text
FPGA: Xilinx Kintex-7 XC7K480T
PCIe: x8 physical / public PCIe projects
RAM: ~4 GB DDR3 total
     two independent DDR3 interfaces
     each 72-bit physical width including ECC-style width
DDR3 devices: 18 x MT41K256M8DA-125 class parts reported in reverse engineering
```

Public/community support is excellent for surplus hardware:

- reverse-engineered pinout;
- XDC/QSF-equivalent board constraints;
- MIG configurations for both DDR3 banks;
- LiteX board target;
- LiteDRAM validation;
- LitePCIe validation;
- openFPGALoader support;
- PCIe x4/x8 work.

Observed search-session prices moved through roughly US$51–62, with one later listing around US$54.92 plus shipping.

Transit verdict:

> **Best low-risk first physical proof board.** More expensive per DDR channel than the cheapest Storey Peak, but two channels and much cleaner open-source bring-up reduce engineering risk.

### 4.2 Nallatech PCIe-385N / P385

Exact identifiers observed:

- `NT1D1-0473-V0502`
- `P385-A72-0813P-81`
- IBM `00NK0000`

Architecture:

```text
FPGA: Stratix V family
PCIe: Gen3 x8
RAM: two independent local DDR3 banks
Known 385-family implementations: 2 x 4 GB = 8 GB
Product-family documentation: up to 16 GB on some variants
```

Historical Altera/Nallatech OpenCL BSP support exists, and OpenCL global memory used the two DDR3 banks. FPGA image flashing through the board software stack was supported historically.

Observed price snapshots:

- approximately US$104.95 for `NT1D1-0473-V0502`;
- approximately US$189 for `P385-A72-0813P-81 / 00NK0000`;
- much higher listings also exist and are not economically interesting.

Transit verdict:

> **Strong buy-one/test candidate.** Two independent DDR banks and 8 GB-class implementations are useful, but low-level board documentation is less open than YPCB.

---

## 5. Other surplus programmable candidates found

### 5.1 SimpliVity OmniCube accelerator

Observed identifiers:

- `510-000003`
- `500-000004`
- `503-000004`

Observed architecture/information:

```text
PCIe x8
FPGA-based accelerator architecture
1 x DDR3/DDR3L RDIMM slot on known variants
known configuration: 8 GB PC3L-12800R
additional NVRAM/flash circuitry on some boards
```

A research publication repurposed a `510-000003` board as an FPGA platform and used memory-connector pins as custom I/O, proving that external researchers have reprogrammed the FPGA hardware.

Observed prices included approximately:

- EUR 14.50 in one surplus listing;
- US$33.95 in another;
- historical ~US$14.99 sold-out listings.

Risk:

- FPGA part identification and complete FPGA-to-DDR pin mapping/toolchain were not recovered during this search;
- therefore this remains a reverse-engineering candidate, not a ready Transit tile.

Verdict: **buy 1–3 only if cheap stock remains; reverse-engineer before bulk.**

### 5.2 Napatech NT20E2-CAP

Observed identifiers:

- `810-0024-03-04`
- `810-0024-03-14`
- PBA `073-008700-*`

Observed architecture:

```text
PCIe Gen2 x8
FPGA-based packet processor
1 GB onboard SDRAM documented as DDR3 in Napatech material
2 x 10Gb SFP+
dual FPGA image banks / firmware reflashing support
```

Observed price snapshots:

- US$14.99 in one listing;
- US$19.99 with ~10 units in another.

Risk:

- exact FPGA part and arbitrary custom-bitstream flow were not established;
- vendor image flashing is documented, but that is not equivalent to an open custom FPGA flow.

Verdict: **very cheap sacrificial reverse-engineering board; not bulk until JTAG/pinout/custom bitstream are proven.**

### 5.3 Gidel HawkEye-20G-48

Architecture:

```text
FPGA: Arria 10 GX 480 class
PCIe: Gen3 x8 hard endpoint
RAM: 1 GB onboard memory plus SODIMM option up to 16 GB on documented variants
Memory generation: DDR4, not DDR3
Programming: Quartus + Gidel ProcDev / JTAG
I/O: 2 x SFP+
```

Observed prices around US$80–91.

Verdict: technically clean development platform, but too expensive for bulk and DDR4 rather than DDR3.

### 5.4 Microsoft Catapult V3 / Longs Peak

Observed identifiers:

- `M1040125-001`
- `M1037382-001`
- `M1030299-001`

Architecture:

```text
FPGA: Intel/Altera Arria 10 class
Local memory: two independent 72-bit DDR4 interfaces
PCIe: two FPGA Gen3 x8 endpoints on relevant Catapult V3 designs
Programming: Quartus, USB/JTAG; public reverse engineering and example designs
```

Public work includes DDR tests, PCIe tests, DMA, OpenCL BSP experiments and custom logic. A sister Catapult variant has been demonstrated through USB-cabled PCIe x1-style riser arrangements, which is highly relevant to Transit, although that exact demonstration must not be silently generalized to every Longs Peak board.

Observed price around EUR 59, with historical lower sold-out pricing around US$25 class.

Verdict: **excellent platform if it falls below ~EUR 25–30; currently too expensive for bulk.**

---

## 6. Non-FPGA memory-compute candidates

These are not as flexible as FPGA, but they already integrate memory controllers and programmable compute.

### 6.1 NVIDIA GRID K1 16 GB DDR3

Architecture:

```text
PCIe Gen3 x16
4 x NVIDIA GK107 GPUs
4 GB DDR3 local to each GPU
4 independent 128-bit DDR3 memory systems
16 GB total board memory
768 CUDA cores total
~130 W board class
```

Observed prices:

- EUR ~29.99/card in Europe;
- lot of 8 for US$165.95 (~US$20.74/card) in a later search.

Transit advantages:

- one physical board gives four independent compute+DDR3 islands;
- 16 GB total capacity;
- mature CUDA/OpenCL-era programming model.

Risks:

- GK107 / Kepler compute capability is legacy and requires old software stacks;
- the four 4 GB memories are not one unified 16 GB pool;
- x1 mining-riser behavior with all four GPUs was not demonstrated during this search.

Verdict: **very interesting alternative proof platform; buy/bench before any architectural commitment.**

### 6.2 Netronome NFE-3240 / SMA-AMDA0021

Observed architecture:

```text
NFP-3240 programmable processor
40 programmable RISC microengines
8 hardware threads per microengine
2 independent 64-bit DDR3-1333 controllers at chip level
known NFE configuration: ~4 GB DDR3
PCIe Gen2, known appliance card documentation includes x4 host link
```

Observed price around US$21.99 with >10 available in one listing.

Risk: old NFP-32xx SDK/compiler availability is much weaker than newer Netronome generations.

Verdict: **promising cheap compute+dual-DDR architecture, but software archaeology is the gating item.**

### 6.3 Netronome Agilio CX / NFP-4000

Observed identifier:

- `ISA-4000-25-2-2`

Architecture:

```text
NFP-4000
~60 programmable flow-processing cores
8 hardware threads/core
2 GB DDR3 onboard
DDR3 controller supports 2 x 32-bit or 1 x 64-bit organization
PCIe Gen3 x8
```

Software situation is much stronger than NFP-3240:

- upstream Linux NFP driver;
- host-loaded firmware model;
- public programmable data-plane ecosystems including P4/eBPF/XDP-era examples.

Observed price around US$65.55–69.

Verdict: **clean programmable-SoC path, but too expensive for mass DDR bandwidth compared with Storey Peak.**

### 6.4 Cavium / Marvell OCTEON II CN6870C board

Observed exact model:

- `CN6870C-210NV-M8-3.0-G`

Observed board configuration:

```text
OCTEON II CN6870
24 x cnMIPS64 v2 cores
2 x 4 GB PC3-1600 DDR3 mini-DIMM = 8 GB installed
PCIe x8 board edge
```

CN68XX family architecture supports multiple DDR3 ECC interfaces at SoC level; the exact routing of all chip-level interfaces on this specific surplus PCB must be verified.

Observed prices during the search:

- US$27.99 with a 40-unit stock snapshot from one seller;
- US$22.77 in another listing;
- later eBay indexing around US$13.69 in one search result.

Software assets include OCTEON SDK lineage, Linux/Simple Executive and GCC/MIPS tooling.

Risk:

- exact surplus-board boot chain, flash replacement, UART/JTAG and host-endpoint mode must be proven;
- x1 fanout behavior is unproven.

Verdict: **one of the strongest non-FPGA bargain candidates if custom code can be booted cleanly.**

---

## 7. PCIe fanout findings

### 7.1 Mining risers

Powered PCIe x1-to-x16 mining risers remain useful as cheap physical extenders.

Their role in Transit is:

```text
PCIe downstream x1 port
  |
powered x1-to-x16 mining riser
  |
Transit endpoint card
```

They are **not** PCIe switches and do not multiply one root lane into many endpoints by themselves.

Observed Alibaba pricing was approximately US$2.25–4.25 depending on model/MOQ, with examples around US$2.86–2.95 during the search.

### 7.2 Cheap 1-to-many mining splitters

Alibaba listings were found for cards marketed as one-upstream to multiple USB/riser endpoints, including `BS-PE-8USB`-class 1-to-8 boards at roughly US$15.80 in bulk.

These are attractive for experiments but must not be assumed to support an arbitrarily deep/cascaded 539-endpoint PCIe topology.

Test sequence:

```text
1 root port -> 4 Storey Peak
then 8
then 16/40
measure enumeration, reset, AER, link width, DMA stability
```

### 7.3 Gigabyte CPBG8A0

Exact board:

- `CPBG8A0 2OZ Rev 1.0`

It is associated with Gigabyte G431-MM0-class GPU-server infrastructure documented for **10 x PCIe Gen3 x1 GPU positions**.

Observed listing snapshot:

- around EUR 69 in one search;
- another snapshot around EUR 49.72;
- very large stock was observed.

Companion power board:

- `CPDGD31 2OZ`
- observed around EUR 49 with cabling in one listing.

Important limitation:

> CPBG8A0 has not been proven to be a self-contained 1-to-10 PCIe switch that can simply be connected to one R920 slot.

Its upstream connectors/topology/power are proprietary to the original server ecosystem until reverse engineered. It may be useful as a physical carrier/distribution board, but it cannot currently be counted as the missing R920 fanout switch.

### 7.4 Enterprise PCIe switch cards

True enterprise multi-port PCIe switch AICs were found on Alibaba and surplus channels, but representative prices in the search were hundreds of dollars per switch card (e.g. ~US$242 or ~US$440+ class listings depending on generation/port count).

At hundreds of Storey Peak endpoints, enterprise switches can dominate system cost and destroy the low-price advantage of the FPGA cards.

Therefore the **single biggest unresolved system-level economic question is cheap stable PCIe fanout**.

---

## 8. R920 role and compatibility

The Dell PowerEdge R920 remains the intended host/orchestrator.

It provides:

- checkpoint/atlas loading;
- topology discovery;
- expert placement;
- router/scheduler;
- command queues;
- activation distribution;
- result collection/reduction;
- host-side state where appropriate;
- NVMe staging;
- telemetry/recovery.

Transit deliberately does not require the R920 PCIe fabric to carry the full ~52 GB active-weight payload per token. Large weights remain local to the tile.

Therefore x1 downstream links can be viable if measured activation/result traffic fits comfortably and if endpoint compatibility/stability is proven.

A Storey Peak has been demonstrated in another Dell PowerEdge generation (R720) on PCIe Gen3 x8, which reduces server-compatibility risk but is not a substitute for testing the exact R920 + switch/riser topology.

---

## 9. 100 tok/s Storey Peak BOM — analytical build

This is the build required to reproduce ~5.2 TB/s of **measured-equivalent local DDR read bandwidth** using one-channel Storey Peak micro-tiles.

### Core compute quantity

```text
required boards = ceil(5200 GB/s / 9.662 GB/s)
                = 539 Storey Peak boards
```

This provides:

```text
539 independent DDR3 channels
~2.156 TB local DDR3 capacity
~5.208 TB/s aggregate measured-read equivalent
~100.2 weight-path tok/s equivalent
```

### Price snapshots

Storey Peak unit prices observed during the hunt ranged from roughly US$9.59 at a temporary bulk tier through US$18.39–19.95 at later/current surplus snapshots.

At US$18.39:

```text
539 * $18.39 = $9,912.21
```

At US$19.95:

```text
539 * $19.95 = $10,753.05
```

At the exceptionally low US$9.59 tier, if 539 units were actually obtainable at that same price:

```text
539 * $9.59 = $5,169.01
```

The low tier must **not** be budgeted as guaranteed because available quantity at that price was not 539 units.

### Riser estimate

For 539 x1 powered risers at approximately US$2.86–3.30:

```text
low  ~= $1,542
high ~= $1,779
```

### Switch/fanout estimate

This remains unresolved.

Two qualitatively different outcomes exist:

1. **cheap mining-style switch/splitter tree works reliably** — potentially low thousands of dollars;
2. **enterprise PCIe switches are required** — potentially tens of thousands of dollars at this endpoint count.

No responsible fixed final system price can be given until this is measured.

### Power envelope

Storey Peak is slot-powered and server-oriented. A conservative system sizing exercise used ~25 W/card class as an upper planning value before real Transit-kernel power measurements.

For 539 cards:

```text
539 * 25 W = 13.475 kW card-side planning envelope
```

After R920, switches, riser conversion losses, fans and storage, total electrical infrastructure should be planned in the ~16–18 kW class if worst-case card power is sustained.

This is not a measured Transit workload power number.

### Rough total

If cheap fanout is proven:

```text
Storey Peak cards      ~$9.9k–10.8k at current-ish $18.39–19.95 snapshots
powered risers         ~$1.5k–1.8k
cheap switch/fanout    rough ~$1k–2k placeholder
PSUs/distribution      rough ~$1.5k–3k placeholder
rack/cooling           rough ~$1k–2k placeholder
NVMe/checkpoint        rough ~$0.15k–0.3k
R920                    reuse existing target host, or add market price
-------------------------------------------------------
rough hardware envelope ~$16k–20k if cheap fanout works
```

If enterprise switching is required, the total can move into the **US$30k–40k+** class.

This is why the 539-board Storey Peak architecture is valuable as a proof and quantitative ceiling, but is probably not the desired final physical product.

---

## 10. Why 4–8 DDR3 channels per physical tile remain the real target

Storey Peak solves the hard logical question:

```text
programmable compute
+ local resident DDR3
+ PCIe endpoint
+ public bring-up path
```

But it gives only one channel per PCB.

The final economic hardware should ideally combine several channels behind one physical endpoint:

```text
1 physical tile
  |
  +-- DDR3 channel 0
  +-- DDR3 channel 1
  +-- DDR3 channel 2
  +-- DDR3 channel 3
  +-- ...
  +-- DDR3 channel 7
```

At the original 304-channel target:

```text
1 channel/board -> 304 physical boards
2 channels/board -> 152 physical boards
4 channels/board -> 76 physical boards
8 channels/board -> 38 physical boards
```

The channel count, not DIMM count, is the relevant bandwidth primitive.

Therefore the ongoing surplus hunt should prioritize, in order:

1. PCIe + FPGA/SoC + 4–8 independent DDR3 channels at <= US$30–60 per board;
2. enterprise surplus accelerators with multiple DDR3 banks or DIMM sockets;
3. boards with public/recoverable pinout, DDR PHY and programming flows;
4. true cheap PCIe switch/fanout hardware that enumerates many x1 endpoints from the R920;
5. only then custom PCB approaches.

---

## 11. Current ranking

### Tier A — immediate physical proof

**YPCB-00338-1P1**

- 2 real DDR3 banks;
- Kintex-7;
- PCIe;
- LiteX/LiteDRAM/LitePCIe path;
- strongest low-risk bring-up.

**Storey Peak X930613-001**

- cheapest proven FPGA+DDR3 micro-tile found;
- 4 GB;
- 1 independent DDR3 channel;
- public reverse engineering;
- measured local DDR result;
- excellent for x1 fanout experiments.

### Tier B — buy-one reverse engineering

**Nallatech PCIe-385N** — 2 DDR3 banks, Stratix V, 8 GB-class known implementation, Gen3 x8, but less-open board support.

**SimpliVity 510-000003 family** — extremely cheap FPGA+DIMM possibility, but FPGA/DDR toolchain needs recovery.

**Napatech NT20E2-CAP** — extremely cheap FPGA+DDR3 card, but arbitrary custom bitstream flow needs proof.

**Cavium CN6870C board** — 8 GB DDR3 + 24 programmable MIPS cores, strong if boot/custom-code path is recovered.

**GRID K1** — 4 independent GPU+DDR3 islands per physical card; strong alternative compute proof, legacy software caveat.

### Tier C — technically strong but price/format less attractive

**Gidel HawkEye-20G-48** — excellent programmable platform, but DDR4 and ~US$80+.

**Catapult V3 Longs Peak** — excellent public reverse engineering, dual DDR4 and multiple PCIe paths; attractive only at much lower surplus price.

**Netronome Agilio CX NFP-4000** — software-friendly programmable NFP + DDR3, but expensive per local-memory bandwidth.

### Fanout research priority

**CPBG8A0** — interesting 10-position x1 server backplane, but upstream behavior is not yet solved.

**Alibaba 1-to-8 mining splitters** — cheap enough to buy/test, but cascade/stability is unknown.

**Enterprise PCIe switch AICs** — technically clean but may be too expensive at hundreds of endpoints.

---

## 12. Minimum purchase before any bulk order

Do **not** buy hundreds of boards before the following experiment succeeds.

Recommended first hardware proof:

```text
Dell R920
  |
1 cheap PCIe fanout/splitter
  |
  +-- powered riser -> Storey Peak 0
  +-- powered riser -> Storey Peak 1
  +-- powered riser -> Storey Peak 2
  +-- powered riser -> Storey Peak 3
```

Acceptance criteria:

```text
[ ] all four endpoints enumerate after cold boot
[ ] all four survive reset/re-enumeration
[ ] each board loads a known-good custom bitstream
[ ] DDR3 calibrates on each board
[ ] each board passes a real DDR read/write test
[ ] all four run DDR workloads concurrently
[ ] each board receives command/activation traffic over the intended PCIe path
[ ] each board returns reduced results
[ ] no weight payload is sent back through the host
[ ] AER/link errors stay zero or understood
[ ] measured per-board DDR bandwidth is recorded under simultaneous load
[ ] x1 PCIe is shown not to bottleneck the measured activation/result workload
```

Then scale:

```text
4 -> 8 -> 16 -> 40 endpoints
```

Only after the 40-endpoint topology is stable should procurement for a hundreds-of-board Storey Peak build even be considered.

In parallel, continue the search for 4–8-channel physical tiles because they reduce endpoint count much more effectively than optimizing the riser tree.

---

## 13. Software/RTL path for the physical test

The existing Transit architecture already defines the logical flow:

```text
host:
  discover devices
  load resident weight block
  verify checksum
  send activation + RUN command
  wait completion
  compare result against software golden reference

tile:
  PCIe endpoint
  command parser
  activation buffer
  DDR reader
  bitplane/native-format compute
  local accumulator
  result DMA
```

First prove the already exact signed INT4 x INT8 reference decomposition on real local DDR data.

Then move to the actual K3 numerical bridge:

- native MXFP4/MXFP8-compatible logic; or
- validated one-time Transit-native low-bit conversion with stored scales.

No 100 tok/s end-to-end claim is valid until the complete K3 path is measured.

---

## 14. Open blockers

The current blockers are concrete:

1. **PCIe fanout** — stable, cheap, scalable R920 -> many x1 endpoint fabric;
2. **exact Storey Peak x1 endpoint configuration** — supported by Stratix V architecture in principle, but must be tested on the board/riser lane mapping;
3. **multi-board simultaneous DDR bandwidth** — must be measured rather than multiplied indefinitely from one-board results;
4. **power** — real Transit bitstream power per board under sustained memory+compute load;
5. **cooling** — dense server cards require directed forced airflow;
6. **MXFP bridge** — K3-native numerical correctness;
7. **real K3 expert** — one complete routed expert shard on physical tile hardware;
8. **routing/reduction** — multi-tile MoE stage;
9. **full-token profiling** — attention/non-expert/KV overhead;
10. **4–8-channel surplus tile** — the desired density breakthrough.

---

## 15. Procurement decision as of 2026-08-17

### Safe to buy now for experiments

```text
4 x Storey Peak X930613-001, if found near the low surplus price
1 x YPCB-00338-1P1 as the safest dual-DDR FPGA reference
1 x cheap 1-to-4/8 PCIe mining splitter
4–8 x powered x1-to-x16 risers
sufficient external PCIe power distribution
forced-air fans
```

### Do not buy in bulk yet

```text
539 x Storey Peak
hundreds of mining risers
large quantities of CPBG8A0
enterprise switch fabric
large PSU farm
```

until the 4/8/16/40 endpoint experiments establish the real fanout behavior and the local FPGA kernel proves that compute can consume DDR bandwidth.

---

## 16. Bottom line

The surplus hunt has established that the Transit primitive is not hypothetical hardware. Multiple cheap enterprise boards already contain combinations of:

```text
PCIe endpoint
+ programmable FPGA/GPU/SoC
+ local DDR3
+ server-grade power/PCB
```

Storey Peak is currently the strongest **price + documentation + proven DDR/PCIe** one-channel micro-tile. YPCB is the strongest **open-source dual-DDR FPGA laboratory tile**. Nallatech 385N is a strong two-bank higher-capacity alternative. Several non-FPGA cards provide additional compute+DDR options.

The unresolved problem has narrowed substantially:

> **find or build an inexpensive way to concentrate 4–8 independent DDR3 channels per physical Transit endpoint, and prove a cheap scalable PCIe x1 fanout from the R920.**

If Storey Peak is used alone, the analytical 100 weight-path tok/s configuration is approximately **539 boards**, **539 independent DDR3 channels**, **~2.156 TB local DDR3**, and **~5.208 TB/s measured-read-equivalent aggregate bandwidth**. That configuration is physically and electrically large enough that it should be treated as a quantitative reference architecture, not the preferred final purchase.
