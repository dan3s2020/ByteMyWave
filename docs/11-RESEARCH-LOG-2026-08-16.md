# 11 — Research Log: Cheap-Memory K3 Hardware, 2026-08-16

## Why preserve the search path

This file records not only the current preferred design, but also the routes that were considered and rejected. The point is to avoid rediscovering the same dead ends later and to preserve the reasoning behind part-number searches.

Marketplace prices below are observations from user-provided screenshots or search leads on 2026-08-16. They are **not** guaranteed current offers.

---

## 1. Starting point: large quantities of inexpensive 8 GB server DIMMs

The original hardware question was effectively:

```text
If 8 GB ECC server DIMMs can be bought extremely cheaply,
what is the cheapest hardware that exposes enough real DIMM sockets
for a frontier-scale model checkpoint?
```

This moved the search away from buying expensive high-density modern RAM and toward old multi-socket enterprise servers whose resale value is close to e-waste.

The first rough capacity arithmetic was:

```text
160 × 8 GB = 1.28 TB
240 × 8 GB = 1.92 TB
300 × 8 GB = 2.40 TB
320 × 8 GB = 2.56 TB
```

After Kimi K3 appeared as the concrete model target, the official Hugging Face repository size of ~1.56 TB made the distinction critical:

- 160 × 8 GB: not enough for full RAM residency;
- ~200 × 8 GB: too close to checkpoint size to be safe;
- 240 × 8 GB: plausible with headroom;
- 300–320 × 8 GB: comfortable capacity.

---

## 2. DDR3 R920 investigation

Dell PowerEdge R920 became attractive because it has **96 real DDR3 ECC RDIMM/LRDIMM sockets**.

The supplier conversation for the intended 8 GB DIMM lot initially had to clarify:

```text
server use
DDR3 ECC
RDIMM (or supported LRDIMM), not UDIMM
exact frequency/rank/part number required
```

### Important dead end: multiplying DIMM sockets through passive risers

A brainstorming route considered whether large numbers of PCIe/mining risers or controller boards could let one R920 physically host hundreds of ordinary DDR3 DIMMs.

The corrected conclusion is:

```text
R920 supported CPU memory sockets = 96
```

A passive PCIe riser does not create additional CPU memory channels or address decoding. The 96 sockets are behind Dell's documented CPU/riser/SMI-2/memory-buffer topology.

Therefore the correct way to use hundreds of R920-compatible DIMMs is **multiple R920 nodes**, not hundreds of passive DIMM adapters in one server.

Useful arithmetic:

```text
2 R920 = 192 slots = 1.536 TB at 8 GB/DIMM
3 R920 = 288 slots = 2.304 TB
4 R920 = 384 slots; 300 populated DIMMs = 2.40 TB
```

---

## 3. DDR2 hunt: search for obsolete 32/64-slot machines

The goal shifted to machines so old that normal enterprise buyers no longer value them.

Initial 32-slot-class candidates included:

- Dell PowerEdge R905;
- Dell PowerEdge R900 (FB-DIMM caveat);
- IBM System x3850 M2;
- HP DL580 G5 (FB-DIMM caveat);
- HP DL585 G5;
- Supermicro H8QMi-2 board family.

The key insight was then to search for **64-DIMM platforms** so only five machines would be required for 320 × 8 GB.

---

## 4. 64-DIMM DDR2 targets

### HP ProLiant DL785 G6

Target geometry used in planning:

```text
64 × 8 GB = 512 GB/server
5 servers = 2.56 TB
```

Search evolved from server names to system/FRU strings:

```text
AM437A
AM438A
AM439A
588797-001
```

Additional DL785 G5-family search leads retained:

```text
AM422A
AM423A
AM424A
AM427A
AM428A
AM429A
AM430A
AM431A
491104-001
AH233-2109D
AH233-60005
```

The G5 must not automatically be treated as identical to a G6; exact memory-cell and DIMM-size support must be verified before purchase.

### Sun Fire X4640

Oracle verifies:

```text
8 CPU modules × 8 DIMM slots = 64 DIMMs
maximum = 512 GB
```

Part-number search leads preserved:

```text
511-1387 motherboard
511-1461 8-DIMM CPU/memory board
541-4146 CPU module lead
541-4147 CPU module lead
350-1476 chassis subassembly
599-3661 base chassis
X8486A / X8487A option leads
```

A user Alibaba screenshot then showed the exact `511-1387-0` motherboard at about **5,808–7,720 RON**.

That result was useful precisely because it was bad: it showed that conventional legacy-spare listings are the wrong market. TensorWave needs **scrap/barebone/dismantler pricing**, not a rare replacement board sold to someone maintaining old Sun infrastructure.

### Sun Fire X4600 M2

The useful variant requires:

```text
501-7817 split-plane CPU module
```

Oracle explicitly documents the 8-DIMM module and maximum 64 DIMM / 512 GB configuration. Not every X4600/X4600 M2 listing qualifies.

---

## 5. Romanian market evidence: complete old servers can be very cheap

User-provided OLX screenshots included:

### HP ProLiant ML370 G5 — ~250 RON

Seller text indicated 16 DDR2 slots and existing RAM.

This looked attractive as cost/slot evidence. Subsequent vendor-spec verification corrected an important assumption:

```text
ML370 G5 = PC2-5300 FB-DIMM
16 slots
official maximum 64 GB = 16 × 4 GB
```

So it is not a 16 × 8 GB Registered-DDR2 solution. It remains evidence that entire G5-era chassis can reach ~250 RON.

### HP ProLiant BL680c G5 — ~500 RON

Observed as a 4-CPU blade with 16 × 2 GB installed.

Rejected as primary because a blade depends on enclosure/power/fabric infrastructure.

### Dell PowerEdge 1950 — ~600 RON

DDR2-era complete machine, but too few slots for the observed price.

### Dell CS24-SC — ~800 RON

Again not enough slot density for the observed price.

### Intel SR1530SH / S3200SHL — ~700 RON

Too few useful memory sockets.

### Intel S5000VSA system — ~600 RON

Old FB-DIMM platform; inferior price/density for this project.

### HP DL380 G7 — ~450 RON

DDR3 generation, 2 × Xeon X5660, no RAM/HDD. Useful evidence that DDR3 chassis pricing can also be low.

### Dell R610 — ~500 RON

The marketplace description called its memory DDR2; the machine generation is DDR3. This is a reminder that seller descriptions are not specifications.

### Supermicro X8DTU-F 2U — ~600 RON

DDR3-generation board/server, no RAM. Retained as price evidence but lower density than R920.

### Dell PowerEdge 750 — ~300 RON

Too old/low-capacity to matter for the K3 RAM target.

---

## 6. DL585 G5 search by component number

The investigation found the processor/memory drawer search key:

```text
454592-001
```

and complete/chassis leads:

```text
455349-B21
448188-421
534498-001
534499-001
534500-001
```

A very cheap marketplace-looking `454592-001` result initially appeared promising.

Correction:

> A processor/memory drawer is not automatically a complete bootable server.

The procurement requirement must include:

```text
chassis
system I/O board
processor/memory drawer
CPUs
PSUs/power distribution
fans
cabling
all required risers/modules
```

This principle applies to every old enterprise platform.

---

## 7. Side exploration: DDR1

DDR1 server memory still exists, and the search briefly explored even older high-slot-count hardware.

Candidates included HP DL585 G1, Tyan S4881/M4881 and HP Integrity rx8620-class systems.

The route was deprioritized because:

- common practical DIMM density is much lower;
- the number of servers/modules rises sharply for ~2 TB;
- some large-memory systems are IA-64 rather than x86-64;
- legacy rarity can make “older” hardware more expensive rather than cheaper.

DDR1 is therefore not part of the current K3 procurement shortlist unless an exceptional scrap lot appears.

---

## 8. Breakthrough candidate: H12DGQ-NT6 DDR4 board listing

User-provided Alibaba screenshots showed:

```text
Supermicro H12DGQ-NT6
~303.21 RON each at MOQ 10
supplier displayed: Shenzhen All True Tech Electronic Co., Ltd.
```

Official Supermicro documentation shows:

```text
dual AMD EPYC 7002/7003
32 DIMM slots
DDR4-3200 ECC Registered
up to 8 TB
```

This changes the DIMM strategy completely.

Instead of 300 × 8 GB:

```text
10 boards × 16 DIMMs/board × 16 GB
= 160 DIMMs
= 2.56 TB
```

With dual EPYC and 8 DIMMs per CPU, this is one DIMM per memory channel across 16 channels/board.

### Why this is not yet a purchase recommendation

The board is proprietary and optimized for AS-4124GQ-TNMI. The apparent Alibaba price is suspiciously low.

Before ordering 10 boards, verify:

```text
real unit price, not deposit
actual stock photos
POST tested
board revision/serials
BIOS/BMC
power distribution board
PSUs
power harness
power-on/front-panel interface
heatsinks
mandatory chassis/riser parts
DDP Romania cost
```

If those dependencies are cheap, this route supersedes the DDR2 hunt technically.

---

## 9. Evolution of the K3 execution idea

### Version 1: split layers across servers

Initial concept:

```text
node 1 -> layers 0..x
node 2 -> next layers
...
```

This is valid for fitting the model, but a correction was necessary:

> pure layer pipeline does not multiply single-sequence autoregressive decode speed by the number of nodes, because the stages remain serial for each token.

### Version 2: hybrid layer + expert parallel

K3 is MoE and selects 16 of 896 experts per token.

Current architecture:

```text
layer/stage placement for capacity
+
expert sharding for concurrent selected-expert compute
+
NUMA-local weight ownership
+
activation/result transport over the network
```

This is the route that can make multiple independent memory controllers useful to **the same token**.

---

## 10. Throughput calculation that replaced hand-waving

Known official K3 quantities:

```text
104B activated parameters/token
2.8T total parameters
~1.56 TB released checkpoint
```

Optimistic pure 4-bit active-weight lower bound:

```text
104B × 0.5 byte = 52 GB/token
```

Checkpoint-average storage-ratio screening estimate:

```text
1.56 TB / 2.8T ~= 0.557 byte/parameter
104B × 0.557 ~= 58 GB/token
```

Approximate linear math:

```text
2 FLOP/active weight × 104B ~= 208 GFLOP/token
```

Therefore the initial 1 tok/s procurement screening target is approximately:

```text
72.5 GB/s effective critical-path parallel weight stream
312 GFLOP/s useful kernel capability
```

including simple headroom factors.

This does not prove 1 tok/s. The final proof is the measured full-model result:

```text
T_token <= 1.0 s -> >= 1 tok/s
```

The complete logic is in `09-KIMI-K3-THROUGHPUT-MODEL.md`.

---

## 11. Current shortlist

As of this research snapshot:

```text
#1 H12DGQ-NT6 + EPYC + DDR4
   if the suspiciously cheap board/power/chassis offer is real

#2 Dell R920 + cheap bulk DDR3 ECC RDIMM
   proven 96-slot server topology

#3 HP DL785 G6 / Sun X4640 / X4600 M2 DDR2
   only at actual e-waste prices
```

The next action is not to buy the largest lot immediately. The next action is to obtain and benchmark **one representative node** from the leading economically viable route.
