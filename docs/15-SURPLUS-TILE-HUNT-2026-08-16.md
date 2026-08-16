# 15 — Surplus Tile Hunt — 2026-08-16

This note records a materially new candidate found during the ongoing Transit DDR3 tile search. It is intentionally conservative: marketplace labels are not treated as proof of exact populated memory capacity or firmware state.

## Nallatech PCIe-385N / Nallatech 385

### Why it is important

The Nallatech PCIe-385N is unusually close to a ready-made Transit memory-compute tile:

- Altera/Intel Stratix V FPGA;
- PCIe Gen3 x8 host interface;
- two independent DDR3 SDRAM banks directly coupled to the FPGA;
- product literature specifies up to 16 GB DDR3 on PCIe-385N variants;
- another Nallatech 385 product summary specifies 8 GB DDR3 arranged as two independent banks;
- half-length/low-profile PCIe accelerator/NIC form factor;
- historical Altera OpenCL support, so the board was explicitly sold as a programmable FPGA compute accelerator rather than a fixed-function NIC only.

This gives the exact Transit primitive:

```text
PCIe Gen3 x8 endpoint
        |
        v
Stratix V FPGA
   |          |
DDR3 bank A  DDR3 bank B
   |          |
local resident weights
        |
local Transit kernel / reduction
```

### Exact currently observed listings

1. **Nallatech 385N, model NT1D1-0473-V0502**
   - seller title: `Nallatech 385N Stratix V PCIe Altera FPGA Accelerator NT1D1-0473-V0502`
   - current observed price: **US$104.95**
   - condition: used, described as from a dust-free datacenter
   - interface: PCI Express
   - marketplace: eBay

2. **Nallatech 385N, P385-A72-0813P-81 / IBM 00NK0000**
   - current observed price: **US$189.00 or best offer**
   - quantity observed: 3 available, 2 sold
   - condition: fully tested / clean
   - interface: PCI Express
   - marketplace: eBay

3. A second `NT101-0473 PCIe-385N` listing was observed at **US$499.95**, which is not attractive relative to the US$104.95 unit but confirms active surplus availability of the same family.

### Public documentation state

Public product summaries document:

- PCIe Gen3 x8;
- Stratix V FPGA family;
- two independent DDR3 banks;
- up to 16 GB local DDR3 on PCIe-385N;
- direct coupling between DDR3 and FPGA;
- Altera OpenCL SDK support on the 385 accelerator family.

A historical Altera community thread confirms that PCIe-385N has **two DDR3 SDRAM banks** used as OpenCL global memory. Another thread states that boards shipped with a default FPGA image compatible with the Altera OpenCL SDK and could be enumerated/used after installing the matching software stack.

The weak point is board-specific low-level documentation. Historical users report that some Nallatech documentation required a registered account/serial number. Therefore we currently have stronger public architectural documentation than we have complete open schematics/pin constraints.

### Transit fit

**Strong prototype candidate.** Compared with the YPCB-00338-1P1, the 385N is attractive because it keeps the same basic `FPGA + PCIe + two local DDR banks` topology while offering a much stronger host link and potentially much larger local capacity.

Potential benefits:

- PCIe Gen3 x8 is far beyond what Transit needs for activation/result traffic and gives comfortable DMA/debug bandwidth;
- two independent memory banks can be treated as two local channels for placement/scheduling experiments;
- up to 8–16 GB local memory per card is large enough to hold nontrivial real-model shards/experts;
- Stratix V has substantial programmable logic and DSP resources for a first real local bitplane/MXFP kernel;
- low-profile datacenter form factor is compatible with dense server deployment;
- current US$104.95 surplus pricing is cheap enough to justify buying **one** for reverse-engineering/proof, not bulk yet.

### Important uncertainty

Do **not** assume the US$104.95 listing is populated with the maximum 16 GB. The marketplace listing identifies the card model but does not state installed DDR3 capacity in the visible metadata. Before bulk purchase, require either seller confirmation or inspection of the memory devices/part numbers from board photos.

Do **not** assume a mining-riser x1 path preserves the board's Gen3 x8 link. Transit can tolerate x1 for command/activation/result traffic if weights remain resident locally, but link training and compatibility must be tested.

Do **not** assume the original OpenCL BSP is sufficient for custom Transit RTL. The FPGA is programmable, but the exact flash/update/recovery route and pin/DDR constraints must be recovered before we call it a fully open platform.

### Recommended experiment

If one US$104.95 unit remains available, it is worth treating as the next alternative lab board after/in parallel with YPCB:

```text
1. identify exact FPGA device and DDR3 chips from photos
2. confirm installed DDR3 capacity
3. confirm PCIe enumeration in R920/Linux
4. obtain/recover original 385N BSP or board support package
5. dump/preserve factory flash before writing anything
6. prove host DMA
7. prove both DDR3 banks independently
8. port the Transit 64-element reference kernel
9. compare exact FPGA output against host golden reference
10. test through the intended x1 mining-riser/fanout path
```

### Current decision

> **Promising new candidate — buy/test-one tier, not bulk tier.**

The current ranking for inexpensive laboratory hardware becomes:

1. **YPCB-00338-1P1** — best documented/reverse-engineered cheap board, two DDR3 banks, strongest open-community path.
2. **Nallatech PCIe-385N at ~US$104.95** — materially stronger capacity/PCIe/compute potential; documentation risk is higher, but price is low enough that one-unit reverse engineering is justified.
3. **SQRL Acorn CLE-215+** — exceptionally good open LiteX/LitePCIe ecosystem, but only a 16-bit 1 GB DDR3 interface and current observed used pricing around EUR 122–147 or ~US$100 per unit, so it is less attractive than YPCB/385N for Transit bandwidth-per-dollar.

## Other candidate checked: SQRL Acorn CLE-215+

Useful properties:

- Artix-7 XC7A200T-3;
- 1 GB DDR3 on a 16-bit interface;
- PCIe Gen2 x4 through M.2;
- public LiteX board target;
- documented LitePCIe use and PCIe-based flash update;
- generic FPGA repurposing is well documented.

It is excellent as a software/toolchain fallback because the open-source path is unusually complete, but it is **not a better Transit memory tile** at current pricing because local DDR bandwidth/capacity per board are much lower than the 385N/YPCB options.

## Search conclusion for this run

The genuinely new result is **Nallatech PCIe-385N at US$104.95**. It is the first new surplus candidate in this search that combines:

```text
server-ready PCIe form factor
+ programmable high-end FPGA
+ two independent local DDR3 banks
+ multi-GB capacity
+ current low used price
```

That is significant enough to keep and investigate physically.
