# 23 — CebaTech / Blue Coat C30244X-B4 Transit Candidate — 2026-08-20

## Summary

A new low-cost enterprise surplus FPGA accelerator candidate was found: Blue Coat / CebaTech `102-02607`, board marking `C30244X-B4`.

Current evidence supports:

- Altera Stratix IV GX FPGA `EP4SGX230HF35C4N` on the populated board variant;
- one included 4 GB memory module, part `XZ8EH8E4GM-C-BC`;
- PCI Express deployment in the Blue Coat SG9000 platform; the SG9000 maintenance manual specifies PCIe x8 slots for option/acceleration cards;
- original CebaTech architecture was explicitly an FPGA/programmable compression offload platform built around C-to-hardware / Verilog compression engines;
- current surplus pricing around US$42.99–52.99, with several units visible at Technology-Traderz/eBay.

This is potentially a cheap `PCIe + FPGA + 4 GB local memory` micro-tile, but it is NOT yet promoted to a DDR3-qualified Transit tile because the public evidence found so far does not establish the exact memory technology, bus width/channel count, board pinout, JTAG header, or arbitrary-bitstream development flow.

## Exact identification

Observed part numbers / markings:

```text
Blue Coat P/N: 102-02607
CebaTech board: C30244X-B4
FPGA: Altera Stratix IV GX EP4SGX230HF35C4N
Memory module: XZ8EH8E4GM-C-BC
Memory capacity: 4 GB (seller-confirmed)
```

The seller has one listing at US$42.99 for the card with the 4 GB memory module and another at US$52.99 explicitly identifying both the FPGA and the 4 GB memory module. Some localized eBay listings show five units available.

## PCIe interface

The SG9000 service manual documents the appliance option-card slots as PCI Express x8 and states that the compression acceleration card is installed in an SG9000 PCIe expansion slot. This establishes the host-side form factor/topology at the system level.

The Stratix IV GX device family has Intel/Altera PCIe IP support, so the FPGA itself is suitable for PCIe endpoint logic. That does not prove that the factory bitstream or PCB will down-train to x1 through a mining riser; this must be tested physically.

## Programmability

The populated FPGA is a standard Stratix IV GX `EP4SGX230HF35C4N`, which is reconfigurable hardware in principle.

Historical CebaTech material is relevant because the company developed compression subsystems using an ANSI-C-to-hardware flow and Verilog/CebaRIP compression cores. The Blue Coat card is therefore not merely an opaque fixed-function ASIC board: it contains a real FPGA-based acceleration path.

However, no public board-specific Quartus project, QSF pinout, schematic, JTAG mapping, DDR constraints, flash mapping, or open-source BSP for `C30244X-B4` was found in this search. Arbitrary user RTL on this exact board remains unproven.

## Memory state

Confirmed:

```text
capacity: 4 GB
module part: XZ8EH8E4GM-C-BC
```

Not yet confirmed from a primary/public technical source:

```text
DDR3 vs another DRAM generation
data bus width
ECC width
number of independent channels
DIMM/SODIMM/proprietary module electrical details
FPGA-to-memory pin mapping
sustained memory bandwidth
```

Because Transit specifically needs known local DDR bandwidth, this uncertainty is currently the main blocker.

## Current price / supply snapshot

Observed on 2026-08-20:

- Technology-Traderz direct: about US$42.99 for `102-02607 / C30244X-B4` with one 4 GB module.
- Technology-Traderz/eBay variant explicitly naming the Stratix IV FPGA + 4 GB module: about US$52.99.
- localized eBay listings showed several units / up to five visible units.

Marketplace prices and stock are snapshots and must be rechecked before purchasing.

## Transit fit

Conceptually, if the memory is confirmed as a useful DDR3 path and the FPGA can be reprogrammed:

```text
R920
  |
PCIe x8 / x1-riser test
  |
C30244X-B4
  |
Stratix IV GX EP4SGX230
  |
4 GB local memory
  |
resident weights + Transit kernel
```

### Advantages

- low surplus price;
- genuine server acceleration card;
- exact FPGA model known;
- 4 GB local memory already populated;
- PCIe host deployment is documented at the SG9000 platform level;
- FPGA family is fully programmable and supports PCIe IP.

### Blockers

- DDR3 type/bus width/channel count not yet proven;
- no public board schematic or pin constraints found;
- no public JTAG/flash reverse-engineering found;
- arbitrary custom Quartus bitstream flow not demonstrated;
- PCIe x1 mining-riser operation not demonstrated;
- local-memory bandwidth not measured.

## Decision

> **BUY/TEST-ONE tier only; not bulk.**

The board is interesting enough to reverse-engineer at ~US$43 because it combines a known Stratix IV GX with 4 GB local memory and a server PCIe form factor. It should not displace YPCB/Celestica/Storey Peak/GIDEL in the preferred list until the memory technology and FPGA control path are recovered.

## Required physical test sequence

1. photograph both PCB sides and all memory/module markings;
2. identify the 4 GB module electrically and confirm DDR generation;
3. locate JTAG/config flash and dump the factory image before modification;
4. enumerate in Linux/R920 and record PCIe IDs/link width;
5. test down-training through the intended x1 powered mining riser;
6. recover FPGA pinout enough to identify the memory bus;
7. build a minimal Quartus heartbeat/JTAG image;
8. bring up the local memory controller;
9. measure sustained sequential read bandwidth;
10. only then port the Transit bitplane/MXFP kernel and compare against the software reference.
