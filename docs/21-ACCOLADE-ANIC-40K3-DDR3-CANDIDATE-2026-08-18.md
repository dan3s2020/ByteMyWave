# 21 — Accolade ANIC-40K3 DDR3 Transit Candidate — 2026-08-18

## Why this is a material find

A current surplus listing exposes the Accolade ANIC-40K3 at **US$29.99 each**, with **more than 10 available**. This is materially interesting for Transit because the original manufacturer announcement documents the board family as an **FPGA-based PCIe Gen3 x8 accelerator with 8 GB onboard DDR3 DRAM**.

## Exact model

- Model / MPN: **Accolade ANIC-40K3**
- Current seller: TechyParts / eBay item 286444469298
- Current price observed: **US$29.99**
- Quantity observed: **>10 available**
- Condition: tested used pull
- Host interface: **PCIe Gen3 x8**
- Network I/O: 4 x 10GbE SFP+

## Memory

Accolade's 2013 ANIC-K3 announcement specifies for ANIC-40K3:

- **8 GB onboard DDR3 DRAM buffer**
- PCIe Gen3 x8
- FPGA implementation of Accolade APP v5.0

The exact DDR3 bus width, number of independent memory-controller channels, DRAM part numbers and sustainable memory bandwidth have **not yet been recovered from public documentation**. Therefore this board must not yet be counted as 2/4/etc. Transit channels simply from its total capacity.

## Programmable logic / controller path

The board is explicitly described by Accolade as an **FPGA-based Packet Capture and Application Acceleration adapter**. The APP 5.0 processing engine is implemented in the FPGA.

Important limitation: the exact FPGA device/package has not yet been verified from a reliable public source, and the publicly visible material does not establish an open arbitrary-RTL programming flow. Accolade historically shipped SDKs/drivers/APIs for its adapters, but that is not equivalent to published board constraints/schematics or a LiteX-style open FPGA flow.

## Transit fit

Conceptually the hardware already contains the desired primitive:

```text
R920
 |
PCIe Gen3 x8
 |
ANIC-40K3
 |-- FPGA / APP compute fabric
 |-- 8 GB local DDR3
 `-- local packet/flow processing datapath
```

At US$29.99, the capacity economics are unusually good:

```text
8 GB DDR3 / card
$3.75 per GB of local DDR3 before shipping
```

It is potentially much denser than Storey Peak because one endpoint carries 8 GB rather than 4 GB.

## What must be proven before bulk purchase

1. Identify the exact FPGA device from board photos or PCIe/firmware metadata.
2. Identify DDR3 chips and bus topology; determine real independent channel count.
3. Locate JTAG/configuration flash and preserve the factory image.
4. Determine whether custom FPGA images can be loaded, or whether the board is practically locked to Accolade firmware.
5. Measure sustained local DDR3 read bandwidth.
6. Verify PCIe enumeration through the intended x1 mining-riser/fanout path.
7. Run a minimal local integer/bitwise compute kernel against resident DDR3 data.

## Current verdict

**BUY/TEST-ONE TIER; do not bulk-buy yet.**

The price + 8 GB DDR3 + PCIe Gen3 x8 + real FPGA makes ANIC-40K3 one of the strongest unexplored surplus candidates in the current search. Its main risk is not hardware density but programmability/documentation: until arbitrary FPGA control and DDR topology are verified, it cannot replace Storey Peak/YPCB as the documented Transit reference tile.
