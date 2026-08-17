# 20 — Sugon NetFirm-4E40-C Transit candidate — 2026-08-17

## Why this is a meaningful new candidate

The Sugon `NetFirm-4E40-C` is a surplus FPGA SmartNIC/accelerator built around a Xilinx Kintex-7 `XC7K325T-FFG900`. It is notable because a recent public reverse-engineering writeup (April 2026) documents enough of the board to move it beyond a blind surplus gamble: JTAG has been identified, PCIe is physically x8, DDR3 DIMM signaling is being traced, clocks have been identified, and custom Vivado bitstreams have already been loaded for probing.

For Transit, the primitive is:

```text
PCIe x8 endpoint
      |
Kintex-7 XC7K325T
      |
DDR3 DIMM/interface
      |
resident weight shard + local compute
```

## Exact observed listing

- Model: `Sugon NetFirm-4E40-C`
- FPGA: Xilinx Kintex-7 `XC7K325T-FFG900`
- Current observed eBay price: **US$81.15** plus approximately **US$3 shipping** from China
- Condition: pre-owned
- Seller: olddays168

This is not yet a bulk-buy recommendation; the current value is primarily the unusually good reverse-engineering state relative to many anonymous surplus cards.

## DDR3 configuration

The board has a real DDR3 memory-module interface/slot directly connected to the FPGA. The reverse-engineering author explicitly discusses probing the DDR3 DIMM signals, notes that one DDR3 group involves more than 100 signals, identifies DDR I2C `SCL/SDA`, and confirms the DDR3 I/O bank voltage as 1.5 V.

At this stage the publicly visible writeup supports treating it conservatively as **one DDR3 interface/channel group**. Installed DIMM capacity and validated sustained DDR3 bandwidth are not yet established in the material inspected for this update.

Do not assume arbitrary server RDIMMs are supported until the exact DIMM electrical topology and MIG configuration have been validated.

## PCIe interface

The public reverse-engineering work identifies **eight FPGA GTX lanes allocated to PCIe** and explicitly describes the board PCIe interface as **x8**. The expected GTX lane mapping is on Kintex-7 GTX quads 115 and 116.

The board also has four optical/SFP positions, but these are not required for the initial Transit role.

For mining-riser-style use, x1 downtraining is an experiment, not yet a proven property of this exact board.

## Programmability and documentation

The FPGA is fully programmable Kintex-7 logic. The recent reverse-engineering work has already established:

- JTAG header pinout (`TMS/TCK/TDO/TDI`);
- a usable custom Vivado design using JTAG-to-AXI and AXI GPIO;
- 100 MHz board differential clock pins;
- PCIe reset signal;
- DDR3 I2C pins;
- several board LEDs/control pins;
- GTX reference clocks including 125 MHz, 156.25 MHz and 622.08 MHz;
- the four SFP GTX lane mappings;
- likely PCIe x8 GTX lane mapping across quads 115/116.

This is materially better documentation than a marketplace-only board because we have a reproducible path to custom bitstreams and ongoing pin recovery.

## Transit fit

### Strengths

- real PCIe endpoint form factor;
- full Kintex-7 FPGA compute/control path;
- real external DDR3 module interface rather than tiny onboard SRAM;
- custom Vivado bitstreams already demonstrated by a third party;
- fresh public reverse-engineering effort specifically targeting DDR3/PCIe/JTAG;
- current used price around US$81 is low enough for a one-board proof.

### Weaknesses / unknowns

- only one DDR3 channel/interface is currently justified by public evidence, so it does not solve the final 4–8-channel-per-tile density target by itself;
- installed DDR3 capacity is not confirmed from the current listing information;
- no sustained DDR3 bandwidth measurement yet;
- no confirmed open LitePCIe/LiteDRAM board target at the time of this note;
- no demonstrated R920/mining-riser x1 enumeration yet;
- exact PCIe reference clock/lane implementation still requires final verification on the physical board.

## Current ranking

For the first physical Transit proof, `YPCB-00338-1P1` remains safer because it already has stronger LiteX/LiteDRAM/LitePCIe validation and two DDR3 banks.

However, `NetFirm-4E40-C` becomes a **strong secondary lab candidate** because the documentation gap is actively collapsing and it exposes a removable DDR3 memory interface plus x8 PCIe on a substantial Kintex-7 FPGA.

## Recommendation

> **BUY/TEST ONE if it can be obtained near US$80; do not bulk-buy yet.**

Required proof sequence:

1. inspect exact DIMM population and SPD;
2. preserve/dump any factory flash image;
3. reproduce JTAG programming;
4. complete DDR3 pin constraints/MIG setup;
5. measure sustained read bandwidth;
6. prove PCIe enumeration and DMA on x8;
7. test x1 mining-riser link training;
8. run the Transit resident-weight bitplane/MXFP proof kernel;
9. compare exact output against host reference;
10. only then evaluate cost/channel relative to Storey Peak and YPCB.

## Sources observed on 2026-08-17

- eBay current marketplace index: `Sugon NetFirm-4E40-C FPGA Smart NIC XILINX KINTEX-7 XC7K325T Acceleration card`, ~US$81.15.
- LM358 / cnblogs reverse-engineering article dated 2026-04-29: `XC7K325T_NetFirm-4E40-C`, documenting JTAG, DDR3 interface probing, board clocks and PCIe/SFP GTX allocation.
