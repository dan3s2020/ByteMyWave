# Transit DDR2 Tile Rev A — engineering handoff

## Purpose

Build a manufacturable DDR2 memory-compute tile for Transit using a proven BEE3-style topology instead of inventing a new DDR PHY. Rev A is intended to validate the electrical architecture, DDR2 bandwidth, FPGA compute kernel, power, and host connectivity before any large production run.

## Proven reference architecture

Microsoft Research BEE3 used:

- 4 × Xilinx Virtex-5 FPGAs in FF1136 package (LX110T / LX155T / SX95T family);
- 16 × DDR2 DIMM sockets;
- 2 independent DDR2 channels per FPGA;
- up to 2 DIMMs per channel;
- 8 independent DDR2 channels total;
- standard PC-style power/cooling infrastructure;
- per-FPGA Ethernet / high-speed I/O.

Microsoft still publishes the BEE3 DDR2 DRAM controller (DDRCHv11.zip), including Verilog source and an ISE project. The new board should remain electrically compatible with the assumptions of that controller or an equivalent Xilinx MIG DDR2 implementation.

Reference links:
- https://www.microsoft.com/en-us/research/project/bee3/
- https://www.microsoft.com/en-us/download/details.aspx?id=52605
- https://casper.berkeley.edu/wiki/Dram

## Rev A architecture

### Memory/compute

- 4 × XC5VLX110T-1FF1136 (acceptable alternates: XC5VLX155T or XC5VSX95T in compatible FF1136 footprint only after pin/power review).
- 8 physically independent DDR2 channels total, 2 per FPGA.
- 16 × 240-pin DDR2 ECC Registered DIMM sockets, two sockets per channel.
- Target memory rate: DDR2-400 first-spin guaranteed target; layout should preserve margin for DDR2-533/667 where SI/timing permits.
- Target DIMM type for first bring-up: JEDEC DDR2 ECC RDIMM, 1.8 V, preferably single-rank or BEE3-known-compatible 4 GB modules.
- Do not claim 8 GB DIMM support until the controller/address mapping has been verified on hardware. PCB address routing should not unnecessarily preclude larger density devices.
- Each FPGA owns its two DDR2 channels and executes local Transit bitplane/masked arithmetic; weight data must not traverse the board-level host link during steady-state inference.

### Host/control interface

For Rev A, reliability and bring-up simplicity are more important than maximum host-link bandwidth.

- 1 × Gigabit Ethernet RGMII PHY + RJ45/magnetics per FPGA (4 ports total), or an equivalent four-port implementation if it materially lowers BOM/area.
- 1 × JTAG chain covering all four FPGAs, exposed on a standard Xilinx-compatible header.
- Optional UART per FPGA or one shared management UART through a mux.
- FPGA-to-FPGA fabric: at minimum 32 single-ended or 16 differential bidirectional-capable inter-FPGA signals between adjacent FPGAs for reduction/synchronization; route for timing closure at >=200 MHz where possible.
- Reserve test points/connectors for clock, reset, rail monitoring, and DDR calibration status.

### Clocks

- One low-jitter differential reference oscillator plus low-skew 1:4 fan-out to the four FPGAs, or independent oscillators if this improves DDR timing closure.
- Provide the clocks required by the selected BEE3 controller/MIG configuration.
- Do not finalize FPGA pinout before running the memory-controller implementation and confirming DQS, clock, PLL, bank, and pin-placement legality in Xilinx ISE.

### Power

Input:
- 12 V DC only, through two high-current Mini-Fit Jr / EPS-style connectors.

Required rails include at minimum:
- FPGA VCCINT ~1.0 V (sized from Xilinx power estimate with margin);
- FPGA auxiliary/configuration rails per Virtex-5 requirements;
- 1.8 V DDR2 VDD/VDDQ and compatible FPGA VCCO banks;
- DDR2 VTT ~0.9 V, source/sink capable;
- DDR2 VREF ~0.9 V, low-noise;
- PHY rails as required by selected Ethernet PHY.

Requirements:
- power sequencing must comply with Virtex-5 and DDR2 requirements;
- each major rail gets current/voltage test points;
- include power-good/reset supervisor;
- reserve current-monitor footprints on 12 V and DDR rail;
- thermal design must support heatsinks on all four FPGAs and forced airflow across DIMMs.

### PCB/mechanical target

- Maximum assembled board outline: 480 mm × 240 mm so it fits within PCBWay's published 250 mm × 500 mm assembly envelope.
- Target stack-up: 12–16 layers; fabricator/layout engineer should select final layer count from BGA escape, DDR2 SI, power integrity, and cost.
- Controlled impedance on clocks, DQS, Ethernet, and any differential inter-FPGA links.
- DDR2 routing must use byte-lane grouping, length matching, appropriate termination, low-stub topology, and SI verification for two RDIMMs/channel.
- Use a placement that keeps every FPGA physically centered between its four DIMM sockets (two channels × two sockets).
- Provide heatsink mounting holes around each FPGA.
- Provide four board mounting holes plus additional support near the DIMM socket field.
- Use press-fit/THT/SMT DIMM socket technology compatible with the chosen assembly process. Assembly house must confirm tooling for the selected 240-pin sockets.

## FPGA pin-planning gate — mandatory before fabrication

No Gerber is approved until the following is demonstrated in Xilinx ISE for the exact FPGA package and speed grade:

1. Two independent DDR2 controllers instantiate per FPGA.
2. All DQ/DQS/DM pins land in legal memory-capable I/O groups.
3. Address/command/clock pins fit legal banks.
4. Required PLL/clock resources are available.
5. Post-route timing passes at the chosen DDR2 target frequency.
6. VCCO/VREF bank assignments are electrically consistent.
7. The remaining pins are sufficient for Ethernet, JTAG, reset, status, and inter-FPGA fabric.

If two channels per FPGA do not pass these gates with the chosen controller, Rev A must be reduced before PCB sign-off rather than routing an unimplementable board.

## Bring-up plan

1. Assemble 5 prototype PCBAs; do not order 30 production boards initially.
2. First power-up without DIMMs: verify sequencing, rails, clocks, JTAG, FPGA configuration, and thermals.
3. Populate one RDIMM per channel and run DDR calibration/memory tests on all 8 channels independently and simultaneously.
4. Populate the second DIMM per channel and repeat signal-integrity/stress tests.
5. Load a Transit bitplane kernel and measure sustained DDR traffic and Gweights/s per FPGA/channel.
6. Run all four FPGAs concurrently and verify result reduction over the local inter-FPGA fabric / Ethernet.
7. Only after measured scale-out evidence, quote 30-board production.

## Manufacturing/assembly requirements

Preferred one-stop vendor for Rev A: PCBWay, because its published services include schematic/PCB design, multilayer fabrication, SMT + THT assembly, BGA assembly with X-ray, consigned/kitted parts, and functional testing.

Ask for two quotes:
- NRE/design + 5 fully assembled Rev A prototypes;
- 30-unit production price after prototype approval.

For obsolete/reclaimed Virtex-5 FPGAs, use a consigned/partial-turnkey flow unless PCBWay can source authenticated parts at an acceptable price. Require X-ray on every FPGA BGA in the prototype lot.

## Deliverables required from layout/design house

- editable schematic source;
- editable PCB layout source;
- complete BOM with manufacturer part numbers;
- FPGA pin-assignment spreadsheet;
- stack-up and impedance table;
- DDR2 SI constraints/report;
- Gerbers/ODB++;
- drill files;
- pick-and-place/centroid;
- assembly drawings;
- stencil files;
- test-point map;
- DFM/DFA report;
- X-ray images for FPGA BGAs;
- first-article power-up report if functional testing is purchased.

## Status

This file is an engineering design/handoff specification, not a claim that Gerbers already exist. The next gate is vendor schematic/layout review plus FPGA DDR2 pin-fit/timing validation before fabrication.