# Hardware Candidates — Carrizo APU DDR3 UMA

## Scope

This file records hardware that can be used to falsify or validate the APU+DDR3 idea. Marketplace listings are **leads**, not procurement guarantees.

The first objective is not to find 160 units. It is to find 1-4 electrically boring, Linux-bootable nodes on which we can measure the memory and kernel path.

## Reference silicon: AMD A8-8600P

Why it is interesting:

```text
Carrizo APU
2 DDR3 channels
Radeon R6 integrated GPU
15 W default TDP
HSA / shared-memory architecture
```

Official APU-level maximum memory rate is up to DDR3-2133. The motherboard may run less.

Primary source:

https://www.amd.com/en/support/downloads/drivers.html/processors/a-series/a8-series-apu-for-laptops/6th-gen-a8-8600p-apu.html

## Candidate A — Dell Inspiron 3656 motherboard, Dell part 0W6FD

### Why it is interesting

Marketplace listings exist for bare Inspiron 3656 motherboards with an A8-8600P soldered/on-board and two DDR3L memory slots. This can be cheaper and denser than buying complete mini-PCs if power/front-panel/boot constraints are manageable.

Official Dell Inspiron 3656 manual/support surface:

https://www.dell.com/support/manuals/en-us/inspiron-3656-desktop/inspiron-3656-desktop

### Marketplace observation — 2026-08-17

Fresh listing searches showed examples around roughly:

```text
US$23.85 - US$26.99 / motherboard
```

for Dell 0W6FD/A8-8600P-class boards.

This is intentionally recorded as a **dated lead only**. Earlier searches found lower prices; prices and available quantity move. We must not derive a 160-node fleet budget from one listing.

### Must verify before even a 4-node purchase

```text
[ ] exact Dell part/revision
[ ] A8-8600P actually included
[ ] both memory slots functional
[ ] dual-channel mode confirmed
[ ] maximum DIMM size per slot
[ ] actual DDR clock with chosen DIMMs
[ ] Linux headless boot
[ ] amdgpu + RADV Vulkan compute
[ ] onboard Ethernet speed
[ ] PCIe slot electrical width/generation
[ ] boot without proprietary case/front-panel hardware
[ ] power connector/pinout and suitable PSU
[ ] BIOS auto-power-on after AC loss
[ ] quantity from same seller/revision if later scaling
```

### Risk

A cheap bare OEM motherboard can turn into an expensive node if it requires proprietary PSU wiring, front-panel adapters, special cooling, or a separate high-speed NIC that the board cannot feed well.

## Candidate B — HP EliteDesk 705 G2 Mini / A8-8600B

A8-8600B is a closely related Carrizo Pro part and is useful as a low-risk complete-system POC candidate.

HP support surface:

https://support.hp.com/us-en/product/details/hp-elitedesk-705-g2-desktop-mini-pc/8741011

Memory configurator/documentation sources commonly show two SODIMM sockets and a 16 GB-class platform limit for the mini configuration. Exact HP QuickSpecs/manual revision should be checked against the purchased product number.

Secondary memory reference:

https://www.kingston.com/en/memory/search/model/93756/hp-hpe-elitedesk-705-g2-mini-pc

### Advantages

- complete enclosure, PSU and cooling;
- low friction for Linux/Vulkan proof;
- easy wall-power measurement;
- good first-node benchmark vehicle.

### Disadvantages

- RAM capacity can be too small for an efficient final fleet layout;
- mini-PC networking/expansion may limit high-speed collective experiments;
- cost/node can be higher than bare motherboard lots.

Therefore this is a strong **POC machine**, not automatically the fleet winner.

## Capacity target per node

At 160 nodes, simple average checkpoint ownership is:

```text
1.56 TB / 160 ~= 9.75 GB/node
```

A final two-slot platform should ideally exceed that comfortably, because we also need:

- uneven tensor placement;
- metadata/scales;
- temporary activation/result buffers;
- local state where applicable;
- expert replication;
- spare/recovery capacity.

A practical fleet candidate should preferably allow at least 16 GB usable/node; 32 GB/node gives much more placement freedom. This is a design preference, not a statement that every candidate above supports 32 GB.

## Memory module rules

Two DIMMs do not automatically mean two independent active channels. For every candidate:

1. install a matched pair;
2. verify memory-controller channel mode in firmware/Linux tooling;
3. measure bandwidth, not just SPD frequency;
4. verify the GPU-visible path separately from CPU STREAM bandwidth.

The metric that matters for this track is:

```text
useful bytes of resident K3-format weights consumed by Radeon compute / second
```

not DIMM count.

## NIC / interconnect selection criteria

Do not select the fleet NIC solely by link rate.

Required board facts:

```text
PCIe slot physical size
PCIe negotiated electrical width
PCIe generation
onboard NIC
IOMMU behavior
interrupt/MSI support
Linux driver support
```

Required network measurements:

```text
7 KiB payload latency
14 KiB payload latency
28 KiB payload latency
small-message p50/p95/p99
2/4/8-node collective latency
CPU utilization during collectives
simultaneous throughput / switch bisection
```

1 GbE should be treated as a basic control/boot network until measurements prove it useful for the target layer cadence.

## Power criteria

Never scale from APU TDP alone.

For each candidate measure at the wall:

```text
idle
GPU-visible DDR streaming
MXFP4-shaped kernel
network collective load
combined inference-shaped load
```

Record:

```text
watts/node
GB/s useful weight traffic
watts per useful GB/s
estimated watts per active channel
```

This will let the project compare APU tiles against FPGA, R920/CPU and other tracks fairly.

## What constitutes a fleet-quality candidate

A board family does not graduate from POC to fleet candidate until all are known:

```text
[ ] >=2 true independent DDR channels
[ ] enough RAM capacity for planned placement
[ ] sustained GPU-visible memory bandwidth passes Gate 1
[ ] K3-shaped kernel passes Gate 2
[ ] stable Linux/RADV stack
[ ] viable low-latency NIC path
[ ] headless unattended boot/restart
[ ] reproducible PSU/cooling solution
[ ] measured wall power
[ ] seller/industrial source with meaningful quantity
[ ] delivered cost including shipping/VAT/adapters/NIC/RAM
[ ] spare-unit strategy
```

## Current buying stance

Buy enough hardware to answer the measurements, not enough hardware to assume the answers.

Recommended order:

```text
1 node -> local memory/kernel proof
2 nodes -> communication proof
4 nodes -> first collective/scaling proof
8 nodes -> scaling decision
fleet  -> only after all gates pass
```