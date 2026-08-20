# 24 — Tsingetech PCIE720 — Five-DDR3-Channel Transit Candidate — 2026-08-20

## Why this is materially new

Tsingetech PCIE720 is unusually close to the desired Transit physical tile because one PCIe add-in card already integrates five independently programmable Xilinx Kintex-7 FPGA processing nodes, and each node owns one 64-bit DDR3 SDRAM group.

That means the board exposes five independent local DDR3 paths behind one host PCIe endpoint/card rather than requiring five separate Storey Peak cards and five separate host endpoints.

## Exact model

- Manufacturer: Beijing Tsingetech / 青翼科技
- Model: `PCIE720`
- Form factor: full-height PCIe accelerator card

## Programmable compute topology

Current manufacturer documentation specifies:

- 1 × main FPGA: `XC7K420T-2FFG901I`
- 4 × subordinate FPGAs: `XC7K325T-2FFG676I`
- subordinate nodes may optionally use `XC7K160T-2FFG676I`
- main FPGA links to each subordinate FPGA with GTX serial links
- each processing node has its own independent configuration flash

Older manufacturer material describes the main FPGA as XC7K410T/XC7K325T compatible; the current product page lists XC7K420T. Treat exact board revision as something to confirm from photos/part markings before purchase.

## DDR3 topology

The manufacturer states that **each processing node has one 64-bit DDR3 SDRAM group**.

Therefore:

```text
PCIE720
├── FPGA 0 (main)  → 1 × 64-bit DDR3 channel
├── FPGA 1         → 1 × 64-bit DDR3 channel
├── FPGA 2         → 1 × 64-bit DDR3 channel
├── FPGA 3         → 1 × 64-bit DDR3 channel
└── FPGA 4         → 1 × 64-bit DDR3 channel

TOTAL = 5 independent 64-bit DDR3 paths
```

The current manufacturer page specifies DDR3 at a 500 MHz working clock and a maximum capacity of 4 GB per processing node. Older manufacturer material explicitly calls it DDR3-1600 for both main and subordinate nodes. The exact populated capacity and actual memory clock of any used board must therefore be verified physically.

If a fully populated board actually has 4 GB per node, the theoretical maximum local capacity is 20 GB per card. Do not assume 20 GB unless the seller/revision confirms population.

## PCIe

The current manufacturer page specifies:

- PCI Express Gen2 x8 host interface
- approximately 3.0–3.2 GB/s DMA up/down performance

Older manufacturer documentation is especially important for Transit because it explicitly states that the host PCIe interface can be configured in **x1, x4, or x8 modes**.

This is the first strong multi-DDR3-channel FPGA candidate found in this search for which manufacturer material explicitly names x1 operation, making it unusually relevant to the R920 + mining-riser/fanout plan.

The x1 claim is an architectural/support claim; physical enumeration through the exact mining riser/switch tree still needs a real test.

## Documentation availability

Public/current:

- manufacturer product page with block-level specifications
- manufacturer downloadable PCIE720 datasheet
- DDR3 topology, PCIe mode, FPGA models, power and physical dimensions documented
- optional BSP advertised by manufacturer
- DDR3 test program advertised
- PCIe/fiber demo software and Windows drivers/API advertised

Not found publicly during this search:

- full schematic
- full XDC pin constraints
- open-source LiteX/LitePCIe target
- public source for the vendor BSP

Therefore documentation is stronger than an unidentified surplus appliance board but weaker than YPCB-00338/Celestica boards with public community constraints and open-source flows.

## Power and physical notes

Manufacturer lists:

- board size approximately 106.65 × 167.65 mm
- supply maximum approximately 1.7 A @ +12 V (~20.4 W stated board supply figure)
- forced-air cooling

The stated 20 W-class figure is surprisingly low for five large Kintex-7 FPGAs and must be verified on a real board under Transit workload before using it for rack power sizing.

## Current availability / price

As of 2026-08-20 the PCIE720 remains present in Tsingetech's current data-center product catalog and its datasheet remains downloadable from the manufacturer support site.

No reliable public Alibaba/eBay/Taobao used listing with a verifiable current unit price was found in this run. The manufacturer site is quote-based and does not publish a unit price.

Therefore this candidate is **not yet a bulk-buy recommendation** despite its extremely attractive topology. We need a current quote or surplus listing before comparing $/channel against Storey Peak, YPCB and GIDEL ProceV.

## Why it matters for Transit

One board effectively supplies five Storey-Peak-like local memory paths while consuming one physical add-in-card position:

```text
R920 / PCIe switch
        |
        | x1 / x4 / x8
        v
     PCIE720
        |
        +-- K7 + DDR3 ch0
        +-- K7 + DDR3 ch1
        +-- K7 + DDR3 ch2
        +-- K7 + DDR3 ch3
        +-- K7 + DDR3 ch4
```

For the original Transit target of ~304 independent channels, this topology would require roughly:

```text
ceil(304 / 5) = 61 PCIE720 cards
```

rather than ~304 one-channel Storey Peak cards.

That reduction in PCIe endpoint count, risers, mechanical positions and cabling is architecturally significant even before price is known.

If each channel truly runs DDR3-1600 x64, nominal payload would be 12.8 GB/s/channel and 64 GB/s/card. That is only a theoretical interface-rate calculation, not a measured PCIE720 sustained bandwidth result. The manufacturer does not publish a sustained five-channel aggregate memory benchmark in the sources found.

## Transit verdict

**VERY STRONG TOPOLOGY / PRICE REQUIRED BEFORE BUY.**

PCIE720 is one of the closest off-the-shelf boards found to the missing Transit tile:

- one PCIe card
- manufacturer-documented x1/x4/x8 host mode
- five programmable Kintex-7 compute nodes
- five independent 64-bit DDR3 paths
- local flash per FPGA
- vendor DDR3/PCIe BSP history

The next action is not bulk procurement. It is to obtain a current quote or locate used/surplus stock, then compare price per independent DDR3 channel and test one board in x1 mode through the intended R920 fanout.
