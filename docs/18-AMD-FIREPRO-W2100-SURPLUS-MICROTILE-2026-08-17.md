# 18 — AMD FirePro W2100 as an ultra-cheap Transit micro-tile — 2026-08-17

This note records a materially new surplus candidate found during the ongoing Transit memory-compute tile hunt.

## Candidate

**AMD FirePro W2100**, including Dell/HP OEM variants such as Dell **02P8XT** and HP **762896-002 / 763264-001**.

The W2100 is not an FPGA. It is a small GCN GPU with a fixed local DDR3 memory subsystem, but it implements the core Transit primitive economically:

```text
PCIe endpoint
   |
GCN programmable compute
   |
2 GB local DDR3
   |
resident weight shard
```

## Verified hardware properties

AMD's current legacy support/specification page reports:

- GPU architecture: Graphics Core Next (GCN)
- 320 stream processors
- 400 GFLOP/s peak FP32 vector performance
- 2 GB dedicated DDR3
- 28.8 GB/s peak local memory bandwidth
- PCIe 3.0 add-in card
- OpenCL support
- 26 W total board power
- no external power connector
- half-height, single-slot form factor

An archived AMD W2100 datasheet additionally specifies a **128-bit DDR3 memory interface**. Lenovo's OEM technical page identifies the GPU as Oland and likewise specifies a 128-bit memory interface and up to eight memory devices.

Sources:

- AMD legacy product/support page: https://www.amd.com/en/support/downloads/drivers.html/graphics/firepro/firepro-wx100-series/firepro-w2100.html
- archived AMD datasheet mirror: https://pdf.directindustry.com/pdf/amd/w2100/102295-617964.html
- Lenovo OEM technical page: https://support.lenovo.com/gb/en/accessories/acc500041-amd-firepro-w2100-2gb-ddr3-two-display-port-graphics-card-by-thinkstation-4x60h45061

## Current surplus pricing observed on 2026-08-17

The most interesting current listing is:

- **Lot of 12 Dell AMD FirePro W2100, P/N 02P8XT**
- 12 cards for **US$85.49 or Best Offer**
- effective board cost: **US$7.12/card** before shipping/tax
- condition: used, pulled from working environment
- eBay item 167844045682
- listing: https://www.ebay.com/itm/167844045682

Other current/observed supply confirms that the low price is not unique to one card:

- lot of 29 W2100 cards: **US$250**, or **US$8.62/card**
- lot of 4: **US$38.99**, or **US$9.75/card**
- lot of 3: **US$29.26**, or **US$9.75/card**
- lot of 2 HP 762896-002: **US$29.99**, with an observed 10% coupon reducing the pair to **US$26.99**; listing showed 18 lots available
- another lot of 2: **US$12.50**, or **US$6.25/card**, though supply/ship-to-region must be checked at purchase time

Marketplace prices are snapshots and must be rechecked before purchase.

## Why this is interesting for Transit

At the 12-pack price:

```text
12 cards
24 GB aggregate local DDR3 capacity
12 independent GPU-local memory subsystems
12 programmable GCN engines
12 x 28.8 GB/s = 345.6 GB/s theoretical aggregate local DDR3 bandwidth
12 x 26 W = 312 W board-power ceiling
US$85.49 total card cost
```

The raw theoretical bandwidth-per-dollar is therefore unusually high for a documented programmable PCIe endpoint with real DDR3 attached.

One board provides only 2 GB of capacity, but it is a complete self-contained memory-compute island and needs no auxiliary PCIe power connector. This makes it especially attractive for cheap multi-endpoint experiments and for determining whether commodity legacy GPUs can implement the same stationary-weight scheduling model as the FPGA Transit path.

## Important distinction versus Storey Peak

Storey Peak gives full FPGA control over the DDR/compute datapath and has roughly 4 GB local memory per physical board, but only one DDR3 channel per board and a measured read rate around the 9.662 GB/s figure currently used in the Transit research notes.

W2100 gives a fixed GPU memory controller rather than an FPGA-controlled PHY, but AMD specifies **28.8 GB/s peak local DDR3 bandwidth** and 320 programmable GCN stream processors. It is therefore potentially much faster as a commodity local-memory engine if a Transit-compatible integer/bitwise kernel can sustain useful utilization.

Do not compare the 28.8 GB/s W2100 peak directly to Storey Peak's measured 9.662 GB/s as if they were equally demonstrated values. W2100 needs a real sustained-read/custom-kernel benchmark before using it in a measured Transit throughput claim.

## Software/documentation advantage

Unlike several abandoned enterprise ASIC/NPU candidates, the W2100 has a standard GPU programming path. AMD still hosts legacy Windows, Windows Server and Linux driver packages for the W2100; historical packages explicitly include an OpenCL runtime. This substantially lowers bring-up risk compared with NFP-3240 or undocumented appliance accelerators.

What is not available is FPGA-style control over the DDR PHY/controller or open board schematics. The local memory system is fixed by the GPU.

## Risks / blockers

1. **Only 2 GB per card.** Capacity density is poor compared with GRID K1, Cavium 8 GB cards, or multi-DIMM FPGA boards.
2. **Legacy GCN software stack.** The board is programable through GPU APIs, but modern ROCm support should not be assumed; use the documented legacy OpenCL/driver path first.
3. **No custom DDR controller.** Transit must map its arithmetic onto the GCN instruction/memory model instead of implementing a bespoke FPGA bitplane engine.
4. **Peak bandwidth is not sustained Transit bandwidth.** 28.8 GB/s is the vendor peak; measure STREAM-like reads plus the actual resident-weight kernel.
5. **PCIe x1 mining-riser operation is not yet demonstrated for this exact board in the Transit lab.** The large weight stream stays local, so x1 could be sufficient for command/activation/result traffic, but enumeration and stability must be physically tested.
6. **Full-model capacity would require many endpoints.** The low 2 GB/card capacity makes a complete K3 deployment physically large even if bandwidth is excellent.

## Recommended experiment

Buy one cheap pair or one 12-pack only after confirming shipping economics, then test:

```text
1. enumerate W2100 normally in R920/Linux
2. enumerate through powered PCIe x1 mining riser
3. install a known-working legacy AMD OpenCL runtime
4. measure sustained sequential DDR3 read bandwidth
5. implement a small integer/bitwise resident-weight dot/MVM kernel
6. compare result exactly against host reference
7. measure useful GB/s while computing, not a graphics benchmark
8. repeat with 4+ cards behind the intended fanout
9. measure PCIe activation/result traffic and scheduler overhead
```

## Current verdict

> **Genuinely promising new ultra-cheap non-FPGA micro-tile; BUY/TEST tier, not bulk-deployment tier yet.**

The W2100 is interesting because the current surplus price can be about **US$7–10 per independent 2 GB DDR3 + programmable-compute PCIe island**, with an AMD-rated 28.8 GB/s local-memory peak and only 26 W board power. It does not replace the FPGA path, but it is cheap enough that a real Transit kernel benchmark is justified immediately.
