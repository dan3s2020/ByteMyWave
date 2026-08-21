# 26 — GIDEL ProceV Price Update — 2026-08-21

This note records a material price improvement for one of the strongest currently known Transit surplus-tile candidates.

## Current listing change

A current eBay listing shows:

- `4 x GIDEL ProceV Rev 3 (PROCE VD8-BXSM) PCIe FPGA Accelerator Card Daughter 16GB`
- price: **US$199.99** for the four-card lot
- shipping: **US$5.93**
- effective card price before shipping: **~US$50/card**
- seller: liquigator

A separate single-card listing shows:

- `GIDEL ProceV Rev 2 (PROCE VD8-BM) PCIe FPGA Accelerator Card Daughter Board 16GB`
- price: **US$59.99**
- shipping: **US$7.04**
- seller: liquigator

The previous observed four-card price was approximately US$259.99 (~US$65/card), so the new four-card lot is a material improvement of roughly 23% in total purchase price and ~US$15/card.

## Hardware configuration already established

The ProceV family is unusually well aligned with Transit:

- Intel/Altera Stratix V GS FPGA (`5SGSD8` family on documented ProceV variants);
- PCIe Gen3 x8 host interface;
- **two independent DDR3 ECC SO-DIMM banks** per card;
- documented configurations of **2 x 8 GB = 16 GB DDR3 per card**;
- each bank is **72-bit (64 data + 8 ECC)**;
- DDR3-1600;
- documented sustained bandwidth of approximately **9.6 GB/s per bank**, **19.2 GB/s/card** using Gidel's controller assumptions;
- Quartus/HDL programmable FPGA path; historical Gidel development/software support.

Conceptually:

```text
R920 / PCIe switch
        |
     PCIe Gen3 x8
        |
   GIDEL ProceV
      Stratix V
      /       \
 DDR3 ch B   DDR3 ch C
  8 GB        8 GB
 72-bit      72-bit
```

## Transit economics at the new price

For the four-card lot:

```text
4 cards
= ~US$199.99
= 64 GB local DDR3
= 8 independent DDR3 channels
= ~76.8 GB/s documented sustained aggregate DDR bandwidth
= 4 programmable Stratix V compute engines
```

That makes one four-card group a particularly clean implementation of the current logical `8 DDR3 channels/tile` concept, although it exposes four PCIe endpoints unless grouped behind a local switch/controller.

Using the current ByteMyWave K3 sizing lower bound of ~52 GB active-weight payload/token:

```text
76.8 GB/s / 52 GB/token ~= 1.48 weight-path tok/s equivalent
```

This is an analytical weight-path number, not measured end-to-end Kimi K3 throughput.

## Why this update matters

At ~US$50/card, ProceV now gives:

- 16 GB DDR3 local capacity;
- 2 independent wide DDR3 channels;
- one large Stratix V FPGA;
- PCIe Gen3 x8;
- much better capacity/channel density than Storey Peak;
- fewer physical cards and endpoints for the same local DDR capacity.

Storey Peak still wins on raw US$/channel when found around US$12, but ProceV is becoming much more competitive once endpoint count, local capacity, fanout complexity and rack density are included.

## Current decision

> **Material price improvement — buy/test the four-card lot tier is now stronger than before.**

Before bulk scaling, still verify on the exact purchased revision:

1. both DDR3 banks enumerate/train independently;
2. installed DIMMs are actually 2 x 8 GB;
3. sustained simultaneous bandwidth of both banks;
4. custom bitstream/Quartus flow and flash recovery;
5. PCIe enumeration and DMA in the R920;
6. whether the endpoint can operate through the intended reduced-lane / fanout topology.
