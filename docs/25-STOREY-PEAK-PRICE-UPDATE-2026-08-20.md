# 25 — Storey Peak Price Update — 2026-08-20

This note records a material procurement update for the Transit GPU surplus-tile search.

## Candidate

**Microsoft / HP Azure Storey Peak `X930613-001`**

Current observed active listings on 2026-08-20:

- eBay seller `g-electronic`: **US$11.99**, used, free shipping, 58 sold in the indexed listing.
- eBay seller `JB Electronic`: **US$13.99**, used, free shipping, more than 10 available, 80 sold.

The US$11.99 listing is the most important current procurement update. It does not beat the historical US$9.59 bulk price previously observed, but it materially improves on the more recent active pricing around US$18–20 and appears to be directly purchasable without a bulk threshold.

## Transit-relevant hardware

Storey Peak remains the best-documented ultra-cheap FPGA+DDR3 micro-tile in the current search:

```text
PCIe endpoint
   |
Stratix V GS FPGA
   |
1 local DDR3L channel / bank
   |
~4 GB usable local DDR3
```

Public reverse-engineering and measurements previously recorded in this branch include:

- FPGA: Intel/Altera Stratix V GS family;
- approximately 4 GB usable local DDR3L on a 72-bit physical bus (64 data + ECC);
- two PCIe Gen3 x8 interfaces exposed by the board design;
- custom PCIe DMA demonstrated publicly;
- DDR3 operation at 1600 MT/s;
- measured DDR read throughput around **9.662 GB/s** on a real board;
- public Quartus/QSF/reference-design/JTAG work sufficient to make custom Transit experimentation realistic.

## Current economics

At US$11.99 per board:

```text
10 boards  = US$119.90
40 boards  = US$479.60
304 boards = US$3,644.96
390 boards = US$4,676.10
539 boards = US$6,462.61
```

Those totals are board-only and exclude fanout, risers, power, cooling, shipping/import costs where applicable, and the still-unresolved large-scale PCIe switch fabric.

Using the measured 9.662 GB/s DDR read figure:

```text
40 boards  -> ~386.5 GB/s aggregate -> ~7.43 weight-path tok/s equivalent
304 boards -> ~2.937 TB/s aggregate -> ~56.5 weight-path tok/s equivalent
390 boards -> ~3.768 TB/s aggregate -> ~72.5 weight-path tok/s equivalent
539 boards -> ~5.208 TB/s aggregate -> ~100.15 weight-path tok/s equivalent
```

The conversion uses the current Transit Kimi K3 sizing assumption of approximately 52 GB of active 4-bit-equivalent weight payload per token. These are **analytical weight-path equivalents, not measured end-to-end Kimi K3 generation rates**.

## Why the price update matters

At US$11.99, Storey Peak becomes extremely difficult to beat on:

```text
price
+ real local DDR3
+ programmable FPGA
+ PCIe endpoint capability
+ public reverse-engineering
+ measured local-memory bandwidth
```

The board still provides only one practical DDR3 channel per physical card, so scaling to hundreds of independent channels produces an ugly physical/fanout problem. That remains the main reason to continue searching for 2/4/8-channel surplus boards even though Storey Peak currently wins the dollar-per-micro-tile comparison.

## Procurement decision

**Status: materially improved current price; strong buy/test tier.**

For immediate experimentation, buying several US$11.99 units is now more attractive than at the recent US$18–20 active price. For a 304–539 board deployment, do not bulk-buy until the R920 PCIe switch/fanout topology and x1 downstream compatibility have been demonstrated at 4, 8, 16 and then 32/40 endpoints.

## Sources observed

- eBay item `286881499825`: Microsoft Azure `X930613-001`, US$11.99, free shipping.
- eBay item `405123019416`: Microsoft HP Azure `X930613-001`, US$13.99, >10 available.

Marketplace state changes quickly; re-check price and stock immediately before purchase.
