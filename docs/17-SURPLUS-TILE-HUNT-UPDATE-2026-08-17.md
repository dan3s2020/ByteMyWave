# 17 — Surplus Tile Hunt Update — 2026-08-17

This note records only changes that crossed the notification threshold after the consolidated Transit surplus/BOM document. Marketplace prices and stock are snapshots and must be rechecked before purchase.

## Material price improvement: Cavium CN6870C-210NV-M8-3.0-G

A materially lower listing was found at retail.era for the exact `CN6870C-210NV-M8-3.0-G` family, described as **Cavium CN6870C-210NV-M8-3.0-G 8GB DDR3 Network Interface Card**, **tested used and working**, at a displayed sale price of **$10** (down from $20 on that listing). This is lower than the previous ~$13.69 low indexed during the Transit search and far below the ~$22.77–27.99 listings previously recorded.

Important caveat: retail.era also exposes a separate exact-part product page titled `CN6870C-210NV-M8-3.0-G Cavium 2Port 10G 8MB Interface Card` at $45. The $10 result explicitly says **8GB DDR3**, while the $45 page is a different catalog entry. Do not assume unlimited quantity or combine the two entries; verify the actual $10 SKU, photos, DIMMs and checkout availability before buying a lot.

### Hardware identity and memory

Independent surplus sources confirm the board form factor and installed memory for this model:

```text
Part: CN6870C-210NV-M8-3.0-G
SoC: OCTEON II CN6870
CPU: 24 x cnMIPS64 v2 cores
Board interface: PCIe x8
Installed memory on documented units:
  2 x Innodisk 4 GB DDR3-1600 ECC ULP Mini-DIMM
  = 8 GB total
```

The Cavium CN68XX product architecture provides up to **four 72-bit ECC DDR3 interfaces** and PCIe Gen2 controllers; CN6870 is the 24-core member. However, the exact PCB routing of the two populated Mini-DIMM sockets on `CN6870C-210NV-M8-3.0-G` still must be verified before counting them as two independently usable Transit memory channels.

### Transit significance

At $10, this becomes the cheapest currently observed **8 GB local DDR3 + programmable many-core processor + PCIe card** in the Transit candidate set.

It is not automatically superior to Storey Peak for bandwidth because:

- Storey Peak has a fully reconfigurable FPGA path and measured DDR read bandwidth;
- CN6870C still lacks a Transit-specific memory-bandwidth benchmark;
- the exact boot/flash/UART/JTAG route on this surplus PCB is not yet proven;
- host PCIe endpoint behavior through an x1 mining-riser/fanout is not yet proven;
- exact mapping of the two Mini-DIMMs to CN68XX memory controllers must be measured/verified.

But the economics are strong enough to change the test priority. If the $10 unit is genuinely available with both 4 GB DIMMs, **buy-one/test-one is justified immediately**.

### Required proof sequence

```text
1. verify the $10 SKU is physically CN6870C-210NV-M8-3.0-G
2. confirm both 4 GB DDR3-1600 ECC Mini-DIMMs are included
3. enumerate the board in the R920
4. identify firmware/boot flash and console/JTAG access
5. boot custom OCTEON code or Linux/Simple Executive payload
6. benchmark sequential DDR3 read bandwidth per populated memory interface
7. determine whether the two DIMMs can be driven independently/concurrently
8. run a minimal integer/bitwise Transit kernel using resident DDR3 data
9. test PCIe x1/mining-riser enumeration
10. only then evaluate bulk procurement
```

## Other new hardware checked but not promoted

### Tilera EMP-125-00301

A current eBay listing shows a Tilera `EMP-125-00301` PCIe accelerator with **16 GB memory** at **$228**, with three available. Other active listings are $210–399 and a historical unit sold for $109.99. The board is interesting because TILE-Gx is a programmable many-core architecture and the card combines large local memory with PCIe and high-speed I/O.

It is **not promoted above the current Transit shortlist** because current pricing is too high relative to Storey Peak/CN6870C, and exact DDR3 channel topology plus a clean modern software bring-up path for this specific EMP board were not established in this run.

### Mellanox/NVIDIA Innova-2 Flex

Innova-2 remains technically excellent: XCKU15P FPGA, 8 GB DDR4 on relevant variants, PCIe x8 through ConnectX-5, public user-image programming instructions, and a public XDMA-to-DDR4 example. Current useful 8 GB variants are around **$450**, so this is not a low-cost Transit tile despite excellent documentation.

### NetFPGA SUME

NetFPGA SUME is almost a textbook Transit research platform: Virtex-7 XC7V690T, PCIe Gen3 x8, **two 4 GB DDR3 SODIMMs with separate DDR3 A/B test projects**, and extensive open reference designs. Current used pricing around **$2,400** makes it irrelevant for the cheap modular build.

## Current decision

The only change in this run that crosses the notification threshold is:

> **CN6870C-210NV-M8-3.0-G at a displayed $10 sale price for an 8 GB DDR3 tested-used card.**

Priority: verify one physical unit before bulk purchase. The key engineering unknown is no longer whether the card contains useful compute + DDR3; it is whether we can take full low-level control of this exact surplus board and sustain independent DDR3 bandwidth suitable for Transit.