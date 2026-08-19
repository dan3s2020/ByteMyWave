# 22 — Broadcom Stingray PS225-H16 as a Transit Candidate — 2026-08-19

## Why this is a material new candidate

A current surplus listing for the Broadcom Stingray PS225-H16 (`BCM958802A8046C`) is active at US$39 each, with an automatic bulk discount to **US$33.15 each for 4+**, and more than 10 units available. The listing identifies the board as a PCIe x8 SmartNIC with **16 GB DDR4 DRAM**.

This candidate is not DDR3, so it does not satisfy the original DDR3-only preference. It is nevertheless materially relevant to Transit because it combines, on one cheap enterprise PCIe card:

- host-facing PCIe endpoint;
- **two independent 64-bit + ECC local DDR4 channels**;
- **16 GB local memory**;
- **8 programmable ARMv8 Cortex-A72 cores at 3.0 GHz**;
- Linux-capable software environment;
- onboard UART and documented host access path;
- hardware accelerators in the same SoC.

That makes it a much lower-risk software target than several proprietary FPGA/NPU surplus boards whose custom firmware path is still unknown.

## Exact board / part number

Observed board:

```text
Broadcom Stingray PS225-H16
BCM958802A8046C
BCM958802 / CD16M identifiers also used by seller
```

Current marketplace snapshot:

```text
eBay item 356739081687
US$39.00 each
US$33.15 each for quantity >= 4
>10 available
7 sold at observation time
Used; seller warranty
```

This price is a marketplace snapshot and must be rechecked before procurement.

## Memory subsystem

Broadcom's PS225 documentation specifies:

```text
DDR4 channel 0: 64-bit data + ECC
DDR4 channel 1: 64-bit data + ECC
DDR4 rate:       2400 MT/s
PS225-H16:       16 GB onboard DDR4
```

The official block diagram shows the two DDR channels directly attached to the BCM58802H SoC.

Theoretical payload ceiling before efficiency losses is therefore approximately:

```text
19.2 GB/s per 64-bit DDR4-2400 channel
38.4 GB/s for two channels
```

This is a theoretical interface calculation, not a measured Transit kernel result.

Broadcom has separately described Stingray deployments as providing very high aggregate memory bandwidth. Do not substitute that marketing/system number for a measured PS225-H16 sequential-read benchmark until the card is tested directly.

## Programmable controller / compute path

The PS225 uses the Broadcom BCM58802H Stingray SoC. Board documentation specifies:

```text
8 x ARMv8 Cortex-A72 cores
3.0 GHz
64-bit
16 MB aggregate L2+L3 cache
```

The board also contains fixed-function/configurable accelerators for networking, crypto and RAID, but the important Transit path is the general-purpose ARM subsystem.

Broadcom documentation explicitly describes:

- access from an x86 host;
- SSH connection to the PS225;
- software upgrade;
- setting up a Linux distribution root filesystem;
- running Ubuntu on the PS225.

Thus, unlike a mystery enterprise ASIC, the primary compute path is conventional programmable ARM64 software.

## PCIe interface

Documented host interface:

```text
PCI Express Gen3 x8
SR-IOV support
Function Level Reset
```

For Transit, Gen3 x8 is more than adequate for the intended command/activation/result traffic if weights remain resident locally.

### Mining-riser caveat

No evidence has yet been found that this exact PS225-H16 reliably down-trains and functions through the intended mining-style PCIe x1 riser/fanout topology.

This must be tested physically before bulk purchase.

## Transit topology

Potential use:

```text
Dell R920
   |
PCIe switch / fanout
   |
   v
PS225-H16
   |
   +-- ARM64 Transit worker
   |
   +-- DDR4 channel 0
   |      local resident weights
   |
   +-- DDR4 channel 1
          local resident weights
```

The host would send commands/activations. The ARM worker would operate on resident local shards and return reduced results.

This is a near-memory software-compute tile rather than an FPGA bitplane tile.

## Rough bandwidth economics

At the observed 4+ price:

```text
US$33.15 / card
16 GB local memory / card
2 independent memory channels / card
```

Cost metrics:

```text
~US$2.07 per GB of local memory
~US$16.58 per independent memory channel
```

If both DDR4-2400 x64 channels approached their nominal interface rate, one card would have a theoretical ~38.4 GB/s local payload ceiling.

Using the current Transit lower-bound model of ~52 GB active 4-bit-equivalent weights per token:

```text
38.4 / 52 ~= 0.74 weight-path tok/s equivalent per card
```

This is only a theoretical memory-interface roofline. It is NOT measured card bandwidth and NOT end-to-end Kimi K3 generation speed.

## Why it could be better than many DDR3 candidates

Advantages:

1. **16 GB per endpoint** is much denser than 4 GB Storey Peak.
2. Two local memory channels reduce endpoint count relative to one-channel micro-tiles.
3. Eight high-frequency ARM64 cores give a normal C/C++/Linux programming model.
4. Broadcom documents Linux/Ubuntu use on the card.
5. PCIe host access and UART are documented.
6. Current used price is extremely low for 16 GB local RAM + independent compute.
7. No FPGA bitstream reverse engineering is required for the first software prototype.

## Why it may fail as the final Transit tile

Important limitations:

1. **DDR4, not DDR3.** It deviates from the original low-cost DDR3 sourcing strategy.
2. ARM compute may not consume both DDR channels efficiently for low-bit K3 MVM workloads.
3. The fixed SoC memory hierarchy gives less freedom than custom FPGA near-memory datapaths.
4. Secure boot / firmware constraints must be understood before assuming arbitrary low-level firmware control.
5. Mining-riser x1 compatibility is not demonstrated.
6. PCIe fanout at high endpoint count remains a system-level issue.
7. Actual sustained DRAM bandwidth accessible to user-space ARM code has not yet been measured.

## Recommended experiment

This candidate is cheap enough to justify a one-card test immediately, with four cards justified only after basic software access is confirmed.

Test sequence:

```text
1. Buy one BCM958802A8046C / PS225-H16.
2. Enumerate it in the R920 under Linux.
3. Connect UART and preserve all factory boot/flash information.
4. Boot/access the documented ARM Linux environment.
5. Run a memory-bandwidth benchmark on both DDR channels.
6. Verify whether NUMA/interleaving controls expose independent channel behavior.
7. Implement a simple INT4/INT8 or MXFP-oriented local MVM microkernel.
8. Measure bytes/s, operations/s and power.
9. Test PCIe through the intended powered x1 mining riser.
10. Only then test 4 cards behind one fanout domain.
```

## Current decision

> **HIGH-PRIORITY BUY-ONE / SOFTWARE TILE TEST.**

At ~US$33.15 bulk, PS225-H16 is one of the cheapest currently observed enterprise boards combining 16 GB of local memory, two real local memory channels, a documented PCIe endpoint and a fully programmable general-purpose compute subsystem.

It does not replace the FPGA Transit path. It is a strong alternative implementation path that may dramatically reduce bring-up risk and endpoint count if measured ARM-side memory bandwidth is high enough.
