# Future hardware target — Dell PowerEdge R7525

Date: 2026-08-15

Status: **future-budget option only; not the current TensorWave baseline**

This note preserves the R7525 conclusions from the Phase-6 hardware discussion so they can be revisited when budget changes. The current R920 path remains the active low-cost platform.

## Why R7525 is interesting for TensorWave

The attraction is not merely DDR3 -> DDR4. It is the combination of:

```text
2-socket AMD EPYC
8 memory channels / socket
DDR4-3200
~204.8 GB/s published memory bandwidth / socket on EPYC 7502
PCIe 4.0 x128 / CPU
GPU-capable full-length x16 riser options
simpler 2-domain NUMA topology
```

Reference EPYC 7502 profile:

```text
32 cores / 64 threads per CPU
2P capable
8 DDR4 channels
up to DDR4-3200
204.8 GB/s published memory bandwidth per socket
PCIe 4.0 x128
```

A 2 x EPYC 7502 machine therefore gives a published aggregate memory-bandwidth envelope of about 409.6 GB/s before topology, efficiency, NUMA and workload effects.

Official sources:

- AMD EPYC 7502: https://www.amd.com/en/support/downloads/drivers.html/processors/epyc/epyc-7002-series/amd-epyc-7502.html
- Dell R7525 memory guidelines: https://www.dell.com/support/manuals/en-us/poweredge-r7525/r7525_ism_pub/system-memory-guidelines
- Dell R7525 expansion/riser specification: https://www.dell.com/support/manuals/en-us/poweredge-r7525/r7525_ts_pub/expansion-card-riser-specifications
- Dell R7525 GPU kit: https://www.dell.com/support/manuals/en-us/poweredge-r7525/r7525_ism_pub/gpu-kit

## Preferred future 1 TiB layout

For a 2-socket EPYC R7525, the clean TensorWave layout is one DIMM per memory channel:

```text
CPU0: 8 x 64 GiB = 512 GiB
CPU1: 8 x 64 GiB = 512 GiB
TOTAL: 1 TiB
```

This populates all eight channels per socket and avoids the deliberately unbalanced NUMA layouts TensorWave is trying to avoid.

## K2.5 CPU-expert implications

The Phase-6 K2.5 routed-expert workload is unchanged by the host platform. With two CPU sockets, the per-socket requirements are:

```text
5 tok/s target:
~33.030 GB/s selected Q4 reads/socket
~105.696 GFLOP/s logical/socket
~52.848 Gweights/s/socket

10 tok/s target:
~66.060 GB/s selected Q4 reads/socket
~211.393 GFLOP/s logical/socket
~105.696 Gweights/s/socket
```

Against EPYC 7502's published 204.8 GB/s/socket memory bandwidth, the pure memory-side fractions are approximately:

```text
5 tok/s  -> 16.1%
10 tok/s -> 32.3%
```

This is only a bandwidth envelope. The actual low-bit expert kernel must still be benchmarked; no tok/s claim is made from the published memory number alone.

## Why this is a real architectural upgrade over R920

R920 Phase-6 baseline:

```text
4 x E7-4890 v2
4 memory channels/socket
85 GB/s published memory BW/socket
AVX-era CPU
PCIe 3.0
```

R7525 + EPYC 7502 target:

```text
2 x EPYC 7502
8 memory channels/socket
204.8 GB/s published memory BW/socket
much newer CPU core/vector implementation
PCIe 4.0
```

For TensorWave this attacks both important paths:

1. NUMA-local CPU routed-expert execution.
2. GPU host feed / model sharding through PCIe Gen4 rather than Gen3.

It also reduces NUMA complexity from four CPU domains to two.

## GPU note

Dell documents up to eight PCIe Gen4 expansion cards depending on riser configuration, with several x16 GPU-capable slots. Dell also explicitly warns against consumer-grade GPUs in enterprise servers, so an RTX 3060 remains an unsupported/custom integration even though the electrical interface is compatible.

The R7525 GPU kit can provide supported enterprise-GPU cooling/power hardware for specific cards. Any future consumer-GPU use must validate physical fit, shroud/fan configuration and power cabling rather than assuming a bare server is GPU-ready.

## Budget gate — corrected for our actual target market

The first version of this note used high-markup eBay/refurbisher asking prices and therefore overstated the practical purchase cost. That was the wrong comparator for this project.

Our relevant working market anchors are:

```text
R920-class bare/low-RAM server: ~1,000 RON target
R7525-class bare/low-RAM server: ~5,000-6,000 RON target
```

These are acquisition targets for used/off-lease hardware, not permanent market facts. RAM, CPUs, GPU kit/riser, shipping and power hardware must be priced separately unless explicitly included.

The correct purchase rule is therefore not "R7525 only if it falls to 2x-3x R920". The technical R7525 upgrade may justify roughly a 5x-6x chassis/platform premium if the total build still makes sense, because it changes CPU ISA/core performance, memory channels/bandwidth and PCIe generation at the same time.

Before revisiting the purchase, search the actual used/off-lease market first, especially Romania and low-markup EU surplus channels, for:

```text
R7525 / EPYC 7002 or 7003
2 CPUs installed or cheap compatible Dell-locked CPUs available
GPU-capable risers/shroud/power kit
1 TiB DDR4 included or RAM priced separately
1400 W PSU configuration where relevant
rails/iDRAC included
shipping/VAT/import to Romania
```

Also compare equivalent dual-EPYC platforms rather than paying a Dell premium:

```text
HPE ProLiant DL385 Gen10 Plus
Lenovo ThinkSystem SR665
Supermicro dual-SP3/H12 platforms
```

The objective is the EPYC memory/PCIe architecture, not the R7525 badge itself.

## Market-source correction — 2026-08-15

Do not use high-markup refurbisher/eBay listings as the primary fair-value signal for this project. They frequently include warranty, VAT, storage, enterprise support margin or simply unrealistic asking prices.

Local evidence also shows how cheap the underlying EPYC ecosystem can be: a Dell-locked EPYC 7402 was listed on OLX Romania for only a few hundred RON per CPU. This reinforces the need to price chassis, CPU and RAM separately rather than infer platform value from turnkey refurbisher listings.

When market pricing is discussed again, report at least two layers separately:

```text
1. bare/low-RAM server acquisition price
2. completed TensorWave build price with target RAM + CPUs + GPU integration
```

## Decision

```text
NOW:
R920 remains the low-cost experimental platform.

LATER:
R7525 remains the preferred architectural upgrade target if a bare/low-RAM unit appears around the ~5,000-6,000 RON working band, then price 1 TiB DDR4 and GPU integration separately.

DO NOT:
use eBay/refurbisher turnkey asking prices as the main fair-value comparator for this project.

DO NOT:
pay a large Dell badge premium if an equivalent dual-SP3 server exposes the same 8-channel DDR4 + PCIe Gen4 advantages.
```
