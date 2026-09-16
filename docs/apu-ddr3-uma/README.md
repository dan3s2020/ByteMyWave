# Track D — Carrizo APU + DDR3 UMA for Kimi K3

**Status:** research / conditional GO for a small proof-of-concept; **NO-GO for fleet procurement until the gates in `POC-VALIDATION-AND-PROCUREMENT-GATES.md` pass.**

**Date:** 2026-08-17

This branch documents an alternative ByteMyWave/Transit direction discovered after the CPU-memory-channel and discrete-GPU investigations:

> use a cheap AMD Carrizo APU as a two-channel DDR3 memory-compute tile, keep the model weights in system DDR3, let the integrated Radeon GPU consume those weights through the shared-memory architecture, and reduce the CPU cores to orchestration/driver/network duties rather than matrix multiplication.

This is deliberately a separate research track. It does **not** silently replace:

- PR #10 / `transit-ddr3-architecture` — FPGA/local-memory-compute tile track;
- PR #11 / `docs/kimi-k3-ddr-cluster` — cheap distributed server-memory track;
- PR #9 / `research/heterogeneous-moe-kimi-v1` — heterogeneous NUMA CPU/GPU MoE track.

## Why this track exists

The discrete-GPU route has a structural bandwidth mismatch:

```text
DDR3 channels -> CPU memory controller -> PCIe -> discrete GPU
```

Even if CPU cores do almost no arithmetic, the host-to-GPU PCIe link remains a bottleneck for streaming large weight matrices.

Carrizo changes the topology:

```text
DDR3 channel A ----\
                    memory controller ---- Radeon R6 iGPU
DDR3 channel B ----/                 \
                                      CPU cores (control/runtime)
```

AMD's Carrizo hUMA design gives CPU and GPU one shared memory address space and access to platform memory. The A8-8600P officially exposes two DDR3 channels, up to 2133 MT/s, Radeon R6 graphics, HSA, Vulkan and a 15 W default TDP.

The architectural question therefore changes from:

> How do we get hundreds of DDR3 channels through CPUs and PCIe into GPUs?

into:

> Can hundreds of cheap two-channel APU tiles each execute K3-shaped memory-bound kernels locally, and can their small activations/partial results be synchronized fast enough for one autoregressive token?

## Current verdict

### What is supported by evidence

- Carrizo provides shared CPU/GPU memory semantics (hUMA/HSA).
- A8-8600P has two DDR3 channels, Radeon R6, 15 W default TDP and memory support up to 2133 MT/s.
- LLVM still identifies Carrizo/A8-8600P as AMDGPU target `gfx801`.
- Mesa RADV supports GFX8-class GPUs through Vulkan; current Mesa documents Vulkan 1.4 for GFX8+.
- Kimi K3 is a 2.8T-parameter MoE with 104B activated parameters, 93 layers, 896 routed experts and 16 selected experts/token.
- K3's released checkpoint is about 1.56 TB.
- K3's quantization is not equivalent to “every active parameter is exactly 4 bits”; the official config explicitly excludes several component families from the MXFP4 linear-weight rule.

### What is **not** demonstrated

- Radeon R6 sustained K3-shaped weight-stream bandwidth from dual-channel DDR3.
- MXFP4 unpack/dequant + GEMV efficiency on `gfx801`.
- KDA and non-routed path performance on `gfx801`.
- an efficient 2/4/8-node collective runtime on this hardware.
- scaling from 8 nodes to ~160 nodes.
- any full-model K3 token/s number on this architecture.

## The most important numerical correction

Older ByteMyWave documents use:

```text
104B active parameters/token * 0.5 byte = 52 GB/token
```

That remains a valid **absolute four-bit lower bound / historical architecture-sizing model**, not an exact K3 active-byte count.

The official K3 config uses MXFP4 groups of 32 with `uint8` scale metadata for quantized linear weights and explicitly ignores several families such as self-attention and shared experts. Therefore the exact active weight bytes/token must be generated from the actual checkpoint tensor inventory and execution graph.

This track carries three models until that exact inventory exists:

1. **52 GB/token** — absolute all-active-weights-at-4-bit lower bound.
2. **~58 GB/token** — checkpoint-average screening model already used in PR #11 (`1.56 TB / 2.8T * 104B`).
3. **~136.6 GB/token conservative envelope** — routed experts modeled with MXFP4+one byte scale/32 weights while every remaining active parameter is pessimistically treated as BF16. This is deliberately **not called exact** because some non-expert `Linear` weights may also be quantized by the released config.

The exact tensor-level number is Gate 0 before fleet procurement.

## Proposed physical scale

A two-channel tile gives the following nominal DDR payload ceilings:

```text
DDR3-1600: 2 * 12.8  = 25.6 GB/s/tile
DDR3-2133: 2 * 17.07 = 34.14 GB/s/tile
```

A 320-channel thought experiment requires 160 such tiles:

```text
160 tiles * 2 channels = 320 independent DDR3 channels
```

Nominal aggregate interface ceilings:

```text
DDR3-1600: 4.096 TB/s
DDR3-2133: ~5.46 TB/s
```

These are interface rooflines, not measured useful K3 bandwidth.

## Why 320 channels do not automatically become one-token bandwidth

K3 selects 16 of 896 routed experts per token in each MoE layer. If whole experts are merely assigned to separate boards, only the boards owning selected experts work on that token; most boards idle.

To make a single token use a large fraction of the fleet bandwidth, active matrices must be tensor-sharded across multiple APUs, followed by reductions/collectives. That turns **network latency and synchronization at layer cadence** into the central system risk.

A pure layer pipeline solves capacity and multi-request throughput but does not add the memory bandwidth of sequential stages for one autoregressive decode token.

## Documents in this track

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — data path from prompt to token, CPU/GPU/RAM roles, sharding topology.
- [`K3-PERFORMANCE-MODEL.md`](K3-PERFORMANCE-MODEL.md) — K3 arithmetic, byte models, DDR rooflines and latency budgets.
- [`EVIDENCE-AND-SOURCES.md`](EVIDENCE-AND-SOURCES.md) — primary evidence, software support and research/prior-art boundary.
- [`HARDWARE-CANDIDATES.md`](HARDWARE-CANDIDATES.md) — concrete Carrizo boards/systems and procurement caveats.
- [`POC-VALIDATION-AND-PROCUREMENT-GATES.md`](POC-VALIDATION-AND-PROCUREMENT-GATES.md) — 1 -> 2 -> 4 -> 8-node proof ladder and GO/NO-GO rules.
- [`RESEARCH-LOG-2026-08-17.md`](RESEARCH-LOG-2026-08-17.md) — chronological reasoning, rejected paths and corrections.
- [`../../tools/apu_ddr3_k3_roofline.py`](../../tools/apu_ddr3_k3_roofline.py) — reproducible calculations used by the documents.

## Required performance-language rule

Use these evidence classes consistently:

1. **vendor/model fact** — stated by an authoritative source;
2. **measured physical/kernel result** — executed on named hardware;
3. **analytical roofline** — arithmetic from stated assumptions;
4. **screening estimate** — intentionally rough model for deciding what to test/buy;
5. **end-to-end K3 tok/s** — only a complete full-checkpoint generation benchmark.

Never report classes 3 or 4 as class 5.

## Purchase decision

Current purchase decision:

```text
1-4 prototype nodes: GO
160-node fleet:       NO-GO until validation gates pass
```

The next valuable information is physical measurement on Carrizo, not another multiplication of nominal DDR rates.