# Evidence and Sources — Carrizo APU + DDR3 UMA

## Purpose

This file separates four things that are easy to mix together:

1. vendor/model facts;
2. software-support facts;
3. research/prior-art evidence;
4. ByteMyWave hypotheses that still require measurement.

A source showing that an architectural mechanism exists is **not** automatically evidence that Kimi K3 will achieve a particular token rate on Carrizo.

## Evidence table

| Item | Evidence | What it supports | What it does not prove |
|---|---|---|---|
| Carrizo hUMA | AMD Carrizo architecture disclosure | CPU and GPU use a unified/shared memory architecture and can access platform memory | sustained Radeon R6 bandwidth for K3-shaped kernels |
| A8-8600P memory topology | AMD product page | two DDR3 channels; memory support up to DDR3-2133 | exact speed on a particular OEM motherboard/BIOS/DIMM set |
| A8-8600P integrated compute | AMD product page | Radeon R6 iGPU, HSA/Vulkan-class capabilities, 15 W default TDP | modern AI tensor-core performance |
| Carrizo compiler target | LLVM AMDGPU usage docs | Carrizo/A8-8600P maps to AMDGPU target `gfx801` | a turnkey modern ROCm inference stack |
| Vulkan driver path | Mesa RADV docs | GFX8-class hardware remains in the RADV support family | K3 kernels already implemented or fast |
| Modern ROCm matrix | AMD ROCm compatibility docs | current supported compute targets are modern generations, not Carrizo/gfx801 | that old experimental stacks cannot run anything; only that modern supported ROCm cannot be assumed |
| K3 model dimensions | official K3 model card/config/paper | 2.8T total, ~104B active, 93 layers, 896 routed experts, 16 selected, relevant dimensions | exact physical bytes read by our runtime |
| K3 quantization config | official K3 `config.json` | MXFP4 group size/scale representation and component exclusions | that every active tensor is MXFP4 |
| K3 checkpoint size | official K3 repository | released checkpoint is about 1.56 TB | active-byte traffic per decode token |
| K3 expert tensor structure | official model implementation | three routed-expert matrices and dimensions used by our expert arithmetic | exact runtime traffic after caching/fusion |
| Expert tensor sharding | MoEShard | expert matrices can be tensor-sharded to spread work/load across devices | 160 Carrizo nodes will scale similarly |
| Edge GPU/NDP MoE | research on edge GPU + near-data processing | tensor parallel expert execution near memory is a researched mechanism | direct K3/Carrizo performance |
| Shared-memory/zero-copy GPGPU | HSA/zero-copy research | shared-memory accelerators can avoid discrete PCIe copy patterns | our end-to-end distributed latency |

## Primary hardware/software sources

### AMD Carrizo / hUMA

AMD, “AMD Discloses Architecture Details of High-Performance, Energy-Efficient Carrizo System-on-Chip”:

https://ir.amd.com/news-events/press-releases/detail/599/amd-discloses-architecture-details-of-high-performance-energy-efficient-carrizo-system-on-chip

Relevant architectural point: Carrizo was designed as an HSA-capable SoC with CPU and GPU sharing a coherent/unified memory view rather than requiring the classic discrete-GPU copy model for all data.

### AMD A8-8600P

Official product/support page:

https://www.amd.com/en/support/downloads/drivers.html/processors/a-series/a8-series-apu-for-laptops/6th-gen-a8-8600p-apu.html

Working facts used in this track:

```text
CPU cores/threads: 4 / 4
GPU:               Radeon R6
GPU cores:          6 graphics cores
memory channels:    2
memory:             DDR3, up to 2133 MT/s at APU level
default TDP:        15 W
```

The exact OEM board can impose lower memory clocks/capacity or different expansion constraints.

### LLVM AMDGPU target documentation

https://rocm.docs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUUsage.html

Carrizo is identified under `gfx801`; the device list includes A8-8600P / Pro A8-8600B-class parts.

### Mesa RADV

https://docs.mesa3d.org/drivers/radv.html

RADV is the Vulkan driver path relevant to a current Linux proof-of-concept. Current documentation includes GFX8-class support and also notes that the optional GFX8 float16 path is not generally beneficial. That is a warning against assuming modern FP16 AI behavior from an old iGPU.

### Modern ROCm compatibility

https://rocm.docs.amd.com/en/develop/compatibility/compatibility-matrix.html

Carrizo/gfx801 is not a target on which this project should assume supported current ROCm/vLLM deployment. The POC therefore treats a custom Vulkan/LLVM-oriented compute path as the baseline unless hardware testing finds another maintained route.

## Primary Kimi K3 sources

### K3 paper

https://arxiv.org/abs/2607.24653

Use this for architecture/deployment details, including KDA/Gated MLA, MoE layout and implementation observations. Do not transfer modern datacenter-GPU performance figures to Carrizo.

### Official model card

https://huggingface.co/moonshotai/Kimi-K3/blob/main/README.md

Important working quantities:

```text
total params           ~2.8T
activated params       ~104B
hidden layers          93
routed experts         896
selected experts       16
hidden size            7168
latent MoE dimension   3584
MoE hidden dimension   3072
```

### Official config

https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json

This is critical for the correction to the old “104B * 0.5 byte” simplification. It declares the MXFP4 packed quantizer and an ignore list, so active storage is a mixed representation.

### Official checkpoint repository

https://huggingface.co/moonshotai/Kimi-K3/tree/main

The repository size is approximately 1.56 TB. This is a capacity fact, not a direct per-token traffic fact.

### Official model implementation

https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py

The routed expert path exposes the three-matrix expert structure used in the `3 * 3584 * 3072` working arithmetic. Exact tensor names/dtypes in the downloaded revision must still be enumerated programmatically before fleet procurement.

## Research / prior art

### MoEShard — expert sharding

“Accelerating MoE Model Inference with Expert Sharding”:

https://arxiv.org/abs/2503.08467

Why it matters here:

- demonstrates row/column tensor sharding of experts as a way to distribute active expert work;
- addresses expert load imbalance;
- supports the mechanism required if one K3 token is to use more memory controllers than the small number of whole-expert owners it happens to select.

Boundary:

- different hardware/model/runtime;
- its reported speedups are not Transit predictions.

### Edge GPU + near-data-processing MoE scheduling

“A Scheduling Framework for Efficient MoE Inference on Edge GPU-NDP Systems”:

https://arxiv.org/abs/2601.03992

Why it matters:

- investigates single-batch MoE inference where expert tensor parallelism spreads work across memory-adjacent processing units;
- confirms that load balance and communication/scheduling are first-order design problems.

Boundary:

- not Carrizo;
- not Kimi K3;
- no direct token/s transfer is valid.

### Shared-memory / zero-copy GPGPU

“Overcoming Limitations of GPGPU-Computing in Scientific Applications”:

https://arxiv.org/abs/1905.05175

Why it matters:

- studies the data-movement limitations of discrete accelerators and alternatives including zero-copy/shared-memory HSA devices;
- provides prior-art context for treating an APU as a memory-compute tile rather than a discrete accelerator behind a separate PCIe weight feed.

Boundary:

- scientific workloads, not LLM inference;
- mechanism evidence only.

## Existing ByteMyWave evidence that must remain in context

### PR #10 — Transit DDR3 tile architecture

https://github.com/dan3s2020/ByteMyWave/pull/10

Relevant lessons retained:

- weights should remain local to their compute endpoint;
- host links should carry smaller commands/activations/results where possible;
- local memory bandwidth must be measured with model-shaped access;
- roofline token equivalents are not end-to-end tokens/s.

### PR #11 — distributed cheap-memory K3

https://github.com/dan3s2020/ByteMyWave/pull/11

Relevant lessons retained:

- capacity != throughput;
- pure layer pipeline does not sum node bandwidth for one decode token;
- expert parallelism can expose concurrent memory buses;
- per-layer collective latency must be measured;
- 52 GB/token is explicitly an optimistic four-bit lower bound and ~58 GB/token a screening model, not exact traffic.

### PR #9 — heterogeneous MoE

https://github.com/dan3s2020/ByteMyWave/pull/9

Relevant lessons retained:

- active/routed parameter arithmetic;
- NUMA locality and memory bandwidth are as important as nominal FLOPS;
- physical validation must precede throughput claims.

## Claims deliberately NOT made by this branch

This research track does **not** claim:

- that Radeon R6 already has a working K3 kernel;
- that Carrizo sustains its nominal dual-channel DDR rate from Vulkan compute;
- that 160 boards will expose all 320 memory channels to every token;
- that 1 GbE, 10 GbE or 25 GbE is definitely sufficient before measurement;
- that 52, 58 or 136.6 GB/token is the exact active traffic;
- that 30, 40, 80 or 100 tok/s has been demonstrated;
- that an OEM motherboard found on a marketplace can be purchased in fleet quantity at the displayed unit price.

The purpose of the POC ladder is to convert these unknowns into measured facts.