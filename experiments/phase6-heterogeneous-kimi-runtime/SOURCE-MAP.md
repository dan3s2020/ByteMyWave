# Phase 6 source map

Checked: **2026-08-15**

This file separates official facts, external benchmark evidence, and TensorWave analytical assumptions.

## Kimi K2.5 — official Moonshot

Repository:

- https://github.com/MoonshotAI/Kimi-K2.5

Deployment guide:

- https://github.com/MoonshotAI/Kimi-K2.5/blob/master/docs/deploy_guidance.md

Hugging Face config:

- https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/config.json

Facts used:

```text
Architecture: MoE
Total parameters: 1T
Activated parameters: 32B
Layers: 61
Dense layers: 1
Hidden dimension: 7168
MoE hidden/intermediate dimension: 2048
Routed experts: 384
Selected experts/token: 8
Shared experts: 1
Context: 256K
Attention: MLA
Native INT4 release
```

Moonshot explicitly lists KTransformers + SGLang as a CPU+GPU heterogeneous inference deployment option for K2.5.

## Kimi K3 — official Moonshot

- https://github.com/MoonshotAI/Kimi-K3

Facts used:

```text
Architecture: MoE
Total parameters: 2.8T
Activated parameters: 104B
Layers: 93
Dense layers: 1
Attention composition: 69 KDA + 24 Gated MLA
Hidden dimension: 7168
MoE hidden/intermediate dimension: 3072
Experts: 896
Selected experts/token: 16
Shared experts: 2
Context: 1,048,576
Quantization: MXFP4 weights / MXFP8 activations
```

## K3 DSpark — vLLM project

- https://github.com/vllm-project/vllm-project.github.io/blob/main/_posts/2026-07-27-k3.md

External sensitivity data used:

```text
open DSpark speculator
7 speculative tokens configured in vLLM recipe
~2.61 accepted tokens/step on high-entropy tasks
~4.73 accepted tokens/step on low-entropy/coding tasks
3.14x single-user speedup in vLLM's GB300 setup
```

These numbers are not R920/TensorWave measurements. They are only used as optional sensitivity factors in the Phase-6 multi-GPU calculations.

## Intel Xeon E7-4890 v2 — Intel

- https://www.intel.com/content/www/us/en/products/sku/75251/intel-xeon-processor-e74890-v2-37-5m-cache-2-80-ghz/specifications.html

Facts used:

```text
15 cores / 30 threads
2.80 GHz base
3.40 GHz turbo
4 memory channels
85 GB/s maximum memory bandwidth
32 PCIe 3.0 lanes
4-socket maximum configuration
Instruction Set Extensions: Intel AVX
```

The AVX-only status is important. No AVX2/AVX-512/AMX throughput is assumed for the R920.

## Dell PowerEdge R920 — Dell

Technical specifications:

- https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/technical-specifications

Expansion-card mapping:

- https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/expansion-card-installation-guidelines

Memory configurations:

- https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/sample-memory-configurations

Facts used:

```text
2 or 4 CPUs
96 DDR3 ECC DIMM sockets
up to 6 TB system RAM
PCIe Gen3
slot 4/5 -> CPU2 x16
slot 6/7 -> CPU3 x16
slot 8/9 -> CPU4 x16
slots 6-10 require all four CPUs
```

Electrical x16 links are kept separate from mechanical GPU fit/power/cooling claims.

## RTX 3060 — NVIDIA

- https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/
- https://www.nvidia.com/en-us/geforce/news/geforce-rtx-3060/

Facts used:

```text
Ampere
3584 CUDA cores
12 GB GDDR6 reference variant
192-bit memory interface
PCIe Gen4 capable
```

TensorWave Phase-5's `12 GB/s effective pinned H2D` is an **analytical assumption**, not an NVIDIA specification and not a measured R920 result.

The Phase-6 `360 GB/s GPU-memory-bandwidth` input is a reference-profile input for the hypothetical full-residency sensitivity test; it must be replaced by measured/validated bandwidth for an actual board.

## RTX 5090 — NVIDIA

- https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/

Facts used:

```text
Blackwell
32 GB GDDR7
1792 GB/s memory bandwidth
PCIe Gen5 capable
```

On R920, host streaming is still limited by the host's PCIe generation/topology. The higher 5090 VRAM bandwidth only dominates once enough weights are resident/reused.

## KTransformers — external heterogeneous-MoE evidence

Repository:

- https://github.com/kvcache-ai/ktransformers

K2.5 guide:

- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/Kimi-K2.5.md

K2 guide:

- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/Kimi-K2.md

AVX2 backend guide:

- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/AVX2-Tutorial.md

Relevant evidence:

```text
KTransformers performs CPU+GPU heterogeneous inference and CPU MoE expert offload.
Kimi K2 docs report roughly 10 TPS single-socket + consumer GPU for Q4 and ~14 TPS with dual-socket NUMA optimization on their tested platform.
The current AVX2-only backend requires AVX2 + FMA.
K2.5 high-performance examples use much newer CPUs/ISA than E7 v2.
```

Those measurements validate the architectural direction, **not** the speed of the R920.

## Compression research references

Candidates only; not implemented/proven in this branch:

- QuIP / extreme PTQ: https://arxiv.org/abs/2307.13304
- AQLM / additive quantization: https://arxiv.org/abs/2401.06118
- BitNet b1.58 training regime: https://arxiv.org/abs/2402.17764

TensorWave does not claim that an arbitrary Kimi checkpoint can be converted to 1.58-bit without quality loss.

## TensorWave analytical inputs inherited from Phase 5

```text
H2D effective bandwidth: 12 GB/s/GPU
GPU effective dense-linear compute: 10 TFLOP/s/GPU
Current Q4 wire format: 0.625 B/weight
```

These remain simulator inputs until real hardware calibration replaces them.
