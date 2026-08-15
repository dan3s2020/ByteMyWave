# TensorWave — Prior Art and Novelty Boundary

Date: 2026-08-15

## Bottom line

TensorWave's foundational idea is **not unique**:

> keep a model larger than VRAM in host memory/storage and stream the needed weights to a smaller GPU.

That family of techniques already exists in published systems and open-source runtimes.

The useful question for TensorWave is therefore **not** “are we the first people to offload weights from RAM to GPU?” The useful question is:

> can we push the mechanism to a smaller, more deterministic, more compressed working set and make a 4 GB consumer GPU productive rather than merely capable of loading an oversized model?

This document defines which ideas are established prior art and where TensorWave may still provide a distinct engineering contribution.

---

## Closest prior art identified so far

### AirLLM

Project:

- https://github.com/lyogavin/airllm

Relevant ideas:

- runs models far larger than GPU VRAM by loading weights progressively;
- layer-wise loading/offload;
- prefetching so loading of future weights can overlap compute;
- low-bit weight compression;
- recent releases also move packed low-bit representations across the host/device boundary and expand on GPU for some supported models;
- MoE-specific paths can avoid loading inactive experts.

Meaning for TensorWave:

- “large model on a 4 GB GPU” is not a novelty claim;
- “prefetch the next weights while the current weights are computed” is not a novelty claim;
- “send compressed weights across PCIe and expand on GPU” is not a novelty claim.

AirLLM also demonstrates the central warning for this entire project: **fitting is not the same thing as being useful**. A model can technically execute with tiny VRAM while end-to-end token latency remains unacceptable if the workload becomes storage/PCIe bound.

---

### DeepSpeed ZeRO-Inference

Project/examples:

- https://github.com/deepspeedai/DeepSpeedExamples/tree/master/inference/huggingface/zero_inference

Relevant ideas:

- GPU + CPU + NVMe inference;
- weight offload;
- low-bit weight representation;
- reduce host/device bytes by quantizing weights;
- overlap communication and compute;
- throughput-oriented inference where batch size provides reuse of each transferred weight block.

Meaning for TensorWave:

- quantized host-to-device weight traffic is established;
- overlap is established;
- large-batch reuse is known to be essential when transfer bandwidth would otherwise dominate.

---

### FlexGen

Paper:

- https://arxiv.org/abs/2303.06865

Relevant ideas:

- aggregate GPU, CPU RAM and disk as one inference memory hierarchy;
- schedule placement and movement of weights/activations/KV state;
- 4-bit compression;
- optimize throughput under severe memory constraints.

Meaning for TensorWave:

- multi-tier memory scheduling is prior art;
- “VRAM as a small fast tier, RAM/disk as large slow tiers” is prior art;
- TensorWave must distinguish itself through the exact granularity, static schedule, compressed ring and low-starvation objective rather than the hierarchy itself.

---

### ATSInfer

Paper identified in the 2026 literature search:

- https://arxiv.org/abs/2607.10183

Relevant ideas:

- tensor-granularity offload rather than only whole-layer offload;
- static tensor placement combined with dynamic transfer;
- asynchronous CPU/GPU coordination.

Meaning for TensorWave:

- “layer-level is too coarse; stream at tensor granularity” is also not unique by itself;
- TensorWave needs to go below this into deterministic sub-tensor tiles and a fixed compressed VRAM ring if it wants a distinct implementation boundary.

---

### PowerInfer / PowerInfer-2

Papers:

- https://arxiv.org/abs/2312.12456
- https://arxiv.org/abs/2406.06282

Relevant ideas:

- exploit hot/cold activation structure;
- keep frequently useful work close to the accelerator;
- divide matrix work into smaller neuron clusters;
- overlap I/O and compute at a granularity below a whole layer.

Meaning for TensorWave:

- sub-matrix/neuron-cluster scheduling already exists as a research direction;
- a TensorWave hot/warm/cold cache is conceptually related prior art;
- TensorWave should measure its contribution in terms of bytes avoided, starvation avoided and working-set size rather than claim that sub-layer partitioning itself is new.

---

### Fast low-bit / NF4 dequantization kernels

Example paper identified during the search:

- https://arxiv.org/abs/2604.02556

Relevant ideas:

- optimize 4-bit dequantization on GPU;
- move decode work close to shared-memory/register-level compute;
- reduce the overhead of repeatedly expanding quantized weights.

Meaning for TensorWave:

- “dequantize close to the kernel” is an established optimization direction;
- TensorWave Phase 4 should use/compare against existing CUTLASS/CuTe/narrow-integer approaches rather than invent a scalar GEMM.

---

## What TensorWave must NOT claim as novel

Do not claim any of the following without qualification:

```text
Large models can run with weights in CPU RAM.
A 4 GB GPU can technically execute models larger than 4 GB.
Weights can be prefetched while the GPU computes.
CPU/NVMe/GPU can form a tiered memory hierarchy.
4-bit weights reduce PCIe traffic.
Compressed weights can be dequantized on GPU.
Layer-level offload can be made finer-grained.
Hot weights can be cached in VRAM.
```

All of those ideas have clear prior art.

---

## The current TensorWave-specific engineering combination

The currently interesting composition is:

```text
large quantized host model store
        |
        v
byte-addressable Weight Atlas
        |
        v
offline/static execution schedule
        |
        v
sub-tensor fixed-size tiles
        |
        v
bounded pinned-host staging
        |
        v
2–3 fixed compressed VRAM slots
        |
        v
scheduled cudaMemcpyAsync
        |
        v
Q4 fragment decode
        |
        v
shared memory / registers
        |
        v
Tensor Core MMA
        |
        v
discard decoded fragment
```

The distinctive objective is not merely memory capacity. It is:

> **minimize unhidden host-to-device transfer and GPU starvation while keeping the fixed VRAM weight working set extremely small.**

The strongest potential differentiators are therefore:

1. **sub-tensor/tile scheduling as a first-class runtime contract**, not an incidental implementation detail;
2. **static/precompiled execution and slot ownership**, eliminating runtime tensor lookup from the hot path when the graph is deterministic;
3. **compressed representation remains compressed across RAM and PCIe**;
4. **future fused decode consumes only the current MMA fragment, avoiding a complete decompressed tensor/tile in VRAM**;
5. **the runtime is explicitly optimized around `GPU starvation %` rather than only “model fits” or raw H2D bandwidth**;
6. **a feasibility map predicts which workload/hardware combinations should work before a full model integration is attempted**.

This combination may still overlap prior patents/papers. No patentability or uniqueness claim should be made until a dedicated patent/literature search is performed.

---

## Research position

The prior-art search is positive for the project even though it reduces novelty claims.

Independent systems validate almost every physical premise:

- progressive/offloaded inference works;
- asynchronous prefetch works;
- low-bit H2D reduces traffic;
- finer granularity can beat whole-layer placement;
- hot/cold placement matters;
- dequantization belongs close to compute.

TensorWave's job is now to determine the **operating envelope** where those techniques make a 4 GB GPU useful.

That operating envelope is the subject of the Feasibility Map phase.
