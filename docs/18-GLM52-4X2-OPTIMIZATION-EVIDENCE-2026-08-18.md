# 18 — GLM-5.2 4×2 Optimization Evidence Review — 2026-08-18

## Purpose

This document freezes the evidence review for the selected near-term deployment target:

```text
4 physical servers
2 GPU cards/devices per server as the purchased 4×2 setup
all practical server RAM populated
GLM-5.2
GGUF / approximately Q3-class model representation
batch=1 interactive / agentic decode as the primary latency target
```

The objective is not to collect generic LLM tricks. It is to identify optimizations that can materially improve this exact regime: a very large sparse MoE whose complete weights live primarily in cheap host memory, with a limited amount of GPU memory acting as a hot execution/cache tier.

The evidence rules from the earlier ByteMyWave work still apply:

1. measured end-to-end model throughput;
2. measured subsystem/kernel throughput;
3. analytical/modelled result;
4. third-party benchmark with exact command/hardware;
5. paper speedup on another model/hardware;
6. idea/search lead.

A category-5 paper speedup must never be multiplied into a ByteMyWave token/s number as though it were category 1.

---

# 1. GLM-5.2 facts that matter to the runtime

Current official/public deployment material describes GLM-5.2 as a roughly `743B` total-parameter MoE with about `39B` active parameters per token.

The current Hugging Face configuration records:

```text
num_hidden_layers       = 78
n_routed_experts        = 256
num_experts_per_tok     = 8
n_shared_experts        = 1
num_nextn_predict_layers = 1
model_type              = glm_moe_dsa
```

Primary sources:

- https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json
- https://huggingface.co/zai-org/GLM-5.2
- https://recipes.vllm.ai/zai-org/GLM-5.2

Important implication:

> Only eight routed experts are selected per sparse layer, so expert ownership/cache placement can matter much more than blindly streaming all experts or treating the model as a dense 743B network.

---

# 2. Q3 storage anchor: a real GLM-5.2 GGUF build exists

A particularly useful community benchmark is `SixVolts/GLM-5.2-ewaste-edition-GGUF`.

Its published Q3-class build reports:

```text
model size       = 295.71 GiB
average bpw      = 3.37 bits/weight
runtime          = llama.cpp
zero-spill test  = 10 × AMD MI100 32 GB
Q3 decode        = 13.2 tok/s
```

It also reports a Q2 variant at `226.9 GiB` and `14.7 tok/s` on eight MI100s with zero CPU spill.

Source:

- https://huggingface.co/SixVolts/GLM-5.2-ewaste-edition-GGUF/blob/main/README.md

Evidence class: **third-party end-to-end benchmark with model layout, hardware and command line**.

This is not a performance prediction for our old GPUs. It is valuable for three reasons:

1. it proves a ~296 GiB Q3-class GLM-5.2 representation is practical today;
2. it provides a real measured target on a much stronger all-VRAM system;
3. it shows that avoiding expert spill is extremely important even when aggregate GPU compute is large.

A rough active-weight payload screen using the published `3.37 bpw` average is:

```text
39B active parameters × 3.37 / 8
≈ 16.43 GB of quantized parameter bits/token
```

This is a **screening number**, not an exact byte trace. The Q3 build deliberately uses higher precision for attention/shared experts and mixed Q2/Q3/Q4 choices across routed experts, so the real per-token traffic must ultimately be measured from the exact GGUF and routing trace.

---

# 3. Native GLM-5.2 MTP is real and is the highest-priority decoding optimization

The official vLLM recipe exposes GLM-5.2 native Multi-Token Prediction (MTP) with:

```text
method                 = mtp
num_speculative_tokens = 5
```

Primary sources:

- https://recipes.vllm.ai/zai-org/GLM-5.2
- https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/

The GLM-5.2 model card states that its MTP layer was improved and that acceptance length increased by up to 20% relative to the preceding path.

Source:

- https://huggingface.co/zai-org/GLM-5.2

Important interpretation:

> `5 speculative tokens` does **not** mean `5× token/s`.

The useful quantity is accepted target tokens per expensive verification pass. That depends on prompt/task, sampling parameters, MTP implementation, memory traffic and verification cost.

---

# 4. llama.cpp GLM-5.2 MTP status: changed during July 2026

This needs to be recorded carefully because a stale statement is easy to make.

On 2026-06-30 / early July, llama.cpp GLM-5.2 loaded NextN/MTP tensors but did not yet wire them into the GLM_DSA compute graph. The upstream discussion explicitly described the limitation.

Source:

- https://github.com/ggml-org/llama.cpp/discussions/25175

Later in July, upstream PR `#25980` added NextN/MTP speculative-decoding support for GLM_DSA / GLM-5.2. Subsequent issue `#26290` confirms that the change now causes the GLM-5.2 MTP tensors to be loaded by default when present and discusses the extra memory footprint.

Source:

- https://github.com/ggml-org/llama.cpp/issues/26290

Decision consequence:

> llama.cpp is now a viable Q3/GGUF base on which to test native GLM-5.2 MTP. However, the ByteMyWave branch must pin an exact llama.cpp commit and benchmark MTP acceptance + memory use rather than assuming the vLLM/Hopper behavior transfers to our hardware.

---

# 5. Model-free n-gram speculation: immediately usable, workload-dependent

The same GLM-5.2 Q3 community benchmark measured llama.cpp `ngram-simple` speculation on the 10×MI100 zero-spill system:

```text
plain prose       13.1 -> 13.1 tok/s   ~0%
structured code   13.0 -> 15.9 tok/s   +22%
verbatim repeat   13.1 -> 16.6 tok/s   +27%
```

The author reports that `ngram-simple` abstains on unpredictable output and has near-1.0 acceptance when it fires.

Source:

- https://huggingface.co/SixVolts/GLM-5.2-ewaste-edition-GGUF/blob/main/README.md

Decision consequence:

> `ngram-simple` is a low-risk first-stage optimization for coding/agentic workloads, but it must not be counted as a universal speedup.

Native MTP remains the preferred general speculative path once validated on the exact quant/runtime.

---

# 6. MoE-Infinity: request-aware expert tracing, caching and prefetch

`MoE-Infinity: Offloading-Efficient MoE Model Serving` tracks expert activation at request level and uses that trace for expert caching and prefetching.

The paper reports `2–20×` latency improvements over several offloading baselines across its tested models/tasks.

Source:

- https://arxiv.org/abs/2401.14361
- public code referenced by the paper: https://github.com/TorchMoE/MoE-Infinity

Evidence class: **paper/system result on other MoE models/hardware**.

What transfers to ByteMyWave:

- per-request expert activation traces;
- expert hit-frequency / reuse statistics;
- hot-expert VRAM cache;
- async host→GPU prefetch;
- cache-policy evaluation from actual GLM-5.2 traces.

What does not transfer automatically:

- the published speedup multiplier;
- its exact kernels/runtime;
- assumptions about PCIe topology or GPU generation.

---

# 7. KTransformers: direct GLM-5.2 heterogeneous scheduling reference

KTransformers announced GLM-5.2 support on 2026-06-17 and provides a current launch recipe using SGLang + KT-Kernel.

The GLM-5.2 tutorial exposes:

```text
--kt-num-gpu-experts
--kt-enable-dynamic-expert-update
--kt-expert-placement-strategy uniform/frequency/...
--kt-threadpool-count
```

The separate expert-scheduling tutorial exposes frequency placement, dynamic expert redistribution and GPU-expert-distribution recording.

Sources:

- https://github.com/kvcache-ai/ktransformers
- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/GLM-5.2-Tutorial.md
- https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/kt-kernel/experts-sched-Tutorial.md

Important limitation for our selected Q3 target:

> The current published GLM-5.2 KT tutorial is built around FP8/BF16 and a modern SGLang/CUDA stack. It is excellent reference code for dynamic expert placement, but it is not the production base for a Q3 deployment on a heterogeneous collection of old NVIDIA GPUs.

Decision consequence:

> Reuse the scheduling ideas and, where practical, algorithms/data structures — not the whole KT GLM-5.2 serving stack.

---

# 8. HybriMoE: hybrid CPU/GPU execution + cache management

`HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference` adds:

1. dynamic intra-layer CPU/GPU scheduling;
2. impact-driven inter-layer prefetching;
3. score-based expert caching.

The paper reports average speedups of:

```text
prefill ~1.33×
decode  ~1.70×
```

over its state-of-the-art hybrid MoE baseline on the tested systems.

Source:

- https://arxiv.org/abs/2504.05897
- code: https://github.com/PKU-SEC-Lab/HybriMoE

This is unusually relevant because HybriMoE is implemented on top of KTransformers and directly targets the same broad bottleneck class as the 4×2 design: GPU-resident hot experts plus CPU execution/offload for the rest.

Decision consequence:

> ByteMyWave should implement a scheduler that can choose per expert between `GPU cached`, `GPU prefetched`, and `CPU local` rather than treating every miss as a mandatory synchronous H2D transfer.

---

# 9. SpecMoEOff / SpecOffload: speculation can hide offload latency

`Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding` (SpecMoEOff) explicitly targets offloaded MoE inference.

It reports up to `2.5×` decode-throughput improvement by using speculative decoding to enlarge useful expert work and orchestrating CPU/GPU execution through roofline-guided tuning.

Source:

- https://arxiv.org/abs/2508.21706

`SpecOffload` similarly integrates speculative decoding into an offloading engine and reports `2.54×` throughput improvement over its best baseline.

Source:

- https://arxiv.org/abs/2505.10259
- code referenced by the paper: https://github.com/MobiSense/SpecOffload

Decision consequence:

> Native GLM-5.2 MTP is not merely a decoder trick for us. It can change the economics of host-memory offload because one target verification pass may amortize an expensive expert load over several accepted tokens.

Again, the paper maxima are not ByteMyWave multipliers.

---

# 10. SP-MoE: MTP/speculation-aware expert prefetch is the most relevant research direction

`SP-MoE: Speculative Decoding and Prefetching for Accelerating MoE-based Model Inference` combines:

- speculative expert prefetch;
- a cutoff-layer policy to limit harmful over-prefetch;
- asynchronous prefetch threads;
- batched I/O and compute/communication pipelining.

Reported TPOT speedup over state-of-the-art methods spans about `1.07–3.5×` across the paper's tested models/datasets/environments.

Source:

- https://arxiv.org/abs/2510.10302

Decision consequence:

> This is the closest published architecture to the desired ByteMyWave fast path. Once GLM-5.2 MTP is stable, the next-token/NextN information should drive expert prefetch rather than waiting for the target router to demand a cold expert synchronously.

---

# 11. SpecMoE: self-assisted speculation for MoE

`SpecMoE: A Fast and Efficient Mixture-of-Experts Inference via Self-Assisted Speculative Decoding` reports up to `4.30×` throughput improvement in its evaluated memory-constrained systems without requiring an external separately trained draft model.

Source:

- https://arxiv.org/abs/2604.10152

This supports the general direction of self-assisted/speculative MoE inference, but the exact algorithm must be evaluated against GLM-5.2's already-native MTP mechanism before porting anything.

---

# 12. Cache/prefetch policy research

`In-depth Analysis on Caching and Pre-fetching in Mixture of Experts Offloading` studies LRU behavior, proposes LFU caching improvements, and evaluates speculative expert prefetching.

Source:

- https://arxiv.org/abs/2511.05814

Decision consequence:

> The ByteMyWave cache policy must be trace-driven. We should benchmark LRU, LFU, frequency-decay and route-prediction-aware scoring on actual GLM-5.2 agentic traces instead of choosing a policy by intuition.

---

# 13. KV-cache compression: useful, secondary for short-context batch=1 decode

KIVI and GEAR demonstrate that quantized/compressed KV caches can greatly increase memory capacity and throughput when KV memory/batching is the bottleneck.

Sources:

- KIVI: https://arxiv.org/abs/2402.02750
- GEAR: https://arxiv.org/abs/2403.05527

The official GLM-5.2 vLLM recipe also uses FP8 KV cache on supported modern hardware.

Source:

- https://recipes.vllm.ai/zai-org/GLM-5.2

For the 4×2 Q3 target, the first bottleneck is expected to be expert weight movement / CPU-GPU execution, not KV capacity at a short benchmark context.

Decision consequence:

> Implement Q8/low-bit KV options and memory accounting, but do not delay expert/MTP work to chase KV optimizations first.

---

# 14. Sparse-attention/indexer kernels: important for long context, not first decode gate

`LiteTopK` targets the indexer/top-k path used in sparse attention and reports a `1.2×` prefill acceleration for GLM-5.2 in its deployment tests.

Source:

- https://arxiv.org/abs/2607.11976

This is relevant when long-context prefill becomes a priority. It is not expected to dominate short-context batch=1 token generation in our first prototype.

Decision consequence:

> Track it as phase-2/long-context work, not the first implementation milestone.

---

# 15. Old-GPU software compatibility is a real engineering constraint

The purchased GPU pool discussed in this project includes legacy NVIDIA devices, including Tesla K80-class hardware and Maxwell-class devices such as M10.

This matters because:

- Tesla K80 is Kepler `sm_37`;
- NVIDIA removed Kepler support from CUDA 12.0 libraries/toolchain;
- NVIDIA's current architecture matrix lists CUDA 11.x as the last toolkit family for Kepler;
- current llama.cpp CUDA CMake defaults target Maxwell `sm_50` as the lowest CUDA-12 architecture.

Primary sources:

- https://docs.nvidia.com/cuda/archive/12.0.0/cuda-toolkit-release-notes/index.html
- https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html
- https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/CMakeLists.txt

There is also a concrete llama.cpp K80 report showing a CUDA-11.4 build targeting `sm_37`, and another issue showing a multi-K80 llama.cpp deployment, which proves that the legacy path is possible with the right source/toolchain combination.

Sources:

- https://github.com/ggml-org/llama.cpp/issues/12140
- https://github.com/ggml-org/llama.cpp/issues/13661

Decision consequence:

> Do not require one identical CUDA binary on every GPU generation. ByteMyWave should use worker isolation: each server/GPU family can run the newest compatible backend build behind the same network/runtime protocol.

This avoids letting a K80 force the entire cluster onto a legacy toolchain.

---

# 16. What the literature does **not** prove

The following multiplication is invalid:

```text
HybriMoE 1.70×
× SpecMoEOff 2.5×
× SP-MoE 3.5×
× ngram 1.22×
= huge ByteMyWave speedup
```

These systems overlap heavily in what they optimize: cache misses, prefetch, speculative verification, CPU/GPU overlap and I/O hiding.

The correct ByteMyWave model is:

```text
T_token =
    serial non-MoE work
  + sum over MoE layers(
        max(local CPU expert critical path,
            local GPU cached expert critical path,
            prefetched expert critical path,
            unavoidable cold-transfer critical path)
        + route/reduction/collective cost
    )
  + attention/state/head overhead

Effective TPS with speculation = accepted output tokens / measured wall time
```

Only a physical trace can supply those terms.

---

# 17. Evidence-ranked optimization backlog

## Tier A — implement first

1. exact GLM-5.2 Q3 GGUF atlas and per-tensor byte accounting;
2. pin a current llama.cpp revision with GLM_DSA support;
3. NUMA-local mmap/read placement per server;
4. per-GPU hot-expert cache;
5. frequency/decay-based expert placement;
6. async H2D prefetch and double buffering;
7. CPU execution fallback for cold experts instead of mandatory transfer;
8. request-level expert trace recorder;
9. `ngram-simple` speculation for structured/repetitive agent output;
10. native GLM-5.2 MTP benchmark and acceptance logging;
11. four-server expert ownership rather than a naive four-stage layer pipeline where possible.

## Tier B — implement after the baseline trace

1. MTP-aware speculative expert prefetch;
2. prefetch cutoff / overfetch controller;
3. LFU vs LRU vs score cache policy auto-tuning;
4. overlap remote expert dispatch with local compute;
5. expert replication for globally hot experts;
6. Q8/low-bit KV cache;
7. fused Q3 dequant + GEMV/MMQ paths specific to the actual GPU family.

## Tier C — long-context / later

1. LiteTopK/indexer optimization;
2. aggressive KV compression beyond the proven llama.cpp options;
3. model-format changes beyond Q3 unless quality/capacity measurements justify them.

---

# 18. Source-of-truth conclusion

The evidence supports continuing with the 4-server × 2-GPU prototype.

It **does not** support a claim that any single public optimization turns the system into a 10+ tok/s GLM-5.2 machine.

The best-supported architecture is a hybrid of existing public ideas:

```text
llama.cpp / GGUF Q3 execution base
        +
ByteMyWave expert atlas + distributed ownership
        +
NUMA-local host-resident experts
        +
GPU hot-expert cache
        +
dynamic cache/placement policy inspired by KTransformers/HybriMoE/MoE-Infinity
        +
async expert prefetch
        +
GLM-5.2 native MTP
        +
MTP-aware speculative prefetch inspired by SP-MoE / SpecMoEOff
        +
per-hardware worker builds for legacy CUDA compatibility
```

That is the architecture to implement and benchmark next.