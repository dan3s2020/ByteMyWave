# Conversation audit — what was retained, corrected, or rejected

Date: **2026-08-15**

This file is the explicit second-pass audit requested after documenting the discussion. It exists to prevent useful ideas from being lost and, equally important, to prevent conversational shortcuts from becoming project facts.

## 1. Retained: TensorWave is a runtime/backend, not a UI

Decision retained:

```text
ComfyUI / OpenAI-compatible API / CLI / other frontend
                         |
                         v
                 TensorWave runtime
                         |
             NUMA RAM + CPU + GPU
```

A ComfyUI integration should be a thin custom-node/plugin layer. TensorWave must keep ownership of:

```text
Weight Atlas
compression format
NUMA placement
pinned buffers
VRAM resident set/cache/ring
prefetch schedule
CPU/GPU expert ownership
metrics
```

Reason: allowing a generic frontend/runtime to independently move model tensors would destroy the memory-placement experiment.

## 2. Retained: current Phase-5 model is a roofline, not a complete tok/s simulator

The general decode lower-bound/roofline must include at least:

```text
T_step >= max(
    host->GPU weight transfer,
    GPU VRAM weight reads,
    GPU compute,
    low-bit unpack/dequant,
    CPU expert execution,
    attention/KV/recurrent-state work,
    CPU<->GPU handoff,
    GPU<->GPU synchronization
)
```

Phase 5 includes host transfer and simple dense-linear GPU compute. Phase 6 adds several missing terms/sensitivity gates. No single term should be presented as the final measured speed.

## 3. Retained: a faster GPU can have nearly the same decode speed on the same host feed

For M=1 weight streaming, replacing an RTX 3060 by a much faster GPU does not automatically increase token rate when fresh weights still cross one PCIe Gen3 x16 host link.

A faster GPU helps when:

```text
more weights are resident/reused
VRAM is larger
GPU memory bandwidth is higher
compute becomes dominant
M is large (prefill/batching/image/video)
```

A real RTX 5090 has much more VRAM and much higher VRAM bandwidth than a 3060, but on an R920 its host feed is still constrained by the R920 PCIe topology.

## 4. Corrected: hypothetical 3060 with 2 TB VRAM

Earlier conversation used only the compute equation for K3 and obtained roughly 48 tok/s at the Phase-5 10-TFLOP/s effective-compute assumption.

That was incomplete.

Once weights are resident, host PCIe disappears but GPU-memory reads remain. With current TensorWave Q4:

```text
104B active * 0.625 B = 65 GB active weight reads/token
360 GB/s reference RTX-3060-class VRAM bandwidth / 65
= ~5.54 tok/s
```

So the simple full-residency roofline becomes about **5.54 tok/s**, not 48 tok/s. Compute-only remains a secondary ceiling (~48 tok/s).

This correction is encoded in the Phase-6 simulator and unit test.

## 5. Retained: compression in RAM before transport

Weights should live in host RAM already compressed.

Preferred pipeline:

```text
compressed host store
    |
    +-- Weight Atlas finds next fragment
    +-- pinned/DMA queue prepared while current fragment computes
    v
async compressed H2D
    v
compressed VRAM ring/cache
    v
GPU low-bit decode
    v
GEMV/GEMM/MMA
```

Do **not** recompress the same weights every token. The host should prepare addresses/queues, not redo model quantization on the hot path.

## 6. Retained: fuse decode with compute later

Current Phase 3:

```text
Q4 -> VRAM -> full FP16 dequant tile -> GEMM
```

Preferred later kernel:

```text
compressed fragment
-> decode only needed fragment
-> registers/shared memory
-> direct compute
```

This reduces full-tile FP16 materialization, VRAM workspace and VRAM traffic.

## 7. Retained: Q4 is not a fundamental limit

Compression candidates carried forward:

```text
Q3
Q2
mixed Q2/Q3/Q4 by tensor/tile sensitivity
Q2 + sparse higher-precision residual/outliers
vector/additive quantization
sub-2-bit only after measured quality validation
```

But the audit preserves the mathematical result that compression alone cannot solve K3 5–10 tok/s through one 12 GB/s link if every active weight crosses that link.

## 8. Retained: Kimi K3 one-GPU boundary

Using official 104B active parameters and current TensorWave Q4:

```text
65 GB fresh active bytes/token
12 GB/s host feed
= 0.1846 tok/s stream-all ceiling
```

Even the idealized 1-bit stress test is below 1 tok/s if all active K3 weights still cross one feed.

Therefore `2 TiB RAM + 1x RTX 3060 12 GiB` solves K3 capacity in current Q4 but not useful single-user speed.

## 9. Retained: Kimi K2.5 is the practical proving target, not a claim of K3-equivalent capability

K2.5 is used because it preserves the important giant-MoE execution problem while lowering the active set to 32B/token.

It must **not** be documented as having K3-level capability. K3 remains the stronger stress/frontier target from this conversation.

## 10. Retained and elevated: CPU expert execution is the main new Phase-6 hypothesis

This was the most important new result.

For K2.5, derived routed active parameters are ~21.139B/token. Instead of moving their compressed weights over PCIe each token, keep routed experts NUMA-local and execute them on CPU.

Then CPU<->GPU moves compact activations/results rather than ~13.2 GB of routed Q4 weight bytes/token.

Exact four-socket Q4 gates are preserved in README/RESULTS/simulator:

```text
5 tok/s:
16.515 GB/s/socket selected-weight reads
52.848 GFLOP/s/socket logical
26.424 Gweights/s/socket

10 tok/s:
33.030 GB/s/socket
105.696 GFLOP/s/socket
52.848 Gweights/s/socket
```

This is a measurable hypothesis, not a performance claim.

## 11. New caveat found during audit: E7-4890 v2 is AVX-only

The R920 CPU has old vector ISA support. Modern heterogeneous-MoE prior art validates CPU expert offload but commonly relies on AVX2/FMA, AVX-512 or AMX-class kernels.

Therefore R920 memory bandwidth alone is insufficient proof. The AVX-era expert microbenchmark in this directory is a mandatory gate.

If the socket cannot meet selected-weights/s, the next actions are concrete:

```text
move more experts to GPU ownership
add independent GPUs/PCIe feeds
use stronger compression if quality holds
or replace the CPU platform
```

## 12. Retained: 4 CPUs are preferable to 2 for CPU expert execution

Four sockets halve the K2.5 per-socket routed workload versus two sockets.

At Q4:

```text
2 sockets, 5 tok/s  -> 33.030 GB/s + 105.696 GFLOP/s/socket
4 sockets, 5 tok/s  -> 16.515 GB/s + 52.848 GFLOP/s/socket
```

The value is not merely 120 threads. It is four NUMA memory domains and four parallel expert engines.

## 13. Retained: multiple GPUs help single-request speed only after real sharding

Replicated GPU workers improve aggregate request throughput, not latency of one request.

For one request, TensorWave needs model/expert ownership:

```text
NUMA-local host shard -> local GPU
NUMA-local host shard -> local GPU
...
```

Prefer compact activation/reduction traffic between devices. Do not send tens of GB of model weights through GPU0 to other GPUs.

RTX 3060 has no NVLink; P2P availability/bandwidth must be measured on the actual topology.

## 14. Retained: DSpark/speculative decoding is a multiplier after target-pass cost

The external K3 DSpark acceptance values are useful sensitivity inputs, not TensorWave measurements.

Correct accounting:

```text
output tok/s
= target verification steps/s
* accepted output tokens/verification
```

Only after the target verification cost includes transfer/compute/state/synchronization.

## 15. Retained: image/video/prefill differ from M=1 decode

The severe PCIe result applies most strongly to single-token dense/MoE decode with low reuse.

When one weight tile is reused across many rows/tokens (`M` large), transfer can be hidden under compute. Therefore image/video generation and LLM prefill/batched serving may be far more favorable to the existing GPU streaming architecture.

This is why TensorWave should remain workload-generic rather than hard-coding the K2.5 CPU path into the whole runtime.

## 16. Corrected/blocked: MiniMax H3 public-weight claims

The earlier conversation stated that `MiniMaxAI/MiniMax-H3` weights had appeared and then reasoned from an alleged exact parameter breakdown.

During this audit, that claim could **not** be verified from MiniMax's official public Hugging Face model organization. The official model listing checked on 2026-08-15 did not expose `MiniMax-H3`.

Therefore this Phase-6 branch deliberately does **not** encode the earlier 33B/13B/FL2VA checkpoint claims as facts or use them in simulations.

Project rule:

```text
H3 stays release-gated until an official MiniMax repository/checkpoint can be fetched and inspected.
```

Once official weights exist, TensorWave should build a real Weight Atlas from the checkpoint before estimating its exact runtime.

## 17. Nothing from the base branch was overwritten

All additions in this research branch are isolated under:

```text
experiments/phase6-heterogeneous-kimi-runtime/
```

plus, if enabled, one dedicated Phase-6 CI workflow. Existing Phase 1–5 source/docs remain untouched.

## 18. Final checklist against the conversation

Captured:

- bandwidth vs compute distinction
- 3060 vs faster GPU on same PCIe host
- K3 2 TiB capacity and <1 tok/s one-feed boundary
- K2.5 exact active/routed arithmetic
- static GPU residency without invented cache-hit ratios
- Q3/Q2/mixed compression direction
- why compression alone eventually fails
- precompressed host store + async prefetch
- fused GPU low-bit decode direction
- 2-vs-4 CPU derivation
- CPU NUMA expert execution
- activation-volume calculation and handoff-latency caveat
- AVX-only R920 caveat
- KTransformers heterogeneous prior-art distinction
- multiple independent PCIe feeds and real sharding
- GPU-GPU/P2P design rule
- DSpark as a post-pass multiplier
- hypothetical huge-VRAM correction with VRAM bandwidth
- standalone runtime + ComfyUI/API/CLI adapters
- high-M image/video/prefill distinction
- H3 claim re-audited and release-gated

No probability-based cache hit or unmeasured 5–10 tok/s claim is treated as a result.