# 16 — Kimi K3 Consumer-GPU vs Transit Reality Check — 2026-08-17

## Purpose

This document freezes the 2026-08-17 reality check triggered by the question:

> If a very large Kimi K3 checkpoint can be made resident in many consumer GPUs, does that only solve capacity, or can a large consumer-GPU rig also produce high single-stream decode speed — and how should that be compared fairly with Transit’s 304-channel DDR3 target?

The main correction is methodological:

> **Do not sum all physical memory bandwidth in one architecture while treating the other architecture as serial. Count only bandwidth/compute that can actually participate concurrently on the same token at the relevant layer/stage, then include serial layer boundaries and communication.**

This note does **not** replace the Transit architecture. It adds an apples-to-apples performance framework and a current external benchmark ledger.

---

## 1. Review completeness

Before writing this note, the following current ByteMyWave surfaces were inspected:

```text
[x] main / CURRENT-ARCHITECTURE.md
[x] main / README.md
[x] main / docs/REPOSITORY-MAP.md
[x] PR #10 / transit-ddr3-architecture
[x] PR #11 / docs/kimi-k3-ddr-cluster
[x] PR #9  / research/heterogeneous-moe-kimi-v1
```

The required Transit implementation/evidence path was also inspected, including:

```text
docs/08-EVIDENCE-BENCHMARKS.md
docs/09-BITPLANE-MATH-AND-KERNEL.md
docs/10-KIMI-K3-TARGET.md
docs/11-DDR3-TILE-ARCHITECTURE.md
docs/13-IMPLEMENTATION-ROADMAP.md
docs/14-CURRENT-SOLUTION.md
benchmarks/README.md
benchmarks/transit_bitplane_kernel.asm
host/reference_bitplane.py
host/atlas.py
host/protocol.py
```

The distributed-K3 track was checked against:

```text
docs/07-KIMI-K3-DISTRIBUTED-RUNTIME.md
docs/08-HARDWARE-DDR2-DDR3-DDR4.md
docs/09-KIMI-K3-THROUGHPUT-MODEL.md
docs/10-IMPLEMENTATION-PROCUREMENT-PLAN.md
docs/11-RESEARCH-LOG-2026-08-16.md
```

Therefore this is a cross-branch current-state note rather than a conclusion inferred from `main` alone.

---

## 2. Evidence language used here

Every number must remain in one of these classes:

1. **Measured end-to-end model decode** — a real full or explicitly modified model generating tokens on named hardware.
2. **Measured subsystem/kernel result** — memory/kernel/network measurement, not full model generation.
3. **Analytical roofline** — arithmetic upper bound under stated assumptions.
4. **Weight-path tok/s equivalent** — memory/weight-processing rate normalized by modeled active weight bytes/token.
5. **Unverified conversation lead** — useful clue not yet recovered from a strong enough source.

Classes 2–4 must never be silently presented as class 1.

---

## 3. K3 facts that constrain every architecture

Current project working values, consistent with the official Kimi K3 release:

```text
total parameters                 ~2.8T
activated parameters / token     ~104B
transformer layers               93
routed experts                   896
selected routed experts/token    16
native headline weights          MXFP4
native headline activations      MXFP8
released checkpoint footprint    ~1.56 TB project working value
```

Official project source:

- https://github.com/MoonshotAI/Kimi-K3

The simple four-bit active-weight lower bound remains:

```text
104e9 active weights/token × 0.5 byte/weight
= 52 GB/token
```

The distributed-K3 branch also uses a checkpoint-average screening estimate:

```text
1.56 TB / 2.8T ~= 0.557 byte/parameter
104B × 0.557 ~= 58 GB/token
```

These two values answer different questions:

- `52 GB/token` = optimistic Q4-equivalent lower-bound weight payload;
- `~58 GB/token` = rough storage-ratio screening estimate;
- neither is a tensor-traced measurement of exact K3 bytes touched per token.

K3’s real hot path also includes non-routed/shared/attention state, scales, metadata, KV/KDA state and repeated kernel/collective overhead.

---

## 4. External reality anchors — systems that actually ran K3

### 4.1 Official vLLM / NVIDIA-class result — strongest performance anchor

vLLM reports Kimi K3 on **16 NVIDIA GB300 NVL72 GPUs** at:

```text
118 tok/s per user   without speculative decoding
370 tok/s per user   with DSpark speculative decoding
```

This is important because it proves that `>100 tok/s` end-to-end single-user K3 is physically achievable when the system combines very high local HBM bandwidth, strong low-precision kernels, fast collectives and K3-specific runtime work.

Source:

- https://vllm.ai/blog/2026-07-27-k3

Evidence class: **measured/reproducible framework benchmark**.

### 4.2 80 × RTX 5090 — direct proof that consumer GPUs are not only a capacity trick

Corespan summarizes the July 27, 2026 demonstration in which the **full Kimi K3** ran on:

```text
80 × RTX 5090
plain Ethernet
official MXFP4 weights
single untuned stream
~20 tok/s day-one decode
```

The same report says the fleet had previously taken GLM-5.2 from roughly `30 tok/s` to `110 tok/s` after optimization, showing that software/kernel/orchestration tuning can move the result dramatically even with unchanged physical GPUs.

Source:

- https://www.corespan.ai/resources/blog/frontier-intelligence-on-gaming-silicon

Evidence class: **operator/demo report summarized by an infrastructure vendor**, not an independently reproduced academic benchmark. Strong enough to prove feasibility; not strong enough to derive a universal efficiency coefficient.

### 4.3 8 × H20 / llama.cpp K3 layer split — useful software-topology anchor

A llama.cpp issue reports:

```text
8 × NVIDIA H20 141 GB
Kimi K3 Q2_K_XL
split-mode layer
~13 tok/s decode
```

The author explicitly requests `split-mode row/tensor` because K3 currently worked only with layer split in that path and per-layer parallel computation was desired to improve decode speed.

Source:

- https://github.com/ggml-org/llama.cpp/issues/26365

Evidence class: **user self-report in an upstream issue**. Useful for topology/software evidence; not an official benchmark.

### 4.4 M3 Ultra K3-derived benchmark — useful memory-roofline utilization reference, not full K3 equivalence

PipeNetwork reports a pruned/modified K3-derived MLX build on a **512 GiB M3 Ultra** at about:

```text
5.51 tok/s decode
~87 GB weights read/token for that build
~819 GB/s machine bandwidth
~9.5 tok/s simple memory ceiling
~58% of that simple ceiling achieved
```

The same model card explicitly states that pruning changes model quality and that the result is bandwidth-bound.

Source:

- https://huggingface.co/pipenetwork/Kimi-K3-REAP73-MLX-mxfp4-q8/blob/main/README.md

Evidence class: **project benchmark on a modified/pruned K3-derived checkpoint**. It is a useful demonstration that real decode can land far below a simple bandwidth quotient; its `58%` must **not** be copied as a Transit or RTX-3060 efficiency constant.

---

## 5. Claims discussed earlier but not promoted to project facts

During the conversation, additional examples were mentioned, including large RTX 3090/4090 rigs and earlier Mac K3 figures. Those claims are not promoted here because this review did not recover a strong enough primary/reproducible source for every exact number, and some K3-derived MLX model cards have since corrected earlier results.

Rule:

> Keep such examples as search leads only until exact hardware, model revision, quantization, runtime, context, batch and command line are recoverable.

This prevents an anecdote from becoming a permanent architectural coefficient.

---

## 6. The central correction: aggregate physical bandwidth != one-token effective bandwidth

A useful symbol is:

```text
B_physical_total = sum of all memory-interface bandwidth physically present
B_eff_token      = memory bandwidth that can actually serve the SAME token concurrently
```

They can differ by an order of magnitude.

For each serial model layer/stage `l`:

```text
T_layer[l] >= max(
    T_weight_critical[l],
    T_compute_critical[l]
) + T_collective[l] + T_misc[l]
```

Then:

```text
T_token ~= sum over serial layer boundaries of T_layer[l]
          + final head/sampling/state overhead

TPS = 1 / T_token
```

For MoE expert parallelism, selected experts inside one layer can run concurrently. The critical expert time is closer to the **slowest selected owner/group on the critical path** than to the sum of every GPU/node in the machine.

This is the same principle already captured in `docs/kimi-k3-ddr-cluster`:

> Pure pipeline parallelism solves capacity and aggregate request throughput, but does not automatically multiply single-sequence decode speed.

---

# 7. 20 × R920 with 6 × RTX 3060 12 GB each

## 7.1 Physical inventory

```text
20 servers × 6 GPUs = 120 RTX 3060
120 × 12 GB         = 1,440 GB raw VRAM
120 × 360 GB/s      = 43.2 TB/s aggregate physical VRAM bandwidth
```

The `360 GB/s` planning value is the RTX-3060-class bandwidth already used by the Phase-6 ByteMyWave audit.

### Capacity consequence

The project working full K3 checkpoint size is approximately `1.56 TB`, therefore:

```text
1.44 TB raw VRAM < ~1.56 TB native checkpoint
```

and runtime/KV workspace also needs memory.

Therefore the native released checkpoint does **not** fit comfortably in 120 × 12 GB. A lower-bit/alternate K3 representation is required if the model is to be fully GPU-resident on this fleet.

This is a capacity statement only.

---

## 7.2 Why `43.2 TB/s / 52 GB = 830 tok/s` is not a valid end-to-end prediction

That division assumes that all 120 GPU memory systems can contribute usefully and concurrently to every active-weight read on the same token with zero communication/synchronization penalty.

A simple topology such as:

```text
inside each R920: 6-way GPU group
between R920s:    pipeline/stage partition
```

has a very different critical path.

Per six-GPU stage:

```text
6 × 360 GB/s = 2.16 TB/s physical local VRAM bandwidth
```

Pure memory rooflines for that stage-equivalent group are:

```text
2,160 / 52 ~= 41.5 weight-path tok/s equivalent
2,160 / 58 ~= 37.2 screening-weight tok/s equivalent
```

These are still **rooflines**, not K3 decode predictions.

If the model is partitioned as 20 serial pipeline stages, all 20 stages can be busy on different requests, but one sequence still traverses serial layer boundaries. Pipeline filling raises aggregate serving throughput much more naturally than it raises one user’s decode rate.

---

## 7.3 Why the 120-GPU fleet may still be useful for single-stream speed

The useful parallelism is not limited to pipeline parallelism.

K3 provides:

```text
896 routed experts
16 selected experts/token
```

so a runtime can attempt combinations of:

- expert parallelism;
- tensor/row parallelism inside a layer;
- pipeline parallelism between stage groups;
- data-parallel attention where appropriate;
- fused quantized GEMV/GEMM kernels;
- communication/compute overlap;
- speculative decoding where a compatible draft/verification path exists.

The 80×5090 K3 demonstration proves that large consumer-GPU fleets can do more than hold the model. The 8×H20 llama.cpp issue simultaneously shows that **split mode matters**: layer-only placement can leave per-layer parallelism unused.

Therefore the correct ByteMyWave statement for `120×3060` is:

> **Capacity is plausible with an appropriate lower-bit representation; raw aggregate bandwidth is enormous; actual single-stream K3 tok/s is currently unknown and must be measured from a real six-GPU stage plus inter-node trace.**

The earlier conversational `~10–20 tok/s` estimate should **not** be treated as a project result. It was not backed by a measured RTX-3060 K3 layer path.

Likewise, `100 tok/s` should not be claimed or rejected before the actual stage/collective measurements exist.

---

# 8. Transit-304 on the same ruler

Transit is not conventional host-DDR offload. The current architecture is:

```text
R920 host/orchestrator
       |
PCIe switch/fan-out
       |
38 active memory-compute tiles
       |
8 independent local DDR3 channels/tile
       |
resident weights + local compute + local reduction
```

The host is intended to move activations/commands/results, not all selected expert weights.

## 8.1 Existing canonical Q4-equivalent weight-path roofline

For 304 independent x64 channels:

```text
DDR3-1600: 304 × 12.80 = 3.891 TB/s
DDR3-1866: 304 × 14.93 = 4.539 TB/s
DDR3-2133: 304 × 17.07 = 5.189 TB/s
```

Against the project’s `52 GB/token` four-bit lower bound:

```text
DDR3-1600: 3,891 / 52 ~= 74.8 weight-path tok/s
DDR3-1866: 4,539 / 52 ~= 87.3 weight-path tok/s
DDR3-2133: 5,189 / 52 ~= 99.8 weight-path tok/s
```

This reproduces the existing source-of-record calculation in `docs/10-KIMI-K3-TARGET.md`.

**The ~99.8 number remains valid as an idealized Q4-equivalent weight-path sizing target. It has not been disproved. It is still not end-to-end K3 throughput.**

## 8.2 Same channels under the 58 GB/token screening model

If the rough checkpoint-average screening value is used instead:

```text
DDR3-1600: 3,891 / 58 ~= 67.1 tok/s ideal weight-path screen
DDR3-1866: 4,539 / 58 ~= 78.3 tok/s ideal weight-path screen
DDR3-2133: 5,189 / 58 ~= 89.5 tok/s ideal weight-path screen
```

This does **not** replace the 52 GB lower-bound target. It shows sensitivity to the assumed exact bytes/token.

The real answer needs the exact K3 tensor trace and stored Transit format.

---

## 8.3 Why Transit may expose its aggregate bandwidth differently from a layer-pipeline GPU cluster

Transit’s intended concurrency is spatial inside a routed expert operation:

```text
router selects experts
 -> selected tile/channel owners read resident weights concurrently
 -> local compute consumes those streams
 -> local reductions shrink the data
 -> small outputs return upward
```

Therefore summing independent local DDR channels can be physically meaningful **only if**:

1. the selected expert/tensor placement activates enough channels concurrently for one layer/token;
2. local compute keeps up with each DDR stream;
3. selected work is balanced rather than concentrated on a few tiles;
4. reductions do not serialize the critical path;
5. MXFP4/MXFP8 or the validated Transit representation is implemented correctly;
6. attention/shared/non-expert work does not become the dominant serial bottleneck.

Those conditions are exactly what the physical tile and K3-layer milestones must prove.

---

# 9. The apples-to-apples rule for both architectures

The same questions must be asked of Transit and the 120-GPU rig:

| Question | Transit | 120 × RTX 3060 |
|---|---|---|
| Full/quantized model fits? | Capacity target says yes with enough tile DDR | Needs lower-bit representation; raw VRAM = 1.44 TB |
| Weights stationary on hot path? | Yes, by architecture | Yes if the chosen quantized model fully fits VRAM |
| Same-token parallel work? | Selected tile/channel compute engines | TP/EP GPU groups |
| Serial boundaries? | 93 model layers + non-expert path | 93 model layers + PP/TP/EP boundaries |
| Main local bandwidth | DDR3 channels near compute | GDDR6 per GPU |
| Main fabric traffic | activations/commands/reduced results | activations/collectives/PP handoffs |
| Exact K3 numerical path proven? | Not yet; INT4×INT8 proof exists, MXFP bridge pending | Depends on chosen runtime/quant; Ampere K3 path must be tested |
| End-to-end K3 tok/s measured? | No | No for proposed 120×3060 system |

The project should reject both of these invalid shortcuts:

```text
304 × DDR bandwidth / 52 GB = guaranteed end-to-end TPS
```

and:

```text
120 × GPU bandwidth / 52 GB = guaranteed end-to-end TPS
```

The numerator must be **effective same-token critical-path bandwidth**, not merely installed interface bandwidth.

---

# 10. Historical ByteMyWave host result: useful reference, not a Transit efficiency coefficient

The measured host bitplane proof on the laptop reached approximately:

```text
raw DDR5 path                   ~50.015 GB/s
V3 Q4-equivalent consumed       ~26.836 GB/s
ratio                           ~53.7%
```

This proves a real bitplane compute engine can fail to consume all raw memory bandwidth because compute itself becomes limiting.

It does **not** prove that an FPGA Transit tile will run at 53.7% efficiency. The final tile may be much better or worse depending on:

- DDR controller efficiency;
- popcount/low-bit lane count;
- clock rate;
- MXFP scale path;
- expert dimensions/layout;
- bank/row scheduling;
- activation and reduction pipeline.

The correct use of the host result is as a warning:

> **Do not assume raw DDR bandwidth becomes useful weight-processing bandwidth automatically. Measure both counters.**

---

# 11. What the real-world results teach us

### Lesson A — capacity is necessary, not sufficient

The full model fitting in VRAM avoids catastrophic CPU/RAM offload, but still leaves per-layer critical-path bandwidth, kernels and collectives.

### Lesson B — consumer GPU clusters can be genuinely fast

The 80×5090 K3 result disproves the idea that a large consumer-GPU rig only solves capacity.

### Lesson C — software can change the result by multiples

The reported 5090 fleet GLM optimization history and vLLM’s K3-specific kernel/speculative work show that runtime design is a first-class hardware multiplier.

### Lesson D — layer split and tensor/expert split are not interchangeable

The H20 llama.cpp report directly motivates measuring row/tensor/expert parallelism instead of assuming a layer pipeline is the best single-stream topology.

### Lesson E — Transit’s 304-channel idea must earn its aggregate bandwidth physically

Transit has a legitimate architectural reason to aggregate many local memory channels: compute is placed beside resident weights. But a channel count is only useful if a real tile consumes the bandwidth and real K3 expert placement exposes concurrency.

---

# 12. Current verdict — deliberately narrower than the conversation

## Transit-304

Proven today:

```text
~99.8 Q4-equivalent weight-path tok/s ideal sizing roofline at DDR3-2133
exact signed INT4×INT8 host bitplane arithmetic
measured host/kernel evidence
architecture for resident local weights + local compute
```

Not proven today:

```text
MXFP4/MXFP8 final tile kernel
8-channel physical final tile
304-channel physical fabric
end-to-end K3 tok/s
```

Therefore:

> **Keep ~99.8 as a category-4 weight-path target; do not publish it as category-1 K3 performance.**

## 20 × R920 / 120 × RTX 3060

Known today:

```text
1.44 TB raw VRAM
43.2 TB/s aggregate installed VRAM bandwidth
native ~1.56 TB checkpoint does not fit with runtime headroom
lower-bit K3 residency is conceptually possible
consumer-GPU K3 clustering is proven in the world at 80×5090 scale
```

Unknown today:

```text
real K3 quant chosen for Ampere
six-GPU one-layer decode latency
best TP/EP split on RTX 3060
P2P/NCCL behavior in the exact R920 topology
inter-node collective latency
full 20-node single-stream decode rate
```

Therefore:

> **Do not preserve the earlier 10–20 tok/s guess as a result. Do not claim 100 tok/s either. Benchmark the six-GPU stage and two-node fabric first.**

---

# 13. Procurement consequence

No bulk purchase should be justified from capacity or aggregate-bandwidth arithmetic alone.

Before buying 120 RTX 3060s:

```text
1. validate one R920 with 6 GPUs;
2. measure real K3 one-layer / one-expert decode work;
3. measure TP/row/expert split if the runtime supports it;
4. measure GPU P2P/NCCL topology and PCIe contention;
5. connect two representative nodes and replay K3’s layer cadence;
6. only then extrapolate to 20 nodes.
```

Before buying 38 final Transit tiles:

```text
1. prove one physical local-DDR compute tile;
2. measure sustained model-shaped DDR payload;
3. measure useful Gweights/s / expert outputs/s;
4. solve the K3 numerical-format bridge;
5. run one real K3 expert/shard;
6. prove multi-tile concurrency before scaling to 38.
```

The dedicated benchmark protocol is documented in:

`docs/17-K3-PARALLELISM-BENCHMARK-PLAN.md`

---

# 14. Decision metric after both prototypes exist

The final comparison should not be “how much RAM/VRAM does it contain?” It should be:

```text
single-stream decode tok/s
p50/p95 inter-token latency
prefill tok/s
aggregate serving throughput
watts at steady decode
acquisition cost
cost / measured tok/s
watts / measured tok/s
failure/recovery behavior
model-quality delta from quantization
```

The winning architecture may be Transit, consumer GPUs, or a hybrid. The project should let the measurements decide.

---

# 15. Source ledger

## Project-local current-state sources

- `CURRENT-ARCHITECTURE.md`
- `AGENTS.md`
- `docs/REPOSITORY-MAP.md`
- PR #10 — `transit-ddr3-architecture`
- PR #11 — `docs/kimi-k3-ddr-cluster`
- PR #9 — `research/heterogeneous-moe-kimi-v1`

## External sources checked 2026-08-17

- Moonshot Kimi K3: https://github.com/MoonshotAI/Kimi-K3
- vLLM K3 performance/implementation: https://vllm.ai/blog/2026-07-27-k3
- 80×RTX5090 K3 demo summary: https://www.corespan.ai/resources/blog/frontier-intelligence-on-gaming-silicon
- llama.cpp K3 row/tensor split request and 8×H20 layer-split report: https://github.com/ggml-org/llama.cpp/issues/26365
- PipeNetwork K3-derived M3 Ultra benchmark/correction: https://huggingface.co/pipenetwork/Kimi-K3-REAP73-MLX-mxfp4-q8/blob/main/README.md

---

## Final rule frozen by this note

> **For every future K3 architecture, report installed bandwidth, effective same-token bandwidth, compute, collective latency, numerical format and serial layer time separately. Never convert an aggregate interface sum directly into end-to-end tok/s.**
