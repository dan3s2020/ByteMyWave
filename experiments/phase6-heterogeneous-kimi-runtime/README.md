# Phase 6 — Heterogeneous MoE runtime for Kimi K2.5 / K3

Date: **2026-08-15**

Status: **isolated research area; analytical proof + implementation plan**

This directory is intentionally separate from TensorWave Phases 1–5. It does not replace the existing CUDA Q4 streaming proof, feasibility simulator, or R920 hardware documentation.

The question studied here is:

> How can an R920-class machine with huge host RAM and small VRAM reach useful single-user decode speed on frontier-scale MoE models without pretending PCIe bandwidth is unlimited?

## Central result

For M=1 LLM decode, compression alone is not enough for very large active sets. TensorWave must also reduce the number of weight bytes that cross PCIe per output token.

The strongest concrete architecture found in the discussion is:

```text
large compressed model in NUMA-local host RAM
        |
        +--> routed experts stay beside RAM and execute on CPU NUMA engines
        |
        +--> small always-active/non-routed path stays resident on GPU where possible
        |
        +--> only compact activations/reductions cross CPU<->GPU
        |
        +--> optional GPU-owned experts / multiple GPUs later
```

The CPUs are therefore not merely memory controllers. For MoE decode they can become **expert engines**.

Nothing in this branch claims that an E7-4890 v2 actually reaches 5 or 10 tok/s. Instead, this branch derives the exact bandwidth/compute thresholds that the real CPU kernel must meet.

---

# 1. Hardware studied

Primary host:

```text
Dell PowerEdge R920
2 or 4 Xeon E7 v2 supported by platform
preferred TensorWave layout: 4 x E7-4890 v2
96 DDR3 ECC DIMM sockets
up to 6 TB system RAM
PCIe Gen3
```

Important E7-4890 v2 facts from Intel:

```text
15 cores / 30 threads
2.8 GHz base / 3.4 GHz turbo
4 memory channels
85 GB/s published max memory bandwidth
32 PCIe 3.0 lanes
4-socket scalability
Instruction Set Extensions: Intel AVX
```

Critical limitation:

```text
AVX only
no AVX2
no AVX-512
no AMX
```

This matters because modern heterogeneous MoE runtimes use newer vector/AMX kernels. The architecture is validated by prior art, but their throughput numbers cannot be copied to the R920.

R920 x16 electrical topology from Dell with four CPUs:

```text
CPU2 -> slot 4 x16, slot 5 x16
CPU3 -> slot 6 x16, slot 7 x16
CPU4 -> slot 8 x16, slot 9 x16
```

These are electrical links, **not six guaranteed internal RTX 3060 positions**. Mechanical fit, power and cooling remain separate hardware gates.

Phase-5 analytical H2D input remains:

```text
12 GB/s effective pinned host->GPU per PCIe x16 feed
```

It is an assumption until measured on the real server.

---

# 2. Model inputs

## Kimi K2.5

Moonshot publishes:

```text
1T total parameters
32B activated parameters/token
61 layers including 1 dense layer
60 MoE layers
hidden size 7168
MoE intermediate size 2048
384 routed experts
8 routed experts selected/token
1 shared expert
MLA
native INT4 release
```

For one routed SwiGLU expert:

```text
params/expert = gate + up + down
              = 3 * 7168 * 2048
              = 44,040,192
```

Selected routed parameters per token across all MoE layers:

```text
60 * 8 * 44,040,192
= 21,139,292,160
= 21.13929216B
```

Planning non-routed active remainder:

```text
32B - 21.13929216B
= 10.86070784B
```

The official 32B active number is rounded, so this remainder is a planning value. The real implementation must derive exact tensor bytes from the checkpoint through Weight Atlas.

## Kimi K3

Moonshot publishes:

```text
2.8T total parameters
104B activated parameters/token
93 layers including 1 dense layer
92 MoE layers
69 KDA + 24 Gated MLA
hidden size 7168
MoE intermediate size 3072
896 experts
16 selected/token
2 shared experts
MXFP4 weights / MXFP8 activations
1M context
```

Derived routed active parameters:

```text
params/expert = 3 * 7168 * 3072
              = 66,060,288

92 * 16 * 66,060,288
= 97,240,743,936
= 97.240743936B routed active/token
```

Planning non-routed active remainder:

```text
104B - 97.240743936B
= 6.759256064B
```

---

# 3. Capacity

Current TensorWave Phase-3 format:

```text
Q4_SYM_G32_F32S
32 weights -> 16 packed int4 bytes + 4-byte FP32 scale
20 bytes/group
0.625 bytes/weight
```

K2.5 full-model host storage:

```text
1T * 0.625 B = 625 GB decimal
```

Therefore 1 TiB RAM passes the raw capacity gate.

K3 full-model host storage:

```text
2.8T * 0.625 B = 1.75 TB decimal
```

2 TiB RAM is about 2.199 TB decimal, therefore 2 TiB passes the current TensorWave Q4 capacity-only gate. 1 TiB does not.

Capacity does not imply speed.

---

# 4. Current single-3060 streaming baseline

For K2.5 stream-all with current Q4:

```text
32B * 0.625 B = 20 GB/token
20 / 12 GB/s = 1.6667 s/token
=> 0.600 tok/s bandwidth ceiling
```

Phase-5 compute assumption:

```text
2 * 32B = 64 GFLOP/token
64 GFLOP / 10 TFLOP/s = 6.4 ms
```

So K2.5 is overwhelmingly host-transfer-bound in the current stream-all design.

For K3:

```text
104B * 0.625 B = 65 GB/token
65 / 12 = 5.4167 s/token
=> 0.1846 tok/s
```

This is the strict current-Q4 stream-all analytical ceiling for one 12 GB/s feed, before attention/KV/dequant/other overhead.

---

# 5. Deterministic static residency

Instead of inventing a cache-hit ratio, first place the always-active/non-routed part deterministically.

K2.5 planning non-routed active weights in current Q4:

```text
10.86070784B * 0.625
= 6.7879424 GB decimal
≈ 6.32 GiB
```

A 12 GiB RTX 3060 can, by capacity arithmetic, reserve 4 GiB for runtime/KV/workspace and still fit this derived non-routed Q4 set in the remaining 8 GiB weight budget.

Then only routed expert weights are streamed:

```text
21.13929216B * 0.625
= 13.2120576 GB/token

13.2120576 / 12
= 1.1010 s/token
=> 0.908 tok/s
```

This is a concrete improvement over 0.600 tok/s without assuming probabilistic reuse.

For K3, the derived non-routed remainder is much smaller relative to routed work, so static residency helps little:

```text
97.240743936B * 0.625
= 60.77546496 GB routed/token
=> ~0.197 tok/s at 12 GB/s
```

---

# 6. Compression research

Q4 is a baseline, not a final format.

Candidate packed formats modeled in this phase:

```text
TW Q4 G32 + FP32 scale = 0.625 B/weight
Q3 G64 + FP16 scale    = 0.40625 B/weight
Q2 G32 + FP32 scale    = 0.375 B/weight
Q2 G64 + FP16 scale    = 0.28125 B/weight
ideal 1-bit stress test = 0.125 B/weight
```

K2.5 routed-only streaming after non-routed residency:

```text
Q4:       13.212 GB/token -> 0.908 tok/s
Q2 G64:    5.945 GB/token -> 2.018 tok/s
ideal 1b:  2.642 GB/token -> 4.541 tok/s
```

Even an ideal 1-bit routed representation remains below 5 tok/s on one 12 GB/s feed.

For K3, if **every active weight** still crosses one 12 GB/s link, the effective representation needed is:

```text
5 tok/s:
12/5 = 2.4 GB/token
2.4*8/104 = 0.1846 bit/active-weight

10 tok/s:
12/10 = 1.2 GB/token
1.2*8/104 = 0.0923 bit/active-weight
```

Therefore compression alone cannot solve K3 5–10 tok/s on one PCIe 3.0 x16 link.

Research directions retained:

```text
Q3
Q2
mixed Q2/Q3/Q4 per tensor/tile
Q2 + sparse high-precision residual/outliers
vector/additive quantization
sub-2-bit only when quality measurements justify it
```

Weight Atlas should eventually carry:

```text
tensor_id
compression_type
bits_per_weight
scale/codebook location
quality/sensitivity score
expected reuse
cache_priority
preferred NUMA node
execution_engine = CPU|GPU
preferred GPU
```

---

# 7. Main new architecture: CPU routed experts

The core observation is that MoE expert weights are huge while the token activation vector is small.

Instead of moving tens of GB of selected expert weights to the GPU every token:

```text
RAM weights -> PCIe -> GPU expert GEMV
```

move the much smaller activation to the NUMA-local CPU engine that already owns those experts:

```text
GPU resident path / router
        |
        +--> activation -> NUMA0 expert shard
        +--> activation -> NUMA1 expert shard
        +--> activation -> NUMA2 expert shard
        +--> activation -> NUMA3 expert shard
                           |
                    partial outputs
                           |
                           v
                         GPU
```

The large routed tensors stay in host RAM.

This attacks the correct variable: **fresh bytes crossing PCIe per token**.

---

# 8. Exact K2.5 CPU thresholds

Routed Q4 bytes/token:

```text
21.13929216B * 0.625
= 13.2120576 GB/token
```

## Four sockets

5 tok/s memory requirement:

```text
13.2120576 * 5 / 4
= 16.515072 GB/s/socket
```

10 tok/s:

```text
33.030144 GB/s/socket
```

Relative to Intel's published 85 GB/s max/socket:

```text
5 tok/s  -> 19.43%
10 tok/s -> 38.86%
```

Logical routed-expert arithmetic:

```text
2 * 21.13929216B
= 42.27858432 GFLOP/token total
```

Per socket:

```text
5 tok/s  -> 52.8482304 GFLOP/s/socket
10 tok/s -> 105.6964608 GFLOP/s/socket
```

Selected weight processing rate/socket:

```text
5 tok/s  -> 26.4241152 Gweights/s
10 tok/s -> 52.8482304 Gweights/s
```

At base clock:

```text
15 cores * 2.8 GHz = 42 billion aggregate core cycles/s/socket
```

Raw aggregate cycle budget:

```text
5 tok/s  -> ~1.589 core-cycles/weight
10 tok/s -> ~0.795 core-cycles/weight
```

This is the critical empirical test. E7-4890 v2 is AVX-only, so low-bit unpack + scale + dot-product efficiency may become the real blocker.

## Two sockets

The exact work over two sockets doubles the per-socket requirement:

```text
5 tok/s:
33.030144 GB/s/socket
105.6964608 GFLOP/s/socket

10 tok/s:
66.060288 GB/s/socket
211.3929216 GFLOP/s/socket
```

Therefore **4 CPUs are materially better than 2 for this architecture**.

## Q2 memory side

Q2 G64 + FP16 scale:

```text
18 bytes / 64 weights
= 0.28125 B/weight
```

K2.5 routed bytes/token:

```text
21.13929216B * 0.28125
= 5.94542592 GB/token
```

Four sockets:

```text
5 tok/s  -> 7.4317824 GB/s/socket
10 tok/s -> 14.8635648 GB/s/socket
```

The logical matrix arithmetic does not disappear; unpack/dequant work is added. Q2 makes memory easier and kernel execution more important.

---

# 9. CPU<->GPU activation volume vs latency

K2.5 hidden size = 7168.

One BF16 hidden vector:

```text
7168 * 2 = 14,336 bytes
```

A deliberately conservative volume upper bound duplicates one vector to all four sockets and returns one partial vector from all four sockets for all 60 MoE layers:

```text
2 directions * 4 sockets * 14,336 * 60
= 6.88128 MB/token
```

At 5 tok/s:

```text
~34.4 MB/s = 0.0344 GB/s
```

So activation **volume** is tiny compared with 13.2 GB/token of routed Q4 weights.

But there are 60 sequential MoE layer dependencies. The important unknown becomes **per-layer handoff/synchronization latency**.

The simulator therefore includes handoff sensitivity rather than assuming it is free.

---

# 10. External validation and the AVX gap

Moonshot's K2.5 deployment guide explicitly includes **KTransformers + SGLang CPU+GPU heterogeneous inference**.

KTransformers describes offloading MoE experts to CPU. Its Kimi K2 documentation reports approximately:

```text
~10 tok/s: single-socket CPU + one consumer GPU, Q4
~14 tok/s: dual-socket with NUMA optimization
```

This is strong evidence that the heterogeneous architecture is real.

It is **not** evidence that R920 reaches those numbers.

KTransformers' modern AVX2 CPU backend requires AVX2 + FMA; high-performance K2.5 deployments use even newer AVX-512/AMX-class CPUs. E7-4890 v2 has only Intel AVX.

Therefore the actual R920 acceptance test is:

```text
Can our AVX/SSE low-bit expert kernel meet at least:

5 tok/s target:
>= 26.424 Gweights/s/socket
>= 16.515 GB/s/socket Q4 selected-weight reads

10 tok/s target:
>= 52.848 Gweights/s/socket
>= 33.030 GB/s/socket Q4 selected-weight reads
```

If not, stop pretending orchestration can fix it: more experts must move to GPU ownership, more GPUs must be added, or the host CPU platform must change.

---

# 11. Multiple GPUs

Multiple GPUs help single-request latency only when real model/expert sharding exists. Replicated workers only improve aggregate request throughput.

For independent local PCIe feeds, ideal host bandwidth scales as:

```text
aggregate H2D ~= GPU_count * 12 GB/s
```

until RAM/NUMA/synchronization becomes the next bottleneck.

Preferred topology conceptually:

```text
CPU2 local RAM -> GPU0 / GPU1
CPU3 local RAM -> GPU2 / GPU3
CPU4 local RAM -> GPU4 / GPU5
CPU1 -> orchestration and/or CPU expert shard
```

Do not route giant weights through GPU0 and then copy them GPU-to-GPU.

RTX 3060 has no NVLink. PCIe 3.0 x16 theoretical one-direction payload is ~15.75 GB/s. Real P2P is topology/driver dependent and must be measured.

GPU<->GPU traffic should be compact activations, routing and reductions, not tens of GB of weights.

---

# 12. Faster GPU does not automatically fix host streaming

A real RTX 5090 has 32 GB GDDR7 and NVIDIA publishes 1792 GB/s VRAM bandwidth. On the same R920, however, fresh host weights still enter through PCIe Gen3.

Therefore the correct decode bound is not `GPU FLOPS only`.

At minimum:

```text
T_token >= max(
  host->GPU weight transfer,
  GPU VRAM weight reads,
  low-bit decode/dequant,
  GPU GEMV/GEMM,
  attention/KV/recurrent state,
  CPU expert execution,
  CPU<->GPU handoff latency,
  multi-GPU synchronization
)
```

A faster GPU matters enormously once weights are resident/reused, and for prefill/image/video where arithmetic intensity is much higher. It can matter very little for a pure M=1 stream that remains host-bandwidth-bound.

---

# 13. Correction to the hypothetical 2 TB-VRAM 3060 thought experiment

A compute-only calculation for K3 gives:

```text
104B active * 2 FLOP = 208 GFLOP/token
10 TFLOP/s / 208 GFLOP ~= 48.1 tok/s
```

That is not the final bound because resident weights still have to be read from GPU memory.

Using a 360 GB/s RTX-3060-class VRAM-bandwidth input and current TensorWave Q4:

```text
104B * 0.625 = 65 GB active reads/token
360 / 65 = 5.54 tok/s VRAM-bandwidth ceiling
```

So the simplified resident roofline is:

```text
min(48.1 compute, 5.54 VRAM BW)
= ~5.54 tok/s
```

If the active representation were ideal 4-bit at 52 GB/token, VRAM bandwidth would give ~6.9 tok/s.

This shows the bottleneck sequence clearly:

```text
small VRAM -> host PCIe wall
huge resident VRAM -> GPU memory-bandwidth wall
high reuse/batching -> compute can become dominant
```

---

# 14. K3 boundaries

For K3 current Q4 and one GPU:

```text
stream-all: ~0.185 tok/s
static non-routed residency: ~0.197 tok/s
Q2 G64 stream-all: ~0.410 tok/s
Q2 G64 routed-only: ~0.439 tok/s
ideal 1-bit stream-all: ~0.923 tok/s
```

CPU-only routed-expert execution is far harder for K3 than K2.5.

Four sockets, K3 routed Q4, 5 tok/s would require approximately:

```text
75.97 GB/s/socket
243.10 GFLOP/s/socket
```

At 10 tok/s:

```text
151.94 GB/s/socket
486.20 GFLOP/s/socket
```

The 10 tok/s Q4 memory requirement alone exceeds the published 85 GB/s/socket maximum. K3 therefore likely needs substantial GPU expert ownership and/or multiple GPUs even if K2.5 validates the CPU-expert design.

---

# 15. Speculative decoding

vLLM publishes an open DSpark speculator for K3 and reports approximately:

```text
2.61 accepted tokens/verification step on high-entropy tasks
4.73 accepted tokens/verification step on low-entropy/coding tasks
3.14x single-user speedup on their GB300 production setup
```

TensorWave should treat this as an **output-token multiplier after target-pass cost is modeled**, not as a replacement for bandwidth accounting.

These values are included only as sensitivity factors in the simulator. They are not R920 measurements.

---

# 16. GPU low-bit path

Phase-3 currently does:

```text
compressed Q4 tile
-> GPU dequant
-> reusable full FP16 tile
-> GEMM
```

A stronger future path remains:

```text
compressed fragment
-> decode only the needed fragment
-> registers/shared memory
-> direct GEMV/GEMM/MMA
```

Avoiding full FP16 materialization reduces VRAM workspace and VRAM traffic.

Weights should remain already compressed in host RAM. Do not recompress the same model every token. CPU/RAM should prepare addresses/queues while the GPU computes the current tile and PCIe asynchronously transfers the next one.

For MoE, prefetch should become router-aware.

---

# 17. Runtime/frontend conclusion

TensorWave should remain a standalone engine/backend.

```text
ComfyUI custom nodes -----+
OpenAI-compatible API ----+
CLI ----------------------+--> TensorWave runtime --> NUMA RAM + CPUs + GPUs
other frontends ----------+
```

ComfyUI can be a thin plugin/front-end, but must not own TensorWave's memory placement, compression, NUMA policy, VRAM cache/ring or prefetch scheduler.

The same engine can therefore later serve LLM, image and video workloads.

---

# 18. Implementation order

## P0 — exact checkpoint census

Weight Atlas must classify/count real K2.5/K3 tensors:

```text
attention / recurrent
router
routed expert
shared expert
dense MLP
embedding / LM head
vision encoder
other
```

Replace rounded active-remainder planning values with exact bytes.

## P1 — resident-set planner

Given 12 GiB VRAM:

```text
reserve runtime/KV/workspace
place always-active weights
place hot compressed cache if space remains
leave transient ring
```

## P2 — CPU ExpertEngine interface

```text
ExpertEngine.run(layer_id, expert_ids, activation)
```

One engine per NUMA node, with expert weights allocated local to that node.

## P3 — AVX-era low-bit expert benchmark

Measure exact K2.5 pass thresholds above before integrating the whole model.

## P4 — CPU/GPU handoff benchmark

Measure:

```text
GPU -> host activation
NUMA dispatch
expert execution
partial reduction
host -> GPU result
GPU continuation
```

Record p50/p95/p99 per MoE layer.

## P5 — adaptive compression

Benchmark Q4/Q3/Q2/mixed formats for both:

```text
quality loss
weights/s
GB/s
unpack/dequant cost
```

## P6 — topology-aware expert placement

Add metadata:

```text
preferred_host_numa
execution_engine
preferred_gpu
expert_id
quant_format
cache_priority
hotness
```

## P7 — multi-GPU expert ownership

Only after one heterogeneous path is measured. Prefer whole expert/layer ownership before fine-grained tensor parallelism.

## P8 — speculative decode adapter

Integrate DSpark or another compatible speculator only after target-pass correctness.

---

# 19. Falsification gates

Revise/reject the R920 heterogeneous hypothesis if:

```text
1. AVX-only expert GEMV cannot approach the 5 tok/s K2.5 threshold.
2. Real local DDR3 selected-weight bandwidth is below the requirement.
3. 60 sequential CPU/GPU handoffs consume most of the 100-200 ms/token budget.
4. NUMA remote traffic dominates despite placement controls.
5. Q2/Q3 quality loss is unacceptable.
6. 12 GiB cannot hold the proposed static set plus KV/workspace.
7. Multi-GPU topology forces large cross-socket or staged transfers.
```

---

# 20. Bottom line

For `R920 + 1 TiB + 1x RTX 3060 12 GB + Kimi K2.5`:

```text
current TensorWave Q4 stream-all:       ~0.600 tok/s ceiling
static non-routed Q4 residency:         ~0.908 tok/s ceiling
static residency + Q2 G64 routed:       ~2.018 tok/s ceiling
```

Therefore 5–10 tok/s is **not** a one-GPU weight-streaming problem.

The concrete Phase-6 hypothesis is:

```text
4 NUMA CPU expert engines
+
GPU-resident non-routed path
+
compressed local expert store
+
small activation transfers
+
strict NUMA placement
+
measured handoff latency
+
GPU expert ownership / more GPUs when CPU ISA is insufficient
```

Exact Q4 gates for four E7-4890 v2 sockets:

```text
5 tok/s:
16.515 GB/s/socket selected-weight reads
52.848 GFLOP/s/socket equivalent
26.424 Gweights/s/socket

10 tok/s:
33.030 GB/s/socket selected-weight reads
105.696 GFLOP/s/socket equivalent
52.848 Gweights/s/socket
```

Those measurements decide whether the CPU idea works on this exact platform.