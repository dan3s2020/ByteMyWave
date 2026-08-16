# 09 — Kimi K3 Throughput Model and Proof Conditions

## Why this file exists

It is easy to prove that a 1.56 TB checkpoint can be split across enough RAM. It is **not** honest to turn that capacity proof into an invented “K3 will run at X tokens/s” claim.

This document defines exactly what can be calculated now, what is only an upper bound, and what measurements will convert the design into a real token/s proof.

The central rule is:

> **Capacity is arithmetic. Throughput is a benchmarked property of the complete execution path.**

TensorWave will therefore publish a token/s number only after the relevant hardware and runtime pass the acceptance tests below.

---

# 1. Known model quantities

From the official Kimi K3 release:

```text
P_total  = 2.8e12 total parameters
P_active = 104e9 activated parameters per token
L        = 93 layers
E        = 896 routed experts
K        = 16 selected experts per token
```

The official checkpoint repository currently reports about:

```text
S_checkpoint ~= 1.56e12 bytes
```

K3 uses MXFP4 weights / MXFP8 activations as its headline quantization, while the actual config excludes some components from the 4-bit linear-weight rule.

Primary sources:

- https://github.com/MoonshotAI/Kimi-K3
- https://huggingface.co/moonshotai/Kimi-K3/tree/main
- https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json

---

# 2. First lower bound: active compressed weight traffic

If every one of the 104B activated parameters were a four-bit weight read exactly once from DRAM for batch-1 decode:

```text
W_4bit_min
= 104e9 parameters × 4 bits / 8
= 52e9 bytes
= 52 GB/token
```

This is an **optimistic lower bound**, not a complete K3 traffic measurement, because:

- not every relevant tensor is stored as 4-bit MXFP4;
- scales/metadata must be read;
- shared experts and attention paths have different representations;
- activations and state generate additional traffic;
- CPU kernels can reread data due to blocking/cache behavior;
- multimodal components may be active for multimodal requests.

A second rough estimate can be derived from the released checkpoint's average storage ratio:

```text
average checkpoint bytes/parameter
~= 1.56e12 / 2.8e12
~= 0.557 bytes/parameter

W_storage_ratio
~= 104e9 × 0.557
~= 58 GB/token
```

This **~58 GB/token** value is useful as an engineering screening model, but it assumes that the active parameter subset has approximately the same average storage density as the full checkpoint. That must be replaced by an exact tensor-level byte accounting from `tw-k3-inspect`.

For planning, this document uses:

```text
W_screen = 58 GB/token
```

and explicitly labels all results derived from it as modeled, not measured.

---

# 3. First compute estimate

For a matrix-vector path, one active weight normally contributes approximately one multiply and one add:

```text
C_linear ~= 2 × P_active
          ~= 208e9 FLOP/token
          ~= 208 GFLOP/token
```

This is again a screening estimate rather than a full operation count. It does not fully account for:

- dequantization/unpack;
- router work;
- normalization and nonlinearities;
- KDA/MLA state updates;
- softmax/attention details where applicable;
- transport packing/unpacking;
- scheduling and synchronization.

The actual CPU bottleneck may be integer/bit manipulation or memory latency rather than nominal floating-point throughput.

---

# 4. Roofline upper bound

Let:

```text
B_eff = effective simultaneous DRAM weight-stream bandwidth available to work on ONE token [GB/s]
F_eff = effective useful kernel throughput for ONE token [GFLOP/s]
W_eff = actual bytes of weight/state traffic per token [GB/token]
C_eff = actual useful operation count per token [GFLOP/token]
```

Ignoring network/serial overhead for a moment, the ideal roofline is:

```text
TPS_memory <= B_eff / W_eff
TPS_compute <= F_eff / C_eff

TPS <= min(TPS_memory, TPS_compute)
```

Using the screening values:

```text
TPS_memory <= B_eff / 58
TPS_compute <= F_eff / 208
```

These are **upper bounds**. Meeting them is necessary but not sufficient.

---

# 5. Screening thresholds for X tokens/s

The following table answers the question “what does the cluster have to be able to sustain before X token/s is even plausible?”

Two columns are shown:

- **ideal minimum** from the simple 58 GB / 208 GFLOP model;
- **screening target** with 25% bandwidth headroom and 50% compute headroom for early procurement decisions.

| Target decode rate | Ideal weight BW | Screening weight BW | Ideal compute | Screening compute |
|---:|---:|---:|---:|---:|
| 0.5 tok/s | 29 GB/s | 36.25 GB/s | 104 GFLOP/s | 156 GFLOP/s |
| **1 tok/s** | **58 GB/s** | **72.5 GB/s** | **208 GFLOP/s** | **312 GFLOP/s** |
| 2 tok/s | 116 GB/s | 145 GB/s | 416 GFLOP/s | 624 GFLOP/s |
| 5 tok/s | 290 GB/s | 362.5 GB/s | 1.04 TFLOP/s | 1.56 TFLOP/s |
| 10 tok/s | 580 GB/s | 725 GB/s | 2.08 TFLOP/s | 3.12 TFLOP/s |

Again: passing the “screening” row does **not** prove that rate. It only prevents buying hardware that is obviously orders of magnitude short before network and serial work are included.

---

# 6. Why pure layer pipeline does not aggregate bandwidth for one decode token

Assume 10 equal servers each own one tenth of the sequential layers. If every server can stream local weights at 20 GB/s, it is incorrect to say:

```text
10 × 20 = 200 GB/s -> 200 / 58 = 3.4 tok/s
```

if those ten stages execute strictly one after another for a single sequence.

In a pure layer pipeline, token latency is approximately:

```text
T_token ~= T_stage0 + T_stage1 + ... + T_stage9
```

If each stage reads one tenth of the active bytes at the same local bandwidth, total token time is still approximately:

```text
(58/10)/20 × 10
= 58/20
= 2.9 s/token
~= 0.34 tok/s
```

Pipeline parallelism solves **capacity** and can improve multi-request throughput through pipelining, but it does not magically multiply single-sequence decode bandwidth.

This is why the K3 architecture needs MoE **expert parallelism** if we want the same token to use multiple memory controllers concurrently.

---

# 7. Why K3's 16 selected experts can expose parallel memory bandwidth

For a MoE layer, the router selects 16 of 896 experts. If experts are distributed across N workers, those 16 expert operations can be executed by multiple workers at the same time.

Under a purely uniform random-placement thought experiment, the expected number of distinct workers touched by 16 selections is:

```text
E[workers] = N × (1 - (1 - 1/N)^16)
```

Examples:

```text
N = 5  -> ~4.83 distinct workers
N = 10 -> ~8.15 distinct workers
N = 20 -> ~11.20 distinct workers
```

This is **not a claim about K3's actual router distribution**. Real expert popularity can be skewed, and selected experts are not independent uniform samples. It simply demonstrates why 10–20 expert owners can, in principle, place many selected experts on simultaneously active memory controllers.

The implementation therefore needs:

- expert placement aware of observed routing frequency;
- per-node queues;
- expert load balancing / replication if a few experts become hot;
- metrics for `selected_experts_per_node` on real prompts.

vLLM's Expert Parallelism and Expert Parallel Load Balancing features independently reflect the same practical issue.

References:

- https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- https://docs.vllm.ai/en/stable/api/vllm/config/parallel/

---

# 8. Network model: bandwidth is not the only issue

K3's config reports:

```text
hidden_size = 7168
routed_expert_hidden_size / latent MoE dimension = 3584
selected experts = 16
```

As a rough order-of-magnitude example only, a BF16 latent vector of 3584 values is:

```text
3584 × 2 bytes ~= 7 KB
```

Dispatching one such vector to each of 16 selected experts would be on the order of:

```text
~112 KB outbound per MoE layer
```

with a similar order for returned expert data, before headers, padding, collectives or any exact K3-specific representation.

Across roughly 92 MoE layers, a naive illustrative total could therefore be in the tens of MB/token rather than tens of GB/token.

The conclusion is **not** “10 GbE is definitely enough.” The conclusion is:

> Expert-parallel transport can be dramatically smaller than moving weights, but **per-layer collective latency** can dominate because the exchange repeats through the model.

The exact packet/activation format must be taken from the real K3 forward implementation before the network budget is finalized.

Measurements required:

```text
one-way payload bandwidth
one-way small-message latency
all-to-all latency for selected expert fanout
p50/p95/p99 per-layer collective time
CPU time spent packing/unpacking
```

---

# 9. Full token latency equation

For layer `l`, define:

```text
T_weight[l]  = time for all required local weight streams on the critical path
T_compute[l] = useful compute/dequant time on the critical path
T_net[l]     = required inter-node communication/synchronization
T_misc[l]    = router, scheduler, state, barrier and software overhead
```

If weight access and compute overlap imperfectly, a useful lower-bound model is:

```text
T_layer[l] >= max(T_weight[l], T_compute[l]) + T_net[l] + T_misc[l]
```

The actual decode time is approximately:

```text
T_token = sum over serial layer boundaries of measured T_layer[l]
          + final head/sampling overhead

TPS = 1 / T_token
```

For parallel experts inside one layer, `T_weight[l]` is determined by the **slowest selected expert owner on the critical path**, not by a sum over every worker. That is where expert parallelism can turn multiple independent RAM buses into useful simultaneous bandwidth.

---

# 10. DDR4 H12DGQ-NT6 feasibility envelope

The Supermicro H12DGQ-NT6 supports two EPYC 7002/7003 sockets and 32 DDR4 DIMMs. AMD documents an EPYC 7262 as having:

```text
8 memory channels/socket
DDR4-3200
204.8 GB/s theoretical memory bandwidth/socket
```

Therefore:

```text
2 sockets/board -> 409.6 GB/s theoretical interface ceiling
10 boards       -> 4096 GB/s theoretical summed interface ceiling
```

Official sources:

- https://www.supermicro.com/en/products/motherboard/h12dgq-nt6
- https://www.amd.com/en/support/downloads/drivers.html/processors/epyc/epyc-7002-series/amd-epyc-7262.html

If one naively divided this theoretical sum by the 58 GB/token screening weight traffic:

```text
4096 / 58 ~= 70.6 tok/s
```

That number is **NOT a prediction**. It is intentionally documented as an impossible-to-use-as-a-benchmark ceiling because:

- no workload sustains 100% of every socket's theoretical interface bandwidth;
- the 16 selected experts may not occupy every socket simultaneously;
- attention/shared/dense work creates serial sections;
- CPU MXFP4 unpack/dequant may bottleneck long before DRAM;
- NUMA placement can waste bandwidth;
- the network introduces repeated synchronization;
- a cheap 8-core EPYC may not generate enough memory-level parallelism for every access pattern.

The useful inference from the ceiling is much weaker but still important:

```text
1 tok/s screening target = 72.5 GB/s
72.5 / 4096 ~= 1.77% of the summed theoretical socket bandwidth
```

Thus the DDR4 cluster has **large raw bandwidth headroom** for a 1 tok/s experiment. Whether TensorWave can expose that headroom to one K3 token is exactly what the expert-parallel benchmark must measure.

---

# 11. DDR3 R920 and DDR2 routes cannot receive a fabricated token/s estimate

For R920, DL785 G6, X4640 and X4600 M2, the purchased CPU model, DIMM rank/population, memory mode, NUMA configuration and NIC are not fixed yet.

In particular, Dell documents multiple R920 memory operating modes/speeds depending on DIMM population and SMI-2 configuration. It would be wrong to multiply “1600 MT/s” by 96 DIMMs and call the result bandwidth; DIMM count is not independent channel count.

Therefore these platforms must pass the same benchmark before we assign them a K3 token/s estimate.

Required measurements per NUMA node:

```text
STREAM copy/read/triad bandwidth
single-thread random/stream latency
all-core sustained read bandwidth
MXFP4 packed expert GEMV bandwidth
KDA/MLA representative kernel throughput
NUMA-local vs remote penalty
NIC round-trip and bulk throughput
```

Only then can we fill this table:

| Platform | Nodes | Measured effective K3 weight BW | Measured K3 kernel GFLOP/s | Distributed trace ms/token | Real K3 tok/s |
|---|---:|---:|---:|---:|---:|
| DL785 G6 | TBD | TBD | TBD | TBD | TBD |
| X4640 | TBD | TBD | TBD | TBD | TBD |
| R920 | TBD | TBD | TBD | TBD | TBD |
| H12DGQ-NT6/EPYC | TBD | TBD | TBD | TBD | TBD |

Until measured, `TBD` is the technically correct value.

---

# 12. The actual proof protocol for “K3 runs at X tok/s”

TensorWave will call rate `X` demonstrated only when all gates pass.

## Gate A — capacity and correctness

- full official checkpoint tensor inventory accounted for;
- full checkpoint resident in aggregate RAM for the target configuration;
- no expert pruning used to obtain the result;
- distributed forward matches reference within documented tolerance.

## Gate B — kernel microbench

Run the exact packed K3 tensor shapes through the custom CPU kernels and record:

```text
GB/s of compressed weights consumed
output vectors/s
cycles/weight
GFLOP/s equivalent
power
NUMA locality
```

## Gate C — network microbench

Reproduce K3's real selected-expert communication sizes and layer cadence over the actual switch/NIC topology.

Record p50/p95/p99.

## Gate D — distributed synthetic K3 trace

Before loading 1.56 TB of weights, run a trace with the same:

- 93 layer barriers;
- expert selection fanout;
- payload sizes;
- local memory bytes per expert;
- synchronization sequence.

If the trace cannot meet `1/X` seconds/token, full K3 will not either.

## Gate E — real full-model decode

Use a pinned prompt, fixed context, fixed sampling settings and a warm model.

Measure:

```text
TTFT
inter-token latency p50/p95/p99
decode tokens/s
per-node CPU utilization
per-node DRAM bandwidth
network bytes and latency
expert load balance
power draw
```

## Gate F — reproducibility

Run at least three independent trials after warm-up. Store:

- exact git commit;
- model revision;
- BIOS settings;
- CPU/DIMM/NIC inventory;
- kernel ISA path;
- raw logs.

Then and only then write:

```text
Kimi K3 full checkpoint
N nodes
M GB RAM
X.Y decode tokens/s
prompt/context = ...
```

---

# 13. Initial project performance target

The first meaningful target is:

```text
FULL K3, no expert pruning, batch=1
>= 1.0 decoded token/s after warm-up
```

This is not claimed as achieved. It is an **acceptance target**.

The screening conditions derived above are:

```text
~72.5 GB/s effective parallel weight stream
~312 GFLOP/s effective useful kernel capability
plus network/serial overhead small enough that total measured T_token <= 1.0 s
```

If the cluster's real distributed trace is, for example:

```text
T_token = 0.72 s
```

then the demonstrated rate is:

```text
X = 1 / 0.72 = 1.39 tok/s
```

If it is:

```text
T_token = 2.5 s
```

then:

```text
X = 0.4 tok/s
```

The architecture does not get to choose the answer. The measurement does.

---

# 14. What we can already demonstrate mathematically

Without owning the final hardware, the repository can already demonstrate three things:

### 14.1 Capacity

Any supported hardware set with comfortably more than ~1.56 TB of usable aggregate RAM can store the full released checkpoint. Examples documented elsewhere include 2.304–2.56 TB configurations.

### 14.2 Distribution correctness is possible in principle

The full tensor set can be partitioned across nodes, and MoE expert computation is inherently shardable. Existing inference systems implement pipeline, tensor and expert parallelism; TensorWave's novelty here is applying a custom CPU/cheap-memory runtime to old/commodity hardware, not inventing distributed MoE mathematics.

### 14.3 A falsifiable performance requirement

For the approximate 58 GB/token storage-ratio model, **1 tok/s cannot be memory-feasible unless the token's simultaneous critical-path memory system supplies roughly 58 GB/s ideal, and our current procurement screening threshold is 72.5 GB/s effective.**

That criterion gives us a way to reject hardware *before* buying a whole cluster, and a way to prove the final X with measurements instead of rhetoric.
