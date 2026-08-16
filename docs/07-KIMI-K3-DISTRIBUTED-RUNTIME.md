# 07 — Kimi K3 Distributed Runtime

## Purpose

This document defines the second TensorWave execution track: **run the complete Kimi K3 checkpoint across a cluster of inexpensive, high-DIMM-count servers**, where no individual server must hold the whole model.

This is deliberately different from the original TensorWave track, where one large host-RAM pool feeds a small-VRAM GPU. The two tracks share the same core idea — persistent weights live in cheap memory and compute sees only the data needed for the current operation — but the K3 cluster adds **distributed ownership of weights and distributed CPU execution**.

The objective is not to pretend that multiple DDR2/DDR3/DDR4 servers become one physically coherent RAM machine. They do not. The objective is to expose the cluster externally as **one Kimi K3 inference endpoint** while every node owns and computes a defined shard of the real K3 model.

---

## 1. Kimi K3 facts that constrain the design

The current official Moonshot Kimi K3 release reports:

- architecture: Mixture-of-Experts (MoE);
- total parameters: **2.8 trillion**;
- activated parameters per token: **104 billion**;
- **93** transformer layers;
- one dense layer;
- attention composition: **69 KDA + 24 Gated MLA**;
- text hidden size: **7168**;
- **896 routed experts**;
- **16 routed experts selected per token**;
- **2 shared experts**;
- context length: **1,048,576** tokens;
- native quantization: **MXFP4 weights / MXFP8 activations** from quantization-aware training.

The Hugging Face repository currently reports approximately **1.56 TB** for the released checkpoint files.

The released `config.json` additionally confirms `num_hidden_layers=93`, `num_experts=896`, `num_experts_per_token=16`, `num_shared_experts=2`, `hidden_size=7168`, and an `mxfp4-pack-quantized` compressed-tensors configuration with 4-bit weight groups of 32 for the targeted linear layers. The config also explicitly excludes some components from that 4-bit rule, including attention, shared-expert paths, `lm_head`, vision tower, and projector. Therefore **104B × 4 bits is only a lower-bound traffic model, not the exact bytes read by a real token**.

Primary sources:

- https://github.com/MoonshotAI/Kimi-K3
- https://huggingface.co/moonshotai/Kimi-K3/tree/main
- https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json

---

## 2. What “100% of Kimi K3 distributed” means

For TensorWave, a valid full-model deployment means all of the following are present somewhere in the cluster:

- every released transformer layer;
- every routed expert required by the checkpoint;
- all 896 experts for every MoE layer — **no pruning as a requirement for correctness**;
- shared experts;
- routers/gates;
- KDA/MLA/attention weights and state;
- embeddings;
- normalization parameters;
- final `lm_head`;
- multimodal/vision components if the chosen endpoint advertises multimodal K3 rather than text-only K3.

A node does **not** need to contain the full checkpoint. It only needs the complete tensors assigned to that node by the placement plan.

A deployment that removes experts, drops layers, substitutes a different model, or silently stores missing weights nowhere is not considered “full K3” by this project.

---

## 3. The wrong mental model: remote RAM over Ethernet

Do **not** design the system as:

```text
CPU on node A -> reads arbitrary remote DIMM on node B -> reads node C -> ...
```

Commodity Ethernet does not turn independent server RAM into local coherent memory, and old DDR2/DDR3 platforms do not expose a CXL-style coherent fabric.

Instead:

```text
node owns weights -> node computes those weights -> compact activations/results move over network
```

Weights should stay resident in the NUMA domain that uses them. The interconnect transports:

- activation tensors;
- router decisions / expert dispatch metadata;
- selected expert inputs and outputs;
- synchronization/control messages;
- optional KV/KDA state movement only when the placement algorithm explicitly requires it.

It should **not** transport entire weight matrices every token.

---

## 4. Why simple pipeline parallelism is sufficient for capacity, but not necessarily speed

A first implementation can partition the 93 layers across machines:

```text
client
  |
  v
node 0: embedding + layers 0..N
  |
  v
node 1: next layer range
  |
 ...
  |
  v
last node: final layers + norm + lm_head
```

This is a real distributed-inference technique. vLLM officially supports pipeline parallelism and multi-node inference, and pipeline parallelism is specifically useful when a model does not fit on one device/node.

However, **pipeline parallelism alone must not be advertised as a way to multiply single-request decode token/s**. Autoregressive token `t+1` cannot complete before token `t` traverses the serial model path. For one sequence, stage latencies largely add.

Therefore TensorWave uses layer/pipeline partitioning primarily for **capacity** and combines it with parallelism inside MoE work for **throughput/latency**.

Reference:

- https://docs.vllm.ai/en/latest/serving/parallelism_scaling/

---

## 5. The important parallelism for K3: expert parallelism

K3 activates only 16 of 896 routed experts for each token. This is the exploitable structure.

A TensorWave MoE layer can assign experts across workers:

```text
896 experts
   |
   +-- worker A owns experts 0..55
   +-- worker B owns experts 56..111
   +-- ...
   +-- worker P owns final expert range
```

For one token, the router produces 16 expert IDs. The runtime groups selected experts by owner and dispatches the corresponding activation slices concurrently:

```text
router -> [e17, e49, e103, e184, ... 16 total]

           +-> node A: e17, e49
           +-> node B: e103
           +-> node C: e184, ...
           ... in parallel

all selected outputs -> reduce/combine -> next sublayer
```

This allows multiple servers' memory controllers and CPUs to work on the same token at the same time instead of placing every layer in a purely serial server chain.

vLLM's current MoE design independently validates this general architecture: it supports **Expert Parallelism (EP)** where expert networks are sharded across ranks, while attention can use a different strategy. TensorWave cannot simply run stock vLLM on the oldest DDR2 CPUs, but the parallel decomposition itself is established rather than invented here.

References:

- https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
- https://docs.vllm.ai/en/stable/configuration/optimization/

---

## 6. Proposed hybrid placement

The current design target is a hybrid:

1. **Layer/stage partitioning** where necessary to keep dense/attention state local and fit total capacity.
2. **Expert parallelism** across a worker group for each MoE stage, so the selected experts can consume aggregate memory bandwidth concurrently.
3. **Tensor parallelism only where benchmarks justify it**, because old-network collectives may cost more than the compute they save.
4. **NUMA-local ownership** inside every multi-socket server.

A placement file should be explicit and reproducible:

```yaml
model: moonshotai/Kimi-K3
checkpoint_revision: <pinned-hf-revision>

layers:
  0:
    attention_owner: node00/numa0
    shared_expert_owner: node00/numa1
    routed_experts:
      0-55: node00/numa0
      56-111: node01/numa0
      112-167: node02/numa0
      # ...

  1:
    # explicit ownership again
```

The placement generator must verify that every tensor in the checkpoint index has exactly one intended storage/compute owner (or an explicitly declared replica).

---

## 7. Runtime components

Proposed components are intentionally small and testable.

### `tw-k3-inspect`

- parses `config.json` and `model.safetensors.index.json`;
- emits the exact tensor inventory and byte sizes;
- identifies quantized vs non-quantized paths;
- fails if a model revision differs from the pinned manifest.

### `tw-k3-reshard`

- reads the official safetensors shards;
- maps tensors to nodes/NUMA domains according to the placement plan;
- writes local shard manifests;
- hashes every output tensor;
- proves no checkpoint tensor was lost.

### `tw-k3-worker`

Runs on every server/NUMA domain and provides:

- mmap/pinned local weight store;
- KDA/MLA/attention kernels;
- MXFP4 unpack/dequant + expert GEMV/GEMM kernels;
- shared-expert execution;
- expert-routing receive/send queues;
- local KV/KDA state;
- NUMA CPU and memory affinity.

### `tw-k3-router`

- evaluates or receives MoE router output;
- maps selected expert IDs to physical owners;
- groups dispatches by owner;
- schedules concurrent expert execution;
- combines returned expert results in the mathematically correct order.

### `tw-k3-transport`

Phase 1:

- persistent TCP sockets;
- preallocated binary buffers;
- no JSON/HTTP inside the hot path.

Phase 2, if supported by purchased NICs/platforms:

- RDMA / InfiniBand / RoCE path;
- registered memory buffers;
- batched all-to-all style exchange.

### `tw-k3-controller`

- tokenizer interface;
- generation loop;
- sampling;
- request state;
- layer/stage sequencing;
- error handling and worker health.

### `tw-k3-gateway`

Exposes one endpoint to the rest of the network:

```text
POST /v1/chat/completions
```

The client does not know whether K3 is physically running on 3, 5, 10 or 20 servers.

---

## 8. User-facing architecture

The intended operator experience is:

```text
Developer PC
  |
  | Kimi Code / agentic framework / OpenAI-compatible client
  v
http://transit-k3:8000/v1
  |
  v
TensorWave K3 gateway/controller
  |
  +---------------- distributed inference fabric ----------------+
  |                                                               |
worker 00 <-> worker 01 <-> worker 02 ... <-> worker N
  |             |             |
local RAM     local RAM     local RAM
local shard   local shard   local shard
```

The official K3 Hugging Face page already documents serving via an OpenAI-compatible `/v1/chat/completions` interface using supported runtimes. TensorWave's gateway keeps that external contract even though the internal CPU/old-server runtime is custom.

Source:

- https://huggingface.co/moonshotai/Kimi-K3/tree/main

---

## 9. Why stock vLLM is not the DDR2 solution

Current vLLM CPU documentation recommends AVX-512 and has limited-feature AVX2 fallback for x86; its AMD Zen optimization path targets Zen 4/5, and current AMD CPU support guidance requires modern processors for the AVX-512 path.

The DDR2 targets in this repository are Xeon/Opteron generations many years older. Therefore the project must assume:

- stock vLLM is **not** the runtime for DDR2-era CPU execution;
- K3's MXFP4 representation needs a custom old-x86 unpack/dequant/compute path, likely SSE/SSE2/SSE4 depending on the chosen CPU;
- kernels must be selected at runtime by CPU feature detection;
- correctness comes before assembly optimization.

The DDR4 EPYC route is much closer to software that modern frameworks can support, although K3 CPU MXFP4 serving still requires verification rather than assumption.

Source:

- https://docs.vllm.ai/en/stable/getting_started/installation/cpu/

---

## 10. Full-RAM is preferred; SSD is a fallback

If aggregate RAM is smaller than the checkpoint, TensorWave can technically build a tiered store:

```text
RAM: dense/attention/shared weights + hot experts
SSD: cold expert backing store
```

but this is not the primary procurement plan. An SSD miss can add orders of magnitude more latency than a local DRAM read, and expert selection is dynamic.

For interactive K3, the preferred configuration is:

```text
complete checkpoint + runtime headroom <= aggregate usable RAM
```

and every expert shard needed by a worker is resident in that worker's DRAM.

---

## 11. Correctness proof ladder

Before making any token/s claim, the distributed implementation must pass:

### P0 — tensor inventory proof

For a pinned K3 revision:

```text
union(all worker tensor manifests) == official checkpoint tensor set
```

and hash checks pass.

### P1 — primitive/kernel proof

For every custom MXFP4/KDA/MLA primitive:

```text
custom_CPU_output ~= reference_output
```

under a documented numerical tolerance.

### P2 — one-layer proof

One real K3 layer executed by TensorWave matches the reference implementation for the same weights and input.

### P3 — two-process distributed proof

Split a small real layer range and/or expert group between two processes. Distributed output must match the monolithic reference within tolerance.

### P4 — full forward proof

All 93 layers and all required experts are represented and one complete forward pass returns numerically valid logits.

### P5 — generation proof

Given a fixed prompt, sampling settings and random seed, generated tokens/logits are compared against the reference implementation using a documented tolerance policy.

### P6 — API proof

An OpenAI-compatible client completes a tool-capable K3 request through the TensorWave gateway.

Only after P0–P6 do we call the system a functional distributed K3 runtime.

---

## 12. What is proven now vs what remains experimental

### Proven by architecture/documentation

- K3 is MoE and activates 16 of 896 experts per token.
- A 1.56 TB checkpoint can be partitioned across multiple independent RAM pools.
- Pipeline and expert parallelism are established distributed-inference techniques.
- The external client can be decoupled from physical model placement through one API endpoint.
- Sufficient aggregate RAM can hold the complete released checkpoint.

### Not yet proven in this repository

- old Xeon/Opteron MXFP4 kernel throughput;
- actual STREAM bandwidth of any purchased DDR2/DDR3 machine;
- inter-node latency and all-to-all efficiency of the actual NIC/switch combination;
- exact K3 bytes read per decoded token after non-MXFP4 paths and caches are included;
- any measured end-to-end K3 token/s number.

The next documents separate **capacity arithmetic** from a falsifiable **throughput model**, so TensorWave does not convert a plausible architecture into an invented benchmark.
