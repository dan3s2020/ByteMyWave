# Large-MoE target selection and 4×DL360p Transit runtime — 2026-09-16

Status: **current web research + analytical architecture/sensitivity model; not an end-to-end benchmark**.

This note extends the 2026-09-16 Q4_K×Q8_K/x8meta work. It asks a different question:

> Given the actual project machine class — four dual-socket HP DL360p Gen8 servers, large DDR3 capacity and two mixed-VRAM GPUs per server — which current large model is structurally favorable, where should every class of work execute, and what token-rate range is worth engineering toward?

It deliberately does not multiply unrelated paper/kernel speedups.

---

## 1. Hardware constraints used

Current project topology:

```text
4 × HP ProLiant DL360p Gen8
2 CPU sockets/server
8 CPU sockets total
~2 TB aggregate DDR3 discussed for the cluster
2 GPUs/server = 8 GPUs total
mixed cards: some ~12 GB VRAM, some ~6 GB VRAM
```

At least the E5-2680 v2 target has:

```text
10 cores / 20 threads
AVX, no AVX2
4 DDR3 channels/socket
DDR3 up to 1866
59.7 GB/s theoretical max memory bandwidth/socket
PCIe 3.0
```

Sources:

- Intel E5-2680 v2: https://www.intel.com/content/www/us/en/products/sku/75277/intel-xeon-processor-e52680-v2-25m-cache-2-80-ghz/specifications.html
- HPE DL360p Gen8 QuickSpecs: https://support.hpe.com/hpesc/public/api/document/c04128242

The 59.7 GB/s value is an interface ceiling, not a low-bit GEMV rate. The decisive physical measurement remains useful compressed-weight GB/s / Gweights/s per socket.

---

## 2. Model search result

### Immediate implementation target: Step-3.7-Flash / Step-3.5-Flash text backbone

Step-3.7-Flash is a 198B multimodal model whose language backbone is the same Step3p5-style sparse architecture:

```text
~198B total (196B language backbone + vision)
~11B active/token
45 transformer layers
42 MoE layers (layers 3..44)
288 routed experts/layer
Top-8 routed experts/token
1 shared expert
hidden size 4096
MoE intermediate 1280
3 NextN/MTP layers
256K context
```

Official/current sources:

- https://huggingface.co/stepfun-ai/Step-3.7-Flash
- https://huggingface.co/stepfun-ai/Step-3.7-Flash/blob/main/config.json
- https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF

The official Step-3.7 material lists a Q4_K_S language-model GGUF at about **111.5 GB**. Step-3.5 has an upstream ggml Q4_K file at about **119 GB**, and the public architecture is 196.81B / ~11B active / Top-8.

Why Step is unusually well matched to the current cluster:

```text
model Top-K experts = 8
physical CPU sockets = 8
```

The hardware can therefore map one selected routed expert to one independent NUMA/memory domain at every MoE layer.

### Ultimate research target: DeepSeek-V4.1-Flash

DeepSeek-V4.1-Flash is even more interesting architecturally:

```text
552B backbone
8B active during prefill / 16B during decode
40 layers
384 routed experts
Top-6 routed experts/token
1 shared expert
~196B Engram conditional-memory parameters
DSpark speculative head
1M context
```

Official source:

- https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash

The huge Engram component is sparsely accessed by token-derived n-gram lookup, which is an unusually good fit for a large cheap-RAM system. A recent functional llama.cpp fork on one RTX 5090 measured 5.12 tok/s on new content and 21.27 tok/s on resident content; its Engram prefetch reduced Engram lookup from 10.44 ms/token to 0.92 ms/token. This directly demonstrates how much residency/prefetch can matter for this model.

Source:

- https://github.com/JigSawPT/deepseek-v41-flash-on-5090

However, upstream llama.cpp support is not yet complete. Community Q3_K_M is ~323.4 GiB and Q2_K ~246.3 GiB, while a working port used a ~502 GB target representation. Therefore V4.1 is Phase 2, not the fastest path to a reproducible cluster result.

### Other candidates

- **GLM-5.3-Flash**: 320B total / 18B active, attractive hybrid sparse+linear attention, but upstream llama.cpp support is still immature as of September 2026.
- **MiniMax-M2.5**: ~230B total / 10B active / 256 experts / Top-8 / 62 layers; community Q4_K_M ~129 GB. Structurally good, but 62 serial layer boundaries make network latency less favorable than Step's 45 layers.
- **Qwen3-Next-80B-A3B**: only ~3B active and mature upstream support, excellent proof workload but less interesting if the goal is maximum capability from the available RAM.

Decision:

> **Build the first full 8-socket Transit runtime around Step-3.7/3.5 Q4 K-quants. Preserve DeepSeek-V4.1-Flash as the next target once the distributed runtime is proven.**

---

## 3. The new key use of excess RAM: replicate experts for compute parallelism

A normal deployment asks whether the model fits in RAM.

Transit should ask a more useful question:

> Can RAM capacity buy independent same-token memory bandwidth by replicating the routed expert pool into every NUMA node?

For Step-3.7 Q4_K_S, the whole language model is only ~111.5 GB. Eight entire copies would be roughly:

```text
111.5 GB × 8 = 892 GB
```

Even before optimizing replication granularity, this is below the project's ~2 TB aggregate RAM budget.

The better implementation is not eight independent model runtimes. It is:

```text
socket 0 RAM: complete routed-expert store, NUMA local
socket 1 RAM: complete routed-expert store, NUMA local
...
socket 7 RAM: complete routed-expert store, NUMA local

shared/non-routed weights: one logical copy, placed on GPU/layer owners
KV/state: one logical execution state, not duplicated eight times
```

This means every socket can execute **any** expert selected by the router. There are no expert-placement collisions and no remote-NUMA weight reads on the intended fast path.

The 2026-09-16 x8meta/predecoded layout can be generated independently inside each socket-local expert store at model load time. Its measured laptop cost was +2.78% runtime storage for the repacked Q4_K subset, which is acceptable in this capacity regime, but its Ivy-Bridge speedup is not assumed until measured.

---

## 4. Step routed-expert byte screen

Step shape:

```text
hidden = 4096
expert intermediate = 1280
routed expert weights ~= 3 × 4096 × 1280
                     = 15,728,640 weights/expert
MoE layers = 42
Top-K = 8
```

Thus selected routed expert weights/token:

```text
15,728,640 × 42 × 8
= 5.28482304B logical weights/token
```

Using the whole-file Q4_K_S average as a **screening ratio only**:

```text
111.5 GB / 196.81B params
~= 0.5665 bytes/parameter
```

routed selected-weight traffic screen:

```text
5.2848B × 0.5665
~= 2.994 GB/token
```

With perfect Top-8 -> 8-socket assignment:

```text
2.994 / 8
~= 0.374 GB/socket/token
```

This is the central architectural advantage.

If one socket sustains useful expert-kernel compressed-weight rates of:

```text
5 GB/s  -> routed critical path ~74.9 ms/token
8 GB/s  -> ~46.8 ms
10 GB/s -> ~37.4 ms
12 GB/s -> ~31.2 ms
15 GB/s -> ~25.0 ms
```

These are routed-weight-path screens, not complete token times.

The socket's theoretical DDR interface is 59.7 GB/s, so even 10 GB/s of useful Q4 expert consumption requires only ~17% of the nominal interface ceiling. Whether AVX-era unpack/dequant can sustain it is the benchmark gate.

---

## 5. Full runtime topology

### Per CPU socket

One pinned worker pool owns:

```text
NUMA-local replicated routed-expert Q4 store
predecoded/x8meta-style runtime layout where beneficial
10 physical cores as one expert engine
local thread pool pinned to that socket
local result accumulator
```

Do not use one CPU to issue memory operations and another CPU to collect them. Each CPU computes against its own local memory.

### Per server

```text
socket A -> one selected expert
socket B -> one selected expert
local reducer -> combine the two 4096-element partial outputs
GPU 0 -> layer-spine shard / attention / router / KV
GPU 1 -> shared expert + hot routed-expert cache and/or draft path
network worker -> activation dispatch + result return
```

A 6 GB GPU is large enough to be useful for the relatively small non-routed layer shard. The 12 GB GPUs should preferentially receive cache/draft roles where extra capacity creates more avoided CPU work.

### Across four servers

Split the 45 layer spine into contiguous ranges, approximately:

```text
server 0: layers 0..11
server 1: layers 12..22
server 2: layers 23..33
server 3: layers 34..44
```

The current layer owner:

1. runs attention/router/non-routed work on its resident GPU shard;
2. quantizes the expert input activation to Q8_K once;
3. broadcasts the tiny Q8_K activation + selected expert IDs to all four servers;
4. schedules the eight selected experts one-per-socket across all eight sockets;
5. simultaneously executes the shared expert on GPU where possible;
6. each server locally reduces its two CPU expert results;
7. three remote servers return one partial vector each;
8. layer owner combines all four routed partials + shared-expert output;
9. execution continues.

At a layer-range boundary, the hidden state moves to the next server's spine GPU. There are only three such ownership handoffs per token.

---

## 6. Q8_K activations should be the inter-server CPU-expert input format

Today's kernel work makes another useful optimization obvious.

Instead of sending FP32 hidden vectors to every CPU socket and quantizing them eight times:

```text
layer owner
  FP32 activation
      |
      +-- quantize once to Q8_K
      |
      +-- send the same Q8_K bytes to 4 servers / 8 sockets
```

For hidden size 4096:

```text
4096 / 256 = 16 Q8_K blocks
16 × 292 bytes = 4,672 bytes
```

So the dynamic expert input is only ~4.6 KiB/layer.

Each socket then consumes exactly the Q8_K representation expected by the Q4_K×Q8_K kernel. This removes seven redundant activation-quantization jobs and reduces network payload.

The expert's down-projection uses an expert-specific intermediate activation and is quantized locally after SwiGLU; that part cannot be shared across experts.

---

## 7. CPU expert kernel to build

The Ivy Bridge kernel should not merely port the laptop function instruction-for-instruction.

Expert execution has three projections:

```text
gate(x)
up(x)
SwiGLU(gate, up)
down(intermediate)
```

Load-time representation:

```text
Q4_K original bytes
 -> exact decoded scale/min metadata sidecar
 -> 8-row interleave / x8meta-style layout
 -> NUMA-local huge-page-backed store
```

Hot path:

```text
Q8_K x shared by gate + up
fused scheduling of gate/up row groups
SSSE3/AVX arithmetic for E5-2680 v2
all cores of one socket cooperate on one expert
local intermediate
Q8_K quantize intermediate
Q4_K × Q8_K down projection
FP32 local output
```

No new weight approximation is required.

Today's laptop evidence says the layout/predecode idea can beat llama-style AVX2 by ~11% on Q4_K×Q8_K. The server path is AVX/SSSE3 and must be benchmarked separately; **zero speedup is assumed in the base system estimate** until that benchmark exists.

---

## 8. GPU roles

### GPU role A — permanent spine shard

Keep permanently resident:

```text
attention weights for the server's layer range
router weights
norms
first dense layers when owned
output/lm-head when owned
KV cache for owned layers
```

These weights execute every token. They deserve VRAM more than randomly selected routed experts.

### GPU role B — shared expert + hot routed-expert cache

Routed experts are only about ~9 MB each at the Step Q4 screening density:

```text
15.73M weights × 0.5665 B ~= 8.9 MB
```

Therefore even several GB of spare VRAM can hold hundreds of `(layer, expert)` objects.

Cache policy must be trace-driven:

```text
rolling frequency
per-layer frequency
recency decay
predicted next-token route
measured CPU cost avoided
measured GPU hit execution cost
```

Critical rule:

> A cache miss must **not** block on host->GPU transfer. CPU local execution is the fallback. Cache fills happen asynchronously.

This is the useful HybriMoE/MoE-Infinity lesson adapted to our hardware.

### Dynamic use of spare CPU sockets after GPU hits

With no GPU expert hits:

```text
8 selected experts -> 8 CPU sockets -> 1 socket/expert
```

If enough selected experts are already resident on GPUs, those experts can execute on GPU and release CPU socket capacity. Spare sockets can then tensor-shard the remaining CPU experts across two NUMA domains by splitting the 1280 intermediate dimension and reducing partial down outputs.

This makes **partial GPU cache hits useful**, rather than requiring all eight experts to hit before the layer gets faster.

The scheduler should compare:

```text
GPU-resident expert time
1-socket CPU expert time
2-socket-sharded CPU expert time + reduction
```

and solve a small assignment problem per layer.

---

## 9. Speculation / MTP

Step-3.7 carries three NextN/MTP layers. Current llama.cpp contains generic draft-MTP infrastructure and Step-family conversion support, but speculative decoding has had backend/context-specific regressions in 2026. MTP therefore remains an A/B optimization, not a baseline multiplier.

Useful placement idea:

- put a more aggressively quantized **draft/MTP sidecar** on a 12 GB GPU;
- draft quantization can be lower precision than the target because rejected draft tokens do not change final target-model correctness;
- use draft router decisions as **prefetch hints**, never as mandatory blocking loads;
- if MTP wall-clock gain is not positive, disable it.

A conservative sensitivity range after measurement is ~1.10–1.25× wall-clock, not “three MTP layers = 3×.”

For highly repetitive code/tool output, n-gram speculation can be tested separately.

---

## 10. Network budget

Hidden size = 4096.

CPU expert activation sent as Q8_K:

```text
~4.7 KiB per layer per remote server
```

Returned server partial is naturally accumulated in FP32:

```text
4096 × 4 = 16 KiB
```

For three remote servers:

```text
~3 × (4.7 + 16) KiB
~= 62 KiB/MoE layer
42 MoE layers
~= 2.6 MiB/token
```

Thus bandwidth is not tens of GB/token. The harder problem is **42 serial small-message synchronization points**.

If the installed DL360p NICs expose multiple independent 1GbE ports, a full-mesh/direct-peer layout can use three links/server so three peers communicate concurrently. If 10GbE is available, use it. Actual NIC inventory and p50/p95 round-trip latency must be measured before freezing the transport.

---

## 11. Sensitivity model for Step on the current cluster

The Step Q4_K_S whole-file screening ratio gives:

```text
active bytes/token            ~6.23 GB
routed Top-8 bytes/token       ~2.99 GB
routed bytes/socket/token      ~0.374 GB
non-routed + shared screen     ~3.24 GB/token
```

A standard one-server or serial layer-pipeline design cannot use eight independent socket memory domains on the same MoE layer. Its rough weight-path ceiling with two sockets each sustaining useful rate `B` is approximately:

```text
TPS_normal_weight ~= 2B / 6.23
```

Examples:

```text
B=5 GB/s/socket   -> ~1.6 tok/s weight-path ceiling
B=8               -> ~2.6
B=10              -> ~3.2
B=12              -> ~3.9
```

This is not an end-to-end prediction, but it shows why simply spreading serial layers over four servers does not aggregate all eight socket bandwidths for one token.

For the Transit expert-parallel design, assume a placeholder **45 ms/token** for all work outside the routed CPU critical path (GPU spine + network + reductions + runtime overhead). This is deliberately exposed as an assumption, not a measured value.

| Useful CPU expert stream / socket | Routed CPU ms/token | Transit screen | If future Ivy expert kernel gives 1.10× only on routed CPU | Plus hypothetical 1.15× MTP wall gain |
|---:|---:|---:|---:|---:|
| 5 GB/s | 74.9 | 8.34 tok/s | 8.85 | 10.17 |
| 8 GB/s | 46.8 | 10.90 | 11.42 | 13.14 |
| 10 GB/s | 37.4 | 12.13 | 12.65 | 14.55 |
| 12 GB/s | 31.2 | 13.13 | 13.63 | 15.68 |
| 15 GB/s | 25.0 | 14.30 | 14.77 | 16.99 |

The x8meta and MTP columns are **counterfactual sensitivity only**. The current measured x8meta result is AVX2 laptop code, not Ivy Bridge.

### Engineering interpretation before hardware calibration

A useful target envelope is:

```text
weak / low socket kernel or high network latency    ~4–8 tok/s
credible first optimized region                     ~8–15 tok/s
strong result with good Ivy kernel + low RTT        ~12–18 tok/s
stretch with useful MTP/cache hits                  ~15–20 tok/s
```

Do not use the upper numbers for procurement claims until the server measurements exist.

Compared with a normal same-sequence layout that effectively exposes only one dual-socket server's weight path at a time, the architecture is worth testing for roughly **2–4× end-to-end decode speedup**. The dominant gain is 8-way expert parallelism and permanent GPU residency of always-on work, not the ~11% laptop x8meta result.

---

## 12. External sanity anchors, not coefficients

Step-3.5 Q4_K_S public llama.cpp benchmarks report approximately:

```text
M4 Max 128GB unified memory      48.18 tok/s at tiny KV
DGX Spark                         19.79 tok/s
Ryzen AI Max+ 395 Vulkan           3.43 tok/s
```

Source:

- https://github.com/stepfun-ai/Step-3.5-Flash/blob/main/llama.cpp/docs/step3.5-flash.md

These systems are radically different from four Ivy Bridge servers. They are useful only to show that the same model spans a very wide range depending on memory architecture/runtime. They are not scaling coefficients.

A Transit target around 10–15 tok/s would sit below the modern DGX-Spark result while attempting to obtain useful speed from old commodity memory domains through expert parallelism.

---

## 13. DeepSeek-V4.1 Phase-2 mapping

Once the Step runtime exists, adapt it to DeepSeek-V4.1:

```text
Top-6 experts -> six of eight CPU sockets
remaining two sockets -> second shards / Engram work / overlap
Engram tables -> RAM-resident, partitioned/replicated hot buckets
Engram addresses -> precompute from token ids and issue batched prefetch
routed experts -> NUMA ownership + selective replication, not eight full copies
always-on attention/shared path -> GPU spine
DSpark -> optional draft GPU
```

Because Q3_K_M alone is ~323 GiB and most of the file is Engram memory, eight complete model copies are inappropriate. Replicate only what creates useful same-token bandwidth and keep the giant lookup memory sharded with a hot-bucket replica policy.

The 5090 report's exact Engram prefetch improvement is strong evidence that this is a legitimate optimization direction, but its token rates are not portable to E5-2680 v2.

---

## 14. What to measure next

The architecture now depends on a short list of physical numbers:

```text
A. one-socket Q4_K expert Gweights/s / compressed GB/s on E5-2680 v2
B. x8meta/predecoded SSSE3/AVX variant versus stock llama.cpp on Ivy Bridge
C. one expert with all 10 cores pinned, NUMA-local
D. two sockets simultaneously, one expert/socket
E. Q8_K activation dispatch + 4096-FP32 result RTT between two servers
F. four-server p50/p95/p99 per-layer fanout/reduce
G. GPU spine ms/token for Step non-routed path on each actual GPU type
H. GPU hot-expert execution time and cache hit rate on real prompts
I. MTP A/B wall-clock gain
J. complete Step token trace
```

After A, E and G, the token/s sensitivity range will narrow substantially. After J, replace this document's tok/s envelope with measured class-1 evidence.

---

## 15. Current decision

Build around **Step-3.7/3.5 Flash Q4 K-quants first** because it simultaneously offers:

```text
large model capability
~11B active/token
Top-8 exactly matching 8 CPU sockets
~111–119 GB Q4 model size
mature/current llama.cpp family support
3-way MTP architecture
small 4096-d hidden vectors for network transport
enough RAM to replicate the routed expert pool per NUMA socket
```

The architecture is:

```text
RAM = resident replicated expert fabric
CPUs = 8 parallel routed-expert engines
small GPUs = permanent layer-spine / attention / KV engines
large GPUs = shared/hot expert cache + optional draft/MTP
network = only Q8_K activations + reduced output vectors
x8meta/predecode = local CPU-kernel accelerator if Ivy benchmark validates it
```

DeepSeek-V4.1-Flash is the next model specifically because its Top-6 sparse MoE + giant Engram lookup memory can exploit the same runtime plus the project's large RAM capacity in a different way.
