# 19 — GLM-5.2 4×2 Decision, Throughput Target and Implementation Readiness

Date: 2026-08-18

## Executive decision

Continue with the already selected prototype:

```text
4 servers × 2 GPU devices/cards per server
host RAM filled as planned
GLM-5.2
Q3-class GGUF representation
single-user / batch=1 agentic decode first
```

Do **not** expand to a much larger server fleet yet.

The software direction is now frozen as:

```text
llama.cpp / GGUF as the Q3 execution substrate
+ ByteMyWave distributed expert scheduler
+ NUMA-local host-resident expert store
+ GPU hot-expert cache
+ CPU execution fallback for cold experts
+ asynchronous expert prefetch
+ request-level routing trace
+ native GLM-5.2 MTP
+ MTP-aware expert prefetch
+ ngram-simple optional fast path for structured/repetitive output
+ per-GPU-family worker builds for legacy hardware
```

KTransformers, HybriMoE, MoE-Infinity, SpecMoEOff and SP-MoE are reference implementations/research inputs. They are not adopted wholesale as the production runtime.

---

# 1. Why llama.cpp is the selected base

For this target, llama.cpp currently has the best combination of properties:

- GGUF;
- Q3/Q2/Q4 support;
- CPU+GPU execution;
- explicit tensor/layer placement;
- broad old-GPU/community use;
- GLM_DSA / GLM-5.2 support;
- recently integrated GLM-5.2 NextN/MTP path;
- low-level code that can be forked and instrumented.

A real GLM-5.2 Q3 build is already public at roughly `295.71 GiB / 3.37 bpw` and runs end-to-end in llama.cpp.

Source:

- https://huggingface.co/SixVolts/GLM-5.2-ewaste-edition-GGUF/blob/main/README.md

KTransformers remains important because its GLM-5.2 path already demonstrates dynamic expert updates and expert placement, but its currently documented GLM-5.2 serving path is FP8/BF16-oriented and assumes a modern CUDA/SGLang environment.

---

# 2. The model is not a 743B-dense decode workload

Official/public working values:

```text
total parameters          ~743B
active parameters/token    ~39B
routed experts/layer       256
selected routed experts    8
layers                     78
NextN/MTP layers           1
```

Source:

- https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json
- https://recipes.vllm.ai/zai-org/GLM-5.2

This is why the selected architecture is expert-parallel and cache-aware.

A four-server **layer pipeline only** would mainly solve capacity and would repeatedly serialize one sequence across server boundaries.

The preferred mapping is instead:

```text
all four servers own experts across the model
router selects eight experts
selected expert owners execute concurrently
hot experts execute from local GPU VRAM
cold experts execute locally on CPU/RAM or are prefetched when beneficial
small activations/results cross the server fabric
```

Layer partitioning can still be used for non-MoE/shared work where it is cheaper, but it is not the default principle for routed experts.

---

# 3. Q3 active-byte screen

Using the measured/public Q3 build average of `3.37 bits/weight`:

```text
39e9 active params × 3.37 / 8
≈ 16.43 GB/token
```

This is only a screening payload.

The exact target GGUF uses mixed precision by tensor class and the real trace includes:

- shared experts;
- attention/DSA tensors;
- quant scales/metadata;
- router/indexer work;
- KV/state traffic;
- MTP verification work.

The implementation must produce an exact per-token byte trace before this number is promoted to a calibrated throughput model.

---

# 4. Current working throughput estimate

## 4.1 Evidence boundary

There is **no measured full GLM-5.2 run yet on the exact four purchased servers**.

Therefore every number below is a planning estimate, not a benchmark.

The previous project discussion converged on an approximately `~1–1.3 tok/s` baseline class for the unoptimized 4×2 host-memory/hybrid configuration. This note preserves that only as the pre-optimization planning baseline; it is not evidence class 1.

The new literature does **not** justify multiplying every reported paper speedup together.

Several techniques target the same stall time, so their gains overlap.

---

## 4.2 Expected ranges after implementation

### A. First functional Q3 distributed baseline

Target:

```text
~1.0–1.4 tok/s
```

This is the expected class before sophisticated expert prediction/MTP-aware prefetch, but after correct NUMA placement and a sane four-node execution plan.

### B. Tier-A optimized runtime

Includes:

```text
hot-expert GPU cache
frequency/decay placement
async H2D double buffering
CPU fallback for cold experts
request routing traces
communication/compute overlap
ngram-simple when it naturally matches
```

Expected general decode:

```text
~1.4–2.1 tok/s
```

Expected structured/code workloads when n-gram speculation is useful:

```text
~1.7–2.6 tok/s
```

The n-gram increment is workload-dependent. A public GLM-5.2 Q3 benchmark measured ~0% on ordinary prose and about +22% on structured code / +27% on verbatim repetition.

---

### C. Full selected architecture with native MTP + MTP-aware expert prefetch

Includes Tier A plus:

```text
native GLM-5.2 NextN/MTP
accepted-token instrumentation
MTP-driven speculative expert prefetch
prefetch cutoff to avoid overfetch
cache-policy auto-tuning
remote dispatch overlap
hot-expert replication where traces justify it
```

**Current expected general-purpose range:**

```text
~2.0–3.0 tok/s
```

**Expected structured/agentic/code range:**

```text
~2.4–3.5 tok/s
```

**Stretch region, not a promise:**

```text
~4 tok/s
```

The stretch result requires simultaneously good MTP acceptance, high expert-cache hit rate, useful four-server expert concurrency and no severe legacy-GPU kernel bottleneck.

---

# 5. Why the new answer is not 10 tok/s

Relevant papers report large maxima:

```text
HybriMoE      ~1.70× average decode speedup
SpecMoEOff    up to ~2.5×
SpecOffload   up to ~2.54×
SP-MoE        ~1.07–3.5× TPOT speedup
SpecMoE       up to ~4.30×
```

But these are not independent factors.

They all attack overlapping portions of:

```text
expert miss latency
H2D/I/O latency
CPU/GPU under-utilization
prefetch timing
speculative verification
```

If 70% of a token is offload wait and two different systems both eliminate much of that same 70%, applying both does not eliminate 140% of the token.

Therefore the present source-of-record position is:

> **2–3 tok/s is a defensible implementation target for the full 4×2 optimization stack. 3–4 tok/s is the stretch range to prove. 10 tok/s is not currently supported by evidence for this exact hardware.**

This number must be replaced by the first physical benchmark as soon as the four-node runtime exists.

---

# 6. External performance anchor

The public `GLM-5.2-ewaste-edition-GGUF` benchmark reports:

```text
Q3 model         295.71 GiB / 3.37 bpw
10 × MI100       all weights resident, zero spill
Decode           13.2 tok/s
```

A Q2 build on eight MI100 reports `14.7 tok/s`.

These MI100 systems have much stronger memory bandwidth and modern enough execution paths to keep the whole model in HBM, so our 4×2 host-memory rig should not be expected to match that result.

However, it provides a useful upper external comparison:

> If our optimized old-hardware hybrid reaches 2–3 tok/s on a ~296 GiB Q3 GLM-5.2, it is operating in a plausible fraction of a proven zero-spill multi-accelerator result rather than relying on an impossible arithmetic roofline.

---

# 7. Implementation architecture

## 7.1 Coordinator

One logical coordinator owns:

```text
request/session state
GLM layer progression
router results
expert ownership table
worker health
MTP state/acceptance statistics
distributed reduction ordering
```

The coordinator must not proxy model weights.

---

## 7.2 Four server workers

Each server worker owns:

```text
NUMA-aware model shards in host RAM
local CPU expert engine
local GPU worker(s)
GPU expert cache
async transfer queues
prefetch queues
routing-frequency statistics
local result reducer
```

Workers exchange activations/results and metadata, not bulk model tensors on every token.

---

## 7.3 Expert atlas

Every routed expert receives a stable record:

```text
layer
expert id
GGUF tensor offsets
quant type
exact bytes
primary server owner
optional replica owners
NUMA node
GPU-cache eligibility
rolling access frequency
last-used timestamp
estimated CPU execution cost
estimated H2D + GPU execution cost
```

This extends the existing ByteMyWave Weight Atlas concept rather than creating a second unrelated metadata system.

---

## 7.4 Hot-expert cache

Cache decisions are based on measured utility, not just recency.

Initial score components:

```text
recent frequency
global frequency
predicted next-token probability
H2D cost avoided
CPU-vs-GPU execution delta
expert byte size
replication value across server ownership
```

Policies to benchmark:

```text
LRU
LFU
frequency decay
cost-aware LFU
MTP/prediction-aware score
```

---

## 7.5 Cold-expert choice

For a cache miss, runtime computes both expected paths:

```text
T_cpu_local
T_prefetch_to_gpu + T_gpu
```

and chooses the lower critical-path contribution.

This is the key HybriMoE-style change from ordinary offload:

> A cold expert does not automatically mean “block until it has crossed PCIe.”

---

## 7.6 Speculation

Two speculation modes are retained:

### ngram-simple

Use immediately for safe structured/repetitive acceleration.

### native GLM-5.2 MTP

Primary long-term mode.

Instrumentation must record:

```text
draft length
accepted length
acceptance distribution
target verification time
extra MTP memory
experts predicted by draft/NextN
prefetched expert hits
prefetched expert false positives
bytes over-fetched
net tok/s delta
```

No MTP feature is considered successful merely because it produces drafts.

---

# 8. Legacy GPU strategy

The old GPU pool must not force one global CUDA toolchain.

In particular:

```text
Tesla K80 = Kepler sm_37, CUDA 11.x-era build path
Maxwell-class M10 = supported by CUDA 12.x-era compilation
newer GPUs = newest compatible llama.cpp/CUDA path
```

Use a stable ByteMyWave worker protocol between them.

This permits:

```text
server A worker -> legacy CUDA build
server B worker -> Maxwell/newer build
server C worker -> newer optimized build
...
```

while the coordinator sees one logical expert execution API.

The alternative — downgrading the whole cluster to the oldest GPU's dependencies — is rejected.

---

# 9. Do we have enough information to implement it?

## Yes — enough to start and build the software architecture

We now have enough public evidence and project context to implement:

- GGUF atlas parsing;
- expert ownership;
- request-level expert traces;
- cache policies;
- async host/GPU queues;
- worker/coordinator protocol;
- CPU-vs-GPU miss decision;
- ngram speculation integration;
- MTP instrumentation/integration against a pinned llama.cpp revision;
- distributed benchmark telemetry;
- configurable hardware backends so old/new GPUs can use different builds.

## No — not enough to honestly claim the final optimized tok/s before physical measurements

The following values can only come from the real machines:

```text
exact GPU list per server
negotiated PCIe width/speed
which GPU pairs have usable P2P
NUMA locality of each GPU
real host-RAM bandwidth per socket/server
real Q3 CPU expert Gweights/s
real H2D bandwidth per GPU
real Q3 kernel throughput on each GPU generation
NIC speed and small-message latency
GLM-5.2 MTP acceptance on the intended agentic workload
real expert-cache hit rate
```

These are not missing design ideas. They are calibration inputs.

Therefore the implementation can begin now, but final tuning and the `2–3 tok/s` target must be validated on hardware.

---

# 10. Implementation gates

## Gate 0 — exact inventory

Collect machine-readable topology from all four servers.

## Gate 1 — one server / one GLM-5.2 Q3 worker

Generate 128+ tokens and record per-layer/expert timings.

## Gate 2 — two local GPUs

Measure cache placement, two-GPU concurrency, H2D overlap and legacy backend compatibility.

## Gate 3 — two servers

Measure expert dispatch/reduction over the actual NIC.

## Gate 4 — four servers, no speculation

Establish the real full-model baseline.

## Gate 5 — ngram-simple

Separate prose vs. code/structured improvement.

## Gate 6 — native MTP

Measure acceptance and wall-clock gain.

## Gate 7 — MTP-aware prefetch

Measure cache/prefetch hit rate and overfetch.

## Gate 8 — auto-tuner

Search cache size, expert placement, CPU/GPU threshold, prefetch depth and MTP draft depth.

Only Gate 8 produces the final source-of-record throughput.

---

# 11. Final frozen decision

Proceed.

The four-server prototype is now valuable because it is large enough to test the complete architecture while small enough that a wrong assumption does not require a 20-server purchase.

The current performance target to engineer against is:

```text
minimum useful result       >= 1.5 tok/s
expected optimized result   2.0–3.0 tok/s
strong/stretch result       3.0–4.0 tok/s
10 tok/s                    not claimed on present evidence
```

For coding/structured output, `ngram-simple` may temporarily push some workloads above the general range, but this should be reported separately from ordinary decode.

The next project action is implementation + physical calibration, not additional speculative architecture changes unless new evidence materially changes one of these gates.