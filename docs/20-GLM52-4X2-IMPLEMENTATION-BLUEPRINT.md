# 20 — GLM-5.2 4×2 Implementation Blueprint

Date: 2026-08-18

## Goal

Turn the selected four-server / two-GPU-per-server GLM-5.2 Q3 architecture into a falsifiable implementation without replacing the existing ByteMyWave/Transit research tracks.

The first complete milestone is:

```text
one OpenAI-compatible request
-> one real GLM-5.2 Q3 model
-> weights resident across the four servers' host RAM
-> routed experts executed by their owning workers
-> hot experts accelerated from local GPU cache
-> cold experts executed locally on CPU or asynchronously prefetched
-> complete output tokens returned
-> every critical-path component timed
```

Speculation is added only after a correct non-speculative four-node baseline exists.

---

# 1. Repository isolation

Implementation should live under a new runtime namespace rather than modifying the historical proof code in place.

Proposed layout:

```text
runtime/glm52_4x2/
    coordinator/
    worker/
    atlas/
    routing/
    cache/
    prefetch/
    backends/
        cpu/
        llama_cpp/
        legacy_cuda/
    transport/
    telemetry/
    speculation/
    config/

tools/glm52_4x2/
    inventory.py
    build_atlas.py
    trace_report.py
    benchmark.py
    tune.py

tests/glm52_4x2/
    test_atlas.py
    test_routing.py
    test_cache.py
    test_protocol.py
    test_scheduler.py
    test_speculation_accounting.py
```

Existing ByteMyWave Weight Atlas/protocol concepts should be reused where compatible; do not create duplicate metadata semantics without need.

---

# 2. Component A — exact GGUF atlas

Input:

```text
GLM-5.2 Q3 GGUF shard set
```

Required output for every tensor/expert:

```json
{
  "layer": 0,
  "role": "routed_expert|shared_expert|attention|router|mtp|other",
  "expert_id": null,
  "tensor_name": "",
  "quant_type": "",
  "file_shard": "",
  "file_offset": 0,
  "bytes": 0,
  "shape": [],
  "owner_server": 0,
  "numa_node": 0,
  "gpu_cache_eligible": true
}
```

Acceptance:

- sum of atlas tensor bytes equals parsed GGUF tensor payload;
- all 78 layers accounted for;
- all 256 routed expert IDs accounted for in every sparse layer where expected;
- shared expert/attention/MTP tensors classified separately;
- no tensor name inferred from another GLM version.

---

# 3. Component B — physical inventory collector

Each server emits a machine-readable inventory containing:

```text
CPU model/socket count
NUMA nodes
RAM bytes per NUMA node
measured STREAM-like RAM bandwidth per NUMA node
GPU model, VRAM, compute capability
GPU PCIe BDF
PCIe negotiated speed/width
GPU<->NUMA affinity
P2P matrix
NVIDIA driver / CUDA runtime
NIC model/speed
small-message network latency
```

This file is immutable evidence attached to every benchmark run.

No scheduler threshold is hard-coded before these measurements exist.

---

# 4. Component C — stable worker protocol

The coordinator talks to all server workers through one hardware-independent protocol.

Minimum operations:

```text
HELLO / capabilities
LOAD_ATLAS
LOAD_SHARD
PIN_EXPERT
UNPIN_EXPERT
RUN_EXPERT
PREFETCH_EXPERT
RUN_SHARED_BLOCK
RUN_ATTENTION_BLOCK
CANCEL_PREFETCH
GET_TELEMETRY
HEALTH
```

Every `RUN_EXPERT` completion reports:

```text
queue wait us
CPU/GPU backend chosen
host bytes read
H2D bytes
H2D us
kernel us
reduction us
cache hit/miss
prefetch hit/miss
output bytes
```

The protocol must permit different worker binaries/toolchains on different servers.

---

# 5. Component D — backend isolation

## Modern llama.cpp backend

Use a pinned upstream revision that supports:

```text
GLM_DSA / GLM-5.2
GGUF Q3
required CPU backend
GPU offload
NextN/MTP when enabled
```

The exact upstream commit SHA is stored in every benchmark result.

## Legacy CUDA backend

Needed for Kepler-class K80 if retained in the final eight-device pool.

Rules:

- build with the newest source revision that can be made correct under the CUDA-11.x/K80 toolchain, or maintain a minimal backported GGML kernel worker;
- expose exactly the same ByteMyWave worker protocol;
- never force modern workers to link against the K80 toolchain;
- if a kernel cannot be supported efficiently on K80, allow the scheduler to treat that device as a cache/auxiliary accelerator or disable it for that tensor class.

## Maxwell/newer backend

Use the newest compatible CUDA-12.x/mainline path and compile for the actual compute capability.

---

# 6. Component E — distributed expert ownership

Default initial ownership:

```text
hash(layer, expert_id) -> server
```

but balanced by exact expert bytes rather than expert count alone.

Then refine placement from traces.

Objectives:

1. spread the eight selected experts of a typical layer across servers;
2. avoid concentrating globally hot experts on one server;
3. keep server RAM/NUMA balanced;
4. minimize cross-server replication initially;
5. replicate only when measured network/queue savings exceed the added RAM/cache cost.

The ownership table is versioned and stored beside each benchmark result.

---

# 7. Component F — GPU hot-expert cache

Cache state per local GPU:

```text
resident expert IDs
bytes occupied
last use
rolling frequency
estimated saved CPU time
estimated saved H2D time
predicted next-token probability
pin count / in-flight status
```

Initial policies:

```text
LRU
LFU
frequency-decay
cost-aware LFU
```

Later:

```text
MTP-aware score
```

The cache manager must support asynchronous eviction only after all dependent kernels complete.

---

# 8. Component G — CPU-vs-GPU miss scheduler

For each cold selected expert, estimate:

```text
T_cpu = queue_cpu + local_RAM_read_compute

T_gpu = queue_copy + H2D + queue_gpu + GPU_kernel
```

Choose the path that adds less time to the layer's critical path, not necessarily the path with the fastest isolated kernel.

The estimate begins from measured EWMA timings and is corrected after every execution.

This component is required before speculative prefetch because otherwise misses have no rational baseline policy.

---

# 9. Component H — asynchronous prefetch

Maintain independent queues for:

```text
mandatory current-token work
high-confidence next-layer prefetch
speculative next-token prefetch
background cache warming
```

Priority order must prevent speculative work from delaying mandatory work.

Metrics:

```text
prefetch issued
prefetch useful
prefetch late
prefetch false-positive
overfetch bytes
copy overlap percentage
```

A prefetch optimization is accepted only if wall-clock token latency falls.

---

# 10. Component I — request-level routing trace

For every generated token and sparse layer, record:

```text
selected expert IDs
server owners
cache state at selection
execution backend
latency
next-token recurrence
```

Trace-derived outputs:

```text
expert frequency histogram
per-layer hotness
transition matrix / next-token recurrence
cache simulation for LRU/LFU/etc.
server imbalance
expert replication candidates
```

This is the data source for placement tuning.

---

# 11. Component J — speculation

## Stage J1 — ngram-simple

Integrate the already-public llama.cpp path first.

Report separate throughput for:

```text
ordinary prose
structured code
repetitive/tool-format output
```

Never report only the best category.

## Stage J2 — native GLM-5.2 MTP

Pin an upstream revision where GLM_DSA NextN/MTP is functional.

Record:

```text
MTP memory footprint
num proposed tokens
num accepted tokens
acceptance histogram
verification wall time
net tok/s
```

Sweep draft depths rather than assuming five is optimal on our hardware.

## Stage J3 — MTP-aware expert prefetch

Use draft/NextN signals to prefetch likely target experts.

Critical safety property:

> Wrong prefetch predictions may waste bandwidth/cache space but must never change the target model's routing result or output distribution.

The target router remains authoritative.

---

# 12. Component K — auto-tuner

Search only measured parameters:

```text
experts cached per GPU
cache policy
CPU/GPU miss threshold
prefetch depth
prefetch confidence threshold
number of copy queues
number of CPU worker threads per NUMA node
expert replication set
MTP draft depth
ngram settings
```

Objective for the primary profile:

```text
maximize batch=1 accepted output tok/s
subject to correctness and no OOM
```

A secondary objective can optimize energy/token later.

---

# 13. Correctness requirements

Optimization may not change the target model silently.

Every implementation stage must test:

```text
same tokenizer/chat template
same GGUF/checkpoint revision
same sampling configuration
same RNG seed where deterministic comparison is possible
same router decisions when speculation/prefetch is disabled
no dropped experts
no expert pruning unless explicitly creating a different model experiment
```

Prefetch and cache prediction may change timing only.

If a lower-bit quant or custom kernel changes numerical results, it is treated as a separate model-quality experiment.

---

# 14. Benchmark schema

Every run stores at least:

```json
{
  "model": "GLM-5.2",
  "quant": "Q3-class exact build id",
  "runtime_commit": "",
  "byte_my_wave_commit": "",
  "hardware_inventory_hashes": [],
  "ownership_map_hash": "",
  "prompt_class": "prose|code|tool|repeat",
  "prompt_tokens": 0,
  "generated_tokens": 0,
  "context_tokens": 0,
  "batch": 1,
  "mtp": {},
  "cache": {},
  "network": {},
  "per_server": [],
  "per_layer": [],
  "decode_tok_s": 0.0
}
```

Raw traces are retained, not only the final average.

---

# 15. Implementation order

```text
P0  inventory collector
P1  GGUF/expert atlas
P2  one-server worker + CPU path
P3  one-server GPU cache
P4  two-GPU local scheduling
P5  coordinator + second server
P6  four-server non-speculative full model
P7  routing trace + cache tuner
P8  ngram-simple
P9  native MTP
P10 MTP-aware expert prefetch
P11 auto-tuning and final benchmark
```

Do not implement P10 before P6 is correct and measured.

---

# 16. What can be implemented without the physical cluster

Immediately implementable from public code/docs and the existing ByteMyWave repo:

```text
atlas parser/schema
ownership planner
worker protocol
scheduler state machine
cache simulator/policies
routing trace format
benchmark schema
coordinator skeleton
MTP/acceptance accounting interfaces
auto-tuner framework
legacy/modern backend capability abstraction
unit/integration tests with synthetic workers
```

---

# 17. What requires the physical cluster

Cannot be truthfully finalized without the machines:

```text
actual CUDA build compatibility for every purchased GPU
per-device Q3 kernel choice
NUMA thread counts
effective H2D bandwidth
P2P usage
CPU-vs-GPU crossover
NIC transport selection
expert-cache capacity after real runtime/KV allocations
prefetch thresholds
MTP draft depth
final tok/s
```

These are calibration tasks, not unresolved architecture questions.

---

# 18. Definition of done

The optimized runtime is considered implemented when a reproducible run on all four servers produces:

1. full GLM-5.2 Q3 generation;
2. no expert pruning;
3. exact documented model/quant revision;
4. stable four-server execution for at least a representative long generation;
5. per-token/per-layer trace;
6. measured cache hit rate;
7. measured CPU/GPU work split;
8. measured H2D/network overlap;
9. measured MTP acceptance when enabled;
10. separately reported prose/code/tool throughput;
11. complete hardware/runtime provenance;
12. a result compared against the non-speculative baseline.

The current engineering target remains `2–3 tok/s` general decode, with `3–4 tok/s` treated as a stretch target until measured.