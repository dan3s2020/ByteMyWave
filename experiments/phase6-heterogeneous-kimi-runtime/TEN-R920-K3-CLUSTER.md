# 10x R920 + 10x RTX 3060 + 50x worker Kimi K3 cluster

Date: **2026-08-16**

Status: **analytical end-state for the R920 architecture; not a measured K3 benchmark**

This document captures the largest R920 topology currently considered economically sensible before changing architecture class. It also records an important correction to the older Phase-6 K3 arithmetic.

> **Canonical correction:** older Phase-6 files that derive K3 routed experts as `3 * 7168 * 3072` are wrong for Kimi K3. K3 uses Stable LatentMoE: routed experts operate on the **3584 latent MoE dimension**, not directly on hidden size 7168. Until the older K3 sections are rewritten, the K3 routed arithmetic in this file supersedes them.

No throughput value below is a physical measurement unless explicitly labeled. Values are divided into official facts, user price inputs, project planning assumptions, analytical ceilings, and engineering targets.

---

## 1. Target configuration

```text
10 x Dell PowerEdge R920

per R920:
    4 x Xeon E7-4890 v2 class CPU sockets
    1 x RTX 3060 12 GB
    5 x cheap 6 GB NVIDIA expert workers (GTX 1060-class planning worker)
    distributed DDR3 shard
    low-latency cluster NIC

cluster totals:
    10 x RTX 3060 12 GB = 120 GB aggregate GDDR6
    50 x 6 GB worker GPUs = 300 GB aggregate worker VRAM
    40 x E7 CPU sockets
    ~2 TB DDR3 under the stated 10,000 lei RAM budget
```

The 120 GB of RTX 3060 VRAM is **not unified memory**. The fixed K3 path must be explicitly tensor/model-sharded across the ten GPUs.

Likewise, the 50 workers improve one-request latency only if expert work is genuinely sharded so that all worker feeds participate in the same token/layer. Replication would improve aggregate serving throughput, not single-request tok/s.

---

## 2. R920 PCIe topology: why 1x3060 + 5 workers/node is electrically plausible

Dell's R920 expansion-card guide gives six native PCIe links at x16 width with four processors installed:

```text
slot 4 -> Processor 2 -> x16
slot 5 -> Processor 2 -> x16
slot 6 -> Processor 3 -> x16
slot 7 -> Processor 3 -> x16
slot 8 -> Processor 4 -> x16
slot 9 -> Processor 4 -> x16
```

Therefore one x16 link can be assigned to the RTX 3060 and the remaining five to expert workers.

This is an **electrical topology result, not a stock mechanical-fit claim**. Ten R920s containing sixty consumer GPUs require external risers/cabling, additional power delivery and custom cooling/open-frame mounting. The R920 chassis is not assumed to physically accept six dual-slot consumer cards internally.

Dell source:
- https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/expansion-card-installation-guidelines

---

## 3. RAM budget and population

User price input:

```text
16 GB DDR3 RDIMM = 80 lei
RAM budget        = 10,000 lei
```

Arithmetic:

```text
10,000 / 80 = 125 DIMMs
125 * 16 GB = 2,000 GB nominal RAM
```

A simple balanced planning layout is:

```text
12 x 16 GB per R920 = 192 GB/node
10 nodes            = 1,920 GB installed
cost                = 9,600 lei
5 DIMMs              = 80 GB spare / validation stock
```

The official Kimi K3 Hugging Face repository currently reports about **1.56 TB** for the released checkpoint, so 1.92 TB nominal host RAM passes the raw checkpoint capacity gate and leaves roughly 360 GB nominal headroom before OS/runtime/pinned-buffer overhead.

K3 checkpoint source:
- https://huggingface.co/moonshotai/Kimi-K3/tree/main

### 16 GB R920 compatibility

Dell's detailed R920 memory documentation explicitly states support for **16 GB RDIMMs** and shows 16 GB sample configurations. The short technical-specification table elsewhere in the manual is inconsistent because it omits 16 GB while the detailed memory section includes it. Exact rank/voltage/part-number compatibility must still be verified before a bulk purchase.

Dell memory source:
- https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/system-memory
- https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/sample-memory-configurations

### Important bandwidth caveat

Twelve DIMMs per four-socket R920 is a capacity-first population, not Dell's maximum-bandwidth population. Dell recommends population in channel-balanced groups for best performance. Therefore the assumed worker feed and CPU expert bandwidth must be benchmarked with the actual 12-DIMM/node layout.

A clean higher-cost alternative is to buy enough DIMMs to improve channel population. At the same 80 lei/DIMM, an extra 35 DIMMs cost 2,800 lei. The exact optimum population should follow Dell's memory-riser/channel rules rather than simply filling arbitrary sockets.

---

## 4. Critical K3 correction: routed experts use latent 3584

Official Moonshot K3 facts:

```text
total parameters             = 2.8T
activated parameters/token   = 104B (published rounded value)
num hidden layers            = 93
first dense layers           = 1
MoE layers                   = 92
hidden size                  = 7168
latent MoE dimension         = 3584
MoE intermediate/expert dim  = 3072
routed experts               = 896
selected experts/token       = 16
shared experts               = 2
```

Moonshot's model implementation constructs routed experts with `hidden_size = routed_expert_hidden_size`, and explicitly applies:

```text
routed_expert_down_proj: 7168 -> 3584
routed experts:          3584 -> 3072 -> 3584
routed_expert_up_proj:   3584 -> 7168
```

Official sources:
- https://huggingface.co/moonshotai/Kimi-K3/blob/main/README.md
- https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json
- https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py

Therefore one selected routed expert contains:

```text
3 * 3584 * 3072
= 33,030,144 weights
```

Selected routed expert weights per token:

```text
92 layers * 16 experts * 33,030,144
= 48,620,371,968
= 48.620371968B routed weights/token
```

The previous Phase-6 value `97.240743936B` incorrectly used 7168 as the expert input/output dimension and is approximately 2x too high.

Using Moonshot's rounded 104B activated figure, the planning remainder is:

```text
104B - 48.620371968B
= 55.379628032B active non-routed/planning remainder
```

Because 104B is rounded, this 55.3796B number is a planning remainder, not a byte-exact checkpoint census.

---

## 5. Native MXFP4 routed traffic

K3's official quantization config uses 4-bit grouped routed Linear weights with group size 32 and `uint8` scale. As a payload model:

```text
32 weights * 4 bits = 16 bytes
+ 1 byte scale      = 17 bytes/group
17 / 32             = 0.53125 bytes/weight
```

This excludes container/alignment/metadata overhead and is therefore an analytical payload model, not a measured DMA byte count.

Routed payload per normal decode token:

```text
48.620371968B * 0.53125 B/weight
= 25.829572608 GB/token
```

This `25.83 GB/token` is the central routed-expert bandwidth number for this document.

---

## 6. Expert fabric: 50 workers

Project planning assumption retained from Phase 6:

```text
one clean PCIe Gen3 x16 worker feed
= 12 GB/s sustained compressed H2D
```

This is **not measured on the physical R920**.

Fifty workers give:

```text
50 * 12 GB/s = 600 GB/s aggregate H2D
```

At the native-MXFP4 payload model:

```text
600 / 0.53125
= 1129.41 Gweights/s
```

Routed-expert ceiling with workers only:

```text
1129.41 / 48.62037
= 23.23 routed token-equivalents/s
```

This is not end-to-end K3 tok/s. It is the ideal expert-service ceiling before kernel inefficiency, RAM contention, routing imbalance, network synchronization and the fixed KDA/MLA path.

---

## 7. CPU contribution sensitivity

The E7-4890 v2 is an old AVX-only CPU. Its real low-bit expert throughput is unknown. Intel publishes:

```text
15 cores / 30 threads
2.8 GHz base / 3.4 GHz turbo
155 W TDP
4 max memory channels
85 GB/s published max memory bandwidth
AVX, but no AVX2/AVX-512/AMX
```

Intel source:
- https://www.intel.com/content/www/us/en/products/sku/75251/intel-xeon-processor-e74890-v2-37-5m-cache-2-80-ghz/specifications.html

Let:

```text
c = measured routed expert kernel throughput in Gweights/s/socket
```

There are 40 sockets across ten servers. Expert capacity becomes:

```text
C_expert = 1129.41 + 40*c   Gweights/s
TPS_routed_ceiling = C_expert / 48.62037
```

Sensitivity:

| measured CPU kernel `c` | routed-expert analytical ceiling |
|---:|---:|
| 0 Gweights/s/socket | 23.23 tok-eq/s |
| 5 | 27.34 tok-eq/s |
| 10 | 31.46 tok-eq/s |
| 20 | 39.68 tok-eq/s |
| 30 | 47.91 tok-eq/s |

The 20–30 Gweights/s rows are **sensitivity points**, not claims that E7-4890 v2 achieves them.

---

## 8. What the ten RTX 3060 GPUs do

The RTX 3060 fleet must not be treated as one 120 GB unified GPU. The fixed K3 path must be deliberately sharded.

Preferred roles:

```text
10 x RTX 3060:
    KDA / MLA fixed path shards
    routing
    latent-MoE 7168<->3584 projection shards
    dense layer / lm_head shards where appropriate
    orchestration and reductions
    optional speculative-draft components if capacity permits

50 x 6 GB workers:
    routed-expert tiles streamed from local DDR3
    optional persistent shared-expert shards

40 x E7 sockets:
    local routed-expert work in parallel with GPU worker feeds
```

NVIDIA confirms the RTX 3060 12 GB model uses Ampere, 3584 CUDA cores, 12 GB GDDR6 and a 192-bit memory interface; reference graphics-card power is 170 W. It has no NVLink.

NVIDIA source:
- https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/

### Fixed-path capacity

Using the rounded active planning remainder:

```text
55.3796B active fixed/non-routed weights
```

At BF16 this is approximately 110.8 GB of weight reads/resident payload across the fleet, leaving almost no aggregate VRAM for runtime if all of it must remain on the ten 12 GB cards.

Therefore a practical design needs one or more of:

```text
1. custom 8-bit/mixed quantization for fixed-path weights;
2. persistent placement of shared experts on the 50 workers;
3. selective host residency for low-frequency/non-critical tensors;
4. tighter KV/workspace budgeting;
5. model-specific fused kernels that avoid large temporary materializations.
```

A planning 8-bit representation of the 55.38B remainder is ~55.38 GB total, or ~5.54 GB/RTX3060 when perfectly balanced, leaving about 6.46 GB/card for runtime/KV/workspace. Quality and kernel speed must be measured; the official checkpoint intentionally leaves several fixed-path module classes outside routed MXFP4 quantization.

### Fixed-path bandwidth roofline

The project has historically used approximately **360 GB/s/card** as the RTX-3060-class local VRAM planning input. This is a planning number, not a measured TensorWave sustained GEMV result.

Ten cards therefore provide a notional:

```text
10 * 360 GB/s = 3.6 TB/s aggregate local VRAM bandwidth
```

If the fixed path were represented at 1 byte/weight and perfectly tensor-sharded:

```text
55.38 GB/token
3.6 TB/s / 55.38 GB
= ~65 fixed-path token-equivalents/s raw roofline
```

This says only that the fixed path need not be the dominant bandwidth bottleneck after aggressive 8-bit placement. It does **not** prove 65 K3 tok/s because ten consumer GPUs without NVLink require frequent cross-node reductions.

---

## 9. Distributed execution topology

The design must shard each dependent layer so that all nodes participate in one request.

A high-level token/layer path is:

```text
            fixed path / router shards on 10 x RTX3060
                          |
                          | compact hidden/latent activations
                          v
        +---------------------------------------------+
        | 10 x R920 local expert domains             |
        |                                             |
        | each node:                                  |
        |   DDR3 shard                                |
        |   5 x GPU worker feeds                      |
        |   4 x CPU expert engines                    |
        +---------------------------------------------+
                          |
                          | compact partial results
                          v
               cross-node reduction / next layer
```

Do **not** assign whole sequential layer ranges to different R920s if the goal is maximum single-request tok/s; that would pipeline nodes but leave most expert bandwidth idle for one decode stream.

Do **not** move selected expert weights across the cluster network. The network should carry hidden/latent activations, route metadata and reductions. Weights stay NUMA-local.

---

## 10. Network is the main scaling risk after bandwidth

K3 has 92 sequential MoE layers. At a target output rate:

```text
25 tok/s -> 40.0 ms/token -> 435 us/layer total budget
35 tok/s -> 28.6 ms/token -> 311 us/layer total budget
```

Those budgets include **everything**, not merely network RTT.

Therefore ordinary high-latency Ethernet orchestration is not sufficient for the stretch target. This cluster needs a low-latency fabric and a runtime designed around preallocated pinned buffers, asynchronous transfer, batched collectives and minimal host scheduling jitter.

Candidate class:

```text
InfiniBand / RoCE / low-latency 25-100 GbE-class fabric
```

The exact NIC/switch is intentionally not frozen here because GPUDirect/RDMA behavior with consumer RTX 3060 GPUs and this old PCIe/NUMA platform must be tested rather than assumed.

The important acceptance metrics are:

```text
p50/p95 one-layer activation dispatch latency
p50/p95 reduction latency
sustained concurrent H2D while NIC traffic is active
CPU NUMA locality under NIC + GPU load
number of collectives required per K3 layer
```

---

## 11. Custom kernels/runtime required

This architecture is not expected to perform with a generic offload runtime.

Minimum required software pieces:

### A. E7 AVX expert kernel

```text
packed MXFP4/Q4 tile
-> fused unpack/scale
-> gate/up dot products
-> SiTU/SwiGLU-family activation as required by K3 implementation
-> down projection
-> partial output
```

The existing Phase-6 AVX1 benchmark is a correctness baseline, not the final kernel.

### B. GTX-1060-class worker kernel

```text
pinned local DDR3
-> async compressed PCIe DMA
-> decode in registers/shared memory
-> direct low-bit GEMV/GEMM
-> no full FP16 expert materialization
```

### C. RTX 3060 fixed-path kernels/runtime

Need model-specific low-batch KDA/MLA, router, latent projection and fixed-path 8-bit/mixed-weight execution that works efficiently on Ampere.

### D. distributed layer scheduler

Need explicit ownership tables:

```text
tensor/expert -> R920 node -> NUMA domain -> CPU/GPU engine
```

and preallocated asynchronous activation/reduction buffers.

### E. speculative verification only after normal decode works

Speculative decoding must not be modeled as a free multiplier. A block of draft tokens can select many more unique routed experts than one normal token.

Define:

```text
A = accepted output tokens / verification block
U = unique routed-expert weight traffic of one verification block
    divided by one normal-token routed traffic
```

Then expert-side speculative throughput scales roughly with `A/U`, not simply `A`.

Measure `U` on real K3 routing traces before using speculative tok/s in a purchase decision.

---

## 12. Cost model

User-supplied / conversation planning prices:

```text
R920                   = 1,000 lei each
6 GB worker            =   500 lei each
16 GB DDR3             =    80 lei each
RTX 3060 12 GB         = 1,000 lei planning placeholder
```

Base cluster:

| item | quantity | planning cost |
|---|---:|---:|
| R920 | 10 | 10,000 lei |
| 6 GB workers | 50 | 25,000 lei |
| RTX 3060 12 GB | 10 | 10,000 lei |
| RAM budget | 125 x 16 GB max budget | 10,000 lei |
| **base subtotal** | | **55,000 lei** |

The `1,000 lei/RTX3060` line is a placeholder from the planning conversation, not a current-market quote.

Not included:

```text
10 low-latency NICs
switch/fabric
external GPU risers
external/additional PSUs and distribution
rack/open-frame mechanical work
boot/storage drives
cabling
cooling / HVAC
spares
shipping
labor/rework
```

Therefore **55k lei is not a finished installed-system price**.

---

## 13. Power model

Reference/planning component powers:

```text
50 x GTX 1060-class worker @ 120 W = 6.00 kW
10 x RTX 3060 @ 170 W              = 1.70 kW
40 x E7-4890 v2 @ 155 W            = 6.20 kW
-------------------------------------------------
CPU+GPU component subtotal          = 13.90 kW
```

Official sources:
- GTX 1060 120 W: https://www.nvidia.com/en-us/geforce/news/nvidia-geforce-gtx-1060/
- RTX 3060 170 W: https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/
- E7-4890 v2 155 W: https://www.intel.com/content/www/us/en/products/sku/75251/intel-xeon-processor-e74890-v2-37-5m-cache-2-80-ghz/specifications.html

This excludes RAM, memory buffers, chipset, NICs, drives, fans and PSU losses. A conservative facility planning envelope is therefore roughly:

```text
~16-20 kW at the wall under sustained heavy load
```

This is an engineering provision estimate, not a measured wall figure.

At 230 V, 16-20 kW corresponds to roughly 70-87 A if absurdly placed on a single phase; practical installation therefore requires proper multi-circuit/three-phase power design. Cooling must remove essentially the same electrical power as heat.

At continuous full load for a 30-day month:

```text
16 kW -> 11.52 MWh/month
20 kW -> 14.40 MWh/month
```

HVAC/cooling electricity is additional.

---

## 14. Throughput estimate: what is defensible today

### Analytical internal ceilings

```text
worker-only routed fabric      ~23.23 tok-eq/s
+ CPU c=10 Gw/s/socket         ~31.46 tok-eq/s
+ CPU c=20 Gw/s/socket         ~39.68 tok-eq/s
+ CPU c=30 Gw/s/socket         ~47.91 tok-eq/s
```

The ten-3060 fixed path, after successful 8-bit/mixed quantization and tensor sharding, has a bandwidth roofline above the worker-only routed fabric under the 360 GB/s/card planning input. It is therefore plausible for the routed fabric/network to become the dominant limit.

### End-to-end engineering targets

Before physical benchmarks, the following labels are the most honest:

```text
first useful target:       ~15-25 output tok/s
stretch target:            ~25-35 output tok/s
speculative stretch:       ~30-45 output tok/s ONLY if measured A/U is favorable
100 tok/s:                 not a credible target for this R920 architecture
```

The 15-35 range is **not derived by multiplying one bandwidth number by a fixed efficiency**. It is a planning envelope acknowledging:

```text
50 concurrent H2D feeds may not sustain 12 GB/s each
DDR3 contention and memory-buffer topology
AVX1 low-bit CPU kernel uncertainty
GTX 1060 low-bit kernel efficiency
3060 fixed-path quantization quality/performance
92 sequential MoE synchronization points
cross-node collectives
consumer GPU / RDMA limitations
routing imbalance
runtime scheduling overhead
```

Do not present 15, 25, 35 or 45 tok/s as benchmarks until measured.

---

## 15. Why this is the end of the R920 line for ~100 tok/s

At 100 normal decode tokens/s, routed experts alone require:

```text
25.8296 GB/token * 100
= 2.583 TB/s effective routed-weight service
```

The 50-worker fabric supplies only:

```text
600 GB/s planned compressed H2D
```

Even optimistic CPU contribution does not close the ~4x gap, and 100 tok/s provides only:

```text
10 ms/token / 92 MoE layers
= ~109 us/layer
```

for router + dispatch + expert execution + reduction + continuation.

Therefore the economical R920 design may be useful in the tens-of-tokens/s regime, but a credible 100 tok/s architecture needs a different memory hierarchy: much larger resident accelerator memory/HBM and a scale-up interconnect so expert weights remain beside compute instead of being repeatedly streamed over PCIe.

The R920 work is still valuable because it establishes the low-cost heterogeneous runtime, low-bit kernels, expert ownership, activation transport and measurement framework.

---

## 16. Purchase/measurement gates before building ten nodes

Do not buy the complete 10-node cluster before one-node validation.

### Gate 1 — one R920 RAM

```text
validate exact 16 GB DIMM part number/rank/voltage
measure STREAM/local NUMA bandwidth with intended population
```

### Gate 2 — one worker feed

```text
>= 10 GB/s sustained compressed pinned H2D minimum
12 GB/s remains target planning input
```

### Gate 3 — five workers simultaneously

```text
measure all 5 x16 feeds concurrently
confirm aggregate host-memory feed
measure per-NUMA contention
```

### Gate 4 — E7 expert kernel

```text
measure Gweights/s/socket on actual K3 3584->3072->3584 expert shape
measure combined CPU + GPU-worker load
```

### Gate 5 — one RTX 3060 fixed-path shard

```text
measure K3 fixed-path 8-bit/mixed kernel bandwidth
measure workspace/KV headroom inside 12 GB
```

### Gate 6 — two R920 network prototype

```text
measure per-layer dispatch + reduction RTT
measure p50/p95/p99
run network while all local GPU feeds are active
```

### Gate 7 — only then scale 2 -> 4 -> 10 nodes

Each scale step must demonstrate single-request latency improvement. If throughput scales only for multiple independent requests, the sharding/runtime is wrong for the project objective.

---

## 17. Frozen conclusion

The best current R920 end-state is:

```text
10 x R920
10 x RTX 3060 12 GB fixed-path shards
50 x 6 GB expert workers
40 x E7 CPU expert engines
~2 TB distributed DDR3
low-latency network
custom low-bit CPU/GPU kernels
```

Planning economics:

```text
~55,000 lei base hardware under conversation prices
+ networking/power/riser/cooling infrastructure
~16-20 kW facility load planning envelope
```

Performance status:

```text
23.23 routed tok-eq/s from workers alone (analytic)
31-48 routed tok-eq/s with hypothetical measured CPU contributions
~15-25 output tok/s initial engineering target
~25-35 output tok/s stretch target
~30-45 output tok/s speculative stretch only after A/U measurement
```

This topology is documented as the **practical capstone of the cheap R920/DDR3 approach**, not as a path to 100 tok/s.
