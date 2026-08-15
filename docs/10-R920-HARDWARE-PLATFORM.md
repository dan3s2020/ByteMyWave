# 10 — Dell PowerEdge R920 as a TensorWave host platform

Date: **2026-08-15**

Status: **hardware feasibility profile + analytical simulation target**

This document evaluates the Dell PowerEdge R920 specifically as a host for the TensorWave architecture already implemented through Phases 1–4.

The conclusion is intentionally split into two parts:

1. **R920 is unusually well matched to TensorWave as a RAM/NUMA/PCIe research host.**
2. **R920 is not a native multi-GeForce GPU chassis.** Its electrical PCIe topology is much stronger than its mechanical/power support for long dual-slot RTX 3060 cards.

Nothing in this document should be read as a measured R920 benchmark. Timing values are simulation inputs until the real machine is measured with the existing Phase-4 calibration tooling.

---

## 1. Why this hardware maps unusually well to TensorWave

TensorWave currently investigates this pipeline:

```text
large compressed model in host RAM
        |
        | static Weight Atlas / execution plan
        v
NUMA-local pinned host window
        |
        | cudaMemcpyAsync / PCIe
        v
fixed compressed VRAM slot A/B
        |
        | GPU dequant
        v
reusable FP16 tile / later fused MMA fragment
        |
        v
GPU compute
```

The R920 gives us four things that directly match that architecture:

```text
very large cheap DDR3 ECC capacity
+ four CPU/memory NUMA domains
+ CPU-attached PCIe 3.0 links
+ enough PCIe topology to compare local-vs-remote and single-vs-multi-GPU streaming
```

This is more valuable to TensorWave than simply buying a newer CPU with much less RAM, because the project is explicitly trying to determine whether **large cheap host memory can become the persistent model store while VRAM remains only the active compute window**.

---

# 2. R920 platform facts

## 2.1 CPU sockets

Dell specifies the R920 for:

```text
2 or 4 Intel Xeon E7 v2 processors
E7-2800 v2
E7-4800 v2
E7-8800 v2
```

For a four-socket TensorWave machine, the useful family is E7-4800 v2 or E7-8800 v2.

Recommended cheap/high-end profile for this experiment:

```text
4 × Intel Xeon E7-4890 v2
```

Intel publishes for one E7-4890 v2:

```text
15 cores
30 threads
2.80 GHz base
3.40 GHz turbo
37.5 MB cache
155 W TDP
4 memory channels
DDR3-1066/1333/1600
85 GB/s published maximum memory bandwidth
32 PCIe 3.0 lanes
3 QPI links
4-socket scalability
```

Therefore the four-CPU profile is:

```text
60 physical CPU cores
120 hardware threads
620 W summed CPU TDP
128 CPU PCIe 3.0 lanes before platform routing constraints
4 independent CPU memory domains
```

The CPU is old and is not being selected for modern per-core performance. It is being selected because it provides **memory capacity + memory channels + PCIe root complexes** cheaply.

Official Intel reference:

- https://www.intel.com/content/www/us/en/products/sku/75251/intel-xeon-processor-e74890-v2-37-5m-cache-2-80-ghz/specifications.html

Official Dell R920 processor specification:

- https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/technical-specifications

---

## 2.2 RAM

Dell documents:

```text
96 × 240-pin DDR3 ECC DIMM sockets
registered / load-reduced ECC
1066 / 1333 / 1600 MT/s
up to 6 TB system RAM
```

For TensorWave, the important practical rule is:

```text
96 DIMM slots / 4 CPUs = 24 DIMM slots per CPU domain
```

### 1 TB target configuration

A cheap and symmetric configuration is:

```text
32 × 32 GB DDR3 ECC = 1024 GB nominal
```

Balanced across four sockets:

```text
CPU1: 8 × 32 GB = 256 GB
CPU2: 8 × 32 GB = 256 GB
CPU3: 8 × 32 GB = 256 GB
CPU4: 8 × 32 GB = 256 GB
```

That symmetry matters more than merely reaching 1 TB, because TensorWave wants a GPU's pinned host tiles to come from the RAM local to the CPU/root complex that owns the GPU.

A bad layout would be:

```text
GPU on CPU3 PCIe
       |
       v
pinned weights allocated mostly on CPU1 RAM
       |
       v
cross-socket QPI traffic
       |
       v
CPU3 PCIe
```

The desired layout is:

```text
GPU on CPU3 PCIe
       ^
       |
CPU3-local pinned RAM
```

### Model capacity perspective

TensorWave Q4 v1 uses:

```text
20 bytes / 32 weights = 0.625 bytes/parameter
```

With 1024 GiB of RAM, the raw mathematical Q4-v1 capacity is roughly:

```text
~1.76 trillion parameters
```

before subtracting OS, filesystem cache, Weight Atlas metadata, pinned staging windows, activations, KV cache, runtime buffers and other processes.

For reference:

```text
70B dense model in Q4-v1  ~= 43.75 GB
120B dense model in Q4-v1 ~= 75.00 GB
```

So 1 TB is not needed merely to hold a 70B model. Its value is that TensorWave can test much larger checkpoints, multiple representations, cache layouts, replicated workers and intentionally oversized host stores without RAM capacity being the immediate limit.

Official Dell memory references:

- https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/technical-specifications
- https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/sample-memory-configurations

---

# 3. PCIe topology — the most important hardware detail

The R920 has PCIe 3.0 and the main slots are attached to specific processors.

Dell's documented mapping is:

| PCIe slot | CPU/root | electrical link | card height | card length |
|---|---|---:|---|---|
| 1 | CPU1 | x8 | full height | half length |
| 2 | CPU1 | x8 | full height | half length |
| 3 | CPU1 | x8 | full height | half length |
| 4 | CPU2 | **x16** | full height | **full length** |
| 5 | CPU2 | **x16** | full height | half length |
| 6 | CPU3 | **x16** | full height | half length |
| 7 | CPU3 | **x16** | full height | half length |
| 8 | CPU4 | **x16** | full height | half length |
| 9 | CPU4 | **x16** | full height | half length |

Slots 6–10 require all four processors installed.

Electrically this is excellent:

```text
CPU2 -> 2 × PCIe 3.0 x16
CPU3 -> 2 × PCIe 3.0 x16
CPU4 -> 2 × PCIe 3.0 x16

CPU1 -> 3 × PCIe 3.0 x8
```

Therefore there are **six CPU-attached x16 links** available across CPU2–CPU4.

That does **not** mean six RTX 3060 cards fit in the stock chassis.

Official Dell slot mapping:

- https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/expansion-card-installation-guidelines

---

## 3.1 PCIe bandwidth

PCIe 3.0 runs at 8 GT/s per lane with 128b/130b encoding.

The theoretical one-direction payload ceiling for x16 is approximately:

```text
8 GT/s × 128/130 × 16 lanes ÷ 8 bits/byte ≈ 15.75 GB/s
```

Real CUDA H2D is lower.

TensorWave's existing Phase-4 analytical default is:

```text
12 GB/s effective pinned-memory H2D
```

The R920 simulation uses the same **12 GB/s assumption** so it stays compatible with the project's existing feasibility equations.

This is **not** presented as a measured R920 number. The first real R920 run must replace it.

---

# 4. Why old DDR3 is not automatically the bottleneck

One E7-4890 v2 has a published maximum memory bandwidth of:

```text
85 GB/s
```

One simulated RTX 3060 PCIe feed consumes:

```text
12 / 85 = 14.1%
```

Two simultaneous local GPU feeds on the same CPU would request:

```text
24 / 85 = 28.2%
```

of that published CPU maximum.

This does **not** prove that a populated R920 will deliver 85 GB/s to CUDA pinned memory. It does show why the platform is interesting: with NUMA-local allocation, the CPU's theoretical DDR3 bandwidth is substantially above the bandwidth of one or two PCIe 3.0 x16 H2D streams.

Therefore the first intended bottleneck remains exactly the one TensorWave is designed to study — **PCIe / exposed H2D latency** — rather than raw host-RAM capacity.

The dangerous case is **remote NUMA**, where a GPU on one CPU repeatedly consumes memory allocated behind another CPU and the traffic must cross the socket interconnect.

---

# 5. RTX 3060 12 GB profile

NVIDIA publishes for the 12 GB RTX 3060 family:

```text
Ampere
3584 CUDA cores
third-generation Tensor Cores
12 GB GDDR6
192-bit memory interface for the 12 GB variant
PCIe Gen 4 capable
1.78 GHz boost reference
170 W graphics-card power
2-slot reference size
~242 mm reference length
1 × 8-pin supplementary power on the reference specification
no NVLink
```

On an R920:

```text
RTX 3060 native link: PCIe 4.0 x16
R920 link:            PCIe 3.0
negotiated maximum:   PCIe 3.0 x16 when installed in a proper x16 slot
```

Official NVIDIA reference:

- https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3060-3060ti/

---

# 6. How many RTX 3060 GPUs does the R920 really take?

There are three different answers.

## 6.1 Electrical answer

With four CPUs:

```text
6 × PCIe 3.0 x16 links
+ CPU1 x8 links
```

So electrically the board topology is rich.

## 6.2 Mechanical stock-chassis answer

A standard RTX 3060 is a long, dual-slot card.

The R920 manual exposes only one normal **full-length x16** position:

```text
slot 4 -> CPU2 -> x16 -> full length
```

Slots 5–9 are documented as half-length in the standard layout.

Optional I/O risers can create other full-length positions, but those positions do not preserve the same x16 layout and Dell's manual contains generation limitations for those optional risers.

Therefore:

> **do not buy six RTX 3060s on the assumption that six x16 electrical links mean six internal RTX 3060 slots.**

A realistic stock-chassis starting point is:

```text
1 × RTX 3060 candidate in slot 4
```

and even that must pass actual AIB card dimensions, dual-slot clearance, fan/airflow behavior, auxiliary 8-pin power availability, BIOS/device enumeration and CUDA initialization.

The R920 is not documented by Dell as a GeForce GPU workstation.

## 6.3 Custom / external-riser answer

If GPUs are mounted externally or through correctly rated PCIe extensions and powered through a proper supported power path, the electrical topology becomes much more useful.

The ideal TensorWave topology for three GPUs would be:

```text
CPU2 local RAM -> x16 -> GPU0
CPU3 local RAM -> x16 -> GPU1
CPU4 local RAM -> x16 -> GPU2
```

This gives each GPU its own CPU memory controller, its own 256 GB local RAM region in the 1 TB layout and its own PCIe x16 root.

That is a far better TensorWave experiment than attaching every GPU to one socket.

The project must still treat external mounting/power as a hardware integration task, not as a capability guaranteed by Dell.

---

# 7. Power is not the same as PSU wattage

Dell documents R920 power-supply modules of:

```text
750 W
1100 W
1600 W when available
```

That does **not** automatically mean the motherboard exposes arbitrary PCIe 8-pin GPU power.

An RTX 3060 reference profile needs:

```text
170 W board power
supplementary 8-pin power
```

The practical GPU limit can therefore be set by PDB/cable availability, per-rail/current limits, redundancy mode, connector availability and thermal load before total PSU nameplate wattage becomes the limit.

TensorWave should not treat improvised GPU power wiring as part of the software experiment. The GPU power path should be known-good before any benchmark result is trusted.

Official Dell PSU reference:

- https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/power-supplies

---

# 8. Proposed TensorWave placement

## One RTX 3060

Best first experiment:

```text
CPU2
├── 256 GB local DDR3
├── TensorWave host worker pinned to CPU2
├── pinned Q4 staging allocation on CPU2 NUMA node
└── slot 4 PCIe 3.0 x16
      └── RTX 3060 12 GB
```

CPU1 can run orchestration, checkpoint parsing, Weight Atlas generation, disk/network services and non-hot control work. The hot path stays on CPU2.

## Two GPUs

The easiest software scaling with the **current code** is not tensor parallelism. It is two independent workers:

```text
Worker 0: local RAM -> GPU0 -> independent request/batch
Worker 1: local RAM -> GPU1 -> independent request/batch
```

Prefer one GPU per socket/root complex if physical integration allows it. This doubles aggregate service capacity without requiring GPU-GPU reductions, NVLink, cross-root P2P or layer synchronization.

## Three GPUs

The most attractive R920 TensorWave topology is conceptually:

```text
CPU2 + local RAM -> GPU0
CPU3 + local RAM -> GPU1
CPU4 + local RAM -> GPU2

CPU1 -> orchestration / storage / preprocessing
```

This is one reason the four-socket R920 is interesting even though CPU1's own expansion links are mostly x8.

---

# 9. Why multiple GPUs do not automatically fix dense batch=1 decode

Suppose a dense 70B model is stored in TensorWave Q4-v1.

Physical streamed bytes with no persistent cache:

```text
70B × 0.625 B = 43.75 GB / dense pass
```

At an assumed 12 GB/s:

```text
43.75 / 12 = 3.646 seconds
```

of H2D bandwidth floor for one complete dense streamed pass.

For `M=1`, the Phase-4 compute model at 10 effective TFLOP/s gives only:

```text
2 × 70B × 1 / 10 TFLOP/s = 14 ms
```

So the GPU is starved almost the entire time.

Adding more independent GPUs gives more independent streams/requests, but each request remains transfer-bound.

A perfectly equal model shard across N GPUs would reduce both streamed bytes/GPU and compute/GPU by roughly N. In an ideal mathematical world that can reduce single-step time by N.

But the current project does **not** implement that path, and a real Transformer introduces activation exchange, collectives/reduction, synchronization, graph partitioning and cross-root PCIe/QPI behavior. RTX 3060 also has no NVLink.

Therefore the first multi-GPU TensorWave implementation should be **replicated NUMA-local workers**, not tensor parallelism.

---

# 10. Simulation that follows the existing TensorWave code

Branch:

```text
hardware/r920-rtx3060-simulation-v1
```

Simulator:

```text
tools/simulate_r920_tensorwave.py
```

The simulator deliberately follows two existing project contracts.

## Phase-3 contract

From `src/tensorwave_q4_stream_proof.cu`:

```text
Q4_SYM_G32_F32S
two compressed VRAM slots
slot(i) = i % 2
copy(i) waits for compute(i-2)
compute(i) waits for copy(i)
one reusable FP16 dequant tile
one copy stream
one compute stream
```

The simulator reproduces that as a discrete-event schedule.

## Phase-4 contract

From `tools/build_feasibility_map.py`:

```text
T_transfer = P * bytes_per_param * (1-r) / H2D_bandwidth
T_compute = 2 * P * M / effective_FLOPS

M_cross = bytes_per_param * (1-r) * effective_FLOPS
          ------------------------------------------------
                       2 * H2D_bandwidth
```

---

# 11. Reference simulation workload

The project already uses dense model sizes:

```text
7B
13B
33B
70B
120B
```

For this hardware simulation we use the existing **70B dense reference point**.

This does **not** claim that MiniMax H3 is a 70B model. H3 parameter count remains governed by `docs/07-H3-RELEASE-GATE.md`.

Simulation profile:

```text
model: 70B dense reference
wire format: Q4_SYM_G32_F32S
wire bytes/param: 0.625
H2D: 12 GB/s assumed
effective GEMM: 10 TFLOP/s assumed
persistent cache: 0
K: 8192
N: 256
tiles: 32
M: 1,4,16,64,128,256,512,1024,2048
dequant time: 0 in the reference lower-bound run
```

`K=8192/N=256` is only a representative tile geometry for the schedule simulator. It is **not** asserted to be the architecture of H3 or of a specific 70B checkpoint.

The dequant time is zero in the reference simulation specifically so the result remains a lower-bound consistent with the current Phase-4 roofline. The tool accepts an explicit dequant time when a measured value exists.

---

# 12. Main simulation result

Predicted crossover:

```text
M_cross ~= 260.4
```

Model-level result:

| M | 70B Q4 stream | H2D floor | compute | hidden transfer | starvation lower bound | classification |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 43.75 GB | 3645.83 ms | 14.00 ms | 0.38% | 99.62% | transfer-bound |
| 64 | 43.75 GB | 3645.83 ms | 896.00 ms | 24.58% | 75.42% | transfer-bound |
| 128 | 43.75 GB | 3645.83 ms | 1792.00 ms | 49.15% | 50.85% | transfer-bound |
| 256 | 43.75 GB | 3645.83 ms | 3584.00 ms | **98.30%** | **1.70%** | near-balanced |
| 512 | 43.75 GB | 3645.83 ms | 7168.00 ms | 100% | 0% | compute-bound |

This directly agrees with the project's Phase-4 equation.

It says:

```text
dense batch=1 decode: bad operating point
large prefill/batch reuse: plausible operating point
```

It does not say the final end-to-end 70B runtime will hit those numbers.

---

# 13. Phase-3 ring simulation result

For:

```text
K=8192
N=256
Q4=0.625 B/weight
```

one Q4 tile is:

```text
1.25 MiB
```

The current Phase-3 fixed VRAM calculation is:

```text
X activations
+ 2 × compressed Q4 slots
+ 1 × complete FP16 dequant tile
+ Y output
```

Result:

| M | fixed VRAM | copy/tile | compute/tile | hidden | starvation |
|---:|---:|---:|---:|---:|---:|
| 1 | 6.52 MiB | 0.1092 ms | 0.0004 ms | 0.38% | 99.60% |
| 64 | 7.56 MiB | 0.1092 ms | 0.0268 ms | 24.58% | 74.83% |
| 128 | 8.63 MiB | 0.1092 ms | 0.0537 ms | 49.15% | 50.05% |
| 256 | 10.75 MiB | 0.1092 ms | 0.1074 ms | **98.30%** | **1.64%** |
| 512 | 15.00 MiB | 0.1092 ms | 0.2147 ms | 100% | 0% |
| 2048 | 40.50 MiB | 0.1092 ms | 0.8590 ms | 100% | 0% |

The important observation is not that a complete model needs only tens of MiB. The important observation is that the **transient Phase-3 weight ring is tiny relative to 12 GB VRAM** for this tile geometry.

That means an RTX 3060 12 GB gives TensorWave room to add a large persistent compressed cache while keeping the fixed streaming ring small.

---

# 14. A concrete improvement exposed by the simulation: use the rest of the 12 GB as cache

Current Phase-3 proof intentionally does not implement a persistent weight cache.

Suppose a later runtime safely reserves:

```text
8 GiB of the 12 GiB
```

for compressed persistent weights after leaving enough room for activations, KV, CUDA/cuBLAS workspace, ring slots and temporary outputs.

For a 70B Q4-v1 model:

```text
Q4 wire model = 43.75 GB
8 GiB cache ~= 8.59 GB
resident fraction ~= 19.6%
streamed bytes ~= 35.16 GB instead of 43.75 GB
M_cross ~= 209 instead of 260
```

This moves the operating boundary materially toward smaller batches/prefills.

Therefore a 12 GB RTX 3060 is more useful to TensorWave than a 4 GB GPU not merely because it has more VRAM, but because the extra VRAM can become a **persistent compressed hot-weight/expert cache** while the architecture still retains the fixed-window principle.

---

# 15. Multi-GPU result

For current-code-compatible **independent workers**, using the same analytical assumptions:

### M=1 dense reference

```text
1 GPU -> ~0.27 aggregate rows/s
2 GPU -> ~0.55 aggregate rows/s
3 GPU -> ~0.82 aggregate rows/s
```

Per-request latency is not improved by replication.

### M=256

```text
1 GPU -> ~70.22 aggregate rows/s
2 GPU -> ~140.43 aggregate rows/s
3 GPU -> ~210.65 aggregate rows/s
```

This is why the R920 is potentially much more attractive as a **multi-worker batched/prefill host** than as a machine for one dense batch=1 stream.

The full multi-GPU table, including the intentionally optimistic unimplemented equal-shard lower bound, is written by the simulator into the Phase-5 result JSON/Markdown.

---

# 16. What the simulation discovered

## Observation A — 1 TB solves capacity, not bandwidth

The R920 makes model capacity almost disappear as a constraint. It does **not** make PCIe disappear. For dense decode, more host RAM alone does not fix repeated weight streaming.

## Observation B — the NUMA layout is a first-class TensorWave object

The Weight Atlas currently knows:

```text
tensor
offset
tile
execution order
slot
```

On R920 it should eventually also know:

```text
host NUMA node
preferred GPU
preferred PCIe root
host replica/shard location
```

That means topology should become part of the execution plan, not just an OS tuning detail.

## Observation C — 12 GB VRAM changes the best optimization target

With a tiny fixed ring, most of the 12 GB can potentially be repurposed. Instead of making streaming tiles enormous, a better use is likely a **persistent compressed cache**, because residency directly decreases streamed bytes and the Phase-4 crossover.

## Observation D — two x16 links per socket are more than enough electrically

At the assumed 12 GB/s/GPU:

```text
2 GPU feeds = 24 GB/s
```

against the E7-4890 v2 published maximum:

```text
85 GB/s host-memory bandwidth
```

So a two-GPU-per-socket electrical topology is not absurd from a memory-bandwidth perspective. The stock R920 mechanical layout is the larger problem for RTX 3060.

## Observation E — old CPU performance is not the first reason to reject the server

If Weight Atlas is prebuilt, runtime schedule is static, the hot path does not search tensors, GPU does dequant/math and DMA performs H2D, then CPU work should be deliberately kept outside the critical per-tile path.

The E7 v2 CPUs are therefore acceptable for the experiment unless measurements show CPU scheduling/pinned-memory overhead becoming visible.

## Observation F — current code is naturally replicated, not naturally tensor-parallel

The existing CUDA proof owns one CUDA device, one copy stream, one compute stream and one pair of Q4 slots. The clean extension is:

```text
N devices
-> N independent instances of that state
-> N NUMA-local host queues
```

before adding distributed collectives.

---

# 17. Recommended implementation changes

## P0 — before buying multiple GPUs

1. Validate one RTX 3060 physically in the actual R920 configuration.
2. Validate an approved/known-good 8-pin power path.
3. Verify `nvidia-smi`, negotiated PCIe width/speed and CUDA device operation.
4. Measure pinned H2D locally.
5. Measure pinned H2D from every remote NUMA node.

If local and remote are indistinguishable, our NUMA assumptions need revision. If remote is materially worse, the runtime must enforce affinity.

## P1 — NUMA-aware runtime

Add a hardware-topology layer that records:

```text
GPU index
PCI bus ID
PCIe width/speed
root CPU / NUMA node
host allocation NUMA node
measured local H2D GB/s
measured remote H2D GB/s
```

Then allocate each pinned Q4 pack/staging window on the GPU-local node.

## P2 — persistent compressed cache

Add:

```text
fixed stream ring
+ persistent Q4 cache
```

Metrics:

```text
cache hit rate
bytes avoided
cache residency GB
cache eviction count
starvation with/without cache
```

Do not optimize cache hit rate alone. Optimize wall time/starvation.

## P3 — one worker per GPU

Implement:

```text
GPU worker 0 -> NUMA node A
GPU worker 1 -> NUMA node B
GPU worker 2 -> NUMA node C
```

Start with independent jobs/batches. This is low-risk because it preserves the current single-GPU execution semantics.

## P4 — only then test one model across multiple GPUs

Candidate approaches:

```text
pipeline by layer ranges
expert ownership for MoE
tensor parallelism
```

Preference order for R920 should likely be:

```text
MoE expert ownership
> pipeline by layer range
> fine-grained tensor parallelism
```

because avoiding frequent cross-GPU collectives is especially valuable on separate PCIe roots without NVLink.

---

# 18. What could make the R920 a bad purchase for TensorWave?

The hardware should be rejected or reconsidered if any of these become true on the actual unit:

```text
RTX 3060 cannot be powered safely/cleanly
slot 4 cannot physically cool the installed AIB card
BIOS/firmware refuses or destabilizes the GPU
measured local pinned H2D is unexpectedly poor
NUMA remote traffic cannot be controlled
DDR3 population drops local bandwidth enough to starve PCIe
power draw/noise is unacceptable for the deployment
we require 3–4 internal long dual-slot GPUs rather than external/custom mounting
```

The last point is important:

> if the final requirement becomes “four normal RTX 3060 cards entirely inside one stock chassis,” an R920 is the wrong chassis even though its CPU/RAM/PCIe topology is excellent.

A purpose-built GPU server would then be the better mechanical platform.

---

# 19. Verdict

For TensorWave as it exists today:

```text
RAM host / model store:          EXCELLENT
cheap capacity per leu:          EXCELLENT
NUMA experimentation:            EXCELLENT
PCIe electrical topology:        EXCELLENT
single RTX 3060 research node:   PROMISING, MUST VALIDATE POWER/FIT
multi-RTX3060 stock chassis:     POOR / NOT NATIVE
multi-GPU external integration:  PROMISING
dense batch=1 70B:               TRANSFER-BOUND
prefill / batching:              STRONGER TARGET
MoE / expert-local sharding:     VERY INTERESTING NEXT TARGET
```

The R920 is therefore not attractive because it is a fast modern server. It is attractive because it is almost a **physical embodiment of the TensorWave hypothesis**:

```text
enormous cheap host memory
+ several independent memory domains
+ direct PCIe roots
+ small GPU working windows
```

The decisive purchase question is not RAM capacity. It is:

> **Can the actual R920 unit cleanly host and power the GPU configuration we want while preserving NUMA-local pinned H2D?**

That is the next hardware gate.

---

# 20. Reproduce the analytical simulation

From branch:

```text
hardware/r920-rtx3060-simulation-v1
```

Run:

```powershell
python .\tools\simulate_r920_tensorwave.py `
  --output-dir .\runs\r920-reference
```

Explicit reference parameters:

```powershell
python .\tools\simulate_r920_tensorwave.py `
  --output-dir .\runs\r920-reference `
  --model-b 70 `
  --host-ram-gib 1024 `
  --h2d-gbps 12 `
  --effective-tflops 10 `
  --cache-gib-per-gpu 0 `
  --k 8192 `
  --n 256 `
  --tiles 32 `
  --m-values "1,4,16,64,128,256,512,1024,2048" `
  --gpu-counts "1,2,3" `
  --dequant-us-per-tile 0
```

Cache sensitivity:

```powershell
python .\tools\simulate_r920_tensorwave.py `
  --output-dir .\runs\r920-cache8 `
  --cache-gib-per-gpu 8
```

Once the real R920 exists, the assumed timing inputs must be replaced by the measured calibration produced by the existing Phase-4 hardware runner.
