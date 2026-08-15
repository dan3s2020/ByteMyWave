# TensorWave — Current Project State

Last updated: **2026-08-15**

This file is the fast synchronization point for contributors. Read it before starting a new branch.

## Current objective

Determine the **measured operating envelope** in which a GPU with a small fixed VRAM working set can productively execute operations over a much larger model resident in host RAM.

The project optimizes:

```text
correctness
bounded VRAM working set
physical H2D bytes
GPU starvation / unhidden transfer
wall time
```

not merely “the model fits.”

---

## Current branch chain

```text
main
  |
  +-- bench/h2d-overlap-proof-v1              # PR #1 — Phase 1
        |
        +-- model/real-weight-atlas-proof-v1  # PR #2 — Phase 2
              |
              +-- quant/q4-streaming-proof-v1 # PR #4 — Phase 3
                    |
                    +-- research/feasibility-map-v1 # Phase 4
                          |
                          +-- hardware/r920-rtx3060-simulation-v1 # Phase 5
```

Each phase targets its immediate parent so reviewers can isolate one architectural increment.

---

## Phase 0 — project definition

Status: **captured**

Preserved context:

- original intent;
- model-memory mental model;
- Weight Atlas concept;
- fixed-VRAM streaming concept;
- compression/dequantization concept;
- collaboration rules;
- original transcript + dated continuation segments.

Transcript index:

- `docs/TRANSCRIPT-INDEX.md`

---

## Phase 1 — synthetic transfer/compute overlap

Branch: `bench/h2d-overlap-proof-v1`

Status: **implementation ready; target-hardware measurement required**

Implements:

- pinned host weights;
- real FP16 cuBLAS GEMM;
- two fixed VRAM weight slots;
- separate copy/compute streams;
- CUDA-event slot ownership;
- sequential vs overlapped execution;
- explicit GPU starvation measurement.

Strong-support criterion:

```text
correctness_ok = true
steady_starvation_pct <= 10%
steady_hidden_transfer_pct >= 80%
```

---

## Phase 2 — real checkpoint bytes

Branch: `model/real-weight-atlas-proof-v1`

Status: **Python tooling and CUDA compile validated; target-hardware measurement required**

Implements:

- `tools/safetensors_atlas.py`
  - header-only safetensors scan;
  - exact tensor names/shapes/dtypes/byte offsets;
  - no payload materialization.

- `tools/pack_stream_tiles.py`
  - deterministic homogeneous rank-2 tensor/tile selection;
  - exact raw row slices;
  - F16/BF16;
  - SHA-256 provenance.

- `tools/build_runtime_schedule.py`
  - static fixed-slot schedule;
  - no tensor lookup required in the measured hot path.

- `src/tensorwave_real_weight_proof.cu`
  - real checkpoint bytes in pinned RAM;
  - two fixed VRAM weight buffers;
  - F16/BF16 cuBLAS;
  - sequential vs overlapped correctness and timing.

- `scripts/run-real-weight-proof.ps1`

---

## Phase 3 — Q4 host store + compressed H2D + GPU dequant

Branch: `quant/q4-streaming-proof-v1`

Status: **implemented; Python validation passes; CUDA compile/hardware measurement tracked separately**

Current proof format:

```text
Q4_SYM_G32_F32S
32 weights/group
4-byte float32 scale
16 packed signed-int4 bytes
20 bytes/group
```

Physical density:

```text
20 / 32 = 0.625 bytes/weight
5 effective bits/weight including scale
3.2x fewer bytes than a 16-bit source
```

Implements:

- offline F16/BF16 -> Q4 pack;
- per-tile SHA-256 and corruption verifier;
- two fixed compressed Q4 VRAM slots;
- one reusable FP16 dequant tile;
- CUDA dequant kernel;
- cuBLAS GEMM;
- separate compressed-H2D/dequant/GEMM/starvation metrics;
- source-equivalent feed-rate metric.

Phase 3 intentionally materializes one complete FP16 tile after dequantization so each cost can be measured independently.

Issue #5 defines the later fused path:

```text
Q4 tile -> decode only MMA fragment -> shared/registers -> Tensor Core MMA
```

with no complete FP16 weight tile.

---

## Phase 4 — Feasibility Map

Branch: `research/feasibility-map-v1`

Status: **analytical map + hardware experiment runner implemented; CI validation in progress/required**

### Purpose

Turn “does TensorWave work?” into a measurable operating envelope.

Main variables:

```text
M / activation rows / batch / prefill reuse
K,N tile geometry
physical wire bytes/parameter
measured H2D GB/s
measured effective GEMM TFLOP/s
dequant time
resident/cache fraction
active parameter fraction
```

### Key equation

For ideal dense-linear streaming:

```text
M_cross = bytes_per_param * (1-resident_fraction) * effective_FLOPS
          ---------------------------------------------------------
                         2 * H2D_bandwidth
```

At `M << M_cross`, the system is expected to be transfer-bound.

At `M ~ M_cross`, compute and H2D are balanced.

At `M > M_cross`, H2D can theoretically be hidden beneath compute.

Important consequence: total dense model parameter count cancels from the ideal overlap ratio, although it still controls absolute latency/RAM requirements.

### Phase-4 tooling

- `tools/build_feasibility_map.py`
  - JSON/CSV/Markdown/SVG maps;
  - dense model sizes and M sweep;
  - residency and active-fraction controls.

- `tools/calibrate_feasibility_map.py`
  - extracts measured physical H2D GB/s;
  - estimates effective GEMM TFLOP/s from correct TensorWave runs;
  - handles Phase-2 and Phase-3 schemas.

- `tools/aggregate_feasibility_runs.py`
  - combines multiple TileN / 16-bit / Q4 sweeps.

- `scripts/run-feasibility-experiments.ps1`
  - real checkpoint -> Q4 M sweep -> calibration -> measured map;
  - optional matching Phase-2 16-bit comparison;
  - TileN sweep;
  - master manifest/CSV/Markdown aggregation.

- `maps/default/FEASIBILITY-MAP.md`
  - committed analytical baseline;
  - assumptions are explicit and must not be presented as hardware measurements.

- `experiments/phase4-feasibility-map/README.md`
  - E1 M crossover;
  - E2 tile-size crossover;
  - E3 16-bit vs Q4;
  - E4 dense-LLM bandwidth floor;
  - E5 MoE/cache map;
  - E6 prediction-vs-measurement validation.

---

## Phase 5 — R920 + RTX 3060 hardware profile and simulation

Branch: `hardware/r920-rtx3060-simulation-v1`

Status: **hardware topology documented + analytical/discrete-event simulator implemented; real R920 measurement still required**

### Hardware profile

Reference target:

```text
Dell PowerEdge R920
4 x Intel Xeon E7-4890 v2
~1 TiB balanced DDR3 ECC
RTX 3060 12 GB as first GPU candidate
```

Detailed analysis:

- `docs/10-R920-HARDWARE-PLATFORM.md`

Key hardware conclusion:

```text
R920 RAM/NUMA/PCIe topology: excellent match for TensorWave experiments
R920 stock multi-GeForce mechanics/power: not native; validate one card first
```

The electrical topology exposes multiple CPU-attached x16 links, but this is deliberately not treated as equivalent to an equal number of internal long dual-slot RTX 3060 positions.

### Simulator

- `tools/simulate_r920_tensorwave.py`
- `tests/test_r920_simulator.py`
- `experiments/phase5-r920-hardware-simulation/README.md`
- committed reference result under `experiments/phase5-r920-hardware-simulation/results/reference/`

The simulator follows the existing Phase-3 two-slot schedule:

```text
slot(i) = i % 2
copy(i) waits compute(i-2)
compute(i) waits copy(i)
```

and the existing Phase-4 crossover model. It does **not** claim to emulate the complete GPU microarchitecture or end-to-end Transformer graph.

### Reference workload

The simulation uses the already-supported generic **70B dense reference point**:

```text
70B dense
Q4 v1 = 0.625 B/param
43.75 GB streamed if resident_fraction=0
12 GB/s H2D assumption
10 TFLOP/s effective dense-linear assumption
K=8192
N=256
32 tiles
M = 1..2048 sweep
```

This does **not** claim MiniMax H3 is 70B. The H3 release gate remains authoritative.

### Main reference findings

```text
predicted M_cross ~= 260.4
M=1:   ~99.6% starvation lower bound
M=256: ~1.7% starvation lower bound / near-balanced
M=512: idealized compute-bound
```

For the current Phase-3 ring at `K=8192,N=256`:

```text
Q4 tile = 1.25 MiB
fixed VRAM at M=256 = 10.75 MiB
fixed VRAM at M=512 = 15.00 MiB
```

The transient ring is therefore tiny versus 12 GiB VRAM. This exposed a high-priority optimization candidate:

```text
fixed streaming ring
+
persistent compressed hot-weight / hot-expert cache
```

A sensitivity example reserving 8 GiB for compressed cache moves the ideal 70B crossover from roughly:

```text
M 260 -> M 209
```

under the same timing assumptions.

### Multi-GPU interpretation

Current-code-compatible scaling:

```text
one independent TensorWave worker per GPU
one NUMA-local pinned host queue per GPU
one copy/compute/ring state per GPU
```

One-model equal sharding is output only as an optimistic lower bound and is explicitly **not implemented**. Real sharding requires activation placement, synchronization/collectives and topology-aware partitioning.

### Real-hardware gates

Before treating R920 as validated:

```text
GPU physical fit
safe auxiliary power path
thermals
PCIe negotiated width/generation
local NUMA pinned H2D
remote NUMA pinned H2D
simultaneous H2D + GEMM overlap
Phase-3 correctness
measured-vs-predicted M crossover
```

---

## Prior art / novelty position

See:

- `docs/08-PRIOR-ART-AND-NOVELTY.md`
- `docs/09-WHY-LARGE-LLM-WORKS-OR-FAILS.md`

Do **not** claim the following as novel:

```text
large model in CPU RAM
offloaded inference on tiny VRAM
prefetch next weights
quantized H2D
GPU dequantization
tensor/sub-layer offload
hot-weight caching
```

Relevant prior systems include AirLLM, ZeRO-Inference, FlexGen, ATSInfer, PowerInfer/PowerInfer-2 and low-bit GPU dequant kernels.

The currently interesting TensorWave combination is:

```text
Weight Atlas
-> static/precompiled tile schedule
-> sub-tensor compressed tiles
-> fixed compressed VRAM ring
-> scheduled async H2D
-> near-compute Q4 decode
-> starvation-driven optimization
```

No uniqueness/patentability claim is established.

---

## Why large dense LLM decode is the adversarial case

For batch=1 dense autoregressive decode, almost all dense active weights are consumed again for each generated token.

TensorWave Q4-v1 requires:

```text
0.625 bytes / active parameter / streamed pass
```

Thus a dense 70B model with no persistent weight residency would require roughly:

```text
43.75 GB streamed per dense step
```

before considering other overheads.

This makes single-stream decode strongly PCIe-limited unless large portions are resident/cached or another sparsity mechanism applies.

Expected stronger regimes:

```text
prefill
batched serving
MoE / sparse expert activation
```

The Feasibility Map must prove or reject that expectation experimentally.

---

## Shared interface contracts

### Weight Atlas v1

```text
tensorwave.weight-atlas.v1
```

### Execution Plan v1

```text
tensorwave.execution-plan.v1
```

### Runtime Schedule v1

```text
tensorwave.runtime-schedule.v1
slot(i) = i % slots
copy(i) waits compute(i-slots)
compute(i) waits copy(i)
```

### Q4 Plan v1

```text
tensorwave.q4-plan.v1
Q4_SYM_G32_F32S
```

### Feasibility Map v1

```text
tensorwave.feasibility-map.v1
```

### Feasibility Calibration v1

```text
tensorwave.feasibility-calibration.v1
```

### R920 Simulation v1

```text
tensorwave.r920-simulation.v1
```

This last schema is a simulation/report schema only; it does not replace a runtime interface contract.

No shared schema/packing rule may change silently. Update the identifier/ADR/current-state documentation when a contract changes.

---

## Immediate hardware commands

Phase-4 real-checkpoint experiment:

```powershell
.\scripts\run-feasibility-experiments.ps1 `
  -ModelDir "D:\models\some-safetensors-model" `
  -TileN @(128,256) `
  -M @(1,4,16,64,128,256,512,1024) `
  -MaxTiles 32
```

R920 analytical reference simulation:

```powershell
python .\tools\simulate_r920_tensorwave.py `
  --output-dir .\runs\r920-reference
```

R920 cache sensitivity example:

```powershell
python .\tools\simulate_r920_tensorwave.py `
  --output-dir .\runs\r920-cache8 `
  --cache-gib-per-gpu 8
```

The decisive real-hardware outputs remain:

```text
local vs remote NUMA H2D
starvation vs M
hidden transfer vs M
Q4 vs 16-bit at same K/N/M
measured map crossover vs predicted M_cross
```

---

## Contributor rule

If a change affects shared contracts, measurement definitions or map equations:

1. update this file;
2. update/add an ADR;
3. add/adjust tests;
4. never replace a measured failure with an analytical assumption.
