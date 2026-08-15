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

No shared schema/packing rule may change silently. Update the identifier/ADR/current-state documentation when a contract changes.

---

## Immediate hardware command

From `research/feasibility-map-v1`:

```powershell
.\scripts\run-feasibility-experiments.ps1 `
  -ModelDir "D:\models\some-safetensors-model" `
  -TileN @(128,256) `
  -M @(1,4,16,64,128,256,512,1024) `
  -MaxTiles 32
```

Quick smoke version:

```powershell
.\scripts\run-feasibility-experiments.ps1 `
  -ModelDir "D:\models\some-safetensors-model" `
  -TileN @(256) `
  -M @(1,16,64,256,512) `
  -MaxTiles 16
```

The decisive outputs are:

```text
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
