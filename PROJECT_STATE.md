# TensorWave — Current Project State

Last updated: **2026-08-15**

This file is the fast synchronization point for contributors. Read it before starting a new branch.

## Current objective

Demonstrate, with measured evidence, that a GPU with a small fixed VRAM working set can execute operations over a much larger model resident in host RAM by streaming deterministic weight tiles ahead of compute.

The project must optimize **unhidden transfer / GPU starvation**, not merely quote raw PCIe bandwidth.

## Current branch chain

```text
main
  |
  +-- bench/h2d-overlap-proof-v1              # PR #1, Phase 1
        |
        +-- model/real-weight-atlas-proof-v1  # PR #2, Phase 2
              |
              +-- quant/q4-streaming-proof-v1 # PR #4, Phase 3
```

Each PR targets its immediate parent phase so contributors can review one architectural increment at a time.

## Phase 0 — project definition

Status: **captured**

Contains:

- original intent;
- model-memory mental model;
- Weight Atlas concept;
- streaming runtime concept;
- compression/dequantization concept;
- collaboration rules;
- original conversation + verbatim user input.

No experimental claim is established merely because it appears in the original discussion.

## Phase 1 — synthetic transfer/compute overlap

Branch: `bench/h2d-overlap-proof-v1`

PR: `#1 Phase 1: fixed-VRAM H2D/compute overlap proof`

Status: **implementation ready; target-hardware measurement pending**

Implements:

- pinned host weight memory;
- real FP16 cuBLAS GEMM;
- two fixed VRAM weight slots;
- separate non-blocking copy/compute CUDA streams;
- event-enforced slot ownership;
- sequential baseline;
- overlapped pipeline;
- GPU-starvation measurement;
- JSON results.

Strong evidence criterion for a tested shape:

```text
correctness_ok = true
steady_starvation_pct <= 10%
steady_hidden_transfer_pct >= 80%
```

No result has been fabricated or assumed. The experiment must run on target CUDA hardware.

## Phase 2 — real checkpoint bytes

Branch: `model/real-weight-atlas-proof-v1`

PR: `#2 Phase 2: real checkpoint Weight Atlas + streaming proof`

Status: **Python tooling CI passes; CUDA compile/hardware gate separate**

Implemented:

- `tools/safetensors_atlas.py`
  - header-only safetensors scan;
  - exact tensor names/shapes/dtypes/byte offsets;
  - no tensor payload materialization.

- `tools/pack_stream_tiles.py`
  - deterministic homogeneous rank-2 group selection;
  - F16/BF16;
  - fixed K and tile-N;
  - exact raw row slices;
  - per-tile provenance + SHA-256;
  - `weights.pack` + `execution-plan.json`.

- `tools/build_runtime_schedule.py`
  - compiles the ordered execution plan into explicit fixed-slot dependencies;
  - tile `i` maps to `slot = i % slots`;
  - slot reuse is guarded by `compute(i-slots):done`;
  - compute is guarded by `copy(i):done`;
  - no runtime tensor lookup is required for this static plan.

- `src/tensorwave_real_weight_proof.cu`
  - real checkpoint bytes loaded into pinned host RAM;
  - two fixed VRAM weight buffers;
  - F16/BF16 cuBLAS input;
  - FP32 accumulator;
  - every tile contributes to final correctness output;
  - sequential versus overlapped measurement.

- `scripts/run-real-weight-proof.ps1`
  - checkpoint directory -> atlas -> plan -> static runtime schedule -> pack -> CUDA build -> M sweep -> summary.

- unit tests + GitHub Actions
  - exact safetensors offset test;
  - exact packed-byte equality;
  - per-tile SHA-256 verification;
  - static slot ownership/reuse dependency validation;
  - Python CI passing on the implemented tool chain.

## Phase 3 — Q4 host store + compressed H2D + GPU dequant

Branch: `quant/q4-streaming-proof-v1`

PR: `#4 Phase 3: Q4 host store + compressed H2D + GPU dequant`

Status: **implemented; Python tests running/passing; CUDA compile and hardware measurement pending**

Current Q4 proof format:

```text
Q4_SYM_G32_F32S
32 weights/group
4-byte float32 scale
16 packed signed-int4 bytes
20 bytes/group
```

Compared with a 16-bit source:

```text
64 source bytes / 20 Q4 bytes = 3.2x compression
Q4 physical byte ratio = 31.25%
effective density = 5 bits/weight including scale
```

Implemented:

- `tools/quantize_q4_pack.py`
  - vectorized F16/BF16 source conversion;
  - symmetric signed int4 quantization;
  - BF16 decoding without requiring native NumPy bfloat16;
  - per-tile Q4 SHA-256;
  - RMS/max reconstruction error + signal RMS + SNR.

- `src/tensorwave_q4_stream_proof.cu`
  - two fixed compressed-Q4 VRAM slots;
  - one fixed reusable FP16 dequantized weight tile;
  - CUDA Q4 dequant kernel;
  - cuBLAS GEMM with FP32 accumulator;
  - sequential-Q4 versus overlapped-Q4 correctness;
  - separate physical H2D, dequant, GEMM and starvation metrics;
  - source-equivalent feed-rate metric separated from physical PCIe bandwidth.

- `scripts/run-q4-proof.ps1`
  - real checkpoint -> atlas -> source pack -> static schedule -> Q4 pack -> CUDA build -> M sweep -> summary.

- `tests/test_q4_quantization.py`
  - signed-int4 nibble encoding;
  - bounded known-group error;
  - zero-group handling;
  - F16/BF16 source decoding.

Phase 3 deliberately dequantizes into one full FP16 tile before cuBLAS. This keeps the proof measurable and debuggable. A later fused kernel should avoid materializing even that full decompressed tile.

## MiniMax H3 status

MiniMax H3 is officially announced.

However, TensorWave currently treats exact architecture/checkpoint internals as **unverified until read from an official checkpoint/config/report/implementation**.

See:

- `docs/07-H3-RELEASE-GATE.md`

Do not introduce hardcoded H3 tensor names or dimensions from secondary claims.

When official weights are available, first actions are:

```powershell
.\scripts\run-real-weight-proof.ps1 -ModelDir "<official H3 checkpoint directory>"
.\scripts\run-q4-proof.ps1 -ModelDir "<official H3 checkpoint directory>"
```

Archive the exact atlas/plan before writing H3-specific graph code.

## Shared interface contracts — DO NOT silently change

### Weight Atlas v1

Schema identifier:

```text
tensorwave.weight-atlas.v1
```

Important fields:

```text
name
shard
dtype
shape
rank
nbytes
data_start_absolute
tensor_offset_relative
tensor_offset_absolute
size_check
```

### Execution Plan v1

Schema identifier:

```text
tensorwave.execution-plan.v1
```

Per tile:

```text
tile_id
tensor_name
shard
dtype
source_shape
row_start
row_end
k
n
source_offset_absolute
pack_offset
nbytes
sha256
```

### Runtime Schedule v1

Schema identifier:

```text
tensorwave.runtime-schedule.v1
```

Current rule:

```text
slot(i) = i % slots
copy(i) waits compute(i-slots) before slot reuse
compute(i) waits copy(i)
```

This schedule is compiled before the measured loop.

### VRAM ring v1

No per-tile `cudaMalloc/cudaFree` is allowed in the measured loop.

Phase 1/2:

```text
2 full-precision weight slots
```

Phase 3:

```text
2 compressed Q4 slots
1 reusable full-precision dequant tile
```

### Q4 Plan v1

Schema identifier:

```text
tensorwave.q4-plan.v1
```

Current quantizer name:

```text
Q4_SYM_G32_F32S
```

Do not change nibble encoding, group size or scale representation without a new quantizer identifier and ADR update.

## Facts vs hypotheses

### Implemented facts

- safetensors headers provide sufficient metadata to locate raw tensor byte ranges;
- Phase-2 tooling builds an atlas and exact byte pack without numerical deserialization;
- static two-slot dependencies are materialized and unit-tested before runtime;
- Q4 v1 mathematically reduces 16-bit weight storage/H2D bytes to 31.25% including its float32 group scales;
- Q4 encoding/decoding logic has unit tests;
- Python toolchain tests run in GitHub Actions.

### Pending measurement / compile gates

- target machine H2D bandwidth;
- actual copy/compute concurrency on RTX 3050 Ti;
- starvation percentage and speedup;
- BF16 cuBLAS behavior on target driver/toolkit;
- CUDA compile result for all proof targets in the current CI container;
- real checkpoint Phase-2 correctness on target hardware;
- Q4 CUDA dequant kernel correctness on target hardware;
- Phase-2 versus Phase-3 wall time/starvation comparison.

### Later hypotheses

- sub-layer/tensor tiling can keep a 4 GB GPU productive on a model far larger than VRAM;
- compressed H2D plus GPU dequant materially reduces unhidden transfer;
- fused dequant-GEMM can remove the full decompressed weight tile from VRAM;
- H3 graph order can be prefetched deterministically several operations ahead;
- hot/warm/cold residency can reduce bytes transferred per denoising step;
- much of H2D latency can be hidden under model compute.

## Next gates

### Gate A — run Phase 1

Get the starvation-versus-M curve on the target RTX 3050 Ti.

### Gate B — run Phase 2 on any real F16/BF16 safetensors checkpoint

Verify real-checkpoint bytes before H3-specific integration.

### Gate C — run Phase 3 on the exact same source tensor family

Compare 16-bit H2D against Q4 H2D + GPU dequant at identical M/K/N/tile count.

### Gate D — official H3 checkpoint

Generate and archive the exact H3 Weight Atlas.

### Gate E — graph-derived plan

Replace storage-order tiles with actual operation dependencies.

### Gate F — fused/tiled Q4 execution

Consume compressed fragments directly into registers/shared memory/matrix multiply so the complete FP16 weight tile is never materialized.

## Contributor rule

If a change modifies one of the shared contracts above, update this file and add/update an ADR in the same branch.
