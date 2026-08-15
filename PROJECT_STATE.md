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
  +-- bench/h2d-overlap-proof-v1        # PR #1, Phase 1
        |
        +-- model/real-weight-atlas-proof-v1   # PR #2, Phase 2
```

Phase 2 intentionally depends on the Phase-1 fixed-VRAM ring. PR #2 therefore targets the Phase-1 branch rather than duplicating the Phase-1 diff against `main`.

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

Status: **implementation ready; hardware measurement pending**

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

Status: **tooling CI passes; CUDA hardware measurement pending**

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

- `src/tensorwave_real_weight_proof.cu`
  - real checkpoint bytes loaded into pinned host RAM;
  - two fixed VRAM weight buffers;
  - F16/BF16 cuBLAS input;
  - FP32 accumulator;
  - every tile contributes to final correctness output;
  - sequential versus overlapped measurement.

- `scripts/run-real-weight-proof.ps1`
  - checkpoint directory -> atlas -> plan -> pack -> CUDA build -> M sweep.

- unit tests + GitHub Actions
  - exact safetensors offset test;
  - exact packed-byte equality;
  - per-tile SHA-256 verification;
  - current Python CI result: passing.

## MiniMax H3 status

MiniMax H3 is officially announced.

However, TensorWave currently treats exact architecture/checkpoint internals as **unverified until read from an official checkpoint/config/report/implementation**.

See:

- `docs/07-H3-RELEASE-GATE.md`

Do not introduce hardcoded H3 tensor names or dimensions from secondary claims.

When official weights are available, first action is:

```powershell
.\scripts\run-real-weight-proof.ps1 -ModelDir "<official H3 checkpoint directory>"
```

Then archive the generated atlas before writing model-specific graph code.

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

### VRAM ring v1

Current weight-residency contract:

```text
slot 0 -> tile i
slot 1 -> tile i+1

before slot reuse:
copy stream waits for compute-end event of the tile that previously owned the slot

before compute:
compute stream waits for copy-end event of the tile it will consume
```

No per-tile `cudaMalloc/cudaFree` is allowed in the measured loop.

## Facts vs hypotheses

### Implemented facts

- safetensors headers contain sufficient metadata to locate raw tensor byte ranges;
- Phase-2 tooling can build an atlas and exact byte pack without numerical deserialization;
- unit tests verify the offset/packing/hash logic on a controlled safetensors file;
- the Python tooling CI passes.

### Pending measurement

- target machine H2D bandwidth;
- compute time for each tested geometry;
- actual copy/compute concurrency on RTX 3050 Ti;
- starvation percentage;
- speedup versus sequential;
- BF16 cuBLAS behavior on the target driver/toolkit;
- real checkpoint Phase-2 correctness on target hardware.

### Later hypotheses

- sub-layer/tensor tiling can keep a 4 GB GPU productive on a model far larger than VRAM;
- quantized bytes can cross PCIe and be dequantized only at/inside GPU compute;
- H3 graph order can be prefetched deterministically several operations ahead;
- hot/warm/cold residency can reduce bytes transferred per denoising step;
- much of H2D latency can be hidden under model compute.

## Next gates

### Gate A — run Phase 1

Get the starvation-versus-M curve on the target RTX 3050 Ti.

### Gate B — run Phase 2 on any real F16/BF16 safetensors checkpoint

This verifies real-checkpoint bytes before waiting on H3 integration.

### Gate C — official H3 checkpoint

Generate and archive the exact H3 Weight Atlas.

### Gate D — graph-derived plan

Replace storage-order tiles with actual operation dependencies.

### Gate E — quantized streaming

Move compressed bytes across PCIe and fuse/defer dequantization on GPU.

## Contributor rule

If a change modifies one of the shared contracts above, update this file and add/update an ADR in the same branch.
