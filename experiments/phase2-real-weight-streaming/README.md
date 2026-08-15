# Phase 2 — Real checkpoint bytes through the fixed-VRAM ring

## Question

Phase 1 uses synthetic FP16 weights to test whether H2D transfer can overlap real cuBLAS GEMM.

Phase 2 removes the synthetic-weight assumption:

> Can exact bytes from a real neural-network checkpoint be indexed in RAM, packed into deterministic tiles, streamed through only two fixed VRAM weight slots, and produce the same accumulated GEMM result as a sequential copy-then-compute baseline while keeping GPU starvation low?

## Important scope

This is deliberately **not yet an H3 end-to-end inference run**.

As of the project status recorded in `docs/07-H3-RELEASE-GATE.md`, we do not hardcode unverified H3 internals. The Phase-2 machinery accepts any local safetensors checkpoint containing suitable F16/BF16 rank-2 weights. When official H3 weights are available, the same command can point at that checkpoint directory.

## Data path

```text
.safetensors shards on disk
        |
        | header-only scan
        v
weight-atlas.json
        |
        | deterministic selection: one dtype + one K
        v
execution-plan.json
        |
        | exact contiguous row slices; no float parsing
        v
weights.pack
        |
        | loaded once into pinned host RAM
        v
+------------------------------+
| pinned RAM: real model bytes |
+------------------------------+
        |
        | cudaMemcpyAsync(N+1)
        v
+-------------+  +-------------+
| VRAM slot A |  | VRAM slot B |
+-------------+  +-------------+
        |              |
        +------GPU-----+
               |
               | cuBLAS GEMM using F16/BF16 checkpoint bytes
               v
         float32 accumulator
```

## Why raw safetensors rows can be used directly

A common linear weight tensor is stored row-major as:

```text
[source rows, K]
```

A full row slice `[N,K]` is contiguous in safetensors. Those bytes are also the memory layout of a column-major cuBLAS matrix `[K,N]` with leading dimension `K`.

Therefore this experiment does not transpose, decode or reserialize the selected weights. It feeds the exact checkpoint bytes to cuBLAS.

This is a layout identity used for the experiment; it is not a claim that every model tensor is semantically a linear projection.

## Weight Atlas

`tools/safetensors_atlas.py` reads only the first 8-byte header length and the JSON header of each shard. It records:

- tensor name;
- shard;
- dtype;
- shape/rank;
- element count;
- exact relative and absolute byte offsets;
- payload bytes;
- consistency check between shape/dtype and byte range.

No tensor payload is loaded into Python.

## Tile pack

`tools/pack_stream_tiles.py` selects a homogeneous rank-2 group:

- dtype: F16 or BF16;
- same second dimension `K`;
- at least one complete `tile-N` row block.

Selection is deterministic. By default it chooses the `(dtype,K)` group that yields the largest number of full tiles for the requested `tile-N`.

Each execution-plan entry records:

```text
tile_id
tensor_name
shard
source_shape
row_start / row_end
K / N
absolute source offset
pack offset
byte count
SHA-256
```

Every tile is copied from the checkpoint as bytes. SHA-256 lets contributors prove exactly which source bytes entered the experiment.

## CUDA experiment

`tensorwave_real_weight_proof` performs two runs over the exact same tile sequence.

### Sequential baseline

```text
COPY tile 0 -> GEMM tile 0
COPY tile 1 -> GEMM tile 1
COPY tile 2 -> GEMM tile 2
...
```

### TensorWave overlap

```text
compute stream:   GEMM 0 | GEMM 1 | GEMM 2 | GEMM 3 | ...
copy stream:             COPY 1 | COPY 2 | COPY 3 | COPY 4 | ...
VRAM slots:        A        B        A        B
```

Before slot A/B is overwritten, the copy stream waits on the CUDA event from the last GEMM that consumed that slot.

The compute stream waits only for the CUDA event saying its required tile has arrived.

## Correctness design

Each real tile contributes to the same float32 accumulator:

```text
Y <- X * W_tile + Y
```

Both sequential and overlapped runs begin from zero and consume identical tiles in identical compute order.

The complete final float32 output is copied back after each run and compared using:

- max absolute error;
- RMS error;
- finite-value check.

This is stronger than checking only the last tile because every streamed tile influences the final accumulator.

## Measurements

Per `M` shape the program reports:

- sequential wall time;
- overlapped wall time;
- H2D GB/s;
- sum of GEMM time;
- startup latency;
- steady GPU starvation time;
- steady starvation percentage;
- estimated hidden-transfer percentage;
- pipeline speedup;
- correctness error;
- pinned checkpoint bytes represented;
- fixed VRAM working-set bytes.

The critical metric remains:

```text
steady_starvation_pct
```

Raw PCIe bandwidth is secondary. If transfer happens while the GPU is busy, the hidden portion should not be counted as end-to-end penalty.

## Strong support criterion for one real shape

For a shape to count as strong evidence:

```text
correctness_ok = true
steady_starvation_pct <= 10%
steady_hidden_transfer_pct >= 80%
```

The curve across `M` matters more than one lucky point.

## Run on Windows / RTX 3050 Ti

Checkout the Phase-2 branch and provide a local safetensors checkpoint directory:

```powershell
git fetch origin
git checkout model/real-weight-atlas-proof-v1
git pull

.\scripts\run-real-weight-proof.ps1 -ModelDir "D:\models\some-model"
```

Optional controls:

```powershell
.\scripts\run-real-weight-proof.ps1 `
  -ModelDir "D:\models\some-model" `
  -TileN 256 `
  -MaxTiles 64 `
  -DType auto `
  -M 64,128,256,512,1024
```

To force a known family:

```powershell
.\scripts\run-real-weight-proof.ps1 `
  -ModelDir "D:\models\some-model" `
  -DType BF16 `
  -K 4096 `
  -TileN 256
```

## Generated run record

Each run creates:

```text
runs/phase2-real-YYYYMMDD-HHMMSS/
  run-config.json
  nvidia-smi.txt
  prepared/
    weight-atlas.json
    execution-plan.json
    weights.pack
  m-64.json
  m-64.log.txt
  m-128.json
  ...
```

`runs/` remains local/ignored because the weight pack contains model weights and may be large or license-restricted.

## Next gate

If Phase 2 succeeds, Phase 3 should replace storage-order selection with a **graph-derived execution plan**:

```text
actual operation N
  -> exact tensor/tile dependencies
  -> prefetch N+1 / N+2
  -> fixed VRAM slots
  -> GPU kernel
```

Only after that should TensorWave claim that the scheduler reflects a model's real inference order.
