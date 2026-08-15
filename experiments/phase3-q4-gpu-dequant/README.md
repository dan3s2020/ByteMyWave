# Phase 3 — Q4 in RAM, Q4 over PCIe, GPU-side dequantization

## Question

Phase 2 streams exact F16/BF16 model bytes. Phase 3 attacks the amount of data that must cross PCIe:

> If the host model store remains quantized, can TensorWave transfer only Q4 bytes, dequantize the active tile on GPU, run the GEMM, and still keep GPU starvation low with a tiny fixed VRAM working set?

This is the first experiment that directly implements the original TensorWave compression idea.

## Representation

Current proof format:

```text
Q4_SYM_G32_F32S
```

Per 32 source weights:

```text
4 bytes  float32 scale
16 bytes packed signed int4 values
----------------------------------
20 bytes total
```

F16/BF16 source cost:

```text
32 * 2 = 64 bytes
```

Therefore:

```text
Q4 byte ratio = 20 / 64 = 31.25%
compression   = 64 / 20 = 3.2x
effective     = 5 bits / weight including scale
```

This deliberately favors a simple/aligned proof format over maximum theoretical Q4 density. Later formats can reduce scale overhead.

## Quantization

Each group uses symmetric quantization:

```text
scale = max(abs(x)) / 7
q     = round(x / scale)
q     = clamp(q, -7, +7)
```

Zero groups use scale `1.0` and q=0.

Signed q values are stored as two's-complement 4-bit nibbles.

The offline quantizer records:

- source and Q4 byte counts;
- compression ratio;
- weight RMS error;
- weight max absolute error;
- weight signal RMS;
- weight SNR;
- SHA-256 for every compressed tile.

## Memory path

```text
large model/checkpoint
       |
       | offline once
       v
Q4 pack in host RAM
       |
       | only compressed bytes cross PCIe
       v
+----------------+  +----------------+
| Q4 VRAM slot A|  | Q4 VRAM slot B|
+----------------+  +----------------+
          |
          | custom CUDA dequant kernel
          v
+---------------------------+
| one reusable FP16 W tile  |
+---------------------------+
          |
          | cuBLAS GEMM
          v
     FP32 accumulator
```

The full decompressed model is never resident in VRAM.

Only the current tile is expanded to FP16.

## Why one decompressed tile is enough

The compute stream is serial:

```text
dequant tile N -> GEMM tile N -> dequant tile N+1 -> GEMM tile N+1
```

Therefore a single fixed FP16 dequant buffer can be reused after each GEMM.

Meanwhile the copy stream can fill the *other compressed slot* with tile N+1.

This gives a fixed weight-memory footprint:

```text
2 * compressed Q4 tile
+ 1 * decompressed FP16 tile
```

rather than:

```text
2 * decompressed FP16 tile
```

or the full model.

## Scheduling

The same static schedule contract is retained:

```text
tile i -> slot (i % 2)

copy(i) waits for compute(i-2) before reusing its compressed slot
compute(i) waits for copy(i)
```

There is no tensor search or dependency discovery in the measured loop.

`tools/build_runtime_schedule.py` materializes this schedule before execution.

## Metrics added in Phase 3

Besides Phase-1/2 metrics:

- `dequant_ms` — total GPU dequant kernel time;
- `gemm_ms` — total cuBLAS GEMM time;
- `compressed_h2d_gbps` — actual physical rate of compressed Q4 bytes;
- `source_equivalent_h2d_gbps` — how many original 16-bit bytes those Q4 bytes represent per second.

`source_equivalent_h2d_gbps` is **not** a claim about physical PCIe bandwidth. It is a useful feed-rate metric for comparing Phase 2 to Phase 3.

## Correctness

The sequential Q4 path and overlapped Q4 path both:

1. copy the same compressed tile;
2. run the same GPU Q4 dequantization kernel;
3. run the same cuBLAS GEMM;
4. accumulate every tile into the same FP32 output geometry.

Their complete final FP32 outputs are compared.

This validates scheduling/stream correctness independently from quantization quality.

Quantization quality relative to original F16/BF16 weights is reported separately by `q4-plan.json`.

## Run

```powershell
git fetch origin
git checkout quant/q4-streaming-proof-v1
git pull

python -m pip install numpy

.\scripts\run-q4-proof.ps1 -ModelDir "D:\models\some-safetensors-model"
```

The command executes:

```text
safetensors headers
 -> Weight Atlas
 -> exact 16-bit source pack
 -> static two-slot schedule
 -> offline Q4 pack
 -> CUDA build
 -> Q4 H2D + GPU dequant + GEMM sweep
 -> SUMMARY.md + summary.csv
```

## What we compare against Phase 2

For identical source checkpoint family, K, N, tile count and M:

```text
Phase 2:
16-bit H2D -> GEMM

Phase 3:
Q4 H2D -> GPU dequant -> GEMM
```

Important deltas:

```text
H2D bytes
starvation %
wall time
dequant overhead
GEMM time
fixed VRAM bytes
speedup vs own sequential baseline
```

The decisive result is not simply `Q4 transfers faster`.

The desired result is:

> The reduced copy time is large enough that GPU-side dequantization plus GEMM can continuously cover the next compressed transfer, substantially reducing unhidden H2D time.

## Current limits

- Q4 converter currently requires NumPy offline.
- Q4 v1 uses one float32 scale per 32 weights, so effective density is 5 bits/weight rather than 4.
- dequantization is a separate CUDA kernel followed by cuBLAS; it is not yet fused into GEMM.
- only the selected rank-2 F16/BF16 tile family is quantized.
- this still does not represent a complete model graph.

## Phase 4 direction if this succeeds

The next optimization target is **fused/tiled dequant-GEMM**:

```text
Q4 bytes
 -> registers/shared-memory fragments
 -> matrix multiply
 -> discard fragment
```

That would remove the full FP16 dequantized weight tile from VRAM entirely and make VRAM primarily activations + compressed ring + GEMM workspace.
