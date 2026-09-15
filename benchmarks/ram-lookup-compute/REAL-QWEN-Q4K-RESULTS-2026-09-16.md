# Real Qwen Q4_K codebook/LUT results — 2026-09-16

This file records the first ByteMyWave lookup-compute experiments executed on a real Ollama Qwen tensor rather than synthetic weights.

## Source model

Ollama manifest:

```text
qwen3.5/4b-q4_K_M
```

Model blob:

```text
sha256:81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490
file size: 3.157 GiB
GGUF tensors: 834
```

Tensor inventory measured by the local GGUF inspector:

```text
F32    429 tensors   0.005 GiB
Q4_K   230 tensors   1.710 GiB
F16    148 tensors   0.621 GiB
Q6_K    27 tensors   0.810 GiB
```

Test tensor:

```text
name:       blk.0.ffn_gate.weight
type:       Q4_K
shape:      2560 x 9216
elements:   23,592,960
stored:     13,271,040 bytes = 12.656 MiB
Q4_K blocks: 92,160
```

The expected Q4_K storage check `92,160 × 144 = 13,271,040 bytes` passed exactly.

## Experiment A — one 8-bit code per 64 real weights

The Q4_K tensor was dequantized through `gguf` Python support. Every contiguous 64-weight block was assigned to one of 256 learned 64-dimensional centroids using MiniBatchKMeans.

Geometry:

```text
rows:                    2,560
columns:                 9,216
64-weight blocks/row:      144
total blocks:          368,640
codebook:              256 x 64 FP32 = 64 KiB
stored code per block: 1 byte
```

Measured weight approximation quality:

```text
RMSE:                   0.00848490
MAE:                    0.00674469
max abs error:          0.19450802
normalized RMSE:        0.938454 x weight std
relative L2:            0.938454
weight cosine:          0.34565896
SNR:                    0.55 dB
```

Storage:

```text
original Q4_K:          12.656 MiB
codes:                   0.352 MiB
codes + FP32 codebook:   0.414 MiB
compression vs Q4_K:    30.57x
```

Dynamic-LUT GEMV method:

```text
For each 64-element activation block:
    compute 256 centroid dot products
    -> one 256-entry LUT row

For each matrix 64-weight block:
    read its 8-bit centroid code
    lookup the precomputed dot result
    accumulate
```

This removed the artificial activation codebook used in synthetic V8: activation values in this experiment were arbitrary FP32 values.

Operation accounting:

```text
original MAC/GEMV:      23,592,960
LUT-build MAC/GEMV:      2,359,296
runtime lookups/GEMV:      368,640
heavy-MAC reduction:          10.00x
```

The LUT arithmetic was verified against explicitly reconstructed codebook weights:

```text
LUT vs reconstructed max diff: 0.00000048
```

Therefore the lookup implementation computes the represented codebook matrix correctly. The dominant error is representation error, not LUT arithmetic error.

Random-FP32-activation output quality:

```text
activations tested:      8
mean output cosine:      0.34180174
worst output cosine:     0.30988004
mean output relative L2: 0.939896
worst output relative L2:0.952460
```

NumPy wall-clock measurement on the test laptop:

```text
CPU: Intel64 Family 6 Model 154 Stepping 3
Python: 3.11.15
NumPy: 2.4.6

FP32 dequantized GEMV:   2.095 ms
LUT build:               0.204 ms
LUT lookup + reduce:     1.540 ms
LUT total:               1.744 ms
measured speed ratio:    1.20x vs this FP32 NumPy GEMV
```

Important boundary: this is not a comparison against llama.cpp/Ollama's optimized Q4_K kernel. The baseline is NumPy operating on the dequantized FP32 matrix.

### Experiment A conclusion

The lookup mechanism worked on a real Qwen tensor and was already faster than this FP32 NumPy GEMV path, but `64 weights -> one 8-bit centroid code` destroys too much information. An output cosine around 0.34 is not remotely sufficient for a real-model inference claim.

## Experiment B — additive residual codebooks on 64-weight blocks

To test whether more codes could recover fidelity while preserving lookup arithmetic, the same 64-weight block was approximated as a sum of successively learned residual codewords:

```text
Wblock ~= C1[code1] + C2[code2] + ...
```

Measured residual quality after training 1 through 8 stages:

```text
stage 1: weight rel-L2 0.939094, SNR 0.55 dB
stage 2: weight rel-L2 0.884012, SNR 1.07 dB
stage 3: weight rel-L2 0.832088, SNR 1.60 dB
stage 4: weight rel-L2 0.784637, SNR 2.11 dB
stage 5: weight rel-L2 0.743273, SNR 2.58 dB
stage 6: weight rel-L2 0.701618, SNR 3.08 dB
stage 7: weight rel-L2 0.665746, SNR 3.53 dB
stage 8: weight rel-L2 0.631636, SNR 3.99 dB
```

Quality/performance ladder:

| Codes / 64 weights | Storage MiB | Compression vs Q4_K | Logical ops represented / lookup | Weight rel-L2 | Output cosine | Output rel-L2 | LUT ms/GEMV | Ratio vs FP32 NumPy GEMV |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.414 | 30.57x | 64 | 0.939094 | 0.331476 | 0.943846 | 1.822 | 1.21x |
| 2 | 0.828 | 15.28x | 32 | 0.884012 | 0.459956 | 0.888437 | 3.090 | 0.71x |
| 4 | 1.656 | 7.64x | 16 | 0.784637 | 0.620431 | 0.784491 | 6.362 | 0.35x |
| 8 | 3.312 | 3.82x | 8 | 0.631636 | 0.775416 | 0.632337 | 12.373 | 0.18x |

Baseline for this run:

```text
FP32 direct NumPy GEMV: 2.208 ms
```

### Experiment B conclusion

Adding residual codebooks to the same 64-dimensional block is the wrong tradeoff on this tensor. Fidelity improves too slowly while runtime cost grows approximately with stage count. Even eight codes per 64-weight block leave very large representation error and are much slower than the FP32 NumPy baseline.

The next experiment should therefore reduce the vector dimension rather than keep adding residual codebooks to a 64-weight vector.

## New hypothesis — reduce the codebook block dimension

For one 8-bit code per block, test `GROUP = 32, 16, 8` (and optionally 4).

A useful identity makes this direction attractive. Dynamic LUT construction work is:

```text
(columns / GROUP) * 256 * GROUP
= columns * 256
= 9,216 * 256
= 2,359,296 centroid MACs/GEMV
```

So the heavy LUT-build MAC count is constant with group size.

What changes is the number of code lookups:

```text
GROUP 64:   368,640 lookups/GEMV
GROUP 32:   737,280 lookups/GEMV
GROUP 16: 1,474,560 lookups/GEMV
GROUP  8: 2,949,120 lookups/GEMV
GROUP  4: 5,898,240 lookups/GEMV
```

And the plain one-code storage envelope is approximately:

```text
GROUP 64: 0.414 MiB, 30.57x smaller than Q4_K
GROUP 32: 0.734 MiB, 17.23x smaller
GROUP 16: 1.422 MiB,  8.90x smaller
GROUP  8: 2.820 MiB,  4.49x smaller
GROUP  4: 5.629 MiB,  2.25x smaller
```

A second promising variant is gain/scale-aware vector quantization:

```text
Wblock ~= alpha * centroid[code]
```

where the code captures direction/shape and a small per-block scalar captures magnitude. Runtime becomes one centroid-dot LUT lookup plus one scalar multiply per block. This may dramatically improve quality because the codebook no longer has to spend centroids representing both shape and magnitude.

## Evidence boundary

These experiments establish:

1. the LUT arithmetic path can operate on a real Qwen tensor;
2. dynamic activation-specific LUT construction can reduce heavy arithmetic substantially;
3. the tested Python/NumPy one-code path was 1.20-1.21x faster than a dequantized FP32 NumPy GEMV baseline;
4. 64-dimensional 256-centroid representation quality is far too poor;
5. residual additive codebooks on the same 64-dimensional block do not recover quality efficiently.

They do **not** establish:

- end-to-end Qwen speedup;
- quality-preserving model inference;
- superiority over Ollama/llama.cpp Q4_K kernels;
- physical compute inside passive DRAM;
- a final Transit representation.

The next falsifiable gate is smaller block dimensionality and/or scale-aware codebooks, followed by a native optimized comparison against the real Q4_K kernel.