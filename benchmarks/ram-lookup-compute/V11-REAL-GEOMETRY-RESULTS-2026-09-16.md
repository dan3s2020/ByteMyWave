# V11 real-geometry Q4_K x Q8_K results — 2026-09-16

Host: Intel Core i5-12500H laptop, 16 GB DDR5-4800 (2x8 GB Samsung), Windows PowerShell, GCC 13.2.0 MinGW-W64.

Tensor: `blk.0.ffn_gate.weight` from the local Qwen Q4_K model.

Correct GGUF/GEMV geometry:

- `ne[0] = 2560` input columns / contiguous row length;
- `ne[1] = 9216` output rows;
- matrix = `9216 x 2560`;
- 10 Q4_K blocks/row;
- 92,160 total Q4_K blocks;
- 13,271,040 bytes / 12.656 MiB Q4_K.

This supersedes the legacy V10/harness run that interpreted the same bytes as `2560 x 9216`.

## Correctness

All SIMD variants matched the scalar Q4_K x Q8_K operator within floating-point accumulation-order noise. Representative native-build errors:

- llama-style AVX2 baseline: max abs `5.960464e-07`, rel-L2 `2.713212e-07`, cosine `1.0`;
- AVX2 repacked 8-row: max abs `3.576279e-07`, rel-L2 `1.401535e-07`, cosine `1.0`;
- exact nibble LUT: max abs `0`, rel-L2 `0`, cosine `1.0`.

Q8_K activation quantization relative to the earlier FP32-activation reference:

- max abs `1.402263e-02`;
- rel-L2 `7.040635e-03`;
- cosine `0.999975214431`.

This is the expected Q8_K activation quantization effect, not extra weight loss.

## AVX2-only build — kernel-only median

| Kernel | ms/GEMV | speed vs llama-style AVX2 |
|---|---:|---:|
| llama-style AVX2 baseline | 0.910 | 1.000x |
| Transit AVX2 raw 1-row | 1.235 | 0.737x |
| AVX2 raw 4-row | 2.122 | 0.429x |
| AVX2 raw 8-row | 2.068 | 0.440x |
| AVX2 repacked 1-row | 1.214 | 0.750x |
| AVX2 repacked 4-row | 1.602 | 0.568x |
| AVX2 repacked 8-row | 1.650 | 0.552x |
| nibble LUT raw | 8.931 | 0.102x |
| nibble LUT repacked | 9.132 | 0.100x |

## Native `-march=native` build — kernel-only median

The native build confirms `__AVXVNNI__ = yes` on the i5-12500H.

| Kernel | ms/GEMV | speed vs llama-style AVX2 |
|---|---:|---:|
| llama-style AVX2 baseline | 0.860 | 1.000x |
| Transit AVX2 raw 1-row | 1.272 | 0.676x |
| AVX2 raw 4-row | 1.640 | 0.524x |
| AVX2 raw 8-row | 1.502 | 0.572x |
| AVX2 repacked 1-row | 1.005 | 0.855x |
| AVX2 repacked 4-row | 1.230 | 0.699x |
| AVX2 repacked 8-row | 1.274 | 0.675x |
| AVX-VNNI raw 1-row | 1.510 | 0.569x |
| AVX-VNNI repacked 1-row | 1.226 | 0.701x |
| AVX-VNNI raw 4-row | 2.245 | 0.383x |
| AVX-VNNI raw 8-row | 1.601 | 0.537x |
| AVX-VNNI repacked 4-row | 1.451 | 0.593x |
| AVX-VNNI repacked 8-row | 1.628 | 0.528x |
| nibble LUT raw | 8.732 | 0.098x |
| nibble LUT repacked | 9.193 | 0.094x |

## Preparation costs

Native build representative one-time/per-activation costs:

- Q8_K quantization: ~`0.004 ms` median once measured in the dynamic benchmark;
- model repack once: `3.721 ms` in this run;
- Q8 nibble LUT build: ~`0.006 ms` median;
- predecoded per-block layout overhead: `+2.778%`.

The LUT build itself is cheap. The failure is the lookup-compute kernel: ~9 ms versus ~0.86 ms for the current AVX2 arithmetic shape.

## Decision

1. The exact nibble-LUT path is rejected for modern x86 CPU inference. It is roughly 10x slower than the llama-style AVX2 baseline despite exact arithmetic.
2. The naive Transit raw/repacked 4-row and 8-row loops are rejected in their current form.
3. The custom AVX-VNNI path is rejected in its current form. Merely replacing multiply/add instructions with VNNI does not beat the current AVX2 dataflow.
4. The active baseline is now the geometry-correct llama-style AVX2 path, ~`0.86–0.91 ms/GEMV` on this laptop run.
5. Previous V15 `x8meta = 0.894 ms` must NOT be described as a win over the current ggml-style baseline. V15 reported `1.403x` only versus its own weaker `packed SIMD control = 1.254 ms`. The next gate is an apples-to-apples same-harness comparison between current llama-style AVX2 and V15 x8meta.
6. The repeated single-layer benchmark is likely at least partly LLC-hot because the 12.656 MiB tensor fits inside the i5-12500H's 18 MiB L3. A realistic model-streaming gate must include cache-evicted/cold measurements or multiple real tensors streamed sequentially.
7. End-to-end model tok/s remains unmeasured. No operator result should be promoted to a Transit inference-speed claim until integrated into current llama.cpp and tested with `llama-bench` / real decode.
