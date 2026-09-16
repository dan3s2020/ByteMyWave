# Native Q4_K x Q8_K real-laptop results — 2026-09-16

Status: measured on the project laptop after the native harness was delivered.

## Environment

```text
CPU: 12th Gen Intel Core i5-12500H
RAM: 16 GiB, 2 x 8 GiB Samsung DDR5-4800
Compiler: GCC 13.2.0, MinGW-W64 x86_64-ucrt-posix-seh
Target: x86_64-w64-mingw32
Thread pinning: disabled (`Pinned logical CPU = -1`)
Warmup: 5
Timed iterations: 30
Tensor: blk.0.ffn_gate.weight
Shape: 2560 x 9216
Logical weights/GEMV: 23,592,960
Q4_K storage: 13,271,040 bytes = 12.656 MiB
Q8_K activation: 10,512 bytes
```

Two binaries were measured:

```text
AVX2 control build:
  -O3 -std=c++17 -mavx2 -mfma -mssse3 -msse4.1
  __AVX2__ = yes
  __FMA__ = yes
  __AVXVNNI__ = no

Native build:
  -O3 -std=c++17 -march=native -mtune=native
  __AVX2__ = yes
  __FMA__ = yes
  __AVXVNNI__ = yes
```

The `NOMINMAX redefined` message is a harmless compile warning and did not affect execution.

## Preparation footprint/cost

AVX2 build:

```text
Q8_K quantize once       0.023 ms initial / 0.013 ms median dynamic
model repack once        4.089 ms
repacked Q4_K            13.008 MiB = +2.778% per block
Q8 nibble LUT build      0.014 ms initial / 0.018 ms median dynamic
Q8 nibble LUT            0.281 MiB
```

Native build:

```text
Q8_K quantize once       0.019 ms initial / 0.013 ms median dynamic
model repack once        3.985 ms
repacked Q4_K            13.008 MiB = +2.778% per block
Q8 nibble LUT build      0.015 ms initial / 0.011 ms median dynamic
Q8 nibble LUT            0.281 MiB
```

## Correctness

All SIMD paths matched the scalar Q4_K x Q8_K operator to normal floating-point accumulation-order tolerance. Representative relative-L2 values were `~1.8e-7` to `~4.5e-7`, with cosine reported as `1.000000000000`.

The exact nibble-LUT paths matched scalar exactly:

```text
Nibble LUT raw exact       max_abs=0  rel_L2=0  cosine=1
Nibble LUT repacked exact  max_abs=0  rel_L2=0  cosine=1
```

The Q8_K activation quantization effect relative to the earlier FP32-activation Python reference was:

```text
max_abs = 2.437210e-02
rel_L2  = 7.000090e-03
cosine  = 0.999975500129
```

This is the normal change from moving the activation side from FP32 to Q8_K, not additional weight approximation.

## Kernel-only results

### AVX2 control build

| Path | Median ms | Speed vs llama-style AVX2 |
|---|---:|---:|
| scalar reference | 2.538 | 0.320x |
| llama-style AVX2 baseline | **0.812** | **1.000x** |
| Transit AVX2 raw 1-row | 1.056 | 0.769x |
| AVX2 raw 4-row | 1.566 | 0.519x |
| AVX2 raw 8-row | 1.361 | 0.597x |
| AVX2 repacked 1-row | 0.898 | 0.904x |
| AVX2 repacked 4-row | 1.163 | 0.698x |
| AVX2 repacked 8-row | 0.974 | 0.834x |
| Nibble LUT raw | 8.909 | 0.091x |
| Nibble LUT repacked | 8.830 | 0.092x |

### Native build (AVX-VNNI available)

| Path | Median ms | Speed vs llama-style AVX2 |
|---|---:|---:|
| scalar reference | 1.257 | 0.625x |
| llama-style AVX2 baseline | **0.786** | **1.000x** |
| Transit AVX2 raw 1-row | 1.342 | 0.586x |
| AVX2 raw 4-row | 1.579 | 0.498x |
| AVX2 raw 8-row | 1.537 | 0.511x |
| AVX2 repacked 1-row | 1.164 | 0.675x |
| AVX2 repacked 4-row | 1.105 | 0.711x |
| AVX2 repacked 8-row | 1.042 | 0.754x |
| AVX-VNNI raw 1-row | 1.120 | 0.702x |
| AVX-VNNI repacked 1-row | 0.979 | 0.803x |
| AVX-VNNI raw 4-row | 1.514 | 0.519x |
| AVX-VNNI raw 8-row | 1.676 | 0.469x |
| AVX-VNNI repacked 4-row | 1.285 | 0.612x |
| AVX-VNNI repacked 8-row | 1.112 | 0.707x |
| Nibble LUT raw | 8.917 | 0.088x |
| Nibble LUT repacked | 8.719 | 0.090x |

## Activation-to-output results

These include dynamic Q8_K activation quantization and, for the LUT path, LUT construction.

AVX2 build:

```text
Q8 + llama-style AVX2          1.067 ms  1.000x
Q8 + AVX2 repacked 4-row      1.252 ms  0.852x
Q8 + LUT + repacked LUT       8.882 ms  0.120x
```

Native build:

```text
Q8 + llama-style AVX2          1.043 ms  1.000x
Q8 + AVX2 repacked 4-row      0.987 ms  1.056x
Q8 + LUT + repacked LUT       8.760 ms  0.119x
```

The 1.056x total-path result is only a small unpinned single-thread signal; the kernel-only repacked 4-row path itself was slower than the AVX2 baseline. It is not promoted as a winning kernel.

## Decision

This run falsifies the simple native LUT direction for modern x86 CPU execution:

- exact LUT arithmetic is correct but roughly **11x slower** than the native AVX2 baseline;
- exposing AVX-VNNI by itself does not help this custom loop enough; layout/scheduling dominates;
- simple raw/repacked 4-row and 8-row batching in this harness does not beat the current ggml-style packed AVX2 microkernel;
- the model-load-time +2.778% repack remains technically useful, but only if paired with a better compute layout/kernel.

Therefore this harness is retained as a falsification/control benchmark. The active optimization track remains the separate V15 `x8meta` implementation, whose real-Qwen standalone benchmark previously measured 0.894 ms versus 1.254 ms packed SIMD (1.403x) on the native build.

No end-to-end model token/s claim is made from either microbenchmark. The next gate is real llama.cpp integration and `llama-bench` / decode measurement.