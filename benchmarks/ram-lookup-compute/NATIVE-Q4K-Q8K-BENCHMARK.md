# Native Q4_K x Q8_K benchmark — 2026-09-16

Status: implemented, compile/correctness smoke-tested, and measured on the project i5-12500H laptop. The simple native LUT/batching variants did **not** beat the ggml-style AVX2 baseline; this harness is now a falsification/control benchmark. Exact measured results are in `NATIVE-Q4K-Q8K-RESULTS-2026-09-16.md`.

## Purpose

This is the next gate after the managed exact-Q4_K nibble-LUT result. It compares Transit ideas against a native SIMD baseline on the arithmetic path that matters for llama.cpp-style Q4_K CPU inference: original Q4_K weights times Q8_K activations.

The benchmark contains:

1. scalar Q4_K x Q8_K reference;
2. standalone llama.cpp-style AVX2 baseline, adapted from `ggml-org/llama.cpp` snapshot `930e2fa5995789efbf249a8bf61325bb626e417b`;
3. readable Transit AVX2 1-row baseline;
4. AVX2 4-row and 8-row batched kernels that reuse Q8 loads;
5. model-load-time Q4_K metadata repack: 144 -> 148 bytes/block (+2.778%) with scales/mins predecoded, preserving original Q4 nibbles and FP16 `d`/`dmin`;
6. repacked AVX2 1/4/8-row variants;
7. AVX-VNNI raw/repacked 1/4/8-row variants when `-march=native` defines `__AVXVNNI__`;
8. exact activation-dependent Q8 nibble LUT, raw and repacked;
9. Q8_K quantization timing;
10. dynamic LUT-build timing;
11. activation-to-output timings that include Q8 quantization and, for the LUT path, LUT construction;
12. correctness metrics against the scalar Q4_K x Q8_K reference and optional comparison against the earlier FP32-activation Python reference.

## Source layout

`transit_q4k_q8k_native.cpp` is the translation unit. The implementation is stored in four adjacent `.inc` fragments only because connector writes were deliberately kept reviewable:

- `transit_q4k_q8k_native_01.inc`
- `transit_q4k_q8k_native_02.inc`
- `transit_q4k_q8k_native_03.inc`
- `transit_q4k_q8k_native_04.inc`

Concatenating the four fragments produces the monolithic development source with SHA-256:

```text
e6dca4b773062202fb91aaad6ede6f3837448983385e66a3fca33f7330c5fde2
```

The split translation unit was compiled and run against a synthetic valid Q4_K fixture after upload-layout assembly. The exact nibble-LUT paths matched the scalar Q4_K x Q8_K reference with `max_abs = 0`, `relative L2 = 0`, `cosine = 1`; AVX2 variants differed only by floating-point accumulation ordering at approximately 1e-6 to 1e-8 relative scale in the smoke fixture.

## Measured laptop result

Platform:

```text
12th Gen Intel Core i5-12500H
16 GiB DDR5-4800
GCC 13.2.0 MinGW-W64
5 warmup / 30 timed iterations
thread unpinned
```

The native build exposed AVX-VNNI successfully.

Headline kernel-only medians:

```text
llama-style AVX2 baseline     0.786 ms   1.000x
AVX2 repacked 1-row           1.164 ms   0.675x
AVX2 repacked 4-row           1.105 ms   0.711x
AVX2 repacked 8-row           1.042 ms   0.754x
AVX-VNNI repacked 1-row       0.979 ms   0.803x
AVX-VNNI repacked 4-row       1.285 ms   0.612x
AVX-VNNI repacked 8-row       1.112 ms   0.707x
Nibble LUT repacked            8.719 ms   0.090x
```

All SIMD paths retained cosine `1.0` versus scalar at only floating-point accumulation-order differences. Both LUT variants were exactly equal to scalar (`max_abs=0`, `rel_L2=0`).

The native activation-to-output comparison showed a small `1.056x` result for `Q8 + AVX2 repacked 4-row` (`0.987 ms` vs `1.043 ms`), but the kernel-only path was slower and the run was unpinned. This is recorded as a weak scheduling/noise-sensitive signal, not a winning kernel.

Full numbers and the AVX2-control build are preserved in `NATIVE-Q4K-Q8K-RESULTS-2026-09-16.md`.

## Decision

The native run changes the direction:

- the managed C# nibble-LUT win does **not** survive against optimized native SIMD;
- exact LUT compute is roughly an order of magnitude slower than the ggml-style AVX2 microkernel;
- AVX-VNNI alone does not rescue the custom loop;
- naive 4-row/8-row reuse and the simple 148-byte repack are not enough;
- layout and metadata-decode cost are the more promising lever.

Therefore no more engineering time should be spent on the simple nibble-LUT path for host x86 inference unless a materially different table organization is proposed.

The active track is the separate `q4k-q8k-x86-v15` implementation, especially `x8meta`, which predecodes scale/min metadata across eight rows. Its recorded native real-Qwen result is `0.894 ms` versus `1.254 ms` packed SIMD (`1.403x`) with +2.78% runtime weight storage.

## Run/reproduce this control benchmark

From `benchmarks/ram-lookup-compute`:

```powershell
.\run_transit_q4k_q8k_native.ps1 -Iterations 30 -Warmup 5
```

Optional thread pinning:

```powershell
.\run_transit_q4k_q8k_native.ps1 -Cpu 0 -Iterations 30 -Warmup 5
```

Because the i5-12500H is hybrid P/E-core hardware, CPU pinning is a measurement control, not an assumed optimization.

Output logs:

```text
%TEMP%\q4k_exact_v10\transit_q4k_q8k_avx2-results.txt
%TEMP%\q4k_exact_v10\transit_q4k_q8k_native-results.txt
```

## Interpretation rule

The standalone `llama-style AVX2` kernel ports the relevant current Q4_K x Q8_K arithmetic shape for a fair native microkernel comparison, but it is not the entire ggml runtime. Likewise, the V15/x8meta microbenchmark is not yet an end-to-end model result.

The decisive next gate is integrating the x8meta runtime layout and kernel into the real llama.cpp CPU repack path and measuring `llama-bench` plus actual decode.