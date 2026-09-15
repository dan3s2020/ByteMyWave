# Native Q4_K x Q8_K benchmark — 2026-09-16

Status: implementation added and compile/correctness smoke-tested; real laptop measurement pending execution on the i5-12500H.

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

## Run on the current laptop

From `benchmarks/ram-lookup-compute`:

```powershell
.\run_transit_q4k_q8k_native.ps1
```

The runner reuses `%TEMP%\q4k_exact_v10` when the previously prepared files are present. Otherwise it calls `prepare_q4k_exact.py` against the configured Ollama model blob.

It builds and runs two executables:

```text
transit_q4k_q8k_avx2.exe
  -O3 -std=c++17 -mavx2 -mfma -mssse3 -msse4.1

transit_q4k_q8k_native.exe
  -O3 -std=c++17 -march=native -mtune=native
```

The second build deliberately exposes CPU-specific instructions such as AVX-VNNI when the compiler/CPU combination enables them.

For a more stable run:

```powershell
.\run_transit_q4k_q8k_native.ps1 -Iterations 30 -Warmup 5
```

Optional thread pinning:

```powershell
.\run_transit_q4k_q8k_native.ps1 -Cpu 0 -Iterations 30 -Warmup 5
```

Because the i5-12500H is hybrid P/E-core hardware, CPU pinning is a measurement control, not an assumed optimization. Core identity should be characterized separately before interpreting pinned-vs-unpinned differences.

## Output logs

```text
%TEMP%\q4k_exact_v10\transit_q4k_q8k_avx2-results.txt
%TEMP%\q4k_exact_v10\transit_q4k_q8k_native-results.txt
```

## Interpretation rule

The standalone `llama-style AVX2` kernel ports the relevant current Q4_K x Q8_K arithmetic shape for a fair native microkernel comparison, but it is not the entire ggml runtime. A winning Transit microkernel still has to be integrated into the real llama.cpp/Ollama path and measured at model level before claiming an inference speedup.

Likewise, the nibble LUT is exact with respect to Q4_K x Q8_K arithmetic but may still lose to SIMD because a table lookup is not automatically cheaper than `maddubs`/VNNI on modern x86.
