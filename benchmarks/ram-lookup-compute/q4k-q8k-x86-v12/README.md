# Transit V12 — apples-to-apples x8meta gate

V12 exists to resolve one ambiguity left by V11 versus V15.

V11 measured the current llama-style AVX2 Q4_K × Q8_K arithmetic shape at roughly `0.86–0.91 ms/GEMV` on the geometry-correct real Qwen tensor. V15 measured `x8meta = 0.894 ms`, but its reported `1.403x` gain was only versus V15's weaker `packed SIMD control = 1.254 ms`.

Therefore V12 puts the two relevant kernels in the **same executable and same paired timing loop**:

- current llama-style AVX2 baseline;
- V15 x8meta AVX2.

It uses the V11 geometry-correct fixture:

- matrix `9216 rows × 2560 cols`;
- 10 Q4_K blocks/row;
- 12.656 MiB original Q4_K;
- x8meta overhead `+2.778%`.

## Measurement controls

The pair order alternates every iteration (`baseline -> x8meta`, then `x8meta -> baseline`) to reduce turbo/thermal ordering bias.

Two regimes are measured:

- `HOT`: repeated layer without explicit eviction;
- `EVICTED`: a 64 MiB buffer is touched before each timed kernel. This is intended to reduce the unrealistic advantage of repeatedly benchmarking a 12.656 MiB tensor that can fit inside the i5-12500H's 18 MiB LLC.

The runner builds twice:

1. normal `-O3 -march=native`;
2. the same build plus `-fno-unroll-loops`, because the earlier V15 assembly audit found compiler unrolling could hurt the x8 path.

## Run

From this directory, after V11 has created `%TEMP%\q4k_exact_v11`:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_v12_compare.ps1 -Iterations 60 -Warmup 8
```

Optional affinity control:

```powershell
.\run_v12_compare.ps1 -Iterations 60 -Warmup 8 -Cpu 0
```

Do not assume logical CPU 0 is necessarily the best P-core. Use affinity only as a measurement control; if V12 is within a few percent, the next step is a logical-CPU sweep rather than declaring a winner.

## Decision metric

The output prints:

```text
HOT      ... paired_speedup=X
EVICTED  ... paired_speedup=Y
```

- `paired_speedup > 1.00` means x8meta is faster;
- `paired_speedup < 1.00` means llama-style AVX2 is faster.

If x8meta does not beat the baseline under either HOT or EVICTED in both build modes, reject the `+2.778%` x8meta layout on the Alder Lake laptop and move to full llama.cpp plus real Gen8/Ivy Bridge testing rather than spending more time on laptop-only microkernels.
