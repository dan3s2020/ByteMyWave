# 17 — Real Qwen Q4_K exact-LUT continuation (2026-09-16)

## Status

Measured exploratory software benchmark on the same stock Windows laptop used by the 2026-09-16 lookup-compute track. This is a continuation of `docs/16-STOCK-LAPTOP-RAM-LOOKUP-COMPUTE-2026-09-16.md` and `benchmarks/ram-lookup-compute/REAL-QWEN-Q4K-RESULTS-2026-09-16.md`.

This continuation answers two questions that remained open after the earlier codebook experiments:

1. Does smaller position-specific product quantization recover enough quality on a real Qwen Q4_K tensor?
2. Can lookup remove arithmetic from the **original Q4_K representation itself**, with zero additional weight approximation?

The decisive result is the second one: a 16-entry exact nibble LUT produced output identical to the direct Q4_K decoder and was measured at **15.223 ms/GEMV versus 16.344 ms/GEMV**, a **1.074x speedup**, in the managed C# proof kernel. The result is not yet a comparison against llama.cpp's optimized native `Q4_K x Q8_K` kernel.

---

## 1. Test platform now identified

The laptop inventory was captured after the benchmark session:

```text
CPU:                 12th Gen Intel(R) Core(TM) i5-12500H
reported cores:      12
reported logical:    16
reported max clock:  2500 MHz (Win32_Processor field)
RAM:                 16 GiB total
DIMMs:               2 x 8 GiB
manufacturer:        Samsung
part number:         M425R1GB4BB0-CQKOL
reported speed:      4800 MT/s
configured speed:    4800 MT/s
available C++ tool:  C:\Strawberry\c\bin\g++.exe
shell:               Windows PowerShell
```

The memory inventory supersedes the earlier note in the V1-V8 document that exact DIMM topology had not been captured.

No claim is made here about exact cache residency, CPU package power state, thread placement, or DRAM traffic because hardware performance counters were not captured.

---

## 2. Real tensor under test

The experiment used the same real Ollama Qwen Q4_K tensor already recorded in the preceding real-Qwen result file:

```text
Tensor:              blk.0.ffn_gate.weight
Shape:               2,560 x 9,216
Logical weights:     23,592,960
Original Q4_K bytes: 13,271,040
Original Q4_K size:  12.656 MiB
Q4_K block size:     256 weights / 144 bytes
```

For the exact-LUT test, the script copied the raw Q4_K bytes directly from the GGUF tensor and built an independent dequantized Python reference using `gguf.dequantize`.

A deterministic FP32 activation vector of length 9,216 was generated with seed `20260916`.

---

## 3. Position-specific product-quantization sweep

### 3.1 Purpose

The previous real-Qwen experiment showed that one 8-bit code per 64 weights was far too destructive, and additive residual codebooks recovered quality too slowly. The next test reduced vector dimension and trained an independent 256-centroid codebook for each tensor position-group.

The tested groups were:

```text
GROUP 16
GROUP 8
```

The stored representation was:

```text
codes[row, subspace]  : uint8
centers[subspace]      : 256 x GROUP, stored FP16
```

Dynamic LUT construction for either group size requires the same heavy arithmetic count:

```text
(cols / GROUP) * 256 * GROUP
= cols * 256
= 2,359,296 MAC/GEMV
```

The group size changes codebook fidelity, storage, LUT size and the number of runtime lookups.

### 3.2 GROUP 16 measured result

```text
Subspaces:            576
Vectors/codebook:     2,560
Centroids/codebook:   256
Training total:       144.245 s

Weight rel-L2:        0.673526
Weight cosine:        0.739166
Weight SNR:           3.43 dB

Output cosine mean:   0.738135
Output cosine worst:  0.724402
Output rel-L2 mean:   0.674645
Output rel-L2 worst:  0.689466

Codes:                1.406 MiB
FP16 codebooks:       4.500 MiB
Total PQ storage:     5.906 MiB
Compression/Q4_K:     2.14x

LUT-build MACs:       2,359,296
Runtime lookups:      1,474,560
Heavy MAC reduction:  10.00x
Dynamic LUT:          576.00 KiB

PQ-LUT GEMV:          6.645 ms
FP32 NumPy baseline:  2.229 ms/GEMV
Speed vs FP32:        0.34x
```

### 3.3 GROUP 8 measured result

```text
Subspaces:            1,152
Vectors/codebook:     2,560
Centroids/codebook:   256
Training total:       337.935 s

Weight rel-L2:        0.494713
Weight cosine:        0.869056
Weight SNR:           6.11 dB

Output cosine mean:   0.868671
Output cosine worst:  0.863564
Output rel-L2 mean:   0.495419
Output rel-L2 worst:  0.504424

Codes:                2.812 MiB
FP16 codebooks:       4.500 MiB
Total PQ storage:     7.312 MiB
Compression/Q4_K:     1.73x

LUT-build MACs:       2,359,296
Runtime lookups:      2,949,120
Heavy MAC reduction:  10.00x
Dynamic LUT:          1,152.00 KiB

PQ-LUT GEMV:          19.371 ms
FP32 NumPy baseline:  2.229 ms/GEMV
Speed vs FP32:        0.12x
```

### 3.4 PQ conclusion

Smaller vector dimensions materially improve fidelity, but not enough:

```text
G16 output cosine ~= 0.738
G8  output cosine ~= 0.869
```

Both remain too destructive for a quality-preserving representation, and the NumPy implementation is slower than the FP32 baseline. This branch therefore does **not** promote these PQ formats as a Transit model representation.

The experiment is still useful because it falsifies the idea that simply shrinking the position-specific vector to 8 or 16 dimensions is sufficient.

---

## 4. Exact Q4_K LUT experiment — no new weight approximation

### 4.1 Goal

The next experiment abandoned centroid/codebook compression and consumed the **original Q4_K bytes directly**.

The question became:

> Can repeated Q4 nibble multiplication by the current activation be replaced by a small activation-specific lookup table while retaining the original Q4_K scale/min semantics exactly?

Three paths were implemented in the same C# class:

```text
A. DIRECT
   decode original Q4_K nibbles and multiply directly by FP32 activation

B. NIBBLE LUT
   for each activation position precompute 16 values:
       LUT[pos, q] = q * x[pos]
   then use the original Q4 nibble as the lookup index

C. BYTE LUT
   for each packed Q4 byte position precompute 256 low/high contributions
   then index by the original packed byte
```

No centroid, learned codebook, residual approximation or weight reconstruction is used in paths B/C.

### 4.2 Q4_K semantics preserved

The C# kernel decodes:

```text
256 weights / Q4_K super-block
144 bytes / super-block
FP16 d
FP16 dmin
12 bytes packed 6-bit scale/min metadata
128 bytes packed Q4 nibbles
```

The `GetScaleMin` routine implements the Q4_K 6-bit scale/min unpack, and the min correction uses one activation sum for every 32 activation positions.

### 4.3 Independent reference correctness

The raw Q4_K tensor was independently dequantized in Python and multiplied by the same FP32 activation vector.

Measured direct-kernel agreement:

```text
DIRECT vs Python reference
max abs diff:    1.90734863E-06
relative L2:     6.02043245E-07
cosine:          1
```

This small difference is consistent with FP32 accumulation/order effects; the cosine is 1 at the printed precision.

### 4.4 Exact lookup equality

Measured:

```text
NIBBLE LUT vs DIRECT
max abs diff:    0
relative L2:     0
cosine:          1

BYTE LUT vs DIRECT
max abs diff:    0
relative L2:     0
cosine:          1
```

Within this implementation, both lookup paths are **output-identical** to the direct original-Q4_K path.

This is the key difference from the preceding PQ/codebook experiments: there is no additional model representation error.

---

## 5. Exact-LUT memory footprint

For the 9,216-column activation:

```text
Nibble LUT:
9,216 positions * 16 FP32 values
= 147,456 floats
= 0.563 MiB

Byte-pair LUT:
4,608 packed-byte positions * 256 entries * 2 FP32 contributions
= 9.000 MiB
```

The original Q4_K tensor itself is 12.656 MiB.

The large difference in lookup-table footprint is central to interpreting the benchmark.

---

## 6. Exact-LUT benchmark result

The benchmark includes rebuilding activation-dependent sums/LUTs inside each measured GEMV iteration.

```text
DIRECT original Q4_K:  16.344 ms/GEMV
NIBBLE LUT exact:       15.223 ms/GEMV
BYTE LUT exact:         47.505 ms/GEMV

Nibble/direct speedup:  1.074x
Byte/direct speedup:    0.344x
```

So the small exact nibble LUT was measured approximately **7.4% faster** than the direct managed C# Q4_K path, while the 9 MiB byte-pair LUT was approximately **2.91x slower** than direct.

### Interpretation

The result supports two narrow statements:

1. Exact activation-specific lookup can replace part of the direct arithmetic without introducing additional numerical loss.
2. Lookup-table footprint/locality is critical: the 0.563 MiB nibble table helped slightly; the 9 MiB byte-pair table was much worse.

It does **not** establish that lookup is faster than llama.cpp, Ollama, an optimized AVX2 kernel, or a hardware Transit tile.

---

## 7. PowerShell/CodeDom compilation chronology

The interactive proof used Windows PowerShell `Add-Type`.

The first attempted compile used:

```powershell
-CompilerOptions "/unsafe /optimize+"
```

which is not accepted by the Windows PowerShell `Add-Type` available on the test system.

The source was then compiled through `System.CodeDom.Compiler.CompilerParameters`:

```powershell
$cp = New-Object System.CodeDom.Compiler.CompilerParameters
$cp.CompilerOptions = "/unsafe /optimize+"
$cp.GenerateInMemory = $true
$cp.GenerateExecutable = $false
```

A second compile failure reported that `Stopwatch` could not be resolved. The benchmark then added the assembly containing `System.Diagnostics.Stopwatch` to `ReferencedAssemblies` before compiling.

These failures are environment/tooling issues; no benchmark timing was taken until the type compiled successfully.

---

## 8. What is proven by this continuation

Measured/proven in this software experiment:

- the real Qwen tensor geometry and raw Q4_K byte count were consumed directly;
- the direct C# Q4_K path matches an independent Python dequantize+GEMV reference to very small FP32 error;
- exact nibble-LUT output equals direct output exactly in the test;
- exact byte-LUT output equals direct output exactly in the test;
- the exact nibble LUT was 1.074x faster than the direct managed path;
- the 9 MiB byte LUT was much slower, demonstrating a strong footprint/locality penalty in this implementation;
- position-specific PQ at G16 and G8 improved over the earlier 64-dimensional codebook but remained too inaccurate.

Not proven:

- faster-than-llama.cpp Q4_K inference;
- faster-than-native AVX2 `Q4_K x Q8_K`;
- end-to-end Qwen token/s improvement;
- end-to-end model quality preservation under any new codebook format;
- physical DRAM compute/PIM;
- server/Xeon performance;
- a final Transit tile datapath.

---

## 9. Architectural consequence for Transit

The exact-LUT result changes the most interesting follow-up from "compress weights into an aggressive codebook" to "preserve the real quantized format and materialize only the smallest useful activation-dependent function."

The next serious comparison should therefore use the runtime path that matters for llama.cpp-style CPU inference:

```text
original Q4_K weights
        x
Q8_K activations
        |
        +-- native direct SIMD baseline
        +-- exact local/block LUT variant
        +-- batched multi-row variant
        +-- optional predecoded scale/min metadata
```

The goal is not to claim that lookup wins by extrapolating the C# result. The falsifiable gate is a native optimized comparison on identical Q4_K/Q8_K inputs, with identical outputs/tolerance and hardware counters where possible.

For the current laptop, the available compiler discovered in the interactive session is:

```text
C:\Strawberry\c\bin\g++.exe
```

The server port must be treated separately because the laptop result is not a Xeon Gen8 measurement.

---

## 10. Reproducible source files preserved with this document

```text
benchmarks/ram-lookup-compute/qwen_position_pq_lut.py
    position-specific G16/G8 PQ experiment

benchmarks/ram-lookup-compute/prepare_q4k_exact.py
    exports raw Q4_K tensor bytes, deterministic activation, Python reference, metadata

benchmarks/ram-lookup-compute/q4k_exact_lut_v10.cs
    direct Q4_K + exact nibble LUT + exact byte-pair LUT kernels and benchmark

benchmarks/ram-lookup-compute/REAL-QWEN-Q4K-EXACT-LUT-RESULTS-2026-09-16.md
    compact raw measured outputs and environment inventory
```

The source model remains external and is not committed.
