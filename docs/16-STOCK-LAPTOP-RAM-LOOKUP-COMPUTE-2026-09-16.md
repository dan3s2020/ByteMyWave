# 16 — Stock-laptop RAM lookup-compute experiments (2026-09-16)

## Status

**Measured exploratory software benchmark on one ordinary Windows laptop.**

This document records the complete V1→V8 experiment sequence performed interactively on 2026-09-16, including the exact measured numbers, the method used at each stage, the observed crossover, and the limitations that prevent these results from being misread as physical DRAM/PIM or end-to-end LLM inference results.

This is a **separate ByteMyWave research track**. It does not replace the current Transit active memory-compute tile architecture in `docs/14-CURRENT-SOLUTION.md`.

---

## 1. What was tested

The experiment asked a narrow question:

> Can arithmetic that would normally be recomputed by the CPU be materialized as a memory function, so inference-like inner-loop work becomes address construction + memory lookup + accumulation, and can that path ever outrun direct scalar CPU arithmetic on a stock laptop?

The basic transformation is:

```text
ordinary path
-------------
result += f(weight, activation)

lookup path
-----------
address = encode(weight, activation)
result += LUT[address]
```

The number of logical arithmetic operations represented by one lookup was then increased experimentally:

```text
V1   1 MAC / lookup
V2   4 MAC / lookup
V3   4 MAC / lookup, universal LUT
V4   4 MAC / lookup, codebook + tiny LUT
V5   4 MAC / lookup, fair same-codebook CPU control + cache/DRAM-pressure control
V6  16 MAC / lookup
V7  32 MAC / lookup
V8  64 logical Q4×Q1 operations / lookup
```

The important conceptual boundary is that **the CPU still performs address formation, LOADs and accumulation**. The DRAM cells are not being commanded to perform an Ambit/ComputeDRAM-style electrical AND/MAC inside the DIMM. This is software **compute-by-lookup / materialized-function** research.

---

## 2. Hardware/software environment actually known

The run was performed interactively from:

```text
Windows PowerShell x64
stock laptop
managed C# compiled in-memory with Add-Type
```

The exact laptop CPU model, DIMM topology, RAM speed, cache sizes, Windows build and power state were **not captured in the console transcript**, so they are intentionally not invented here.

That missing inventory is a required improvement for the next reproduction run.

---

## 3. Evidence discipline

These numbers are:

- measured wall-clock results from the executed benchmark code;
- synthetic inference-like arithmetic, not a trained language model;
- `tok/s` labels are **synthetic benchmark iterations called tokens**, not natural-language LLM decode tokens;
- exact arithmetic checks were used between reference and lookup paths (`PASS`);
- later versions use increasingly restrictive codebooks to make many logical operations addressable by a finite LUT.

These numbers are **not**:

- measured Kimi K3 generation speed;
- evidence that passive DDR3/DDR4 DIMMs independently execute neural arithmetic;
- an AVX2/AVX-512/AMX-optimized CPU comparison;
- a proof that every access in the 256 MiB pressure tests missed LLC and reached physical DRAM;
- a generic arbitrary-weight 64-MAC kernel result.

---

# 4. Complete measured result summary

## V1 — one Q4×INT8 product per lookup

Configuration:

```text
Hidden size        256
Layers             4
Vocabulary         256
Parameters         393,216
Q4 weight RAM      0.19 MiB
RAM product LUT    64.00 MiB
LUT replicas       4,096
Prompt tokens      58
Generated tokens   128
Model/LUT build    0.020 s
```

Method:

```text
CPU reference:
    sum += weight * activation

RAM lookup path:
    address = replica | Q4_weight | INT8_activation
    sum += ProductLut[address]
```

Measured:

| Metric | CPU direct | RAM lookup |
|---|---:|---:|
| Prefill | 14.23 ms | 219.46 ms |
| Generation | 0.0396 s | 0.5590 s |
| Speed | **3230.24 tok/s** | **228.96 tok/s** |
| CPU/RAM ratio | | **14.11× CPU faster** |
| Correctness | | **PASS** |

Finding: replacing one cheap integer multiply with one large-table lookup is a losing trade.

---

## V2a — four MACs per lookup, per-weight-group compiled tables

Configuration:

```text
Hidden size              256
Layers                   4
Vocabulary               256
Parameters               393,216
Q4 weight memory         0.19 MiB
Compiled RAM tables      40.00 MiB
MACs/generated token     327,680
RAM lookups/token        81,920
Generated tokens         256
```

Method:

Four fixed Q4 weights are compiled against every 4-element Q2 activation pattern:

```text
TABLE[activation_pattern]
    = w0*x0 + w1*x1 + w2*x2 + w3*x3
```

Thus one inference lookup replaces four products plus the within-group additions.

Measured:

| Metric | CPU direct | RAM 4-MAC lookup |
|---|---:|---:|
| Prefill | 15.38 ms | 38.12 ms |
| Generation | 0.0806 s | 0.2149 s |
| Speed | **3177.98 tok/s** | **1191.40 tok/s** |
| Effective rate | **1.041 G MAC/s** | **0.390 G MAC/s eq.** |
| Lookup rate | — | **97.60 M lookup/s** |
| CPU/RAM ratio | | **2.67× CPU faster** |
| Correctness | | **PASS** |

V1→V2 lookup path improvement:

```text
228.96 tok/s -> 1191.40 tok/s
~5.20×
```

---

## V2b — same V2 method, larger model

Configuration:

```text
Hidden size              1,024
Layers                   4
Vocabulary               256
Parameters               4,718,592
Q4 weight memory         2.25 MiB
Compiled RAM tables      544.00 MiB
MACs/generated token     4,456,448
RAM lookups/token        1,114,112
Prompt tokens            58
Generated tokens         64
Compile/build time       3.015 s
```

Measured:

| Metric | CPU direct | RAM V2 |
|---|---:|---:|
| Prefill | 223.19 ms | 923.04 ms |
| Generation | 0.2623 s | 1.0789 s |
| Speed | **243.99 tok/s** | **59.32 tok/s** |
| Effective rate | **1.087 G MAC/s** | **0.264 G MAC/s eq.** |
| Lookup rate | — | **66.09 M lookup/s** |
| CPU/RAM ratio | | **4.11× CPU faster** |
| Correctness | | **PASS** |

Important correction: the original V2 console footer accidentally printed the V2a hard-coded reduction numbers (`327,680 -> 81,920`). For this V2b run the correct values are:

```text
4,456,448 MAC/token
        ->
1,114,112 lookup/token
```

Finding: the per-weight-group table construction scales badly; 2.25 MiB of Q4 weights expanded into 544 MiB of compute tables.

---

## V3 — universal 16 MiB LUT

Goal: eliminate V2's model-dependent table explosion.

Address:

```text
16 bits = four Q4 weights
 8 bits = four Q2 activations
-----------------------------
24-bit universal lookup address
```

Configuration:

```text
Hidden size              1,024
Layers                   4
Vocabulary               256
Parameters               4,718,592
Q4 weight memory         2.25 MiB
Universal RAM LUT        16.00 MiB
Total weights + LUT      18.25 MiB
MACs/token               4,456,448
RAM lookups/token        1,114,112
Generated tokens         64
Universal LUT build      0.020 s
Model creation           38.98 ms
```

Measured:

| Metric | CPU direct | RAM V3 |
|---|---:|---:|
| Prefill | 144.36 ms | 873.96 ms |
| Generation | 0.1694 s | 1.1861 s |
| Speed | **377.87 tok/s** | **53.96 tok/s** |
| Effective rate | **1.684 G MAC/s** | **0.240 G MAC/s eq.** |
| Lookup rate | — | **60.11 M lookup/s** |
| CPU/RAM ratio | | **7.00× CPU faster** |
| Correctness | | **PASS** |

Finding: memory footprint became sane, but a nearly random 16 MiB lookup structure was much slower than the scalar CPU path.

---

## V4 — 64 KiB codebook LUT, four MACs/lookup

The representation was changed so a byte selects one of 256 four-weight Q4 patterns, and one byte encodes four Q2 activations.

```text
weightCode     = 8 bits -> selects 4 Q4 weights
activationCode = 8 bits -> represents 4 Q2 values

address = (weightCode << 8) | activationCode

LUT[address]
    = w0*x0 + w1*x1 + w2*x2 + w3*x3
```

The compute LUT is only:

```text
256 × 256 × 1 byte = 64 KiB
```

Configuration:

```text
Hidden size              1,024
Layers                   4
Vocabulary               256
Logical Q4 parameters    4,718,592
CPU packed-Q4 memory     2.25 MiB
RAM coded-weight memory  1.19 MiB
Compute LUT              64.00 KiB
MACs/token               4,456,448
RAM lookups/token        1,114,112
Generated tokens         128
Model build              12.64 ms
```

Measured:

| Metric | CPU packed-Q4 | RAM V4 |
|---|---:|---:|
| Prefill | 142.58 ms | 53.42 ms |
| Generation | 0.3359 s | 0.1302 s |
| Speed | **381.12 tok/s** | **982.84 tok/s** |
| Effective rate | **1.698 G MAC/s** | **4.380 G MAC/s eq.** |
| Lookup rate | — | **1094.99 M lookup/s** |
| Relative speed | 1.00× | **2.58× faster** |
| Correctness | | **PASS** |

This was the first lookup-compute win.

However, V4 was not yet a fair architecture comparison because:

1. the 64 KiB LUT is plausibly cache-resident;
2. the lookup path also uses a more compressed 1-byte code for four weights;
3. the CPU reference used ordinary packed Q4 rather than exactly the same codebook representation.

Those issues motivated V5.

---

# 5. Fairer controls and the large-working-set crossover

## V5 — same codebook representation for CPU and LUT + 256 MiB pressure path

Configuration:

```text
Hidden size              4,096
Layers                   4
Vocabulary               256
Logical MACs/token       68,157,440
Lookup operations/token  17,039,360
Weight-code memory       16.25 MiB
Small LUT                64.00 KiB
Large LUT                256.00 MiB
Large LUT replicas       4,096
Generated test tokens    4
Warmup verification      PASS
```

All paths use the same one-byte-per-four-weights codebook representation.

The 256 MiB path contains replicated copies of the same correct 64 KiB table. A hash/replica selector deliberately spreads lookups across the large address range to defeat small-cache locality. It is a **cache/large-memory pressure proxy**, not proof that every lookup physically reached DRAM.

Measured:

| Path | Speed | Lookup/MAC rate | Relative to direct CPU |
|---|---:|---:|---:|
| A. Direct CPU, same codebook | **23.07 tok/s** | **1.572 G MAC/s** | 1.00× |
| B. 64 KiB LUT | **67.37 tok/s** | **1147.89 M lookup/s**, 4.592 G MAC/s eq. | **2.92×** |
| C. 256 MiB large LUT | **4.32 tok/s** | **73.55 M lookup/s**, 0.294 G MAC/s eq. | **0.19×** |

Arithmetic equality: **PASS**.

The large-LUT result gives a useful empirical crossover estimate for this code path:

```text
1.572 G CPU MAC/s / 73.55 M large-memory lookup/s
≈ 21.4 MAC per lookup
```

At only 4 MAC/lookup, the large lookup path cannot win.

---

## V6 — 16 MACs per lookup

Representation:

```text
Q4 weight codebook
Q1 activations {-1,+1}
16 weights per group
16 activation signs per pattern
1 LUT result = sum of 16 Q4×Q1 products
```

Configuration:

```text
Hidden size              4,096
Layers                   4
Vocabulary               256
Activation format        Q1 (-1 / +1)
Logical MACs/token       68,157,440
Lookups/token            4,259,840
MACs per lookup          16
Weight-code memory       4.06 MiB
Base LUT                 32.00 MiB
Large LUT                256.00 MiB
Large LUT replicas       8
Base LUT build           0.025 s
Large LUT build          0.078 s
Warmup verification      PASS
```

Measured:

| Path | Speed | Rate | Relative to direct CPU |
|---|---:|---:|---:|
| A. Direct CPU | **29.35 tok/s** | **2.000 G MAC/s** | 1.00× |
| B. 32 MiB LUT | **36.07 tok/s** | 153.65 M lookup/s; **2.458 G MAC/s eq.** | **1.23×** |
| C. 256 MiB large LUT | **17.27 tok/s** | 73.56 M lookup/s; **1.177 G MAC/s eq.** | **0.59×** |

Arithmetic equality: **PASS**.

At 16 MAC/large lookup the pressure path improved substantially but remained below CPU.

---

## V7 — 32 MACs per lookup: first large-working-set crossover

Representation:

```text
1 weight code      -> 32 Q4 weights
1 activation code  -> 32 Q1 values
1 LUT lookup       -> sum of 32 Q4×Q1 products
```

Configuration:

```text
Hidden size               4,096
Layers                    4
Vocabulary                256
Weight representation     1 code / 32 Q4 weights
Activation representation 1 code / 32 Q1 values
Logical MACs/token        68,157,440
Lookups/token             2,129,920
MACs per lookup           32
Weight-code memory        2.03 MiB
Base LUT                  128.00 KiB
Large LUT                 256.00 MiB
Large LUT replicas        2,048
Base LUT build            1.94 ms
Large LUT build           59.73 ms
Warmup verification       PASS
```

Measured:

| Path | Speed | Rate | Relative to direct CPU |
|---|---:|---:|---:|
| A. Direct CPU | **29.49 tok/s** | **2.010 G MAC/s** | 1.00× |
| B. 128 KiB LUT | **506.54 tok/s** | 1078.89 M lookup/s; **34.525 G MAC/s eq.** | **17.17×** |
| C. 256 MiB large LUT | **32.81 tok/s** | 69.89 M lookup/s; **2.236 G MAC/s eq.** | **1.11×** |

Arithmetic equality: **PASS**.

This is the first measured crossover in the deliberately dispersed 256 MiB working-set path:

```text
32.81 / 29.49 ≈ 1.11×
```

A second threshold estimate from this run is:

```text
2.010 G direct CPU MAC/s / 69.89 M lookup/s
≈ 28.76 logical MACs per large-memory lookup
```

V7 uses 32 MAC-equivalents/lookup and crosses that measured threshold.

---

## V8 — 64 logical operations per lookup + branchless CPU control

V8 tested whether the lookup path could also beat a scalar CPU path that does not use multiplication for Q1 signs.

Representation:

```text
1 weight code      -> one of 256 predefined 64-element Q4 vectors
1 activation code  -> one of 256 predefined 64-element Q1 vectors
1 LUT entry        -> exact dot product of those two coded vectors
```

Configuration:

```text
Hidden size               4,096
Layers                    4
Vocabulary                256
Weight block              64 Q4 values/code
Activation block          64 Q1 values/code
Logical ops/token         68,157,440
Lookups/token             1,064,960
Ops per lookup            64
Weight-code memory        1.02 MiB
Base LUT                  128.00 KiB
Large LUT                 256.00 MiB
Large LUT replicas        2,048
Tokens                    8
Base LUT build            3.19 ms
Large LUT build           49.93 ms
Warmup verification       PASS
```

CPU branchless control for Q1 signs uses sign-select arithmetic instead of multiplication.

Measured:

| Path | Speed | Rate | Relative result |
|---|---:|---:|---:|
| A. CPU multiply | **13.03 tok/s** | **0.888 Gop/s** | baseline |
| B. CPU branchless, no multiply | **12.95 tok/s** | **0.883 Gop/s** | baseline control |
| C. 128 KiB LUT | **1071.19 tok/s** | 1140.78 M lookup/s; **73.010 Gop/s eq.** | **82.69× branchless** |
| D. 256 MiB large LUT | **58.55 tok/s** | 62.36 M lookup/s; **3.991 Gop/s eq.** | **4.52× branchless** |

Arithmetic equality: **PASS**.

Reported ratios:

```text
Large LUT / multiply    4.49×
Large LUT / branchless  4.52×
Small LUT / branchless 82.69×
```

---

# 6. The crucial V8 caveat

V8 is a real measured result for the **coded-vector problem it defines**, but it must not be generalized to arbitrary Q4×Q1 vectors.

Why:

```text
64 arbitrary Q4 weights would contain 256 raw bits.
V8 represents the entire 64-weight vector with one 8-bit code.
```

That byte does not losslessly encode every possible 64-element Q4 vector. It selects one of only 256 predefined vectors from a codebook.

Likewise:

```text
64 arbitrary Q1 activations contain 64 raw bits.
V8 represents the activation vector with one 8-bit code selecting 1 of 256 patterns.
```

Therefore the domain is deliberately restricted:

```text
256 allowed weight vectors × 256 allowed activation vectors
= 65,536 possible block pairings
```

A 128 KiB table can materialize all exact block dot products for that restricted domain.

This is precisely why V8 can turn 64 logical products into one lookup.

Consequences:

1. The V8 speedup is valid for this coded representation.
2. It is **not** evidence that one byte can losslessly represent 64 arbitrary Q4 weights.
3. A real model would require training/quantization/product-quantization/vector-quantization machinery to map real weight and activation blocks into such codebooks, and model-quality loss must be measured.
4. For exactly the V8 representation, the best CPU implementation is itself allowed to use the same LUT. Comparing the LUT against a CPU that expands all 64 values is useful to measure materialization benefit, but it is not a generic CPU-vs-RAM hardware verdict.

This caveat is mandatory whenever V8 is cited.

---

# 7. What the experiment genuinely established

The measured sequence supports these narrow claims:

### A. One operation per lookup is too fine-grained

V1:

```text
1 MAC/lookup
228.96 tok/s lookup
3230.24 tok/s direct CPU
```

### B. More useful work per lookup changes the economics

The trend under deliberate large-working-set pressure was:

```text
V5   4 MAC/lookup   large path = 0.19× direct CPU
V6  16 MAC/lookup   large path = 0.59× direct CPU
V7  32 MAC/lookup   large path = 1.11× direct CPU
V8  64 coded ops    large path = 4.52× scalar branchless CPU
```

The V5/V7 measured lookup throughput was roughly 70–74 million dispersed lookups/s on this laptop, while the direct scalar CPU path was around 1.6–2.0 billion logical MAC/s in those versions. This placed the empirical crossover near a few tens of operations represented by each lookup.

### C. Tiny cache-resident tables can be extremely fast

Examples:

```text
V5 64 KiB LUT:    2.92× direct CPU
V7 128 KiB LUT:  17.17× direct CPU
V8 128 KiB LUT:  82.69× scalar branchless control
```

These are better interpreted as **table-driven/coded inference optimization results** than as evidence about raw DRAM compute.

### D. Large random lookup locality is expensive

V5 showed:

```text
64 KiB LUT   67.37 tok/s
256 MiB LUT   4.32 tok/s
small/large  15.61×
```

So locality is decisive unless one lookup represents enough useful work.

---

# 8. What was NOT established

The experiment did not establish any of the following:

- that a stock laptop DIMM internally executed AND, multiply, add or dot-product operations;
- that CPU ALUs were absent from the execution path;
- that a 256 MiB array guarantees a DRAM hit on every access;
- that the Windows/JIT/C# scalar CPU implementation is a competitive optimized inference kernel;
- that `58.55 synthetic tok/s` maps to any real model's token/s;
- that the V8 codebook preserves LLM accuracy;
- that arbitrary 64-element Q4/Q1 blocks can be represented by one byte each;
- that this replaces Transit tile-local FPGA/ASIC compute.

---

# 9. Why this still matters to Transit

The experiment suggests an additional architectural lever that can be investigated alongside the existing bitplane and native-MXFP tile paths:

> **Increase useful arithmetic represented by each memory transaction by changing the model representation, not merely by accelerating a scalar multiply.**

Potential Transit implications to test, not assume:

1. **Vector/codebook weights** — store an index to a learned low-bit block prototype instead of every raw weight nibble where model quality permits.
2. **Block activation coding** — map activation blocks to a small basis/codebook where acceptable.
3. **Tile-local result tables** — compute logic or SRAM/BRAM near DDR could materialize frequently used block dot products.
4. **Hybrid codebook + bitplane path** — common patterns use table/materialized results; uncommon/residual vectors use the exact arithmetic engine.
5. **Residual correction** — codebook approximation plus sparse/low-rank residuals can preserve more model quality without making the LUT exponential.

This is compatible with the current Transit principle that large weights stay local and computation happens in active logic near memory. It is not a return to the rejected claim that passive DDR3 DIMMs magically execute inference.

---

# 10. Required next validation gates

The next experiment should be stricter than V8.

## Gate 1 — capture the actual laptop hardware

Record at minimum:

```text
CPU model
core/thread count
ISA (SSE/AVX/AVX2/AVX-512)
L1/L2/L3 sizes
RAM capacity/type/data rate/channels
Windows version
.NET runtime/JIT version
power plan
```

## Gate 2 — optimized native CPU baseline

Reimplement the direct and codebook paths in native C/C++ with at least:

```text
-O3 /O2
AVX2 if supported
AVX-512 if supported
vectorized sign/dot kernels
multiple independent accumulators
prefetch where justified
```

A C# scalar loop is useful for discovery but not a final CPU baseline.

## Gate 3 — hardware counters

Measure whether the large-LUT path is actually reaching DRAM:

```text
LLC references/misses
memory-controller read bytes
DRAM bandwidth
cycles stalled on memory
TLB misses
IPC
```

The 256 MiB replicated LUT is a pressure proxy until these counters are captured.

## Gate 4 — arbitrary-vector baseline

Compare against real packed arbitrary Q4 blocks, not only codebook-selected vectors.

## Gate 5 — learned codebook on a real model tensor

For a real model layer:

```text
train/fit codebook
encode real weight blocks
encode or quantize real activation blocks
run LUT/materialized dot products
apply residual/correction if needed
compare output error to reference
measure task/perplexity degradation
```

Only after this can the V7/V8 representation be discussed as an LLM technique rather than a synthetic coded-vector benchmark.

## Gate 6 — real model-shaped end-to-end slice

Required sequence:

```text
one real tensor slice
one real layer
one real expert
one routed layer
then full model
```

Report exact model revision, tensor shape, quantization and error tolerance.

---

# 11. Reproduction and evidence files

This branch stores the measured table in:

```text
benchmarks/ram-lookup-compute/results-2026-09-16.csv
```

and a compact reproduction/method guide in:

```text
benchmarks/ram-lookup-compute/README.md
```

The numbers in those files are copied from the actual console outputs from the interactive 2026-09-16 run; values that were not captured are explicitly marked unknown rather than inferred.

---

# 12. Bottom line

The experiment started with a losing design and found a measurable crossover by increasing the amount of precomputed/coded arithmetic represented by one memory lookup:

```text
V1  1 MAC/lookup    -> lookup path 14.11× slower than CPU
V5  4 MAC/lookup    -> 256 MiB pressure path 0.19× CPU
V6 16 MAC/lookup    -> 256 MiB pressure path 0.59× CPU
V7 32 MAC/lookup    -> 256 MiB pressure path 1.11× CPU
V8 64 coded ops     -> 256 MiB pressure path 4.52× scalar branchless CPU
```

The strongest scientifically defensible conclusion is:

> On this stock-laptop software benchmark, table/materialized-function execution can outperform scalar recomputation when each lookup represents enough useful work; a deliberately dispersed 256 MiB lookup path crossed the scalar direct-compute baseline at 32 coded MAC-equivalents/lookup and reached 4.52× the scalar branchless control at 64 coded operations/lookup. The result is exact for the synthetic coded domains tested, but it is not physical in-DRAM compute and not yet a generic or accuracy-validated LLM kernel.

That is the result to carry forward into the next ByteMyWave/Transit experiments.