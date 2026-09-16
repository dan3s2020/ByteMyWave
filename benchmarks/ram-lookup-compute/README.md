# Stock-laptop RAM lookup-compute benchmark

Date: 2026-09-16  
Status: measured exploratory software benchmark  
Platform: Windows PowerShell x64; Intel Core i5-12500H; 2 x 8 GiB Samsung M425R1GB4BB0-CQKOL DDR5-4800. The exact CPU/DIMM inventory was captured after the original V1-V8 transcript and is preserved in the exact-Q4_K continuation results.

The complete interpretation, exact numbers and caveats are in:

- [`../../docs/16-STOCK-LAPTOP-RAM-LOOKUP-COMPUTE-2026-09-16.md`](../../docs/16-STOCK-LAPTOP-RAM-LOOKUP-COMPUTE-2026-09-16.md) — synthetic V1-V8 chronology;
- [`REAL-QWEN-Q4K-RESULTS-2026-09-16.md`](REAL-QWEN-Q4K-RESULTS-2026-09-16.md) — first real-Qwen codebook experiments;
- [`../../docs/17-REAL-QWEN-Q4K-EXACT-LUT-2026-09-16.md`](../../docs/17-REAL-QWEN-Q4K-EXACT-LUT-2026-09-16.md) — position-specific PQ plus lossless/original-Q4_K exact-LUT continuation;
- [`REAL-QWEN-Q4K-EXACT-LUT-RESULTS-2026-09-16.md`](REAL-QWEN-Q4K-EXACT-LUT-RESULTS-2026-09-16.md) — compact raw measured output and hardware inventory;
- [`results-2026-09-16.csv`](results-2026-09-16.csv) — V1-V8 machine-readable result table;
- [`V29-NUMA-PLAN-AND-LUT-CODEBOOK-PLACEMENT-2026-09-16.md`](V29-NUMA-PLAN-AND-LUT-CODEBOOK-PLACEMENT-2026-09-16.md) — next physical-server gate: V27 META per NUMA socket, plus explicit retained roles for LUTs/codebooks.

## What the benchmark does

The benchmark progressively replaces scalar inner-loop arithmetic with a precomputed/materialized function:

```text
CPU arithmetic:
    contribution = f(weight_block, activation_block)

lookup arithmetic:
    address = encode(weight_block, activation_block)
    contribution = LUT[address]
```

The CPU still forms addresses, executes the memory loads and accumulates returned partial sums. This is **not physical in-DRAM arithmetic**.

## Evolution

| Version | Work represented by one lookup | Table/representation | Purpose |
|---|---:|---|---|
| V1 | 1 Q4×INT8 product | 64 MiB universal product LUT | Establish naive baseline |
| V2 | 4 Q4×Q2 MACs | model-dependent compiled tables | Amortize lookup over more arithmetic |
| V3 | 4 Q4×Q2 MACs | 16 MiB universal 24-bit table | Remove V2 table explosion |
| V4 | 4 Q4×Q2 MACs | 64 KiB codebook LUT | Test tiny/local table |
| V5 | 4 MACs | same codebook for CPU and LUT; 64 KiB + 256 MiB pressure table | Fairer control + cache-pressure split |
| V6 | 16 Q4×Q1 MACs | 32 MiB base + 256 MiB pressure table | Increase work/lookup |
| V7 | 32 coded Q4×Q1 MACs | 128 KiB base + 256 MiB pressure table | Find large-working-set crossover |
| V8 | 64 coded Q4×Q1 ops | 128 KiB base + 256 MiB pressure table | Test larger block + branchless CPU control |

## Exact core formulas

### V1

```csharp
// CPU
sum += weight * activation;

// lookup
int address = replica | (q4 << 8) | activationByte;
sum += productLut[address];
```

### V2/V3/V4/V5: 4-element block

```text
partial = w0*x0 + w1*x1 + w2*x2 + w3*x3
```

V2 precompiles the table per fixed four-weight group. V3 uses a universal address containing four raw Q4 weights plus four Q2 activations. V4/V5 replace the raw four-weight block with an 8-bit codebook ID.

V5 is the first comparison where the direct CPU and lookup path use the same 1-byte-per-four-weights codebook representation.

### V6: 16-element block

```text
partial = sum(i=0..15) wi * xi
wi = Q4 value
xi ∈ {-1,+1}
```

A weight code selects a predefined 16-element Q4 vector. A 16-bit activation pattern contains the Q1 signs.

### V7: 32-element block

```text
weightCode     ∈ [0,255] -> one predefined 32-element Q4 vector
activationCode ∈ [0,255] -> one predefined 32-element Q1 vector

LUT[(weightCode << 8) | activationCode]
    = dot(weightVector, activationVector)
```

The 128 KiB base table contains `256 * 256` signed 16-bit results.

### V8: 64-element block

V8 uses the same table dimensions but each codebook entry expands to 64 elements:

```text
weightCode     ∈ [0,255] -> one predefined 64-element Q4 vector
activationCode ∈ [0,255] -> one predefined 64-element Q1 vector
```

The lookup therefore returns the exact 64-element dot product **for the restricted codebook pair**.

## Mandatory V8 interpretation rule

Do not write:

> one byte losslessly stores 64 arbitrary Q4 weights

That is false.

Write:

> one byte selects one of 256 predefined 64-element Q4 codebook vectors.

Likewise, one activation byte selects one of 256 predefined Q1 activation vectors.

The benchmark is exact for this coded domain and the equality checks passed, but model-quality impact of mapping real tensors to these codebooks has not been measured.

## Small-table vs large-table control

V5-V8 deliberately use two lookup surfaces.

### Small/base table

The real materialized function table. It is small enough in V5/V7/V8 to be cache-friendly.

### 256 MiB pressure table

The same correct table is replicated many times. A deterministic hash/replica selector spreads accesses through the 256 MiB array:

```text
replica = hash(row, group, tokenSalt) & replicaMask
address = replica * baseEntries + localAddress
result  = bigLut[address]
```

Every replica stores the same value for the same local address, therefore arithmetic remains identical.

The purpose is to destroy tiny-table locality and create a large working set.

**It is not proof that every lookup reaches physical DRAM.** Hardware counters are required for that claim.

## Key measured V1-V8 trend

```text
V5  4 MAC/lookup  : large path = 0.19x direct CPU
V6 16 MAC/lookup  : large path = 0.59x direct CPU
V7 32 MAC/lookup  : large path = 1.11x direct CPU
V8 64 coded ops   : large path = 4.52x scalar branchless CPU
```

V7 is the first observed large-working-set crossover.

## Real-Qwen continuation

The synthetic coded-domain work was followed by experiments on the real Q4_K tensor `blk.0.ffn_gate.weight` from the local Qwen checkpoint.

Position-specific product quantization did not recover sufficient fidelity:

```text
GROUP 16: output cosine mean 0.738135, 6.645 ms/GEMV
GROUP  8: output cosine mean 0.868671, 19.371 ms/GEMV
```

The next experiment abandoned codebook approximation and consumed the original Q4_K bytes directly. A 16-entry activation-specific nibble table preserved the direct kernel output exactly:

```text
DIRECT original Q4_K : 16.344 ms/GEMV
NIBBLE LUT exact     : 15.223 ms/GEMV
BYTE LUT exact       : 47.505 ms/GEMV

Nibble/direct        : 1.074x
Byte/direct          : 0.344x

NIBBLE vs DIRECT max abs diff = 0
BYTE   vs DIRECT max abs diff = 0
```

The nibble LUT is 0.563 MiB for this activation width; the byte-pair LUT is 9.000 MiB. The result therefore points toward **small/local exact materialized functions** rather than large lookup surfaces or lossy codebooks.

This is a managed C# proof kernel, not yet a comparison against llama.cpp's optimized native `Q4_K x Q8_K` path.

Reproducible continuation sources:

- [`qwen_position_pq_lut.py`](qwen_position_pq_lut.py)
- [`prepare_q4k_exact.py`](prepare_q4k_exact.py)
- [`q4k_exact_lut_v10.cs`](q4k_exact_lut_v10.cs)
- [`run_q4k_exact_lut_v10.ps1`](run_q4k_exact_lut_v10.ps1)

## Why the `tok/s` unit is synthetic

One benchmark "token" executes the configured matrix-like workload once. It is a convenient iteration-rate label inherited from the toy language-model harness.

It is not a real tokenizer/model decode token and must not be compared directly to llama.cpp/vLLM/Kimi/GLM token/s.

## Reproduction protocol for the next run

Capture or verify the hardware inventory before comparing runs:

```powershell
Get-CimInstance Win32_Processor | Format-List Name,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed
Get-CimInstance Win32_PhysicalMemory | Format-Table Manufacturer,PartNumber,Capacity,Speed,ConfiguredClockSpeed
Get-CimInstance Win32_OperatingSystem | Format-List Caption,Version,BuildNumber
```

Also capture cache/ISA information with a trusted CPU tool or a small CPUID utility.

Then benchmark each path after warm-up, with the same generated codebooks/activations and exact checksum/equality validation.

For publication-grade comparison, port the exact-Q4_K experiment to native C/C++ and compare against the real optimized `Q4_K x Q8_K` path, with AVX2 on the i5-12500H where appropriate and hardware counters for cache misses, DRAM traffic, stalls and TLB behavior.

## Acceptance rules

A run is valid only if:

1. all compared paths use the same logical input/codebooks or the same original quantized bytes as appropriate;
2. checksums/output arithmetic match the stated exactness/tolerance criterion;
3. build/warm-up time is separated from timed inference work unless explicitly included;
4. table size and representation are reported;
5. small-table and large-working-set results are not conflated;
6. coded-vector results are not described as arbitrary-vector results;
7. synthetic tok/s is not presented as real model token/s;
8. the C# exact-LUT result is not presented as superiority over llama.cpp before the native `Q4_K x Q8_K` comparison is executed.

## Current next gate after V28

The V28 exact hot-dictionary experiment reported zero selected exact patterns at every tested threshold (`R2` through `R64`) and the fused path lost to same-run META. That mechanism is therefore closed unless a new source of reuse is demonstrated.

The next benchmark is V29 on the real dual-socket server:

```text
V27 META exact kernel
+ NUMA-local weight allocation
+ one persistent worker pool per socket
+ disjoint output-row shards
+ duplicated activation
+ local-vs-remote controls
+ exact correctness
```

LUT/codebook research is **not abandoned**. The linked V29 plan keeps it as a separate lever for regimes with real amortization/locality, near-memory SRAM/BRAM, numerically safe speculative expert prefetch, network/state compression, and small repeated decode/control transforms.