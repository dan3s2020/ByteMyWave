# Transit exact Q4_K × Q8_K LUT1/LUT2/LUT3 V21

Date: 2026-09-16

This benchmark is the next gate after the committed Native V11 and Sol V20 measurements. It tests whether exact lookup tables that return one, two, or three Q4×Q8 products can beat the existing exact arithmetic paths without changing the Q4_K weights.

## What was audited first

The benchmark branch `bench/q4k-q8k-native-avx2-2026-09-16` contains the exact Native V11 source archive at:

`benchmarks/ram-lookup-compute/native-q4k-q8k/sources/kernel_v11.cpp.gz.b64`

Expected restored V11 SHA-256:

`a4f25e15f6d4667894b9f726b4ffbbd2f1198bcc925f2f4dbefb2c006630a938`

The same branch contains the committed V19/V20 results and V20 layout/scheduler specification, but the V20 C++ source itself is not present in the audited tree or the `research/ram-lookup-compute-2026-09-16` branch. Therefore this V21 harness does **not** falsely label a reconstructed function as the original V20 source. It uses a `V20-contract-COMPACT` control reconstructed from the committed V20 invariants:

- 148 B per Q4_K block (`13.008 MiB` for the real 23,592,960-weight tensor);
- original 128 Q4 payload bytes unchanged;
- FP16 `d/dmin` unchanged;
- eight scales + eight mins pre-unpacked;
- large rotating working set (`520.31 MiB` by default);
- persistent static worker pool and dispatch included in timing.

The PowerShell runner independently restores, hashes, compiles, and runs the exact archived Native V11 checkpoint before the new harness.

## Critical Q4_K correction

A raw Q4_K byte does **not** mean “two consecutive weights in the same scale group.” In the real Q4_K packing, low and high nibbles belong to two different 32-weight groups and therefore can have different scales/mins.

V21 fixes this before lookup benchmarking. At model-load time it losslessly repacks nibbles so every LUT pair/triple is formed only from consecutive weights inside the **same 32-weight scale group**. No weight value is approximated and no extra quantization is introduced.

The compact LUT layouts retain the same information footprint as V20 COMPACT: `148 B/block`.

## LUTs tested

### LUT1

Exact universal table:

`LUT1[a][q] = q * a`

- Q4 states: 16
- Q8 states: 256
- entries: 4,096
- int16 footprint: **8 KiB**

### LUT2

Exact universal table:

`LUT2[a0,a1][q0,q1] = q0*a0 + q1*a1`

- activation-pair states: 65,536
- Q4-pair states: 256
- entries: 16,777,216
- int16 footprint: **32 MiB**
- active slice per activation pair: **512 B**

### LUT3

Exact algebra:

`LUT3[a0,a1,a2][q0,q1,q2] = q0*a0 + q1*a1 + q2*a2`

A full universal LUT3 is:

- activation triples: `256^3`
- Q4 triples: `16^3`
- int16 footprint: **128 GiB**

The laptop harness cannot allocate the full 128 GiB table. Instead it materializes only the slices needed by the current activation vector. For the real `2560`-wide activation this is **6.25 MiB**. The slice-build time is reported separately and is not hidden inside kernel-only timing. On the Transit servers, the proposed universal 128-GiB table would replace that build with fetching the selected 8-KiB slices from RAM.

Each 32-weight scale group uses ten exact LUT3 triples plus one exact LUT2 tail pair: `30 + 2 = 32` weights. Scales and min corrections are applied with the original Q4_K algebra after exact integer accumulation.

## Two lossless lookup layouts are autotuned

V21 does not assume one memory order wins on every CPU.

It tests both:

- `*-row`: sixteen same-group Q4-pair bytes for each output row remain adjacent;
- `*-trans`: each pair position is transposed across output rows for contiguous row streaming.

Both carry exactly the same Q4 values and metadata. The result prints an `AUTOTUNE WINNERS` section for LUT1/LUT2/LUT3 independently under both benchmark gates.

## Correctness gates

Before timing anything, the executable compares all paths against the scalar Q4_K × Q8_K reference:

- Native11-compatible control;
- V20-contract COMPACT control;
- LUT1 row/transposed;
- LUT2 row/transposed;
- LUT3 row/transposed;
- four-thread persistent-pool versions of every lookup path.

The benchmark refuses to continue unless every path is **bit-exact** (`memcmp` on output floats), not merely close in relative L2.

The audited source was compiled locally with both:

- `-march=native -mtune=native`
- `-march=ivybridge -mtune=ivybridge -mno-avx2`

using `-Wall -Wextra -Wshadow -Wconversion -Wpedantic` with **zero compiler warnings**, and both native and Ivy self-tests passed bit-exactly.

A scheduler race discovered during this audit was fixed before delivery: the persistent pool now has a readiness barrier so the first published job cannot be missed by a worker that has not yet initialized its epoch.

## Benchmark methodology

### Gate A — Native V11-style streaming

Default rotating working set: **101.25 MiB**.

- one pinned worker;
- real `9216 × 2560` Q4_K tensor;
- baseline: `Native11-4row-compatible`;
- LUT1/2/3 row and transposed candidates.

The exact original V11 executable is also run separately by the PowerShell script as a provenance checkpoint.

Historical anchor from the repository, not substituted for the new run:

`0.7295 ms/GEMV`, `16.94 GiB/s`, one pinned fast logical CPU.

### Gate B — V20-style large streaming

Default rotating working set: **520.31 MiB**.

- persistent worker pool;
- default 12 threads on the i5-12500H;
- V20 physical-first CPU order from the recorded run when that CPU is detected;
- baseline: `V20-contract-COMPACT`;
- all LUT candidates use the same large-working-set gate;
- worker dispatch is included in timing.

Historical V20 anchor from the repository:

`0.228 ms/GEMV`, `59.95 GB/s`, 12-thread COMPACT best recorded result.

The controls in V21 are measured again. Historical numbers are printed only as audit anchors; they are never silently substituted for live measurements.

### Order-bias protection

Variants are not benchmarked in a fixed `control -> LUT1 -> LUT2 -> LUT3` sequence. Each timed round changes the order deterministically and reverses alternating rounds. Every variant collects its own samples and reports its median. This is intended to suppress turbo, thermal, and benchmark-order bias that affected earlier kernel experiments.

## Local pre-delivery full-path test

A synthetic GGUF with the exact real tensor geometry (`9216 × 2560`, 92,160 Q4_K blocks, 12.656 MiB) was used only to validate the complete code path: GGUF parsing, layout conversion, LUT construction, correctness, worker dispatch, rotating working sets, CSV generation, and interleaved benchmarking.

That synthetic run is **not a performance claim for the project laptop**. On the audit machine the lookup candidates were slower than the arithmetic controls; LUT2 was generally the strongest lookup candidate. This is precisely why V21 measures rather than assumes a 2×/3× gain.

## Run on the project laptop

From the new benchmark directory:

```powershell
.\run_lut123_v21.ps1 `
    -ModelBlob "C:\path\to\the\Qwen\GGUF-or-Ollama-blob" `
    -Tensor "blk.0.ffn_gate.weight" `
    -Iters 15 `
    -Threads 12 `
    -V11Cpu 4
```

Outputs:

- `RESULTS-LUT123-V21.txt`
- `RESULTS-LUT123-V21.csv`

Optional Ivy-compatible full run on the laptop as an ISA proxy:

```powershell
.\run_lut123_v21.ps1 `
    -ModelBlob "C:\path\to\blob" `
    -RunIvyProxyFull
```

That remains a proxy. The E5-2680 v2 servers are the authoritative Ivy Bridge measurement target.

## What to send back

Paste the complete console output or `RESULTS-LUT123-V21.csv`. The next calculation can then use the measured LUT1/LUT2/LUT3 ratios against both controls and extrapolate to a large model without running that large model.

## Delivered source integrity

Repository-friendly storage keeps the audited C++ source as three gzip+base64 text parts:

- `transit_lut123_v21.cpp.gz.b64.part00`
- `transit_lut123_v21.cpp.gz.b64.part01`
- `transit_lut123_v21.cpp.gz.b64.part02`

`run_lut123_v21.ps1` transparently restores `transit_lut123_v21.cpp` when it is absent and refuses to compile unless the restored source SHA-256 is exactly:

`8a1ddf084be0f63194e246cb12f756e243d35c42c2005ec569d706cb171fdc1a`

This is separate from the original Native V11 provenance gate, which independently verifies its historical source SHA before compiling it.
