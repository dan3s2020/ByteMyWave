# Transit host benchmark sources

These are the host-side experiments that produced the measured results documented in `docs/08-EVIDENCE-BENCHMARKS.md`.

They are preserved as evidence/history, not presented as the final Transit runtime. The expanded V1 assembly proof is kept directly in this directory; the **exact full historical source bundle**, including V2/V3/V4/final-overlap scripts, is preserved under [`archive/`](archive/README.md).

## Expanded files in this branch

- `transit_bitplane_kernel.asm` — first exact native x64 bitplane dot-product kernel for the 640×2048 test tensor.
- `transit_asm_runner.py` — builds/loads/runs the V1 flat binary kernel and compares it with the reference path.
- `run_transit_asm.ps1` — convenience launcher used on Windows.

## Exact archived source bundle

[`archive/README.md`](archive/README.md) explains how to reconstruct and verify the archived ZIP. Its expected SHA-256 is:

```text
b5fffeaa9af6d016350660f6108d811eddb9837c6ef0cbb5d06639d83f8a7e37
```

That exact archive contains:

- `transit_ollama_ssd_test.py` — real GGUF tensor extraction, simple signed-INT4 requantization, packed-Q4 vs bitplane representation, SSD/RAM probes and exact integer verification.
- `transit_asm_v2.py` — multicore hot-compute, direct-NVMe and NVMe+assembly pipeline benchmark.
- `transit_ddr5_bench_v3_lowram.py` — large working-set DDR5 benchmark proving the result is not merely cache throughput.
- `transit_maskedsum_v4_1_fixed.py` — corrected V4 masked-activation-sum experiment; exact but slower than V3 on the tested CPU.
- `transit_final_host_overlap.py` — final simultaneous CPU+DDR5 + SSD + GPU overlap experiment.
- the exact historical V1 assembly kernel, runner and PowerShell launcher.

The archive is immutable project evidence. New benchmark versions should be added as new source files/commits rather than replacing it.

## Historical result summary

```text
V1 exact                     max_abs_diff = 0
V1 best                      8.354 Gweights/s
V2 16-worker hot             53.697 Gweights/s
V2 NVMe read                 ~6.16 GB/s
V2 NVMe+ASM                  12.318 Gweights/s transport-fed
DDR5 raw 6 GiB set           ~50.015 GB/s
DDR5 V3 16-worker            53.673 Gweights/s
V4.1 16-worker               39.982 Gweights/s
V4.1/V3                      0.702777x
final CPU standalone         56.451 Gweights/s
final CPU simultaneous       51.385 Gweights/s
final SSD standalone         6.153 GB/s
final SSD simultaneous       5.787 GB/s
```

Do not interpret SSD Q4-equivalent throughput as computation. Do not interpret these weight-path rates as end-to-end K3 token rates.

## 2026-09-16 stock-laptop lookup-compute track

A separate software experiment records the progression from synthetic compute-by-lookup tests to real Qwen Q4_K experiments.

Start here:

- [`ram-lookup-compute/README.md`](ram-lookup-compute/README.md) — current track index, method, hardware inventory, reproduction rules and evidence boundaries;
- [`ram-lookup-compute/RAW-MEASUREMENTS-2026-09-16.md`](ram-lookup-compute/RAW-MEASUREMENTS-2026-09-16.md) — transcribed synthetic V1-V8 console measurements;
- [`ram-lookup-compute/results-2026-09-16.csv`](ram-lookup-compute/results-2026-09-16.csv) — machine-readable V1-V8 result table;
- [`../docs/16-STOCK-LAPTOP-RAM-LOOKUP-COMPUTE-2026-09-16.md`](../docs/16-STOCK-LAPTOP-RAM-LOOKUP-COMPUTE-2026-09-16.md) — full V1-V8 interpretation and caveats;
- [`ram-lookup-compute/REAL-QWEN-Q4K-RESULTS-2026-09-16.md`](ram-lookup-compute/REAL-QWEN-Q4K-RESULTS-2026-09-16.md) — first real-Qwen codebook/residual experiments;
- [`../docs/17-REAL-QWEN-Q4K-EXACT-LUT-2026-09-16.md`](../docs/17-REAL-QWEN-Q4K-EXACT-LUT-2026-09-16.md) — position-specific PQ and exact original-Q4_K lookup continuation;
- [`ram-lookup-compute/REAL-QWEN-Q4K-EXACT-LUT-RESULTS-2026-09-16.md`](ram-lookup-compute/REAL-QWEN-Q4K-EXACT-LUT-RESULTS-2026-09-16.md) — compact exact measured output plus laptop inventory.

Synthetic V1-V8 headline sequence:

```text
V5  4 MAC/lookup  : 256 MiB pressure path = 0.19x direct CPU
V6 16 MAC/lookup  : 256 MiB pressure path = 0.59x direct CPU
V7 32 MAC/lookup  : 256 MiB pressure path = 1.11x direct CPU
V8 64 coded ops   : 256 MiB pressure path = 4.52x scalar branchless CPU
```

These are synthetic software lookup benchmarks, not physical DRAM/PIM operations and not end-to-end LLM token/s. V8 additionally uses a restricted 256-entry vector codebook and must not be described as 64 arbitrary Q4 weights losslessly encoded in one byte.

### Real-Qwen continuation

Position-specific PQ on `blk.0.ffn_gate.weight` improved representation quality as the group size fell, but remained too lossy:

```text
G16 output cosine mean  0.738135
G8  output cosine mean  0.868671
```

The following test abandoned codebook approximation and consumed the original Q4_K bytes directly. The exact 16-entry nibble-LUT path matched the direct managed kernel output exactly and was slightly faster:

```text
DIRECT original Q4_K     16.344 ms/GEMV
NIBBLE LUT exact         15.223 ms/GEMV   = 1.074x direct
BYTE LUT exact           47.505 ms/GEMV   = 0.344x direct

NIBBLE LUT vs DIRECT     max abs diff = 0
BYTE LUT vs DIRECT       max abs diff = 0
```

The nibble LUT is 0.563 MiB; the byte-pair LUT is 9.000 MiB. This result motivates small/local exact materialized functions, not large lookup surfaces or further lossy codebook compression.

This remains a managed C# proof. The next falsifiable gate is a native optimized `Q4_K x Q8_K` comparison against the real direct SIMD baseline before any llama.cpp/Ollama or end-to-end model speedup claim.
