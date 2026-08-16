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
