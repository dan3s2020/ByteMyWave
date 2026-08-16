# Exact historical Transit host benchmark bundle

This directory preserves the exact source bundle used during the host-side Transit experiments.

The ZIP is stored as four base64 text fragments because the GitHub connector used to document this branch could write UTF-8 text files but could not upload the local binary ZIP directly.

## Archive

Concatenate, in name order:

```text
transit_host_benchmarks_full.zip.b64.part00
transit_host_benchmarks_full.zip.b64.part01
transit_host_benchmarks_full.zip.b64.part02
transit_host_benchmarks_full.zip.b64.part03
```

Decode the resulting base64 stream to:

```text
transit_host_benchmarks_full.zip
```

Expected SHA-256:

```text
b5fffeaa9af6d016350660f6108d811eddb9837c6ef0cbb5d06639d83f8a7e37
```

The reconstructed ZIP contains the exact historical files:

```text
transit_ollama_ssd_test.py
transit_asm_v2.py
transit_ddr5_bench_v3_lowram.py
transit_maskedsum_v4_1_fixed.py
transit_final_host_overlap.py
transit_bitplane_kernel.asm
transit_asm_runner.py
run_transit_asm.ps1
```

## Linux / macOS reconstruction

Run from this directory:

```bash
cat transit_host_benchmarks_full.zip.b64.part* \
  | tr -d '\n\r' \
  | base64 -d \
  > transit_host_benchmarks_full.zip

sha256sum transit_host_benchmarks_full.zip
unzip -l transit_host_benchmarks_full.zip
```

The printed SHA-256 must match the value above before treating the archive as the historical source bundle.

## PowerShell reconstruction

Run from this directory:

```powershell
$parts = Get-ChildItem 'transit_host_benchmarks_full.zip.b64.part*' | Sort-Object Name
$b64 = ($parts | ForEach-Object { Get-Content $_ -Raw }) -join ''
$b64 = $b64 -replace '\s',''
[IO.File]::WriteAllBytes(
    'transit_host_benchmarks_full.zip',
    [Convert]::FromBase64String($b64)
)

Get-FileHash .\transit_host_benchmarks_full.zip -Algorithm SHA256
```

Then inspect/extract normally:

```powershell
Expand-Archive .\transit_host_benchmarks_full.zip .\transit_host_benchmarks_full
```

## Why preserve the exact bundle

The documentation in `docs/08-EVIDENCE-BENCHMARKS.md` records measured results, but these files preserve the actual scripts that produced those experiments. This matters because several design decisions depended on negative as well as positive results:

- Python LUT lookup was too slow;
- native assembly made the bitplane identity practical;
- SSD saturated around the measured ~6 GB/s range;
- the 6 GiB V3 test proved the result was DRAM-scale rather than cache-only;
- V4 masked-sum was exact but slower on the CPU;
- the final host test showed CPU+DDR5 and SSD could overlap strongly while the mobile GPU suffered under simultaneous load.

The archive is therefore project evidence/history. It should not be edited in place. New experiments belong in normal source files with their own commits.
