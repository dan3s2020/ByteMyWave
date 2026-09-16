# Native Q4_K × Q8_K AVX2 source manifest — 2026-09-16

This directory belongs to the follow-up branch `bench/q4k-q8k-native-avx2-2026-09-16`, created from the head of PR #16 (`research/ram-lookup-compute-2026-09-16`).

The full experiment history/results are in:

- `docs/18-NATIVE-Q4K-Q8K-AVX2-2026-09-16.md`
- `docs/19-CROSS-BRANCH-NATIVE-KERNEL-AUDIT-2026-09-16.md`

## Exact source checkpoints stored here

The source snapshots are gzip-compressed and Base64-encoded so the GitHub text connector can preserve exact bytes.

| Source | Role | Source SHA-256 | Encoded file SHA-256 |
|---|---|---|---|
| `kernel_v7_1.cpp` | shuffled paired A/B/A tile-width sweep | `2127a87662f05ff76308c98fe38e208712975fa38f578727e347ee79cf6baf2b` | `2720c2c6c3f2afcd2906eaf53ea44dd0f07f1aa935f412126b0c2de7da70d01e` |
| `kernel_v9.cpp` | explicit 4-row exact vs generic/fused | `55a7b6ab73e7df2ae7f45ee65014a07d9fe76396c651b46df7c6aedd1dd650e3` | `72f855208fb8531f33a04c21c6508b9e4b68c1cd16b498cecec069fda58e1e55` |
| `kernel_v11.cpp` | row-major vs physical 4-row repack, ~101 MiB streaming | `a4f25e15f6d4667894b9f726b4ffbbd2f1198bcc925f2f4dbefb2c006630a938` | `cd426c25d672dd13428587818965388c5df77720d0e09128e9d1b516503be289` |

Files:

```text
sources/kernel_v7_1.cpp.gz.b64
sources/kernel_v9.cpp.gz.b64
sources/kernel_v11.cpp.gz.b64
```

## Restore on Windows/PowerShell

Example for v11, using Python already available on the benchmark host:

```powershell
python -c "import base64,gzip,pathlib; p=pathlib.Path(r'benchmarks/ram-lookup-compute/native-q4k-q8k/sources/kernel_v11.cpp.gz.b64'); pathlib.Path('kernel_v11.cpp').write_bytes(gzip.decompress(base64.b64decode(p.read_text().strip())))"

Get-FileHash .\kernel_v11.cpp -Algorithm SHA256
```

Expected source SHA-256:

```text
a4f25e15f6d4667894b9f726b4ffbbd2f1198bcc925f2f4dbefb2c006630a938
```

Compile:

```powershell
g++ .\kernel_v11.cpp `
    -O3 `
    -std=c++17 `
    -march=native `
    -mtune=native `
    -o .\kernel_v11.exe

.\kernel_v11.exe 4
```

The `4` is the logical CPU that measured fastest in the original i5-12500H affinity sweep. Do not blindly use that CPU number on another host.

## Historical source hashes

The detailed history document records these historical source hashes even when the source is not duplicated here:

```text
kernel.cpp
  a5d0735347444cba68bde0d97be48abe2a50cb58bf63d873004d14b3993f5184

kernel_v2_1.cpp
  267451f5c36f2ee7ad2382bc3684008f55937e5bc21f14a05b8a575cb51e7419

kernel_vs_llama.cpp
  ea45a05242abadfefb8d6f7a1a3e796ce2293a0f2c098fe5c9f10ee43166bc38

kernel_v3.cpp
  6680e1733b8a88afc262b29fb5e8c414646ed2756d3c132bc8b795f229c35b36

kernel_v7_1.cpp
  2127a87662f05ff76308c98fe38e208712975fa38f578727e347ee79cf6baf2b

kernel_v8_asm_probe.cpp
  9bd6bdeb311f069a072886f7173000e32ad2400cfd2e6af2d4b40e9c7326a09f

kernel_v9.cpp
  55a7b6ab73e7df2ae7f45ee65014a07d9fe76396c651b46df7c6aedd1dd650e3

kernel_v10.cpp
  c0ac03f68ffc54844313383fcc935f89adfa929f5fd50af54fb37cf9fd2eb0a8

kernel_v11.cpp
  a4f25e15f6d4667894b9f726b4ffbbd2f1198bcc925f2f4dbefb2c006630a938
```

## Evidence rules for future agents

- The measured native result is a **kernel/subsystem** result, not a measured full-model token/s result.
- The strongest beyond-LLC controlled result is the V10 ~101 MiB streaming A/B/A result: median `1.1809x`, IQR `[1.1504, 1.2031]` against the standalone reproduced direct llama.cpp Q4_K×Q8_K path.
- V11 row-major reached median `1.2640x`, IQR `[1.2499,1.2723]` in that run.
- Physical 4-row repacking was slightly slower than row-major on the i5-12500H (`0.9735x` candidate ratio); do not promote it as a win.
- Current llama.cpp also has production repacked GEMV Q4_K paths; a real `llama-bench` integration is still mandatory before claiming end-to-end gain.
- The AVX2 source does not run on Ivy Bridge E5-2680 v2. A separate AVX/SSE/POPCNT port is required.
