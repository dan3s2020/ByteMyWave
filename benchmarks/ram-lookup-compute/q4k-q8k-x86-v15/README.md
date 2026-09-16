# Transit V15 — exact x86 Q4_K × Q8_K kernel

This experiment moves the RAM lookup-compute track from approximate vector codebooks to the real llama.cpp CPU quantized path.

## Target

- Weight format: `Q4_K`, unchanged from GGUF/Ollama.
- Activation format: `Q8_K`, matching llama.cpp's `GGML_TYPE_Q4_K` CPU dot path.
- Main test tensor: `blk.0.ffn_gate.weight` from the local Qwen 3.5 4B Q4_K_M model (`2560 × 9216` in GGUF ne-order; GEMV interpretation `9216 output rows × 2560 input columns`, 12.656 MiB Q4_K).
- No extra model approximation is introduced. Correctness is measured against a scalar implementation of the same Q4_K × Q8_K arithmetic.

## Kernels in the standalone benchmark

`transit_q4k_q8k_exact_v15.cpp` compares:

1. scalar exact Q4_K × Q8_K;
2. packed SIMD control using original Q4_K layout;
3. llama-compatible `block_q4_Kx8` fused kernel — storage neutral;
4. `x8meta` — eight-row fused kernel with unpacked scale/min metadata, +2.78% runtime weight storage.

The x8 kernels reuse each Q8_K activation fragment across eight output rows. The core integer dot work is `pmaddubsw` / `pmaddwd` (`vpmaddubsw` / `vpmaddwd` on AVX2).

## Why x8 instead of a large random LUT

The exact packed-byte LUT tested earlier replaces only two low-bit products per lookup and loses badly to SIMD. V15 instead spends a small amount of RAM on a compute-friendly runtime layout and removes repeated metadata decode while preserving dense SIMD dot instructions.

This is consistent with the broader Transit observation: memory helps when it materializes enough useful work per access or removes CPU bookkeeping; simply replacing a cheap multiply with a random lookup is not enough.

## Real Qwen benchmark — 2026-09-16

User-run benchmark on the real Ollama Qwen 3.5 4B Q4_K_M blob, tensor `blk.0.ffn_gate.weight`, 23,592,960 weights / 12.656 MiB Q4_K. Single thread, median, 15 iterations.

All optimized paths were exactly equal to the scalar Q4_K × Q8_K operator for this run:

- packed SIMD: relative-L2 `0`, cosine `1.000000000`, max abs `0`;
- llama x8: relative-L2 `0`, cosine `1.000000000`, max abs `0`;
- x8meta: relative-L2 `0`, cosine `1.000000000`, max abs `0`.

Native build (`-march=native`, AVX2 + SSSE3 + F16C):

- scalar exact: `2.279 ms`, `10.35 GOP/s` logical;
- packed SIMD control: `1.254 ms`, `18.81 GOP/s`;
- llama-x8 fused: `1.303 ms`, `18.11 GOP/s`, `0.963×` vs packed;
- x8meta fused: `0.894 ms`, `26.39 GOP/s`, `1.403×` vs packed;
- x8meta RAM overhead: `2.78%`.

Ivy-compatible build (`-march=ivybridge -mno-avx2`, SSSE3 + F16C), executed on the same laptop as an ISA-target proxy rather than on the HP Gen8 Xeons themselves:

- scalar exact: `3.320 ms`, `7.11 GOP/s` logical;
- packed SIMD control: `1.331 ms`, `17.72 GOP/s`;
- llama-x8 fused: `1.295 ms`, `18.22 GOP/s`, `1.028×` vs packed;
- x8meta fused: `1.238 ms`, `19.07 GOP/s`, `1.076×` vs packed;
- x8meta RAM overhead: `2.78%`.

### Important baseline correction after V11

A later geometry-correct V11 harness added a closer port of the current ggml Q4_K × Q8_K AVX2 arithmetic shape. On the same laptop and the same real `9216 × 2560` tensor it measured approximately `0.860–0.910 ms/GEMV` kernel-only, faster than V15's `packed SIMD control = 1.254 ms` and roughly comparable to / slightly faster than V15's `x8meta = 0.894 ms` in those separate runs.

Therefore the earlier `1.403×` x8meta number is **only a win over V15's packed control**, not a demonstrated win over the best current ggml-style baseline. Do not cite V15 as a 40% improvement over llama.cpp.

The next valid gate is an apples-to-apples same-harness comparison of:

- current llama-style AVX2 baseline;
- V15 x8meta;
- same CPU affinity;
- paired/interleaved timing;
- hot-cache and cache-evicted streaming measurements.

See `../V11-REAL-GEOMETRY-RESULTS-2026-09-16.md` for the newer baseline results.

## Development sanity benchmark

These figures are development-machine sanity checks over valid synthetic Q4_K bytes, not measurements from the user's Qwen tensor or HP Gen8 servers. Repeated runs vary with CPU scheduling/turbo.

Representative ranges with `-O3 -fno-unroll-loops`:

- AVX2/F16C: llama x8 typically ~1.1× over packed control; x8meta commonly ~1.35–1.45×.
- Ivy Bridge / SSSE3/F16C target build: llama x8 roughly ~1.03–1.14×; x8meta roughly ~1.09–1.21×.

All custom paths returned relative-L2 `0` against the scalar exact implementation in the development checks. Different accumulation order may still produce tiny floating-point rounding differences on other compilers/builds.

`-fno-unroll-loops` is intentional for the Ivy build. An assembly audit showed aggressive compiler unrolling causing substantial stack traffic/spills around the SSSE3 hot loop.

## llama.cpp integration

Current llama.cpp already has a runtime repack structure named `block_q4_Kx8` and generic `q4_K_8x8_q8_K` GEMV/GEMM support. `llama_x86_q4k8x8_transit.inc` supplies an experimental x86 specialization with:

- AVX2 path;
- SSSE3 path for Ivy Bridge-class CPUs;
- generic fallback.

`llama-repack-selector-v15.patch` broadens the Q4_K x8 repack selection from AVX2-only to AVX2 or SSSE3 so the Gen8/Ivy Bridge target can reach the new path.

This integration remains experimental until it is built inside a full current llama.cpp tree and compared with llama.cpp's own benchmark/full-model inference.

## Run on the real local Qwen tensor

From PowerShell in this directory:

```powershell
.\run-v15.ps1 `
  -ModelBlob "C:\Users\DSV\.ollama\models\blobs\sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490"
```

If neither `clang++` nor `g++` is installed:

```powershell
winget install -e --id LLVM.LLVM
```

Open a new PowerShell and rerun the benchmark.

Important output fields:

- `packed relative L2`
- `llama-x8 relative L2`
- `x8meta relative L2`
- `packed SIMD control`
- `llama-x8 fused`
- `x8meta fused`
- `llama-x8 / packed`
- `x8meta / packed`

The standalone benchmark is an operator benchmark. The decisive next gate is compiling the same x86 kernel into llama.cpp and measuring `llama-bench` and actual model decode on the Gen8 servers.
