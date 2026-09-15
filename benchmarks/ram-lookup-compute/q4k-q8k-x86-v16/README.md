# Transit V16 production-oriented Q4_K × Q8_K kernel

V16 keeps the original Q4_K weights exactly and changes only the runtime layout and compute kernel.

## Main upgrades over V15

- exact Q4_K × Q8_K arithmetic; no codebook approximation;
- 64-byte-aligned runtime blocks;
- FP32 `d` / `dmin` metadata materialized offline;
- scales/mins unpacked offline;
- lane-friendly row reorder so horizontal reductions produce canonical output order;
- **late-scale** integer accumulation: accumulate Q4×Q8 dot fragments first, apply the Q4_K 6-bit scales once per 32-weight group;
- x16 fused AVX2 path;
- **AVX-VNNI `vpdpbusd`** path when `-march=native` exposes it (Alder Lake laptop target);
- prefetch autotune PF0/PF1/PF2/PF4;
- production-style multi-thread throughput sweep;
- Ivy Bridge SSSE3/SSE4.1/F16C path for HP Gen8 servers;
- V15 x8meta compiled into the native benchmark as an apples-to-apples control.

Runtime weight expansion is 1.05556× Q4_K (+5.56%). GGUF is unchanged; the expansion exists only in the runtime repacked buffer.

## Run

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\run-v16.ps1 `
  -ModelBlob "C:\Users\DSV\.ollama\models\blobs\sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490" `
  -Target Both `
  -Iters 21 `
  -Threads 8
```

For the laptop-only race, use `-Target Native`.

## Important output

Native:
- `V15 x8meta`
- `V16 x16 PF*`
- `V16 VNNI PF*`
- `Best V16 / V15`
- thread scaling

Correctness must remain:

```text
rel-L2=0
cosine=1.000000000
max_abs=0
```

The VNNI hot loop was assembly-audited on the development build: `vpdpbusd` is emitted and there were no vector spills to the stack in the inspected PF2 kernel. Ivy remains a separate optimization target because register pressure is materially worse without AVX2 integer operations.
