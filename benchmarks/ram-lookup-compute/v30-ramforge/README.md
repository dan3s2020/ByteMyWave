# Transit V30 RAMFORGE — experiment entry

Date: 2026-09-16  
Status: first pinned-real-Qwen laptop run complete; physical E5-2680 v2 result still pending  
Base: `research/ram-lookup-compute-2026-09-16`

## Purpose

Evolve V27 META without deleting or weakening it. V27 remains the mandatory exact fallback. V30 candidates are promoted only when they are bit/exact-output compatible with the same Q4_K × Q8_K contract and beat V27 in a same-run benchmark in the relevant hardware/regime.

## Why this track exists

V28 showed that an exact hot dictionary had no useful repeated-pattern coverage on the tested real tensor and lost to META. V30 therefore does **not** repeat that mechanism. It explores space-for-time compilation: multiple offline executable layouts of the same logical Q4_K weights, plus multi-token reuse and an explicit high-RAM NUMA replication experiment.

## First real pinned-model result

The FIX2 runner used exactly the prior ByteMyWave model blob:

`C:\Users\DSV\.ollama\models\blobs\sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490`

and tensor:

`blk.0.ffn_gate.weight`

Measured tensor geometry: `9216 x 2560`, 12.6562 MiB original Q4_K bytes. Every reported exact candidate passed `max_abs_diff = 0`, cosine `1.0`.

Key single-token result on the i5-12500H while compiling to the strict-Ivy ISA contract:

```text
12T V27_META      0.3925 ms  1.000x
12T V30_META_X8F 0.3403 ms  1.153x
12T V30_PACK8    0.3504 ms  1.120x
```

Therefore V30 has produced the first same-run exact single-token win over the V27 baseline reconstruction. This is **not** yet a claim for E5-2680 v2: the Alder-Lake laptop is heterogeneous and the tested tensor is small enough for substantial LLC residency after warm-up.

The scaling shape is important: V27 improved through 8T then regressed at 12T, while META_X8F continued improving. This points V31 toward shared-activation load reuse, row blocking, affinity/topology, cache-line layout and reduction/uop structure.

Multi-token reuse was also strong:

```text
V30_VEC8_META vs serial V27x8:
1T  1.473x
4T  1.343x
8T  2.176x
12T 2.048x
```

This remains a speculative-verification/prefill throughput result, not a batch=1 decode speedup.

Negative results remain part of the evidence:

- SUM4: ~0.17-0.27x V27 in this run.
- exact BR base+residual: ~50.717% residual density and no useful sparse residual regime for the tested thresholds.
- COL8: below V27 in all tested thread counts.

## Implemented V30 candidates

- `V27_META` — preserved baseline/fallback, x8meta-style 8-row layout, predecoded scale/min metadata, strict-Ivy SSSE3/SSE4.1 arithmetic.
- `V30_META_X8F` — FP16 d/dmin expanded to FP32 and scale/min metadata transposed by group for 8-row SIMD/shared-activation processing.
- `V30_PACK8` — lossless pairwise Q4 repack across 8 output rows.
- `V30_COL8` — expanded-Q4 column-major 8-row representation.
- `V30_SUM4` — activation-dependent 16-entry subset-sum functions with Q4 bitplane masks and SSSE3 `PSHUFB` lookup.
- `V30_BR_R4/R8/R16/R32` — arithmetic-progression base plus exact residual correction.
- `V30_VEC4` / `V30_VEC8` — 4/8-activation weight/decode reuse for speculative verification or prefill-style batches.
- `META-Atlas` — measures candidates and preserves V27 whenever a candidate does not pass correctness + speed gates.
- `V29R_DUAL_REPLICATED` — explicit high-RAM NUMA replication experiment for the real dual-socket machine.
- SSD compiled-layout cache — persists weight-dependent compiled layouts rather than rebuilding them in the timed hot path.

## V31 research / next experiment design

Read:

[`V31-POST-V30-RESEARCH-2026-09-16.md`](V31-POST-V30-RESEARCH-2026-09-16.md)

The V31 plan is derived from the measured V30 winner/losers plus current BitNet, Vec-LUT, T-MAC/T-MAN, llama.cpp old-x86 and Ivy-Bridge instruction evidence. Highest-priority new mechanism: fuse Q4_K group scale into `PMADDWD`, accumulate the four int32 partial lanes across groups, and defer horizontal reduction to the end; test FSA4/FSA8 with row-block autotuning and explicit hot-cache vs rotating/beyond-LLC modes.

## Evidence boundary

The physical dual-socket E5-2680 v2 run remains authoritative for the target deployment. Development-machine results are mechanism evidence, not server throughput claims.

The historical branch does not preserve a standalone original `v27/` source directory. The package therefore labels its V27 code honestly as a **V27-META contract reconstruction** from the documented x8meta/V27 semantics; it must not be described as byte-for-byte recovered V27 source.

## Promotion rule

```text
correctness passes
AND same input
AND same machine
AND same-run timing
AND candidate median < V27 median in the claimed regime
```

V27 is never removed. A final META-Atlas may keep different exact representations for cold/streaming tensors, hot experts, batch=1, VEC-N speculative verification, and each NUMA socket.
