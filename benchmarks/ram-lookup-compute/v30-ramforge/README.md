# Transit V30 RAMFORGE — experiment entry

Date: 2026-09-16  
Status: runnable benchmark package prepared; **awaiting physical E5-2680 v2 results**  
Base: `research/ram-lookup-compute-2026-09-16`

## Purpose

Evolve V27 META without deleting or weakening it. V27 remains the mandatory exact fallback. V30 candidates are promoted only when they are bit/exact-output compatible with the same Q4_K × Q8_K contract and beat V27 in a same-run benchmark on the target Ivy Bridge server.

## Why this track exists

V28 showed that an exact hot dictionary had no useful repeated-pattern coverage on the tested real tensor and lost to META. V30 therefore does **not** repeat that mechanism. It explores space-for-time compilation: multiple offline executable layouts of the same logical Q4_K weights, plus multi-token reuse and an explicit high-RAM NUMA replication experiment.

## Implemented candidates in the runnable package

- `V27_META` — preserved baseline/fallback, x8meta-style 8-row layout, predecoded scale/min metadata, strict-Ivy SSSE3/SSE4.1 arithmetic.
- `V30_META_X8F` — FP16 d/dmin expanded to FP32 and scale/min metadata transposed by group for 8-row SIMD.
- `V30_PACK8` — lossless pairwise Q4 repack across 8 output rows; same logical Q4 payload, rearranged so one compact load feeds 8 row partials.
- `V30_COL8` — aggressive expanded-Q4 column-major 8-row representation; intentionally spends RAM to reduce repeated decode/load work.
- `V30_SUM4` — activation-dependent 16-entry subset-sum functions with Q4 bitplane masks and SSSE3 `PSHUFB` lookup; exact relative to Q4_K arithmetic.
- `V30_BR_R4/R8/R16/R32` — arithmetic-progression base plus exact column-major residual correction. Reports residual density and falls back to V27 groups when the configured residual gate is not met.
- `V30_VEC4` / `V30_VEC8` — 4/8-activation weight/decode reuse for speculative verification or prefill-style batches; separate from batch=1 decode claims.
- `META-Atlas` — measures all exact candidates and selects a V30 path only if it beats the same-run V27 baseline; otherwise V27 wins automatically.
- `V29R_DUAL_REPLICATED` — explicit high-RAM NUMA experiment: one V27 representation per socket, disjoint output-row work, duplicated activation. This is intentionally a replication variant, not the no-duplication primary V29 plan.
- SSD compiled-layout cache — persists weight-dependent META, META_X8F, PACK8, COL8, SUM4 and BR representations so cold weights/experts can eventually be stored directly in executable layouts.

## Evidence boundary

Development-machine tests are only implementation/correctness checks. All implemented single-token candidates pass the exactness gate in the development harness, but relative speed changes with geometry/hardware. The physical dual-socket E5-2680 v2 run is authoritative.

The historical branch does not preserve a standalone original `v27/` source directory. The package therefore labels its V27 code honestly as a **V27-META contract reconstruction** from the documented x8meta/V27 semantics; it must not be described as byte-for-byte recovered V27 source.

## Promotion rule

```text
correctness passes
AND same input
AND same machine
AND same-run timing
AND candidate median < V27 median
```

If that rule is not satisfied, keep V27.

## Next gate

Run the supplied `TRANSIT_V30_RAMFORGE.zip` on the real HP DL360p Gen8 / dual E5-2680 v2 server. Preserve `MASTER_RESULTS.csv`, `SUMMARY.txt`, hardware inventory, candidate logs, and the generated 256-GB compiled-layout capacity plan. The next revision must follow the measured winner/loser pattern rather than the branch names.