# V31 FUSIONFORGE — real laptop results

Date: 2026-09-16
Machine: Intel Core i5-12500H, 12 cores / 16 logical processors, 16 GB RAM
Input: pinned prior Qwen GGUF blob `sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490`
Tensor: `blk.0.ffn_gate.weight`, Q4_K, 9216 x 2560, 12.6562 MiB raw Q4_K bytes
Build contract: strict Ivy-compatible SSSE3/SSE4.1 + AVX, no AVX2/VNNI/FMA required

## What is definitely real

All tested V31 candidates passed the exactness gate with `max_abs_diff = 0` and cosine 1.0 relative to the preserved V27-META contract baseline.

Across two complete runs, fused-scale accumulation (FSA) repeatedly beat V27 in selected thread regimes:

- 1 thread HOT: `FSA_RB16` ~1.14–1.20x; COLD: `FSA_RB2/RB16` ~1.13–1.28x.
- 4 threads HOT: `FSA_RB4` reached 1.37x in the cleaner second run; COLD: `FSA_RB4/RB16` ~1.16–1.20x.
- 12 threads HOT in the cleaner second run: `FSA_RB4` 1.240x and `FSA_RB16` 1.234x.
- 8 threads did not show a reliable single-token win; V27 remained best in both HOT and COLD in the second run.

This is strong evidence that the V31 mechanism is real but thread-count/layout sensitive.

## High-RAM representations

`EXP8` and `SCALED16` are not general HOT winners. They expand the weight payload substantially and usually lose in HOT mode. However, in the second 12-thread COLD run:

- `EXP8_PF2`: 1.314x vs same-run V27.
- `SCALED16_PF2`: 1.377x vs same-run V27.

These results are interesting because they explicitly trade RAM bytes for less decode/arithmetic, but they are not yet stable enough to promote. They need paired/interleaved repetition because the 12-thread COLD baseline varied heavily between runs.

## Important benchmark artifact discovered

Some headline values in the first run are invalid as performance claims because the V27 baseline itself became a scheduling outlier. Example: first-run 12-thread HOT V27 was 3.3114 ms while X8F was 0.3448 ms, producing an apparent 9.60x speedup. In the immediately repeated run, 12-thread HOT V27 returned to 0.3428 ms and X8F was 0.4353 ms.

Therefore:

- do **not** promote the 9.60x X8F headline;
- do **not** promote multi-token 5x headlines from rounds where the serial baseline is similarly unstable;
- next benchmark must use paired/interleaved candidate-vs-baseline timing, randomized/alternating order, and report ratio distributions rather than dividing separately measured medians.

## Stable interpretation

The robust signal is not a single giant number. It is:

1. FSA reduces work enough to beat V27 repeatedly in several physical-core regimes while remaining exact.
2. Optimal row block changes with thread count; fixed RB8 is not universally optimal.
3. 8-thread behavior is a negative/neutral regime on this hybrid laptop and must be retained as evidence.
4. Expanded representations only look promising in cold/high-thread regimes, exactly where RAM-for-compute could matter on the target Xeon servers.
5. Manual prefetch is generally harmful or inconsistent except for selected 12-thread cold expanded-representation cases.
6. The next experiment must separate true mechanism speedup from Windows hybrid scheduling and thermal/order noise.

## Next gate: V32 STABILITYFORGE

Run only the survivors in a paired, interleaved harness:

HOT shortlist:
- V27_META
- V31_FSA_RB4
- V31_FSA_RB16
- V30_META_X8F control

COLD shortlist:
- V27_META
- V31_FSA_RB2
- V31_FSA_RB4
- V31_FSA_RB16
- V31_EXP8_PF2
- V31_SCALED16_PF2

For each candidate, time baseline and candidate in alternating AB/BA order over many rounds and calculate the median of per-round speedup ratios plus dispersion. A candidate is promoted only if the paired distribution remains >1 and exactness remains zero-diff.

Physical dual-socket E5-2680 v2 remains the production authority.