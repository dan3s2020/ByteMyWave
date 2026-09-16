# Transit V32 STABILITYFORGE

Date: 2026-09-16
Base: `bench/v31-fusionforge-2026-09-16`

## Purpose

V31 produced real exact wins, but two complete runs also exposed severe scheduling/order artifacts on the hybrid Windows laptop. V32 does not add another family of kernels. It validates only the survivors with a paired/interleaved measurement protocol.

## Evidence inherited from V31

Reliable repeated signal:
- FSA is real and exact; RB4/RB16 repeatedly beat V27 in selected 1/4/12-thread HOT regimes.
- FSA RB2/RB4/RB16 repeatedly beat V27 in selected COLD regimes.
- 8-thread single-token behavior was neutral/negative on the laptop and is retained as evidence.
- EXP8/SCALED16 are generally poor HOT paths but showed potentially useful 12-thread COLD wins with PF2 in the second run.

Known artifact:
- first-run 12-thread HOT V27 jumped to 3.3114 ms while X8F stayed ~0.345 ms, creating a false 9.60x headline. Immediate repeat put V27 back at 0.3428 ms and X8F at 0.4353 ms. That 9.60x value is invalid for promotion.

## V32 method

For every survivor, time V27 and candidate as a close pair for at least 21 rounds. Alternate order each round:

- even round: V27 -> candidate
- odd round: candidate -> V27

Promotion metric is the **median of per-round `V27_time / candidate_time` ratios**, not the ratio of separately collected medians. Also report p10/p90 of the paired-speedup distribution.

HOT shortlist:
- V30_META_X8F control
- V31_FSA_RB4
- V31_FSA_RB16

COLD shortlist:
- V31_FSA_RB2
- V31_FSA_RB4
- V31_FSA_RB16
- V31_EXP8_PF2
- V31_SCALED16_PF2

All candidates retain exactness gates against V27. The pinned prior model and tensor remain unchanged. Physical E5-2680 v2 results remain the production authority.
