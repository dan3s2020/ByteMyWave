# TRANSIT V31 FUSIONFORGE

V31 is the next exact Q4_K x Q8_K kernel study after the measured V30 RAMFORGE run.

## Fixed input contract

Default model blob is the exact prior ByteMyWave model:

`C:\Users\DSV\.ollama\models\blobs\sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490`

Default tensor: `blk.0.ffn_gate.weight` (9216 x 2560 Q4_K in the prior run).

There is no model autodiscovery and no normal synthetic fallback in RUN.ps1.

## Why V31 exists

Measured V30 on the real tensor showed:

- V27 META best at 8T: 0.3764 ms;
- V30 META_X8F at 12T: 0.3403 ms vs same-run V27 0.3925 ms = 1.15339x;
- V30 PACK8 at 12T: 0.3504 ms = 1.12015x;
- VEC8 META reached 2.17558x vs eight serial V27 calls at 8T;
- SUM4 and exact base+residual were clear negatives;
- base+residual density was 50.717%, so sparse residual correction is not a good exact-Q4_K path for this tensor.

V31 therefore stops widening LUT/dictionary searches and attacks the observed loop/reuse bottleneck.

## New exact candidates

- `V31_FSA_RB2/RB4/RB8/RB16`: fused-scale accumulation. The Q4 scale is folded into `PMADDWD` so the kernel accumulates scaled int32 lanes and performs horizontal reduction only once per row/block instead of once per group.
- `V31_FSA_RB8_PF1/PF2/PF4`: same exact arithmetic with prefetch-distance sweep.
- `V31_EXP8_FSA`: Q4 nibbles expanded offline to bytes. This deliberately spends roughly 2x the Q payload to remove nibble extraction.
- `V31_EXP8_PF2`: expanded representation plus prefetch.
- `V31_SCALED16`: stores exact `q * scale` as int16 offline. This is a much larger compiled representation and removes nibble unpack + runtime scale multiply.
- `V31_SCALED16_PF2`: same with prefetch.
- `V31_VEC2/4/8/16_META`: activation-parallel / speculative-verification reuse with the exact V27 META weights.

Retained controls:

- `V27_META` — mandatory fallback;
- `V30_META_X8F` — previous measured winner;
- `V30_PACK8` — previous measured runner-up.

## Hot vs cold control

V31 reports two separate single-token suites:

- `HOT`: repeated kernel timing, useful for microarchitecture/cache-resident comparison;
- `COLD_EVICT`: touches a large eviction surface before each timed call (outside the timed region) to reduce reuse from the prior iteration and better expose memory-streaming sensitivity.

RUN.ps1 chooses 256 MiB on small-RAM machines, 1 GiB on >=64 GiB machines, and 2 GiB on >=512 GiB machines unless `-EvictMB` is supplied.

This is not a claim that COLD_EVICT perfectly reproduces full-model inference. It is a stricter control against optimizing only for a ~12.7 MiB hot tensor.

## Physical-core policy

The native runner enumerates one logical processor per physical core with `GetLogicalProcessorInformationEx(RelationProcessorCore)`. On heterogeneous Windows CPUs it sorts higher `EfficiencyClass` first. This avoids treating SMT siblings as independent physical cores in the sweep.

## Acceptance rule

V27 is never removed. A V31 single-token path is promoted only when:

1. exactness gate passes;
2. same real tensor and activation contract are used;
3. same-run candidate median beats the matching V27 mode;
4. HOT and COLD claims are kept separate;
5. final Transit production promotion is repeated on the E5-2680 v2 server.

## Run

From the package directory:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\RUN.ps1
```

The runner writes `MASTER_RESULTS.csv`, `SUMMARY.txt`, per-thread logs, hardware inventory, compiled-layout caches, and a multi-budget representation capacity plan.
