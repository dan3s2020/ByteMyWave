# V32 STABILITYFORGE — paired laptop result

Date: 2026-09-16
Machine: Intel Core i5-12500H, 12 physical cores / 16 logical processors
Input: pinned prior Q4_K GGUF blob, tensor `blk.0.ffn_gate.weight`, geometry 9216 x 2560
Method: alternating AB/BA paired timings against V27 META; exactness required (`diff=0`).

## Stable findings

The strongest repeatable HOT result is `V31_FSA_RB16` at 4 physical threads:
- paired median speedup: 1.26162x
- p10: 1.10758x
- p90: 1.61619x
- diff: 0

`V31_FSA_RB4` at 4 physical threads is also stable:
- paired median speedup: 1.12614x
- p10: 1.05047x
- p90: 1.23803x
- diff: 0

The strongest repeatable COLD/beyond-LLC results occur at one physical thread:
- `V31_FSA_RB2`: median 1.19746x, p10 1.03182x, p90 1.52065x, diff=0
- `V31_FSA_RB16`: median 1.18996x, p10 1.0681x, p90 1.38394x, diff=0

The 4-thread COLD medians are >1 for multiple FSA variants, but p10 falls below 1, so those wins are not yet promotion-grade under the V32 stability rule.

## Hybrid-core interpretation

The benchmark's CPUID classification reports the first four selected physical cores as type `0x40` and the remaining eight as `0x20`. On Intel hybrid CPUs, 0x40 is the P-core type and 0x20 is the E-core type. Therefore the 4-thread sweep is P-core-only, while the 8- and 12-thread sweeps mix P- and E-cores.

This is a critical interpretation boundary: the loss at 8/12 threads on the i5-12500H must not be treated as evidence that FSA fails to scale on the homogeneous E5-2680 v2 target. The physical server run remains authoritative for 6/8/10 homogeneous cores per socket.

## Negative / rejected results

- `V30_META_X8F` does not beat V27 under paired-stable timing on this laptop.
- `EXP8_PF2` and `SCALED16_PF2` lose under paired-stable cold timing despite earlier single-run wins; those earlier wins were benchmark noise.
- Multi-token VEC reuse remains useful directionally, but the current V32 run does not apply the same AB/BA paired stability method to batch paths, so batch speedups are not promoted here.

## Next mechanism

V33 should preserve V27 and the stable FSA controls, then attack redundant work inside FSA rather than add another large representation:

1. Fuse the even/odd Q4_K nibble groups that share the same 32 packed bytes. Current FSA reloads the same packed Q bytes separately for low- and high-nibble groups. Pair-fusion can load once, derive low/high nibbles once, and feed both activation groups.
2. Replace four independent horizontal reductions with packed 4-row `PHADDD` reduction.
3. Vectorize the Q4_K min-correction across four rows using SSE4.1 integer multiply/add.
4. Keep AB/BA paired timing and add a rotating-address beyond-LLC mode instead of relying only on a 256 MiB eviction sweep.
5. On the hybrid laptop, promote from P-core-only 1/2/4 sweeps. On homogeneous E5-2680 v2, sweep the full physical-core range.

## Evidence boundary

These are measured kernel/subsystem results on the pinned 9216 x 2560 real Q4_K tensor. They are not end-to-end model decode throughput and they do not yet establish dual-socket server scaling.
