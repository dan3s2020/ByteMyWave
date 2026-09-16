# V35 MADDFUSE — exact PMADDWD folding after V34 APM

V35 starts from the real pinned-laptop V34 result. `V34_PAIR4_APM` is the strongest stable 1-P-core path measured so far: HOT 1.28357x (p10 1.1198) and ROTATE 1.24448x (p10 1.09907), diff=0. The RAM-heavy `APM_SB` variant did not reliably pass the stable gate, so V35 attacks arithmetic redundancy instead of expanding metadata.

## Exact mechanism

For one Q4_K 32-weight group, the existing path produces two 8-lane int16 vectors using `PMADDUBSW`, one for weights/activations 0..15 and one for 16..31. Both halves use the same Q4_K group scale.

Old form:

```text
p0 = PMADDUBSW(q0, a0)
p1 = PMADDUBSW(q1, a1)
out = PMADDWD(p0, scale) + PMADDWD(p1, scale)
```

V35 exact fold:

```text
p0 = PMADDUBSW(q0, a0)
p1 = PMADDUBSW(q1, a1)
p  = ADD16(p0, p1)
out = PMADDWD(p, scale)
```

This is algebraically identical because the scale is common to the full 32-weight group. It is also int16-safe for Q4 0..15 and Q8 -128..127: each PMADDUBSW lane is within [-3840,3810], and the lane-wise sum of the two 16-byte halves is within [-7680,7620], so the fold cannot overflow int16.

## Candidates

- `V35_PAIR4_MF`: V33 PAIR4 + PMADDWD fold only; isolates the new mechanism.
- `V35_PAIR4_APM_MF`: V34 APM + PMADDWD fold; primary V35 candidate.

V27 META, V31 FSA, V33 PAIR4 and V34 APM remain controls.

## Hardware contract

Build for `-march=ivybridge -mssse3 -mavx -mno-avx2 -mno-fma`. No AVX2/VNNI/FMA dependency.

## Promotion gate

Paired AB/BA, real pinned model/tensor. Promote only when `diff=0`, median speedup > 1, p10 > 1, and win rate >= 0.80. HOT and ROTATE remain separate claims. The 12500H uses only 1/2/4 P-core runs for promotion; E5-2680 v2 physical-server measurements remain the production authority.
