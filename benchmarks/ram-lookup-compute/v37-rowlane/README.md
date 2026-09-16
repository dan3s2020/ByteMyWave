# V37 ROWLANE — exact 4-row SIMD lane layout

V37 follows the real pinned V36 laptop run. V36 did not replace the V35 champion overall, so V35 remains the previous champion/control.

## Mechanism

V37 changes the execution layout rather than adding another arithmetic fold. The original packed Q4_K payload is permuted offline into 4-row × 4-position chunks. Each 16-byte XMM then contains four Q4 positions for each of four output rows. Low/high nibbles still encode the same paired Q4_K groups.

Activation 4-tuples are swizzled once per Q8_K activation block and reused across all output row-groups. Eight PMADDUBSW chunk results accumulate in int16 lanes; each lane covers only 16 Q4×Q8 products and is bounded by [-30720, 30480], so the accumulation is exact. One PMADDWD then produces four exact 32-weight dot products, one per int32 SIMD lane. Four row scales are applied together with PMULLD. No final horizontal reduction is required for those four output rows.

For the 9216×2560 test tensor, ROW4LANE has the same stored byte size as X8F: it is a permutation of the Q4 payload, not an expansion.

## Controls

V27 META is never removed. V31 FSA, V33 PAIR4/PAIR8, V34 APM, V35 MF/APM_MF and V36 HF/APM_HF remain in the same paired harness.

## Promotion gate

A V37 result is promoted only with diff=0, paired median speedup >1, p10>1 and win_rate>=0.80. The runner also compares the best stable V37 result against the best stable pre-V37 control in the same run and prints a final SUCCESS/FAIL percentage verdict.

Laptop results select mechanisms; E5-2680 v2 physical-server results decide production scaling.