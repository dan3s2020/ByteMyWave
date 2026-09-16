# V36 real pinned-laptop result — 2026-09-17

Input remained the pinned prior GGUF blob and `blk.0.ffn_gate.weight`, Q4_K 9216×2560, strict Ivy ISA contract, P-core-only 1/2/4 thread promotion sweep.

Stable V36 result: V36 itself did not advance the overall champion. The best stable HOT path in that run remained `V35_PAIR4_APM_MF @1T = 1.3132x vs V27` with p10 1.22476, win 0.952381, diff=0. The best stable ROTATE path likewise remained `V35_PAIR4_APM_MF @1T = 1.27613x vs V27`, p10 1.08425, win 1.0, diff=0.

`V36_PAIR4_APM_HF` reached 1.29631x HOT at 1T and 1.26556x ROTATE at 1T, but did not beat the V35 champion in either regime. At 4T, `V36_PAIR4_HF` was a stable ROTATE win over V27 (1.10118x, p10 1.06231, win 1.0) but still below the best existing path.

Conclusion: V36 is retained as an exact control, but it is FAIL TO ADVANCE. V37 therefore attacks row-parallel layout and reduction structure rather than another incremental horizontal fold.