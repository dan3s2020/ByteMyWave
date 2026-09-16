# V37 ROWLANE — real pinned laptop result — 2026-09-17

Pinned model blob:
`C:\Users\DSV\.ollama\models\blobs\sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490`

Tensor: `blk.0.ffn_gate.weight`

Promotion machine: Intel Core i5-12500H. Only 1/2/4 P-core runs are promotion-grade. Exact strict-Ivy ISA contract: SSSE3/SSE4.1 + AVX, no AVX2/FMA/VNNI required.

## Result

V37_ROW4LANE passed exactness (`diff=0`) and the paired AB/BA stability gate.

- HOT 1T: 1.46251x vs V27, p10=1.12624, win=1.0.
- HOT 2T: 1.42302x, p10=1.27241, win=1.0.
- HOT 4T: 1.27102x, p10=1.10872, win=1.0.
- ROTATE 1T: 1.41463x vs V27, p10=1.23325, win=0.967742.
- ROTATE 2T: 1.38532x, p10=1.20202, win=0.967742.
- ROTATE 4T: 1.20553x, p10=1.14528, win=0.967742.

Final benchmark verdict:
- HOT: SUCCESS +46.25% vs V27 and +8.38% vs previous V35 champion.
- ROTATE: SUCCESS +41.46% vs V27 and +12.14% vs previous V35 champion.

## Mechanism

ROW4LANE keeps the same Q4 payload bytes as X8F but repacks them offline so four output rows map directly to the four int32 SIMD lanes. This removes per-row scale broadcasts and the final horizontal reduction used by earlier row-block kernels.

V27 remains preserved as fallback/control. Physical E5-2680 v2 measurements remain the authority for production scaling.