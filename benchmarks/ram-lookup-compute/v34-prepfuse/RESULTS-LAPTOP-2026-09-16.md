# V34 PREPFUSE — real pinned laptop result (2026-09-16)

Input stayed pinned to the prior ByteMyWave real GGUF blob and tensor `blk.0.ffn_gate.weight` (Q4_K, 9216x2560). The 12500H promotion sweep used only 1/2/4 P-cores. V27 META remained the paired AB/BA baseline. Stable promotion requires `diff=0`, paired median speedup > 1, p10 > 1, and win rate >= 0.80.

## Stable results

### 1 P-core
- `V34_PAIR4_APM` HOT: **1.28357x**, p10 **1.1198**, win **0.952381**, diff=0.
- `V34_PAIR4_APM` ROTATE: **1.24448x**, p10 **1.09907**, win **0.967742**, diff=0.
- `V31_FSA_RB16` HOT remains a strong control: 1.22856x, p10 1.17315.

### 2 P-cores
- `V34_PAIR4_APM` HOT: **1.21328x**, p10 **1.16946**, win **0.952381**, diff=0.
- ROTATE APM median was 1.17664x but p10 0.948495, so it is not promoted.
- `V33_PAIR4_VR` ROTATE remains stable at **1.18801x**, p10 **1.15474**.

### 4 P-cores
- `V31_FSA_RB16` HOT: **1.28058x**, p10 **1.07563**, win=1, diff=0.
- `V33_PAIR4_VR` HOT: **1.19677x**, p10 **1.06984**, win=1.
- `V33_PAIR4_VR` ROTATE: **1.08910x**, p10 **1.02484**, win **0.903226**.
- `V34_PAIR4_APM` HOT median 1.13179x but p10 0.872936; ROTATE median 1.07358x but p10 0.861659. These are not promoted at 4 P-cores.

## RAM-for-broadcast result

`V34_PAIR4_APM_SB` increased the tensor-side representation from 14,008,320 bytes (X8F) to 16,220,160 bytes (PAIRSCALE) in order to pre-expand scale broadcast metadata. It did **not** justify the extra memory:
- 1T HOT median 1.1985x but p10 0.926724 -> unstable.
- 1T ROTATE median 1.15988x but p10 0.960907 -> unstable.
- 4T ROTATE = 0.987979x -> slower than V27.

Conclusion: keep APM, reject scale-broadcast RAM expansion as a production direction for now.

## Interpretation

V34 proves that moving activation-invariant qsum/min coefficients out of the output-row loop is a real exact win, especially at 1 P-core. It also shows that the next step should reduce arithmetic inside the Q4_K dot path rather than simply increasing metadata size.

V27 META remains preserved. Physical E5-2680 v2 results remain the production authority.
