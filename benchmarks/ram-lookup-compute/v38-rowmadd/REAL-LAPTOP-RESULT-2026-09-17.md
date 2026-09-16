# V38 ROWMADD — real pinned laptop result — 2026-09-17

Pinned model blob:
`C:\Users\DSV\.ollama\models\blobs\sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490`

Tensor: `blk.0.ffn_gate.weight`

Promotion machine: Intel Core i5-12500H. Only 1/2/4 P-core runs are promotion-grade. Strict-Ivy ISA contract: SSSE3/SSE4.1 + AVX, no AVX2/FMA/VNNI.

## Stable result

- HOT: `V38_ROW4MADD_I2 @1T` = 1.48837x vs V27, p10=1.00444, win=0.904762, diff=0.
- ROTATE: `V38_ROW4MADD @1T` = 1.47948x vs V27, p10=1.01279, win=0.903226, diff=0.
- ROTATE 2T: `V38_ROW4MADD` = 1.38739x, p10=1.17318, win=0.903226, diff=0.
- ROTATE 4T: `V38_ROW4MADD_I2` = 1.20205x, p10=1.12468, win=0.935484, diff=0.

Final benchmark verdict:
- HOT: SUCCESS +48.84% vs V27; +6.95% vs V37.
- ROTATE: SUCCESS +47.95% vs V27; +24.04% vs stable V37 control in the same V38 run.

V38 therefore becomes the previous champion/control for V39. V27 remains preserved as fallback/control.
