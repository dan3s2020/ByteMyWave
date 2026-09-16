# TRANSIT V39 ILPFORGE

V39 starts from the real V38 ROWMADD champion and preserves V27 META, V37 ROW4LANE, and V38 ROW4MADD as mandatory controls.

Pinned input:
- model blob: `C:\Users\DSV\.ollama\models\blobs\sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490`
- tensor: `blk.0.ffn_gate.weight`

## Real V38 starting point

Pinned laptop result:
- HOT best stable: `V38_ROW4MADD_I2` = 1.48837x vs V27 (+48.84%), exact.
- ROTATE best stable new V38 path: `V38_ROW4MADD` = 1.47948x vs V27 (+47.95%), exact.
- Final V38 verdict: SUCCESS TO ADVANCE in HOT and ROTATE.

The V38 split is informative: I2 helps HOT, while simple ROW4MADD is often more robust under ROTATE. V39 therefore changes scheduling/dependency structure without changing Q4 bytes.

## New exact candidates

- `V39_ROW4MADD_I4`: four independent c-step PMADDUBSW accumulation chains. Tests deeper latency hiding versus register pressure.
- `V39_ROW4MADD_P2`: two independent pair-level dot/min accumulation chains. Breaks pair-to-pair dependencies with modest extra register state.
- `V39_ROW4MADD_I2P2`: combines V38 I2 even/odd c-chains with V39 pair-level dual accumulation.

All candidates use the existing compact `row4madd_tile`; no Q4 payload growth and no new static representation are required.

## Gate

V39 advances only when:
- `diff=0`;
- paired median speedup > 1;
- p10 > 1;
- win-rate >= 80%;
- and a V39 candidate beats a stable V38 candidate in the same run.

The runner prints explicit final verdicts with percentages vs V27 and vs V38.

ISA remains strict Ivy:
`-march=ivybridge -mssse3 -mavx -mno-avx2 -mno-fma`.

Development synthetic runs are only a gate. Pinned real laptop and then E5-2680 v2 measurements decide promotion/production.
