# TRANSIT V38 ROWMADD

> **REPRODUCIBILITY RULE:** the authoritative implementation is the exact benchmark artifact under [`EXACT-PACKAGE/`](EXACT-PACKAGE/), not a reconstruction from this design document. Run `EXACT-PACKAGE/RESTORE_EXACT_V38.ps1`; it must reproduce package SHA256 `4b6a717f5d30725246088a07783c754100183fe59b46c10023cf79b0709b8088`, source SHA256 `61a1d4aa0364a1f3d8daa0b9d704b4b7339b25f97a86771fc4649d538714bf45`, and runner SHA256 `c203bbb4d010528b17851614d78c9a353a905a0d62abc9660f1ca3b0b8ee2a07`. **Do not call reconstructed code “V38 exact.”**

V38 starts from the real V37 ROW4LANE champion and preserves V27 META.

Pinned input:
- model blob: `C:\Users\DSV\.ollama\models\blobs\sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490`
- tensor: `blk.0.ffn_gate.weight`

## Exact source / package

The complete original `TRANSIT_V38_ROWMADD.zip` that generated the recorded benchmark is preserved losslessly in `EXACT-PACKAGE/` as six Base64 transport parts because the repository connector writes UTF-8 text. `RESTORE_EXACT_V38.ps1` concatenates those exact bytes, verifies the ZIP SHA256, extracts the original files, and verifies the exact `.cpp`, runner, and manifest hashes. The original runner is also exposed directly as `EXACT-SOURCE/RUN.ps1` for inspection.

This exact-package artifact supersedes any earlier incomplete `EXACT-SOURCE` fragment or any source reconstructed from README/design notes.

## Why V38 exists

V37 maps four output rows directly into the four int32 SIMD lanes and reached the real laptop record: HOT 1.46251x vs V27 and ROTATE 1.41463x vs V27, exact diff=0.

Inside ROW4LANE, each 32-weight group still used `PMADDWD(sum16, ones)` followed by `PMULLD(dot, scale)`. V38 uses distributivity and the proven int16 bounds to compute `PMADDWD(sum16, repeated_scale16)` directly, preserving the exact result while removing the per-group PMULLD scale stage.

## New candidates

- `V38_ROW4MADD`: direct PMADDWD scaling. Scale bytes remain compact and are expanded at runtime with `PSHUFB`. Same static bytes as V37 ROW4LANE.
- `V38_ROW4MADD_I2`: same arithmetic with two independent accumulation chains to expose ILP around PMADDUBSW latency.
- `V38_ROW4MADD_PRE`: RAM-for-instructions control; pre-expands scale vectors offline at +192 B/tile.

Min metadata is also permuted pair-major without increasing min byte count.

## Promotion gate

A V38 path advances only if `diff=0`, paired median >1, p10 >1, win-rate >=80%, and it beats stable V37 ROW4LANE in the same run. The runner prints explicit `SUCCESS +X%` or `FAIL TO ADVANCE` for HOT and ROTATE.

ISA contract remains strict Ivy: `-march=ivybridge -mssse3 -mavx -mno-avx2 -mno-fma`.

Development sanity is only a gate. Pinned real laptop and physical E5-2680 v2 results decide promotion/production.
