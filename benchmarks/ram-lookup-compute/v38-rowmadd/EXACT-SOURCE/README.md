# TRANSIT V38 ROWMADD

V38 starts from the real V37 ROW4LANE champion and preserves V27 META.

Pinned input:
- model blob: `C:\Users\DSV\.ollama\models\blobs\sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490`
- tensor: `blk.0.ffn_gate.weight`

## Why V38 exists

V37 maps four output rows directly into the four int32 SIMD lanes and reached the real laptop record:
- HOT: 1.46251x vs V27
- ROTATE: 1.41463x vs V27
- exact `diff=0`

Inside ROW4LANE, each 32-weight group still used:

```
PMADDWD(sum16, ones) -> int32 dot
PMULLD(dot, scale)
```

V38 exploits distributivity and the proven int16 bounds:

```
PMADDWD(sum16, repeated_scale16)
```

This computes the identical scaled dot directly and removes the per-group PMULLD stage.

## New candidates

- `V38_ROW4MADD`: direct PMADDWD scaling. Scale bytes stay compact and are expanded at runtime using SSSE3/AVX `PSHUFB`. Same static bytes as V37 ROW4LANE.
- `V38_ROW4MADD_I2`: same arithmetic, but splits the 8 c-steps into two independent accumulation chains to expose more ILP around PMADDUBSW latency.
- `V38_ROW4MADD_PRE`: RAM-for-instructions control. Pre-expands scale vectors offline; +192 bytes/tile versus ROW4LANE.

Min metadata is also permuted pair-major without increasing its byte count, removing the older convert+unpack sequence.

## Promotion gate

A V38 path advances only if:
- exactness: `diff=0`
- paired median speedup > 1
- p10 > 1
- win rate >= 80%
- and it beats the stable V37 ROW4LANE champion in the same run.

The PowerShell runner prints an explicit final `SUCCESS +X%` or `FAIL TO ADVANCE` for HOT and ROTATE.

## ISA contract

```
-march=ivybridge
-mssse3
-mavx
-mno-avx2
-mno-fma
```

The generated assembly was checked: the V38 ROW4MADD hot function contains `VPSHUFB` + `VPMADDWD` for scale handling and does not use `VPMULLD` for that stage.

Development synthetic sanity is only a gate. The pinned real laptop run and then E5-2680 v2 measurements decide promotion/production.
