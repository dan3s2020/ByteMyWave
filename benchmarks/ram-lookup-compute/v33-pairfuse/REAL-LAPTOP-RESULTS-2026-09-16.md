# V33 PAIRFUSE — real pinned laptop results — 2026-09-16

Hardware: Intel Core i5-12500H, promotion-grade sweep restricted to the four P-cores selected by CPUID leaf 0x1A. Build contract: `-march=ivybridge -mssse3 -mavx -mno-avx2 -mno-fma`.

Input is the same pinned prior ByteMyWave GGUF blob and `blk.0.ffn_gate.weight`, geometry 9216 x 2560, exact Q4_K. Every promoted path returned `diff=0`, cosine=1.

## Stable gate

Promotion requires all of:
- paired AB/BA median speedup > 1.0 versus V27;
- p10 > 1.0;
- win-rate >= 0.80;
- diff=0.

## Key measured results

### HOT
- 1T `V33_PAIR4_VR`: 1.22788x median, p10 1.09007, p90 1.31925, win 1.0.
- 2T `V31_FSA_RB16`: 1.25726x median, p10 1.20361, p90 1.34422, win 1.0. This remains the strongest HOT control.
- 2T `V33_PAIR8_VR`: 1.15976x median, p10 1.11204, p90 1.20782, win 0.952381.
- 4T `V33_PAIR4_VR`: 1.23077x median, p10 1.03112, p90 1.39352, win 0.904762.

### ROTATING / beyond-LLC
The harness rotated 16 address-distinct META/X8F copies, ~421.875 MiB total, with no eviction sweep in the timed region.

- 1T `V33_PAIR4_VR`: 1.22865x median, p10 1.1088, p90 1.34675, win 0.967742.
- 2T `V33_PAIR8_VR`: 1.17368x median, p10 1.11721, p90 1.30237, win 0.967742.
- 4T `V33_PAIR4_VR`: 1.17731x median, p10 1.10063, p90 1.27131, win 0.935484.
- 4T `V33_FSA4_VR`: 1.17396x median, p10 1.08429, p90 1.21261, win 0.935484.

## Decision

`PAIR4_VR` is promoted as a real exact mechanism on the laptop selector harness. It is not yet a production/server claim. `V27_META` remains the mandatory fallback and `V31_FSA_RB16` remains an important HOT control.

The evidence supports the mechanism: low/high Q4_K groups share the same 32 packed bytes; loading once and extracting both nibbles removes redundant Q4 loads. Four-row vectorized min correction/reduction is a good balance between reuse and register pressure. `PAIR8` is useful in some 2T cases but is less robust at 4T.

## Direction after V33

The next exact target is to remove work that is invariant across output rows:
- precompute activation-side group qsum coefficients once per Q8_K block;
- replace separate 32-bit min multiply chains with packed `PMADDWD` across a low/high group pair;
- separately test a small RAM-for-instructions representation that pre-expands scale broadcast metadata.

Do not reinterpret these kernel/subsystem numbers as end-to-end model tok/s. Physical E5-2680 v2 NUMA results remain the production gate.
