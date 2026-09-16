# TRANSIT V34 PREPFUSE

V34 is the next exact Q4_K evolution after the real-laptop V33 PAIRFUSE result.

## What V33 proved on the pinned real Qwen tensor

On the i5-12500H P-core-only sweep, V33_PAIR4_VR passed the strict stable gate in both HOT and ROTATE modes:
- 1T HOT: 1.22788x, p10 1.09007, win 100%, diff=0
- 1T ROTATE: 1.22865x, p10 1.1088, win 96.8%, diff=0
- 4T HOT: 1.23077x, p10 1.03112, win 90.5%, diff=0
- 4T ROTATE: 1.17731x, p10 1.10063, win 93.5%, diff=0

V31_FSA_RB16 remains a strong HOT control, especially at 2 P-cores.

## V34 mechanism 1: APM (activation-precomputed min correction)

The Q8_K `qsum` for each 32-weight activation group depends only on the activation block. V33 recomputed the two group sums inside every output row-group.

V34 builds 4 small coefficient vectors per activation block once:
`[qsum(g0), qsum(g1)] x 4 rows`

The min correction then uses one packed `PMADDWD` per low/high group pair across four rows instead of two independent 32-bit multiply chains. This is exact: qsum fits signed int16 and the products/sums fit int32.

Candidate: `V34_PAIR4_APM`.

## V34 mechanism 2: APM + scale-broadcast metadata (space for time)

V33 loads an 8-bit scale and the compiler must expand it to all int16 lanes.

V34 optionally stores each scale offline as a 32-bit bit pattern:
`scale | (scale << 16)`

AVX1 `VBROADCASTSS` then loads that 32-bit pattern from memory and replicates it to the XMM register in one memory-broadcast operation. This costs more metadata RAM but removes scale-broadcast instruction sequences from the hot loop.

Candidate: `V34_PAIR4_APM_SB`.

For the 9216x2560 tensor, approximate static representation bytes in the harness:
- META: 13,639,680
- X8F: 14,008,320
- PAIRSCALE: 16,220,160

This is intentionally a small space-for-time trade compared with EXP8/SCALED16.

## Methodology

- pinned prior model blob only
- pinned tensor `blk.0.ffn_gate.weight`
- strict Ivy Bridge build: SSSE3/SSE4.1 + AVX, no AVX2/FMA/VNNI
- exactness gate: diff=0
- P-core-only 1/2/4 on the 12500H
- paired alternating AB/BA against V27
- HOT and rotating address-distinct beyond-LLC claims kept separate
- stable promotion requires median>1, p10>1, win_rate>=0.80, diff=0
- V27 is never removed
- physical E5-2680 v2 remains production authority
