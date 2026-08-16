# 09 — Bitplane Math and Kernel Contract

This document defines the arithmetic that was actually verified on the host and the contract a future FPGA/ASIC implementation must preserve.

## 1. Signed INT4 representation

The proven host kernel uses four-bit two's-complement signed integers.

For one weight `q` with bits `b0..b3`:

```text
q = b0 + 2*b1 + 4*b2 - 8*b3
```

where each `bi` is either 0 or 1.

Therefore a vector of weights can be stored as four bitplanes:

```text
W0 = all b0 bits
W1 = all b1 bits
W2 = all b2 bits
W3 = all sign bits
```

For N weights:

```text
packed Q4 storage = 4N bits
4 × one-bit planes = 4N bits
```

There is no logical storage expansion.

## 2. Signed INT8 representation

For one signed INT8 activation `x` with bits `a0..a7`:

```text
x = a0 + 2*a1 + 4*a2 + 8*a3
  + 16*a4 + 32*a5 + 64*a6 - 128*a7
```

Define activation bitplanes `A0..A7` in the same way.

## 3. Exact dot-product decomposition

For a block of elements, define:

```text
P[i][j] = popcount(W_i & A_j)
```

Weight coefficients:

```text
CW = [1, 2, 4, -8]
```

Activation coefficients:

```text
CA = [1, 2, 4, 8, 16, 32, 64, -128]
```

Then:

```text
dot(W, A) = sum(i=0..3) sum(j=0..7) CW[i] * CA[j] * P[i][j]
```

This is exactly equal to the scalar signed-integer dot product as long as both representations use the same two's-complement values.

The host experiments verified equality with:

```text
maxdiff = 0
```

## 4. What the datapath needs

A direct bitplane engine can therefore be built from:

1. bitwise intersection (`AND`);
2. population count / bit reduction;
3. fixed shifts or hard-wired coefficients;
4. signed accumulation;
5. optional scaling after the integer accumulator.

No general-purpose multiplier is required for the bitplane intersection itself.

On an FPGA the coefficients are constants, so the `×2`, `×4`, `×8`, etc. operations are wiring/shifts and sign changes rather than arbitrary multipliers.

## 5. CPU V3 realization

The proven x64 assembly kernel uses scalar 64-bit operations:

```text
load 64-bit weight plane word
AND with activation plane word
POPCNT
accumulate with fixed signed coefficient
```

The V1 code intentionally does not depend on AVX-512. The important result is not that scalar `POPCNT` is the final hardware choice; it is that the mathematical decomposition survives a native implementation and produces exact results.

## 6. V4 masked-activation-sum formulation

An alternative exact expression is:

```text
S0 = sum(x[k] where W0[k] = 1)
S1 = sum(x[k] where W1[k] = 1)
S2 = sum(x[k] where W2[k] = 1)
S3 = sum(x[k] where W3[k] = 1)

dot = S0 + 2*S1 + 4*S2 - 8*S3
```

This removes activation bitplane decomposition from the conceptual formula and treats each weight bit as a gate over the original activation values.

On the tested CPU this realization was slower than V3. It remains interesting for hardware in which the mask/gate operation can be implemented spatially rather than synthesized as a sequence of CPU instructions.

## 7. Matrix multiplication from dot products

For a weight matrix `W[M,K]` and activation vector `x[K]`, each output row is an independent dot product:

```text
y[m] = dot(W[m,:], x)
```

The bitplane storage can therefore be row-major, blocked, expert-major or otherwise tiled according to memory-channel layout.

A useful physical layout for one tile is:

```text
channel 0 -> rows/block subset 0
channel 1 -> rows/block subset 1
...
channel 7 -> rows/block subset 7
```

or, for MoE experts:

```text
channel 0 -> expert shards A
channel 1 -> expert shards B
...
```

The layout should be chosen to maximize sequential DDR bursts and minimize cross-channel reduction traffic.

## 8. Local reduction

A Transit tile should not return all intermediate popcounts to the R920.

It should return the most reduced mathematically safe result possible.

Preferred order:

```text
DDR3 bitplanes
   -> local AND/popcount lanes
   -> local signed integer accumulation
   -> local scale/dequant step if needed
   -> local row/expert output
   -> only then PCIe result DMA
```

This is the reason a relatively narrow PCIe uplink can coexist with much larger local DDR bandwidth.

## 9. K3 caveat: MXFP4/MXFP8 is not plain INT4/INT8

The current proof kernel is **not an exact implementation of K3's published numerical format**.

K3 routed experts are described with MXFP4 weights and MXFP8 activations. Those formats include floating-point-style encodings and block scaling behavior that differ from simple signed two's-complement INT4/INT8.

Therefore the K3 kernel needs one of two paths.

### Path A — native MXFP decode

Keep the checkpoint representation close to MXFP4/MXFP8 and implement:

- unpack/decode of the low-bit element representation;
- block-scale fetch;
- local integer/significand accumulation where possible;
- scale application at the correct granularity;
- reference validation against a known-good K3 implementation.

This is the preferred path if it maps efficiently into FPGA LUTs/DSPs.

### Path B — Transit internal quantization

Offline, convert selected K3 tensors to a Transit-native signed INT4/INT8 format with stored scales.

Advantages:

- directly reuses the exact bitplane engine already demonstrated;
- simple hardware.

Risks:

- numerical loss may be larger than native MXFP;
- calibration/quantization policy becomes part of model quality;
- K3 accuracy must be measured, not assumed.

The repository must keep these two paths distinct.

## 10. Candidate FPGA microarchitecture

A first FPGA engine does not need to be elegant. It needs to be measurable and correct.

Conceptual pipeline:

```text
PCIe command
     |
activation DMA -> activation buffer
     |
transpose/encode activation planes
     |
DDR3 burst reader -> W0/W1/W2/W3 FIFOs
     |
N parallel 64-bit AND + popcount lanes
     |
coefficient accumulator
     |
row/expert accumulator RAM
     |
scale/format stage
     |
result DMA -> host
```

The first hardware version should expose counters for:

- DDR bytes read;
- useful weight elements processed;
- PCIe bytes in/out;
- cycles stalled on DDR;
- cycles stalled on activation input;
- cycles stalled on result output;
- compute-lane active cycles;
- total command latency.

Without those counters it will be too easy to confuse one bottleneck for another.

## 11. Correctness contract for a tile

For every development stage the tile must be tested against a software reference using the same exact input bytes and weight bytes.

Required checks:

```text
1. bitplane unpack/reference reconstruction
2. one 64-element dot product
3. one row
4. one small matrix-vector multiply
5. one real model tensor slice
6. one expert shard
7. one complete expert
```

For the signed INT4×INT8 proof format, the acceptance condition is exact integer equality:

```text
max_abs_diff == 0
```

For MXFP or scaled formats, define an exact reference for the encoded arithmetic first, then separately evaluate model-level numerical quality.

## 12. What not to optimize prematurely

Do not repeat the host benchmark spiral by introducing many new CPU versions that do not change the physical architecture.

The next optimization questions should be hardware questions:

- can local DDR3 feed the lanes continuously?
- how many popcount lanes fit at the target FPGA clock?
- does the activation transpose become expensive?
- can block scales be fused without stalling?
- how much result reduction can happen locally?
- can the design exploit DRAM row operations later?

The host proof has already served its purpose: the arithmetic works. The next proof must happen where Transit intends to run it.
