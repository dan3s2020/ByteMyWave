# TensorWave — Why a Large LLM Works or Fails

Date: 2026-08-15

## Core result

For a dense autoregressive LLM, a tiny-VRAM streaming runtime is not primarily limited by whether the model *fits* in host RAM. It is limited by whether the GPU can do enough useful work on the current weight tile while the next compressed tile crosses PCIe.

The steady-state condition is approximately:

```text
T_compute(current tile) >= T_H2D(next tile)
```

If true, most transfer can be hidden.

If false, the GPU starves.

---

## Dense decode is the hardest case

For one generated token, a dense transformer uses almost all of its dense weights again:

```text
token t
  -> layer 0
  -> layer 1
  -> ...
  -> final layer
  -> token t+1
```

If most weights do not fit in VRAM, they must be streamed again for the next token.

For a model with `P` parameters and an effective wire representation of `b` bytes/parameter:

```text
bytes_per_step ~= P * b * (1 - resident_fraction)
```

For TensorWave Q4 v1:

```text
32 weights -> 20 bytes
b = 20 / 32 = 0.625 bytes/weight
```

This is **5 effective bits/weight**, including the float32 group scale.

At effective H2D bandwidth `BW`:

```text
T_transfer ~= bytes_per_step / BW
```

Example, assuming **15 GB/s effective H2D** and zero useful VRAM residency:

```text
13B Q4-v1: 13e9 * 0.625 = 8.125 GB -> ~0.542 s transfer floor
33B Q4-v1: 33e9 * 0.625 = 20.625 GB -> ~1.375 s transfer floor
70B Q4-v1: 70e9 * 0.625 = 43.75 GB -> ~2.917 s transfer floor
```

These are bandwidth-only floors. They exclude attention, KV traffic, dequantization, launch overhead and compute.

That is why “a 70B model can run on 4 GB” can be true while “a 70B model is productively interactive on 4 GB” is false.

---

## Why batching/prefill changes the result

Let `M` be the number of activation rows that reuse the same weight tile before it is discarded.

Examples:

```text
M = 1       single-sequence decode
M = 16      batch of 16 decode streams
M = 512     large batch/prefill-like matrix
M = 2048    long-prompt prefill-like matrix
```

For a dense linear operation, approximate compute per streamed parameter is:

```text
FLOPs ~= 2 * P * M
```

If effective GPU throughput is `F` FLOP/s:

```text
T_compute ~= 2 * P * M / F
```

The transfer time is:

```text
T_transfer ~= P * b * (1-r) / BW
```

where `r` is the fraction of weights already resident in VRAM/cache.

Set them equal to find the crossover:

```text
2 * P * M / F = P * b * (1-r) / BW
```

`P` cancels:

```text
M_crossover = b * (1-r) * F / (2 * BW)
```

This is a critical TensorWave insight:

> **for a dense GEMM-dominated model, the amount of reuse needed to hide streaming is largely independent of total model size.**

Model size still determines absolute latency and RAM capacity, but the transfer-vs-compute crossover is mainly controlled by:

```text
wire bytes / parameter
effective GPU compute
PCIe H2D bandwidth
VRAM-resident fraction
activation rows / batch / prefill length
```

---

## Counter-intuitive consequence: a faster GPU can starve more easily

Suppose PCIe bandwidth stays fixed.

A faster GPU finishes the current tile sooner, leaving less time to hide transfer of the next tile.

From:

```text
M_crossover = b * F / (2 * BW)
```

larger `F` means larger `M_crossover`.

This does **not** mean a slow GPU is faster overall. It means a slower GPU can have an easier time hiding PCIe because compute takes longer.

This is one reason a small consumer GPU is not automatically disqualified from TensorWave: its lower compute rate can improve overlap, even though absolute speed remains lower than a large GPU.

---

## Example crossover values

Assume:

```text
Q4-v1 b = 0.625 B/param
BW = 12 GB/s effective H2D
resident_fraction = 0
```

Then:

```text
F = 5 TFLOP/s  -> M_cross ~130
F = 10 TFLOP/s -> M_cross ~260
F = 20 TFLOP/s -> M_cross ~521
F = 30 TFLOP/s -> M_cross ~781
```

These are only roofline-style predictions. Real TensorWave measurements must replace `F` and `BW` with effective values from the actual kernel/tile geometry.

---

## Why batch=1 decode is difficult

At `M=1`:

```text
one activation row
x
large weight tile
```

The arithmetic intensity per transferred byte is very low.

The GPU can consume the tile faster than PCIe can feed the next tile:

```text
GPU:   COMPUTE ██      WAIT ███████
PCIe:           LOAD █████████
after load:     COMPUTE ██
```

The weight is used once, discarded, then needed again for the next token.

That is the worst case for TensorWave.

---

## Why prefill is favorable

For a prompt with many tokens, the same weight tile can be used across many activation rows:

```text
X[M,K] * W[K,N]
```

Instead of:

```text
X[1,K] * W[K,N]
```

The H2D bytes for the weight tile remain roughly constant while compute grows with `M`.

The timeline can become:

```text
GPU:   COMPUTE TILE N ███████████████
PCIe:       LOAD TILE N+1 █████
```

and transfer disappears under compute.

This is why TensorWave is expected to be much more promising for prefill than single-stream decode.

---

## Why batched serving is favorable

If one streamed tile serves a batch of 64 sequences:

```text
load W once
compute W against 64 activation rows
```

The transfer is amortized across the batch.

Aggregate throughput can become good even if per-sequence latency is not ideal.

This is the operating regime targeted by systems such as ZeRO-Inference/FlexGen and is likely one of TensorWave's strongest LLM use cases.

---

## Why MoE can be much better

A Mixture-of-Experts model can have a huge total parameter count while each token activates only a small subset of experts.

Let:

```text
P_total  = total parameters
A        = active parameter fraction per token/step
```

Then the streamed bytes can be closer to:

```text
P_total * A * b
```

rather than:

```text
P_total * b
```

If routing is known early enough, TensorWave can prefetch only active experts.

If expert reuse exists across nearby tokens/batch entries, a VRAM hot-expert cache can reduce bytes further.

For MoE, the important additional variables are:

```text
active parameter fraction
expert prediction lead time
expert cache hit rate
unique experts per batch
expert reuse across consecutive tokens
```

MoE is therefore a more promising target than a dense model of the same total parameter count.

---

## Residency/cache changes the crossover

If a fraction `r` of repeatedly used weights stays in VRAM:

```text
M_cross = b * (1-r) * F / (2 * BW)
```

Examples:

```text
r=0.00 -> stream 100% of active weights
r=0.25 -> stream 75%
r=0.50 -> stream 50%
```

A hot-weight or hot-expert cache can materially move the system from transfer-bound toward compute-bound.

This provides a concrete way to evaluate whether using 0.5–1.5 GB of the 4 GB VRAM as persistent cache is worth the activation/workspace tradeoff.

---

## Quantization changes the crossover

Smaller wire representation means less H2D traffic.

Ignoring quantization compute overhead:

```text
16-bit = 2.0 B/param
8-bit  ~= 1.0 B/param + metadata
Q4-v1  = 0.625 B/param including scale
```

Going from 16-bit to Q4-v1 reduces the transfer side by 3.2x.

However, the GPU must pay dequantization cost. Therefore the real condition is closer to:

```text
T_current_compute >= T_next_H2D
```

with:

```text
T_current_compute = T_dequant + T_GEMM + other kernels
```

If dequantization is fused into MMA and does not materialize a full FP16 tile, it can both reduce VRAM working set and avoid extra global-memory write/read traffic.

---

## The actual TensorWave feasibility question

For every tested geometry we want:

```text
T_H2D(tile)
T_dequant(tile)
T_GEMM(tile)
T_starvation(tile)
bytes_H2D(tile)
VRAM_working_set
```

Then classify:

```text
COMPUTE-BOUND:
    transfer is almost fully hidden

BALANCED:
    transfer is partially hidden; optimization may matter

TRANSFER-BOUND:
    GPU spends a material fraction waiting for the next tile
```

The project should never decide “works/doesn't work” from model size alone.

The decision must come from the measured operating point:

```text
(model architecture,
 active parameter bytes,
 quantization,
 M/batch/prefill,
 tile geometry,
 PCIe bandwidth,
 GPU effective compute,
 cache residency)
```

That multidimensional operating envelope is the TensorWave Feasibility Map.
