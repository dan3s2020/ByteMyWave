# 00 — Project Intent

## Origin

The project starts from a practical hardware constraint: **large AI models are limited by VRAM cost**, while ordinary system RAM can be purchased in much larger capacities for far less money.

The working question is therefore not:

> How do we buy enough VRAM?

It is:

> How do we restructure inference so that a GPU with very little VRAM can continuously compute on a model that physically lives in much larger system RAM?

## Initial hardware target

The concrete target discussed is deliberately difficult:

- GPU with about **4 GB VRAM**;
- large system RAM pool (64 / 128 / 256 GB or more);
- potentially inexpensive older DDR3/DDR4 ECC server RAM;
- PCIe link as the bridge between host RAM and GPU VRAM;
- MiniMax H3 as a demanding first target workload.

The larger motivation is to make a useful AI rig inside a low hardware budget instead of spending the budget on several mediocre low-VRAM GPUs.

## User-originated hypotheses to preserve

1. **The model can live in RAM.** VRAM should not be treated as the mandatory permanent home of all model weights.
2. **The GPU only needs the data required for the next computation.** If execution order is known, the next data can be loaded before the current calculation finishes.
3. **VRAM can behave like a working cache/window.** It does not need to represent total model size.
4. **The execution sequence is predictable.** For a known model graph, the runtime can know what tensor/tile will be needed next.
5. **Indexing should be precomputed.** Runtime should not spend time searching for weights. A prebuilt execution plan should map each future operation to fixed RAM offsets and fixed VRAM slots.
6. **Compression must not create a new bottleneck.** The model should already be stored in a compact/quantized form in RAM, transferred in that form, then dequantized on the GPU during compute.
7. **Transfer and compute should overlap.** While the GPU computes tile N, DMA should transfer tile N+1.
8. **The key metric is uncovered transfer / GPU starvation.** Raw transfer time is not the final problem if most of it is hidden under GPU compute.
9. **The model should be representable as a graph/map/atlas.** This should identify exactly what every piece is, where it lives, what depends on it, and what is missing if a piece is unavailable.
10. **Data representation itself may be exploitable.** Shared basis, quantization, parity/recovery, sparse structure, or other relationships between weights may reduce how much independent information must move.

## What ByteMyWave is not

ByteMyWave is not based on the claim that software can change the physical bandwidth of PCIe or turn DDR3 into HBM.

It instead investigates whether the total required traffic and exposed transfer latency can be reduced enough that those physical limitations stop dominating wall-clock inference time.

## Primary research question

Given a large model in host RAM and a 4 GB GPU, can a custom runtime maintain sufficiently high GPU utilization by combining:

- deterministic execution planning;
- quantized/compressed host representation;
- tiled tensor execution;
- pinned-memory transfer;
- asynchronous DMA;
- double/triple buffering;
- fixed VRAM slots;
- GPU-side dequantization;
- selective caching;
- computation-result caching where mathematically valid;
- prefetch several operations ahead?

## Success criterion

The first success criterion is **not** "the model technically runs".

The desired result is a system that remains useful/productive.

A critical metric proposed in the discussion:

```text
GPU starvation time = time the GPU is ready to compute but waits for required data
```

If starvation can be reduced to a small percentage of total execution time, then a small-VRAM GPU may be able to operate as the compute engine for a model whose persistent representation is far larger than VRAM.

## Current phase

This repository is currently in **documentation-only phase**. No implementation claim has yet been demonstrated here. The next phase will be experimental proof on a single model block/tensor path before attempting complete MiniMax H3 execution.
