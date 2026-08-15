# ADR 0001 — Phase-1 Fixed VRAM Ring Proof

**Status:** Proposed / under experimental verification

## Context

TensorWave's central hypothesis is that model capacity can live primarily in host RAM while a small-VRAM GPU is treated as a compute accelerator with a moving weight window.

Before modifying MiniMax H3 or designing a production Weight Atlas format, the project needs to know whether the fundamental scheduling mechanism works on the target hardware.

The specific mechanism is:

```text
compute(tile N)
        ||
H2D(tile N+1)
```

with a fixed small number of VRAM slots and a deterministic execution plan.

## Decision

The first implementation is a synthetic but real CUDA/cuBLAS benchmark with:

- FP16 host weight tiles;
- pinned host backing memory;
- two fixed VRAM weight slots;
- one copy stream;
- one compute stream;
- CUDA events for cross-stream dependencies;
- real cuBLAS GEMM for compute;
- sequential and overlapped executions of identical arithmetic;
- correctness comparison;
- explicit measurement of GPU starvation between GEMMs.

No H3-specific code is included in this phase.

## Why two fixed slots

Two slots are the minimum structure that permits:

```text
slot A: compute current tile
slot B: receive next tile
```

The slot addresses do not change during the run. The transfer stream is not allowed to overwrite a slot until the compute event from that slot's previous use has completed.

This isolates the desired mechanism from allocator overhead and avoids relying on implicit unified-memory paging.

## Why pinned backing memory first

Pinned RAM is used to isolate the H2D/DMA overlap question. Pageable RAM introduces an additional driver staging path that would make a negative result ambiguous.

A later ADR/experiment must decide whether production TensorWave uses:

- fully pinned model storage;
- `cudaHostRegister` windows;
- normal RAM plus a small pinned staging ring;
- or another host-memory strategy.

## Why cuBLAS GEMM

The project is not trying to prove that a synthetic sleep/kernel can overlap with DMA. It needs overlap against a workload structurally similar to the dominant dense matrix operations in neural inference.

A real FP16 cuBLAS GEMM therefore provides a better first falsification target while remaining independent of H3 graph details.

## Measured evidence required before acceptance

This ADR is not accepted until a real target machine records:

- GPU model;
- CUDA/driver state;
- measured H2D GB/s;
- sequential wall time;
- overlapped wall time;
- per-tile GEMM time;
- steady-state GPU starvation;
- estimated hidden-transfer percentage;
- correctness error;
- matrix/tile dimensions.

Strong support for one tested shape is defined as:

```text
correctness passes
steady_starvation_pct <= 10%
hidden transfer >= 80%
```

## Consequences

If the hypothesis is supported for realistic compute/transfer ratios, Phase 2 can replace synthetic weights with real H3 tensor tiles while preserving the same ring/scheduler contract.

If no reasonable shape can hide transfer on the target GPU, the project should reconsider the architecture before investing in H3-specific integration.
