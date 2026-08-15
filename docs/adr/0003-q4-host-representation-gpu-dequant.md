# ADR 0003 — Keep streamed weights quantized through H2D

Status: **experimental / implemented for Phase 3; measurement pending**

## Context

The fixed-VRAM ring can hide some host-to-device latency, but a 16-bit tile still requires every F16/BF16 byte to cross PCIe.

TensorWave's larger goal requires reducing bytes transferred, not merely scheduling the same bytes more carefully.

If a checkpoint is already quantized in host RAM, decompressing it on CPU before H2D would destroy most of the bandwidth advantage.

## Decision

Phase 3 stores and transfers a Q4 representation and performs dequantization on GPU.

The first format is deliberately simple:

```text
Q4_SYM_G32_F32S
32 weights/group
float32 scale
16 packed int4 bytes
20 bytes/group
```

The 16-bit source requires 64 bytes for the same 32 weights, so Q4 v1 is 3.2x smaller including scale overhead.

## VRAM contract

Weight-related residency is:

```text
Q4 slot A
Q4 slot B
one reusable FP16 dequantized tile
```

The full model and full decompressed model remain outside VRAM.

The copy stream fills compressed slot `i % 2` while the compute stream dequantizes and consumes the previous tile.

## Why dequantize to a full tile in Phase 3

This is an intermediate proof.

Using a separate GPU dequant kernel and cuBLAS allows us to isolate and measure:

- compressed H2D;
- dequant cost;
- GEMM cost;
- starvation;
- scheduling correctness.

A fused Q4 GEMM would be faster and use less VRAM, but implementing it first would make failures harder to attribute.

## Quantization quality vs scheduling correctness

These are separate questions.

`q4-plan.json` measures weight reconstruction error relative to the original F16/BF16 source.

The CUDA experiment compares sequential-Q4 output to overlapped-Q4 output. That detects stream/ring corruption without conflating it with unavoidable Q4 approximation error.

## Alternatives considered

### CPU dequant before H2D

Rejected for the target architecture because it would transfer expanded 16-bit bytes and lose the main bandwidth reduction.

### Float16 scale

Deferred. It would reduce overhead to 18 bytes/group, but float32 scale makes the first CUDA format easier to inspect and decode while preserving 4-byte alignment of every group record.

### Q8 first

Rejected as the primary Phase-3 proof because the central question is whether substantial compression changes H2D starvation. Q4 creates a larger, clearer effect.

### Fused dequant-GEMM immediately

Deferred to Phase 4. Separate kernels provide better instrumentation and lower implementation risk for the first proof.

## Consequences

Positive:

- physical weight H2D bytes fall to 31.25% of the 16-bit source representation;
- quantized host representation remains compact until the GPU actually needs the tile;
- only one full FP16 weight tile is resident;
- dequant overhead becomes directly measurable;
- next step to fused execution is explicit.

Negative:

- Q4 conversion is lossy;
- v1 scale overhead means 5 effective bits/weight;
- current conversion uses NumPy offline;
- separate dequant writes one full FP16 tile to VRAM before GEMM;
- a slow dequant kernel could make transfer look easy to hide for the wrong reason, so `dequant_ms` and `gemm_ms` must always be reported separately.

## Evidence required before acceptance

On target hardware, for several M values and a real checkpoint family:

```text
correctness_ok
compressed H2D GB/s
dequant ms
GEMM ms
starvation %
hidden transfer %
wall-time delta vs Phase 2
fixed VRAM bytes
```

The Q4 approach is useful only if end-to-end behavior improves or enables a working set that would otherwise be impractical; compression ratio alone is not sufficient evidence.
