# ADR — exact Q4_K x8meta runtime layout

Date: 2026-09-16

Status: **Accepted for llama.cpp integration experiment; not yet accepted as default runtime layout.**

Owner/scope: PR #16, x86 Q4_K×Q8_K CPU kernel research only.

## Context

The real-Qwen experiments tested several ways to reduce CPU work around Q4_K inference while preserving model quality.

Lossy PQ/codebook representations were rejected. Exact activation-dependent LUTs were numerically correct but approximately an order of magnitude slower than a native AVX2 baseline in the geometry-correct benchmark. Simple multi-row and AVX-VNNI variants also failed to beat the relevant baseline.

The remaining measured candidate is `x8meta`: a model-load-time, lossless runtime repack of eight Q4_K output rows.

## Decision

Carry `x8meta` forward to a real llama.cpp integration experiment.

Do **not** replace the on-disk GGUF format.

Do **not** overload the existing `block_q4_Kx8` structure because x8meta is larger and has different metadata semantics.

Use a distinct runtime layout approximately equivalent to:

```cpp
struct Q4KX8Meta {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t  scales[8][8];
    uint8_t  mins[8][8];
    uint8_t  qs[1024];
};
```

This is 1184 bytes versus 1152 bytes for eight ordinary 144-byte Q4_K blocks: `+2.7778%` runtime weight storage.

Preserve the original Q4 nibbles and FP16 `d/dmin` exactly. Decode the packed six-bit scale/min metadata once at repack time.

The AVX2 kernel processes eight output rows together and reuses Q8_K activation fragments across rows while keeping the standard Q4_K×Q8_K arithmetic semantics.

For Ivy Bridge, implement a separate SSSE3/AVX path; do not compile AVX2 for E5-2680 v2.

## Measured evidence

On the i5-12500H, real Qwen `blk.0.ffn_gate.weight`, correct `9216×2560` GEMV geometry, single-thread paired V12 comparison:

```text
-O3 -march=native -fno-unroll-loops

HOT
llama-style AVX2   1.313 ms median
x8meta             1.165 ms median
paired speedup     1.115×

64 MiB EVICTED
llama-style AVX2   1.304 ms median
x8meta             1.169 ms median
paired speedup     1.110×
```

No additional quantization is introduced.

## Alternatives considered

### Lossy PQ/codebook

Rejected because measured output error was too large.

### Exact nibble/byte LUT

Rejected because native AVX2 was roughly 10× faster in the geometry-correct test.

### Simple 4-row/8-row batching without the x8meta layout

Rejected because it was slower than the llama-style AVX2 baseline.

### AVX-VNNI replacement

Rejected on this CPU because the tested VNNI kernels were slower than the baseline. ISA choice alone did not solve the layout/hot-loop problem.

### Existing packed `block_q4_Kx8`

Kept as a separate control/llama-compatible layout. It does not contain fully predecoded scale/min arrays and is not equivalent to x8meta.

## Consequences

Positive:

- measured ~11% operator gain on the development laptop;
- exact original Q4_K representation semantics;
- activation-fragment reuse across eight rows;
- removes scale/min unpack from token hot loop.

Costs/risks:

- +2.78% runtime bytes on repacked Q4_K tensors;
- possible loss of advantage when many threads saturate DRAM bandwidth;
- compiler unrolling can create spills and erase much of the gain;
- current result is AVX2/Alder Lake, not Ivy Bridge;
- current result is an operator benchmark, not model tok/s.

## Verification gates before default adoption

1. compile inside current llama.cpp rather than standalone harness;
2. correctness A/B against the existing Q4_K path;
3. `llama-bench` and full decode on a real Q4_K model;
4. thread-count/DRAM bandwidth sweep;
5. generated-assembly inspection;
6. SSSE3/AVX implementation and real E5-2680 v2 measurement;
7. NUMA-local two-socket measurement.

If full-model or socket-scale performance does not improve, this ADR remains an accepted experiment but x8meta must not become the default representation.