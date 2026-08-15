# ADR 0002 — Real-weight Atlas + deterministic raw tile pack

Status: **proposed / implemented for Phase 2; awaiting hardware measurement**

## Context

Phase 1 demonstrates the fixed two-slot VRAM ring with synthetic FP16 weight tiles. The next proof must remove the synthetic data assumption without coupling TensorWave to an unverified or unreleased model graph.

Large safetensors checkpoints already expose exactly the metadata TensorWave needs for the storage layer:

- tensor identity;
- dtype;
- shape;
- byte range inside a shard.

For rank-2 F16/BF16 tensors, a full row slice is contiguous and can be transferred as exact checkpoint bytes.

## Decision

Phase 2 introduces two explicit persistent artifacts:

### `weight-atlas.json`

A header-only inventory of the checkpoint. Tensor payloads are not decoded or loaded by the atlas builder.

### `execution-plan.json` + `weights.pack`

For the Phase-2 storage proof, a deterministic homogeneous family of rank-2 tensors is selected:

- one dtype (`F16` or `BF16`);
- one second dimension `K`;
- fixed `N` rows per tile.

Each complete row tile is copied byte-for-byte into `weights.pack` and receives a plan entry containing source tensor, shard, row range, source offset, pack offset, byte count and SHA-256.

The CUDA runtime loads the pack into pinned host RAM and treats the two VRAM weight buffers as a ring.

## Why not parse values in Python

Parsing/reserializing would add several confounders:

- Python memory amplification;
- accidental dtype conversion;
- transpose/copy overhead;
- uncertainty about whether benchmark bytes match checkpoint bytes.

Raw byte slicing keeps Phase 2 focused on the memory-transport question.

## Why storage order is acceptable in Phase 2

The Phase-2 claim is deliberately narrow: real checkpoint bytes can be indexed and streamed through the fixed-VRAM execution mechanism.

Storage order is **not** claimed to equal the model's inference order.

A later graph-integration phase will replace the storage-order plan with operation-derived dependencies.

## Correctness contract

Every streamed tile contributes to one FP32 accumulator:

```text
Y <- X * W_tile + Y
```

The sequential and overlapped paths process the same plan and compute order. Their final full accumulators must agree within the experiment tolerance and contain no non-finite values.

## Consequences

Positive:

- works before H3-specific code exists;
- exact traceability from source shard to GPU input bytes;
- no third-party Python dependency for atlas/packing;
- pack can be inspected independently by all contributors;
- dtypes/layout are explicit rather than inferred in CUDA code.

Negative / limits:

- only F16/BF16 rank-2 tensors are executable in Phase 2;
- pack duplicates selected checkpoint bytes on disk temporarily;
- entire selected pack is pinned in host RAM for the current proof;
- true graph order and quantized dequant-on-GPU are not addressed yet.

## Follow-up decisions

Expected later ADRs:

- graph-derived execution plan;
- bounded pinned-RAM ring instead of pinning the entire selected pack;
- quantized tile representation + fused GPU dequantization;
- hot/warm/cold cache policy.
