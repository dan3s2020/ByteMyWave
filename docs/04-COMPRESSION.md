# 04 — Compression / Quantization Placement

A key question in the discussion was: if the GPU is already computing while the next data is loading, **when and where should compression happen?**

The proposed answer is that most compression/quantization should happen **before inference**, not dynamically on the critical path.

## Offline conversion

Start with the original checkpoint representation and convert it once into the execution format required by TensorWave.

Conceptually:

```text
Original BF16 / FP16 checkpoint
          |
          | one-time conversion
          v
TensorWave packed representation
(Q8/Q6/Q5/Q4/Q3/Q2, metadata, atlas, execution index)
```

The compact representation is what lives on disk and is loaded into large host RAM.

## Do not expand before PCIe

Bad path:

```text
RAM Q4
  |
CPU dequantizes
  |
large FP16 tensor
  |
PCIe transfer
  v
GPU
```

This increases host CPU work and sends more bytes through the narrowest link.

Desired path:

```text
RAM Q4
  |
compressed PCIe transfer
  v
GPU compressed tile
  |
GPU dequantization while consuming fragments
  v
GEMM / model operation
```

## Fused or near-fused dequantization

The ideal kernel should not create a complete expanded FP16 copy of every large weight tile.

Conceptually:

```text
compressed weight block
       |
       v
dequantize tiny fragment
       |
       v
register/shared-memory fragment
       |
       v
matrix multiply / accumulation
       |
       v
fragment discarded
```

The project should test existing quantized GEMM paths before writing custom kernels.

## Mixed precision / mixed quantization

Not all model tensors need the same quantization.

A future atlas may classify sensitivity:

```text
high sensitivity       -> Q8 / Q6
normal                  -> Q4
highly tolerant         -> Q3 / Q2
small but important     -> FP16/BF16 if needed
```

Possible example:

```text
Block 00 QKV      Q4
Block 00 FFN      Q3
Block 01 QKV      Q4
Block 01 FFN      Q2
...
Block 34 QKV      Q8  # if measured sensitive
```

The point is not to maximize compression at any cost. The point is to minimize transferred bytes while preserving useful model quality and GPU kernel efficiency.

## Compression vs structural representation

The conversation also distinguished ordinary quantization from more ambitious representation changes.

### Ordinary quantization

Store independent weight values using fewer bits plus scales/metadata.

### Shared codebooks / vector quantization

Multiple values point into a shared set of representative values.

### Low-rank / tensor decomposition

Represent a large matrix using smaller factor matrices where approximation error is acceptable.

### Shared-basis representation

Multiple tensors/tiles may share a common basis plus small per-tile coefficients.

Conceptual form:

```text
W1 ~= B * C1
W2 ~= B * C2
W3 ~= B * C3
```

This was discussed as one interpretation of "data entanglement": independent tensors may not need entirely independent storage if useful common structure can be extracted.

## Recovery coding is separate from compression

Parity / erasure coding should not be confused with model compression.

Compression reduces storage/traffic.

Recovery coding adds controlled redundancy to allow exact restoration after missing or corrupt chunks.

Both can coexist in the Weight Atlas, but they solve different problems.

## What must be benchmarked

For each candidate format, measure:

- compressed bytes per tile;
- effective host-to-device GB/s;
- GPU dequantization cost;
- resulting GEMM throughput;
- total exposed transfer time;
- output quality loss;
- VRAM working-set requirement.

A smaller file format is not automatically better if its dequantization path makes the GPU significantly slower.
