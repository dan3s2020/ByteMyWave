# 01 — What Is Actually in Model Memory?

The central clarification from the discussion is that a model checkpoint is not stored as sentences, images, rules, or symbolic concepts. It is mostly **large numerical tensors** plus enough metadata to interpret them.

## Persistent model data

For a Transformer/diffusion model such as MiniMax H3, persistent parameters include categories such as:

- attention projection weights (`Wq`, `Wk`, `Wv`, `Wo`);
- FFN/MLP projection matrices;
- normalization parameters;
- embedding and projection weights;
- conditioning / AdaLN parameters;
- text/vision encoder parameters;
- video/audio VAE parameters;
- biases, scales, quantization metadata and tensor layout metadata.

Conceptually, a tensor may be viewed as:

```text
layer_17.attention.q_proj.weight
shape = [rows, columns]

[
  0.00431, -0.0182,  0.00077, ...
 -0.00214,  0.0311, -0.00921, ...
 ...
]
```

Those numerical values are the learned parameters.

## Quantized representation

The values do not need to remain BF16/FP16 in host memory. A quantized block can conceptually contain:

```text
scale = 0.0137
codes = [-3, +1, +7, 0, -2, +4, ...]
```

with approximate reconstruction such as:

```text
weight ~= code * scale
```

The project therefore assumes that host RAM should hold a **compact execution representation**, not necessarily the original training checkpoint format.

## Temporary execution memory

Separate from persistent weights are values created during inference:

- activations;
- latent video/audio tensors;
- Q/K/V intermediate activations;
- attention workspace;
- GEMM workspace;
- conditioning embeddings;
- residuals;
- temporary dequantized fragments.

These values have different lifetimes. Many can be discarded or reused once the downstream operation no longer needs them.

This distinction is crucial:

```text
persistent model state != temporary working state
```

The 4 GB VRAM target should primarily hold the **currently required working state**, not the complete persistent model.

## Why tensors can be split

A large matrix multiplication can often be decomposed into smaller mathematically equivalent operations.

Conceptually:

```text
Y = X * W
```

If `W` is partitioned into compatible tiles:

```text
W = [W0 W1 W2 W3 ...]
```

then the output can be accumulated or concatenated from operations using those tiles without requiring the complete `W` in VRAM at once.

Therefore the important question is not "does the whole matrix fit?" but:

> What is the smallest mathematically useful tile that can be transferred, dequantized and consumed efficiently without destroying GEMM efficiency?

That becomes an experimental design variable for ByteMyWave.

## Read-only property during inference

A major simplifying property is that most model weights are **read-only during inference**.

They can therefore remain in host memory, be referenced repeatedly, streamed on demand, cached selectively, and discarded from VRAM without synchronization required for weight updates.

## Practical hierarchy envisioned

```text
SSD / checkpoint storage
        |
        v
Large host RAM
(compact persistent tensors)
        |
        v
Pinned / DMA-visible host window
        |
        v
Small VRAM slots
(compressed tiles + active state)
        |
        v
Shared memory / registers
(tiny dequantized fragments)
        |
        v
GPU matrix operations
```

The project will treat this hierarchy as explicit architecture rather than relying entirely on generic framework memory management.
