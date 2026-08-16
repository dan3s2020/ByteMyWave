# 10 — Kimi K3 Target

Kimi K3 is the current stress-test model for Transit GPU. This document records the **working model numbers used in the project** and the conclusions drawn from them. Before a production implementation, the exact checkpoint/configuration used must be re-parsed and these values regenerated automatically rather than copied by hand.

## 1. Working model facts used by the project

The current design calculations have used approximately:

```text
total parameters             ~2.8 trillion
active parameters / token    ~104 billion
layers                       93
experts                      896
selected experts / token     16
checkpoint size              ~1.56 TB
hidden dimension             7168
expert latent-ish dimension  3584
expert hidden dimension      3072
```

A routed expert has been approximated at about:

```text
~33.03 million weights
~16.5 MB at 4 bits/weight
```

The working precision assumption discussed in the project is:

- routed expert weights: MXFP4;
- activations: MXFP8;
- other model components may use different precision.

These values are sufficient for architecture sizing but must not substitute for a real tensor inventory generated from the checkpoint.

## 2. Why MoE changes the physical architecture

K3 is not a dense 2.8T model that reads every parameter for every token.

The router selects a small fraction of experts. That means Transit should physically exploit sparsity rather than merely celebrate it in software.

Desired behavior:

```text
router selects experts
      |
      +--> only tiles holding selected expert shards wake/read/compute
      |
      +--> inactive expert tiles remain idle
```

This creates three important placement opportunities:

1. **expert locality** — keep a complete expert or large expert shards within one tile whenever possible;
2. **replication** — replicate statistically hot experts to reduce contention;
3. **routing-aware scheduling** — route a token to the copy/tile with available bandwidth.

## 3. Active-weight traffic roofline

Using the working approximation:

```text
104e9 active weights/token
× 0.5 byte/weight at Q4-equivalent packing
= ~52 GB/token
```

At 100 tokens/s:

```text
52 GB/token × 100 token/s = ~5.2 TB/s
```

This number is deliberately called a **weight-path roofline**, not an end-to-end K3 performance prediction.

It answers one question only:

> If every active weight had to be fetched once from external memory for every token, how much payload bandwidth would 100 tok/s imply at 4 bits/weight?

It does not include:

- attention traffic;
- KV cache;
- router compute/metadata;
- scale metadata;
- activation movement;
- synchronization;
- PCIe protocol overhead;
- expert-output aggregation;
- non-expert weights;
- batch effects.

## 4. Why 304 DDR3 channels appeared

The target `38 tiles × 8 channels = 304 channels` came from matching cheap DDR3 aggregate bandwidth to the active-weight roofline.

Nominal x64 payload rates:

```text
DDR3-1600  12.8 GB/s/channel
DDR3-1866  14.93 GB/s/channel
DDR3-2133  17.07 GB/s/channel
```

For 304 channels:

```text
DDR3-1600  ~3.89 TB/s
DDR3-1866  ~4.54 TB/s
DDR3-2133  ~5.19 TB/s
```

At the purely ideal Q4 streaming roofline:

```text
DDR3-1600  ~74.8 weight-path tok/s
DDR3-1866  ~87.3 weight-path tok/s
DDR3-2133  ~99.8 weight-path tok/s
```

This coincidence is useful for sizing, but it is **not a promise that 304 DDR3-2133 channels produce 100 K3 tokens/s**.

Real efficiency will be lower unless the architecture avoids unnecessary reads and keeps the channels busy.

## 5. The real reason Transit may beat the naive roofline

The project should not accept `304 channels` as an immutable requirement. It is a baseline if every active Q4-equivalent weight is streamed once per token.

Transit tries to reduce the required effective external movement using:

- MoE routing: inactive experts do not read weights;
- expert-local compute: weights never cross PCIe;
- placement: keep expert shards contiguous and sequential;
- hot-expert replication;
- local caching of repeatedly used blocks where statistically useful;
- activation broadcast rather than weight movement;
- local partial-sum/expert reduction;
- potential future in-DRAM operations;
- batching if latency requirements allow reuse across tokens.

The purpose of the 304-channel design is therefore:

> provide enough cheap physical memory parallelism that the architecture is not doomed even before optimization.

## 6. Capacity at 38 × 8 channels

If each logical DDR3 channel is populated with one DIMM:

```text
8 GB/channel  × 304 = 2.432 TB
16 GB/channel × 304 = 4.864 TB
32 GB/channel × 304 = 9.728 TB
```

Even 8 GB/channel would exceed the ~1.56 TB checkpoint working size used in the project and leave space for metadata/replication depending on exact checkpoint representation.

16 GB/channel is particularly attractive conceptually because it provides enough excess capacity to replicate hot experts rather than place exactly one copy of everything.

This is a capacity argument only. DIMM density does not multiply bandwidth if multiple DIMMs share one channel.

## 7. Expert placement sketch

A first placement strategy should be boring and deterministic.

Example:

```text
Tile 00
  DDR ch0 -> layer 0 expert shard set A
  DDR ch1 -> layer 0 expert shard set B
  ...
  DDR ch7 -> layer 0/1 expert shards

Tile 01
  next expert range
...
```

Better later:

```text
Weight Atlas
  |
  +-- placement optimizer
        |
        +-- keep selected expert's shards within minimal tile count
        +-- replicate hot experts
        +-- distribute bandwidth pressure
        +-- preserve sequential bursts
```

The placement optimizer should consume measured router statistics from real K3 prompts rather than assume uniform expert usage.

## 8. Communication is smaller than weight traffic

One reason the tile architecture is plausible is that activation/result vectors are much smaller than the stored expert matrices.

Using hidden size 7168 as an order-of-magnitude example:

```text
7168 values × 1 byte  ≈ 7 KiB   for an 8-bit activation vector
7168 values × 2 bytes ≈ 14 KiB  for a 16-bit result vector
```

Even when copied to multiple selected experts, this is orders of magnitude smaller than reading tens of gigabytes of weights/token.

Exact communication depends on:

- where routing happens;
- how experts are sharded;
- whether one tile contains multiple selected experts;
- whether expert weighting/reduction happens on the tile;
- the actual activation dtype;
- attention/non-expert paths.

Therefore PCIe x1-class links may be sufficient for some tile topologies even if each tile has >100 GB/s of local DDR bandwidth. That claim must be checked with the real command/activation/result protocol rather than assumed from lane count alone.

## 9. Attention and non-expert work

The 304-channel calculation is dominated by the routed-weight path. A complete K3 runtime must also decide where to execute:

- attention projections;
- attention score/value operations;
- router;
- normalization;
- embeddings/output head;
- KV cache management;
- any shared experts or non-routed MLP paths;
- scale conversion and residual operations.

Possible first architecture:

```text
R920 / optional GPU
  -> router + control + attention/shared work
Transit tiles
  -> routed expert heavy lifting
```

This is an engineering starting point, not a final partition. The partition should follow profiling.

## 10. First real K3 milestones

The project should not jump directly from a synthetic INT4 tensor to full K3.

Order:

1. parse the K3 checkpoint and build an exact tensor/expert atlas;
2. identify one routed expert tensor and its exact MXFP metadata;
3. reproduce that expert numerically in software;
4. implement one equivalent Transit tile kernel;
5. run one expert on one physical tile;
6. run the same expert through PCIe command/response;
7. run a complete routed-expert stage for one layer;
8. integrate router selection;
9. run one complete K3 layer;
10. only then attempt an end-to-end token.

## 11. Performance language rule

When reporting Transit results, use distinct terms:

- **GB/s DDR payload** — physical/local memory traffic;
- **Gweights/s** — weight elements processed by the kernel;
- **weight-path tok/s equivalent** — `Gweights/s / active_weights_per_token`;
- **end-to-end tok/s** — measured complete model generation only.

Do not present a weight-path equivalent as end-to-end K3 tok/s.
