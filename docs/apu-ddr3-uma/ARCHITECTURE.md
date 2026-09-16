# Carrizo DDR3 UMA Architecture

## 1. Goal

Use the integrated Radeon GPU inside a Carrizo APU as the local compute engine for model weights stored in the same tile's DDR3 system memory.

CPU cores remain present because they are part of the APU/platform, but the target division of labor is:

```text
CPU cores
  -> boot Linux
  -> tokenizer / control
  -> Vulkan command submission
  -> routing/scheduling
  -> network transport
  -> telemetry/failure handling

Radeon R6 iGPU
  -> model-shaped vector/matrix arithmetic
  -> MXFP4 unpack/dequant where useful
  -> local reductions
  -> KDA/non-routed kernels where feasible

DDR3
  -> persistent local model-weight store
  -> activation/state buffers where appropriate
```

The design objective is **not** “remove the CPU silicon.” It is “remove CPU cores from the dominant matrix arithmetic and avoid a discrete-GPU PCIe weight-stream bottleneck.”

## 2. One tile

Reference candidate:

```text
             Carrizo APU
      +----------------------+
      |                      |
      | CPU cores  Radeon R6 |
      |     |          |      |
      |     +----+-----+      |
      |          |            |
      | shared memory fabric  |
      |          |            |
      | DDR controller        |
      +-----+------------+----+
            |            |
        DDR3 ch A    DDR3 ch B
```

A8-8600P vendor facts:

- 4 CPU cores / 4 threads;
- Radeon R6, 6 graphics cores;
- 2 DDR3 memory channels;
- memory specification up to 2133 MT/s;
- 15 W default TDP;
- HSA, Vulkan and IOMMU v2 listed as supported technologies.

AMD's Carrizo hUMA description states that CPU and GPU share one memory address space and both can access platform memory. This is the key physical reason this track avoids the `host DDR -> PCIe -> discrete GPU` weight path.

## 3. Startup

For N tiles:

```text
1. boot Linux on every tile
2. verify exact APU/GPU/driver revision
3. verify both DDR channels are populated and active
4. enumerate usable system memory
5. start Transit node daemon
6. load assigned K3 tensor shards from storage/network into local DDR3
7. checksum every resident shard
8. create persistent Vulkan buffers/descriptors/pipelines
9. establish collective/network connections
10. begin inference
```

Weights should remain resident across tokens. Normal decode must not reload resident weights from another machine for every token.

## 4. Prompt to first token

Simplified path:

```text
user text
   |
   v
tokenizer
   |
   v
input token IDs
   |
   v
embedding / initial state
   |
   v
layer 0
   |
   v
layers 1..92
   |
   v
LM head / sampling
   |
   v
next token ID
   |
   v
text output
```

The 93 model layers remain serial boundaries for one autoregressive sequence. Parallelism is exploited **inside** each layer by splitting independent matrix work across memory-compute tiles.

## 5. One model-shaped operation on one tile

A tile receives a small activation vector and a command describing which resident matrix shard to apply:

```text
network/runtime command
       |
       v
activation buffer in shared memory
       |
       v
Vulkan compute dispatch
       |
       +--> read packed weights from local DDR3
       +--> unpack/dequantize
       +--> multiply by activation
       +--> accumulate local partial result
       |
       v
partial/result buffer
       |
       v
collective/network step
```

The CPU does not need to copy the entire weight shard through a separate PCIe device path. It still incurs runtime/driver/network work and therefore cannot be assumed to be free.

## 6. Why simple expert ownership is insufficient for one-token scaling

K3 has 896 routed experts and selects 16 experts/token in each MoE layer.

Naive placement:

```text
board A owns whole expert 10
board B owns whole expert 11
...
```

For one token only the owners of the 16 selected experts perform routed-expert weight reads. With 160 boards, most boards would be idle during that routed stage.

This can be acceptable for high multi-user throughput, but it does not expose all 320 channels to a **single decode sequence**.

## 7. Tensor-sharded active work

To aggregate many memory controllers for one token, split an active matrix across a group of tiles:

```text
                         activation x
                              |
               +--------------+--------------+
               |              |              |
               v              v              v
            tile 0          tile 1         tile N
          weight shard    weight shard    weight shard
               |              |              |
               v              v              v
           partial y0      partial y1      partial yN
               +--------------+--------------+
                              |
                         reduce/combine
                              |
                              v
                         next operation
```

The precise reduction depends on row-wise/column-wise sharding.

### Row parallel

Each tile owns output rows and produces a distinct output slice. Result slices are gathered/concatenated.

### Column parallel

Each tile owns input columns and produces a partial sum for the same output vector. Partial outputs require reduction.

The best K3 partition should minimize collective count and bytes while preserving long sequential DDR bursts.

## 8. MoE layer sketch

A simplified distributed MoE layer may look like:

```text
hidden state
    |
latent/down projection
    |
router -> 16 expert IDs
    |
    +--> expert 1 shard group ----\
    +--> expert 2 shard group -----+
    ...                             +--> weighted expert reduction
    +--> expert 16 shard group ----+
    |                              /
shared expert path ---------------+
    |
latent/up projection
    |
next hidden state
```

The critical design question is not merely network **bandwidth**. The exchange repeats at model-layer cadence, so software and switch latency matter strongly.

## 9. Dense/attention/KDA work cannot be ignored

K3 is not “only the 16 routed experts.” A complete token also contains:

- one dense layer;
- shared experts;
- latent projections;
- router;
- 69 KDA layers;
- 24 Gated MLA layers;
- normalization/residual work;
- embeddings/output head;
- state/KV handling;
- sampling.

Some of this work can also be tensor-sharded, but every additional collective consumes the per-layer latency budget.

The physical prototype must therefore benchmark both:

1. routed-expert MXFP4-shaped kernels;
2. representative high-precision/KDA/non-routed kernels.

## 10. One token across a fleet

Conceptual single-sequence decode:

```text
for layer in 0..92:
    determine operation/routing
    broadcast or distribute the current activation
    run local resident-weight kernels in parallel
    reduce/gather partial outputs
    complete serial layer boundary

run final head/sampling
emit token
repeat
```

No system-level token/s claim is valid until the complete loop is measured.

## 11. Network hierarchy candidate

A flat 160-node all-to-all is undesirable. A likely direction is hierarchical collectives:

```text
APU tiles
  -> small local shard groups
  -> group reduction
  -> rack/switch-level reduction
  -> next operation
```

Potential transport candidates to benchmark include standard Ethernet at progressively higher link rates and any low-cost low-latency NIC/fabric that the selected board can electrically support.

The exact NIC cannot be chosen from link-rate marketing alone. Required measurements are:

- small-message one-way latency;
- ping-pong latency;
- 7/14/28 KiB payload latency;
- simultaneous group collective latency;
- p50/p95/p99 jitter;
- CPU overhead;
- effective switch bisection bandwidth.

## 12. Capacity

The released K3 checkpoint is about 1.56 TB. At 160 nodes the average persistent checkpoint share is roughly:

```text
1.56 TB / 160 ~= 9.75 GB/node
```

This is only an average. Real placement needs extra capacity for:

- non-uniform tensor sizes;
- hot-expert replication;
- metadata/scales;
- temporary buffers;
- state/KV if local;
- spare/recovery copies.

Therefore “aggregate RAM > 1.56 TB” is necessary but not sufficient.

## 13. Failure model

A 160-node second-hand fleet should assume failures.

The runtime should maintain:

```text
node_id
resident tensor/shard IDs
checksums
health/temperature
network status
last completed sequence ID
spare placement capacity
```

A failed node must either be replaced by a replica/spare or force deterministic reload/replacement of its shard group.

## 14. What this architecture is NOT

It is not:

```text
DDR3 computing by itself
```

It is not:

```text
one GPU electrically driving arbitrary DDR3 DIMMs
```

It is not:

```text
160 boards automatically giving 160x single-token speedup
```

It is:

> a distributed collection of integrated CPU/GPU memory-compute tiles whose central feasibility question is whether local DDR bandwidth can be converted into useful K3 kernel throughput faster than repeated distributed synchronization consumes the gain.