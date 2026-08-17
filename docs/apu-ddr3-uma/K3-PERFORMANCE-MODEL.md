# Kimi K3 Performance Model for APU DDR3 UMA

## 1. Purpose

This file defines what can be calculated before hardware exists and prevents nominal DDR bandwidth from being reported as end-to-end Kimi K3 speed.

Central rule:

> **Memory capacity is arithmetic. Interface bandwidth is a roofline. End-to-end token/s is a benchmark.**

## 2. Official K3 model facts used here

From the released Kimi K3 model card/config/paper:

```text
total parameters             ~2.8T
activated parameters/token   ~104B
hidden layers                93
dense layers                 1
routed experts               896
selected routed experts      16/token
shared experts               2
hidden size                  7168
latent MoE dimension         3584
MoE hidden dimension         3072
attention composition        69 KDA + 24 Gated MLA
checkpoint size              ~1.56 TB
```

The config declares MXFP4 packed quantization for target `Linear` modules with:

```text
num_bits    = 4
group_size  = 32
scale_dtype = uint8
```

but also contains an ignore list including self-attention, shared experts, some MLP projections, LM head and vision components. Therefore **not every active parameter can be modeled as exactly 0.5 byte.**

## 3. Routed-expert arithmetic

The K3 architecture uses three expert matrices with latent/expert dimensions corresponding to:

```text
3 * 3584 * 3072
= 33,030,144 weights/expert
```

With 1 dense layer out of 93, the working MoE-layer count is:

```text
93 - 1 = 92 MoE layers
```

With 16 selected experts/token/layer:

```text
33,030,144 * 16 * 92
= 48,620,371,968
~= 48.62B routed-expert weights active/token
```

This reproduces the expert-size arithmetic already present in the Transit branch.

## 4. Three active-byte models

### Model A — absolute four-bit lower bound

If all 104B active parameters were exactly 4 bits and each were read once:

```text
104e9 * 0.5 byte = 52 GB/token
```

This is a useful lower bound but does not represent the released mixed-format checkpoint exactly.

### Model B — checkpoint-average screening model

Use the released checkpoint's average byte density:

```text
1.56e12 bytes / 2.8e12 params
~= 0.557 byte/parameter

104e9 * 0.557
~= 58 GB/token
```

PR #11 already uses this as a screening estimate.

It assumes the active subset has approximately the same storage mix as the full checkpoint. That is not guaranteed.

### Model C — conservative routed-MXFP4 / remainder-BF16 envelope

MXFP4 group storage, assuming one 8-bit scale per 32 four-bit weights:

```text
32 * 4 bits = 16 bytes packed weights
+ 1 byte scale
= 17 bytes / 32 weights
= 0.53125 byte/weight
```

Routed active bytes:

```text
48.620371968B * 0.53125
~= 25.83 GB/token
```

If every other active parameter were pessimistically treated as BF16:

```text
remaining active params
= 104B - 48.620371968B
~= 55.379628032B

BF16 bytes
~= 110.76 GB
```

Combined:

```text
25.83 + 110.76
~= 136.6 GB/token
```

**This is a conservative envelope, not an exact K3 byte count.** The official quantization config can quantize additional non-expert `Linear` weights unless they match the ignore rules, so exact traffic may be materially below this envelope.

## 5. Gate 0: exact tensor-level byte accounting

Before buying a fleet, build an executable inventory from the released checkpoint and forward graph:

```text
for each operation/tensor used in one decode token:
    tensor name
    layer
    expert/shared/attention/etc class
    shape
    storage dtype/packing
    scale/metadata bytes
    selected frequency
    bytes read/token
    reuse/cache assumption
```

Output:

```text
exact_weight_bytes/token by component
exact_scale_bytes/token
state/KV traffic estimate
activation/network bytes per layer
```

Until this exists, all token/s numbers in this document are rooflines over Models A/B/C.

## 6. One A8-8600P tile — nominal memory ceiling

A8-8600P has two DDR3 channels and supports up to DDR3-2133 at the APU level.

Nominal x64 payload:

```text
DDR3-1600: 12.8 GB/s/channel
DDR3-2133: 17.064 GB/s/channel
```

Two-channel tile:

```text
DDR3-1600: 25.6 GB/s
DDR3-2133: 34.128 GB/s
```

The motherboard/BIOS/DIMM population may run lower. Measure actual frequency and sustained GPU-visible bandwidth.

## 7. 160-tile / 320-channel roofline

Nominal aggregate interface bandwidth:

```text
320 * 12.8   = 4096 GB/s  at DDR3-1600
320 * 17.064 = 5460.48 GB/s at DDR3-2133
```

If all that bandwidth could contribute simultaneously to one token, the pure weight-path ceilings are:

| Byte model | DDR3-1600 320ch | DDR3-2133 320ch |
|---|---:|---:|
| 52 GB lower bound | 78.8 tok/s | 105.0 tok/s |
| 58 GB screening | 70.6 tok/s | 94.1 tok/s |
| 136.6 GB conservative envelope | 30.0 tok/s | 40.0 tok/s |

These are **not end-to-end predictions**. They assume:

- all 320 channels contribute to the same token;
- full nominal payload utilization;
- compute keeps up with DRAM;
- perfect routing/sharding;
- zero collective latency;
- no KDA/state/activation overhead beyond the chosen byte model;
- no stragglers.

Real performance is lower unless optimizations reduce traffic/reuse weights.

## 8. Effective bandwidth sensitivity

Let `eta_mem` be the fraction of nominal bandwidth converted into useful K3 weight consumption.

For DDR3-1600 / 320 channels:

```text
B_nominal = 4096 GB/s
B_effective = eta_mem * B_nominal
TPS_weight = B_effective / W_active
```

Using the conservative 136.6 GB model:

| Useful DDR efficiency | Effective BW | Weight-path ceiling |
|---:|---:|---:|
| 100% | 4096 GB/s | 30.0 tok/s |
| 70% | 2867 GB/s | 21.0 tok/s |
| 60% | 2458 GB/s | 18.0 tok/s |
| 50% | 2048 GB/s | 15.0 tok/s |
| 40% | 1638 GB/s | 12.0 tok/s |
| 30% | 1229 GB/s | 9.0 tok/s |

Again, this isolates only one bottleneck.

## 9. Why the 320 channels may not all be useful

### Expert ownership only

K3 selects 16 routed experts. If each whole expert belongs to one board, only a limited number of boards read routed-expert weights for that token.

Therefore:

```text
sum(all installed DDR bandwidth)
!=
useful bandwidth for one token
```

unless active matrices are sharded across enough independent memory controllers.

### Tensor sharding

Tensor sharding can make many APUs process the same active operation concurrently, but requires gather/reduce collectives.

The true per-layer critical path becomes approximately:

```text
T_layer >= max(T_local_weight, T_local_compute)
           + T_collective
           + T_misc
```

and:

```text
T_token ~= sum(T_layer over 93 serial layer boundaries)
           + final head/sampling
```

## 10. Per-layer latency budget

Ignoring final overhead, an average budget per model layer is:

```text
budget = 1 / (target_tps * 93)
```

Examples:

| Target | Token time | Average budget/layer |
|---:|---:|---:|
| 1 tok/s | 1000 ms | 10.75 ms |
| 5 tok/s | 200 ms | 2.15 ms |
| 10 tok/s | 100 ms | 1.075 ms |
| 20 tok/s | 50 ms | 0.538 ms |
| 30 tok/s | 33.3 ms | 0.358 ms |

This budget includes local memory, compute, network and software.

## 11. Activation-size order of magnitude

K3 dimensions give useful payload scales:

```text
hidden BF16 vector:
7168 * 2 bytes = 14,336 bytes ~= 14 KiB

latent BF16 vector:
3584 * 2 bytes = 7,168 bytes ~= 7 KiB
```

At 1 Gb/s, the wire time for a single 14,336-byte payload is already about:

```text
14,336 * 8 / 1e9 ~= 115 microseconds
```

before Ethernet framing, software, switch latency, collective phases or contention.

This does not prove a specific NIC requirement, but it shows why 1 GbE is unlikely to be comfortable for a high-tok/s layer-synchronous design.

## 12. Compute roofline must also be measured

Even if decode GEMV is memory-bound on modern K3 deployments, Radeon R6 is an old GFX8 iGPU without modern tensor/matrix cores.

Required microbenchmarks:

```text
pure GPU-visible DDR read GB/s
MXFP4 packed read + scale + unpack GB/s
K3-shaped GEMV output vectors/s
BF16/FP32 fallback throughput
KDA recurrent-state kernel latency
shared-expert path latency
normalization/residual kernel latency
dispatch overhead
```

If compute cannot consume memory at a high fraction of sustained read bandwidth, adding channels does not help.

## 13. CPU contribution

The CPU cores are intentionally not the primary matrix engine, but they still may become a bottleneck in:

- Vulkan submission;
- command generation;
- packet processing;
- collectives;
- routing;
- synchronization;
- copying/format conversion accidentally performed on CPU;
- interrupt load.

Record CPU utilization and cycles/token in every prototype run.

## 14. Power model

A8-8600P default TDP is 15 W. Using TDP only as a silicon sizing number:

```text
15 W / 2 DDR channels = 7.5 W/channel APU-TDP-equivalent
160 APUs * 15 W = 2.4 kW aggregate APU TDP
```

This is **not wall power**. Fleet power also includes:

- motherboard/chipset;
- DIMMs;
- NIC;
- storage/boot media;
- VRMs;
- PSU losses;
- switches;
- fans.

Measure watts at the wall for one node and then for a small multi-node group before scaling.

## 15. Speculative decoding

K3 includes deployment work for speculative decoding. In principle, verifying multiple proposed tokens can amortize weight traffic across more than one accepted output token.

This track does **not** include speculative-decoding gains in baseline procurement arithmetic.

Baseline must succeed first. Any speculative gain is an optimization measured later as:

```text
accepted output tokens / full-weight verification pass
```

## 16. Decision metric

For each prototype stage report both:

```text
weight-path tok/s equivalent
full synthetic-layer tok/s equivalent
full K3 end-to-end tok/s (only when actually running full model)
```

The fleet purchase decision must use the measured scaling curve, not the nominal `320 channels` sum.