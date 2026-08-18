# 21 — GLM-5.2 4×2 Throughput Simulation Correction — 2026-08-18

## Status / correction

This note overrides the `2–3 tok/s expected optimized` planning number in document 19.

That earlier number was **not produced by an end-to-end simulation of the selected architecture**. It inherited the older host-offload mental model too strongly and did not credit the same-token parallelism available from four 4-socket R920-class servers.

The correct source-of-record position is now:

> No exact end-to-end token/s should be frozen until the simulator is calibrated with the four physical machines. A first architecture-aware sensitivity model puts the plausible region materially above 2–3 tok/s when NUMA expert parallelism is actually used.

The committed simulator is:

```text
tools/glm52_4x2/simulate_4x2_sensitivity.py
```

It is an analytical critical-path sensitivity model, not a measured benchmark and not a cycle-accurate emulator.

---

## 1. Why the previous estimate was structurally pessimistic

The previous estimate effectively treated host-memory expert work as a small number of large serial offload paths.

The R920 reference profile already documented in ByteMyWave is:

```text
4 servers
4 CPU sockets/server
16 NUMA/socket domains total
```

The reference CPU is Xeon E7-4890 v2. Intel specifies four CPU-side memory channels / 85 GB/s maximum memory bandwidth for the processor, while Dell's R920 memory architecture exposes two memory risers per processor and four DDR channels per riser. The exact sustained number depends on DIMM population and R920 memory mode, so nominal bandwidth is not used as a measured simulator input.

Sources:

- https://www.intel.com/content/www/us/en/products/sku/75251/intel-xeon-processor-e74890-v2-37-5m-cache-2-80-ghz/specifications.html
- https://www.dell.com/support/manuals/en-us/poweredge-r920/r920ownersmanual/system-memory

The important architecture fact is not the nominal GB/s number. It is that the 16 NUMA/socket domains are independent compute+memory resources that can participate in the same MoE layer if expert placement is designed for it.

---

## 2. GLM-5.2 shape used by the simulator

Official/public configuration:

```text
78 transformer layers
first 3 dense
75 MoE layers
256 routed experts/MoE layer
8 routed experts selected/token
1 shared expert
hidden size 6144
MoE intermediate size 2048
```

Therefore one routed expert contains approximately:

```text
3 × 6144 × 2048
= 37,748,736 logical weights
```

for gate, up and down projections.

Sources:

- https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json
- https://github.com/FareedKhan-dev/glm-5.2-in-c/blob/main/docs/architecture.md

The public `GLM-5.2-ewaste-edition-GGUF` also confirms the practical quantization layout used for old hardware:

```text
Q3_K_M total size       295.71 GiB
layers 3–34 experts     Q2_K
layers 35–77 experts    Q3_K
attention/shared path   Q6_K
```

and reports 13.2 tok/s on 10×MI100 with the complete Q3 build resident in HBM.

Source:

- https://huggingface.co/SixVolts/GLM-5.2-ewaste-edition-GGUF

That result is an external anchor, not a performance coefficient for our hardware.

---

## 3. The key placement change: replicate for compute parallelism, not merely capacity

The 295.7 GiB Q3 model is much smaller than the total RAM available across four filled R920-class servers.

That means RAM should not merely contain one distributed copy of the model. It can be used to create **replicas/shards chosen to expose independent socket bandwidth**.

Preferred sparse-layer mapping:

```text
router selects 8 experts
        |
        +-- expert 0 -> 2 NUMA/socket shards
        +-- expert 1 -> 2 NUMA/socket shards
        +-- expert 2 -> 2 NUMA/socket shards
        +-- expert 3 -> 2 NUMA/socket shards
        +-- expert 4 -> 2 NUMA/socket shards
        +-- expert 5 -> 2 NUMA/socket shards
        +-- expert 6 -> 2 NUMA/socket shards
        +-- expert 7 -> 2 NUMA/socket shards

8 experts × 2 socket shards = 16 socket domains active on the same layer
```

A two-socket expert split can partition the SwiGLU intermediate dimension. Both sockets receive the same small activation vector, each processes half of the expert's intermediate channels from NUMA-local weights, and the two partial down-projection outputs are reduced.

The simulator deliberately discounts the ideal 2× split using a configurable shard-efficiency value. The default sensitivity scenarios use 80% incremental efficiency, i.e. a two-socket split is modeled as `1.8×`, not `2×`.

This is the central architectural reason the earlier 2–3 tok/s estimate was not valid as the optimized-system target.

---

## 4. Why GPU cache is not modeled as a magic multiplier

A layer selects eight routed experts.

If the per-expert GPU fast-path hit probability is `h`, the probability that **all eight** selected experts are GPU-ready is:

```text
P(all 8 ready) = h^8
```

Examples:

```text
h = 0.60 -> 1.7%
h = 0.75 -> 10.0%
h = 0.82 -> 20.4%
h = 0.88 -> 36.0%
```

If even one CPU expert remains on the critical path, partial GPU hits may save bandwidth and energy but do not automatically remove the CPU layer latency.

This prevents the model from multiplying a cache-hit percentage directly into token/s.

The public colibri/GLM-5.2 measurements nevertheless show strong expert locality: on its workload, 60 GB of VRAM expert cache reached ~65.5% hit rate, 120 GB ~79.4%, and 180 GB ~88.4%. Those are workload-specific observations, not ByteMyWave assumptions.

Source:

- https://github.com/FareedKhan-dev/glm-5.2-in-c/blob/main/results/ANALYSIS.md

---

## 5. Why MTP is also not multiplied by tokens-per-forward

The same public GLM-5.2 C implementation measured its MTP head at roughly 50% draft acceptance and 2.67 tokens/forward, but the wall-clock A/B improvement was much smaller than 2.67× because verification itself costs work.

Therefore the simulator takes **wall-clock MTP speedup** as a separate sensitivity input.

Default scenarios use only:

```text
1.00× to 1.25×
```

rather than multiplying by 2.67.

Sources:

- https://github.com/FareedKhan-dev/glm-5.2-in-c
- https://github.com/FareedKhan-dev/glm-5.2-in-c/blob/main/results/ANALYSIS.md

---

## 6. First sensitivity run

The following scenarios are committed in the simulator. They intentionally span poor to strong implementation quality.

| Scenario | CPU expert rate / socket | Expert socket shards | Hot hit | Always-on path | Network / MoE layer | MTP wall gain | Simulated tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| floor: one socket/expert | 10 Gw/s | 1 | 60% | 80 ms/token | 0.50 ms | 1.00× | **2.52** |
| pessimistic 2-socket | 12 Gw/s | 2 @ 80% eff | 65% | 70 ms/token | 0.45 ms | 1.05× | **4.54** |
| conservative | 18 Gw/s | 2 @ 80% eff | 75% | 50 ms/token | 0.30 ms | 1.10× | **7.17** |
| central sensitivity | 25 Gw/s | 2 @ 80% eff | 82% | 35 ms/token | 0.20 ms | 1.18× | **11.28** |
| strong sensitivity | 32 Gw/s | 2 @ 80% eff | 88% | 28 ms/token | 0.15 ms | 1.25× | **15.86** |

These numbers are **not predictions with equal confidence**. They answer a different question:

> If the physical measurements land at these inputs, what does the architecture imply for the critical path?

The earlier `2–3 tok/s` value corresponds approximately to the **floor / poorly parallelized** region of the table, not to the optimized architecture.

---

## 7. Current engineering interpretation

Before physical calibration, the useful target bands are now:

```text
< 4 tok/s     architecture or kernel implementation is leaving major parallelism unused
~4–7 tok/s    pessimistic / weak socket kernel or high communication overhead
~7–12 tok/s   credible conservative-to-central engineering region
~12–16 tok/s  strong result if socket kernels, GPU always-on path and fabric cooperate
>16 tok/s      stretch; requires measurement before being used for procurement decisions
```

This is a **sensitivity envelope**, not an end-to-end benchmark.

A 10 tok/s result is no longer rejected by the project. It is now an explicit engineering target to test.

---

## 8. Measurements that collapse the uncertainty

Only a short set of real measurements controls most of the range:

1. **Q3/Q2 expert Gweights/s per E7 socket**, NUMA-local, one real GLM-5.2 expert.
2. **1-socket vs 2-socket expert tensor split efficiency** on one R920.
3. **always-on attention/shared path ms/token** using the exact purchased GPU inventory.
4. **four-server reduction latency** for the real ~6k-element activation/result payload cadence.
5. **GPU hot-expert hit rate** on agentic/coding traces.
6. **MTP wall-clock A/B**, not merely acceptance or tokens/forward.

Once values 1–4 exist, the width of the range should shrink dramatically even before full-model generation works.

---

## 9. Decision

Continue with the 4×2 prototype.

Do not use `2–3 tok/s` as the expected optimized number.

Use **10 tok/s as the primary engineering target**, with the current uncalibrated sensitivity envelope centered around roughly **7–12 tok/s** and a strong region around **12–16 tok/s**.

The next evidence gate is not another generic paper search. It is calibration of the committed simulator with one real expert, one real two-socket split, the actual GPU list and the actual inter-server fabric.
