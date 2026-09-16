# Q4_K × Q8_K x8meta kernel and cross-track audit — 2026-09-16

Status: **measured operator result + read-only audit of adjacent ByteMyWave tracks**.

Branch ownership: `research/ram-lookup-compute-2026-09-16` / PR #16.

This note does **not** modify or supersede the implementation ownership of PR #9, #10, #11, #12, #13 or #14. It records how the current x86 Q4_K×Q8_K result was obtained, what it proves, what it does not prove, and how it relates to the other agents' work without multiplying incompatible metrics together.

---

## 1. Evidence classes used in this note

To keep multiple-agent work from collapsing into one optimistic number, every conclusion is classified as one of:

1. **Measured end-to-end model result** — complete model generation on named hardware.
2. **Measured operator/subsystem result** — kernel, memory, PCIe, network, etc.
3. **Analytical/sensitivity model** — derived from explicit assumptions; not a benchmark.
4. **Architecture proposal** — implementation direction awaiting measurement.
5. **Rejected/superseded result** — useful history but must not be used as current evidence.

The x8meta result in this document is evidence class **2**, not class 1.

---

## 2. Exact development machine

The validated Q4_K×Q8_K x8meta benchmark was run on:

```text
CPU                 Intel Core i5-12500H
cores / threads     12 / 16
RAM                 16 GiB
DIMMs               2 × 8 GiB Samsung M425R1GB4BB0-CQKOL
memory speed        DDR5-4800 configured
OS/shell            Windows / Windows PowerShell 5.1
compiler            GCC 13.2.0 MinGW-W64
compiler path       C:\Strawberry\c\bin\g++.exe
native ISA macros   AVX2=yes, FMA=yes, AVX-VNNI=yes
```

The final V12 comparison was single-threaded. The Windows scheduler was allowed to place the thread (`cpu=-1`), but the two kernels were measured in the same process with paired/interleaved timing and alternating order to reduce scheduler/turbo ordering bias.

This is **not** a Xeon E5-2680 v2 result. The actual Gen8/Ivy Bridge path remains a separate required benchmark.

---

## 3. Exact model tensor and the geometry correction

Model source:

```text
Ollama GGUF/blob
sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490
```

Tensor:

```text
blk.0.ffn_gate.weight
GGUF ne[0] = 2560
GGUF ne[1] = 9216
```

In `GGUFReader`, `ReaderTensor.shape` remains in GGML dimension order. Therefore the real GEMV is:

```text
9216 output rows × 2560 input columns
23,592,960 logical weights
92,160 Q4_K blocks
10 Q4_K blocks/row
13,271,040 Q4_K bytes = 12.656 MiB
```

Geometry-correct V11 fixture:

```text
activation FP32 values     2560
activation FP32 bytes      10,240
Q8_K blocks                10
Q8_K activation bytes      2,920
reference FP32 outputs     9,216
reference bytes            36,864
geometry_version           2
```

### Superseded geometry

The earlier V10 fixture interpreted this as `2560 rows × 9216 columns`. Total weights/bytes were identical, so self-consistent correctness tests passed, but row boundaries were wrong for the actual model tensor.

All V10 model-shaped timing conclusions are therefore **superseded for performance interpretation**. They remain useful only as arithmetic-development history.

The corrected preparation code is:

- `benchmarks/ram-lookup-compute/prepare_q4k_exact.py`

The corrected native runner refuses legacy fixtures without `geometry_version >= 2`.

---

## 4. Numerical formats

### Original Q4_K block

The benchmark consumes the real llama.cpp-style 256-weight Q4_K representation:

```cpp
struct Q4KRaw {
    uint16_t d;
    uint16_t dmin;
    uint8_t  scales[12];
    uint8_t  qs[128];
};
```

Size:

```text
144 bytes / 256 logical weights
```

The 12 metadata bytes encode eight 6-bit scales and eight 6-bit minimum coefficients. `qs` contains 4-bit weight nibbles.

### Q8_K activation block

```cpp
struct Q8K {
    float   d;
    int8_t  qs[256];
    int16_t bsums[16];
};
```

Size:

```text
292 bytes / 256 activations
```

The benchmark quantizer mirrors the current Q8_K reference semantics used by the operator. `bsums` provide the 16-element group sums used by the Q4_K minimum correction.

Q8_K activation quantization is part of the normal Q4_K×Q8_K operator path. It is **not** a new x8meta loss.

On the real tensor/activation fixture, Q4_K×Q8_K versus the earlier Q4_K×FP32 reference measured:

```text
max abs     1.402263e-02
relative L2 7.040635e-03
cosine      0.999975214431
```

---

## 5. Baseline kernel actually compared

Source of record:

- `benchmarks/ram-lookup-compute/q4k-q8k-x86-v12/transit_v12_compare.cpp`

The baseline is the standalone port of the relevant llama.cpp-style AVX2 Q4_K×Q8_K arithmetic shape.

Per Q4_K block it:

1. converts Q4_K `d` and `dmin` from FP16;
2. decodes packed 6-bit scale/min metadata;
3. loads Q8_K `bsums` and computes minimum corrections;
4. loads 32 packed Q4 bytes at a time;
5. extracts low/high 4-bit nibbles;
6. uses AVX2 unsigned-byte × signed-byte multiply-add (`vpmaddubsw` semantics through `_mm256_maddubs_epi16`);
7. multiplies pair sums by Q4_K scales with `_mm256_madd_epi16`;
8. accumulates 32-bit integer dot products;
9. applies Q4/Q8 block scales and minimum correction to FP32 accumulation.

This is the baseline that matters for the validated comparison. Earlier managed C# direct loops and the V15 `packed SIMD control` are not used as the final denominator.

---

## 6. x8meta runtime representation

The winning representation groups eight output rows for the same Q4_K block position:

```cpp
struct Q4KX8Meta {
    uint16_t d[8];
    uint16_t dmin[8];
    uint8_t  scales[8][8];
    uint8_t  mins[8][8];
    uint8_t  qs[1024];
};
```

Size:

```text
8 ordinary Q4_K blocks   8 × 144 = 1152 bytes
x8meta block             1184 bytes
overhead                 32 bytes / 8 blocks
                         +2.7778%
```

Important: x8meta does **not** requantize or approximate weights.

It preserves:

- the original Q4 nibbles;
- the original FP16 `d` values;
- the original FP16 `dmin` values;
- the exact decoded six-bit scale/min values.

The only change is runtime layout/predecode.

### Model-load-time repack

For each group of eight output rows and each Q4_K block index:

1. copy `d` and `dmin` into 8-wide arrays;
2. decode the 12 packed metadata bytes once into eight scales + eight mins for every row;
3. interleave Q4 bytes in 8-byte fragments across the eight rows so a Q8 activation fragment can be reused while several output rows are live.

The repack is one-time model-load work. It is not performed per token.

---

## 7. x8meta AVX2 hot loop

For every 8-row group and every Q8_K/Q4_K block:

1. keep two AVX2 integer accumulators for rows 0–3 and rows 4–7;
2. process the eight 32-weight Q4_K groups as four low/high pairs;
3. read already-decoded `scales[g][row]` and `mins[g][row]` rather than reconstructing six-bit metadata in the hot loop;
4. combine Q8_K `bsums` with the eight row-specific minimum coefficients;
5. load an 8-byte Q8 fragment and broadcast it across 64-bit lanes;
6. load interleaved Q4 bytes for four rows at once;
7. extract low/high nibbles with mask/shift;
8. perform `_mm256_maddubs_epi16` followed by `_mm256_madd_epi16` with row-specific scale vectors;
9. repeat for rows 4–7;
10. horizontally reduce per-row integer accumulators;
11. apply each row's exact FP16 `d/dmin` and the Q8_K block scale;
12. accumulate FP32 output for all eight rows.

The gain therefore comes from two mechanisms, not from RAM doing arithmetic:

```text
A. metadata work moved out of the token hot loop
B. one Q8 activation fragment reused across eight output rows
```

DDR/RAM remains storage. CPU SIMD performs the arithmetic.

---

## 8. Correctness

All final candidate paths were checked against a scalar implementation of the same Q4_K×Q8_K operator.

The optimized SIMD paths showed only floating-point accumulation-order differences at approximately `1e-7` relative-L2 scale and cosine `1.0` in the geometry-correct V11 run.

The earlier exact nibble-LUT implementation matched the scalar operator exactly (`max_abs=0`, `relative L2=0`, `cosine=1`) but was much slower than SIMD.

No additional model approximation is accepted in this track.

---

## 9. Rejected and superseded paths

### Position-specific PQ / codebooks — rejected for quality

Real Qwen measurements:

```text
G16 output cosine mean  0.738135
G16 output rel-L2 mean  0.674645

G8 output cosine mean   0.868671
G8 output rel-L2 mean   0.495419
```

These are too inaccurate for a quality-preserving representation.

### Exact nibble LUT — rejected for native x86 performance

Geometry-correct V11 native run:

```text
llama-style AVX2        ~0.86–0.91 ms/GEMV
exact nibble LUT        ~8.7–9.2 ms/GEMV
```

The LUT is numerically exact but roughly an order of magnitude slower than AVX2 on this CPU.

### Simple AVX2 4/8-row batching — rejected

The readable raw/repacked 4-row/8-row candidates were slower than the llama-style baseline.

### Simple AVX-VNNI conversion — rejected

`-march=native` exposed AVX-VNNI on the i5-12500H, but the VNNI candidates were slower than the baseline. The experiment therefore showed that the instruction alone was not the missing optimization; layout and hot-loop scheduling mattered more.

### V15 `1.403×` — superseded as final speedup claim

V15 measured `x8meta = 0.894 ms` versus an internal packed control at `1.254 ms`, yielding `1.403×`.

That result was real for those two functions but the denominator was not the current llama-style AVX2 baseline. It must not be presented as a +40% llama.cpp-class gain.

V12 exists specifically to fix this denominator problem.

---

## 10. Final V12 apples-to-apples result

Real tensor, correct geometry, same process, paired/interleaved timing, 60 timed iterations, 8 warmups.

### Normal `-O3 -march=native`

```text
HOT
llama median       1.228 ms
x8meta median      1.071 ms
paired speedup     1.094×

64 MiB EVICTED
llama median       1.295 ms
x8meta median      1.174 ms
paired speedup     1.032×
```

### `-fno-unroll-loops`

```text
HOT
llama median       1.313 ms
x8meta median      1.165 ms
paired speedup     1.115×

64 MiB EVICTED
llama median       1.304 ms
x8meta median      1.169 ms
paired speedup     1.110×
```

### Validated operator claim

The current defensible claim is:

> On the i5-12500H, for this real 9216×2560 Qwen Q4_K×Q8_K GEMV, the x8meta representation is approximately **1.11× faster** than the current llama-style AVX2 baseline when compiled with `-fno-unroll-loops`, while adding **2.78% runtime Q4_K storage** and no new quantization.

This is a **measured operator speedup**, not measured model tok/s.

---

## 11. Why `-fno-unroll-loops` is retained

Earlier assembly inspection showed aggressive compiler unrolling around the x8 path creating stack traffic/spills. V12 confirms the performance consequence:

```text
normal cold/evicted gain       1.032×
no-unroll cold/evicted gain    1.110×
```

Therefore the x8meta experiment must preserve compiler-codegen control and generated-assembly inspection as part of future ports.

Hand-written assembly is not yet justified: the first integration gate is to preserve the measured C++ intrinsic result inside the real runtime and inspect codegen there.

---

## 12. Translating +11% operator speed into model tok/s

There is no honest exact model tok/s delta until the kernel is integrated and `llama-bench`/full decode is measured.

For a model where fraction `f` of token latency is spent in eligible Q4_K×Q8_K GEMV and that portion receives a kernel speedup `S=1.11`, Amdahl's law gives:

```text
model_speedup = 1 / ((1 - f) + f / 1.11)
```

Examples:

| Eligible Q4_K×Q8_K share of token latency | Projected total speedup | Tok/s increase |
|---:|---:|---:|
| 30% | 1.0306× | +3.1% |
| 50% | 1.0521× | +5.2% |
| 70% | 1.0745× | +7.5% |
| 80% | 1.0861× | +8.6% |
| 90% | 1.0979× | +9.8% |
| 100% | 1.1100× | +11.0% |

Thus the x8meta-only **absolute upper bound** on this measured CPU is approximately +11% model tok/s. A first realistic target for a CPU decode strongly dominated by eligible Q4_K GEMV is roughly **+5–10% tok/s**, but this remains a projection until full-model profiling establishes `f`.

If a baseline model decodes at `B` tok/s, the projected x8meta rate is simply:

```text
B × model_speedup
```

For example, at `f=0.8`, `10 tok/s` would become about `10.86 tok/s`; it would **not** become `11.1 tok/s` unless the entire token critical path were accelerated by the kernel.

---

## 13. Memory-bound limit and the 2.78% storage cost

x8meta is a compute/decode/layout optimization, not a compression or bandwidth reduction.

If execution becomes purely DRAM-bandwidth-bound and every repacked byte must be streamed, the storage overhead alone gives a pessimistic bandwidth ratio:

```text
1 / 1.02778 ≈ 0.973×
```

or about a 2.7% loss if compute savings become irrelevant.

A useful critical-path model is:

```text
baseline T0 = max(T_mem, T_compute)

x8meta T1 = max(
    1.02778 × T_mem,
    T_compute / S_compute
)
```

V12's explicit 64 MiB eviction result is encouraging because x8meta still measured `1.110×` with no-unroll, but this remains a **single-thread** test. A many-core server may saturate DRAM bandwidth differently.

Therefore a multi-thread and real-server bandwidth test is mandatory before claiming that the +11% survives at socket scale.

---

## 14. Applicability to larger dense models

### Directly applicable when

- the model actually contains Q4_K tensors;
- the CPU path consumes them as Q4_K×Q8_K GEMV;
- tensor row groups are compatible with the x8 path or have a correct tail fallback;
- the runtime can afford +2.78% on the repacked Q4_K subset.

The method is not inherently tied to a 4B model. The repack operates block-by-block and row-group-by-row-group.

### Storage scaling

If **all** model bytes were eligible Q4_K, x8meta would add approximately:

```text
100 GB Q4_K  -> +2.78 GB
500 GB Q4_K  -> +13.9 GB
1 TB Q4_K    -> +27.8 GB
```

For mixed-quant models, overhead applies only to tensors actually repacked.

### Why model size does not automatically mean the same +11%

Larger models generally reduce cache reuse and can push the full runtime toward DRAM bandwidth. Full decode also uses multiple threads, attention/KV work, synchronization and possibly GPU/network stages.

The 64 MiB eviction result shows the kernel does not require this 12.656 MiB tensor to remain permanently hot in LLC, but it does **not** prove that a many-core, full-model stream preserves the same ratio.

Required gates:

1. real llama.cpp integration;
2. single-thread `llama-bench`;
3. thread-count sweep;
4. hardware-counter / DRAM bandwidth measurement;
5. exact E5-2680 v2 run.

---

## 15. Applicability to MoE

### Q4_K MoE

For a MoE checkpoint whose routed expert matrices are actually Q4_K and execute through Q4_K×Q8_K on CPU, the same x8meta representation can be used for each resident expert tensor.

Only selected experts execute per token, but repack is model-load-time. The relevant model-level speedup depends on the fraction of the **critical selected-expert path** spent in eligible GEMV.

The router, network, attention, expert reduction and GPU work are unchanged.

For a MoE layer:

```text
T_layer ≈ max(selected expert owner critical paths)
          + routing/reduction/network/other serial work
```

x8meta reduces only the local CPU GEMV component of the selected expert owner(s).

### It does not solve weight transport

x8meta does not reduce active weight bytes. It increases the local runtime representation by 2.78%.

Therefore it cannot turn a K3-style `~0.2 tok/s` single-PCIe-weight-streaming architecture into multi-token/s inference. That problem requires avoiding weight movement, expert locality, independent memory domains, GPU residency or near-memory compute.

---

## 16. GLM-5.2 audit: exact x8meta is not directly applicable to the selected Q3 build

The GLM-5.2 4×2 branch documents a public Q3_K_M build where:

```text
layers 3–34 routed experts   Q2_K
layers 35–77 routed experts  Q3_K
attention/shared path        Q6_K
```

Therefore the current Q4_K×Q8_K x8meta kernel is **not** the kernel that dominates that selected GLM deployment.

The transferable idea is instead:

> predecode quant metadata once, interleave several output rows/expert shards, and reuse activation fragments across them while preserving the original quantized representation.

A GLM-specific continuation should implement equivalent `Q2_K×Q8_K` and `Q3_K×Q8_K` row-interleaved/predecoded kernels and benchmark them against current llama.cpp baselines.

Do **not** multiply GLM's simulated tok/s by 1.11 before those kernels exist.

---

## 17. Kimi K3 audit: exact x8meta is not directly applicable to native MXFP4/MXFP8

The Transit K3 documents explicitly treat K3 routed weights/activations as MXFP4/MXFP8-style representations.

Q4_K×Q8_K is therefore not a drop-in K3 operator.

What transfers is the architecture lesson:

```text
packed/block metadata
    -> decode/materialize once
    -> row/expert-local compute-friendly layout
    -> reuse activation fragments across parallel output work
```

For an FPGA/ASIC/near-memory Transit tile, the predecoded metadata does not necessarily need to live in the same expanded DDR representation used by x86. It may live in BRAM/SRAM/register structures or be generated during weight load.

A native MXFP4×MXFP8 kernel must be numerically validated separately.

---

## 18. Audit of the other ByteMyWave tracks

### PR #10 — `transit-ddr3-architecture`

Evidence types:

- measured host-side bitplane/DDR5/NVMe kernels;
- architecture proposal for active DDR3 compute tiles;
- analytical K3 weight-path rooflines.

Important measured host evidence includes roughly `53–57 Gweights/s` for the exact signed INT4×INT8 bitplane engine on the laptop and about `50 GB/s` raw DDR5 in the 6 GiB working-set experiment.

The K3 `~100 tok/s` number is explicitly a **weight-path roofline** from ~304 ideal DDR3-2133 channels against a ~52 GB/token lower-bound model. It is not end-to-end K3 tok/s.

Relationship to x8meta:

- complementary lesson about amortizing decode/control work;
- different numerical format and execution engine;
- do not multiply `1.11×` into the 304-channel roofline.

### PR #11 — `docs/kimi-k3-ddr-cluster`

This track correctly formalizes throughput as a critical-path model and explicitly rejects summing nominal server bandwidth for a serial layer pipeline.

It uses ~58 GB/token as a screening weight-traffic model and states that only complete-model measurement can become real tok/s.

Relationship to x8meta:

- its methodology is the correct way to translate local kernel gains;
- x8meta changes only `T_compute` for eligible CPU GEMV and slightly increases local `T_mem` bytes;
- network, routing, attention and state terms are unaffected.

### PR #9 — `research/heterogeneous-moe-kimi-v1`

This track models CPU/GPU expert ownership and derives per-socket gates rather than claiming they are measured.

Examples in its current Q4 planning model:

```text
K2.5 at 5 tok/s, four sockets:
selected weights/socket  26.424 Gweights/s
weight read/socket       16.515 GB/s

K3 at 5 tok/s, four sockets:
weight read/socket       75.969 GB/s
compute/socket           243.102 GFLOP/s
```

Relationship to x8meta:

- if the deployed expert representation were Q4_K, an x8meta-like kernel could improve the CPU execution rate;
- the +2.78% representation increases local bytes, so memory gates become slightly tighter;
- it does not alter GPU H2D or network requirements.

### GLM-5.2 4×2 branch — `research/glm52-4x2-optimized-runtime-2026-08-18`

Document #19 originally proposed an expected optimized `2–3 tok/s`, but document #21 explicitly supersedes that value with an architecture-aware sensitivity model.

The #21 simulator assumes:

```text
4 servers
4 sockets/server
16 socket/NUMA domains
8 selected experts
2 socket shards/expert in the main scenarios
```

Its central sensitivity point is ~11.28 tok/s and it describes ~7–12 tok/s as a credible sensitivity region, but the document is explicit that this is **not a benchmark or prediction with calibrated confidence**.

Critical hardware mismatch:

> The current four HP DL360p Gen8 setup is dual-socket per server (8 CPU sockets total), not the simulator's 16-socket R920 reference profile.

Therefore the GLM sensitivity numbers cannot be copied directly to the current HP cluster. With eight CPU sockets, eight selected experts can naturally occupy one CPU socket each; giving every expert two socket shards simultaneously would require more CPU socket domains or serialization/partial GPU replacement.

Also, the selected GLM Q3 build uses Q2_K/Q3_K experts, so the Q4_K x8meta result is not directly applicable.

### PR #12 — DDR2 server fabric

This is documentation/roofline work. It explicitly labels DDR2 figures as weight-path rooflines and asks for a real NUMA-local Gweights/s benchmark before making K3 tok/s claims.

No conflict with x8meta; the same evidence rule should be retained.

### PR #13 — Carrizo APU DDR3 UMA

This track is a separate architecture and explicitly carries multiple K3 active-byte models rather than promoting one guessed value. Its nominal-channel tok/s figures are weight-path rooflines, not end-to-end predictions.

No x8meta multiplier should be applied because execution format/hardware are different.

### PR #14 — Active Memory / Chronicles

This track explicitly targets **agentic wall-clock**, repeated retrieval/filesystem work and prompt/context size. It says it is not claiming raw decoder tok/s.

Therefore Active Memory gains and x8meta decoder gains may both improve user-visible task completion, but they must be measured at different layers and must not be added or multiplied without an end-to-end agent benchmark.

---

## 19. What the audit says about the current HP Gen8 cluster

The biggest cross-agent trap is treating R920 16-socket simulations as though they describe the four dual-socket HP servers.

For the current HP cluster, the CPU-side facts that matter next are:

```text
4 servers × 2 sockets = 8 CPU/socket memory domains
Xeon E5-2680 v2-class Ivy Bridge target
no AVX2
Q4/Q3/Q2 exact model format still model-dependent
```

The current x8meta laptop result can guide the Ivy implementation but cannot provide the HP token rate.

The decisive measurements are:

1. SSSE3/AVX Ivy version of the exact Q4_K x8meta comparator on one E5-2680 v2;
2. thread-count sweep on one socket and one server;
3. NUMA-local versus remote RAM placement;
4. sustained compressed-weight GB/s and Gweights/s;
5. two-socket scaling;
6. four-server activation/reduction latency;
7. only then full model decode.

For GLM-5.2 Q3, build Q2_K/Q3_K equivalents instead of assuming the Q4_K result transfers.

---

## 20. Current tok/s conclusion

### Exact measured statement

```text
Q4_K×Q8_K operator:
~1.11× / +11% on the tested i5-12500H
+2.78% runtime Q4_K storage
no new weight quantization
```

### Conditional full-model statement

For a Q4_K CPU model, x8meta-only model speedup is bounded by the eligible GEMV share of token latency.

Practical pre-integration expectation:

```text
~+5% to +10% tok/s
```

**only if** Q4_K×Q8_K GEMV occupies roughly 50–90% of the token critical path and DRAM bandwidth does not become the dominant bottleneck.

Absolute x8meta-only ceiling on the currently measured CPU:

```text
~+11% model tok/s
```

There is no evidence for +40% model tok/s from this kernel.

### Larger models

Yes, the representation is mechanically applicable to larger Q4_K models. The performance ratio must be remeasured under many-thread DRAM pressure; larger capacity alone does not invalidate the method.

### MoE

Yes, for Q4_K expert tensors, but the overall gain is only on the selected-expert CPU GEMV critical path.

No, not directly for the currently documented GLM-5.2 Q2_K/Q3_K experts or Kimi K3 native MXFP4/MXFP8. Those require format-specific ports of the same layout/predecode/reuse idea.

---

## 21. Next implementation gates

The branch should not create another unrelated lookup experiment before these gates:

```text
Gate A  integrate distinct x8meta runtime layout into current llama.cpp Q4_K CPU path
Gate B  llama-bench / full Q4_K model A/B on i5-12500H
Gate C  thread-count and DRAM-bandwidth sweep
Gate D  SSSE3/AVX Ivy Bridge x8meta kernel
Gate E  run paired comparator on real E5-2680 v2
Gate F  NUMA-local two-socket server benchmark
Gate G  Q2_K/Q3_K x8meta-style prototypes for GLM-5.2
Gate H  MXFP4/MXFP8 metadata/layout prototype only if K3 path is pursued
```

Only class-1 full-model decode may be reported as final tok/s.

---

## 22. Source-of-record files for this track

Measured/current:

- `benchmarks/ram-lookup-compute/q4k-q8k-x86-v12/transit_v12_compare.cpp`
- `benchmarks/ram-lookup-compute/q4k-q8k-x86-v12/run_v12_compare.ps1`
- `benchmarks/ram-lookup-compute/q4k-q8k-x86-v12/RESULTS-2026-09-16.md`
- `benchmarks/ram-lookup-compute/prepare_q4k_exact.py`

Historical/diagnostic:

- `benchmarks/ram-lookup-compute/q4k-q8k-x86-v15/transit_q4k_q8k_exact_v15.cpp`
- `benchmarks/ram-lookup-compute/q4k-q8k-x86-v15/llama_x86_q4k8x8_transit.inc`
- `benchmarks/ram-lookup-compute/transit_q4k_q8k_native.cpp`
- `benchmarks/ram-lookup-compute/q4k_exact_lut_v10.cs`
- `benchmarks/ram-lookup-compute/qwen_position_pq_lut.py`

Cross-track sources audited read-only:

- root `AGENTS.md`
- root `CURRENT-ARCHITECTURE.md`
- root `README.md`
- `docs/REPOSITORY-MAP.md`
- PR #10 / `transit-ddr3-architecture`
- PR #11 / `docs/kimi-k3-ddr-cluster`
- PR #9 / `research/heterogeneous-moe-kimi-v1`
- PR #12 / `agent/transit-ddr2-server-fabric`
- PR #13 / `agent/apu-ddr3-uma-k3`
- PR #14 / `agent/transit-active-memory-chronicles`
- `research/glm52-4x2-optimized-runtime-2026-08-18`

This file is the cross-agent interpretation record for the Q4_K×Q8_K/x8meta work. Other agents should cite it rather than re-deriving the old V10/V15 numbers from chat history.