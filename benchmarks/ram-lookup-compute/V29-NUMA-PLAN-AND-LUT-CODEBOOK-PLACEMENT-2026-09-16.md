# Transit V29 — V27 META per NUMA socket + row sharding

Date: 2026-09-16  
Status: next physical-server benchmark plan  
Target platform: HP DL360p Gen8, dual-socket Xeon E5-2680 v2 class, Windows, DDR3  
Primary numerical path: exact real-Qwen `Q4_K × Q8_K`

## Decision

V29 does **not** add another LUT, codebook, dictionary, dot32 cache, or new inner-loop representation to V27.

V29 keeps the winning strict-Ivy V27 META kernel unchanged inside each NUMA domain and changes the **system topology around the kernel**:

```text
                         Q8 activation
                              |
                      duplicate to both
                      NUMA-local workers
                    +---------+---------+
                    |                   |
                    v                   v
               NUMA/socket 0       NUMA/socket 1
               local META rows      local META rows
               output shard A       output shard B
                    |                   |
                    v                   v
                 V27 META            V27 META
                 unchanged           unchanged
                    |                   |
                    +---------+---------+
                              |
                    concatenate output rows
```

For a row-sharded GEMV, each socket owns disjoint output rows. Therefore the basic `Q4_K × Q8_K` row-sharded case requires **no cross-socket sum reduction**: each socket computes its own `y[row]` values and the final vector is concatenated.

The activation vector is small compared with the streamed weight payload and is duplicated to both NUMA-local worker groups.

## Why V29 follows V27

The strongest same-harness V27 result on the strict-Ivy contract was obtained by predecoded META storage plus direct SSSE3 arithmetic. The V28 hot-dictionary experiment then tested whether exact repeated `dot32` weight patterns could remove more arithmetic.

V28 result supplied from the benchmark run:

```text
DICT-R2  coverage = 0.000%
DICT-R4  coverage = 0.000%
DICT-R8  coverage = 0.000%
DICT-R16 coverage = 0.000%
DICT-R32 coverage = 0.000%
DICT-R64 coverage = 0.000%

12T META-T1-PF0 = 0.224 ms
12T best fused  = FUSED-R32 = 0.249 ms effective
speed vs META   = 0.902x
```

Interpretation:

- the tested real tensor had no useful exact duplicate-pattern coverage for this dictionary definition;
- the fused path added overhead and lost to plain META;
- no further threshold sweep of the same exact-hot-dictionary mechanism is justified without a new source of reuse;
- V28 is a negative result, not a replacement for V27/META.

The next orthogonal resource is memory-controller parallelism. V29 therefore asks whether two independent NUMA-local memory domains can feed two independent copies of the already-good kernel concurrently.

## V29 invariants

V29 must preserve all of these:

1. Same real Qwen tensor geometry: `9216 output rows × 2560 input columns` for `blk.0.ffn_gate.weight`.
2. Same original logical Q4_K weights and Q8_K activation semantics.
3. Same V27 META block representation and exact arithmetic.
4. Same strict Ivy-Bridge-compatible ISA contract; no AVX2/VNNI dependency.
5. Bit-exact correctness against the same reference used by the V27 family.
6. No remote-NUMA weight reads in the timed local-shard path.
7. No duplicate full tensor per socket unless a separate replication experiment explicitly requests it; the primary row-sharded test stores each output-row shard only in its owner NUMA node.
8. Physical-core-first worker placement. Hyperthreads are a separate sweep, not silently mixed into the baseline.
9. Allocation/first-touch and thread affinity must be reported, not assumed.
10. One-socket and two-socket results must be produced by the same executable and data layout so scaling is attributable to NUMA parallelism.

## Required benchmark cases

At minimum:

```text
A. socket 0 local only
B. socket 1 local only
C. socket 0 worker reading socket 1 memory (remote penalty control)
D. socket 1 worker reading socket 0 memory (remote penalty control)
E. dual socket, disjoint row shards, local memory on both sockets
```

For A/B/E, sweep physical-core counts appropriate to the installed CPUs. For an E5-2680 v2 target, include at least 4, 8 and 10 physical workers per socket before testing SMT.

## Metrics

Report:

```text
correctness
rows/socket
bytes/socket
workers/socket
kernel ms/GEMV
aggregate ms/GEMV
payload GB/s per socket
aggregate payload GB/s
logical Gweights/s
local-vs-remote NUMA penalty
2-socket scaling vs best 1-socket result
```

The key V29 number is:

```text
speedup_dual = best_single_socket_ms / dual_socket_row_sharded_ms
```

A 1.5–2.0x scaling region is an engineering target to test, **not a claimed result**.

## What V29 is not

V29 is not:

- another lookup experiment;
- a new codebook;
- a lossy representation;
- a simulated dual-socket laptop benchmark;
- a token/s claim for a complete model;
- a claim that the two-socket system is one coherent bandwidth pool without NUMA locality work.

The benchmark must run on the actual dual-socket server hardware to establish the result.

---

# LUT and codebook placement — retained, not abandoned

The V28 failure only rejects one narrow hypothesis:

> exact duplicate `dot32` weight-pattern reuse in this real Q4_K tensor is common enough to justify a hot dictionary.

It does **not** reject materialized functions, LUTs, vector codebooks, or dictionaries project-wide.

The synthetic V1–V8 track already demonstrated the central rule: a lookup becomes attractive when one lookup represents enough useful work and the lookup surface retains acceptable locality. V7 crossed the large-working-set baseline at 32 logical MACs/lookup; V8 was much faster in its restricted coded domain. The real-Qwen continuation also showed that a small exact nibble LUT can beat a managed direct baseline while a larger byte-pair LUT loses badly. Those results remain valid within their stated domains.

The project should therefore preserve LUT/codebook work in the following roles.

## 1. Activation-dependent tables amortized over many rows

A table built from one activation is potentially useful when its build cost is reused across a very large number of weight rows, experts, or batched output rows.

Acceptance rule:

```text
(table_build + all_lookup_rows) < direct_compute_for_same_rows
```

The V22/V23 exact-Q4_K CPU experiments did not beat META on the current single-vector hot path. That does not forbid a different reuse regime such as larger batch, repeated activation fragments, or hardware-local SRAM/BRAM.

## 2. FPGA / near-memory tile-local materialized functions

The CPU pays address-generation, cache hierarchy, instruction, and load costs for a software lookup. A near-memory FPGA/ASIC tile has a different cost model.

Small tables can live in registers, distributed RAM, SRAM, or BRAM beside the DDR stream and can implement:

- low-bit decode;
- scale/min decode aids;
- fixed coefficient transforms;
- block-format conversion;
- small exact partial-product functions.

A software LUT that loses on x86 can still be useful when mapped to spatial hardware. This must be benchmarked on the target tile, not inferred from CPU timing.

## 3. Predictive expert prefetch/scheduling codebooks — numerically safe

A codebook does not have to replace model arithmetic.

A compact activation signature/codebook may predict likely MoE expert IDs or expert groups and begin NUMA/SSD/GPU/tile prefetch/scheduling early. The authoritative router result still decides the actual experts.

```text
activation/signature
      |
      +--> codebook predictor --> speculative prefetch only
      |
      +--> exact router --------> authoritative expert IDs
```

A wrong prediction costs wasted prefetch work but does **not** change model numerics if the runtime always falls back to the exact router decision.

This is a strong place to use learned codebooks without accepting model-quality loss.

## 4. Activation/network compression between servers

Vector quantization/codebooks can reduce activation or partial-result traffic when the distributed runtime becomes network-bound.

This is lossy unless an exact encoding is found, so it requires:

- measured reconstruction error;
- layer/logit quality validation;
- end-to-end task-quality validation;
- proof that network time saved exceeds encode/decode overhead.

Current architecture work indicates activation volume is often much smaller than weight traffic, so this is **conditional**, not automatically useful.

## 5. KV/state compression

Codebooks/PQ may be useful for KV or other long-lived state when context length makes state bandwidth/capacity material. This is a memory-capacity/bandwidth use rather than a direct Q4_K GEMV acceleration.

Any such path is a separate numerical-quality experiment and must not inherit the poor real-Qwen weight-codebook results as if they were equivalent domains.

## 6. Cold-expert / tiered-memory representation

A secondary approximate codebook representation can be investigated for cold experts only if it is used as:

- a prefetch hint;
- a coarse filter;
- a speculative approximation whose result is not committed until verified;
- or a separately quality-gated model variant.

It must not silently replace the exact production expert weights.

## 7. Repeated control/decode transforms

Small finite-state transforms with high reuse are natural LUT candidates even when weight-vector dictionaries fail. Examples include compact unpack/decode state, quantization metadata transforms, routing metadata transforms, and format-conversion helpers.

On the current CPU path, predecoded META already demonstrates that spending a small amount of extra RAM to remove repeated decode work can win. A LUT is one implementation option where the state space is genuinely small; predecode/layout expansion is another.

---

# Selection rule for future LUT/codebook experiments

Before building another lookup benchmark, require at least one of these to be true:

```text
A. one lookup replaces >= ~32 useful scalar-equivalent operations, based on the V7 crossover evidence;
B. the lookup table is small enough to remain in the intended local storage level;
C. table-build cost is amortized across many rows/tokens/batch items;
D. the lookup is implemented in near-memory hardware with a materially different cost model from x86;
E. the codebook is used only for prediction/prefetch/compression and does not replace authoritative arithmetic.
```

If none is true, do not create another LUT version merely by changing table width or dictionary threshold.

# Current project decision

For the next benchmark:

```text
V29 = V27 META kernel
    + NUMA-local allocation
    + NUMA-local persistent worker pools
    + disjoint output-row sharding across the two sockets
    + duplicated activation
    + local/remote controls
    + exact correctness gate
```

LUT/codebook research remains active as a **separate lever** and should be applied only where measured reuse/amortization/locality or safe speculative use exists.