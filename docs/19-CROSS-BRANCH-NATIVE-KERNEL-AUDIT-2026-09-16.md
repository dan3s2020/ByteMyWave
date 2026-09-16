# Cross-branch audit — native Q4_K kernel, Transit, heterogeneous MoE, K3, GLM and agent stack

Date: 2026-09-16

Status: audit / integration note. This document compares measured results and active research tracks. It deliberately separates subsystem measurements from analytical rooflines and end-to-end token/s.

## Sources inspected

Required current-state surfaces:

```text
AGENTS.md
CURRENT-ARCHITECTURE.md
README.md
docs/REPOSITORY-MAP.md
PR #10 transit-ddr3-architecture
PR #11 docs/kimi-k3-ddr-cluster
PR #9  research/heterogeneous-moe-kimi-v1
```

Additional relevant work inspected:

```text
PR #16 research/ram-lookup-compute-2026-09-16
PR #12 agent/transit-ddr2-server-fabric
PR #13 agent/apu-ddr3-uma-k3
PR #14 agent/transit-active-memory-chronicles
agent/transit-memory-controller-trade-study
research/transit-cosoldex-dspark-2026-09-09
research/glm52-4x2-optimized-runtime-2026-08-18
research/k3-consumer-gpu-vs-transit-2026-08-17
```

## 1. What the new native kernel changes

PR #16 ended at the correct next gate: native optimized `Q4_K × Q8_K`, thread pinning, multi-row reuse, server ISA port and real llama.cpp integration.

The follow-up measurements now establish on the i5-12500H:

```text
reproduced llama.cpp direct Q4_K×Q8_K baseline
+
prepared Q8 sum32 metadata
+
shared activation loads across 4 output rows
+
explicit register-shaped 4-row kernel
=
measured ~15–26% direct-kernel speedup depending on run/method
```

The most conservative streaming result to carry into planning is:

```text
V10 STREAM 101 MiB
median speedup 1.1809x
IQR [1.1504, 1.2031]
```

The strongest later row-major streaming result was:

```text
V11
median speedup 1.2640x
IQR [1.2499, 1.2723]
```

Because CPU frequency and baseline timing move between runs, this audit treats the result as:

> robust direct-kernel gain on this host, approximately 15–26%, not yet an end-to-end model gain.

## 2. Comparison with the older Transit host bitplane kernel

The Transit DDR3 evidence branch measured the earlier signed-INT4×INT8 bitplane engine on the same laptop class:

```text
single-worker DRAM-scale bitplane: roughly 6 Gweights/s
16-worker DRAM-scale bitplane:    roughly 54–57 Gweights/s
raw DDR5 read path:               roughly 50 GB/s
```

The new row-major exact Q4_K path processes the 23,592,960-weight matrix in about:

```text
0.7986 ms in V10 streaming -> ~29.5 Gweights/s on one pinned fast logical CPU
0.7295 ms in V11 streaming -> ~32.3 Gweights/s on one pinned fast logical CPU
```

This is not an apples-to-apples format comparison:

```text
old bitplane = simple signed INT4 × INT8 proof representation
new kernel   = real llama-style Q4_K × Q8_K block-quantized arithmetic
```

However, it changes the host-side priority:

- for modern AVX2 CPUs running GGUF/Q4_K, the new exact SIMD path is much closer to the real inference runtime and should be the preferred host-kernel research path;
- the bitplane path remains valuable as the custom-hardware/FPGA reference because its arithmetic maps naturally to AND/POPCNT/reduction;
- for Ivy Bridge Xeons without AVX2, neither the AVX2 result nor the old laptop all-core bitplane number can be copied directly; the two candidate paths must be measured on the actual server.

## 3. Comparison with the Phase-6 AVX-only CPU-expert kernel

The heterogeneous MoE branch already contains `bench_cpu_expert_q4.cpp` for an old Xeon-class AVX1 target.

That benchmark intentionally uses:

```text
group-32 signed int4
float32 scale
0.625 B/weight
scalar nibble unpack
AVX1 float MAC
OpenMP
```

and explicitly labels itself a correctness-oriented baseline, not the final optimized kernel.

The new Q4_K work supplies concrete optimization ideas for its successor:

```text
precompute activation-side metadata once;
reuse one activation across multiple output rows;
tile output rows;
write the winning tile explicitly rather than through generic arrays;
pin one worker pool per NUMA socket;
benchmark row-major before introducing repacks;
use paired A/B/A timing and a DRAM-scale working set.
```

On E5-2680 v2 / E7-4890 v2 class CPUs, AVX2 instructions cannot be used. A dedicated AVX/SSE4.1/SSSE3/POPCNT design is required.

## 4. CPU versus memory-controller trade study

The memory-controller branch made an important architectural rule:

> compare maximum local read bandwidth with read+Transit-compute bandwidth. If compute barely reduces sustainable memory throughput, external compute offload is unnecessary; if it collapses throughput, dedicated local logic is justified.

The new Q4_K result makes this test more concrete.

On the i5, one pinned fast logical CPU already consumes roughly 15–17 GiB/s of Q4_K payload in the winning path. The next server measurement must determine:

```text
per-socket raw local DRAM GB/s
vs
per-socket Q4_K/Q8_K useful GB/s
vs
Gweights/s
vs
cycles/weight
```

Only this ratio tells us whether old CPU memory controllers plus custom kernels are an economical Transit tile substitute.

## 5. Dense models

For batch-1 dense decode, nearly all weight matrices are touched every token. A faster Q4_K GEMV can therefore accelerate a large fraction of CPU-only inference if those tensors use the matching quantization path.

But larger matrices also push the system toward the DRAM bandwidth roofline.

Consequences:

- the direct kernel gain can survive beyond LLC; V10 already proves that for ~101 MiB on one thread;
- the gain can shrink at multi-core scale if both baseline and optimized kernels saturate the same memory channels;
- model size by itself does not multiply the speedup;
- a full model that fits in GPU VRAM should normally stay GPU-resident rather than use this CPU path unnecessarily.

## 6. MoE models

MoE is a particularly good fit for the new kernel mechanics.

For one selected expert matrix, one activation vector is multiplied by many output rows. This is exactly the reuse pattern exploited by the 4-row kernel.

At the architecture level, PR #9/#10/#11 already agree on the broader rule:

```text
keep expert weights local;
dispatch small activations;
run selected experts in parallel;
return reduced outputs;
avoid moving expert weights over PCIe/network every token.
```

The native 4-row kernel is therefore useful at three levels:

1. CPU fallback for cold experts in the GPU-hot-cache architecture;
2. CPU-owned experts in the heterogeneous NUMA-local MoE architecture;
3. software reference/benchmark for deciding whether custom Transit memory-compute hardware is actually worth replacing the CPU path.

## 7. Kimi K2.5 implications

The Phase-6 branch derives, for four CPU sockets and current Q4 routed bytes:

```text
5 tok/s:
16.515 GB/s selected Q4 reads/socket
26.424 Gweights/s/socket

10 tok/s:
33.030 GB/s/socket
52.848 Gweights/s/socket
```

The new i5 single-fast-thread Q4_K result reaches approximately 29.5–32.3 Gweights/s on its test matrix.

This does **not** prove that an old Xeon socket reaches the K2.5 5 tok/s gate: different CPU ISA; different quantization layout; one tensor is not a full SwiGLU expert; multi-thread memory saturation is unknown; server NUMA and handoff cost are unknown.

It does show that the required `26.424 Gweights/s/socket` is no longer absurd as a kernel-scale number on modern AVX2 hardware. The physical server remains the gate.

## 8. Kimi K3 implications

The K3 Phase-6 model derives approximately:

```text
104B active parameters/token
~97.24B routed active parameters/token
~60.775 GB routed Q4 bytes/token after non-routed residency assumption
```

For four CPU sockets, current Q4 routed-expert requirements are roughly:

```text
5 tok/s:
~75.97 GB/s/socket
~243.1 GFLOP/s/socket

10 tok/s:
~151.94 GB/s/socket
~486.2 GFLOP/s/socket
```

A 15–26% CPU microkernel improvement does not bridge that gap on four old sockets.

For K3, the new kernel is an enabling optimization for CPU shards/fallback, not the whole answer.

The K3 architecture still needs one or more of: many independent memory domains; expert parallelism; GPU-resident hot experts; multiple independent GPU/H2D links; Transit local-memory compute tiles; lower-byte formats where quality allows; speculative decoding / accepted-token multiplication; routing-aware expert placement and replication.

## 9. GLM-5.2 / four-server 4x2 track

The GLM track selected:

```text
llama.cpp/GGUF
+ distributed expert scheduler
+ NUMA-local host expert store
+ GPU hot-expert cache
+ CPU cold-expert fallback
+ asynchronous prefetch
+ native MTP
+ MTP-aware expert prefetch
```

Its planning range is explicitly not measured on the purchased servers, but the selected implementation stack is compatible with the new kernel.

The new kernel can improve the **CPU cold-expert/fallback component**. It should not be multiplied by MTP, cache-hit and prefetch paper speedups as independent factors; those mechanisms overlap in the critical path.

## 10. Co-Sol-Dex, Active Memory and speculative decoding

These mechanisms attack different layers of cost:

```text
Co-Sol-Dex / Active Memory
    -> reduce prompt/context/retrieval work

native Q4_K kernel
    -> reduce cost of one CPU target forward path

GPU hot-expert cache / Transit local-memory compute
    -> remove or parallelize CPU weight work

MTP / speculative decoding
    -> reduce target-model forward passes per accepted output token
```

They can compose because they are not all the same optimization.

However, wall-clock gains must be measured end-to-end. A retrieval-context reduction does not directly multiply decode tok/s, and a speculative accepted-token factor does not make a memory-bound kernel faster.

## 11. Realistic end-to-end speedup translation

Let:

```text
S_kernel = direct-kernel speedup
f        = fraction of baseline token latency actually spent in the accelerated path
```

Then Amdahl's law gives:

```text
S_end_to_end = 1 / ((1-f) + f/S_kernel)
```

Using the measured direct-kernel range `S_kernel = 1.1809 to 1.2640`, representative end-to-end multipliers are:

```text
accelerated fraction f     with 1.1809x kernel    with 1.2640x kernel
30%                         1.048x                1.067x
50%                         1.083x                1.117x
70%                         1.120x                1.171x
80%                         1.140x                1.201x
90%                         1.160x                1.232x
100%                        1.181x                1.264x
```

Therefore the defensible current expectation is:

> if matching Q4_K GEMV work is 50–80% of end-to-end CPU decode time, this kernel alone is worth roughly 8–20% end-to-end tokens/s on an AVX2 CPU before memory-channel saturation.

This is an analytical translation, not a measured model result.

## 12. Combined optimization plan

The current workstreams should be composed in this order, because each stage exposes whether the next is still worth engineering:

```text
A. Exact model/tensor trace
   -> know which tensors/formats/experts dominate time and bytes

B. Production llama.cpp integration
   -> prove the 4-row Q4_K path beats the actual selected GEMV/repack path

C. Thread/NUMA scaling
   -> determine per-socket useful GB/s and Gweights/s before memory saturation

D. CPU/GPU expert split
   -> GPU keeps highest-value/hottest experts
   -> CPU runs cold/miss experts locally without blocking on refill

E. Expert-parallel placement
   -> selected experts execute across independent memory domains concurrently
   -> avoid pure serial layer-pipeline assumptions

F. Speculation/MTP
   -> prefetch the experts likely to be needed by drafted tokens
   -> measure accepted tokens per expensive target step

G. Co-Sol-Dex / Active Memory
   -> reduce executor context and agentic retrieval overhead
   -> keep this metric separate from raw decode tok/s

H. Transit tile path
   -> only replace CPU expert execution where measured local-memory compute economics justify it
```

## 13. Immediate measurements that maximize information value

Before another architecture-scale token/s claim, the highest-value tests are:

```text
1. current llama.cpp patched production Q4_K path + llama-bench;
2. 1/2/4 P-core Q4_K scaling on the laptop with >LLC working set;
3. exact E5-2680 v2 ISA inventory and AVX/SSE kernel benchmark per socket;
4. local vs remote NUMA Q4 expert benchmark on one server;
5. one real MoE expert gate/up/down benchmark, not one isolated matrix;
6. CPU expert + GPU hot-expert concurrent critical-path benchmark;
7. full model routing trace: active expert IDs, hit rate, bytes and per-layer timing;
8. only then convert to measured end-to-end tok/s.
```

## 14. Current decision

Promote the native exact 4-row Q4_K × Q8_K path to an active host-kernel candidate.

Do **not** promote the physical 4-row weight repack on the i5-12500H; it measured slightly slower than row-major.

Do **not** replace the Transit bitplane FPGA path with this CPU kernel; they solve different hardware targets.

Do **not** claim a K3 or GLM end-to-end token/s multiplier yet.

The best synthesis of the current repository is a heterogeneous hierarchy:

```text
Co-Sol-Dex / Active Memory
        |
        v
request/router/MTP
        |
        +--> hot experts -> GPU VRAM / GPU compute
        |
        +--> cold Q4_K experts -> NUMA-local optimized CPU kernel
        |
        +--> future high-bandwidth expert shards -> Transit local-memory tiles
        |
        v
local reductions -> small activation/result traffic -> next serial layer
```

That architecture uses each measured result where it is strongest instead of treating any one microbenchmark as the entire inference system.
