# Transit V31 research after the first V30 real-Qwen run

Date: 2026-09-16  
Input evidence: user-run V30 RAMFORGE FIX2 on the same pinned Qwen GGUF blob and `blk.0.ffn_gate.weight` used by the prior ByteMyWave Q4_K line.  
Status: design/experiment plan; no V31 performance result is claimed here.

## Measured V30 facts that constrain V31

The real tensor was `9216 x 2560`, 12.6562 MiB original Q4_K bytes, all promoted candidates were exact (`max_abs_diff = 0`, cosine = 1.0).

Single-token same-run results:

- 1T: V27 META 1.561 ms; META_X8F 1.986 ms (0.786x); PACK8 1.772 ms (0.881x).
- 4T: V27 META 0.703 ms; META_X8F 0.718 ms (0.980x).
- 8T: V27 META 0.376 ms; META_X8F 0.393 ms (0.957x).
- 12T: V27 META 0.393 ms; **META_X8F 0.340 ms (1.153x)**; **PACK8 0.350 ms (1.120x)**.

Thus V30 has already found an exact path that beats the same-run V27 baseline, but only in one observed high-thread regime on the Alder-Lake laptop. This is not yet an E5-2680-v2 result.

The shape of the curve matters more than the headline number: V27 improves through 8T then regresses at 12T, while X8F continues improving. V31 must therefore investigate shared-activation loads, row blocking, scheduling/affinity, cache-line layout, and per-thread working-set pressure rather than adding another arbitrary LUT.

Multi-token results are independently strong:

- VEC8 META: 1.473x (1T), 1.343x (4T), **2.176x (8T)**, **2.048x (12T)** versus serial V27 x8.
- VEC4 META: 1.764x (1T), 1.162x (4T), 1.330x (8T), 0.776x (12T).

Therefore VEC-N is retained as a speculative-verification/prefill path, not as a batch=1 claim.

Negative results are also decisive:

- SUM4 is ~0.17-0.27x V27: do not spend the next batch=1 iteration on more software PSHUFB subset-LUT width changes.
- Exact base+residual has ~50.717% residual density; only 0.159% of groups have <=8 residuals and only 51.379% have <=16. The tested exact QTEA/FluxBin-inspired residual structure is too dense for the current hot loop.
- COL8 is below V27 at all tested thread counts.

## External research read for V31

### Microsoft BitNet CPU optimization (2026)

Source: https://github.com/microsoft/BitNet/blob/main/src/README.md

The 2026 BitNet CPU update adds parallel weight/activation kernels and explicitly autotunes `ROW_BLOCK_SIZE`, `COL_BLOCK_SIZE`, and `PARALLEL_SIZE`. Their documented sweep includes row blocks 2/4/8/16/32 and column blocks 32..1024, and they report an additional 1.15x-2.1x improvement over the earlier implementation depending on platform/workload.

Direct ByteMyWave implication: an x8 layout must not be frozen just because x8 won once. V31 must sweep multiple row-block geometries and K/column tiles on the exact Q4_K tensor.

BitNet's implementation also explicitly shares activation work across multiple weight rows. This matches the mechanism exposed by META_X8F: group-major processing reduces repeated activation loads compared with the row-major V27 contract reconstruction.

### Vec-LUT / vlut.cpp (MobiSys 2026)

Sources:

- https://arxiv.org/abs/2512.06443
- https://github.com/OpenBitSys/vlut.cpp

Vec-LUT turns repeated scalar lookup into a 1-to-N vector operation and adds vector-LUT-centric layout plus cache-aware streamed lookup. V30's measured VEC8 result independently supports the same general reuse direction on the ByteMyWave tensor. V31 should sweep N=2/4/8/16 and select by speculative window/batch size; it must not mix those numbers into batch=1 decode claims.

### T-MAC / T-MAN

Sources:

- https://github.com/microsoft/T-MAC
- https://arxiv.org/abs/2511.11248

T-MAC uses tiny table lookup (`pshuf`/`tbl`) for low-bit arithmetic. Its own documentation warns that old x86 CPUs can have poor memory-bandwidth behavior, especially for 4-bit token generation. T-MAN adds two-level tables and tiling on NPUs. These remain useful design references, but V30 SUM4 is direct project evidence that another software LUT-width sweep is not currently the highest-value batch=1 experiment.

### Current llama.cpp SSSE3 path

Source: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/arch/x86/quants.c

Current llama.cpp retains explicit SSSE3 implementations and software prefetch in old-x86 quant paths. V31 should therefore benchmark prefetch distance/hint rather than assuming hardware prefetch is sufficient on E5-2680 v2.

### Ivy Bridge instruction economics

Sources:

- https://uops.info/html-instr/VPMADDUBSW_XMM_XMM_XMM.html
- https://uops.info/html-instr/PSHUFB_XMM_XMM.html

On Ivy Bridge, 128-bit VPMADDUBSW is measured at about one instruction/cycle throughput with ~5-cycle dependency latency, while PSHUFB is high-throughput. The V30 SUM4 loss therefore cannot be blamed on PSHUFB being intrinsically slow; it is the total lookup/build/address/instruction structure. V31 should expose more independent PMADDUBSW chains and remove horizontal-reduction/decode work around the existing direct arithmetic.

## Important new kernel idea: fuse scale into PMADDWD and defer horizontal reduction

The current direct dot helper conceptually does, for every 32-weight group:

```text
PMADDUBSW(q, a)
PMADDWD(pair_sums, ones)
horizontal_sum
scalar/group scale multiply
```

For Q4_K, the group scale is constant for the whole 32-weight group. Therefore an exact alternative is:

```text
PMADDUBSW(q, a)
PMADDWD(pair_sums, broadcast(scale))
accumulate 4 x int32 lanes across all 8 groups
horizontal_sum ONCE at the end
```

This is algebraically exact in integer arithmetic within the existing ranges. It removes most per-group horizontal reductions and removes the separate `scale * sg` operation. It needs no lossy representation and can use the same logical Q4 bytes/META information.

V31 must test at least two register-pressure variants:

- FSA4: four output rows live at once; activation vectors are reused across four rows.
- FSA8: eight rows live at once; maximum activation reuse, but may spill on x86-64's 16 XMM registers.

This is the highest-priority unexplored batch=1 kernel after the V30 result.

## Required V31 ablations: explain X8F before adding complexity

X8F changed several things at once. V31 must separate them:

1. `META_PAD64`: V27 semantics with each tile explicitly 64-byte aligned/padded only.
2. `META_F32`: predecode only `d/dmin` FP16 -> FP32; keep row-major metadata and V27 loop order.
3. `META_T`: transpose scale/min metadata and use group-major shared-activation order, but keep FP16 d/dmin.
4. `META_X8F64`: current full X8F with explicit 64-byte base alignment.

This tells us whether the 12T win comes primarily from activation-load reuse, metadata transpose, FP16 removal, cache-line stride/alignment, or their interaction.

## Space-for-time representations worth testing next

### EXP8 exact expanded Q4

Store each Q4 value as one unsigned byte after offline compilation. This doubles the nibble payload footprint but removes low/high nibble mask/shift from the hot path. Crucially, the current compact kernel already reloads a 32-byte packed region for low- and high-nibble groups; EXP8 may increase unique DRAM bytes but does not necessarily double the number of hot-loop load instructions. It must be tested separately in hot-cache and DRAM-streaming modes.

### SCALED16 exact compiled weights

Store `scale * q` as int16 per weight. This is much larger but removes nibble decode and group-scale multiplication and allows integer lane accumulation across groups. It is explicitly a RAM-for-compute experiment for hot tensors/experts, not a default whole-model format. META-Atlas keeps V27/X8F for cold/streaming weights if bandwidth loses.

Do not assume these win: they trade instruction count for more memory traffic. The point of the 200-300 GB budget is to permit multiple compiled representations, not to force the largest representation into every layer.

## Benchmark methodology corrections for V31

### 1. Hot-cache and DRAM-streaming must be separate

The tested tensor is only 12.6562 MiB. Repeated warm runs can be heavily LLC-resident on the laptop, and a row shard is also small relative to an E5-2680-v2 socket's LLC. A real large model does not repeatedly execute only one tensor forever.

V31 therefore needs:

- HOT mode: current repeated-tensor timing, useful for kernel/uop analysis.
- ROTATING mode: enough identical-but-distinct compiled copies (or multiple real tensors) to exceed LLC by a large margin, rotating copies between iterations. Correct output is unchanged, but cache residency is destroyed.
- Report effective representation bytes/GEMV and effective GB/s alongside ms.

Recommended laptop gate: >=256-512 MiB rotating working set. Server gate: >=512 MiB-2 GiB per NUMA socket if free RAM allows.

### 2. Physical-core pinning, not logical-CPU-number assumptions

The i5-12500H is heterogeneous. A 12-thread result can mix P cores, E cores and SMT depending on Windows logical numbering. V31 must enumerate physical cores with `GetLogicalProcessorInformationEx(RelationProcessorCore)`, choose one logical processor per physical core, record EfficiencyClass, and provide separate physical-core and SMT sweeps.

The E5-2680-v2 target is homogeneous, but correct physical-core enumeration is still needed for fair 10-core/socket testing.

### 3. Real NUMA placement

The next dual-socket run must make memory locality explicit. Building a `std::vector` on an arbitrarily scheduled main thread is not enough evidence of NUMA placement. Allocate/touch each representation while pinned to its owner node (or use Windows NUMA-aware allocation APIs), then measure local and remote controls before the dual row-sharded run.

## V31 ordered experiment queue

Run in this order; stop promoting losers but preserve their results:

1. V27 baseline, current X8F, PACK8.
2. X8F ablations: PAD64 / F32 / metadata-transpose+group-major / explicit 64B aligned X8F.
3. FSA4 and FSA8 fused-scale deferred-horizontal-reduction kernels.
4. Row block autotune: 2/4/8/16; K/column tile sweep where applicable.
5. Prefetch: off, T0 distance 1/2/4 tiles, and any architecture-safe alternatives.
6. EXP8 exact expanded Q4.
7. SCALED16 only if HOT mode remains compute/uop limited.
8. Repeat finalists in ROTATING/beyond-LLC mode.
9. VEC N=2/4/8/16 multi-token sweep.
10. Physical E5-2680-v2: 1/4/8/10 physical cores per socket, local/remote NUMA controls, then two sockets.

## Promotion policy

V27 is never deleted.

A V31 representation/kernel can become an Atlas choice only when:

```text
exactness passes
AND same pinned tensor/model
AND same-run comparison
AND relevant regime wins (HOT, ROTATING/DRAM, or VEC-N)
AND the regime is recorded in the Atlas
```

It is acceptable — and expected — for the final system to keep multiple copies:

```text
cold/streaming tensor -> compact V27 or X8F
hot tensor/expert     -> EXP8 or SCALED16 if measured faster
batch=1               -> best exact single-token kernel
speculative N=8       -> VEC8 layout/kernel
socket 0 / socket 1   -> NUMA-local copies
SSD                    -> persisted compiled forms, not random-lookups in the hot loop
```

The 200-300 GB RAM/SSD budget is therefore used as a compiled-model cache/atlas, with performance-selected representations per tensor/regime rather than as one giant universal LUT.
