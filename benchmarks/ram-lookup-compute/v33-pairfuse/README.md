# TRANSIT V33 PAIRFUSE

Date: 2026-09-16
Status: runnable exact-Q4_K benchmark; laptop/server measurement required for promotion
Base: V32 STABILITYFORGE

## Why V33 exists

V32 established a repeatable exact win for the FSA family on the pinned real Q4_K tensor. The strongest stable laptop result was V31_FSA_RB16 at four P-cores (paired median 1.26162x, p10 1.10758x), while the 8/12-thread mixed P+E runs did not generalize. V33 therefore does not add another speculative large LUT. It removes redundant work inside the FSA arithmetic path.

## Main new mechanism: pair-fused Q4 nibble groups

In Q4_K, groups 0/1, 2/3, 4/5, and 6/7 share the same packed 32 Q4 bytes: one group is the low nibble and the other is the high nibble. The V31 FSA path processed each group separately, so it reloaded those same packed bytes twice.

V33 fuses each even/odd group pair:

1. load the packed 32 Q4 bytes once;
2. derive low and high nibbles once;
3. multiply low nibbles against activation group `g`;
4. multiply high nibbles against activation group `g+1`;
5. apply both exact Q4_K scales and accumulate.

This halves packed-Q load operations for the paired group path without expanding the Q4 payload.

## Other exact changes

- `V33_FSA4_VR`: vectorized four-row min correction and packed four-row `PHADDD` horizontal reduction, but retains the old per-group Q loads. This isolates the reduction/min-vectorization benefit.
- `V33_PAIR4_VR`: pair-fused nibble groups + vectorized min correction + packed 4-row reduction. Four dot accumulators are kept live to avoid the severe register pressure of the old 16-row path.
- `V33_PAIR8_VR`: same pair-fusion over eight rows to test whether extra activation reuse outweighs register pressure.

All V33 candidates must match V27 exactly in the harness (`diff=0`). No approximation or new quantization is introduced.

## Benchmark methodology

V33 preserves V32's alternating AB/BA paired measurement and adds a stronger beyond-LLC control:

- `HOT`: repeated same-address representation;
- `ROTATE`: address-distinct copies of both V27 META and X8F/FSA representations are rotated between measurements, so the benchmark walks a large working set instead of touching a synthetic eviction buffer before every timing.

Default rotating copies:
- 16 on small-RAM machines;
- 32 on >=64 GB;
- 64 on >=512 GB.

Promotion gate:

```text
exactness diff == 0
AND paired median speedup > 1.0
AND p10 > 1.0
AND win_rate >= 0.80
```

## Hybrid laptop vs target server

The i5-12500H reports four CPUID core-type `0x40` cores (P-cores) and eight `0x20` cores (E-cores). V33 therefore uses 1/2/4 physical cores as the promotion sweep on that laptop. Mixed 8/12 P+E runs are optional diagnostics only.

On a homogeneous E5-2680 v2 socket, the runner sweeps up to 10 physical cores. Physical server results remain authoritative for production scaling.

## Development sanity only

A strict-Ivy development build (`-march=ivybridge -mssse3 -mavx -mno-avx2 -mno-fma`) passes exactness for all V33 paths. On the synthetic 9216 x 2560 geometry, the pair-fused kernels showed a meaningful advantage over V27 in both HOT and rotating-address modes. These development numbers are mechanism checks, not ByteMyWave production evidence; the pinned real Qwen tensor on the user's machine is the next authority.

## External context

Current llama.cpp itself uses multi-row repacked Q4_K layouts (`block_q4_Kx8`, `block_q4_Kx16`), reinforcing the general direction that row-interleaved/repacked low-bit execution is a useful CPU strategy. BitNet's 2026 CPU update likewise emphasizes configurable tiling and parallel kernel autotuning. V33 adapts those broad lessons to the exact strict-Ivy Q4_K x Q8_K contract rather than copying their ISA/model assumptions.

## Files

- `v33_pairfuse.cpp` - exact benchmark and kernels
- `RUN.ps1` - pinned-model Windows runner
- `README.md` - this experiment contract

V27 META is never deleted or replaced.
