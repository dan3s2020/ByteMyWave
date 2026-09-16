# V31 research notes

## Measured evidence driving this revision

The prior real-Qwen V30 run on the i5-12500H showed the first exact single-token win over the V27 baseline: V30 META_X8F reached 1.15339x at 12 threads and PACK8 1.12015x. The win appeared only at higher thread count, while V27 stopped scaling from 8T to 12T. That points to loop order, activation reuse, reductions, and load pressure rather than a need for larger random LUTs.

The same run showed VEC8 META above 2x versus eight serial calls in some thread configurations, so multi-activation reuse remains a distinct high-value path.

SUM4 and the exact base+residual family were negative. The measured residual density (50.717%) is too high to justify sparse exact correction on this tensor.

## External mechanisms incorporated

### Microsoft BitNet CPU optimization

Current BitNet CPU work exposes configurable row block, column block, and parallelism, and distinguishes weight-parallel from activation-parallel kernels. Their published tuning sweep includes row blocks 2/4/8/16/32 and shows that the optimum is hardware/workload dependent.

Source: https://github.com/microsoft/BitNet/blob/main/src/README.md

V31 adapts the principle, not the BitNet quantization format: exact Q4_K remains authoritative and row-block 2/4/8/16 is swept inside our strict-Ivy kernel.

### Windows physical-core topology

Windows `GetLogicalProcessorInformationEx(RelationProcessorCore)` returns one relationship record per physical core. `PROCESSOR_RELATIONSHIP.EfficiencyClass` is non-zero on heterogeneous systems; higher values are more performance-oriented.

Sources:
- https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getlogicalprocessorinformationex
- https://learn.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-processor_relationship

V31 uses one logical processor per core and sorts higher EfficiencyClass first, avoiding the V30 ambiguity where the first N logical CPUs could include SMT siblings.

## Main new arithmetic transformation: fused-scale accumulation

V27-like group work is conceptually:

```
pair16 = PMADDUBSW(q4, q8)
partial32 = PMADDWD(pair16, 1)
sg = horizontal_sum(partial32)
sumi += scale * sg
```

For Q4_K, q is 0..15 and q8 is -127..127, so the two-product `PMADDUBSW` pair sum does not approach int16 saturation. Because the group scale is constant for all 32 values in the group, V31 instead performs:

```
pair16 = PMADDUBSW(q4, q8)
scaled32 = PMADDWD(pair16, scale)
acc32 += scaled32
...
sumi = horizontal_sum(acc32)   // once after all groups
```

This is integer-distributive and preserves the same exact Q4_K x Q8_K result in the harness. It removes repeated horizontal reductions and scalar/vector multiply stages from the group loop.

## Space-for-time variants

`EXP8` expands each 4-bit q to an 8-bit value offline. It doubles the packed-Q payload but removes nibble extraction.

`SCALED16` stores `q * group_scale` directly as int16. This consumes about four times the packed-Q payload but removes both nibble decode and runtime scale multiplication. It is intentionally a RAM-heavy candidate, suitable for the project's 200-300+ GB compiled-layout budget if and only if cold/DDR measurements justify it.

## Development sanity only

The V31 source was compiled with:

`-O3 -std=c++17 -march=ivybridge -mssse3 -mavx -mno-avx2 -mno-fma`

On a local synthetic 9216x2560 Q4_K geometry, every V31 candidate passed diff=0 against V27. FSA produced clear wins in both hot and cache-evicted controls in this sanity environment. Those timings are not transferred as claims about the user's i5-12500H or E5-2680 v2; the supplied runner is the decision gate.
