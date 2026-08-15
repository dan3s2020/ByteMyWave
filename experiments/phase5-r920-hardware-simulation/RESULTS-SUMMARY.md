# Phase 5 result summary

## Reference conclusion

Under the explicit analytical assumptions used by the current Feasibility Map style of reasoning:

```text
70B dense reference
Q4 v1 = 0.625 B/parameter
12 GB/s effective H2D
10 TFLOP/s effective dense-linear compute
0 persistent residency
```

TensorWave predicts:

```text
M_cross ~= 260.4
M=1   -> 99.6% starvation lower bound
M=256 -> 1.7% starvation lower bound / near-balanced
M=512 -> compute-bound in the ideal model
```

The discrete-event simulation of the actual Phase-3 two-slot ownership schedule produces the same qualitative crossover and gives **1.64% steady starvation at M=256** for the representative `K=8192,N=256` tile geometry.

## Most useful observation

At that geometry the current Phase-3 fixed working set is only:

```text
M=256 -> 10.75 MiB
M=512 -> 15.00 MiB
```

so a 12 GiB RTX 3060 leaves most VRAM available for future persistent compressed residency/cache.

An 8 GiB compressed-cache sensitivity example for the 70B Q4 reference reduces streamed bytes from 43.75 GB to about 35.16 GB and moves the ideal crossover from about M=260 to M=209.

## Hardware conclusion

The R920 is a strong **RAM/NUMA/PCIe research host** for TensorWave but is not a native multi-GeForce chassis. The project should validate one RTX 3060 physically and electrically before treating additional consumer GPUs as a deployment plan.

## Next measurements

```text
local NUMA pinned H2D
remote NUMA pinned H2D
copy+compute overlap
real dequant time
effective GEMM TFLOP/s
cache residency benefit
concurrent one-GPU-per-NUMA-worker scaling
```

All analytical timing values in Phase 5 must yield to those measurements when available.
