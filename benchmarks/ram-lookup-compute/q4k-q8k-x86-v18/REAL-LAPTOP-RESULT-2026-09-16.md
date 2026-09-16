# Transit V18 — real laptop result — 2026-09-16

Tensor: `blk.0.ffn_gate.weight`

- GEMV: 9216 rows x 2560 cols
- logical weights: 23,592,960
- Q4_K: 12.656 MiB
- runtime x16 layout: 14,008,320 bytes, +5.556%
- compiled path: AVX2 + F16C/FMA + AVX-VNNI (`vpdpbusd`)

Correctness remained effectively exact relative to the scalar Q4_K x Q8_K operator:

- rel-L2: 2.139e-07
- cosine: 1.000000000
- max abs: 4.172325134e-07

Key timings from the user's Windows laptop:

- V15 x8meta: 0.813 ms
- best V17 single: 0.516 ms
- V17 persistent 8T: 0.081 ms
- V18 best: 0.064 ms
- V18 best config: 11 threads, dynamic-physical, PF0, chunk 4
- V18 logical throughput: 370.96 GOP/s
- V18 / V15: 12.78x (includes threading)

Important interpretation: the 0.064 ms result is cache-hot. The V18 runtime layout is roughly 13.36 MiB, so 0.064 ms corresponds to more than 200 GB/s of weight traffic before counting other traffic. Therefore V18 establishes a strong hot-cache/scheduler/kernel result, but not full-model DDR throughput.

The next falsifiable gate is V19 streaming/cold-weight testing with a 512 MiB–1 GiB rotating working set, preserving the same exact AVX-VNNI kernel while forcing weight data to stream through LLC/DRAM.
