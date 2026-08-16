# 08 — Evidence and Benchmarks

This file separates **measured facts** from architectural inference. Numbers here are copied from the actual experiments that led to the Transit design.

## 1. Test host

The main host-side experiments were run on a laptop with:

- Intel Core i5-12500H;
- 12 cores / 16 threads;
- AVX2 support, no AVX-512;
- DDR5 system memory;
- NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4 GiB;
- NVMe SSD;
- Windows host for the original assembly/direct-I/O tests.

The CPU topology matters because the measured multicore result is not an abstract bandwidth number: the i5-12500H has a hybrid P-core/E-core design and the kernel used ordinary x64 scalar `POPCNT`, not AVX-512.

## 2. Original GGUF / SSD experiment

Script: `benchmarks/transit_ollama_ssd_test.py`

The script:

- discovered local Ollama GGUF models;
- selected a real Gemma 3 270M Q8_0 model;
- located a real tensor (`blk.0.ffn_down.weight`);
- used a 640×2048 test slice = 1,310,720 weights;
- dequantized the source tensor;
- requantized row-wise to signed INT4 `[-7, 7]`;
- stored the result both as packed Q4 and four one-bit planes;
- generated INT8 activations;
- verified exact integer equivalence between packed-Q4 and bitplane arithmetic;
- measured RAM and direct SSD paths.

Observed values:

```text
Ollama Gemma3 270m baseline     ~106.486 tok/s
weights in test tensor          1,310,720
packed Q4 size                  655,360 bytes = 0.625 MiB
4 bitplanes total               655,360 bytes = 0.625 MiB
logical storage                 4 bits/weight
rowwise Q4 relative RMSE        0.233455 vs original float tensor
SSD sequential probe            3.158 GB/s in the first large-stream probe
RAM int4 dot                    0.673 ms
RAM int4 throughput             1.948 Gweights/s
RAM 16-entry LUT path           14.594 ms = 89.8 Mweights/s
SSD packed Q4                   8.700 ms = 150.7 Mweights/s
SSD bitplanes in Python         16.725 ms = 78.4 Mweights/s
```

Interpretation:

- bitplanes do **not** require 4× the Q4 storage; four 1-bit planes are still 4 bits/weight total;
- the Python implementation was overhead dominated and not a hardware-speed estimate;
- the 16-entry activation lookup idea was decisively slow and rejected;
- the quantization error is from the deliberately simple row-wise INT4 conversion, not from the bitplane representation itself;
- storage delivery for 655,360 bytes at 3.158 GB/s is only about 0.208 ms in an ideal sequential read, much less than the Python compute overhead seen in that first script.

## 3. Assembly V1 — exact bitplane kernel

Files:

- `benchmarks/transit_bitplane_kernel.asm`
- `benchmarks/transit_asm_runner.py`
- `benchmarks/run_transit_asm.ps1`

Kernel characteristics:

- NASM x64 flat binary;
- Windows x64 ABI;
- fixed 640×2048 test shape in V1;
- scalar 64-bit `AND` + `POPCNT` + shifts/add/sub;
- no AVX2 in the actual V1 kernel;
- no BMI2 dependency;
- no general multiply in the matrix datapath.

Correctness:

```text
exact_int_accum = True
max_abs_diff    = 0
```

Measured:

```text
ASM best        0.1569 ms
ASM best        8.354 Gweights/s
ASM best        4.177 GB/s Q4-equivalent weight bytes
ASM median      0.1580 ms
ASM median      8.296 Gweights/s

SSD + ASM best  0.9172 ms
SSD + ASM       1.429 Gweights/s
SSD + ASM med   1.0841 ms
SSD + ASM med   1.209 Gweights/s
Direct-read med 0.9115 ms
Direct-read med 0.719 GB/s
ASM in pipeline 0.1696 ms median
```

What this proves:

- the integer bitplane identity was implemented correctly in native assembly;
- the CPU kernel can be much faster than the Python prototype;
- in that small direct-I/O pipeline, storage delivery dominated total latency.

What it does **not** prove:

- end-to-end K3 inference speed;
- MXFP4/MXFP8 numerical compatibility;
- FPGA speed;
- custom DDR3 tile speed.

## 4. Assembly V2 — multicore and direct NVMe pipeline

Script: `benchmarks/transit_asm_v2.py`

A roughly 2.5 GiB repeated stream was used so the direct-I/O path could be stressed independently of the small source tensor.

Correctness:

```text
exact=True
maxdiff=0
```

### A. Hot assembly compute

```text
workers   Gweights/s
1          7.892
2         14.907
4         26.759
8         38.671
16        53.697
```

At 16 workers:

```text
53.697 Gweights/s
26.849 GB/s Q4-equivalent weight bytes
```

### B. Direct NVMe read-only

```text
workers   GB/s
1         2.675
2         4.455
4         5.976
8         6.163
16        6.080
```

The SSD essentially saturated around 8 readers.

### C. NVMe + assembly pipeline

```text
workers   SSD GB/s   Gweights/s
1         1.451       2.903
2         2.871       5.742
4         5.285      10.571
8         6.151      12.301
16        6.159      12.318
```

Best pipeline result:

```text
12.318 Gweights/s
```

Interpretation:

- the CPU compute kernel can outrun the SSD;
- the pipeline can consume essentially the full physical SSD bandwidth;
- storage is transport only: `6.159 GB/s` should not be arithmetically added to `53.697 Gweights/s` as if both were independent compute engines.

## 5. DDR5 V3 — cache-busting 6 GiB working set

Script: `benchmarks/transit_ddr5_bench_v3_lowram.py`

A 6 GiB allocated/materialized working set was used specifically to avoid pretending cache bandwidth was DRAM bandwidth.

Observed memory state:

```text
available before allocation  ~10.33 GiB
allocated/materialized        6 GiB
available after allocation   ~4.316 GiB
9830 repeated blocks
12.884 billion weights represented
```

The repeated blocks do not represent unique model parameters; they exist to create a DRAM-scale benchmark stream.

### A. Raw DDR5 copy/read path

```text
workers   GB/s
1          9.083
2         17.139
4         37.665
8         33.949
16        50.015
```

### B. DDR5 → V3 bitplane engine

```text
workers   Gweights/s   Q4-equivalent GB/s
1          6.179        3.089
2         10.710        5.355
4         26.064       13.032
8         37.339       18.670
16        53.673       26.836
```

Correctness:

```text
exact=True
maxdiff=0
```

Interpretation:

- this is actual DRAM-scale traffic, not a cache-resident microbenchmark;
- the raw memory path reached about 50 GB/s;
- the V3 kernel consumed about 26.8 GB/s of Q4-equivalent weight bytes;
- V3 therefore used roughly 53.7% of the measured raw DRAM bandwidth at the best point;
- the near-identical ~53.7 Gweights/s result in hot V2 and DRAM-scale V3 suggests the scalar `POPCNT` execution side became the main CPU-side limit before raw DDR5 bandwidth was exhausted.

## 6. V4.1 — masked activation sum

Script: `benchmarks/transit_maskedsum_v4_1_fixed.py`

An initial V4 flat-binary build accidentally executed the helper function at byte 0 instead of the intended kernel entry. That was diagnosed and fixed by placing an entry jump at byte 0. The corrected V4.1 is the only V4 result that should be used.

Correctness after fix:

```text
V3 exact=True  maxdiff=0
V4 exact=True  maxdiff=0
```

### V3 on the same 6 GiB test

```text
workers   Gweights/s
1          5.946
2         11.263
4         24.978
8         35.855
16        56.891
```

At 16 workers:

```text
56.891 Gweights/s
28.446 GB/s Q4-equivalent weights
```

### V4 masked-sum

```text
workers   Gweights/s
1          5.789
2         10.019
4         20.183
8         30.706
16        39.982
```

At 16 workers:

```text
39.982 Gweights/s
19.991 GB/s Q4-equivalent weights
V4/V3 speed ratio = 0.702777×
```

Activation preprocessing:

```text
x -> x+128 prep best    3.1 us
median                  3.3 us
```

Decision:

> V4 is mathematically valid but about 30% slower than V3 on this CPU. Reject it as the host CPU implementation. Keep the masking/gating idea for custom hardware research.

## 7. Final host overlap benchmark

Script: `benchmarks/transit_final_host_overlap.py`

The purpose was to test whether CPU+DDR5 work, SSD prefetch and GPU activity could overlap on the same laptop rather than benchmark each subsystem in isolation forever.

Correctness:

```text
exact=True
maxdiff=0
```

GPU:

```text
NVIDIA GeForce RTX 3050 Ti Laptop GPU
4096 MiB
Driver 592.82 during the test
```

Small Ollama Gemma3 270m warmup:

```text
13 tokens
242.19 tok/s
```

### A. GPU standalone

```text
eval_count      54
eval_duration   0.2203 s
245.148 tok/s
```

### B. CPU + DDR5 standalone

```text
56.451 Gweights/s
28.225 GB/s Q4-equivalent weights
```

### C. SSD standalone

```text
6.153 GB/s
12.306 Gweights/s Q4 transport equivalent
```

Again, the latter is a **transport equivalence**, not SSD computation.

### D. Simultaneous

```text
CPU             51.385 Gweights/s
SSD              5.787 GB/s
GPU            127.822 tok/s on the tiny Gemma run
GPU util avg     23.8%
GPU util max     95%
VRAM avg        592 MiB
```

Retention relative to isolated runs:

```text
CPU  0.910259
SSD  0.940512
GPU  0.521409
```

Interpretation:

- CPU+DDR5 and SSD retain roughly 91% and 94% of their isolated rates, so these two pipelines overlap well;
- the laptop GPU loses much more, likely because CPU and GPU share a mobile power/thermal envelope;
- the GPU measurement is short and noisy: only 54 evaluation tokens / ~0.22 s in the standalone result;
- the 52% GPU retention must not be treated as a universal Transit architecture coefficient.

This was deliberately the **final PC overlap benchmark**. The next meaningful measurement should be on physical Transit hardware or a real end-to-end K3 path.

## 8. Crude K3-equivalent translation

Using the working approximation of `104 billion active weights/token`:

```text
56.451 Gweights/s / 104e9 ≈ 0.543 weight-path tokens/s
51.385 Gweights/s / 104e9 ≈ 0.494 weight-path tokens/s
```

These are **weight-path roofline equivalents only**. They are not complete K3 token-rate predictions because they ignore or simplify:

- attention;
- router cost;
- KV cache traffic;
- expert aggregation;
- MXFP scaling/decoding;
- communication;
- synchronization;
- non-expert weights;
- batching effects.

Similarly:

```text
5.787 GB/s SSD
≈ 11.574 Gweights/s at Q4 packing
≈ 0.111 active-weight-token/s transport equivalent
```

This is storage transport, not compute.

## 9. Evidence summary

### Proven

- four bitplanes can represent signed INT4 weights with the same 4 bits/weight storage as packed Q4;
- the signed INT4×INT8 bitplane formula is exact;
- the native assembly implementation is exact;
- a commodity laptop CPU can sustain ~54–57 Gweights/s on the tested bitplane kernel;
- the result survives a 6 GiB DRAM working set;
- the tested NVMe can sustain ~6.1 GB/s direct stream;
- CPU+DDR5 and SSD I/O can overlap strongly;
- V4 masked-sum is slower than V3 on this CPU.

### Not proven yet

- K3 end-to-end execution;
- K3 numerical quality after any Transit-specific quantization;
- exact MXFP4/MXFP8 bitplane kernel;
- YPCB sustained DDR3+FPGA kernel throughput;
- an eight-channel DDR3 Transit tile;
- 38-tile PCIe enumeration/power/network behavior;
- 100 tok/s K3.

The project should never silently promote an item from the second list into the first.
