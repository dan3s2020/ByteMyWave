# Real Qwen Q4_K exact-LUT raw results — 2026-09-16

## Hardware inventory captured after the run

```text
CPU
Name                      : 12th Gen Intel(R) Core(TM) i5-12500H
NumberOfCores             : 12
NumberOfLogicalProcessors : 16
MaxClockSpeed             : 2500

RAM module 0
Manufacturer          : Samsung
PartNumber            : M425R1GB4BB0-CQKOL
CapacityGB            : 8
Speed                 : 4800
ConfiguredClockSpeed  : 4800

RAM module 1
Manufacturer          : Samsung
PartNumber            : M425R1GB4BB0-CQKOL
CapacityGB            : 8
Speed                 : 4800
ConfiguredClockSpeed  : 4800

Compiler discovered
Name   : g++.exe
Source : C:\Strawberry\c\bin\g++.exe
```

## Position-specific product quantization

```text
================================================================================
 REAL QWEN - POSITION-SPECIFIC PRODUCT QUANTIZATION LUT
================================================================================
Tensor              : blk.0.ffn_gate.weight
Shape               : 2,560 x 9,216
Weights             : 23,592,960
Original Q4_K       : 12.656 MiB

FP32 NumPy baseline : 2.229 ms/GEMV

================================================================================
 POSITION-SPECIFIC PQ -- GROUP 16
================================================================================
Subspaces           : 576
Vectors/codebook    : 2,560
Centroids/codebook  : 256
Training total      : 144.245 s

Weight rel-L2       : 0.673526
Weight cosine       : 0.739166
Weight SNR          : 3.43 dB

Output cosine mean  : 0.738135
Output cosine worst : 0.724402
Output rel-L2 mean  : 0.674645
Output rel-L2 worst : 0.689466

Codes               : 1.406 MiB
FP16 codebooks      : 4.500 MiB
Total PQ storage    : 5.906 MiB
Compression/Q4_K    : 2.14x

LUT-build MACs      : 2,359,296
Runtime lookups     : 1,474,560
Heavy MAC reduction : 10.00x
Dynamic LUT size    : 576.00 KiB

PQ-LUT GEMV         : 6.645 ms
Speed vs FP32       : 0.34x
Checksum            : 14.087387

================================================================================
 POSITION-SPECIFIC PQ -- GROUP 8
================================================================================
Subspaces           : 1,152
Vectors/codebook    : 2,560
Centroids/codebook  : 256
Training total      : 337.935 s

Weight rel-L2       : 0.494713
Weight cosine       : 0.869056
Weight SNR          : 6.11 dB

Output cosine mean  : 0.868671
Output cosine worst : 0.863564
Output rel-L2 mean  : 0.495419
Output rel-L2 worst : 0.504424

Codes               : 2.812 MiB
FP16 codebooks      : 4.500 MiB
Total PQ storage    : 7.312 MiB
Compression/Q4_K    : 1.73x

LUT-build MACs      : 2,359,296
Runtime lookups     : 2,949,120
Heavy MAC reduction : 10.00x
Dynamic LUT size    : 1152.00 KiB

PQ-LUT GEMV         : 19.371 ms
Speed vs FP32       : 0.12x
Checksum            : 10.697013
```

## Exact original-Q4_K LUT proof

```text
============================================================
 Q4_K EXACT LUT V10 - NO ADDITIONAL WEIGHT LOSS
============================================================

Rows                     : 2,560
Columns                  : 9,216
Logical weights/GEMV     : 23,592,960
Original Q4_K bytes      : 13,271,040
Original Q4_K MiB        : 12.656
Nibble LUT               : 0.563 MiB
Byte-pair LUT            : 9.000 MiB

DIRECT vs Python reference
  max abs diff           : 1.90734863E-06
  relative L2            : 6.02043245E-07
  cosine                 : 1

NIBBLE LUT vs DIRECT
  max abs diff           : 0
  relative L2            : 0
  cosine                 : 1

BYTE LUT vs DIRECT
  max abs diff           : 0
  relative L2            : 0
  cosine                 : 1

------------------------------------------------------------
 CUSTOM KERNEL BENCHMARK
------------------------------------------------------------

DIRECT original Q4_K     : 16.344 ms/GEMV
NIBBLE LUT exact         : 15.223 ms/GEMV
BYTE LUT exact           : 47.505 ms/GEMV

Nibble LUT / direct      : 1.074x
Byte LUT / direct        : 0.344x

NO centroid/codebook approximation.
Original Q4_K bytes are consumed directly.
============================================================
```

## Compilation issues encountered before the successful run

```text
1. Add-Type -CompilerOptions was rejected by Windows PowerShell.
2. CompilerParameters with /unsafe /optimize+ compiled further but initially failed to resolve Stopwatch.
3. The assembly containing System.Diagnostics.Stopwatch was added to CompilerParameters.ReferencedAssemblies.
4. Only after successful type compilation was Q4KExactLutV10.Run benchmarked.
```

These setup failures are not benchmark results and are preserved only to make the interactive chronology reproducible.
