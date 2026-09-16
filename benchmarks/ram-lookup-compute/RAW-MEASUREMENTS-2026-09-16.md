# Raw/transcribed measurements — 2026-09-16

These blocks preserve the numeric console results from the interactive PowerShell session. They are transcribed from the run output, not regenerated after the fact.

The original console session did not record exact CPU/DIMM inventory; do not infer it from these results.

## V1 — one product per lookup

```text
Hidden size        : 256
Layers             : 4
Vocabulary         : 256
Parameters         : 393,216
Weight format      : Q4 packed (2 weights/byte)
Q4 weight RAM      : 0.19 MiB
RAM product LUT    : 64.00 MiB
LUT replicas       : 4,096
Prompt tokens      : 58
Generate tokens    : 128
Model/LUT build    : 0.020 s

CPU MULTIPLY VERSION
Prefill time       : 14.23 ms
Generation time    : 0.0396 s
Generation speed   : 3230.24 tokens/s

RAM LOOKUP VERSION
Prefill time       : 219.46 ms
Generation time    : 0.5590 s
Generation speed   : 228.96 tokens/s

IDENTICAL OUTPUT   : PASS
CPU multiply       : 3230.24 tok/s
RAM lookup         : 228.96 tok/s
CPU/RAM ratio      : 14.11x
```

## V2a — four MACs/lookup, Hidden=256

```text
Hidden size              : 256
Layers                   : 4
Vocabulary               : 256
Parameters               : 393,216
Original weights         : Q4 packed
Activation compute       : 2-bit
MACs per RAM lookup      : 4
Q4 weight memory         : 0.19 MiB
Compiled RAM tables      : 40.00 MiB
MACs/generated token     : 327,680
RAM lookups/token        : 81,920
Prompt tokens            : 58
Generated tokens         : 256
Compile/build time       : 0.215 s

CPU DIRECT MAC VERSION
Prefill time             : 15.38 ms
Generation time          : 0.0806 s
Generation speed         : 3177.98 tokens/s
Effective MAC rate       : 1.041 G MAC/s

RAM V2 - 4 MACs PRECOMPUTED PER LOOKUP
Prefill time             : 38.12 ms
Generation time          : 0.2149 s
Generation speed         : 1191.40 tokens/s
RAM lookup rate          : 97.60 M lookup/s
Effective MAC equivalent : 0.390 G MAC/s

IDENTICAL OUTPUT         : PASS
CPU direct MAC           : 3177.98 tok/s
RAM 4-MAC lookup         : 1191.40 tok/s
CPU/RAM ratio            : 2.67x
```

## V2b — same method, Hidden=1024

```text
Hidden size              : 1024
Layers                   : 4
Vocabulary               : 256
Parameters               : 4,718,592
Original weights         : Q4 packed
Activation compute       : 2-bit
MACs per RAM lookup      : 4
Q4 weight memory         : 2.25 MiB
Compiled RAM tables      : 544.00 MiB
MACs/generated token     : 4,456,448
RAM lookups/token        : 1,114,112
Prompt tokens            : 58
Generated tokens         : 64
Compile/build time       : 3.015 s

CPU DIRECT MAC VERSION
Prefill time             : 223.19 ms
Generation time          : 0.2623 s
Generation speed         : 243.99 tokens/s
Effective MAC rate       : 1.087 G MAC/s

RAM V2 - 4 MACs PRECOMPUTED PER LOOKUP
Prefill time             : 923.04 ms
Generation time          : 1.0789 s
Generation speed         : 59.32 tokens/s
RAM lookup rate          : 66.09 M lookup/s
Effective MAC equivalent : 0.264 G MAC/s

IDENTICAL OUTPUT         : PASS
CPU direct MAC           : 243.99 tok/s
RAM 4-MAC lookup         : 59.32 tok/s
CPU/RAM ratio            : 4.11x
```

Note: the interactive script's explanatory footer still contained the earlier V2a hard-coded `327,680 -> 81,920` text. The actual V2b header above is authoritative: `4,456,448 -> 1,114,112`.

## V3 — 16 MiB universal LUT

```text
Universal LUT size       : 16.00 MiB
LUT build time           : 0.020 s

Hidden size              : 1024
Layers                   : 4
Vocabulary               : 256
Parameters               : 4,718,592
Q4 weight memory         : 2.25 MiB
Universal RAM LUT        : 16.00 MiB
Total weights + LUT      : 18.25 MiB
MACs/token               : 4,456,448
RAM lookups/token        : 1,114,112
MACs per lookup          : 4
Generated tokens         : 64
Model creation           : 38.98 ms

CPU DIRECT Q4/Q2 MAC
Prefill                  : 144.36 ms
Generation               : 0.1694 s
Speed                    : 377.87 tok/s
MAC rate                 : 1.684 G MAC/s

RAM UNIVERSAL-LUT COMPUTE
Prefill                  : 873.96 ms
Generation               : 1.1861 s
Speed                    : 53.96 tok/s
RAM lookup rate          : 60.11 M lookup/s
Effective MAC equivalent : 0.240 G MAC/s

IDENTICAL OUTPUT         : PASS
CPU                      : 377.87 tok/s
RAM V3                   : 53.96 tok/s
CPU/RAM ratio            : 7.00x
```

## V4 — 64 KiB codebook LUT

```text
Hidden size              : 1024
Layers                   : 4
Vocabulary               : 256
Logical Q4 parameters    : 4,718,592
CPU packed-Q4 memory     : 2.25 MiB
RAM coded-weight memory  : 1.19 MiB
Compute LUT              : 64.00 KiB
MACs/token               : 4,456,448
RAM lookups/token        : 1,114,112
MACs per lookup          : 4
Generated tokens         : 128
Model build              : 12.64 ms

CPU PACKED-Q4 DIRECT MAC
Prefill                  : 142.58 ms
Generation               : 0.3359 s
Speed                    : 381.12 tok/s
MAC rate                 : 1.698 G MAC/s

RAM V4 64-KiB LUT
Prefill                  : 53.42 ms
Generation               : 0.1302 s
Speed                    : 982.84 tok/s
Lookup rate              : 1094.99 M lookup/s
Effective MAC equivalent : 4.380 G MAC/s

IDENTICAL OUTPUT         : PASS
CPU                      : 381.12 tok/s
RAM V4                   : 982.84 tok/s
CPU/RAM ratio            : 0.39x
```

## V5 — fair same-codebook control + 256 MiB pressure

```text
Hidden size              : 4,096
Layers                   : 4
Vocabulary               : 256
Logical MACs/token       : 68,157,440
Lookup operations/token  : 17,039,360
Weight-code memory       : 16.25 MiB
Small LUT                : 64.00 KiB
Large LUT                : 256.00 MiB
Large LUT replicas       : 4,096
Generated test tokens    : 4
Warmup verification      : PASS

A. SAME CODEBOOK - DIRECT CPU MAC
Speed                    : 23.07 tok/s
Effective MAC rate       : 1.572 G MAC/s

B. 64 KiB LUT
Speed                    : 67.37 tok/s
Lookup rate              : 1147.89 M lookup/s
Effective MAC equivalent : 4.592 G MAC/s

C. LARGE LUT / CACHE-EVICTION TEST
Speed                    : 4.32 tok/s
Lookup rate              : 73.55 M lookup/s
Effective MAC equivalent : 0.294 G MAC/s

IDENTICAL ARITHMETIC     : PASS
Direct CPU               : 23.07 tok/s
64 KiB LUT               : 67.37 tok/s
Large LUT                : 4.32 tok/s
Small LUT / Direct       : 2.92x
Large LUT / Direct       : 0.19x
Small / Large LUT        : 15.61x
```

## V6 — 16 MACs/lookup

```text
Hidden size              : 4,096
Layers                   : 4
Vocabulary               : 256
Activation format        : Q1 (-1 / +1)
Logical MACs/token       : 68,157,440
Lookups/token            : 4,259,840
MACs per lookup          : 16
Weight-code memory       : 4.06 MiB
Base LUT                 : 32.00 MiB
Large LUT                : 256.00 MiB
Large LUT replicas       : 8
Base LUT build           : 0.025 s
Large LUT build          : 0.078 s
Warmup verification      : PASS

A. DIRECT CPU - SAME CODEBOOK
Speed                    : 29.35 tok/s
MAC rate                 : 2.000 G MAC/s

B. 32 MiB LUT - 16 MAC/lookup
Speed                    : 36.07 tok/s
Lookup rate              : 153.65 M lookup/s
Effective MAC equivalent : 2.458 G MAC/s

C. LARGE LUT - DRAM PRESSURE
Speed                    : 17.27 tok/s
Lookup rate              : 73.56 M lookup/s
Effective MAC equivalent : 1.177 G MAC/s

IDENTICAL ARITHMETIC     : PASS
Direct CPU               : 29.35 tok/s
32 MiB 16-MAC LUT        : 36.07 tok/s
Large 16-MAC LUT         : 17.27 tok/s
32 MiB LUT / CPU         : 1.23x
Large LUT / CPU          : 0.59x
```

## V7 — 32 MACs/lookup, first 256 MiB crossover

```text
Hidden size               : 4,096
Layers                    : 4
Vocabulary                : 256
Weight representation     : 1 code / 32 Q4 weights
Activation representation : 1 code / 32 Q1 values
Logical MACs/token        : 68,157,440
Lookups/token             : 2,129,920
MACs per lookup           : 32
Weight-code memory        : 2.03 MiB
Base LUT                  : 128.00 KiB
Large LUT                 : 256.00 MiB
Large LUT replicas        : 2,048
Base LUT build            : 1.94 ms
Large LUT build           : 59.73 ms
Warmup verification       : PASS

A. DIRECT CPU - 32 MACs
Speed                    : 29.49 tok/s
MAC rate                 : 2.010 G MAC/s

B. 128 KiB LUT - 32 MAC/lookup
Speed                    : 506.54 tok/s
Lookup rate              : 1078.89 M lookup/s
Effective MAC equivalent : 34.525 G MAC/s

C. LARGE LUT - DRAM PRESSURE - 32 MAC/lookup
Speed                    : 32.81 tok/s
Lookup rate              : 69.89 M lookup/s
Effective MAC equivalent : 2.236 G MAC/s

IDENTICAL ARITHMETIC     : PASS
Direct CPU               : 29.49 tok/s
128 KiB LUT              : 506.54 tok/s
Large LUT                : 32.81 tok/s
Small LUT / CPU          : 17.17x
Large LUT / CPU          : 1.11x
```

## V8 — 64 coded operations/lookup + CPU branchless control

```text
Hidden size              : 4,096
Layers                   : 4
Vocabulary               : 256
Weight block             : 64 Q4 values/code
Activation block         : 64 Q1 values/code
Logical ops/token        : 68,157,440
Lookups/token            : 1,064,960
Ops per lookup           : 64
Weight-code memory       : 1.02 MiB
Base LUT                 : 128.00 KiB
Large LUT                : 256.00 MiB
Large LUT replicas       : 2,048
Tokens                   : 8
Base LUT build           : 3.19 ms
Large LUT build          : 49.93 ms
Warmup verification      : PASS

A. CPU MULTIPLY
Speed                    : 13.03 tok/s
Logical operation rate   : 0.888 Gop/s

B. CPU BRANCHLESS Q1 - NO MULTIPLY
Speed                    : 12.95 tok/s
Logical operation rate   : 0.883 Gop/s

C. 128 KiB LUT - 64 OPS/LOOKUP
Speed                    : 1071.19 tok/s
Lookup rate              : 1140.78 M lookup/s
Effective op equivalent  : 73.010 Gop/s

D. 256 MiB LARGE LUT - 64 OPS/LOOKUP
Speed                    : 58.55 tok/s
Lookup rate              : 62.36 M lookup/s
Effective op equivalent  : 3.991 Gop/s

IDENTICAL ARITHMETIC     : PASS
CPU multiply             : 13.03 tok/s
CPU branchless           : 12.95 tok/s
Small LUT                : 1071.19 tok/s
Large LUT                : 58.55 tok/s
Large LUT / multiply     : 4.49x
Large LUT / branchless   : 4.52x
Small LUT / branchless   : 82.69x
```

Mandatory caveat: V8's 8-bit weight code selects one of only 256 predefined 64-element Q4 vectors, and the 8-bit activation code selects one of only 256 predefined 64-element Q1 vectors. This is a restricted coded-vector domain, not arbitrary 64-element Q4/Q1 data.
