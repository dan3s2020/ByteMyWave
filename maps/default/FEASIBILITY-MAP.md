# TensorWave Feasibility Map — Default Analytical Baseline

This is a committed **analytical baseline map**, not an observed benchmark.

Assumptions:

```text
effective H2D bandwidth = 12 GB/s
effective dense-linear GPU throughput = 10 TFLOP/s
wire representation = TensorWave Q4-v1 = 0.625 B/parameter
persistent resident fraction = 0%
active parameter fraction = 100%
```

Under those assumptions:

```text
M_cross = 0.625 * 10e12 / (2 * 12e9)
        ≈ 260.4 activation rows
```

Meaning: around `M≈260`, dense-linear compute and streamed-weight transfer become equal in the idealized model. Below it, transfer dominates. Above it, compute can theoretically hide all H2D traffic.

## Regime map

Each cell is the ideal percentage of H2D time that dense-linear compute could hide.

```text
T = strongly transfer-bound
B = balanced
N = near-balanced
C = compute-bound / transfer theoretically fully hidden
```

| model | M=1 | M=4 | M=16 | M=64 | M=128 | M=256 | M=512 | M=1024 | M=2048 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7B   | T 0.4% | T 1.5% | T 6.1% | T 24.6% | B 49.2% | N 98.3% | C 100% | C 100% | C 100% |
| 13B  | T 0.4% | T 1.5% | T 6.1% | T 24.6% | B 49.2% | N 98.3% | C 100% | C 100% | C 100% |
| 33B  | T 0.4% | T 1.5% | T 6.1% | T 24.6% | B 49.2% | N 98.3% | C 100% | C 100% | C 100% |
| 70B  | T 0.4% | T 1.5% | T 6.1% | T 24.6% | B 49.2% | N 98.3% | C 100% | C 100% | C 100% |
| 120B | T 0.4% | T 1.5% | T 6.1% | T 24.6% | B 49.2% | N 98.3% | C 100% | C 100% | C 100% |

The repeated percentages are not a bug. In the ideal dense model, total parameter count scales compute and transfer together, so it cancels from the **overlap ratio**. Model size still changes absolute latency and RAM requirements.

## Absolute example — dense 70B Q4-v1

At 0% VRAM residency:

```text
70B * 0.625 B/param = 43.75 GB streamed per dense step
43.75 GB / 12 GB/s = 3.646 s transfer-only floor
```

| M | H2D ms | dense compute ms | ideal step ms | unhidden H2D ms | aggregate rows/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 3645.8 | 14.0 | 3645.8 | 3631.8 | 0.27 |
| 4 | 3645.8 | 56.0 | 3645.8 | 3589.8 | 1.10 |
| 16 | 3645.8 | 224.0 | 3645.8 | 3421.8 | 4.39 |
| 64 | 3645.8 | 896.0 | 3645.8 | 2749.8 | 17.55 |
| 128 | 3645.8 | 1792.0 | 3645.8 | 1853.8 | 35.11 |
| 256 | 3645.8 | 3584.0 | 3645.8 | 61.8 | 70.22 |
| 512 | 3645.8 | 7168.0 | 7168.0 | 0 | 71.43 |

This makes the central LLM problem visible:

```text
single-stream decode M=1 -> disastrous for pure dense streaming
large batch/prefill M~256 -> close to crossover
M>=512 -> transfer can theoretically disappear beneath compute
```

The aggregate rows/s column is **not** a claim of real 70B token throughput. It excludes attention, KV-cache traffic, dequantization, kernel inefficiency, synchronization and graph irregularity.

## What moves the map left/right

Move the crossover **left** (easier to hide transfer):

```text
higher PCIe H2D bandwidth
smaller bytes/parameter
more persistent VRAM residency
slower/longer compute per tile
fused kernels that avoid extra global-memory traffic
```

Move the crossover **right** (harder to hide transfer):

```text
faster GPU with unchanged PCIe
larger wire representation
poor H2D bandwidth
very small M / batch=1 decode
extra synchronization that prevents true copy/compute concurrency
```

MoE changes the absolute cost by reducing active weights. Expert caching/routing can improve the map further if the runtime avoids streaming inactive experts.

## This map must be replaced by measured calibration

Generate the map from actual TensorWave results:

```powershell
python tools/calibrate_feasibility_map.py --run-dir <phase3-run> --output calibration.json
```

Then feed measured values into:

```powershell
python tools/build_feasibility_map.py `
  --output-dir maps/measured `
  --pcie-gbps <measured> `
  --effective-tflops <measured> `
  --bytes-per-param 0.625
```

The measured map, not this analytical baseline, decides whether the idea works on the target GPU.
