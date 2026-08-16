# Phase 1 — H2D/Compute Overlap Proof

## Question

Can a GPU with very small VRAM behave as a compute accelerator for a model that lives mainly in host RAM if the next weight tile is transferred while the current tile is being computed?

This phase deliberately does **not** use MiniMax H3 yet. It isolates the fundamental systems claim before model-specific complexity is introduced.

If this experiment fails badly, there is no reason to spend time building an H3-specific Weight Atlas or quantized tile format first.

## What the code actually does

The executable allocates:

- a configurable collection of FP16 weight tiles in **pinned host RAM**;
- one activation matrix `X` in GPU memory;
- one output matrix `Y` in GPU memory;
- exactly **two fixed VRAM weight slots**.

For every tile it performs a real Tensor-Core-capable cuBLAS GEMM:

```text
Y = X × W_tile
```

Two execution modes are measured.

### Sequential baseline

```text
copy W0 -> compute W0 -> copy W1 -> compute W1 -> ...
```

### TensorWave two-slot pipeline

```text
GPU:   [compute W0] [compute W1] [compute W2] [compute W3]
PCIe:        [copy W1]    [copy W2]    [copy W3]    [copy W4]
```

A VRAM slot is never overwritten until the compute event that used that slot has completed.

The execution order is known before the run. No runtime graph search, file lookup, allocation or dynamic model decision is part of the measured loop.

## Why pinned RAM is used in Phase 1

The current hypothesis is about whether PCIe transfer can be hidden under compute.

Using pinned host memory removes pageable-memory staging noise from the first experiment and gives `cudaMemcpyAsync()` the conditions required for real asynchronous H2D DMA.

This does **not** mean TensorWave ultimately needs the entire model permanently pinned. A later experiment will compare:

```text
large normal RAM model
        -> small pinned host ring
        -> fixed VRAM ring
```

against fully pinned backing storage.

## Metrics

The program records:

- sequential wall time;
- overlapped wall time;
- sum of actual H2D copy-event durations;
- sum of GEMM durations;
- measured H2D GB/s;
- startup latency;
- **GPU starvation time** between consecutive GEMMs;
- estimated percentage of steady-state transfer hidden by compute;
- pipeline speedup;
- output correctness against the sequential execution.

The most important number is:

```text
steady_starvation_pct
```

not raw PCIe bandwidth alone.

## Falsifiable success criterion

For a tested shape, the strong version of the hypothesis is supported when:

```text
correctness passes
AND
steady_starvation_pct <= 10%
AND
estimated hidden transfer >= 80%
```

The program reports this as:

```text
HYPOTHESIS_SUPPORTED_FOR_THIS_SHAPE
```

This is intentionally shape-specific. One successful shape does not prove that H3 will work; it proves that the transfer/compute overlap mechanism itself can work on the hardware.

## Why sweep M

For a weight tile `W[K,N]`, H2D bytes are mostly determined by `K×N`, while GEMM work grows with `M×K×N`.

So `M` is an easy way to move through the important regimes:

```text
small M  -> transfer dominated
medium M -> break-even region
large M  -> compute dominated, transfer should become hideable
```

The first useful result is therefore not one number but the curve:

```text
M -> steady_starvation_pct
```

That tells us where this particular GPU crosses from PCIe-bound to compute-bound.

## Windows / RTX 3050 Ti initial run

Requirements:

- NVIDIA CUDA Toolkit with `nvcc` and cuBLAS;
- CMake;
- Visual Studio C++ build tools;
- NVIDIA driver.

From the repository root:

```powershell
.\scripts\run-proof.ps1
```

Default sweep:

```text
M = 64, 128, 256, 512, 1024, 2048
K = 4096
N = 2048
Tiles = 32
```

Each FP16 weight tile is approximately 16 MiB, so the default host-resident synthetic model is approximately 512 MiB while only two 16 MiB weight slots are reserved in VRAM.

The VRAM requirement is therefore approximately:

```text
2 × weight_tile + X + Y
```

and does not grow with the number of host tiles.

Raw runs are written under:

```text
runs/YYYYMMDD-HHMMSS/
```

and include one JSON file per `M` value plus `nvidia-smi.txt` when available.

## Manual single run

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release --parallel
.\build\Release\tensorwave_stream_proof.exe --m 512 --k 4096 --n 2048 --tiles 32 --warmup 3 --json result.json
```

## What this does NOT prove

This phase does not yet prove:

- MiniMax H3 can run on 4 GB VRAM;
- Q2/Q4 dequantization can be fused efficiently;
- H3 activations fit in the remaining VRAM;
- all H3 operators tile cleanly;
- quality survives the intended quantization;
- pageable RAM can feed a pinned staging ring without becoming the bottleneck;
- the same overlap remains after adding model-specific kernels.

Those are subsequent experiments.

## Phase 2 if Phase 1 succeeds

Replace synthetic FP16 tiles with one real H3 tensor family:

1. extract actual tensor metadata;
2. generate a minimal Weight Atlas;
3. split one H3 matrix into legal tiles;
4. store the weights in host RAM;
5. precompute the execution plan;
6. transfer with the same fixed VRAM ring;
7. add GPU-side quantized deconversion/dequantization;
8. compare against an unstreamed reference result;
9. measure starvation again.

Only after that should the runtime be expanded to a full H3 block and then the complete model.
