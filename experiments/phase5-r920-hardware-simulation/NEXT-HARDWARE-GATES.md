# Phase 5 — Next physical hardware gates

Run these gates in order on the real Dell R920. Analytical Phase-5 results do not replace them.

## Gate 1 — inventory

Capture:

```text
exact R920 CPU models and count
DIMM population by socket/channel
memory speed negotiated
riser configuration
PSU modules and redundancy mode
BIOS/iDRAC firmware
RTX 3060 exact AIB model/dimensions
```

## Gate 2 — GPU integration

Verify:

```text
physical fit
safe auxiliary power
boot stability
nvidia-smi visibility
CUDA initialization
PCIe generation and width under load
GPU temperature and clocks under sustained load
```

## Gate 3 — NUMA map

Record the OS-visible NUMA topology and map the selected GPU PCI bus/root to its owning CPU/NUMA node.

## Gate 4 — H2D locality sweep

For the same pinned buffer size and same GPU:

```text
allocate on local NUMA node -> measure H2D
allocate on every remote node -> measure H2D
```

Record averages, percentiles and run-to-run variance.

## Gate 5 — Phase-3 proof

Run the existing Q4 streaming proof with real checkpoint bytes and preserve:

```text
correctness
compressed H2D
source-equivalent feed rate
dequant time
GEMM time
starvation
hidden transfer
wall time
```

## Gate 6 — Phase-4 crossover

Sweep M and TileN on the actual machine and compare measured crossover against the Phase-5 simulator after replacing assumed H2D/TFLOPS with measured calibration.

## Gate 7 — persistent compressed cache

Reserve increasing amounts of safe VRAM for compressed residency and measure real bytes avoided + starvation/wall-time improvement.

## Gate 8 — second GPU

Only after one-GPU validation, place a second GPU on a different CPU/root if hardware integration permits it. Compare:

```text
single worker
2 independent workers
one GPU per NUMA root
2 GPUs sharing one CPU/root if physically possible
```

This determines whether multi-GPU scaling is limited by per-GPU PCIe, host memory controllers, QPI/NUMA traffic, or software scheduling.
