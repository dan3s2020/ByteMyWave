# TensorWave / Transit GPU

TensorWave started as an investigation into running models much larger than GPU VRAM by keeping the persistent model in cheap system memory and using the GPU as a small working cache/accelerator. The project has since evolved into **Transit GPU**: a modular inference architecture that tries to keep model weights local to many inexpensive memory-compute tiles and move only activations, commands and reduced results through the host fabric.

The current target workload is **Kimi K3**, used as a deliberately extreme MoE case. The current host candidate is one **Dell PowerEdge R920**, not a fleet of conventional servers. The current scale target is a modular fabric of roughly **38 tiles × 8 DDR3 channels = 304 independent DDR3 channels**, with local computation close to each tile's memory and a PCIe switch/fan-out tree back to the R920.

This branch is a research/design branch. It contains measured host-side evidence, the exact benchmark scripts used to obtain it, the bitplane arithmetic that was verified exactly, the hardware directions that were tried or rejected, and the implementation plan for the first physical Transit tile.

## Core principle

The architecture must not turn the R920 into a 300-DIMM motherboard and must not stream every active weight over PCIe every token.

```text
                         R920
                 scheduler / router
                 reducer / runtime
                          |
                    PCIe switch tree
                          |
          +---------------+---------------+
          |               |               |
       Transit tile    Transit tile    Transit tile
       8 DDR3 ch       8 DDR3 ch       8 DDR3 ch
          |               |               |
      local weights    local weights    local weights
      local compute    local compute    local compute
          |               |               |
          +------ activations/results ----+
```

**Weights stay local. Compute happens local. We unite the system after compute, not by electrically combining hundreds of DDR buses.**

## What is demonstrated already

The host experiments are no longer hypothetical:

- exact signed INT4 × INT8 dot products were implemented with bitplanes, `AND`, `POPCNT`, shifts and add/sub;
- the bitplane representation occupies the same logical 4 bits/weight as packed Q4;
- the x64 assembly kernel was verified bit-exact against a scalar reference;
- multicore execution reached about **53.7 Gweights/s** on the test laptop;
- a 6 GiB DDR5 working set proved the result was not a cache illusion;
- raw DDR5 bandwidth measured about **50 GB/s**;
- the V3 bitplane engine consumed about **26.8 GB/s of Q4-equivalent weight bytes / 53.7 Gweights/s**;
- NVMe direct-read saturated around **6.1 GB/s**;
- CPU+DDR5 and SSD prefetch overlapped well in the final host overlap test;
- the V4 masked-activation-sum formulation was mathematically exact but slower on this CPU and is rejected as a CPU optimization.

The next meaningful benchmark belongs on physical Transit hardware, not another laptop microbenchmark.

## Important distinction: proof kernel vs K3 kernel

The proven bitplane kernel uses a **signed two's-complement INT4 weight representation and INT8 activations**. K3's published routed experts use **MXFP4 weights and MXFP8 activations**. The current proof therefore validates the bitplane execution principle, not exact K3 numerical semantics. A K3 tile must either implement MXFP4/MXFP8 decode/scaling directly or define and validate a conversion into the Transit internal representation.

## Current hardware direction

The first laboratory board candidate is the surplus **YPCB-00338-1P1** FPGA card: Kintex-7, PCIe and two local DDR3 memory channels, with public reverse-engineering work and LiteX board support. It is attractive for proving the endpoint/runtime/kernel path, but it is **not the final 8-channel tile** and it does not directly accept the project's loose DDR3 DIMMs.

The final target remains one endpoint per tile serving several DDR3 channels locally. At 8 channels per tile, 38 endpoints give 304 channels while keeping the PCIe topology manageable.

Mining risers are useful only as cheap powered **PCIe physical extenders**. They do not create lanes or DDR channels. They make sense because the tile keeps weight traffic local, so its upstream PCIe link carries comparatively small command/activation/result traffic.

## Repository map

The original TensorWave documents are preserved as the history of the RAM→VRAM streaming phase:

- [`docs/00-PROJECT-INTENT.md`](docs/00-PROJECT-INTENT.md)
- [`docs/01-MODEL-MEMORY.md`](docs/01-MODEL-MEMORY.md)
- [`docs/02-WEIGHT-ATLAS.md`](docs/02-WEIGHT-ATLAS.md)
- [`docs/03-STREAMING-RUNTIME.md`](docs/03-STREAMING-RUNTIME.md)
- [`docs/04-COMPRESSION.md`](docs/04-COMPRESSION.md)
- [`docs/05-OPEN-QUESTIONS.md`](docs/05-OPEN-QUESTIONS.md)
- [`docs/06-COLLABORATION.md`](docs/06-COLLABORATION.md)
- [`docs/TRANSCRIPT.md`](docs/TRANSCRIPT.md)
- [`docs/USER-INPUT-VERBATIM.md`](docs/USER-INPUT-VERBATIM.md)

The Transit phase is documented in:

- [`docs/07-TRANSIT-EVOLUTION.md`](docs/07-TRANSIT-EVOLUTION.md) — chronological design evolution, including discarded directions.
- [`docs/08-EVIDENCE-BENCHMARKS.md`](docs/08-EVIDENCE-BENCHMARKS.md) — measured results and what they do/do not prove.
- [`docs/09-BITPLANE-MATH-AND-KERNEL.md`](docs/09-BITPLANE-MATH-AND-KERNEL.md) — exact arithmetic and the CPU proof kernel.
- [`docs/10-KIMI-K3-TARGET.md`](docs/10-KIMI-K3-TARGET.md) — model-level target, bandwidth rooflines and MoE implications.
- [`docs/11-DDR3-TILE-ARCHITECTURE.md`](docs/11-DDR3-TILE-ARCHITECTURE.md) — R920 + PCIe fanout + 38×8-channel tile design.
- [`docs/12-HARDWARE-CANDIDATES.md`](docs/12-HARDWARE-CANDIDATES.md) — risers, FPGA boards, open DDR3 controllers and decisions.
- [`docs/13-IMPLEMENTATION-ROADMAP.md`](docs/13-IMPLEMENTATION-ROADMAP.md) — concrete path from one lab tile to a K3 cluster.
- [`docs/14-MEMORY-CONTROLLER-TRADE-STUDY.md`](docs/14-MEMORY-CONTROLLER-TRADE-STUDY.md) — CPU vs FPGA vs minimal logic, discrete-transistor limits, DDR PHY/controller bottleneck, DDR3/DDR4/DDR5 channel economics and the current no-custom-ASIC decision.
- [`benchmarks/`](benchmarks/) — exact host-side scripts and assembly used in the measured experiments.

## Current success criterion

A Transit tile is successful when it demonstrates all of the following on physical hardware:

1. weights stored in local DDR3;
2. activation/command arrival over PCIe;
3. local bitplane or MXFP-compatible compute without shipping those weights to the host;
4. local accumulation/reduction;
5. result returned to the R920;
6. measured sustained local DDR3 bandwidth and compute utilization;
7. deterministic correctness against a software reference.

Only after that proof do we scale to multiple tiles and then to a real K3 layer/expert path.
