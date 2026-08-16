# TensorWave / Transit GPU

TensorWave started as an investigation into running models much larger than GPU VRAM by keeping the persistent model in cheap system memory and using the GPU as a small working cache/accelerator. The project has since evolved into **Transit GPU**: a modular inference architecture that tries to keep model weights local to many inexpensive memory-compute tiles and move only activations, commands and reduced results through the host fabric.

The current target workload is **Kimi K3**, used as a deliberately extreme MoE case. The primary high-throughput direction remains one **Dell PowerEdge R920** plus a modular fabric of roughly **38 tiles × 8 DDR3 channels = 304 independent DDR3 channels**, with local computation close to each tile's memory and a PCIe switch/fan-out tree back to the R920.

This branch also preserves a second low-cost direction discovered during the hardware search: use complete obsolete **64-DIMM-class DDR2 multi-socket servers** as already-manufactured memory-controller + CPU + chassis + network tiles. That path is treated as an experimentally falsifiable alternative/fallback, not as an assumption that old servers will automatically be fast enough.

This branch is a research/design branch. It contains measured host-side evidence, the exact benchmark scripts used to obtain it, the bitplane arithmetic that was verified exactly, the hardware directions that were tried or rejected, the implementation plan for the first physical Transit tile, and the whole-server DDR2 architecture/procurement gates.

## Core principle

The architecture must not turn the R920 into a 300-DIMM motherboard and must not stream every active weight over PCIe or Ethernet every token.

```text
                         R920 / head
                 scheduler / router / API
                 reducer / runtime
                          |
              +-----------+-----------+
              |                       |
        PCIe switch tree          fast network
              |                       |
       Transit DDR3 tiles       DDR2 server nodes
       local weights            local weights
       local compute            local CPU compute
              |                       |
              +---- activations/results ----+
```

**Weights stay local. Compute happens local. We unite the system after compute, not by electrically combining hundreds of DDR buses or by shipping active weight matrices through a slow fabric.**

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

The next meaningful benchmark belongs on physical Transit hardware: either one YPCB/DDR3 tile or one complete high-DIMM-count DDR2 server measured as a NUMA memory-compute node.

## Important distinction: proof kernel vs K3 kernel

The proven bitplane kernel uses a **signed two's-complement INT4 weight representation and INT8 activations**. K3's working routed-expert format uses **MXFP4 weights and MXFP8 activations**. The current proof therefore validates the bitplane execution principle, not exact K3 numerical semantics. A K3 implementation must either implement MXFP4/MXFP8 decode/scaling directly or define and validate a conversion into the Transit internal representation.

## Current hardware directions

### DDR3 / FPGA local-compute path

The first laboratory board candidate is the surplus **YPCB-00338-1P1** FPGA card: Kintex-7, PCIe and two local DDR3 memory channels, with public reverse-engineering work and LiteX board support. It is attractive for proving the endpoint/runtime/kernel path, but it is **not the final 8-channel tile** and it does not directly accept the project's loose DDR3 DIMMs.

The final target remains one endpoint per tile serving several DDR3 channels locally. At 8 channels per tile, 38 endpoints give 304 channels while keeping the PCIe topology manageable.

Mining risers are useful only as cheap powered **PCIe physical extenders**. They do not create lanes or DDR channels. They make sense because the tile keeps weight traffic local, so its upstream PCIe link carries comparatively small command/activation/result traffic.

### Whole-server DDR2 path

The second path asks whether obsolete 8-socket servers are so cheap that buying the complete memory-controller infrastructure is cheaper than building it.

Current named candidates include:

- HP ProLiant DL785 G5/G6 class systems;
- Sun Fire X4640 systems with up to 8 CPU modules × 8 DDR2 DIMM slots = 64 DIMMs;
- other complete 48–64+ DIMM obsolete servers if they are electrically complete and Linux-usable.

These machines are treated as **memory-compute nodes**, not remote RAM boxes. Their CPUs must process resident local weights; the network carries activations/results. The decisive unknown is measured old-CPU `Gweights/s`, not aggregate RAM capacity.

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
- [`docs/14-SURPLUS-DDR2-SERVER-FABRIC.md`](docs/14-SURPLUS-DDR2-SERVER-FABRIC.md) — 10–20-node whole-server architecture, NUMA placement, network/API path and bandwidth/tok/s rooflines.
- [`docs/15-HARDWARE-SEARCH-MATRIX.md`](docs/15-HARDWARE-SEARCH-MATRIX.md) — consolidated parts found, rejected candidates, riser/FPGA/server roles and exact shopping targets.
- [`docs/16-OPEN-PROBLEMS-AND-VALIDATION-GATES.md`](docs/16-OPEN-PROBLEMS-AND-VALIDATION-GATES.md) — unresolved K3/CPU/NUMA/network/power problems and the measurements that decide each one.
- [`benchmarks/`](benchmarks/) — exact host-side scripts and assembly used in the measured experiments.

## Current performance language

For K3 the working model in document 10 uses ~52 GB of active 4-bit-equivalent weights per token.

Therefore:

```text
100 tok/s weight path ~= 5.2 TB/s
```

The DDR3 304-channel design was sized around that roofline.

For the whole-server DDR2 path, a conservative illustrative fully populated 8-socket DDR2-533 model gives roughly:

```text
~68.3 GB/s theoretical/server
~1.31 ideal weight-path tok/s/server
10 servers -> ~13.1 ideal weight-path tok/s
20 servers -> ~26.3 ideal weight-path tok/s
```

Those are **not end-to-end K3 numbers**. The old CPU kernel may be much slower than the local memory roofline. The first one-server `Gweights/s` benchmark decides whether this branch is compute architecture, fallback capacity, or a dead end.

## Current success criteria

### DDR3 Transit tile

A Transit tile is successful when it demonstrates all of the following on physical hardware:

1. weights stored in local DDR3;
2. activation/command arrival over PCIe;
3. local bitplane or MXFP-compatible compute without shipping those weights to the host;
4. local accumulation/reduction;
5. result returned to the R920;
6. measured sustained local DDR3 bandwidth and compute utilization;
7. deterministic correctness against a software reference.

### Whole DDR2 server node

A server node is successful when it demonstrates:

1. stable Linux + full intended NUMA/DIMM inventory;
2. measured local and aggregate DDR2 bandwidth;
3. NUMA-local Transit kernel with measured `Gweights/s`;
4. resident weight shard/expert;
5. activation received over the network;
6. local compute without sending the weights to the head;
7. reduced result returned;
8. wall-power measurement;
9. deterministic correctness against the same software reference.

Only measured physical results decide which architecture scales.