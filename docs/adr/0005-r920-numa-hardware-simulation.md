# ADR 0005 — R920 NUMA hardware simulation and placement rule

Status: **proposed / simulation-stage**

Date: **2026-08-15**

## Context

TensorWave has progressed from a generic small-VRAM/large-RAM hypothesis to a concrete hardware candidate: Dell PowerEdge R920 with four Xeon E7 v2 sockets, large DDR3 ECC capacity and an RTX 3060 12 GB as the first consumer-GPU candidate.

The R920 is a multi-socket NUMA machine. A GPU's PCIe path is attached to a particular CPU/root complex, while host memory can be physically attached to any socket. Treating the entire 1 TiB pool as uniform memory could cause TensorWave H2D traffic to traverse the inter-socket fabric before reaching the GPU.

The current Phase-3 runtime is single-GPU and uses pinned host memory plus a two-slot compressed VRAM ring. Phase 4 defines a calibrated analytical feasibility map but does not encode hardware topology.

## Decision

For R920 experimentation:

1. **GPU-local NUMA placement is the default hypothesis.** Pinned host source windows for a GPU worker should come from the NUMA node local to that GPU's PCIe root when the OS/runtime permits it.
2. **The hardware simulator is explicitly non-measured.** Its H2D/TFLOPS inputs are replaceable calibration parameters, not R920 performance claims.
3. **The simulator must preserve Phase-3 scheduling semantics** rather than inventing a different overlap mechanism:

```text
slot(i) = i % 2
copy(i) waits compute(i-2)
compute(i) waits copy(i)
```

4. **The simulator must preserve the Phase-4 roofline equation** for its model-level prediction.
5. **Current-code multi-GPU scaling means independent per-GPU workers first.** Equal model sharding is allowed only as an explicitly optimistic/unimplemented analytical lower bound until a real distributed runtime exists.
6. **Electrical PCIe slot count is not treated as physical GPU capacity.** Card length, width, auxiliary power and cooling remain separate hardware gates.
7. **12 GiB VRAM may be exploited as persistent compressed cache** once transient ring/workspace requirements are reserved and measured.

## Consequences

The Weight Atlas/execution plan may need future topology metadata such as:

```text
host NUMA node
preferred GPU
PCI bus/root
host replica/shard location
measured local/remote H2D
```

A real R920 run must compare local and remote NUMA H2D. If the difference is material, NUMA affinity becomes a correctness-of-performance requirement for TensorWave scheduling.

The first multi-GPU implementation should instantiate one independent Phase-3 state machine per GPU/NUMA domain. Tensor/pipeline/expert sharding is a later architectural change and must include communication costs.

## Falsification

This ADR does not assert that R920 will be fast. The hardware hypothesis is weakened if measurements show poor local pinned H2D, severe cross-socket contention, unsafe/impractical GPU integration, or a measured crossover substantially worse than calibrated predictions.

## Related

- `docs/10-R920-HARDWARE-PLATFORM.md`
- `tools/simulate_r920_tensorwave.py`
- `experiments/phase5-r920-hardware-simulation/README.md`
- `docs/adr/0004-feasibility-map-calibrated-roofline.md`
