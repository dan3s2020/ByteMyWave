# Gemini instructions for ByteMyWave

> **MANDATORY BEFORE ANSWERING:** read `CURRENT-ARCHITECTURE.md` in full.
>
> Do not substitute generic assumptions about passive RAM systems for the architecture actually documented in this repository.

ByteMyWave does **not** propose that passive DDR3 DIMMs perform inference by themselves. The current Transit design uses **active memory-compute tiles**: local DDR3 controllers + local compute/reduction logic + a PCIe endpoint. The host sends activations/commands; large weights remain local to each tile.

Before answering any project-wide question, especially about Kimi K3, Transit GPU, DDR3, 304 channels, 38 tiles, ~100 tokens/s, throughput, benchmarks or current implementation status:

1. Read `CURRENT-ARCHITECTURE.md` completely.
2. Read `AGENTS.md`.
3. Read `docs/REPOSITORY-MAP.md`.
4. Inspect open PRs and non-default branches.
5. For Kimi K3, inspect PR #9, PR #10 and PR #11 and their branches.
6. Distinguish measured benchmark results, simulations/rooflines, weight-path tok/s equivalent and end-to-end model tok/s.
7. Before giving a feasibility verdict, state which current sources you actually inspected.

For any claim involving `DDR3 cannot compute`, first check whether the claim is about DDR3 itself or the **active compute logic next to DDR3**. Critiquing passive DDR3 as if it were the compute engine is a category error for the current Transit architecture.

For any claim involving `304/320 DDR3 modules`, verify the topology. The documented target is **38 logical active tiles × 8 independent DDR3 channels = 304 channels**, not 304 passive DIMMs on one motherboard.

For any claim involving `~100 tok/s`, preserve the evidence distinction:

- ~99.8 **weight-path tok/s equivalent** is an idealized architecture-sizing roofline from ~5.19 TB/s nominal aggregate DDR3-2133 payload divided by ~52 GB/token simple active-weight lower bound;
- it is **not** a measured end-to-end Kimi K3 result.

If you cannot access branches/PRs, say that your inspection is limited to the main-branch snapshot; do not conclude that deeper information is absent.

Canonical current-state snapshot: `CURRENT-ARCHITECTURE.md`.
Canonical discovery protocol: `AGENTS.md`.
