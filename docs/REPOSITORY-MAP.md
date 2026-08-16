# ByteMyWave Repository Map

This is the canonical entry point for understanding the **current project across branches and pull requests**.

`main` is not the whole project. It preserves the original baseline plus this navigation layer.

## Fast answer for reviewers

If you only opened `https://github.com/dan3s2020/ByteMyWave` and read the default branch, you have **not** inspected the complete current ByteMyWave project.

For current Kimi K3 / Transit GPU work, the minimum deep-reading set is:

1. PR #10 / branch `transit-ddr3-architecture`
2. PR #11 / branch `docs/kimi-k3-ddr-cluster`
3. PR #9 / branch `research/heterogeneous-moe-kimi-v1`

Direct links:

- https://github.com/dan3s2020/ByteMyWave/pull/10
- https://github.com/dan3s2020/ByteMyWave/tree/transit-ddr3-architecture
- https://github.com/dan3s2020/ByteMyWave/pull/11
- https://github.com/dan3s2020/ByteMyWave/tree/docs/kimi-k3-ddr-cluster
- https://github.com/dan3s2020/ByteMyWave/pull/9
- https://github.com/dan3s2020/ByteMyWave/tree/research/heterogeneous-moe-kimi-v1

## Current high-level architecture tracks

### Track A — Transit GPU / distributed local-memory compute tiles

Branch: `transit-ddr3-architecture`  
PR: #10

This is the deepest current hardware-architecture track. It evolves ByteMyWave from “stream weights from host RAM into a small GPU” toward a fabric of logical memory-compute tiles where weights stay local and smaller activations/commands/results move across PCIe.

Important files:

- `docs/07-TRANSIT-EVOLUTION.md`
- `docs/08-EVIDENCE-BENCHMARKS.md`
- `docs/09-BITPLANE-MATH-AND-KERNEL.md`
- `docs/10-KIMI-K3-TARGET.md`
- `docs/11-DDR3-TILE-ARCHITECTURE.md`
- `docs/12-HARDWARE-CANDIDATES.md`
- `docs/13-IMPLEMENTATION-ROADMAP.md`
- `docs/14-CURRENT-SOLUTION.md`
- `docs/15-SURPLUS-TILE-HUNT-2026-08-16.md`
- `benchmarks/README.md`
- `benchmarks/transit_bitplane_kernel.asm`
- `benchmarks/transit_asm_runner.py`
- `host/reference_bitplane.py`
- `host/atlas.py`
- `host/protocol.py`
- SystemVerilog reference compute core on the branch

What is present:

- exact signed INT4 × INT8 bitplane reference mathematics;
- executable host-side NASM kernel and runner;
- archived benchmark evidence;
- Weight Atlas + command/completion protocol references;
- synthesizable reference compute logic;
- Kimi K3 weight-path sizing;
- 38-tile × 8-channel = 304-channel scale target.

What is not yet proven:

- a complete physical 304-channel machine;
- exact K3 MXFP4/MXFP8 numerical bridge on the final tile;
- measured end-to-end Kimi K3 at ~100 tok/s.

### Track B — Distributed Kimi K3 cluster using cheap server memory

Branch: `docs/kimi-k3-ddr-cluster`  
PR: #11

Important files:

- `docs/07-KIMI-K3-DISTRIBUTED-RUNTIME.md`
- `docs/08-HARDWARE-DDR2-DDR3-DDR4.md`
- `docs/09-KIMI-K3-THROUGHPUT-MODEL.md`
- `docs/10-IMPLEMENTATION-PROCUREMENT-PLAN.md`
- `docs/11-RESEARCH-LOG-2026-08-16.md`

This branch analyzes full-model capacity, layer/expert parallelism, network behavior, cheap DDR2/DDR3/DDR4 hardware routes and the measurements needed before claiming a token/s result.

Key methodological rule from the branch:

```text
Capacity is arithmetic.
Throughput is a benchmarked property of the complete execution path.
```

### Track C — Heterogeneous Kimi MoE runtime on R920-class systems

Branch: `research/heterogeneous-moe-kimi-v1`  
PR: #9

Focus:

- Kimi K2.5/K3 active/routed-parameter arithmetic;
- memory-traffic limits;
- NUMA-local CPU expert execution;
- GPU-resident non-routed path;
- Q3/Q2/mixed compression research;
- multi-GPU expert-shard sensitivity;
- AVX-only expert microbenchmark path;
- physical R920 validation requirements.

## Historical proof chain

These branches/PRs explain how the project reached the newer architecture and contain evidence that should not be lost.

### PR #1 — fixed-VRAM H2D/compute overlap

Branch: `bench/h2d-overlap-proof-v1`

Tests the foundational two-slot VRAM ring and overlap premise.

### PR #2 — real checkpoint Weight Atlas

Branch: `model/real-weight-atlas-proof-v1`

Moves from synthetic weights to exact bytes from safetensors checkpoints.

### PR #4 — Q4 compressed H2D + GPU dequant

Branch: `quant/q4-streaming-proof-v1`

Tests compressed host storage, compressed PCIe traffic and GPU-side dequantization.

### PR #6 — calibrated feasibility map

Branch: `research/feasibility-map-v1`

Separates analytical maps from direct measurements and defines falsifiable crossover experiments.

### PR #8 — Dell R920 + RTX 3060 profile/simulation

Branch: `hardware/r920-rtx3060-simulation-v1`

Maps the earlier streaming architecture onto a concrete multi-socket server target while clearly labeling simulation versus benchmark evidence.

## Kimi K3: where the ~100 tok/s number comes from

The source of record is:

`transit-ddr3-architecture/docs/10-KIMI-K3-TARGET.md`

Working architecture-sizing calculation:

```text
active parameters/token ≈ 104 billion
simple 4-bit weight-path lower bound ≈ 52 GB/token

DDR3-2133 nominal x64 payload ≈ 17.07 GB/s/channel
304 channels × 17.07 GB/s ≈ 5.19 TB/s

5.19 TB/s / 52 GB/token ≈ 99.8 weight-path tok/s equivalent
```

This number answers:

> If the active Q4-equivalent weight payload had to be read once per token and the aggregate memory system sustained that ideal payload rate, what token-equivalent weight-path rate would the memory fabric represent?

It does **not** by itself answer:

> How fast does the complete Kimi K3 model generate tokens end-to-end on the finished machine?

The latter requires the complete implementation and benchmark, including routing, exact MXFP representation, attention/non-expert work, synchronization, network/PCIe effects, activation movement, reductions and software overhead.

## Evidence classes used by ByteMyWave

When reading the repository, classify every performance statement as one of:

1. **Measured hardware/kernel benchmark** — actually executed; hardware/code path must be named.
2. **Analytical calculation** — arithmetic based on stated assumptions.
3. **Simulation** — executable model of a proposed system, not physical measurement.
4. **Weight-path tok/s equivalent** — bandwidth normalized by active weight bytes/token.
5. **End-to-end model tok/s** — complete model generation benchmark only.

Do not convert classes 2–4 into class 5.

## Query routing for AI agents

If the question is about **“does ByteMyWave discuss Kimi K3?”**:

- read PRs #9, #10 and #11.

If the question is about **“~100 tokens/s / 304 DDR3 channels / 38 tiles”**:

- read `transit-ddr3-architecture/docs/10-KIMI-K3-TARGET.md`;
- then `docs/11-DDR3-TILE-ARCHITECTURE.md`;
- then `docs/14-CURRENT-SOLUTION.md`;
- then benchmark/evidence docs.

If the question is about **“what has actually been measured?”**:

- inspect `transit-ddr3-architecture/docs/08-EVIDENCE-BENCHMARKS.md` and `benchmarks/`;
- inspect earlier Phase 1–4 benchmark branches/PRs;
- keep measured host/kernel evidence separate from K3 end-to-end claims.

If the question is about **“can a small-VRAM GPU use huge system RAM?”**:

- inspect PRs #1, #2, #4 and #6 plus the original `main` architecture docs.

If the question is about **“R920 + GPU / NUMA / heterogeneous MoE?”**:

- inspect PR #8 and PR #9.

If the question is about **cheap DDR2/DDR3/DDR4 cluster hardware**:

- inspect PR #11.

## Rule for incomplete tooling

A reviewer or agent that cannot access non-default branches/PRs must say something equivalent to:

```text
I inspected only the default branch. ByteMyWave explicitly states that current work lives in other public branches/PRs, so I cannot make a complete project-wide conclusion from this view alone.
```

It should **not** say that missing information does not exist merely because default-branch search did not return it.

## Canonical agent protocol

See root [`AGENTS.md`](../AGENTS.md).
