# ByteMyWave Repository Map — Semantic Index

_Last full branch inventory: 2026-09-16_

This is the canonical semantic map of ByteMyWave. It exists so agents do **not** choose a branch, file, or conclusion from its name alone.

`main` is a baseline + navigation surface. The project state is distributed across branches and pull requests. A branch often inherits hundreds of files from an earlier branch, so **the files visible on a branch are not automatically the work introduced by that branch**.

## Mandatory rule for agents

Before selecting a branch to answer a question:

1. Read `CURRENT-ARCHITECTURE.md` for the high-level current architecture.
2. Read this map for branch semantics.
3. Enumerate the current branches/PRs in case this inventory is stale.
4. Choose the branch from **purpose + method + evidence + next gate**, not from its name.
5. On the chosen branch, read its branch-specific README/decision/result files before searching arbitrary filenames.
6. Distinguish inherited files from branch-local additions.
7. Distinguish measured results, simulations, analytical rooflines, engineering targets, and end-to-end model benchmarks.

If a branch exists that is newer than this map, inspect its diff against its base and add it to this map before making a project-wide conclusion.

---

# 30-second routing table

| Question / topic | Start here | Why |
|---|---|---|
| Current Transit hardware architecture | `transit-ddr3-architecture` / PR #10 | Canonical active memory-compute tile architecture, bitplane proof, 304-channel K3 sizing |
| Distributed Kimi K3 using cheap server RAM | `docs/kimi-k3-ddr-cluster` / PR #11 | Full-model sharding, server-memory routes, network/runtime/procurement model |
| Heterogeneous CPU/RAM + GPU MoE runtime | `research/heterogeneous-moe-kimi-v1` / PR #9 | NUMA-local routed experts, GPU fixed/non-routed path, expert-shard analysis |
| Q4_K × Q8_K CPU kernels, LUTs, codebooks, META, NUMA | `research/ram-lookup-compute-2026-09-16` / PR #16 | Current measured kernel/representation track; contains V27 META, V28 negative result, V29 NUMA plan |
| Native direct Q4_K × Q8_K provenance / AVX2 checkpoint | `bench/q4k-q8k-native-avx2-2026-09-16` / PR #17 | Exact native standalone controls and source archives |
| Exact LUT1/LUT2/LUT3 experiment | `bench/q4k-q8k-lut123-v21-2026-09-16` | Lossless Q4_K LUT gate with packing correction and Ivy-compatible build |
| Four-server GLM-5.2 plan | `research/glm52-4x2-optimized-runtime-2026-08-18` | llama.cpp/GGUF + expert scheduler + NUMA + GPU cache + MTP plan |
| Consumer-GPU K3 vs Transit comparison | `research/k3-consumer-gpu-vs-transit-2026-08-17` | Apples-to-apples effective-bandwidth / same-token parallelism framework |
| Co-Sol-Dex + Transit + speculative decoding | `research/transit-cosoldex-dspark-2026-09-09` | Context reduction + adaptive GPU cache + speculative target verification |
| Agent memory / retrievers / chroniclers | `agent/transit-active-memory-chronicles` / PR #14 | Agentic retrieval/memory subsystem, not model kernel hardware |
| Surplus FPGA/SmartNIC/tile candidates and fan-out | `research/transit-surplus-full-bom-2026-08-17` | Large hardware candidate ledger + BOM/fan-out research |
| Why DDR controller/PHY dominates custom tile design | `agent/transit-memory-controller-trade-study` | CPU vs FPGA vs minimal logic by cost/watt per usable DDR interface |
| Manufacturable DDR2 FPGA tile | `hardware/transit-ddr2-tile-reva` | BEE3-style 4×Virtex-5 / 8-channel engineering handoff |
| Whole old-server DDR2 fabric | `agent/transit-ddr2-server-fabric` / PR #12 | Uses complete obsolete servers as ready-made memory-controller nodes |
| Carrizo APU + DDR3 UMA idea | `agent/apu-ddr3-uma-k3` / PR #13 | iGPU-local DDR compute path and scale-out gates |
| Ten-R920 procurement/build design | `research/ten-r920-procurement-build` | Phase-7 BOM, corrected K3 routed arithmetic, staged procurement |
| Original fixed-VRAM streaming proof chain | PRs #1 → #2 → #4 → #6 → #8 | Shows how original RAM→VRAM design was measured and evolved |
| Recovered Transit tile research from EvoWorld | `archive/import-evoworld-transit-tile-research-2026-09-16` / draft PR #18 | Provenance archive only; not automatically canonical current architecture |
| Heretic + Evo2 orchestrator research | `research/heretic-evo2-orchestrator-2026-08-29` / PR #15 | Offline model-preparation/tool-use research; not Transit token hot path |

---

# Evidence vocabulary — never collapse these categories

1. **Measured end-to-end model decode** — full named model actually generated tokens on named hardware/runtime.
2. **Measured subsystem/kernel result** — real benchmark, but only a kernel, memory path, network path, etc.
3. **Simulation** — executable model of a proposed system.
4. **Analytical roofline / screening estimate** — arithmetic under stated assumptions.
5. **Weight-path tok/s equivalent** — aggregate weight-processing bandwidth normalized by modeled active bytes/token.
6. **Engineering target / acceptance gate** — a number the design is trying to reach, not a result.
7. **External lead / marketplace observation** — useful research input, not a project measurement.

A class 2–6 number must never be silently presented as class 1.

---

# Repository-area map

These directory names recur across branches. Their meaning is more useful than their name alone:

- `docs/` — architecture, research records, evidence interpretation, plans, current-solution notes.
- `docs/adr/` — explicit architecture decisions. An ADR can accept an experiment without making it the default runtime.
- `docs/audits/` — cross-track or model/hardware audits; usually synthesis/evidence documents rather than new kernels.
- `docs/sessions/` — preserved session-level design/research notes; useful provenance, not automatically canonical.
- `benchmarks/` — measured or runnable kernel/system microbenchmarks. Read the benchmark README and evidence boundaries before quoting numbers.
- `experiments/` — staged proof/validation tracks; later phases often supersede assumptions from earlier phases.
- `host/` — host-side Weight Atlas, protocol, and numerical/reference logic for Transit.
- `rtl/` — synthesizable/reference FPGA logic. Existence of RTL is not proof of physical board performance.
- `hardware/` — physical engineering specifications/BOMs and hardware-specific design tracks.
- `tools/` — simulators, converters, feasibility calculators, benchmark helpers. Tool output may be analytical rather than measured.
- `maps/` — feasibility maps; explicitly separate analytical projections from direct measurements.
- `src/` — CUDA/C++ mechanism proofs from the original streaming chain.
- `integrations/` — pinned external dependencies/integration boundaries where present.
- `.github/workflows/` — reproducibility/CI gates; CI passing proves the stated software checks, not physical throughput.

---

# Full branch atlas

## `main` — baseline, project origin, and mandatory navigation

**Purpose:** preserve the original ByteMyWave model-memory/streaming concept while acting as the entry point to newer branch work.

**Method:** project intent → model-memory anatomy → Weight Atlas → RAM→fixed-VRAM streaming → compression → open questions → collaboration/provenance.

**Read first:** `CURRENT-ARCHITECTURE.md`, `AGENTS.md`, this map, root `README.md`, `llms.txt`.

**Foundation docs:** `docs/00-PROJECT-INTENT.md` through `docs/06-COLLABORATION.md`, plus `TRANSCRIPT.md` and `USER-INPUT-VERBATIM.md`.

**Result/status:** useful baseline and navigation, **not** a complete snapshot of current implementation/research.

---

## `transit-ddr3-architecture` — central active memory-compute tile architecture

**PR:** #10.

**Purpose:** move from repeated host-RAM→GPU weight streaming toward weights resident beside local programmable compute and local DDR bandwidth.

**Method:** PCIe host/orchestrator → many logical memory-compute tiles → local DDR channels → local low-bit compute/reduction; host moves mainly commands/activations/results.

**Evidence/results:** exact signed INT4×INT8 bitplane math; executable NASM host kernel/runner; Python golden reference; archived host benchmarks; Weight Atlas/protocol code; SystemVerilog core; K3 sizing.

**Key sizing:** 38 logical tiles × 8 DDR3 channels = 304 channels; nominal DDR3-2133 calculation gives ~99.8 **weight-path tok/s equivalent** at the simple 52 GB/token lower bound.

**Does not prove:** finished 304-channel hardware, production K3 MXFP4/MXFP8 bridge, or ~100 end-to-end K3 tok/s.

**Read first:** `docs/08-EVIDENCE-BENCHMARKS.md`, `09-BITPLANE-MATH-AND-KERNEL.md`, `10-KIMI-K3-TARGET.md`, `11-DDR3-TILE-ARCHITECTURE.md`, `14-CURRENT-SOLUTION.md`, `benchmarks/README.md`.

---

## `docs/kimi-k3-ddr-cluster` — distributed K3 on cheap server memory

**PR:** #11.

**Purpose:** investigate full-model K3 capacity/execution using distributed DDR2/DDR3/DDR4 servers rather than only custom tiles.

**Method:** full-model sharding + layer/expert parallelism + explicit network/runtime model + one-node-first procurement/validation.

**Result/status:** architecture, hardware catalogue, throughput model, implementation/procurement plan and research log. Its core rule is: capacity can be computed; throughput must be measured on the complete path.

**Does not prove:** a claimed final K3 token rate on the proposed cluster.

**Read first:** `docs/07-KIMI-K3-DISTRIBUTED-RUNTIME.md`, `08-HARDWARE-DDR2-DDR3-DDR4.md`, `09-KIMI-K3-THROUGHPUT-MODEL.md`, `10-IMPLEMENTATION-PROCUREMENT-PLAN.md`, `11-RESEARCH-LOG-2026-08-16.md`.

---

## `research/heterogeneous-moe-kimi-v1` — NUMA CPU experts + GPU path

**PR:** #9.

**Purpose:** make very large MoE inference practical on R920-class heterogeneous machines without treating every cache miss as a mandatory PCIe stall.

**Method:** NUMA-local CPU routed-expert execution; GPU-resident non-routed/hot path; Q3/Q2/mixed formats; expert sharding and multi-GPU sensitivity; AVX-only old-Xeon benchmark path.

**Evidence/results:** analytical K2.5/K3 requirements, simulation/modeling, benchmark/runbook code, CI. Numerical requirements are acceptance gates unless explicitly labeled measured.

**Does not prove:** final end-to-end K3 speed on the physical target cluster.

**Read first:** `experiments/phase6-heterogeneous-kimi-runtime/README.md`, `RESULTS.md`, `RUNBOOK.md`, `SOURCE-MAP.md` where present on the branch, plus PR #9.

---

# 2026-09-16 Q4_K × Q8_K / LUT / META line

## `research/ram-lookup-compute-2026-09-16` — current semantic center of the CPU-kernel research

**PR:** #16. The branch evolved significantly after the original PR text; do not stop at the PR description.

**Purpose:** determine which extra-RAM representations, predecode/layout changes, LUTs, and NUMA placement can actually beat direct exact `Q4_K × Q8_K` work.

**Method progression:** synthetic materialized functions V1–V8 → real-Qwen lossy codebooks/PQ → exact original-Q4_K LUT → native C/C++ controls → lossless metadata/layout repacks → strict-Ivy SSSE3/AVX path → NUMA system topology.

**Measured lessons:**

- Synthetic large-working-set crossover appeared around 32 useful logical MACs/lookup; V8 was fast only in its restricted coded domain.
- Real-Qwen PQ/codebook weight approximations were too inaccurate to promote.
- Small exact managed nibble LUT beat its managed direct control slightly; a larger byte LUT lost badly.
- Geometry-correct native AVX2 made the early exact LUT path unattractive on the laptop.
- `x8meta` — lossless 8-row runtime repack/predecoded metadata — measured about ~1.11× standalone AVX2 operator gain on the i5-12500H with +2.78% runtime weight bytes; accepted for an integration experiment, not as default runtime.
- The strict-Ivy line later converged on **V27 META**: predecoded metadata + direct SSSE3 arithmetic as the strongest same-harness path documented in the branch.
- **V28 hot dictionary is a negative result:** 0.000% exact pattern coverage at tested thresholds; best fused path lost to META (~0.902× in the recorded 12-thread comparison).
- **V29 is the next physical gate:** keep V27 META unchanged per NUMA socket; NUMA-local row shards, one persistent worker pool/socket, duplicated activation, local-vs-remote controls, bit-exact correctness on the real dual E5-2680 v2 server.

**LUT/codebook decision:** not abandoned, but moved out of the default hot loop unless there is real amortization/locality. Retained roles include tiny activation-dependent tables reused over many rows, FPGA SRAM/BRAM materialized functions, numerically safe expert-prefetch signatures, activation/network compression, KV/state compression, and repeated decode/control transforms.

**Selection rule before another LUT experiment:** require at least one of: ~32+ useful ops/lookup; tiny intended-local table; amortized build cost; materially different near-memory hardware cost model; or prediction/compression-only use that does not replace authoritative arithmetic.

**Read first:** `benchmarks/ram-lookup-compute/README.md`, `V29-NUMA-PLAN-AND-LUT-CODEBOOK-PLACEMENT-2026-09-16.md`, `docs/adr/2026-09-16-q4k-x8meta-runtime-layout.md`, and the latest result/audit files on the branch.

**Does not prove:** full llama.cpp/model tok/s gain, server NUMA scaling, or that “RAM computes by itself.”

---

## `bench/q4k-q8k-native-avx2-2026-09-16` — native direct-kernel checkpoint/provenance

**PR:** #17; base is the RAM lookup-compute research line.

**Purpose:** establish a serious exact native `Q4_K × Q8_K` control before claiming LUT wins.

**Method:** real Qwen geometry; exact 4-row native kernel; beyond-LLC rotating working set; paired/shuffled runs; archived source checkpoints and cross-branch audit.

**Measured result:** controlled V10 median ~1.1809× and later V11 ~1.2640× versus the reproduced direct control on the i5-12500H standalone harness. A physical 4-row repack was slightly slower and rejected for that CPU at that point.

**Does not prove:** end-to-end llama.cpp token/s. Its AVX2 kernel cannot be copied to E5-2680 v2; Ivy Bridge needs SSSE3/AVX-specific work.

**Read first:** `docs/18-NATIVE-Q4K-Q8K-AVX2-2026-09-16.md`, `docs/19-CROSS-BRANCH-NATIVE-KERNEL-AUDIT-2026-09-16.md`, `benchmarks/ram-lookup-compute/native-q4k-q8k/README.md`.

---

## `bench/q4k-q8k-lut123-v21-2026-09-16` — exact LUT1/LUT2/LUT3 gate

**Purpose:** test exact tables representing one, two, or three Q4×Q8 products without changing Q4_K weights.

**Key correction:** raw Q4_K low/high nibbles do not automatically belong to the same 32-weight scale group; V21 repacks losslessly so pair/triple LUTs never mix incompatible scale/min groups.

**Tables:** LUT1 = 8 KiB universal; LUT2 = 32 MiB universal with 512 B active slice per activation pair; full universal LUT3 = 128 GiB, while laptop tests materialize only needed activation slices and report build cost separately.

**Correctness:** bit-exact gate against scalar/direct controls; both native and Ivy-compatible (`-march=ivybridge -mno-avx2`) self-tests are part of the harness.

**Status/result:** on the audit machine’s synthetic full-path validation, lookup candidates were slower than arithmetic controls; LUT2 was generally strongest among lookup candidates. This is a benchmark gate, **not a claimed LUT speedup**. Real E5-2680 v2 measurement remains authoritative.

**Read first:** `benchmarks/ram-lookup-compute/q4k-q8k-lut123-v21/README.md` and its runner/results.

---

# Model/runtime synthesis and comparisons

## `research/glm52-4x2-optimized-runtime-2026-08-18` — four-server GLM-5.2 design

**Purpose:** freeze a practical 4-server × 2-GPU/server prototype for GLM-5.2 Q3-class GGUF.

**Method:** llama.cpp/GGUF substrate + ByteMyWave distributed expert scheduler + NUMA-local host expert store + GPU hot-expert cache + CPU cold-expert fallback + async prefetch + routing trace + native GLM-5.2 MTP + MTP-aware prefetch + optional `ngram-simple`; per-GPU-family worker builds for legacy hardware.

**Important model framing:** ~743B total but ~39B active/token; routed experts execute concurrently when possible instead of treating it as a dense 743B stream.

**Status:** no full measurement on the exact four purchased servers. Documented ~1–1.4, ~1.4–2.1, ~2–3, and ~4 stretch ranges are planning/engineering targets for successive implementation tiers, not measured results. The document explicitly rejects a 10 tok/s claim on current evidence.

**Read first:** `docs/18-GLM52-4X2-OPTIMIZATION-EVIDENCE-2026-08-18.md`, `19-GLM52-4X2-DECISION-THROUGHPUT-AND-IMPLEMENTATION.md`, `20-GLM52-4X2-IMPLEMENTATION-BLUEPRINT.md`, `21-GLM52-4X2-SIMULATION-CORRECTION-2026-08-18.md`.

---

## `research/k3-consumer-gpu-vs-transit-2026-08-17` — apples-to-apples K3 comparison framework

**Purpose:** compare Transit with large consumer-GPU rigs without cheating by summing all physical bandwidth on one side and serializing the other.

**Method:** distinguish total physical bandwidth from bandwidth/compute that can participate concurrently for the same token/stage; include serial layer boundaries and collectives; classify external benchmark evidence.

**Result:** preserves the Transit ~99.8 weight-path roofline as a sizing statement, while rejecting naive conversions such as aggregate 120×RTX3060 GDDR bandwidth divided by active bytes/token as an end-to-end prediction.

**Read first:** `docs/16-K3-CONSUMER-GPU-VS-TRANSIT-REALITY-CHECK-2026-08-17.md`, `17-K3-PARALLELISM-BENCHMARK-PLAN.md`.

---

## `research/transit-cosoldex-dspark-2026-09-09` — context + target runtime + speculative layer

**Purpose:** combine three orthogonal savings: reduce executor input, make the expensive forward pass cheaper, and reduce expensive target decode steps.

**Method:** Co-Sol-Dex micro-agent retrievers/selectors → compact task context → large executor → speculative drafter/verification → Transit CPU/RAM backing model with GPUs as adaptive hot-weight/hot-expert cache+compute tier.

**Evidence/status:** session note preserves a user-reported Co-Sol-Dex benchmark (~2–3k executor input tokens vs ~24k control on that task) and measured laptop Transit kernel evidence, but old-Xeon aggregate numbers remain estimates until benchmarked. DSpark-style speculation is a research direction, not a measured ByteMyWave model result.

**Read first:** `docs/sessions/2026-09-09-transit-cosoldex-dspark.md`.

---

## `agent/transit-active-memory-chronicles` — agentic memory/retrieval subsystem

**PR:** #14.

**Purpose:** reduce agent task wall-clock and context waste rather than raw decoder tok/s.

**Method:** RAM-backed working tree, structural indexes, retriever guards, GPU batching, early termination, incremental updates/persistence; Chronicle Memory records evidence, provenance, confidence and superseded beliefs when compute is idle.

**Status:** documentation/design only in the PR; claimed speedups are illustrative until baseline-vs-active-memory benchmarks exist.

**Do not confuse with:** model-weight memory-compute tiles. This track is about agent/project memory and retrieval.

---

## `research/heretic-evo2-orchestrator-2026-08-29` — offline model prep + benign Evo2 tool gate

**PR:** #15.

**Purpose:** evaluate Heretic-prepared orchestrator candidates for benign Evo2 tool-use workflows without changing Transit’s token hot path.

**Method:** pin Heretic v1.4.0; preserve provenance/license; explicit GLM compatibility gates; benign tool-use quality/refusal evaluation.

**Status:** no GLM-5.3 or Kimi K3 abliteration success is claimed; no new token/s claim. Heretic is an offline dependency/research input.

---

# Hardware architecture / procurement branches

## `agent/transit-ddr2-server-fabric` — whole obsolete servers as memory-compute nodes

**PR:** #12.

**Purpose:** avoid designing the DDR physical layer by buying obsolete multi-socket servers whose OEM memory controllers, DIMM sockets, power and Linux support already work.

**Method:** nodes such as HP DL785/Sun X4640 keep weights resident; NUMA-local workers compute locally; network carries activations/results rather than full weights.

**Status:** documented alternative/fallback architecture and rooflines. Decisive gate is measured real-server Gweights/s + wall power + scale-out, not the analytical roofline.

---

## `agent/apu-ddr3-uma-k3` — Carrizo APU / DDR3 UMA track

**PR:** #13.

**Purpose:** use local DDR3 directly visible to an APU/iGPU so the Radeon compute engine works near its memory without discrete-GPU PCIe weight streaming.

**Method:** CPU for control/runtime/network; iGPU for model-shaped arithmetic; tensor/expert sharding and collectives for scale-out.

**Important correction:** installed DDR channels do not automatically add for one autoregressive token. Branch carries separate 52 GB/token lower bound, ~57.94 GB screening model and ~136.59 GB conservative envelope until exact trace accounting exists.

**Status:** 1–4 prototype nodes are a measurement path; large fleet is NO-GO until validation gates pass.

---

## `agent/transit-memory-controller-trade-study` — what hardware actually costs

**Purpose:** answer the narrower question: what is the cheapest/watt-efficient way to obtain working DDR interfaces next to a small arithmetic engine?

**Method:** compare CPUs, FPGAs and minimal logic by **cost + watts per usable DDR interface**, not FLOPs. Examine DDR PHY/training/I/O complexity, wider buses, controller silicon, and custom ASIC implications.

**Result/decisions:** arithmetic is not the hard hardware problem; DDR PHY/controller/training is. Cheap old CPUs are attractive because they bundle the hard interface. External gates after CPU reads usually add another transfer. Discrete-transistor DDR controller rejected. Custom ASIC is out of current scope. Procurement metric should include lei/sustained TB/s and watts/sustained TB/s.

**Read first:** `docs/14-MEMORY-CONTROLLER-TRADE-STUDY.md`.

---

## `hardware/transit-ddr2-tile-reva` — manufacturable BEE3-style tile handoff

**Purpose:** produce a real DDR2 memory-compute board without inventing a new DDR PHY.

**Method:** proven BEE3-style topology: 4× Virtex-5 FF1136, 2 DDR2 channels/FPGA, 8 independent channels total, 16 DDR2 ECC RDIMM sockets, local Transit compute; simple Ethernet/JTAG-first host/control path.

**Engineering gate:** Xilinx ISE must prove two legal DDR2 controllers/FPGA, pin/bank legality and timing before fabrication.

**Status:** engineering specification + BOM target, **not existing Gerbers or measured hardware**. Plan is five prototypes first, then scale only after DDR/kernel/thermal/scale-out tests.

**Read first:** `hardware/transit-ddr2-tile-reva/ENGINEERING-BRIEF.md`.

---

## `research/transit-surplus-full-bom-2026-08-17` — surplus tile/fan-out catalogue

**Purpose:** find real cheap programmable/memory-rich surplus hardware and practical PCIe fan-out routes.

**Method:** board-by-board identification, memory topology, programming/JTAG/open-source status, measured/public bandwidth when available, current-market price snapshots, and scaling arithmetic.

**Key findings:** Storey Peak is a strong cheap single-DDR-channel microtile; YPCB-00338-1P1 is a strong low-risk two-channel proof board; Nallatech P385 and many later SmartNIC/FPGA/SoC/GPU candidates are documented. Storey Peak’s public ~9.662 GB/s board read figure is used for a conservative scaling example; hundreds of boards would be required for the 100 weight-path-tok/s class.

**Status:** hardware research/BOM, not a finished machine. Marketplace price/stock is ephemeral and must be rechecked.

**Read first:** `docs/16-SURPLUS-TILES-FANOUT-AND-100TOK-BOM-2026-08-17.md`, then candidate update docs 17–27.

---

## `research/ten-r920-procurement-build` — ten-R920 capstone procurement design

**Purpose:** turn the heterogeneous MoE work into a staged physical build proposal.

**Method:** 10× R920; per node 4 E7-class sockets, 256 GB DDR3, 1× RTX3060 12 GB fixed-path shard, 5× GTX1060-class routed workers, FDR56 network, external GPU power and true x16 risers.

**Corrected model arithmetic:** K3 Stable LatentMoE routed payload model is ~48.620B routed weights/token; branch planning model ~25.83 GB routed payload/token. A 600 GB/s worker-feed calculation gives ~23.23 routed-token-equivalents/s — explicitly not output tok/s.

**Status:** procurement/engineering plan. 15–25 output tok/s “first useful”, 25–35 stretch, etc. are engineering targets, not measurements. Procurement must stage 1→2→4→10; estimated complete power scale is facility-class.

**Read first:** `experiments/phase7-ten-r920-procurement/README.md`, `BOM.md`, `ASSEMBLY.md`, `VALIDATION.md`.

---

# Historical proof chain — do not mistake these for current final architecture

## `bench/h2d-overlap-proof-v1` — PR #1

**Question:** can H2D transfer of tile N+1 hide under real GPU compute of tile N with only two fixed VRAM weight slots?

**Method:** pinned synthetic host weights, two CUDA streams, real FP16 cuBLAS GEMM, sequential vs overlapped pipeline, correctness and starvation metrics.

**Role today:** foundational mechanism proof/provenance. Does not prove a large model fits/runs.

## `model/real-weight-atlas-proof-v1` — PR #2

**Question:** does the same mechanism work with exact bytes from real safetensors checkpoints?

**Method:** header-only atlas, deterministic tensor/tile packing, SHA-256 provenance, real F16/BF16 CUDA path.

**Role today:** provenance for Weight Atlas/real-byte streaming. Does not prove true model graph scheduling.

## `quant/q4-streaming-proof-v1` — PR #4

**Question:** can host weights remain compressed so PCIe carries Q4 and GPU dequantizes only the active tile?

**Method:** explicit Q4 format, compressed VRAM slots, GPU dequant then FP16 GEMM, correctness and transfer/dequant/starvation timing.

**Role today:** compressed-streaming proof. Not the final fused runtime.

## `research/feasibility-map-v1` — PR #6

**Question:** in what compute/bandwidth regimes is streaming physically worthwhile?

**Method:** calibrated roofline + direct measured starvation map + experiment matrix.

**Role today:** performance-methodology provenance. Analytical maps must not be quoted as measured model speed.

## `hardware/r920-rtx3060-simulation-v1` — PR #8

**Question:** what happens when earlier streaming assumptions are mapped onto a concrete Dell R920 + RTX3060-class target?

**Method:** NUMA/PCIe hardware profile + discrete-event simulation + cache sensitivity.

**Role today:** physical-topology/simulation bridge into Phase 6. Simulation is not benchmark evidence.

---

# Cross-repository provenance archive

## `archive/import-evoworld-transit-tile-research-2026-09-16` — draft PR #18

**Purpose:** prevent loss of Transit research that had been written under `dan3s2020/EvoWorld/docs/hardware/transit/TRANSIT_TILE_RESEARCH.md` instead of ByteMyWave.

**Contents:** verbatim archived Transit tile/fan-out research plus a provenance README under `docs/imported/evoworld/transit/`.

**Source blob:** `868583fcd08395f85745c8b27d94434f9433d347`.

**Interpretation:** archive/evidence source only. It must not silently override newer ByteMyWave branch decisions. Compare any hardware claim with the newer Transit/surplus/controller-study branches before treating it as current.

---

# Topic-specific reading recipes

## Kimi K3 / ~100 tok/s / 304 channels

Read in this order:

1. `CURRENT-ARCHITECTURE.md` on `main`.
2. This map.
3. `transit-ddr3-architecture/docs/10-KIMI-K3-TARGET.md`.
4. `docs/11-DDR3-TILE-ARCHITECTURE.md` and `docs/14-CURRENT-SOLUTION.md` on that branch.
5. PR #11 distributed-K3 track and PR #9 heterogeneous runtime for alternative execution paths.
6. If comparing GPUs vs Transit, use `research/k3-consumer-gpu-vs-transit-2026-08-17`.

Never convert the ~99.8 weight-path roofline into measured K3 decode.

## Q4_K / Q8_K / LUT / codebook / V27 META / V29 / Ivy Bridge

Read in this order:

1. `research/ram-lookup-compute-2026-09-16/benchmarks/ram-lookup-compute/README.md`.
2. V29 plan (`V29-NUMA-PLAN-AND-LUT-CODEBOOK-PLACEMENT-2026-09-16.md`).
3. `docs/adr/2026-09-16-q4k-x8meta-runtime-layout.md` for the AVX2 x8meta decision boundary.
4. PR #17/native branch for the native control/provenance.
5. `bench/q4k-q8k-lut123-v21-2026-09-16` only when evaluating exact multi-product LUTs.

Current direction for the real dual-socket E5-2680 v2 gate is **V27 META + NUMA row sharding**, not “add another LUT because LUT is in the branch name.”

## Four-server purchased setup / GLM-5.2

Read `research/glm52-4x2-optimized-runtime-2026-08-18` first, then the Q4_K/META branch if CPU exact-kernel work is relevant. Do not substitute ten-R920 procurement assumptions for the four-DL360p setup.

## Hardware purchase / FPGA tile / PCIe fan-out

Use three layers:

1. `agent/transit-memory-controller-trade-study` for the silicon/interface economics.
2. `research/transit-surplus-full-bom-2026-08-17` for concrete surplus boards/prices/fan-out leads.
3. `hardware/transit-ddr2-tile-reva` only for the custom BEE3-style DDR2 board path.

Use the EvoWorld import only as archived provenance.

## Agentic speed / memory / context reduction

- Context/retrieval: `agent/transit-active-memory-chronicles` and `research/transit-cosoldex-dspark-2026-09-09`.
- Model inference hardware/runtime: Transit/GLM/Kimi branches above.
- Do not report fewer executor prompt tokens as higher decoder tok/s; measure end-to-end task latency separately.

---

# Supersession and contradiction rules

When two documents disagree, do not delete history or pick the one with the more impressive number. Resolve in this order:

1. Newer measured result on the same hardware/input/harness.
2. Newer branch-specific decision/result document that explicitly audits the older result.
3. Raw measurement/result artifact.
4. Current architecture document.
5. Analytical/simulation document.
6. Older baseline/research note.

Examples:

- V28’s failed exact hot-dictionary experiment does not erase the synthetic V7/V8 lookup result; it narrows where that mechanism applies.
- V27 META/V29 NUMA direction supersedes “try another dictionary threshold” for the current strict-Ivy hot loop unless a new reuse source is demonstrated.
- `x8meta` is accepted for an integration experiment but is not automatically the final E5-2680 v2 layout.
- The 304-channel ~99.8 number remains an architecture-sizing weight-path roofline, not a measured end-to-end K3 result.
- Old marketplace prices are research history, not current quotes.

---

# Map maintenance contract

Any agent that creates a meaningful new research branch/PR should update this map **without overwriting another branch’s technical work**. Add or amend the semantic entry with:

- branch/PR;
- purpose/question;
- base/evolution relationship if known;
- method;
- measured result or current evidence class;
- negative results/rejected mechanisms;
- what is explicitly not proven;
- files to read first;
- next falsifiable gate;
- whether it supersedes, complements, or archives older work.

A branch name is only an address. **This map is the index of what the address actually contains.**
