# ByteMyWave — Fast Semantic Repository Map

> **Read this as a routing board, not as a substitute for evidence.**
>
> The canonical detailed semantic index is [`docs/REPOSITORY-MAP.md`](docs/REPOSITORY-MAP.md). `AGENTS.md` already requires agents to read `CURRENT-ARCHITECTURE.md` and the detailed map before making project-wide claims. This root file exists so a human or agent landing on the repository can see the whole territory immediately instead of choosing work from branch names.

_Last synchronized with branch inventory: 2026-09-16 — 28/28 current branches represented._

## How to use this map

For any question, use this sequence:

1. Read `CURRENT-ARCHITECTURE.md` for the current high-level architecture.
2. Use the table below to find the relevant research line by **purpose/method/result**, not by branch name.
3. Open the corresponding entry in `docs/REPOSITORY-MAP.md` for evidence boundaries, negative results, files to read first, and supersession relationships.
4. Read the branch-local result/decision/README files before quoting numbers.
5. Distinguish measured end-to-end decode, measured subsystem/kernel results, simulation, analytical rooflines, weight-path tok/s equivalents, and engineering targets.

---

## 28/28 branch routing board

| Branch | What it is actually for | Method / mechanism | Current result / status | Read next |
|---|---|---|---|---|
| `main` | Baseline + navigation surface, not the whole current project | Project intent → model memory → Weight Atlas → streaming → compression → collaboration | Canonical landing surface; newer work lives across branches/PRs | `CURRENT-ARCHITECTURE.md`, `AGENTS.md`, `docs/REPOSITORY-MAP.md` |
| `transit-ddr3-architecture` | Main active memory-compute Transit architecture | PCIe fan-out → local DDR channels → local low-bit compute/reduction; weights stay local | Exact bitplane math + host/reference/RTL evidence; ~99.8 tok/s is a **weight-path roofline**, not measured K3 decode | `docs/10-KIMI-K3-TARGET.md`, `11-DDR3-TILE-ARCHITECTURE.md`, `14-CURRENT-SOLUTION.md` |
| `docs/kimi-k3-ddr-cluster` | Distributed Kimi K3 on cheap server memory | Full-model sharding + layer/expert parallelism + network/runtime/procurement model | Architecture/procurement research; throughput still requires physical-path measurement | `docs/07-KIMI-K3-DISTRIBUTED-RUNTIME.md` through `11-RESEARCH-LOG-2026-08-16.md` |
| `research/heterogeneous-moe-kimi-v1` | NUMA CPU routed experts + GPU fixed/hot path | NUMA-local expert execution, GPU resident path, mixed quantization, expert sharding | Analytical/simulation + benchmark framework; not final physical K3 tok/s | `experiments/phase6-heterogeneous-kimi-runtime/` |
| `research/ram-lookup-compute-2026-09-16` | Semantic center of Q4_K×Q8_K/LUT/codebook/META/NUMA research | Synthetic LUTs → real codebooks → exact LUT → native controls → metadata repacks → strict-Ivy → NUMA | **V27 META** is the preserved strict-Ivy baseline; V28 dictionary negative; later V30–V33 branches extend the exact path | `benchmarks/ram-lookup-compute/README.md`, `V29-NUMA-PLAN-AND-LUT-CODEBOOK-PLACEMENT-2026-09-16.md` |
| `bench/v30-ramforge-2026-09-16` | First exact post-V27 space-for-time layout sweep | X8F metadata/FP32 predecode, PACK8/COL8, SUM4, exact base+residual, VEC4/VEC8, META-Atlas | Some same-run laptop wins appeared, but later paired-stability work showed X8F/large-RAM single-run gains were not robust; SUM4/BR negative | `benchmarks/ram-lookup-compute/v30-ramforge/README.md` |
| `bench/v31-fusionforge-2026-09-16` | Exact arithmetic fusion after V30 | Fused-scale accumulation (FSA), row-block 2/4/8/16, prefetch, EXP8/SCALED16, physical-core enumeration, HOT/COLD controls, VEC2/4/8/16 | Established FSA as the promising mechanism; large headline outliers were later rejected as scheduling/baseline noise | `benchmarks/ram-lookup-compute/v31-fusionforge/README.md`, `REAL-LAPTOP-RESULTS-2026-09-16.md` |
| `bench/v32-stabilityforge-2026-09-16` | Separate real kernel wins from Windows/scheduler noise | Alternating V27→candidate / candidate→V27 AB/BA paired timing; median+p10+p90; exactness gate | Pinned real Qwen tensor: stable HOT `FSA_RB16` at 4 P-cores = **1.26162x median, p10 1.10758**, diff=0; stable 1T COLD FSA gains ~1.19x; 8/12T laptop runs mix P+E and are not server-scaling evidence | `benchmarks/ram-lookup-compute/v32-stabilityforge/RESULTS-LAPTOP-2026-09-16.md` |
| `bench/v33-pairfuse-2026-09-16` | Remove redundant Q4_K work inside the stable FSA path | Fuse low/high nibble group pairs that share packed bytes; vectorize four-row min correction + horizontal reduction; PAIR4/PAIR8; rotating-address beyond-LLC paired benchmark | Exact strict-Ivy implementation prepared and development sanity passes; **real pinned laptop/server run pending**. V27 and V31 FSA controls remain mandatory | `benchmarks/ram-lookup-compute/v33-pairfuse/README.md`, `RUN.ps1` |
| `bench/q4k-q8k-native-avx2-2026-09-16` | Native exact CPU control/provenance for LUT claims | Real Qwen geometry, native 4-row kernel, beyond-LLC paired runs | AVX2 control improvements measured on i5-12500H; not directly portable to E5-2680 v2 | `docs/18-NATIVE-Q4K-Q8K-AVX2-2026-09-16.md`, `19-CROSS-BRANCH-NATIVE-KERNEL-AUDIT-2026-09-16.md` |
| `bench/q4k-q8k-lut123-v21-2026-09-16` | Exact LUT1/LUT2/LUT3 experiment | Lossless Q4_K repack + 1/2/3-product lookup tables | Correctness proven in harness; lookup candidates slower than arithmetic controls on audit machine; real Ivy server remains authority | `benchmarks/ram-lookup-compute/q4k-q8k-lut123-v21/README.md` |
| `research/glm52-4x2-optimized-runtime-2026-08-18` | Four-server GLM-5.2 runtime plan | llama.cpp/GGUF + expert scheduler + NUMA + GPU cache + CPU fallback + MTP/prefetch | Practical architecture/targets; exact purchased four-server setup not yet fully measured | `docs/18-GLM52-4X2-OPTIMIZATION-EVIDENCE-2026-08-18.md` through `21-...SIMULATION-CORRECTION...md` |
| `research/k3-consumer-gpu-vs-transit-2026-08-17` | Fair K3 comparison: consumer GPUs vs Transit | Same-token/stage concurrency, effective bandwidth, serial boundaries, collectives | Rejects naive aggregate-bandwidth comparisons; preserves Transit roofline only as sizing | `docs/16-K3-CONSUMER-GPU-VS-TRANSIT-REALITY-CHECK-2026-08-17.md` |
| `research/transit-cosoldex-dspark-2026-09-09` | Context reduction + expensive executor + speculative verification | Co-Sol-Dex retrievers/selectors + compact context + Transit backing + speculative layer | Orthogonal latency-saving research; context reduction is not decoder tok/s | `docs/sessions/2026-09-09-transit-cosoldex-dspark.md` |
| `agent/transit-active-memory-chronicles` | Agent/project memory and retrieval, not model-weight compute | RAM working tree, structural indexes, retriever guards, Chronicle evidence/provenance | Design/documentation track; benchmark needed for latency claims | PR #14 / branch docs |
| `research/heretic-evo2-orchestrator-2026-08-29` | Offline model preparation/tool-use evaluation | Pinned Heretic + compatibility/quality/refusal gates | No new Transit hot-path speed claim; no claimed GLM/K3 abliteration success | PR #15 / branch docs |
| `agent/transit-ddr2-server-fabric` | Whole obsolete servers as memory-compute nodes | Reuse OEM memory controllers/DIMMs/NUMA; local workers; network carries activations/results | Alternative/fallback architecture; gate is measured Gweights/s + wall power | PR #12 / branch docs |
| `agent/apu-ddr3-uma-k3` | Carrizo APU/iGPU using local DDR3 UMA | CPU control + iGPU model arithmetic + shard/collective scale-out | Prototype path only; large fleet NO-GO until measured gates | PR #13 / branch docs |
| `agent/transit-memory-controller-trade-study` | Find cheapest/watt-efficient working DDR interface next to compute | Compare CPU/FPGA/minimal logic by cost + watts per usable DDR interface | DDR PHY/controller/training is the hard part; old CPUs attractive because they bundle it | `docs/14-MEMORY-CONTROLLER-TRADE-STUDY.md` |
| `hardware/transit-ddr2-tile-reva` | Manufacturable DDR2 FPGA tile handoff | BEE3-style 4×Virtex-5, 8 channels, 16 RDIMMs, local compute | Engineering spec/BOM, not fabricated/benchmarked hardware | `hardware/transit-ddr2-tile-reva/ENGINEERING-BRIEF.md` |
| `research/transit-surplus-full-bom-2026-08-17` | Surplus FPGA/SmartNIC/tile/fan-out candidate ledger | Board identification, RAM topology, JTAG/tooling, bandwidth, prices, scaling arithmetic | Hardware/BOM research; prices/stock are historical snapshots | `docs/16-SURPLUS-TILES-FANOUT-AND-100TOK-BOM-2026-08-17.md` + update docs |
| `research/ten-r920-procurement-build` | Ten-R920 staged physical build proposal | 10× R920 + CPU/GPU routed workers + FDR56 + staged 1→2→4→10 procurement | Procurement/engineering targets, not measured output tok/s | `experiments/phase7-ten-r920-procurement/` |
| `bench/h2d-overlap-proof-v1` | First proof that H2D of tile N+1 can hide under GPU compute of tile N | Pinned host weights, two CUDA streams, FP16 cuBLAS, sequential vs overlapped | Foundational mechanism proof only; not full-model feasibility | PR #1 / branch benchmark docs |
| `model/real-weight-atlas-proof-v1` | Move from synthetic bytes to exact bytes from real checkpoints | Safetensors atlas, deterministic packing, SHA-256 provenance, F16/BF16 CUDA path | Provenance for Weight Atlas/real-byte streaming; not true model graph scheduling | PR #2 / branch docs |
| `quant/q4-streaming-proof-v1` | Test compressed Q4 host→GPU streaming | Q4 host weights, compressed VRAM slots, GPU dequant then FP16 GEMM | Compression/streaming mechanism proof; not final fused runtime | PR #4 / branch docs |
| `research/feasibility-map-v1` | Determine where streaming is physically worthwhile | Calibrated roofline + measured starvation map + experiment matrix | Performance-methodology provenance; analytical maps are not measured model speed | PR #6 / branch maps/docs |
| `hardware/r920-rtx3060-simulation-v1` | Map streaming assumptions onto a concrete R920 + RTX3060 target | NUMA/PCIe profile + discrete-event simulation + cache sensitivity | Physical-topology/simulation bridge; simulation is not benchmark evidence | PR #8 / branch simulation docs |
| `archive/import-evoworld-transit-tile-research-2026-09-16` | Preserve Transit research accidentally living in EvoWorld | Verbatim import + provenance README under isolated namespace | Archive/evidence only; must not override newer ByteMyWave decisions | `docs/imported/evoworld/transit/` on draft PR #18 |

---

## Topic shortcuts

- **Kimi K3 / 304 channels / ~100 tok/s:** start at `transit-ddr3-architecture`, then compare `docs/kimi-k3-ddr-cluster` and `research/heterogeneous-moe-kimi-v1`.
- **Q4_K / Q8_K / LUT / codebook / META / V27+:** start at `research/ram-lookup-compute-2026-09-16`, then follow `bench/v30-ramforge-2026-09-16` → `bench/v31-fusionforge-2026-09-16` → `bench/v32-stabilityforge-2026-09-16` → `bench/v33-pairfuse-2026-09-16`. V32 is the current stability filter; V33 is the next exact mechanism under test.
- **Four purchased servers / GLM-5.2:** start at `research/glm52-4x2-optimized-runtime-2026-08-18`; do not substitute assumptions from the ten-R920 branch.
- **FPGA/DDR tile procurement/fan-out:** use `agent/transit-memory-controller-trade-study` → `research/transit-surplus-full-bom-2026-08-17` → `hardware/transit-ddr2-tile-reva` as needed.
- **Agent context/memory:** `agent/transit-active-memory-chronicles` + `research/transit-cosoldex-dspark-2026-09-09`; do not confuse this with model-weight memory compute.
- **Historical evolution of streaming:** follow PR #1 → #2 → #4 → #6 → #8.

## Evidence rule

Never flatten these into one performance number:

1. measured end-to-end model decode;
2. measured kernel/subsystem result;
3. simulation;
4. analytical roofline;
5. weight-path tok/s equivalent;
6. engineering target;
7. external hardware/marketplace observation.

For details, contradictions, rejected mechanisms, supersession rules, and exact files to read first, continue to **`docs/REPOSITORY-MAP.md`**.
