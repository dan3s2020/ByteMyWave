# ByteMyWave repository instructions

> STOP before feasibility judgments: read `CURRENT-ARCHITECTURE.md` first.

Do not fall into these common misreadings:
- passive DDR3 is not the compute engine; active tile logic computes/reduces next to local DDR3;
- the topology is 38 active memory-compute tiles × 8 independent DDR3 channels = 304 channels, not hundreds of passive DIMMs on one motherboard;
- Transit intends to keep large expert weights local to tiles and use PCIe mainly for commands/activations/reduced results;
- ~99.8 tok/s is a weight-path roofline/equivalent, not measured end-to-end Kimi K3 throughput;
- `main` is not the complete current project state.

Treat `AGENTS.md` as the canonical repository discovery protocol.

Read `CURRENT-ARCHITECTURE.md`, `docs/REPOSITORY-MAP.md` and inspect relevant public branches/open PRs before project-wide conclusions.

For Kimi K3, Transit GPU, DDR3, 304 channels, 38 tiles or ~100 tok/s questions, inspect PRs #9, #10 and #11 and their branches. Keep measured results, analytical/simulated rooflines, weight-path tok/s equivalent and end-to-end Kimi K3 tok/s as separate evidence classes.

Before a project-wide verdict, state which current-state sources were actually inspected. If branch/PR access is unavailable, say the review is incomplete rather than substituting a generic passive-RAM architecture.
