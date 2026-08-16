# Claude instructions for ByteMyWave

> STOP before technical feasibility judgments: read `CURRENT-ARCHITECTURE.md` first.

Common traps to avoid:
- DDR3 itself is not the compute engine; active tile logic performs compute/reduction next to local DDR3.
- The documented topology is 38 active memory-compute tiles × 8 independent DDR3 channels = 304 channels, not hundreds of passive DIMMs on one motherboard.
- Transit keeps large expert weights local to tiles; PCIe is intended mainly for commands/activations/reduced results.
- ~99.8 tok/s is a weight-path architecture-sizing equivalent, not a measured end-to-end Kimi K3 result.
- `main` is not the complete current project state.

Use `AGENTS.md` as the canonical repository-reading protocol.

Before making current-state claims, inspect `CURRENT-ARCHITECTURE.md`, `docs/REPOSITORY-MAP.md`, open pull requests and relevant non-default branches.

For Kimi K3 / Transit GPU / DDR3 / ~100 tok/s questions, PR #9, PR #10 and PR #11 are mandatory reading. Never conflate analytical rooflines or weight-path tok/s equivalent with measured end-to-end Kimi K3 tok/s.

Before a project-wide feasibility verdict, state which current-state sources you actually inspected.

If branch/PR access is unavailable, explicitly scope the answer to `main` rather than claiming the missing material does not exist.
