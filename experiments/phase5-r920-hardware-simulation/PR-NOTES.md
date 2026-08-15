# Phase 5 review notes

This branch should be reviewed as an extension of `research/feasibility-map-v1`.

Review boundaries:

- no measured R920/RTX 3060 performance is claimed;
- no MiniMax H3 parameter count is inferred;
- Phase-3 Q4 density and two-slot scheduling are preserved;
- Phase-4 crossover math is preserved;
- multi-GPU model sharding is explicitly marked unimplemented;
- physical RTX 3060 count is not inferred from electrical PCIe x16 count;
- all timing assumptions are intended to be replaced by real calibration.

Key new engineering proposals:

```text
NUMA-aware host placement
topology metadata in the execution plan
persistent compressed VRAM cache
one independent worker per GPU first
local-vs-remote NUMA H2D measurement
```
