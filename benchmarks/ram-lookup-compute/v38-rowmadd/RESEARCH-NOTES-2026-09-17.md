# V38 ROWMADD — research notes — 2026-09-17

External mechanism check:

- Microsoft BitNet CPU optimization uses configurable row/column tiling and weight/activation parallelism. This reinforces the direction of compiled row-major layouts and row blocking rather than large random DRAM LUTs.
- uops.info reports Ivy Bridge XMM `VPMADDUBSW` as one uop on p0, latency 5, throughput 1/cycle. `VPSHUFB` has latency 1 and measured throughput 0.5 cycles on Ivy Bridge. This makes compact shuffle-based scale expansion attractive if it removes `PMULLD` from the row-lane dot path.

V38 therefore separates three questions:

1. Can direct `PMADDWD(sum16, scale16)` beat V37 with no extra weight bytes? (`ROW4MADD`)
2. Does splitting accumulation into two independent chains improve ILP? (`ROW4MADD_I2`)
3. Does spending +192 B/tile on pre-expanded scale vectors outperform runtime PSHUFB? (`ROW4MADD_PRE`)

Generated strict-Ivy assembly was inspected: the compact ROW4MADD path emits `VPSHUFB` and `VPMADDWD` for the new scale stage, with no `VPMULLD` in that scale stage.

Development synthetic runs are screening evidence only and must not be reported as the Transit result. The pinned real Qwen tensor and then physical E5-2680 v2 decide.