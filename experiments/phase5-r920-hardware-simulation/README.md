# Phase 5 — Dell R920 + RTX 3060 hardware simulation

## Purpose

This phase maps the existing TensorWave Phase-3/Phase-4 runtime assumptions onto a concrete low-cost server target:

```text
Dell PowerEdge R920
4 x Intel Xeon E7-4890 v2 reference profile
~1 TiB balanced DDR3 ECC
1 x RTX 3060 12 GB as the first stock-chassis GPU candidate
```

The goal is not to claim measured performance before hardware exists. The goal is to answer, reproducibly:

1. how the R920 NUMA domains, RAM and PCIe topology line up with TensorWave;
2. what the current two-slot Q4 runtime would look like on an RTX 3060;
3. which operating regimes should be transfer-bound or compute-bound;
4. what changes with multiple GPUs;
5. what must be measured on the real machine before treating the platform as validated.

Detailed hardware analysis:

- `docs/10-R920-HARDWARE-PLATFORM.md`

Simulator:

- `tools/simulate_r920_tensorwave.py`

Reference result:

- `results/reference/SIMULATION-REPORT.md`
- `results/reference/simulation.json`

Tests:

- `tests/test_r920_simulator.py`

---

## Source contracts preserved

The simulator deliberately follows the current project rather than inventing a new runtime.

### Phase 3 Q4 wire format

```text
Q4_SYM_G32_F32S
32 weights/group
20 physical bytes/group
0.625 bytes/parameter
```

### Phase 3 two-slot ownership rule

```text
slot(i) = i % 2
copy(i) waits compute(i-2) before overwriting that slot
compute(i) waits copy(i)
```

The simulator models a serialized H2D copy engine and serialized compute stream with the same dependencies.

### Phase 4 crossover equation

```text
M_cross = bytes_per_param * (1-resident_fraction) * effective_FLOPS
          ---------------------------------------------------------
                         2 * H2D_bandwidth
```

The default reference simulation uses **12 GB/s effective H2D** and **10 TFLOP/s effective dense-linear compute** only as assumptions. They are not measurements of an R920 or RTX 3060.

---

## Reference workload

MiniMax H3 remains the long-term model target, but TensorWave does not currently treat an exact H3 parameter count/configuration as verified.

Therefore this phase uses one of the generic model sizes already present in the Feasibility Map:

```text
70B dense reference model
```

This is a stress profile, not a claim that MiniMax H3 is 70B.

Reference Q4 wire size:

```text
70e9 * 0.625 B = 43.75 GB
```

---

## Experiment H1 — NUMA placement

For a 4-socket R920 with 1 TiB balanced evenly:

```text
CPU1 NUMA domain ~256 GiB
CPU2 NUMA domain ~256 GiB
CPU3 NUMA domain ~256 GiB
CPU4 NUMA domain ~256 GiB
```

The first real TensorWave GPU worker must allocate/register its staging and pinned host memory on the NUMA domain local to the chosen PCIe slot.

Measure on hardware:

```text
local NUMA pinned H2D GB/s
remote NUMA pinned H2D GB/s
local-vs-remote latency
CPU memory bandwidth while H2D is active
```

A large local/remote penalty would make NUMA-aware Weight Atlas placement mandatory rather than optional.

---

## Experiment H2 — Phase-3 ring on RTX 3060

Reference tile geometry:

```text
K = 8192
N = 256
tiles = 32
M = 1,4,16,64,128,256,512,1024,2048
```

The current Phase-3 working-set equation is preserved:

```text
fixed VRAM = X + Q4 slot A + Q4 slot B + one FP16 dequant tile + Y
```

The simulation shows the exact fixed working-set size for every M and the copy/compute timeline implied by the existing ownership schedule.

Important: this phase does not yet simulate a fused Q4->MMA kernel. It matches the current full-FP16-dequant-tile implementation.

---

## Experiment H3 — 70B dense feasibility

With the default assumptions:

```text
H2D = 12 GB/s
effective GEMM = 10 TFLOP/s
resident fraction = 0
Q4 wire = 0.625 B/parameter
```

we expect a crossover near:

```text
M ~= 260
```

This means the analytical prediction is hostile to batch=1 decode and much more favorable to large-batch/prefill reuse.

The real machine must replace the assumed H2D and compute values with Phase-4 calibration.

---

## Experiment H4 — multiple RTX 3060s

Two different modes must not be confused.

### Mode A — independent workers

This is compatible with the current single-GPU runtime design:

```text
GPU0 serves request/batch group A
GPU1 serves request/batch group B
GPU2 serves request/batch group C
```

Each worker owns its CUDA streams, its two-slot Q4 ring, its NUMA-local pinned source window and its persistent cache allocation.

Aggregate throughput can scale if each GPU has an independent local H2D path and the server memory/NUMA fabric does not become the bottleneck.

### Mode B — one model sharded across GPUs

The simulator reports an equal-shard optimistic lower bound, but current TensorWave code does **not** implement this mode.

A real sharded runtime would need to model/implement:

```text
activation placement
cross-GPU synchronization
collectives or peer transfers
pipeline/tensor parallel schedule
output reduction/concatenation
NUMA placement per shard
```

Do not present the ideal sharded number as a current capability.

---

## Experiment H5 — use the 12 GB VRAM as cache, not merely as a tiny ring

The current ring consumes only a small fraction of a 12 GB RTX 3060 for representative tiles.

That suggests a more useful use of the extra VRAM:

```text
fixed streaming ring/workspace
+
persistent compressed hot-weight/hot-expert cache
```

Run sensitivity sweeps such as:

```text
cache = 0, 2, 4, 6, 8 GiB per GPU
```

For each cache size measure/recompute:

```text
resident fraction
streamed GB/step
M crossover
starvation
wall time
cache hit rate on a real graph
```

A cache policy should ultimately be Weight-Atlas-aware, not first-come-first-served.

---

## Experiment H6 — physical R920 validation gates

Before buying multiple consumer GPUs or claiming the platform is production-suitable, validate the first card.

Required checks:

```text
1. RTX 3060 physically fits the selected slot/riser.
2. Server boots reliably with the card installed.
3. A supported/safe 8-pin auxiliary power path exists.
4. Card stays within temperature limits under sustained TensorWave load.
5. PCIe negotiates the expected width/generation.
6. pinned local-NUMA H2D bandwidth is measured.
7. remote-NUMA H2D bandwidth is measured.
8. simultaneous H2D + GEMM overlap is measured.
9. Phase-3 correctness remains green.
10. Phase-4 M crossover is calibrated against the simulator prediction.
```

Only after H6 should a second/third consumer GPU configuration be treated as a hardware plan.

---

## Run the simulator

From the repository root:

```powershell
python .\tools\simulate_r920_tensorwave.py `
  --output-dir .\runs\r920-3060-reference
```

Reference with an 8 GiB persistent compressed cache:

```powershell
python .\tools\simulate_r920_tensorwave.py `
  --output-dir .\runs\r920-3060-cache8 `
  --cache-gib-per-gpu 8
```

Change the timing assumptions after hardware calibration:

```powershell
python .\tools\simulate_r920_tensorwave.py `
  --output-dir .\runs\r920-measured `
  --h2d-gbps 11.4 `
  --effective-tflops 7.8
```

The numbers above are examples of CLI usage, not expected R920 measurements.

---

## Run tests

```powershell
python -m unittest discover -s tests -p "test_r920_simulator.py" -v
```

The tests lock the simulator to important project invariants:

```text
Q4 wire density = 0.625 B/parameter
Phase-4 crossover equation
70B Q4 transfer-floor arithmetic
Phase-3 two-slot ring behavior near crossover
compute-bound behavior after crossover
```

---

## Reference simulation result

The committed reference run uses:

```text
70B dense
1 TiB host RAM
RTX 3060 12 GB profile
12 GB/s assumed effective H2D
10 TFLOP/s assumed effective dense-linear compute
no persistent cache
K=8192, N=256, 32 tiles
```

Its purpose is to make the assumptions and expected qualitative behavior reviewable before hardware arrives.

It is **not a benchmark**.

---

## Falsification criteria

The R920 hypothesis should be weakened or rejected if real measurements show any of the following:

```text
local pinned H2D is unexpectedly poor or unstable
NUMA traffic collapses under concurrent GPU streams
GPU cannot be powered/cooled safely in the chassis
copy and compute do not overlap despite correct CUDA scheduling
measured crossover is far worse than the calibrated model and the difference cannot be explained
host memory subsystem becomes the dominant bottleneck with the intended GPU count
```

A negative result is useful: it tells us whether to change the server, the GPU chassis, the cache strategy, or the runtime itself.
