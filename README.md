# TensorWave

TensorWave investighează o arhitectură de inferență în care **modelul poate fi mult mai mare decât VRAM-ul disponibil**: modelul stă în RAM într-o reprezentare compactă, iar GPU-ul cu VRAM mic este tratat ca **accelerator + fereastră/cache de lucru**, nu ca depozit complet al modelului.

## Ținta

Sistemul urmărit:

- mult RAM ieftin, inclusiv RAM server vechi dacă măsurătorile îl permit;
- GPU NVIDIA cu VRAM mic, ținta inițială fiind ~4 GB;
- model/index/execuție cunoscute înainte de runtime;
- tile-uri deterministic adresabile;
- fixed VRAM slots, fără `cudaMalloc/cudaFree` per tile;
- `copy(N+1)` în paralel cu `compute(N)`;
- modelul păstrat quantizat în RAM când este avantajos;
- transferul prin PCIe făcut în forma comprimată;
- dequantizarea cât mai târziu posibil, ideal în GPU chiar lângă operația matematică;
- măsurăm **GPU starvation / unhidden transfer**, nu confundăm bandwidth-ul brut cu costul efectiv.

Modelul final de interes este MiniMax H3, dar infrastructura este intenționat generică pentru Transformer/diffusion/LLM. Detaliile H3 sunt acceptate ca fapte numai când provin din checkpoint/config/implementare oficială; vezi [`docs/07-H3-RELEASE-GATE.md`](docs/07-H3-RELEASE-GATE.md).

## Stadiul actual

### Phase 1 — synthetic H2D/compute overlap

Branch: `bench/h2d-overlap-proof-v1` — draft PR #1.

Demonstrează mecanismul minim:

```text
pinned RAM
  -> fixed VRAM slot A/B
  -> real FP16 cuBLAS GEMM
```

Baseline-ul secvențial și pipeline-ul overlapped execută aceeași ordine și se compară matematic. Sunt măsurate H2D, compute, wall time, starvation și hidden-transfer estimate.

Run:

```powershell
.\scripts\run-proof.ps1
```

### Phase 2 — real checkpoint bytes + Weight Atlas

Branch: `model/real-weight-atlas-proof-v1` — draft PR #2.

Înlocuiește weights sintetice cu bytes reali din `safetensors`:

```text
safetensors headers
 -> weight-atlas.json
 -> exact row tiles
 -> execution-plan.json
 -> runtime-schedule.json
 -> weights.pack
 -> pinned RAM
 -> two fixed VRAM slots
 -> F16/BF16 cuBLAS GEMM
```

Nu se deserializează numeric tensorii pentru atlas/pack. Fiecare tile are proveniență, offsets și SHA-256. Static scheduler-ul materializează dinainte slot assignment + wait dependencies, deci measured loop nu caută „ce tensor urmează”.

Run:

```powershell
.\scripts\run-real-weight-proof.ps1 -ModelDir "D:\models\some-safetensors-model"
```

CUDA compile CI pentru Phase 2 este verde; target-hardware timing rămâne gate separat.

### Phase 3 — Q4 în RAM, Q4 prin PCIe, dequant pe GPU

Branch: `quant/q4-streaming-proof-v1` — draft PR #4.

Aici atacăm direct cantitatea de date transferată.

Formatul proof curent:

```text
Q4_SYM_G32_F32S
32 weights/group
4-byte float32 scale
16 packed signed-int4 bytes
20 bytes/group
```

Față de F16/BF16:

```text
64 bytes -> 20 bytes
31.25% din bytes
3.2x compresie
5 effective bits/weight incluzând scale-ul
```

Pipeline:

```text
large Q4 host store
        |
        | compressed cudaMemcpyAsync
        v
 Q4 slot A / Q4 slot B
        |
        | custom CUDA dequant
        v
 one reusable FP16 tile
        |
        v
    cuBLAS GEMM
```

Se măsoară separat compressed physical H2D GB/s, source-equivalent model feed rate, GPU dequant, GEMM, starvation, wall time și correctness.

Run:

```powershell
python -m pip install numpy
.\scripts\run-q4-proof.ps1 -ModelDir "D:\models\some-safetensors-model"
```

Protocol complet: [`experiments/phase3-q4-gpu-dequant/README.md`](experiments/phase3-q4-gpu-dequant/README.md).

### Phase 4 — measured feasibility map

Branch: `research/feasibility-map-v1`.

Phase 4 transformă întrebarea „merge?” într-un operating envelope pe:

```text
M / batch / prefill reuse
K,N tile geometry
wire bytes/parameter
measured H2D GB/s
measured effective GEMM TFLOP/s
resident/cache fraction
active fraction
```

Ecuația roofline principală:

```text
M_cross = bytes_per_param * (1-resident_fraction) * effective_FLOPS
          ---------------------------------------------------------
                         2 * H2D_bandwidth
```

Protocol: [`experiments/phase4-feasibility-map/README.md`](experiments/phase4-feasibility-map/README.md).

### Phase 5 — Dell R920 + RTX 3060 hardware simulation

Branch: `hardware/r920-rtx3060-simulation-v1`.

Am mapat arhitectura actuală TensorWave pe un profil concret de hardware ieftin:

```text
Dell PowerEdge R920
4 × Xeon E7-4890 v2
~1 TiB DDR3 ECC balanced
RTX 3060 12 GB
```

Hardware analysis: [`docs/10-R920-HARDWARE-PLATFORM.md`](docs/10-R920-HARDWARE-PLATFORM.md).

Simulator reproductibil: [`tools/simulate_r920_tensorwave.py`](tools/simulate_r920_tensorwave.py).

Experiment/protocol: [`experiments/phase5-r920-hardware-simulation/README.md`](experiments/phase5-r920-hardware-simulation/README.md).

Reference result: [`experiments/phase5-r920-hardware-simulation/results/reference/SIMULATION-REPORT.md`](experiments/phase5-r920-hardware-simulation/results/reference/SIMULATION-REPORT.md).

Reference assumptions:

```text
70B generic dense reference
Q4 v1 = 0.625 B/param
12 GB/s effective H2D assumption
10 TFLOP/s effective dense-linear assumption
no persistent cache
```

Reference prediction:

```text
M_cross ~= 260
M=1   -> strongly transfer-bound
M=256 -> near-balanced
M>=512 -> compute-bound in the idealized model
```

The current ring is tiny versus 12 GB VRAM for the representative tile geometry, which makes **persistent compressed caching** one of the strongest next optimization targets. These are simulation results, not measured R920 performance.

## Ce vrem să vedem

Strong support pentru un shape rămâne:

```text
correctness_ok = true
steady_starvation_pct <= 10%
steady_hidden_transfer_pct >= 80%
```

Dar nu ne interesează un singur punct norocos. Sweep-ul pe `M` trebuie să arate unde compute-ul începe să acopere transferul și cât de mult mută Q4/cache pragul față de 16-bit.

## Structura proiectului

### Source of truth

- [`PROJECT_STATE.md`](PROJECT_STATE.md) — starea curentă, branch chain, facts vs hypotheses, next gates.
- [`docs/00-PROJECT-INTENT.md`](docs/00-PROJECT-INTENT.md) — scop și constrângeri.
- [`docs/01-MODEL-MEMORY.md`](docs/01-MODEL-MEMORY.md) — ce reprezintă weights/activations/workspace.
- [`docs/02-WEIGHT-ATLAS.md`](docs/02-WEIGHT-ATLAS.md) — modelul de adresare/tile map.
- [`docs/03-STREAMING-RUNTIME.md`](docs/03-STREAMING-RUNTIME.md) — memory choreography.
- [`docs/04-COMPRESSION.md`](docs/04-COMPRESSION.md) — compression/dequant strategy.
- [`docs/05-OPEN-QUESTIONS.md`](docs/05-OPEN-QUESTIONS.md) — ipoteze încă nedemonstrate.
- [`docs/06-COLLABORATION.md`](docs/06-COLLABORATION.md) — reguli pentru workstreams simultane.
- [`docs/07-H3-RELEASE-GATE.md`](docs/07-H3-RELEASE-GATE.md) — gate-ul de verificare MiniMax H3.
- [`docs/08-PRIOR-ART-AND-NOVELTY.md`](docs/08-PRIOR-ART-AND-NOVELTY.md) — prior art/novelty boundary.
- [`docs/09-WHY-LARGE-LLM-WORKS-OR-FAILS.md`](docs/09-WHY-LARGE-LLM-WORKS-OR-FAILS.md) — dense decode/prefill/batching/MoE feasibility reasoning.
- [`docs/10-R920-HARDWARE-PLATFORM.md`](docs/10-R920-HARDWARE-PLATFORM.md) — R920/RAM/NUMA/PCIe/RTX3060 hardware profile and limits.
- [`docs/TRANSCRIPT.md`](docs/TRANSCRIPT.md) — conversația de origine.
- [`docs/USER-INPUT-VERBATIM.md`](docs/USER-INPUT-VERBATIM.md) — inputul original al utilizatorului.

### Tooling / runtime

- [`tools/safetensors_atlas.py`](tools/safetensors_atlas.py)
- [`tools/pack_stream_tiles.py`](tools/pack_stream_tiles.py)
- [`tools/build_runtime_schedule.py`](tools/build_runtime_schedule.py)
- [`tools/quantize_q4_pack.py`](tools/quantize_q4_pack.py)
- [`tools/build_feasibility_map.py`](tools/build_feasibility_map.py)
- [`tools/calibrate_feasibility_map.py`](tools/calibrate_feasibility_map.py)
- [`tools/simulate_r920_tensorwave.py`](tools/simulate_r920_tensorwave.py)
- [`src/tensorwave_stream_proof.cu`](src/tensorwave_stream_proof.cu)
- [`src/tensorwave_real_weight_proof.cu`](src/tensorwave_real_weight_proof.cu)
- [`src/tensorwave_q4_stream_proof.cu`](src/tensorwave_q4_stream_proof.cu)

### ADRs

- [`docs/adr/0001-phase1-fixed-vram-ring.md`](docs/adr/0001-phase1-fixed-vram-ring.md)
- [`docs/adr/0002-real-weight-atlas-pack.md`](docs/adr/0002-real-weight-atlas-pack.md)
- [`docs/adr/0003-q4-host-representation-gpu-dequant.md`](docs/adr/0003-q4-host-representation-gpu-dequant.md)
- [`docs/adr/0004-feasibility-map-calibrated-roofline.md`](docs/adr/0004-feasibility-map-calibrated-roofline.md)

## Principiul care nu trebuie pierdut

TensorWave nu afirmă că 4 GB VRAM devin fizic 24/64/128 GB.

Ținta este mai puternică și mai precisă:

> GPU-ul trebuie să vadă numai working set-ul necesar calculului curent, iar scheduler-ul trebuie să se asigure că următorul working set este deja disponibil când GPU-ul îl cere.

Long-term path:

```text
large compressed RAM model
        |
        | static graph-derived schedule
        v
compressed H2D ring
        |
        | dequant fragments only when needed
        v
register/shared-memory tiles
        |
        v
Tensor Core MMA
```

Dacă ajungem acolo, VRAM-ul nu mai este „mărimea modelului”; este dimensiunea ferestrei active de calcul.
