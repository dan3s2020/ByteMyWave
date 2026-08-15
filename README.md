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

Se măsoară separat:

- compressed physical H2D GB/s;
- source-equivalent model feed rate;
- GPU dequant time;
- GEMM time;
- starvation;
- wall time;
- sequential-Q4 vs overlapped-Q4 full-output correctness;
- offline Q4 RMS/max weight error și SNR.

Run:

```powershell
python -m pip install numpy
.\scripts\run-q4-proof.ps1 -ModelDir "D:\models\some-safetensors-model"
```

Protocol complet: [`experiments/phase3-q4-gpu-dequant/README.md`](experiments/phase3-q4-gpu-dequant/README.md).

## Ce vrem să vedem

Strong support pentru un shape rămâne:

```text
correctness_ok = true
steady_starvation_pct <= 10%
steady_hidden_transfer_pct >= 80%
```

Dar nu ne interesează un singur punct norocos. Sweep-ul pe `M` trebuie să arate unde compute-ul începe să acopere transferul și cât de mult mută Q4 pragul față de 16-bit.

## Structura proiectului

### Source of truth

- [`PROJECT_STATE.md`](PROJECT_STATE.md) — starea curentă, branch chain, facts vs hypotheses, next gates.
- [`docs/00-PROJECT-INTENT.md`](docs/00-PROJECT-INTENT.md) — scop și constrângeri.
- [`docs/01-MODEL-MEMORY.md`](docs/01-MODEL-MEMORY.md) — ce reprezintă weights/activations/workspace.
- [`docs/02-WEIGHT-ATLAS.md`](docs/02-WEIGHT-ATLAS.md) — modelul de adresare/tile map.
- [`docs/03-STREAMING-RUNTIME.md`](docs/03-STREAMING-RUNTIME.md) — memory choreography.
- [`docs/04-COMPRESSION.md`](docs/04-COMPRESSION.md) — compression/dequant strategy.
- [`docs/05-OPEN-QUESTIONS.md`](docs/05-OPEN-QUESTIONS.md) — ipoteze încă nedemonstrate.
- [`docs/06-COLLABORATION.md`](docs/06-COLLABORATION.md) — reguli pentru 5 workstreams simultane.
- [`docs/07-H3-RELEASE-GATE.md`](docs/07-H3-RELEASE-GATE.md) — gate-ul de verificare MiniMax H3.
- [`docs/TRANSCRIPT.md`](docs/TRANSCRIPT.md) — conversația de origine.
- [`docs/USER-INPUT-VERBATIM.md`](docs/USER-INPUT-VERBATIM.md) — inputul original al utilizatorului.

### Tooling / runtime

- [`tools/safetensors_atlas.py`](tools/safetensors_atlas.py)
- [`tools/pack_stream_tiles.py`](tools/pack_stream_tiles.py)
- [`tools/build_runtime_schedule.py`](tools/build_runtime_schedule.py)
- [`tools/quantize_q4_pack.py`](tools/quantize_q4_pack.py)
- [`tools/summarize_runs.py`](tools/summarize_runs.py)
- [`src/tensorwave_stream_proof.cu`](src/tensorwave_stream_proof.cu)
- [`src/tensorwave_real_weight_proof.cu`](src/tensorwave_real_weight_proof.cu)
- [`src/tensorwave_q4_stream_proof.cu`](src/tensorwave_q4_stream_proof.cu)

### ADRs

- [`docs/adr/0001-phase1-fixed-vram-ring.md`](docs/adr/0001-phase1-fixed-vram-ring.md)
- [`docs/adr/0002-real-weight-atlas-pack.md`](docs/adr/0002-real-weight-atlas-pack.md)
- [`docs/adr/0003-q4-host-representation-gpu-dequant.md`](docs/adr/0003-q4-host-representation-gpu-dequant.md)

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
