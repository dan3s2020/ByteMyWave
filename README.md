# TensorWave

TensorWave este proiectul pentru investigarea unei arhitecturi de inferență în care **un model AI mult mai mare decât VRAM-ul disponibil este păstrat în RAM**, iar un GPU cu VRAM mic (ținta inițială: ~4 GB) este folosit ca **accelerator de calcul + fereastră/cache de lucru**, nu ca depozit complet al modelului.

## Ideea centrală

Ținta proiectului este să verificăm dacă putem face util un sistem cu:

- mult RAM ieftin (inclusiv RAM vechi/server, dacă este suficient pentru alimentarea pipeline-ului);
- GPU NVIDIA cu VRAM mic, inclusiv 4 GB;
- model mare ținut în RAM în formă compactă/quantizată;
- transfer predictibil RAM → VRAM pe bucăți mici;
- prefetch al următoarei bucăți în timp ce GPU-ul calculează bucata curentă;
- sloturi VRAM fixe, fără allocate/free continuu;
- dequantizare/decompresie pe GPU, cât mai aproape de operația matematică;
- un atlas/graf al tensorilor și tile-urilor pentru adresare, dependențe, integritate și eventual reconstrucție;
- metrica principală: **GPU starvation time**, nu simplul timp brut de copiere prin PCIe.

Modelul principal de interes discutat până acum este **MiniMax H3**, dar arhitectura este intenționat generică pentru Transformer/diffusion/LLM.

## Stadiul actual

**Faza 2 — real checkpoint bytes + Weight Atlas + fixed-VRAM streaming.**

Faza 1 izolează ipoteza fundamentală folosind weights sintetice:

> în timp ce GPU-ul calculează tile-ul `N`, poate transferul RAM → VRAM al tile-ului `N+1` să fie ascuns suficient încât GPU-ul să nu rămână flămând după date?

Faza 2 păstrează același contract CUDA, dar scoate weights sintetice din ecuație. Tooling-ul actual:

1. citește headerele shard-urilor `safetensors` fără să materializeze tensorii;
2. construiește `weight-atlas.json` cu nume, shape, dtype și byte offsets exacte;
3. selectează determinist tensori reali rank-2 cu același dtype și `K`;
4. copiază direct row-slices din checkpoint într-un `weights.pack` fără conversie numerică;
5. generează `execution-plan.json` cu proveniență și SHA-256 pentru fiecare tile;
6. încarcă pack-ul în pinned RAM;
7. rulează exact bytes checkpoint-ului prin două sloturi VRAM fixe și cuBLAS GEMM;
8. compară baseline-ul secvențial cu pipeline-ul overlapped folosind un accumulator FP32 la care contribuie fiecare tile.

Experimentul este documentat în [`experiments/phase2-real-weight-streaming/README.md`](experiments/phase2-real-weight-streaming/README.md).

Pe Windows:

```powershell
.\scripts\run-real-weight-proof.ps1 -ModelDir "D:\models\some-safetensors-model"
```

Scriptul construiește automat atlasul + pack-ul, citește geometria selectată și face sweep pe mai multe valori `M`.

### MiniMax H3

H3 există oficial, dar TensorWave nu hardcodează încă dimensiuni interne neverificate. Statusul checkpoint-ului și regula de verificare sunt documentate în [`docs/07-H3-RELEASE-GATE.md`](docs/07-H3-RELEASE-GATE.md). Când checkpoint-ul oficial devine disponibil, Faza 2 trebuie să poată porni doar indicând folderul lui local.

## Structura proiectului

### Documentație de bază

- [`docs/00-PROJECT-INTENT.md`](docs/00-PROJECT-INTENT.md) — scop, constrângeri și ipoteze.
- [`docs/01-MODEL-MEMORY.md`](docs/01-MODEL-MEMORY.md) — ce există efectiv în memoria modelului.
- [`docs/02-WEIGHT-ATLAS.md`](docs/02-WEIGHT-ATLAS.md) — reprezentarea modelului ca hartă/graf/tile-uri.
- [`docs/03-STREAMING-RUNTIME.md`](docs/03-STREAMING-RUNTIME.md) — cum ar trebui alimentat GPU-ul din RAM.
- [`docs/04-COMPRESSION.md`](docs/04-COMPRESSION.md) — unde se face quantizarea/decompresia și de ce.
- [`docs/05-OPEN-QUESTIONS.md`](docs/05-OPEN-QUESTIONS.md) — lucrurile care trebuie demonstrate experimental.
- [`docs/06-COLLABORATION.md`](docs/06-COLLABORATION.md) — organizarea pentru 5 persoane care lucrează simultan.
- [`docs/07-H3-RELEASE-GATE.md`](docs/07-H3-RELEASE-GATE.md) — ce este și ce nu este verificat despre checkpoint-ul H3.
- [`docs/TRANSCRIPT.md`](docs/TRANSCRIPT.md) — copia conversației relevante care a dus la proiect.
- [`docs/USER-INPUT-VERBATIM.md`](docs/USER-INPUT-VERBATIM.md) — mesajele utilizatorului care au definit proiectul, păstrate fără reformulare.

### Implementare experimentală

**Phase 1**

- [`src/tensorwave_stream_proof.cu`](src/tensorwave_stream_proof.cu) — benchmark CUDA/cuBLAS cu weights sintetice în pinned RAM.
- [`scripts/run-proof.ps1`](scripts/run-proof.ps1) — build + sweep Phase 1 pe Windows.
- [`experiments/phase1-h2d-overlap/`](experiments/phase1-h2d-overlap/) — protocol și criterii Phase 1.
- [`docs/adr/0001-phase1-fixed-vram-ring.md`](docs/adr/0001-phase1-fixed-vram-ring.md) — fixed two-slot VRAM ring.

**Phase 2**

- [`tools/safetensors_atlas.py`](tools/safetensors_atlas.py) — parser header-only pentru Weight Atlas.
- [`tools/pack_stream_tiles.py`](tools/pack_stream_tiles.py) — exact-byte tile packer + execution plan + SHA-256.
- [`src/tensorwave_real_weight_proof.cu`](src/tensorwave_real_weight_proof.cu) — streaming CUDA/cuBLAS pentru weights F16/BF16 reale.
- [`scripts/run-real-weight-proof.ps1`](scripts/run-real-weight-proof.ps1) — pipeline complet checkpoint → atlas → pack → benchmark.
- [`experiments/phase2-real-weight-streaming/`](experiments/phase2-real-weight-streaming/) — protocolul Phase 2.
- [`tests/test_safetensors_tools.py`](tests/test_safetensors_tools.py) — test exact pentru offsets, pack bytes și SHA-256.

## Principiul care nu trebuie pierdut

TensorWave nu încearcă să pretindă că 4 GB VRAM „devin” fizic 24/64/128 GB VRAM.

În schimb, încearcă să facă astfel încât **GPU-ul să nu aibă nevoie să vadă simultan decât fereastra activă de date necesară operației curente**, iar restul modelului să fie ținut în RAM și pregătit din timp.

Schema conceptuală:

```text
Large RAM model store
        |
        | preplanned / indexed / compressed tiles
        v
Pinned host window / DMA queue
        |
        | PCIe async transfer
        v
Small fixed VRAM ring/cache
        |
        | dequantize + compute
        v
GPU kernels / Tensor Cores
```

Obiectivul experimental este să aflăm dacă transferul poate fi ascuns suficient de mult sub compute încât sistemul să rămână productiv chiar cu VRAM foarte mic.
