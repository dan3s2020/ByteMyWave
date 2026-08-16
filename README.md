# ByteMyWave

ByteMyWave este proiectul pentru investigarea unei arhitecturi de inferență în care **un model AI mult mai mare decât VRAM-ul disponibil este păstrat în RAM**, iar un GPU cu VRAM mic (ținta inițială: ~4 GB) este folosit ca **accelerator de calcul + fereastră/cache de lucru**, nu ca depozit complet al modelului.

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

Modelul principal de interes discutat până acum este **MiniMax H3**, dar arhitectura trebuie gândită generic, astfel încât aceeași idee să poată fi verificată și pe alte modele Transformer/diffusion/LLM.

## Stadiul actual

**Faza 1 — proof experimental pentru transfer/compute overlap.**

Faza 0 de documentare este încheiată. Prima implementare nu încearcă încă să ruleze MiniMax H3. Mai întâi testează separat ipoteza fundamentală:

> în timp ce GPU-ul calculează tile-ul `N`, poate transferul RAM → VRAM al tile-ului `N+1` să fie ascuns suficient încât GPU-ul să nu rămână flămând după date?

Proof-ul actual folosește:

- weights FP16 rezidente în pinned RAM;
- GEMM real prin cuBLAS;
- două sloturi VRAM fixe;
- stream CUDA separat pentru H2D și compute;
- CUDA events pentru dependențe;
- baseline secvențial versus pipeline overlapped;
- măsurare explicită a `GPU starvation`;
- verificarea numerică a rezultatului pipeline-ului față de baseline.

Experimentul este documentat în [`experiments/phase1-h2d-overlap/README.md`](experiments/phase1-h2d-overlap/README.md).

Pe Windows, după instalarea CUDA Toolkit + CMake + Visual Studio C++ tools:

```powershell
.\scripts\run-proof.ps1
```

Scriptul face implicit un sweep pe mai multe valori `M` pentru a găsi punctul în care compute-ul devine suficient de lung încât transferul următorului tile să fie ascuns.

## Structura proiectului

### Documentație de bază

- [`docs/00-PROJECT-INTENT.md`](docs/00-PROJECT-INTENT.md) — scop, constrângeri și ipoteze.
- [`docs/01-MODEL-MEMORY.md`](docs/01-MODEL-MEMORY.md) — ce există efectiv în memoria modelului.
- [`docs/02-WEIGHT-ATLAS.md`](docs/02-WEIGHT-ATLAS.md) — reprezentarea modelului ca hartă/graf/tile-uri.
- [`docs/03-STREAMING-RUNTIME.md`](docs/03-STREAMING-RUNTIME.md) — cum ar trebui alimentat GPU-ul din RAM.
- [`docs/04-COMPRESSION.md`](docs/04-COMPRESSION.md) — unde se face quantizarea/decompresia și de ce.
- [`docs/05-OPEN-QUESTIONS.md`](docs/05-OPEN-QUESTIONS.md) — lucrurile care trebuie demonstrate experimental.
- [`docs/06-COLLABORATION.md`](docs/06-COLLABORATION.md) — organizarea pentru 5 persoane care lucrează simultan.
- [`docs/TRANSCRIPT.md`](docs/TRANSCRIPT.md) — copia conversației relevante care a dus la proiect.
- [`docs/USER-INPUT-VERBATIM.md`](docs/USER-INPUT-VERBATIM.md) — mesajele utilizatorului care au definit proiectul, păstrate fără reformulare.

### Implementare experimentală

- [`src/tensorwave_stream_proof.cu`](src/tensorwave_stream_proof.cu) — benchmark CUDA/cuBLAS pentru pipeline-ul fixed-VRAM.
- [`scripts/run-proof.ps1`](scripts/run-proof.ps1) — build + sweep automat pe Windows.
- [`experiments/phase1-h2d-overlap/`](experiments/phase1-h2d-overlap/) — protocolul experimentului și criteriile de succes/eșec.
- [`docs/adr/0001-phase1-fixed-vram-ring.md`](docs/adr/0001-phase1-fixed-vram-ring.md) — decizia arhitecturală pentru primul proof.

## Principiul care nu trebuie pierdut

ByteMyWave nu încearcă să pretindă că 4 GB VRAM „devin” fizic 24/64/128 GB VRAM.

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
