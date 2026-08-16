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

**Faza 0 — documentare și formalizarea ideii.**

Nu există încă implementare în acest repository. În această fază sunt păstrate:

1. cerințele și ideile originale;
2. modelul mental al memoriei și execuției;
3. arhitectura propusă a runtime-ului;
4. întrebările deschise;
5. regulile pentru lucru simultan în echipă;
6. transcriptul conversației din care a pornit proiectul;
7. inputul utilizatorului păstrat separat, verbatim, ca să nu se piardă intenția originală prin rezumare.

## Structura documentației

- [`docs/00-PROJECT-INTENT.md`](docs/00-PROJECT-INTENT.md) — scop, constrângeri și ipoteze.
- [`docs/01-MODEL-MEMORY.md`](docs/01-MODEL-MEMORY.md) — ce există efectiv în memoria modelului.
- [`docs/02-WEIGHT-ATLAS.md`](docs/02-WEIGHT-ATLAS.md) — reprezentarea modelului ca hartă/graf/tile-uri.
- [`docs/03-STREAMING-RUNTIME.md`](docs/03-STREAMING-RUNTIME.md) — cum ar trebui alimentat GPU-ul din RAM.
- [`docs/04-COMPRESSION.md`](docs/04-COMPRESSION.md) — unde se face quantizarea/decompresia și de ce.
- [`docs/05-OPEN-QUESTIONS.md`](docs/05-OPEN-QUESTIONS.md) — lucrurile care trebuie demonstrate experimental.
- [`docs/06-COLLABORATION.md`](docs/06-COLLABORATION.md) — organizarea pentru 5 persoane care lucrează simultan.
- [`docs/TRANSCRIPT.md`](docs/TRANSCRIPT.md) — copia conversației relevante care a dus la proiect.
- [`docs/USER-INPUT-VERBATIM.md`](docs/USER-INPUT-VERBATIM.md) — toate mesajele utilizatorului care au definit proiectul, păstrate fără reformulare.

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
