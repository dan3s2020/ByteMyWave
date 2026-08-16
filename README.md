# TensorWave

TensorWave este proiectul pentru investigarea unor arhitecturi de inferență în care **modele AI mult mai mari decât memoria acceleratorului sau a unui singur server sunt păstrate în RAM ieftin și executate prin streaming/distribuție controlată**.

Proiectul are acum două piste complementare:

1. **RAM → GPU window/cache** — modelul stă în RAM-ul hostului, iar un GPU cu VRAM mic (ținta inițială: ~4 GB) este folosit ca accelerator de calcul și fereastră/cache de lucru.
2. **Distributed Kimi K3 / cheap-memory cluster** — checkpoint-ul complet este împărțit între mai multe servere DDR2/DDR3/DDR4, fiecare nod păstrează și calculează shard-ul local, iar clusterul este expus către PC/agent ca un singur endpoint Kimi K3.

## Ideea centrală

Ținta proiectului este să verificăm dacă putem face util un sistem cu:

- mult RAM ieftin, inclusiv ECC server RAM din generații vechi;
- GPU NVIDIA cu VRAM mic pentru pista inițială;
- sau mai multe noduri CPU/RAM pentru pista distribuită K3;
- model mare ținut în RAM în formă compactă/quantizată;
- transfer predictibil al datelor active, nu mutarea permanentă a întregului model;
- prefetch și overlap acolo unde topologia permite;
- dequantizare cât mai aproape de operația matematică;
- un atlas/graf al tensorilor și tile-urilor pentru adresare, dependențe, integritate și resharding;
- măsurători reale de bandwidth, NUMA, network latency și kernel throughput înainte de a declara performanță.

Modelele principale discutate în proiect sunt **MiniMax H3** pentru pista RAM→GPU și **Kimi K3** pentru pista de cluster distribuit.

Kimi K3 este o țintă deosebit de interesantă deoarece este un MoE de 2.8T parametri, cu 104B parametri activați per token și 16 experți selectați din 896. Asta permite să investigăm nu doar pipeline/layer sharding pentru capacitate, ci și **expert parallelism** pentru a utiliza simultan bandwidth-ul mai multor noduri.

## Stadiul actual

**Faza 0/1 — documentare, formalizare și proiectarea benchmark-urilor.**

Nu există încă o implementare completă K3 în acest repository și nu este revendicat niciun token/s măsurat. Documentația nouă separă explicit:

- ce este verificat din documentația oficială a modelului/hardware-ului;
- ce este doar un lead de marketplace;
- calculele de capacitate;
- limitele teoretice de bandwidth/compute;
- condițiile măsurabile care trebuie îndeplinite pentru a demonstra X tokeni/s.

În această fază sunt păstrate:

1. cerințele și ideile originale;
2. modelul mental al memoriei și execuției;
3. arhitectura propusă a runtime-ului;
4. întrebările deschise;
5. regulile pentru lucru simultan în echipă;
6. transcriptul conversației din care a pornit proiectul;
7. inputul utilizatorului păstrat separat, verbatim, ca să nu se piardă intenția originală prin rezumare;
8. cercetarea hardware DDR2/DDR3/DDR4 pentru Kimi K3;
9. modelul matematic de throughput și protocolul de benchmark.

## Structura documentației

- [`docs/00-PROJECT-INTENT.md`](docs/00-PROJECT-INTENT.md) — scop, constrângeri și ipoteze.
- [`docs/01-MODEL-MEMORY.md`](docs/01-MODEL-MEMORY.md) — ce există efectiv în memoria modelului.
- [`docs/02-WEIGHT-ATLAS.md`](docs/02-WEIGHT-ATLAS.md) — reprezentarea modelului ca hartă/graf/tile-uri.
- [`docs/03-STREAMING-RUNTIME.md`](docs/03-STREAMING-RUNTIME.md) — cum ar trebui alimentat GPU-ul din RAM.
- [`docs/04-COMPRESSION.md`](docs/04-COMPRESSION.md) — unde se face quantizarea/decompresia și de ce.
- [`docs/05-OPEN-QUESTIONS.md`](docs/05-OPEN-QUESTIONS.md) — lucrurile care trebuie demonstrate experimental.
- [`docs/06-COLLABORATION.md`](docs/06-COLLABORATION.md) — organizarea pentru lucru simultan.
- [`docs/07-KIMI-K3-DISTRIBUTED-RUNTIME.md`](docs/07-KIMI-K3-DISTRIBUTED-RUNTIME.md) — arhitectura completă pentru împărțirea Kimi K3 pe mai multe servere și expunerea ca un singur API.
- [`docs/08-HARDWARE-DDR2-DDR3-DDR4.md`](docs/08-HARDWARE-DDR2-DDR3-DDR4.md) — servere, plăci, FRU/part numbers, compatibilitate RAM, calcule de capacitate și lead-uri de piață.
- [`docs/09-KIMI-K3-THROUGHPUT-MODEL.md`](docs/09-KIMI-K3-THROUGHPUT-MODEL.md) — modelul de bandwidth/compute/network și condițiile pentru a demonstra X tokeni/s fără a inventa benchmark-uri.
- [`docs/10-IMPLEMENTATION-PROCUREMENT-PLAN.md`](docs/10-IMPLEMENTATION-PROCUREMENT-PLAN.md) — ordinea de achiziție, benchmark, kernel development, distributed runtime și integrarea cu agentic frameworks.
- [`docs/TRANSCRIPT.md`](docs/TRANSCRIPT.md) — copia conversației relevante care a dus la proiect.
- [`docs/USER-INPUT-VERBATIM.md`](docs/USER-INPUT-VERBATIM.md) — mesajele utilizatorului care au definit proiectul, păstrate separat.

## Pista 1 — RAM mare, VRAM mic

TensorWave nu încearcă să pretindă că 4 GB VRAM „devin” fizic 24/64/128 GB VRAM.

În schimb, încearcă să facă astfel încât **GPU-ul să nu aibă nevoie să vadă simultan decât fereastra activă de date necesară operației curente**, iar restul modelului să fie ținut în RAM și pregătit din timp.

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

## Pista 2 — Kimi K3 pe cluster de RAM ieftin

Aici nu încercăm să facem „remote RAM” transparent prin Ethernet. Fiecare nod deține weights locale și execută calculele pentru acele weights. Rețeaua mută activations, expert dispatch și rezultate, nu checkpoint-ul întreg la fiecare token.

```text
PC / Kimi Code / agent
        |
        v
TensorWave OpenAI-compatible gateway
        |
        v
controller / router
        |
        +-------- inference fabric --------+
        |               |                 |
     worker 0         worker 1          worker N
     local RAM        local RAM         local RAM
     K3 shard         K3 shard          K3 shard
```

Capacitatea este ușor de demonstrat matematic: de exemplu, 2.304–2.56 TB de RAM agregat depășește checkpoint-ul oficial de aproximativ 1.56 TB. Performanța nu este dedusă din capacitate; ea trebuie demonstrată cu kernel-uri reale și benchmark end-to-end.

Ținta inițială de acceptanță pentru pista K3 este:

```text
full K3 checkpoint
no expert pruning
batch = 1
>= 1.0 decoded token/s after warm-up
```

Aceasta este **o țintă de test, nu o performanță deja revendicată**.
