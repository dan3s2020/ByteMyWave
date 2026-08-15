# TensorWave Conversation Transcript — Continuation

Date: 2026-08-15

This file continues `docs/TRANSCRIPT.md`, which intentionally remains preserved as the original capture. This continuation records the visible user/assistant discussion that followed repository creation. Tool invocation internals are omitted; visible conclusions and project actions are preserved.

---

## User

Ok ,pentru proof experimental avem nevoie de un cod care sa producă experimentul

## Assistant

Da. Primul proof trebuie să fie **mic, falsificabil și fără H3 încă**: ținem sute de MB/GB de „weights” în RAM pinned, rezervăm doar două sloturi fixe în VRAM, rulăm GEMM real prin cuBLAS și măsurăm dacă `copy(N+1)` se ascunde sub `compute(N)`. Asta ne spune imediat dacă premisa fundamentală merită continuată pe H3.

Am pus proof-ul experimental complet în `dan3s2020/TensorWave`, pe branch-ul **`bench/h2d-overlap-proof-v1`**, și am deschis draft PR #1.

Experimentul face exact ce discutam: weights-urile stau în RAM pinned, VRAM-ul are **două sloturi fixe**, iar în timp ce GPU-ul face GEMM real pe tile-ul `N`, un stream separat face `cudaMemcpyAsync()` pentru tile-ul `N+1`. Slotul nu poate fi suprascris până când CUDA event-ul confirmă că GPU-ul a terminat cu el.

Se compară:

```text
BASELINE:
COPY 0 -> COMPUTE 0 -> COPY 1 -> COMPUTE 1 -> ...

TENSORWAVE:
GPU:   COMPUTE 0 | COMPUTE 1 | COMPUTE 2 | COMPUTE 3
PCIe:       COPY 1    | COPY 2    | COPY 3    | COPY 4
```

Metrica centrală este `steady_starvation_pct`, nu doar H2D GB/s.

---

## User

Bun ,apuca-te si continua

## Assistant

Am continuat în Faza 2: checkpoint real, fără să presupun arhitectura H3. Am introdus:

- parser `safetensors` header-only;
- Weight Atlas cu nume/shape/dtype/offseturi exacte;
- pack byte-for-byte din tensori reali;
- `execution-plan.json`;
- static runtime schedule pentru cele două sloturi;
- CUDA/cuBLAS proof cu weights F16/BF16 reale;
- sweep automat și sumarizare;
- CI pentru tooling și compile CUDA.

Faza 2 a trecut compile CUDA în CI. Am separat clar storage order de runtime schedule și am păstrat regula că niciun slot nu poate fi suprascris înainte de `compute-end` al proprietarului anterior.

Apoi am pornit Faza 3: **Q4 în RAM -> Q4 prin PCIe -> GPU dequant -> GEMM**.

Formatul de proof ales este:

```text
Q4_SYM_G32_F32S
32 weights/group
4-byte float32 scale
16 packed signed-int4 bytes
20 bytes/group
```

Față de 16-bit:

```text
64 bytes -> 20 bytes
3.2x compression
31.25% physical byte ratio
5 effective bits/weight including scale
```

Am adăugat două sloturi Q4 fixe în VRAM, un singur buffer FP16 reutilizabil pentru dequant, kernel CUDA de dequantizare, cuBLAS GEMM și măsurare separată pentru H2D/dequant/GEMM/starvation.

Am creat și issue-ul pentru Faza 4: decode Q4 direct în shared-memory/register fragments către Tensor Core MMA, fără materializarea completă a tile-ului FP16.

---

## User

Bun. Acum vreau sa vedem dacă ideea noastră este unică si utilă sau dacă cineva a făcut ceva exact așa . Si de ce merge/nu merge pentru un LLM mare?

## Assistant

Am făcut o căutare de prior art concentrată pe mecanism, nu doar pe termenul generic „CPU offload”.

Concluzia:

**ideea fundamentală nu este unică.** Există sisteme foarte apropiate. Dar combinația exactă TensorWave — sub-tensor tiles, static schedule, fixed compressed VRAM ring, compressed H2D, dequant cât mai aproape de MMA și `GPU starvation` ca obiectiv central — nu a fost identificată ca un singur sistem complet în căutarea inițială.

Cele mai apropiate sisteme discutate:

```text
AirLLM
ZeRO-Inference
FlexGen
ATSInfer
PowerInfer / PowerInfer-2
fast low-bit/NF4 GPU dequant kernels
```

AirLLM confirmă:

- modele mult mai mari decât VRAM pot fi executate prin progressive loading;
- prefetching poate suprapune loading și compute;
- low-bit packed weights pot reduce transferul;
- tiny VRAM poate fi suficient pentru a executa un model, dar viteza poate deveni inutilă dacă sistemul este I/O-bound.

ZeRO-Inference și FlexGen confirmă:

- CPU/NVMe/GPU ca memory hierarchy;
- quantized weight transfer;
- batching/throughput ca metodă de amortizare a transferului.

ATSInfer confirmă că whole-layer offload poate fi prea grosier și că tensor-granularity este o direcție reală.

PowerInfer-2 confirmă că scheduling-ul poate coborî sub layer/tensor, spre neuron/sub-matrix clusters.

Kernels moderne de low-bit dequant confirmă că decode-ul trebuie apropiat de shared memory/register-level compute.

Prin urmare TensorWave nu trebuie să pretindă ca noutate:

```text
model mare în RAM
4GB GPU poate executa model >4GB
prefetch weights
4-bit H2D
dequant pe GPU
tensor-level/sub-layer offload
hot-weight caching
```

Posibila contribuție distinctă este compoziția:

```text
large Q4 RAM weight store
       ↓
Weight Atlas byte-addressable
       ↓
offline/static execution schedule
       ↓
sub-tensor fixed tiles
       ↓
2–3 fixed compressed VRAM slots
       ↓
scheduled async H2D
       ↓
Q4 fragment decode
       ↓
shared-memory/register fragment
       ↓
Tensor Core MMA
       ↓
discard fragment
```

cu obiectiv explicit:

> **minimizăm unhidden transfer / GPU starvation pe un GPU consumer cu VRAM foarte mic.**

### De ce este dificil pentru un LLM dens mare

La decode batch=1, pentru fiecare token un dense LLM folosește aproape toate weights-urile din nou.

Aproximativ:

```text
13B Q4
33B Q4
70B Q4
```

treaga cantitate de active weights trebuie retraversată pentru fiecare token dacă nu încape în cache/VRAM.

Dacă H2D este ~15 GB/s, transferul singur produce un plafon sever de tokens/s.

Condiția TensorWave este:

```text
T_compute(tile) >= T_transfer(next tile)
```

La batch=1 decode, compute-ul pentru un tile poate fi prea scurt, astfel GPU-ul termină și așteaptă PCIe.

### De ce batching și prefill sunt mult mai bune

Dacă același weight tile este folosit pentru `M` activation rows:

```text
X[M,K] * W[K,N]
```

transferăm weight-ul o singură dată, dar compute-ul crește cu `M`.

Astfel transferul poate fi ascuns.

### De ce MoE este foarte interesant

Un MoE poate avea foarte mulți parametri total, dar fiecare token activează doar o fracțiune dintre experți.

Runtime-ul poate transfera numai experții activi și poate cache-ui experții „hot”.

Asta face MoE mult mai promițător decât un dense LLM de aceeași dimensiune totală.

Concluzia formulată:

> Dacă obiectivul era doar „rulăm 70B pe 4GB indiferent de viteză”, proiectul nu este nou și nu este suficient. Obiectivul util este să aflăm viteza maximă fizic posibilă după eliminarea aproape tuturor costurilor în afară de transferul inevitabil și compute.

Întrebarea critică pentru TensorWave devine:

```text
pentru ce (model architecture, batch, tile size, quantization, PCIe bandwidth)
obținem T_compute >= T_transfer?
```

Aceasta trebuie transformată într-o hartă de fezabilitate.

---

## User

Bun,continua sa documentezi tot ce ai zis în git,inclusiv conversația completă .apoi tot în git te apuci sa faci harta . Apoi după ce faci harta propui niște experimente ca sa dovedim ca merge și faci codul necesar ca să testam

## Assistant

Acest mesaj deschide Faza `research/feasibility-map-v1`.

Ordinea cerută este păstrată:

1. arhivarea conversației și a prior-art-ului;
2. construirea hărții matematice/experimentale;
3. definirea experimentelor falsificabile;
4. scrierea codului/runners necesari pentru testarea pe GPU real.

---

# Continuation boundary

Further conversation should be appended as new dated transcript segments rather than rewriting this historical segment.
