# RAW Visible Conversation — After TensorWave Repository Creation

Captured: 2026-08-15

This file preserves the visible user/assistant text for the project-development segment after repository creation. Tool calls and hidden connector payloads are intentionally excluded.

---

## User

Ok ,pentru proof experimental avem nevoie de un cod care sa producă experimentul

## Assistant

Da. Primul proof trebuie să fie **mic, falsificabil și fără H3 încă**: ținem sute de MB/GB de „weights” în RAM pinned, rezervăm doar două sloturi fixe în VRAM, rulăm GEMM real prin cuBLAS și măsurăm dacă `copy(N+1)` se ascunde sub `compute(N)`. Asta ne spune imediat dacă premisa fundamentală merită continuată pe H3. Îl pun într-un branch separat, conform regulilor proiectului.

Am pus acum proof-ul propriu-zis: nu un „sleep benchmark”, ci **GEMM FP16 real prin cuBLAS**, cu model sintetic în RAM pinned, două adrese VRAM fixe, două CUDA streams și evenimente care împiedică overwrite-ul înainte de consum. Mai verific branch-ul ca structură înainte să deschid PR-ul.

Gata. Am pus proof-ul experimental complet în `dan3s2020/TensorWave`, pe branch-ul **`bench/h2d-overlap-proof-v1`**, și am deschis **draft PR #1**.

Experimentul face exact ce discutam: weights-urile stau în RAM pinned, VRAM-ul are **două sloturi fixe**, iar în timp ce GPU-ul face GEMM real pe tile-ul `N`, un stream separat face `cudaMemcpyAsync()` pentru tile-ul `N+1`. Slotul nu poate fi suprascris până când CUDA event-ul confirmă că GPU-ul a terminat cu el. Se măsoară separat copy, compute și mai ales **timpul în care GPU-ul stă efectiv flămând după următorul tile**.

Am pus și baseline secvențial identic matematic, deci comparația este:

```text
BASELINE:
COPY 0 → COMPUTE 0 → COPY 1 → COMPUTE 1 → ...

TENSORWAVE:
GPU:   COMPUTE 0 | COMPUTE 1 | COMPUTE 2 | COMPUTE 3
PCIe:       COPY 1    | COPY 2    | COPY 3    | COPY 4
```

Scriptul face automat sweep pe `M = 64,128,256,512,1024,2048`, cu `K=4096`, `N=2048`, 32 tile-uri și salvează JSON + log + `nvidia-smi`.

Pe PC-ul cu RTX 3050 Ti, din repo:

```powershell
git fetch origin
git checkout bench/h2d-overlap-proof-v1
git pull

.\scripts\run-proof.ps1
```

Outputul important va arăta aproximativ:

```text
Sequential wall:       ...
Sequential H2D:        ... GB/s

Overlapped wall:       ...
Overlapped H2D:        ... GB/s

Compute sum:           ...
GPU starvation sum:    ...
Steady starvation:     ... %
Transfer hidden est.:  ... %
Pipeline speedup:      ...x

Verdict:
HYPOTHESIS_SUPPORTED_FOR_THIS_SHAPE
```

**Ce vrem să vedem:** `steady_starvation_pct <= 10%` și `hidden transfer >= 80%`. Dacă apare asta, am demonstrat experimental că GPU-ul mic poate fi alimentat continuu din RAM fără să plătim integral transferul RAM→VRAM.

Important: **nu l-am rulat eu pe 3050 Ti-ul tău**, pentru că mediul meu de aici nu are CUDA GPU. Codul este pus pentru a produce măsurătoarea pe hardware-ul real, nu am inventat rezultate.

Dacă primul run compilează și îmi dai outputul, următorul pas nu mai este un benchmark sintetic: **luăm un tensor real MiniMax H3 și îl trecem prin exact același pipeline.**

---

## User

Bun ,apuca-te si continua

## Assistant

Continui direct în **Faza 2**: nu mai fac încă un benchmark sintetic, ci leg TensorWave de un checkpoint real MiniMax H3. Următorul milestone este: **inventariem tensorii reali fără să încărcăm modelul complet, generăm Weight Atlas-ul, alegem un tensor mare real, îl tăiem în tile-uri deterministe și îl trecem prin același ring RAM→VRAM**.

Am prins și o problemă importantă înainte să o transformăm în cod fals: **H3 există oficial, dar checkpoint-ul oficial nu este încă vizibil în organizația MiniMaxAI de pe Hugging Face pe care o pot verifica acum**. MiniMax a spus pe 31 iulie că intenționează să deschidă weights „în următoarele zile”; nu voi hardcoda în TensorWave dimensiuni/layere inventate până nu avem checkpoint-ul real.

De aceea Faza 2 pe care o scriu acum este **checkpoint-real, dar model-agnostică**: citește direct orice shard `safetensors`, fără să încarce tensorii în Python, produce atlasul și pachetul de tile-uri din bytes reali. Când H3 apare oficial, îi dăm pur și simplu folderul checkpoint-ului.

Am trecut un gate important: **Phase 2 compilează integral cu CUDA 12.8.1 în CI** și Python CI e verde; deci atlas + static schedule + real-weight CUDA target nu mai sunt doar cod necompilat.

În paralel cu gate-ul de compilare, pornesc acum **Faza 3 pe branch separat**, pentru partea care ne interesează cel mai mult economic: **nu trimitem BF16/F16 prin PCIe; RAM-ul ține Q4, PCIe transportă Q4, iar GPU-ul dequantizează abia în slotul de lucru înainte de GEMM**. Asta atacă direct cantitatea de bytes RAM→VRAM, nu doar suprapunerea transferului.

Q4 Python gate este verde inclusiv testul care corupe intenționat un byte și verifică faptul că pack-ul este respins. Compile-ul CUDA al Q4 încă trage containerul; Phase 2 știm deja că a compilat verde în exact același CI.

Am verificat și direcția pentru faza următoare: NVIDIA CUTLASS 4.6.1 suportă explicit **narrow integer 4-bit** și are exact mecanismele de tiled data movement / shared-memory / register fragments de care avem nevoie ca să ajungem ulterior la `Q4 → registers/shared memory → MMA`, fără să materializăm tot tile-ul FP16. Asta face Phase 4 o direcție tehnică reală, nu o improvizație.

Nu sar încă la ea în cod până nu trece Q4-ul separat prin compile/test; altfel nu mai știm dacă un bug vine din quantizare, scheduling sau kernel fusion.

Am continuat destul de mult proiectul, nu doar cu documentație.

**Phase 2 este acum real-checkpoint proof**, nu sintetic: avem parser `safetensors` header-only, `Weight Atlas`, pack byte-for-byte din tensorii reali, execution plan, **runtime schedule precompilat** și CUDA care rulează bytes reali F16/BF16 prin două sloturi VRAM fixe. Important: **Phase 2 compilează integral cu CUDA în CI**.

Apoi am trecut la **Phase 3**, care este exact direcția mai interesantă discutată:

```text
MODEL REAL
   ↓
RAM: Q4 comprimat
   ↓
PCIe: transferăm NUMAI Q4
   ↓
VRAM SLOT A / SLOT B
   ↓
GPU dequant
   ↓
un singur tile FP16 reutilizabil
   ↓
cuBLAS GEMM
```

Am implementat un format controlat `Q4_SYM_G32_F32S`: 32 weights folosesc **20 bytes în loc de 64 bytes**, deci prin PCIe trece doar **31,25% din cantitatea F16/BF16**, adică **3,2× mai puțini bytes**, inclusiv scale-urile.

Am adăugat și protecții serioase: fiecare tile Q4 are SHA-256; înainte de primul transfer către GPU se verifică geometry, offsets, sizes și fiecare hash. Testele includ inclusiv **coruperea intenționată a unui byte**, care trebuie detectată.

Mai important pentru ideea ta cu indexarea: acum există un **static runtime schedule compiler**. Înainte să înceapă GPU-ul știm deja:

```text
tile 0 → slot A
tile 1 → slot B
tile 2 → slot A
tile 3 → slot B
...

copy(2) nu poate suprascrie A
până când compute(0) = DONE

compute(2) nu poate începe
până când copy(2) = DONE
```

Deci în loop-ul critic nu mai căutăm „ce tensor urmează?”. **Planul este compilat anterior.**

Benchmarkul măsoară separat:

```text
physical compressed H2D
source-equivalent feed rate
GPU dequant time
GEMM time
GPU starvation
hidden transfer
wall time
fixed VRAM working set
```

Singurul lucru pe care nu îl declar încă terminat este compile-ul CUDA pentru noul kernel Q4 până când build-ul lui nu trece efectiv.

Am pregătit și **Phase 4**, dar intenționat n-am început să stric Phase 3 cu optimizări premature. Issue #5 definește următorul salt:

```text
RAM Q4
   ↓
PCIe Q4
   ↓
VRAM compressed tile
   ↓
decode doar micro-fragmentul necesar
   ↓
shared memory / registers
   ↓
Tensor Core MMA
```

Asta ar elimina complet **tile-ul FP16 decomprimat** din VRAM.

---

## User

Bun. Acum vreau sa vedem dacă ideea noastră este unică si utilă sau dacă cineva a făcut ceva exact așa . Si de ce merge/nu merge pentru un LLM mare?

## Assistant

Da. Am căutat exact după mecanismul nostru, nu doar după „CPU offload”. Concluzia e destul de clară:

**ideea fundamentală nu este unică.** Există deja sisteme foarte apropiate. Dar **combinația exactă pe care o construim în TensorWave nu am găsit-o ca un singur sistem**, iar diferența care ar putea conta este tocmai coborârea de la layer → tensor → tile + Q4 transferat comprimat + scheduling static + dequantizare cât mai aproape de MMA.

Cel mai apropiat de noi este surprinzător de apropiat.

### Cine a făcut deja părți din TensorWave

| Sistem | Ce face similar cu noi | Diferența principală |
|---|---|---|
| **AirLLM** | rulează 70B pe 4 GB, layer streaming, prefetch compute/load, 4-bit compression | predominant layer/expert streaming, nu runtime-ul nostru tile-level |
| **ZeRO-Inference** | CPU/NVMe → GPU, INT4, reduce PCIe ~4×, quant/dequant kernels | proiectat mai ales pentru throughput/batching, nu tiny-GPU interactive |
| **FlexGen** | GPU+CPU+disk, 4-bit, scheduler pentru placement | optimizează batch mare/high throughput |
| **ATSInfer (2026)** | **tensor-granularity offload**, static placement + transfer dinamic + async CPU/GPU | foarte aproape de schedulerul nostru, dar nu aceeași arhitectură Q4 tile→GPU-decode |
| **PowerInfer-2** | sparge operațiile în **neuron clusters**, pipeline I/O + compute | rulează compute și pe CPU/NPU; altă arhitectură |
| **Fast NF4 kernels (2026)** | optimizează dequantizarea 4-bit folosind shared memory | nu rezolvă host→GPU streaming-ul |

AirLLM este cel mai direct precedent pentru afirmația „model enorm, GPU de 4 GB”. Din 2023 încarcă modelul layer-by-layer, iar în decembrie 2023 a introdus prefetching pentru a suprapune încărcarea cu compute-ul. Are și weight compression 4/8-bit.

Mai mult, versiunea AirLLM 3.1 din 2026 merge chiar mai aproape de ce discutam noi: pentru Kimi K3, **weights MXFP4 traversează PCIe în forma packed și sunt expandate pe GPU**. Pentru MoE încarcă doar experții activați.

Asta înseamnă că:

> **„Q4 în RAM → Q4 prin PCIe → expandare pe GPU” nu este nou.**

ZeRO-Inference este și mai important ca prior art academic/industrial: folosește GPU+CPU+NVMe pentru modele mai mari decât GPU-ul și are **4-bit weight quantization care reduce atât memoria cât și volumul PCIe aproximativ 4×**, cu kernels dedicate de quantizare/dequantizare.

FlexGen a demonstrat încă din 2023 că poți agrega GPU, CPU și disk, comprima weights și attention cache la 4 biți și rula chiar OPT-175B pe un singur GPU de 16 GB.

ATSInfer intră exact pe problema pe care am identificat-o noi: whole-layer offloading poate fi prea grosier și trece la **tensor-granularity offloading**, combinând static tensor placement cu transfer dinamic și coordonare CPU-GPU asincronă.

PowerInfer-2 merge și mai jos: descompune matrix operations în **neuron clusters**, apoi face scheduling și pipeline I/O-compute la granularitatea clusterelor.

Deci și ideea „nu transfera tot layerul, coboară la tensori/sub-matrix” are prior art.

### Unde TensorWave poate fi totuși diferit

Nu putem revendica drept noutate:

```text
model mare în RAM
prefetch next layer
4-bit H2D
GPU dequant
tensor-level offload
hot-weight caching
```

Ce rămâne interesant ca **combinație inginerească** este:

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

cu obiectivul explicit:

> **să nu existe niciodată tensorul complet decomprimat în VRAM și să optimizăm direct GPU-starvation pe un GPU consumer de ~4GB.**

### Dar există problema mare pentru LLM-uri

Pentru un **LLM dens autoregresiv**, la generarea unui singur token:

```text
token N
 ↓
layer 0
 ↓
layer 1
 ↓
...
 ↓
ultimul layer
 ↓
token N+1
```

Practic **toate weights-urile modelului sunt folosite din nou pentru fiecare token**.

Asta este groaznic pentru streaming PCIe.

Cu o reprezentare Q4 idealizată de ~0.5 B/param, ordinul de mărime este:

```text
70B params ≈ 35 GB weights
33B params ≈ 16.5 GB
13B params ≈ 6.5 GB
```

La ~15 GB/s H2D, fără cache/rezidență și fără overlap suficient, bandwidth-ul singur produce un plafon sever de tokens/s.

Dar noi încercăm să ascundem transferul în compute:

```text
dacă:
Tcompute(tile) >= Ttransfer(next_tile)

atunci transferul poate fi aproape complet ascuns
```

Problema este că la **decode batch=1** operația este apropiată de GEMV: foarte puțină reutilizare a fiecărui weight transferat. GPU-ul poate consuma tile-ul mai repede decât PCIe poate aduce următorul.

### La batch mare se schimbă complet

Dacă aducem weight-ul o dată și îl folosim pentru multe activation rows:

```text
matrix M×K × W
```

același transfer produce mult mai mult compute. La un M suficient de mare:

```text
compute W17 ███████████████
load W18       █████
```

transferul poate dispărea sub compute.

### Prefill este favorabil

Prompturile lungi oferă exact acest tip de reutilizare: același weight tile este aplicat multor token rows.

### MoE este și mai interesant

Un MoE foarte mare poate avea sute de miliarde de parametri total, dar un token folosește doar o fracțiune dintre experți. Dacă routerul spune suficient de devreme ce experți urmează, transferăm doar subsetul activ și cache-uim experții „hot”.

De aceea ordinea de șanse este aproximativ:

```text
MoE LLM                      foarte promițător
Diffusion/video Transformer  foarte promițător
LLM prefill                  promițător
batched LLM serving          foarte promițător
dense LLM batch=1 decode     cazul cel mai greu
```

Concluzia critică:

> **pentru ce `(model architecture, batch, tile size, quantization, PCIe bandwidth)` trecem pragul `Tcompute ≥ Ttransfer`?**

Dacă facem harta aceea, putem spune matematic ce modele merită rulate pe 4GB și care nu, înainte să facem integrarea completă.

---

## User

Bun,continua sa documentezi tot ce ai zis în git,inclusiv conversația completă .apoi tot în git te apuci sa faci harta . Apoi după ce faci harta propui niște experimente ca sa dovedim ca merge și faci codul necesar ca să testam

---

# End of raw segment at capture time
