# Phase 6 validation runbook

The Phase-6 claims are designed to turn into measurements. This runbook defines the first physical R920 tests.

## 1. Analytical simulator

K2.5 / 1 TiB:

```bash
python experiments/phase6-heterogeneous-kimi-runtime/simulate_heterogeneous_kimi.py \
  --model k2.5 \
  --host-ram-gib 1024 \
  --cpu-sockets 4 \
  --h2d-gbps 12 \
  --gpu-vram-gib 12 \
  --reserve-vram-gib 4 \
  --output-dir out/phase6-k25
```

K3 / 2 TiB:

```bash
python experiments/phase6-heterogeneous-kimi-runtime/simulate_heterogeneous_kimi.py \
  --model k3 \
  --host-ram-gib 2048 \
  --cpu-sockets 4 \
  --h2d-gbps 12 \
  --gpu-vram-gib 12 \
  --reserve-vram-gib 4 \
  --output-dir out/phase6-k3
```

Unit tests:

```bash
python -m unittest -v experiments/phase6-heterogeneous-kimi-runtime/test_simulator.py
```

## 2. Build the AVX-only CPU expert benchmark

Linux/GCC:

```bash
cd experiments/phase6-heterogeneous-kimi-runtime
g++ -O3 -std=c++17 -mavx -fopenmp bench_cpu_expert_q4.cpp -o bench_cpu_expert_q4
```

Important: use `-mavx`, not `-mavx2`, because E7-4890 v2 is an AVX-era CPU.

Windows/MSVC:

```powershell
cl /O2 /std:c++17 /arch:AVX /openmp bench_cpu_expert_q4.cpp
```

## 3. Run one socket at a time

First inspect NUMA topology:

```bash
numactl --hardware
lscpu -e=CPU,NODE,SOCKET,CORE
```

Socket/node 0 example:

```bash
OMP_NUM_THREADS=15 \
numactl --cpunodebind=0 --membind=0 \
./bench_cpu_expert_q4 --threads 15 --sockets 4 --warmup 3 --iters 20
```

Repeat for every NUMA node:

```bash
for n in 0 1 2 3; do
  echo "=== NUMA $n ==="
  OMP_NUM_THREADS=15 numactl --cpunodebind=$n --membind=$n \
    ./bench_cpu_expert_q4 --threads 15 --sockets 4 --warmup 3 --iters 20
done
```

Record:

```text
Gweights/s
encoded Q4 GB/s
CPU-expert-only K2.5 tok/s ceiling
PASS/FAIL 5 tok/s gate
PASS/FAIL 10 tok/s gate
```

Exact four-socket gates:

```text
5 tok/s  >= 26.4241152 Gweights/s/socket
10 tok/s >= 52.8482304 Gweights/s/socket
```

The benchmark PASS is necessary, not sufficient.

## 4. Measure raw local/remote memory bandwidth

Use a trusted memory bandwidth tool (for example STREAM) with affinity.

The important matrix is:

```text
CPU execution node x memory allocation node
```

Measure all 16 combinations:

```text
CPU0->RAM0 local
CPU0->RAM1 remote
...
CPU3->RAM3 local
```

Phase-6 needs the selected-expert kernel's actual bandwidth, not only a synthetic memcpy number, but STREAM gives the platform floor/context.

## 5. GPU topology

On the real machine:

```bash
nvidia-smi -L
nvidia-smi topo -m
nvidia-smi topo -p2p p
```

For each GPU capture:

```text
PCI bus id
negotiated generation
negotiated width
closest NUMA node/root
P2P capability to every other GPU
```

Do not infer local NUMA from slot numbering in software; discover and record it.

## 6. H2D calibration

Reuse the Phase-1/Phase-4 pinned-memory H2D benchmark on each local NUMA/GPU pair.

Measure:

```text
local pinned H2D GB/s
remote pinned H2D GB/s
1 simultaneous GPU
2 simultaneous GPUs on same socket
GPUs on different sockets
```

Replace the analytical `12 GB/s` value in experiment inputs with measured values; do not rewrite the historical reference result.

## 7. CPU<->GPU handoff microbenchmark

K2.5 has 60 MoE layers, so per-layer latency matters.

Benchmark a loop of:

```text
GPU produces 7168 BF16/FP16 activation
-> host/NUMA expert dispatch
-> CPU kernel
-> partial reduction
-> GPU receives output
-> next dependent step
```

Record:

```text
p50/p95/p99 roundtrip microseconds/layer
```

Handoff-only sensitivity from the simulator:

```text
10 us/layer   -> 0.6 ms/token
50 us/layer   -> 3.0 ms/token
100 us/layer  -> 6.0 ms/token
250 us/layer  -> 15 ms/token
500 us/layer  -> 30 ms/token
1000 us/layer -> 60 ms/token
```

At 5 tok/s the total token budget is 200 ms. At 10 tok/s it is 100 ms.

## 8. Exact checkpoint census before full integration

Do not permanently use the rounded 32B-active remainder.

Extend Weight Atlas to classify actual K2.5 checkpoint tensors and sum exact bytes for:

```text
routed experts
shared expert
attention/MLA
router
dense MLP
embedding
LM head
vision modules
other
```

Only after this census should the resident-set planner decide exact VRAM placement.

## 9. Compression benchmark matrix

For each candidate format:

```text
TensorWave Q4
Q3 candidate
Q2 candidate
mixed adaptive candidate
Q2 + residual/outliers candidate
```

measure simultaneously:

```text
model/task quality
encoded B/weight
CPU selected Gweights/s
GPU decode speed
host read GB/s
H2D GB/token where applicable
```

A smaller file that makes the AVX CPU kernel much slower may lose overall.

## 10. Full K2.5 heterogeneous acceptance

Do not claim 5 tok/s until all of the following are measured together:

```text
CPU routed-expert work
GPU resident/non-routed work
attention/KV/state
router
60 sequential handoffs
NUMA placement
reductions
real output decoding
```

Acceptance tests:

```text
single request
long enough generation to leave warmup
report prefill separately from decode
p50/p95 tok/s
first-token latency
total energy/power if available
quality/correctness against reference implementation
```

## 11. K3 transition gate

Do not move the complete design to K3 merely because K2.5 works.

K3 has a much larger routed active set. Use measured K2.5 kernel/H2D/handoff data as simulator inputs, then determine how many experts must become GPU-owned and how many independent GPU feeds are required.

K3 5–10 tok/s is a separate topology problem, not a free consequence of K2.5 success.
