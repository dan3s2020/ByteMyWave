# 17 — K3 Parallelism Benchmark Plan: 6×RTX3060, 2×R920, 120-GPU Extrapolation and Transit Parity

## Purpose

This document turns the 2026-08-17 GPU-vs-Transit discussion into a falsifiable experiment.

The immediate question is **not**:

```text
How many total GB/s are installed?
```

It is:

```text
For one batch=1 Kimi K3 decode token,
how much of the installed hardware can execute the current layer concurrently,
and what is the measured critical-path latency?
```

The experiment must answer that on both:

1. one representative **6×RTX3060 R920 GPU stage**;
2. one representative **Transit local-memory compute tile**.

No 20-node GPU cluster and no 38-tile Transit fabric should be purchased because of arithmetic alone.

---

# 1. Output required from every test

Every benchmark run must produce a machine-readable record containing at least:

```json
{
  "timestamp_utc": "",
  "git_commit": "",
  "model_repo": "moonshotai/Kimi-K3",
  "model_revision": "",
  "quantization": "",
  "runtime": "",
  "runtime_commit": "",
  "batch": 1,
  "context_tokens": 0,
  "prompt_tokens": 0,
  "generated_tokens": 0,
  "hardware": {},
  "topology": {},
  "split_mode": "",
  "tp": 1,
  "pp": 1,
  "ep": 1,
  "per_layer_ms": [],
  "collective_ms": [],
  "weight_bytes_read": 0,
  "pcie_bytes": 0,
  "network_bytes": 0,
  "power_w": 0,
  "decode_tok_s": 0.0,
  "notes": ""
}
```

Raw logs must be retained next to the summary. Never retain only the final token/s number.

---

# 2. Freeze the exact K3 representation before benchmarking

The 120×3060 fleet has `1.44 TB` raw VRAM, which is smaller than the project’s `~1.56 TB` released-checkpoint working value before runtime headroom.

Therefore the final candidate representation must be explicit.

Record:

```text
checkpoint revision
conversion/quantizer revision
quant type per tensor class
exact total bytes
routed-expert bytes
non-routed/shared bytes
scale/metadata bytes
KV/state requirement at benchmark context
quality/eval delta versus the source model
```

Do **not** benchmark a tiny synthetic Q4 GEMV and then assume the same behavior for the final K3 quant.

If an aggressive quant is required for capacity, model-quality validation becomes part of the hardware decision.

---

# 3. G0 — one R920 / six-GPU physical inventory

Before K3 kernels, capture the actual machine.

Required:

```text
R920 BIOS/iDRAC version
CPU models and socket count
NUMA topology
all PCIe slot negotiated link widths/speeds
all 6 GPU PCIe BDFs
GPU model/revision/VRAM
GPU clocks/power limits/temperature
NVIDIA driver
CUDA version
NCCL version
OS/kernel
NIC model/link speed
```

Commands may include:

```bash
lscpu
numactl --hardware
lspci -tv
lspci -vv
nvidia-smi -q
nvidia-smi topo -m
nvidia-smi topo -p2p r
ip -br link
ethtool <interface>
```

Acceptance:

> We know exactly which GPUs share root complexes/switches and which links are Gen3 x16/x8/etc. No topology is inferred from the mechanical slot shape.

---

# 4. G1 — GPU memory and P2P baseline

## 4.1 Local VRAM bandwidth

Measure each GPU independently with a real device-memory bandwidth test.

Record:

```text
GPU0..GPU5 read GB/s
GPU0..GPU5 copy GB/s
clock/temperature/power during run
```

The planning value `360 GB/s` is not the measured value.

## 4.2 Pairwise GPU transfer matrix

Measure every directed pair:

```text
GPU0 -> GPU1
GPU0 -> GPU2
...
GPU5 -> GPU4
```

Record:

```text
P2P supported? yes/no
unidirectional GB/s
bidirectional GB/s
small-message latency
host-staged fallback GB/s if P2P unavailable
```

This produces a 6×6 matrix rather than one vague “PCIe Gen3” number.

## 4.3 Collective baseline

With NCCL or the runtime’s real collective backend, measure the sizes closest to K3 tensor/expert synchronization rather than only 1 GiB bulk buffers.

At minimum:

```text
all-reduce latency
all-gather latency
reduce-scatter latency
all-to-all if expert parallel path uses it
```

for multiple payload sizes from KiB to MiB.

Acceptance:

> We know the communication cost that a tensor/expert split will actually pay inside one R920.

---

# 5. G2 — real K3 one-layer decode trace on one GPU

The first model benchmark should isolate one real K3 layer or routed-expert path from the exact target checkpoint/quantization.

Required work:

```text
load exact tensor bytes
use real K3 dimensions
use batch=1 decode-shaped activation
run the exact quantized unpack/dequant path
run expert MVM/GEMV
include scale handling
record bytes and kernel times
```

Measure separately:

```text
routed expert kernel
shared/non-routed expert path
attention/KDA/MLA representative work
router/top-k overhead if available
```

Output:

```text
kernel ms
compressed weight GB/s actually consumed
logical Gweights/s
GPU utilization
memory-controller utilization
power
```

Acceptance:

> We have a real K3-shaped Ampere result, not a theoretical RTX3060 spec-sheet quotient.

---

# 6. G3 — six-GPU split-mode comparison

The key experiment is to run the same layer/workload under progressively more parallel layouts.

Candidate matrix:

| Test | TP | EP | Layer/PP | Goal |
|---|---:|---:|---:|---|
| G3-A | 1 | 1 | single GPU | baseline |
| G3-B | 2 | 1 | same layer | check 2-way tensor scaling |
| G3-C | 3 | 1 | same layer | check 3-way tensor scaling |
| G3-D | 6 | 1 | same layer | max in-node tensor split |
| G3-E | 1/2/3 | multiple expert owners | same layer | check selected-expert concurrency |
| G3-F | mixed | mixed | local mini-pipeline | determine best measured topology |

For every test record:

```text
compute ms
collective ms
GPU idle ms
per-GPU weight bytes read
per-GPU memory BW
per-GPU SM utilization
end-to-end layer ms
```

### Current software gate

As of the 2026-07-31 llama.cpp K3 issue, the reported K3 path supported `split-mode layer` but the reporter requested `row/tensor` support.

Source:

- https://github.com/ggml-org/llama.cpp/issues/26365

Therefore do not assume stock llama.cpp can perform all rows in the table on K3 today.

If the production candidate runtime cannot perform same-layer TP/EP on Ampere, that is itself a procurement result:

> Hardware bandwidth that the available software cannot expose is not effective K3 bandwidth.

Possible paths can include a newer upstream implementation, a compatible distributed framework, or a custom K3 microbenchmark. The final benchmark must identify exactly which code path was used.

---

# 7. G4 — one complete 6-GPU R920 stage

Once the best six-GPU topology is found, create a stage containing the exact layer range that a 20-node deployment would assign to one R920.

Do not assume exactly `93/20` equal layers if K3 layer types and bytes differ.

Use the exact checkpoint atlas to partition by:

```text
bytes
layer type
routed expert ownership
non-routed/shared work
estimated compute
```

Measure:

```text
stage input bytes
stage output bytes
stage execution ms
sum kernel ms
sum collective ms
idle/bubble ms
peak VRAM
steady power
```

This yields the first credible building block for the 20-node model.

---

# 8. N0 — two-R920 network test

The inter-node test must reproduce inference traffic, not merely `iperf3` bulk bandwidth.

First record conventional link metrics:

```text
iperf3 unidirectional/bidirectional GB/s
small-message RTT
CPU utilization
p95/p99 latency
```

Then replay K3-shaped traffic:

```text
stage output -> next stage input
expert dispatch payloads if cross-node EP is tested
expert result payloads
barrier/synchronization cadence
```

Test at the actual candidate fabric speed available to the system:

```text
10 GbE
25 GbE
40 GbE
100 GbE / InfiniBand / RDMA
```

Do not extrapolate a 100 GbE result onto a 10 GbE procurement plan or vice versa.

---

# 9. N1 — 93-layer synthetic timing trace

Before owning 20 servers, use measured G4 + N0 timings to replay the structure of one full token.

The simulator must preserve:

```text
93 serial model layer boundaries
actual KDA vs MLA layer classification
routed-expert fanout cadence
selected expert count
stage boundaries
collective boundaries
network boundaries
non-expert serial work
```

For each modeled layer:

```text
T_layer = measured or conservatively mapped
          max(weight/compute critical path)
          + collective
          + network
          + scheduler/misc
```

Then:

```text
T_token = sum(T_layer) + head/state/sampling
TPS = 1 / T_token
```

The model must report where time is spent:

```text
GPU compute %
VRAM wait %
in-node collective %
inter-node network %
pipeline bubble %
other %
```

Acceptance:

> A full 20-node extrapolation exists only after its components are calibrated by physical 6-GPU and two-node measurements.

---

# 10. G5 — minimum real multi-token model test before bulk purchase

If a complete K3 quantized representation can be made to run on fewer available nodes, run the largest real partial cluster possible before bulk buying 120 GPUs.

Preferred ladder:

```text
1 GPU
2 GPUs
3 GPUs
6 GPUs
2 R920 / 12 GPUs
4 R920 / 24 GPUs if available
20 R920 / 120 GPUs only after trend is understood
```

For every step compare:

```text
expected scaling
measured scaling
collective overhead
network overhead
power
VRAM headroom
```

If scaling reverses or saturates early, stop and identify the bottleneck before adding hardware.

---

# 11. Transit parity benchmark

The Transit test must answer the same questions as the GPU test.

## T0 — one physical local-DDR tile

Measure:

```text
actual DDR channel count
actual DDR data rate
raw sustained DDR sequential read GB/s
model-shaped read GB/s
compute-active cycles
DDR-stall cycles
PCIe RX/TX bytes
result-stall cycles
weight elements processed
power
```

The existing `host/protocol.py` completion record already reserves key counters such as:

```text
cycles_total
cycles_compute_active
ddr_bytes_read
pcie_bytes_rx
pcie_bytes_tx
weight_elements
```

Use them rather than creating an unrelated benchmark vocabulary.

## T1 — exact proven INT4×INT8 resident-weight test

First reproduce the existing golden path:

```text
host/reference_bitplane.py
    ==
physical tile output
```

Acceptance for the proof format:

```text
max_abs_diff == 0
```

This validates the full physical path:

```text
resident local DDR weights
+ activation through PCIe
+ local compute
+ reduced result through PCIe
```

## T2 — real K3 numerical-format bridge

Then benchmark the actual intended K3 representation:

```text
native MXFP4/MXFP8 path
or
explicit Transit-native re-quantized path with measured quality
```

Record:

```text
compressed bytes consumed/s
expert outputs/s
scale/decode cycles
DDR utilization
numerical error versus reference
```

## T3 — one real K3 expert/shard

Use a real checkpoint tensor and real activation.

Measure the exact equivalent of GPU G2:

```text
weight bytes
compute ms
local memory GB/s
result bytes
PCIe ms
power
```

## T4 — 2/4/8 tile concurrency

Scale selected-expert work across multiple tiles.

Measure whether aggregate useful bandwidth grows:

```text
1 tile  -> X GB/s useful
2 tiles -> ?
4 tiles -> ?
8 tiles -> ?
```

and whether result/reduction/host scheduling becomes the bottleneck.

Only after this curve is known should `38 × 8 channels` be projected physically.

---

# 12. Exact Transit roofline checks to keep in the report

For 304 independent channels, retain both assumptions:

### Q4-equivalent lower bound — 52 GB/token

```text
DDR3-1600: ~74.8 weight-path tok/s
DDR3-1866: ~87.3 weight-path tok/s
DDR3-2133: ~99.8 weight-path tok/s
```

### checkpoint-ratio screening — 58 GB/token

```text
DDR3-1600: ~67.1 ideal screening tok/s
DDR3-1866: ~78.3 ideal screening tok/s
DDR3-2133: ~89.5 ideal screening tok/s
```

These remain roofline/screening values until T0–T4 and full-model work exist.

---

# 13. Exact six-RTX3060 roofline checks to keep in the report

Planning physical local bandwidth:

```text
6 × 360 GB/s = 2,160 GB/s
```

Pure memory quotients:

```text
2,160 / 52 ~= 41.5 weight-path tok/s equivalent
2,160 / 58 ~= 37.2 screening tok/s equivalent
```

Again, do not call these K3 decode rates.

The actual stage result is the measured G4 latency after:

```text
quant decode
GEMV/GEMM
expert routing
TP/EP collectives
PCIe topology effects
attention/shared work
runtime overhead
```

---

# 14. Procurement stop/go table

| Gate | GPU route | Transit route | Bulk purchase allowed? |
|---|---|---|---|
| Capacity | final K3 quant + KV fits planned VRAM | checkpoint/format fits tile DDR | no |
| Local memory | measured per-GPU bandwidth | measured DDR bandwidth | no |
| Real K3 kernel | G2 passes | T2/T3 passes | no |
| Parallel scaling | G3 1→6 scaling measured | T4 1→8 scaling measured | no |
| Fabric | N0 measured | PCIe multi-tile measured | no |
| Full-token model | N1 calibrated trace + partial cluster | multi-tile K3 layer/token path | maybe |
| Real end-to-end benchmark | full K3 decode measured | full K3 decode measured | yes, scale based on economics |

The threshold is intentionally strict because the cost of being wrong at 120 GPUs or 38 final tiles is far larger than the cost of one representative prototype.

---

# 15. Final comparison report format

When both routes have prototype data, publish one table with no mixed evidence classes:

| Metric | 6×3060 R920 | projected 20×R920 only from calibrated trace | Transit prototype | projected 38-tile only from calibrated trace |
|---|---:|---:|---:|---:|
| model/quant | | | | |
| usable capacity | | | | |
| measured local BW | | | | |
| measured useful weight BW | | | | |
| real layer ms | | | | |
| collective/reduction ms | | | | |
| network/PCIe ms | | | | |
| measured decode tok/s | | N/A until built | | N/A until built |
| modeled decode tok/s | N/A | | N/A | |
| power W | | | | |
| purchase cost | | | | |
| quality delta | | | | |

The key discipline is:

> **Measured values stay measured; calibrated projections stay projections.**

---

# 16. Immediate action order

### GPU route

```text
1. obtain/identify one R920 capable of the intended 6-GPU topology
2. install 1 GPU and run G0/G1/G2
3. scale 1 -> 2 -> 3 -> 6 GPUs
4. identify the K3 software path that can expose TP/EP on Ampere
5. measure the best six-GPU stage
6. connect a second R920 and run N0
7. build the 93-layer calibrated trace
8. decide whether 20-node acquisition is justified
```

### Transit route

```text
1. finish one physical local-DDR tile
2. reproduce exact INT4×INT8 proof
3. measure raw vs useful DDR bandwidth
4. implement/validate K3 numerical bridge
5. run one real K3 expert/shard
6. scale 1 -> 2 -> 4 -> 8 tile endpoints
7. calibrate the 38-tile trace
8. decide whether bulk tile acquisition is justified
```

Run both tracks with the same evidence vocabulary so the eventual decision is economic and empirical rather than rhetorical.
