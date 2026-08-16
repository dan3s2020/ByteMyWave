# 10 — Implementation, Procurement and Benchmark Plan

## Goal

Turn the distributed-K3 idea into a sequence where **each purchase and each software step has a pass/fail criterion**.

No 10–20 node cluster should be purchased merely because aggregate RAM arithmetic works. We first validate one representative node, one real memory population, one NIC path and one real K3 kernel.

---

# 1. Current target configurations

## Route A — DDR4, preferred if Alibaba H12DGQ-NT6 lead is genuine

```text
10 × Supermicro H12DGQ-NT6
20 × inexpensive compatible AMD EPYC 7002/7003 CPUs
160 × 16 GB DDR4 ECC RDIMM
= 2.56 TB nominal aggregate RAM
```

Population concept:

```text
2 CPUs/board
8 DIMMs/CPU
1 DIMM/channel
16 DIMMs/board
```

Advantages:

- modern EPYC memory controllers;
- 16 memory channels per dual-socket board;
- DDR4-3200;
- PCIe 4.0 platform;
- far more CPU throughput and memory bandwidth than DDR2-era systems;
- only 160 DIMMs required at 16 GB each.

Risk:

- H12DGQ-NT6 is proprietary and normally belongs to AS-4124GQ-TNMI;
- apparent ~303 RON Alibaba board price may be placeholder, damaged stock, incomplete board-only stock or otherwise misleading;
- power distribution / harness / chassis could erase the cost advantage.

### Mandatory seller questions

```text
Is the displayed price the full price for one complete H12DGQ-NT6 motherboard
when buying 10 pieces, not a deposit or placeholder?

Are the boards tested working and able to POST?
Please send actual photos of all available boards including model/serial labels.

Are there repaired, damaged or missing components?
Are BIOS and BMC usable and unlocked?

Please quote all parts required to boot the motherboard:
power distribution board, PSU, power cables/harness, front-panel/power-on interface,
CPU heatsinks and any mandatory riser/control boards.

Do you have stripped AS-4124GQ-TNMI chassis/components from the same dismantled systems?

Quote EXW and DDP Romania for 1 validation unit and for 10 units.
```

---

## Route B — DDR3 Dell R920

Capacity choices:

```text
3 × R920 × 96 × 8 GB = 2.304 TB
4 × R920 -> 384 physical slots
300 × 8 GB = 2.40 TB if the existing plan uses all 300 modules
```

Required RAM:

```text
DDR3 ECC RDIMM or supported LRDIMM
not desktop ECC UDIMM
```

Advantages:

- 96 documented memory sockets per server;
- fewer physical servers than 16/32-slot alternatives;
- much newer CPU and PCIe generation than DDR2 platforms;
- conventional complete rack server, easier to power/manage than proprietary board-only stock.

Risks:

- 8 GB DIMMs at high DIMMs-per-channel can lower memory operating speed;
- complete four-CPU/eight-riser configuration must be present to expose all intended slots;
- used R920 price may be much higher than true DDR2 e-waste.

### R920 seller checklist

```text
[ ] all four CPUs installed
[ ] all eight memory risers installed
[ ] all 96 DIMM sockets physically present/usable
[ ] BIOS/iDRAC usable
[ ] POST test
[ ] PSU pair present
[ ] PCIe risers/slots present for NIC
[ ] exact CPU models
[ ] actual photos, not stock photos
```

---

## Route C — DDR2 64-slot e-waste cluster

Preferred targets:

```text
5 × HP DL785 G6      @ 64 × 8 GB = 2.56 TB
5 × Sun Fire X4640   @ 64 × 8 GB = 2.56 TB
5 × Sun Fire X4600 M2 with 501-7817 CPU modules = 2.56 TB
```

Only pursue if the complete machines are available near scrap value.

Why:

- the capacity is excellent;
- CPU efficiency, power draw and software ISA are poor;
- the system is only economically rational if acquisition cost is extremely low.

### Search terms / FRUs

HP DL785 family:

```text
AM437A
AM438A
AM439A
588797-001
AM422A
AM423A
AM424A
AM427A
AM428A
AM429A
AM430A
AM431A
491104-001
AH233-2109D
AH233-60005
```

Sun X4640:

```text
511-1387
511-1461
541-4146
541-4147
350-1476
599-3661
X8486A
X8487A
```

Sun X4600 M2:

```text
501-7817
```

DL585 G5 fallback leads:

```text
455349-B21
454592-001
448188-421
534498-001
534499-001
534500-001
```

These are search keys. Seller stock must be mapped back to exact vendor documentation before money is sent.

---

# 2. Acquisition rule: buy one representative node first

Before a cluster order:

```text
1. Buy/borrow ONE target node or board.
2. Populate enough memory to exercise every intended memory channel.
3. Install the intended NIC.
4. Boot the exact Linux environment.
5. Run the complete benchmark pack.
6. Run a real K3 tensor/kernel microbenchmark.
7. Only then negotiate the bulk lot.
```

An exception is acceptable only when a bulk e-waste lot is so cheap that the entire lot costs less than a normal single validation unit and the seller will not split it. Even then, the technical risk must be recorded.

---

# 3. Hardware validation pack

Every candidate node gets a machine-readable inventory file:

```text
artifacts/hardware/<node-id>/inventory.json
```

Capture:

```text
DMI board/server model and serial
BIOS version
BMC version
CPU model(s)
CPU flags
NUMA topology
DIMM count/capacity/rank/speed per socket
PCIe topology
NIC model/link rate
SSD model
kernel version
power configuration
```

Suggested commands/tools:

```text
lscpu
numactl --hardware
lsmem
dmidecode
lspci -vv
ethtool
ip -s link
smartctl / nvme-cli
```

---

# 4. Memory benchmark pack

Headline DDR speed is not enough. Measure the actual machine.

For each NUMA node:

1. single-thread sequential read;
2. all-core sequential read;
3. STREAM Copy/Scale/Add/Triad;
4. local NUMA allocation;
5. remote NUMA allocation;
6. all sockets simultaneously;
7. one-DIMM-per-channel vs dense population where available.

Record:

```text
GB/s p50
GB/s best stable
GB/s sustained 60s
NUMA remote penalty
CPU utilization
power draw if measurable
```

Procurement scoring should use **sustained NUMA-local bandwidth**, not DIMM sticker speed.

---

# 5. Network validation pack

Start with a dedicated inference fabric separate from management if possible.

Baseline concept:

```text
1 GbE management: SSH, logs, package management, BMC
10/25/40 GbE or InfiniBand inference fabric: activations/expert traffic
```

The exact fabric depends on what PCIe generations and cheap NICs the purchased nodes actually support.

Measure:

```text
iperf3 bulk throughput
one-way/RTT small-message latency
multiple simultaneous peers
CPU cost per GB/s
p99 latency under load
```

Then run a custom `tw-nettrace` benchmark that reproduces the real K3 MoE cadence rather than trusting `iperf3` alone.

---

# 6. CPU kernel development order

Do not begin by writing the entire 93-layer runtime.

## K0 — inspect exact K3 tensor formats

Use the pinned checkpoint and record:

```text
tensor name
shape
dtype
compressed format
group size
scale format
byte size
layer
expert ID if applicable
```

## K1 — MXFP4 decode correctness

Implement scalar reference unpack/dequant first.

Test against the official/reference implementation for exact known packed values.

## K2 — one expert linear path

Implement the real K3 expert tensor shapes.

Paths by CPU ISA may include:

```text
scalar reference
SSE2 baseline
SSE4.x path where available
AVX/AVX2 path for newer DDR3 CPUs
AVX2/AVX-512 path for compatible DDR4 CPUs
```

Do not assume the same binary is optimal across DL785-era Opteron and EPYC 7002.

## K3 — fused unpack + dot product

The preferred old-CPU hot path should avoid materializing an entire BF16/FP32 copy of a compressed expert weight matrix in RAM.

Conceptually:

```text
load packed MXFP4
-> unpack small block
-> apply scale
-> accumulate dot product immediately
-> discard expanded temporary
```

Measure **compressed bytes consumed per second**, not only FLOP labels.

## K4 — router + 16 selected experts

Build one real MoE layer path:

```text
router
-> top-16 expert IDs
-> local/remote ownership grouping
-> concurrent expert execution
-> combine outputs
```

## K5 — KDA/MLA layer path

Implement and validate the non-MoE path separately because its data movement and compute structure differ.

---

# 7. Distributed software milestones

## M0 — checkpoint manifest

Deliverable:

```text
K3 revision pinned
all 96 safetensor shard files accounted for
all tensor names/bytes indexed
```

## M1 — resharder

Deliverable:

```text
tw-k3-reshard
```

Acceptance:

```text
no missing tensors
no unexplained duplicate tensors
all output hashes stored
reassembling manifests reproduces original tensor set
```

## M2 — two workers on one machine

Use two processes/NUMA domains before using the network.

Acceptance:

```text
distributed layer output ~= reference
```

## M3 — two physical machines

Transport real activation/expert payloads between nodes.

Acceptance:

```text
same numerical result as M2
transport metrics recorded
```

## M4 — N-node synthetic trace

Scale only the execution skeleton to 5/10/20 workers and emulate the K3 layer/expert cadence.

Acceptance:

```text
predicted distributed trace T_token meets the target envelope
```

## M5 — complete full-RAM K3 load

Load all official tensors across workers.

Acceptance:

```text
aggregate worker manifests == checkpoint manifest
no SSD weight faults during steady-state decode target
```

## M6 — first complete token

One prompt produces one correct next-token logits vector through all 93 layers.

## M7 — full generation

Generate multiple tokens, compare against reference behavior/tolerance, then benchmark.

## M8 — API gateway

Expose the cluster as one OpenAI-compatible endpoint.

---

# 8. Agentic-framework integration

The user's PC should not need to understand the cluster topology.

External interface:

```text
http://<tensorwave-head>:8000/v1/chat/completions
```

Moonshot's current Kimi CLI / Kimi Code documentation supports custom OpenAI-compatible providers with an overridable `base_url`.

Official references:

- https://github.com/MoonshotAI/kimi-cli/blob/main/docs/en/configuration/providers.md
- https://github.com/MoonshotAI/kimi-code/blob/main/docs/en/configuration/providers.md

Conceptual client configuration:

```toml
[providers.tensorwave]
type = "openai"
base_url = "http://10.50.0.10:8000/v1"
api_key = "local-tensorwave"

[models."tensorwave/kimi-k3"]
provider = "tensorwave"
model = "kimi-k3"
max_context_size = 1048576
```

The exact provider type/name must follow the installed Kimi Code version. Current documentation calls the Chat Completions-compatible provider `openai` in the `kimi-code` docs and `openai_legacy` in the `kimi-cli` docs; do not blindly copy a config from a different version.

User flow:

```text
PC
 -> Kimi Code / another agent
 -> TensorWave OpenAI-compatible gateway
 -> controller/router
 -> distributed K3 workers
 -> response/tool calls back to agent
```

The agent sees one model alias. It does not know which experts live on which server.

---

# 9. BOM comparison template

Every serious quote should be normalized to this table before purchase:

| Item | DDR2 route | R920 DDR3 route | H12DGQ DDR4 route |
|---|---:|---:|---:|
| Nodes/boards | 5 | 3–4 | 10 |
| CPUs | exact model TBD | exact model TBD | 20 × EPYC candidate |
| DIMMs | 320 × 8 GB | 288–300 × 8 GB | 160 × 16 GB |
| Nominal RAM | 2.56 TB | 2.304–2.40 TB | 2.56 TB |
| NICs | TBD | TBD | TBD |
| Switch/fabric | TBD | TBD | TBD |
| SSDs | TBD | TBD | TBD |
| Chassis/power | complete server | complete server | **must price proprietary components** |
| Acquisition RON | quote | quote | quote |
| Measured GB/s | benchmark | benchmark | benchmark |
| RON / measured GB/s | calculate | calculate | calculate |
| Watts @ decode | benchmark | benchmark | benchmark |
| Real K3 tok/s | benchmark | benchmark | benchmark |

The winner is not necessarily the platform with the lowest purchase price. A 100 RON server that adds 800 W and 0.03 tok/s can be more expensive in use than a newer 400 RON node.

---

# 10. Immediate decision tree

```text
Is H12DGQ-NT6 really ~303 RON and bootable with affordable power/chassis parts?
  |
  +-- YES -> acquire one validation set -> benchmark -> likely primary route
  |
  +-- NO
       |
       v
Can R920 be sourced cheaply enough with all four CPUs/eight risers?
  |
  +-- YES -> benchmark one R920 with representative 8 GB RDIMM population
  |
  +-- NO
       |
       v
Can 64-slot DDR2 machines be bought near scrap value?
  |
  +-- YES -> benchmark one exact CPU/server before bulk order
  |
  +-- NO -> continue e-waste/industrial search; do not pay legacy-spare prices
```

---

# 11. Definition of a successful first prototype

The first TensorWave K3 prototype is successful when all are true:

```text
[ ] official K3 revision pinned
[ ] complete checkpoint accounted for
[ ] complete checkpoint resides in aggregate RAM
[ ] no expert pruning required
[ ] real K3 layer/expert kernels are numerically validated
[ ] at least two physical nodes participate in one token
[ ] end-to-end full-model generation works
[ ] OpenAI-compatible gateway works from a separate PC
[ ] Kimi Code or another agent can use the local endpoint
[ ] full benchmark log reports actual decoded tok/s
```

The initial performance acceptance target is **>= 1.0 decoded token/s for batch=1 after warm-up**, but it remains a target until the measured full-model benchmark passes it.
