# 14 — Surplus DDR2 Server Fabric

This document captures the second low-cost Transit architecture that emerged after the DDR3 tile work: use complete obsolete multi-socket servers as **prebuilt memory-controller + CPU + chassis + power + network tiles** instead of designing every DDR PHY ourselves.

It does **not** replace the DDR3 Transit-tile architecture in documents 07–13. It is an alternative path optimized for acquisition cost and capacity first, with a higher risk of poor compute-per-watt and lower final throughput.

## 1. Why this branch exists

The DDR3 tile architecture solves the right data-movement problem, but the missing hardware object remains difficult:

```text
cheap endpoint
+ 4–8 independent DDR3 channels
+ large DIMM capacity
+ local programmable compute
+ PCIe/network control path
```

Old enterprise servers already contain most of this difficult physical engineering:

- working DDR2 PHYs;
- CPU-integrated memory controllers;
- dozens of validated DIMM sockets;
- BIOS training;
- ECC/RAS;
- power delivery;
- cooling;
- PCIe slots;
- Ethernet;
- Linux support.

The question therefore changed from:

> Can we build hundreds of memory channels cheaply?

into:

> Can we buy obsolete servers so cheaply that their already-built memory channels, CPUs and DIMM infrastructure are cheaper than building custom Transit tiles?

The economic target discussed for this experimental path is extremely aggressive: roughly **1000–3000 RON total for scavenged memory-side hardware where possible**, not a conventional rack-scale procurement plan. That target is a search constraint, not a claim that every candidate can be obtained at that price.

## 2. Candidate servers identified

### 2.1 HP ProLiant DL785 G5

Project role: **64-slot-class DDR2 server tile candidate**.

Verified from HPE documentation:

- 8-socket x86 platform;
- up to 8 AMD Opteron processors;
- up to 512 GB memory in documented configurations;
- processor/memory cells are removable modules;
- DDR2-generation platform.

The project has treated the fully populated machine as a 64-DIMM-class candidate. Procurement must still verify the exact chassis, CPU-cell population, DIMM type and installed CPUs for every listing; a bare chassis or partially populated CPU-cell machine is not equivalent to the desired target.

Why interesting:

- each processor owns local memory, giving natural NUMA memory islands;
- all memory-controller/PHY work is already solved by the OEM;
- many PCIe slots and standard Linux make it easier to add fast networking;
- 8 GB DDR2 RDIMMs can make a fully populated machine a hundreds-of-GB memory node.

Main concern:

- full DIMM population reduces memory clock on this generation;
- old Opteron cores may not execute the Transit bitplane/MXFP kernel fast enough to consume all available memory bandwidth;
- idle and loaded power can be very high compared with modern hardware.

### 2.2 HP ProLiant DL785 G6

Project role: **preferred HP DDR2-generation variant if pricing is equally absurdly low**.

HPE option documentation confirms DDR2 RDIMMs and documents the memory-speed penalty as slots are filled. For PC2-6400 DIMMs, HPE states the memory bus can drop from PC2-6400 with light population to PC2-5300 with six DIMMs/processor and PC2-4200 with eight DIMMs/processor. Other combinations can also fall to PC2-4200 when heavily populated.

This matters because Transit wants both capacity and bandwidth. A 64-DIMM machine should therefore be modeled conservatively as a **DDR2-533-class full-population memory system until measured otherwise**.

### 2.3 Sun Fire X4640

Project role: **especially clean 64-DIMM DDR2 candidate**.

Oracle service documentation verifies:

- up to 8 CPU modules;
- each CPU module contains one processor and 8 DIMM slots;
- 64 DIMMs total;
- up to 512 GB total memory;
- DDR2 memory;
- 8 PCI expansion slots in total: 6 PCIe + 2 PCI-X;
- four onboard Gigabit Ethernet ports.

This is unusually aligned with the Transit concept because the OEM already exposes the machine as eight replaceable CPU+memory islands inside one 4U chassis.

Why interesting:

```text
8 CPU modules
× 8 DIMM sockets/module
= 64 DIMM sockets/server
```

Five such servers would expose 320 DIMM sockets of capacity infrastructure. Ten mixed 64-slot-class servers would expose roughly 640 physical DIMM sockets, although **socket count is capacity, not independent memory-channel count**.

### 2.4 Mixed fleet

A homogeneous fleet is easier, but the architecture should not require it.

The project explicitly considered combinations such as:

```text
5 × HP DL785 G5/G6
5 × Sun Fire X4640
```

The runtime must therefore discover actual per-node properties rather than assume one fixed server type:

```text
node_id
CPU model/count
NUMA nodes
DIMM capacity per NUMA node
measured memory bandwidth per NUMA node
supported ISA
network interfaces
kernel throughput
power draw
health
```

## 3. Important correction: 64 DIMM slots are not 64 independent channels

This is the same lesson learned during the R920/riser work.

A DIMM socket adds capacity behind a memory controller; it does not automatically add one independent x64 memory channel.

For the old 8-socket Opteron systems, the useful abstraction is approximately:

```text
server
  -> 8 NUMA sockets
     -> memory controller(s) per socket
        -> several DIMM sockets sharing those channels
```

Therefore we must measure and schedule at **NUMA-memory-controller granularity**, not at DIMM-socket granularity.

The working conservative roofline below uses an illustrative model of two x64 DDR2 channels per socket across eight sockets. The exact topology of the purchased machine must be verified from firmware/Linux topology and measurement before it is used for performance promises.

## 4. Distributed architecture

The whole-server version of Transit preserves the same central rule as the FPGA-tile version:

> Weights stay where the memory bandwidth is. Dynamic activations move to the node that owns those weights. Only reduced results come back.

Topology:

```text
                 Transit head / API node
          tokenizer / router / scheduler / reducer
                         |
                 fast Ethernet / IB
                         |
       +-----------------+------------------+
       |                 |                  |
 DDR2 server A      DDR2 server B      DDR2 server C
 local experts      local experts      local experts
 local CPU compute  local CPU compute  local CPU compute
       |                 |                  |
       +--------- reduced results ----------+
```

The head can initially be:

- the existing R920;
- a modern PC;
- another Linux server with a sufficiently fast NIC;
- later, a GPU-equipped node for attention/shared work.

The DDR2 nodes are not remote RAM servers in the naive sense. They are **memory-compute nodes**. If a node merely sends 52 GB/token of weights over Ethernet to a central GPU, the architecture fails immediately.

## 5. Model placement

The Weight Atlas must be extended from `tile_id/channel_id/address` to support whole-server NUMA placement:

```text
node_id
numa_id
local_file_or_memory_region
local_address
length
layer
expert
format
scale_metadata
checksum
replica_group
```

Preferred placement order:

1. keep a complete routed expert on one NUMA node if capacity permits;
2. otherwise keep it within one server;
3. only then shard an expert across servers;
4. replicate hot experts when spare capacity exists;
5. place shared/non-routed weights separately based on profiling.

K3's MoE structure is what makes this plausible. The router should wake only nodes containing selected experts instead of causing every server to scan weights for every token.

## 6. Local execution model

Each physical server runs one `transit-node` daemon and several NUMA-bound workers.

Conceptually:

```text
transit-node
  control thread
  network RX/TX
  health/telemetry
  |
  +-- worker numa0 -> local memory only
  +-- worker numa1 -> local memory only
  +-- ...
  +-- worker numa7 -> local memory only
```

Rules:

- `mbind`/NUMA affinity is mandatory;
- first-touch initialization must occur on the owning socket;
- a worker must not casually read another socket's resident weight pages;
- queue descriptors and activation buffers should be duplicated/localized if remote NUMA traffic becomes visible;
- weights are loaded once and reused;
- huge pages should be tested, not assumed to help;
- memory mapping should be deterministic enough that the Weight Atlas can validate residency.

## 7. Compute kernel path

The existing Transit host proof implemented exact signed INT4 × INT8 dot products using bitplanes, AND, POPCNT, shifts and add/sub and reached about 53.7 Gweights/s on the modern test laptop.

That result is **not transferable by clock/core count** to old Opterons.

The DDR2 server path needs a dedicated kernel ladder:

1. detect ISA per CPU (`SSE2`, `SSE4a`, `POPCNT/ABM` where present);
2. compile multiple kernel variants;
3. benchmark one NUMA socket with local memory;
4. scale to all sockets without remote-memory contamination;
5. measure `Gweights/s` and `GB/s` simultaneously;
6. only then multiply by server count.

If the CPU cannot process weights as quickly as local DDR2 can deliver them, the cluster is compute-limited and adding DIMMs will not improve tokens/s.

K3 adds a second numerical problem: its working format is MXFP4/MXFP8, while the proven kernel is signed INT4/INT8. The same numerical bridge described in document 13 is required here.

## 8. Bandwidth roofline for a fully populated DDR2 server

K3 working numbers in document 10 use:

```text
~104 billion active weights/token
~4 bits/active weight
=> ~52 GB active weight bytes/token
```

For a conservative full-population illustration:

```text
16 x64 memory channels/server
DDR2-533 payload/channel ~4.27 GB/s
---------------------------------
ideal aggregate ~68.3 GB/s/server
```

This is a **topology assumption + bus roofline**, not measured server bandwidth.

Corresponding ideal weight-path rate:

```text
68.3 / 52 ~= 1.31 weight-path tok/s/server
```

Cluster roofline:

| Servers | Ideal DDR2-533 payload | Ideal weight-path tok/s | At 60% memory efficiency | At 75% memory efficiency |
|---:|---:|---:|---:|---:|
| 5 | ~341 GB/s | ~6.6 | ~3.9 | ~4.9 |
| 10 | ~683 GB/s | ~13.1 | ~7.9 | ~9.8 |
| 20 | ~1.37 TB/s | ~26.3 | ~15.8 | ~19.7 |

These numbers still assume compute keeps up. They are **not end-to-end K3 generation rates**.

The final model is:

```text
server_effective_Gweights_s = min(memory_weight_rate, CPU_kernel_rate)
cluster_weight_path_tok_s = sum(server_effective_Gweights_s) / 104e9
```

End-to-end tok/s can only be reported after attention, router, network, synchronization, reduction, KV and numerical-format costs are included.

## 9. Capacity roofline

Capacity is where these systems are immediately attractive.

Using the documented maximum 512 GB class for the named candidates:

```text
5 servers  -> up to ~2.56 TB
10 servers -> up to ~5.12 TB
20 servers -> up to ~10.24 TB
```

The current K3 working checkpoint size in document 10 is ~1.56 TB.

Therefore even a partially populated cluster can satisfy checkpoint capacity long before it satisfies desired compute/bandwidth.

This distinction must stay explicit:

> Fitting K3 in aggregate RAM is easy. Running it fast is the hard part.

## 10. Network budget

The architecture is viable only if the network carries activations/results rather than weights.

Using the existing rough hidden-vector example:

```text
activation ~7 KiB at 8-bit
result     ~14 KiB at 16-bit
```

A deliberately conservative routing sketch with 16 experts across 93 routed layers gives an order-of-magnitude dynamic payload of:

```text
(7 KiB + 14 KiB)
× 16 experts
× 93 layers
~= 30.5 MiB/token
```

That is not a precise K3 trace; it is a planning bound using the project's working dimensions.

Approximate central-fabric payload from that sketch:

```text
10 tok/s  -> ~305 MiB/s -> ~2.4 Gbit/s payload
20 tok/s  -> ~610 MiB/s -> ~4.8 Gbit/s payload
100 tok/s -> ~3.05 GiB/s -> ~23.8 Gbit/s payload
```

Consequences:

- 1 GbE can prove the software architecture but is not a serious high-rate target;
- 10 GbE is a sensible minimum for a 10–20 tok/s experiment;
- cheap used 25/40 GbE or InfiniBand becomes attractive if the cluster scales;
- multicast/activation replication and local reductions can lower head-node pressure;
- the exact answer must come from an execution trace, not the rough vector math above.

## 11. Runtime protocol

Start boring.

Head -> node command:

```text
sequence_id
model_version
layer_id
expert_id
shard_id
activation_bytes
activation_format
routing_weight
flags
```

Node -> head completion:

```text
sequence_id
status
result_bytes
cycles
numa_id
ddr_bytes_read
weights_processed
network_rx_bytes
network_tx_bytes
checksum/error flags
```

Phase 1 transport can be TCP with persistent connections and fixed buffers.

Only introduce RDMA/InfiniBand verbs, UCX or a custom transport after counters show TCP/kernel networking is a real bottleneck. The software interface should remain transport-independent.

## 12. How the user actually uses the distributed K3 system

The distributed machine should appear as **one model endpoint**, not 10–20 manually managed servers.

```text
Laptop / desktop / agent framework
              |
      OpenAI-compatible API
              |
        Transit gateway
              |
    tokenizer + generation loop
              |
         Transit scheduler
              |
     distributed K3 runtime
```

The head node should eventually expose endpoints such as:

```text
/v1/models
/v1/chat/completions
/v1/completions
```

An agentic framework then points its `base_url` at the Transit gateway exactly as it would point at another local/self-hosted OpenAI-compatible inference server.

The agent framework does not need to understand DDR2, NUMA, experts or cluster topology.

## 13. Hybrid architecture

The DDR2 fleet and the DDR3/FPGA tiles do not have to compete permanently.

A useful hybrid may be:

```text
fast DDR3/FPGA Transit tiles
  -> hot experts / bandwidth-critical routed path

DDR2 whole servers
  -> cold experts
  -> overflow capacity
  -> replicas / failover
  -> checkpoint staging
  -> development path while custom tiles are incomplete
```

The Weight Atlas can place tensors according to measured cost:

```text
placement_score =
  latency
+ bandwidth pressure
+ network cost
+ replica availability
+ power policy
```

This lets cheap server capacity remain useful even if it fails the final performance target.

## 14. What this architecture solves immediately

It eliminates or postpones several difficult hardware problems:

- no custom DDR2/DDR3 PCB;
- no FPGA DDR PHY bring-up just to access loose DIMMs;
- no reverse engineering of proprietary memory risers before first software proof;
- no need for hundreds of PCIe endpoints;
- Linux already sees the RAM;
- CPU memory controllers already train the DIMMs;
- each server can be validated independently.

## 15. What it does not solve

It does not magically make old RAM fast.

Unresolved limits include:

- old CPU instruction throughput;
- memory speed collapse at full DIMM population;
- NUMA cross-socket penalties;
- high server power consumption;
- network synchronization across selected experts;
- K3 MXFP4/MXFP8 semantics;
- attention/shared/KV placement;
- end-to-end latency;
- reliability of very old DIMMs/PSUs/fans;
- acquisition economics once shipping and power are included.

## 16. First proof sequence

Do not buy 10–20 machines first.

For one candidate server:

```text
1. boot Linux
2. inventory CPU/NUMA/DIMMs
3. measure wall power idle/load
4. run STREAM/NUMA-local bandwidth per socket
5. run aggregate local-memory bandwidth
6. port Transit bitplane kernel
7. measure Gweights/s per socket
8. measure whole-server Gweights/s
9. emulate one K3 expert/shard
10. attach a 10 GbE-class NIC
11. run activation -> local compute -> result over network
12. record latency/bandwidth/correctness
```

Only after this measurement can the project say whether the old-server route is:

- final compute architecture;
- useful low-cost fallback;
- capacity-only support layer;
- or economically dead despite cheap purchase price.

## 17. Source verification status

Primary vendor documentation checked on 2026-08-16 supports the key chassis facts used above:

- HPE ProLiant DL785 G5: 8-socket class, up to 8 AMD Opteron processors, hundreds-of-GB memory capacity;
- HPE ProLiant DL785 G6 option documentation: DDR2 RDIMM options and heavy-population memory-speed reduction;
- Oracle Sun Fire X4640 service documentation: up to eight CPU modules, eight DIMM slots/module, 64 DDR2 DIMMs, up to 512 GB, six PCIe plus two PCI-X slots.

Marketplace prices and exact installed configurations are intentionally **not** treated as stable facts. Every purchase candidate must be re-verified from photos, part numbers and seller configuration.