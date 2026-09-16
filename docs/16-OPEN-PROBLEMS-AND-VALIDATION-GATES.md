# 16 — Open Problems and Validation Gates

This document is the engineering checklist for deciding whether Transit becomes a fast K3 inference machine rather than only a large-memory experiment.

Every open problem is paired with a concrete measurement or implementation path. No architecture is allowed to win by enthusiasm alone.

## 1. Performance terms

Use these terms consistently:

- **capacity** — bytes of model that can remain resident;
- **local memory GB/s** — measured useful bytes read from the memory attached to the compute doing the work;
- **Gweights/s** — weight elements actually consumed by the kernel;
- **weight-path tok/s equivalent** — `Gweights/s / 104e9` using the project's current K3 active-weight working value;
- **network payload** — activation/result/control traffic only;
- **end-to-end tok/s** — complete measured K3 generation, including all model operations.

No memory-bandwidth division is allowed to be presented as end-to-end K3 speed.

## 2. Open problem: exact K3 tensor inventory

### Problem

The repository currently uses working model numbers:

```text
~2.8T total parameters
~104B active parameters/token
~1.56 TB checkpoint
93 layers
896 experts
16 selected experts/token
MXFP4 routed weights
MXFP8 activations
```

These are sufficient for architecture sizing but not for final implementation.

### Resolution

Build `tools/k3_atlas_builder.py` against the exact checkpoint/configuration used.

It must emit:

```text
all tensor names
shape/dtype
bytes on disk
logical weight count
layer ownership
expert ownership
scale metadata layout
shared vs routed classification
checkpoint hash/version
```

### Gate

No final capacity, placement or tok/s claim until the atlas sums exactly to the checkpoint and every tensor required by the reference runtime is accounted for.

## 3. Open problem: MXFP4/MXFP8 vs the proven INT4/INT8 kernel

### Problem

The existing exact proof kernel is signed INT4 × INT8 bitplane arithmetic. K3's working routed-expert format is MXFP4/MXFP8.

### Resolution

Implement two software references:

1. exact native MXFP decode/compute;
2. explicit conversion to a Transit internal integer representation.

Compare:

```text
tensor error
expert output error
layer output error
model quality/perplexity/task behavior
compute cost
memory traffic
```

### Gate

A physical node/tile must match the selected software reference at the agreed numerical tolerance before performance work counts.

## 4. Open problem: old Opteron compute throughput

### Problem

A 64-DIMM DDR2 server may have enough aggregate memory bandwidth to be interesting while its CPUs are too slow to process the stream.

This is the largest unknown in the whole-server path.

### Resolution

Compile a small multi-version Transit kernel library:

```text
scalar reference
SSE2 path
SSE4a path where useful
POPCNT/ABM path where supported
architecture-specific unrolling/prefetch variants
```

Benchmark one NUMA socket first, then all sockets.

Record simultaneously:

```text
Gweights/s
GB/s local DDR reads
instructions/cycle
CPU utilization
remote NUMA bytes if available
wall power
```

### Gate

For a server with measured useful bandwidth `B GB/s`, define:

```text
memory_weight_limit = 2 * B Gweights/s   # 4 bits/weight
```

If the CPU kernel is substantially below that value, the machine is compute-limited.

Example:

```text
50 GB/s useful DDR payload
=> 100 Gweights/s memory-side ceiling
=> ~0.96 K3 weight-path tok/s/server
```

If the CPU reaches only 25 Gweights/s, the practical ceiling is only ~0.24 weight-path tok/s/server regardless of empty memory bandwidth.

## 5. Open problem: real DDR2 bandwidth at full population

### Problem

Full DIMM population can reduce memory clock. Vendor peak numbers are not sustained model-shaped bandwidth.

### Resolution

For every candidate server run:

1. `numactl --hardware` / topology inventory;
2. STREAM or equivalent per NUMA node;
3. aggregate all-socket sequential read test;
4. random/model-shaped large-page test;
5. Transit bitplane weight-stream benchmark;
6. repeat with the exact final DIMM population.

### Gate

Store results in `benchmarks/hardware/<node-model>/` and never use theoretical bus rate for procurement scaling after real measurements exist.

## 6. Open problem: NUMA locality

### Problem

An 8-socket machine can destroy its own effective bandwidth if threads read weights from remote sockets.

### Resolution

Model each server internally as eight small Transit tiles:

```text
server node
  numa0
  numa1
  ...
  numa7
```

Use:

- one worker group per NUMA node;
- CPU affinity;
- `mbind`/NUMA-aware allocation;
- first-touch on the owning CPU;
- local completion queues;
- atlas placement down to NUMA ID.

### Gate

A multi-socket run must report local vs remote memory traffic. If remote reads become a significant fraction of weight traffic, the placement is considered failed even if total GB/s looks acceptable.

## 7. Open problem: network topology

### Problem

The server route replaces custom PCIe fanout with a network fabric. Slow Ethernet can become the next bottleneck.

### Resolution

Bring-up ladder:

```text
1 GbE -> correctness only
10 GbE -> first real cluster
25/40 GbE or InfiniBand -> scale if needed
```

Start with persistent TCP connections and fixed buffers.

Profile:

```text
bytes/token
packets/token
head NIC utilization
per-node NIC utilization
serialization cost
kernel networking CPU time
round-trip latency
barrier/reduction latency
```

Only then decide whether to move to:

- RDMA verbs;
- UCX;
- libfabric;
- InfiniBand multicast/tree distribution;
- custom binary transport.

### Gate

Network time must be smaller than the local expert-compute window or overlap with it sufficiently that it is not the dominant uncovered latency.

## 8. Open problem: head-node network pressure

### Problem

A naive central scheduler may send the same activation separately to many selected experts and receive every expert result separately.

A rough planning sketch from document 14 is ~30.5 MiB dynamic payload/token if all 93 routed layers each send/receive vectors for 16 selected experts using the current 7 KiB/14 KiB examples.

### Resolution

Implement progressively:

1. direct unicast from head;
2. batch selected-expert commands per node;
3. one activation copy per node when multiple selected experts live there;
4. local combine/reduction within the server;
5. tree/multicast distribution if profiling justifies it.

### Gate

Head-node fabric must stay below 70–80% sustained utilization at the target tok/s so bursts, control traffic and failures do not collapse latency.

## 9. Open problem: distributed expert placement

### Problem

Random expert placement can create unnecessary network fanout and bandwidth hotspots.

### Resolution

Collect router traces from real prompts and build a placement optimizer using:

```text
expert activation frequency
co-activation frequency
expert bytes
node/NUMA bandwidth
node compute throughput
network locality
replica count
power state
```

First placement can be deterministic contiguous ranges. Optimization comes only after a correct runtime exists.

### Gate

Compare optimized placement against contiguous baseline using the same trace. Keep it only if measured tail latency or throughput improves.

## 10. Open problem: attention/shared/KV path

### Problem

The active-weight roofline is dominated by routed experts, but complete K3 inference also includes:

- attention projections;
- attention score/value work;
- normalization;
- router;
- embeddings/output head;
- residual operations;
- KV cache;
- shared/non-routed paths.

### Resolution

Initial partition:

```text
head/R920/optional GPU
  -> tokenizer
  -> router
  -> attention/shared operations
  -> KV/state
  -> global reductions

Transit memory-compute nodes
  -> routed expert heavy lifting
```

Then profile layer by layer.

### Gate

No architecture claims end-to-end K3 speed until one full token completes through this partition and every major stage is timed.

## 11. Open problem: can K3 be split 100% across 10–20 servers?

### Answer in architecture terms

Yes, the checkpoint and computation graph can be partitioned across multiple machines. The unresolved question is **performance**, not whether bytes can be assigned to nodes.

A correct implementation must avoid a design where every layer repeatedly reconstructs the whole model on one machine.

### Practical partition

```text
head node
  shared model state + generation loop

server group
  resident expert shards / selected weight-heavy tensors

optional additional nodes/GPU
  attention/shared operations
```

For a pure CPU-server implementation, even the head/shared path can also be distributed, but the first version should keep the control plane centralized to reduce complexity.

### Gate

The first success criterion is one complete token with no missing tensor and deterministic output behavior. Throughput optimization comes after correctness.

## 12. Open problem: startup/model loading

### Problem

A 1.56 TB-class checkpoint cannot be reloaded over the network on every inference request.

### Resolution

Persistent residency:

```text
cluster boot
-> node discovery
-> atlas assignment
-> transfer/read assigned shards once
-> checksum
-> pin/reserve memory
-> mark resident
-> serve many requests
```

Whenever possible, store each node's assigned shard files on local SSD so reboot recovery does not require the head to stream the whole checkpoint again.

### Gate

Normal token generation contains zero checkpoint-weight network transfers except recovery/rebalancing.

## 13. Open problem: API/agent integration

### Problem

A distributed runtime is not useful if every application must understand its topology.

### Resolution

Expose one OpenAI-compatible gateway on the head node.

Minimal endpoints:

```text
GET  /v1/models
POST /v1/completions
POST /v1/chat/completions
```

Internal flow:

```text
request
-> tokenizer
-> generation loop
-> Transit scheduler
-> distributed layer/expert calls
-> sampled token
-> stream response
```

Agent frameworks connect to one `base_url`; cluster details remain private to Transit.

Tool calling, conversation memory and agent loops live above the inference layer and do not change DDR placement.

### Gate

A standard client can send a prompt to the gateway and receive streamed tokens without knowing node IDs.

## 14. Open problem: failure handling

### Problem

With 10–20 old servers and hundreds of old DIMMs, failure probability matters.

### Resolution

Weight Atlas stores replica state and checksums.

Node health states:

```text
READY
DEGRADED
DRAINING
FAILED
RECOVERING
```

On failure:

```text
stop routing new work
use replica if available
or reload shard onto spare capacity
update placement generation
retry affected sequence
```

### Gate

Kill one worker/node during a synthetic multi-node run and demonstrate deterministic recovery or a clean bounded error rather than cluster deadlock.

## 15. Open problem: old-server power

### Problem

A server that costs almost nothing can still be economically terrible if it consumes hundreds of watts or more continuously.

### Resolution

Measure wall power:

```text
powered-off standby
idle Linux
memory benchmark
Transit kernel
full cluster inference
```

Track:

```text
W/server
J/token
W/Gweights/s
monthly energy at expected duty cycle
```

### Gate

Purchase price and energy cost are separate columns in every architecture comparison.

A 20-server design is rejected as a final always-on machine if its power cost dominates the intended low-cost advantage, even if acquisition cost is excellent. It may still be useful as a proof rig run on demand.

## 16. Open problem: cooling/noise/physical installation

### Problem

Old 4U multi-socket servers were designed for datacenters, not a quiet office.

### Resolution

Before scaling beyond two machines, record:

```text
rack units
weight
fan noise
inlet temperature
exhaust temperature
circuit current
PSU connector requirements
```

Plan a dedicated rack/power/cooling location if the architecture survives the compute benchmark.

### Gate

No 10–20-server purchase before confirming the site can power and cool them safely.

## 17. Open problem: DDR2 DIMM sourcing

### Problem

Cheap servers only help if matching high-density DIMMs can also be bought cheaply enough.

### Resolution

Prefer large homogeneous lots and respect OEM population rules.

For every DIMM lot capture:

```text
part number
capacity
speed grade
rank
ECC/registered/FB-DIMM type
quantity
price
seller test status
server compatibility
```

### Gate

Buy a small sample first and pass memtest in the target server before buying the lot.

## 18. Token/s expectation ladder

### DDR3 custom-tile architecture

Existing document 10 gives the ideal Q4 weight-path roofline for 304 channels:

```text
DDR3-1600 ~74.8 tok/s equivalent
DDR3-1866 ~87.3 tok/s equivalent
DDR3-2133 ~99.8 tok/s equivalent
```

These are bandwidth rooflines only.

### DDR2 whole-server architecture

Using the conservative illustrative full-population roofline from document 14:

```text
~68.3 GB/s theoretical/server
~1.31 weight-path tok/s/server ideal
```

Indicative cluster numbers:

```text
5 servers  ~6.6 ideal weight-path tok/s
10 servers ~13.1 ideal weight-path tok/s
20 servers ~26.3 ideal weight-path tok/s
```

At 60–75% memory efficiency before compute overhead:

```text
5 servers  ~3.9–4.9
10 servers ~7.9–9.8
20 servers ~15.8–19.7
```

The CPU kernel may reduce these sharply.

### Honest end-to-end expectation

There is currently **no defensible end-to-end K3 tok/s number for the DDR2 server cluster** because the old-CPU Gweights/s benchmark does not exist yet.

The first useful prediction will be generated from:

```text
measured_server_Gweights_s
measured_network_bytes_s
measured_shared_path_latency
measured_reduction_latency
exact K3 execution trace
```

## 19. Server-path go/no-go formula

After benchmarking one server:

```text
P = measured effective Gweights/s/server
N = affordable server count

expert_weight_path_tok_s ~= N * P / 104
```

when `P` is expressed in Gweights/s.

Examples:

```text
P = 25 Gweights/s, N = 10 -> ~2.4 tok/s equivalent
P = 50 Gweights/s, N = 10 -> ~4.8 tok/s equivalent
P = 100 Gweights/s, N = 10 -> ~9.6 tok/s equivalent
P = 100 Gweights/s, N = 20 -> ~19.2 tok/s equivalent
```

This is the most important benchmark-derived scaling equation for the DDR2 path.

## 20. Decision matrix

| Criterion | DDR3 FPGA tiles | Whole DDR2 servers | Hybrid |
|---|---|---|---|
| Upfront controller engineering | hard | easy | medium |
| Capacity/$ if scrap-priced | good | potentially excellent | excellent |
| Local memory bandwidth/node | high target | moderate | mixed |
| Local compute flexibility | high | low/medium CPU | high for hot path |
| Power efficiency | potentially better | likely poor | tunable |
| Time to first physical proof | medium | fast | fast |
| Final 100 tok/s plausibility | best current path | unlikely without exceptional CPU results/scale | plausible if hot path moves to tiles |
| Use of loose DIMMs | possible with final board | native server slots | both |
| Networking complexity | PCIe fabric | Ethernet/IB fabric | both |

## 21. Immediate measurement sequence

The next experimental milestone for the server path is one complete 64-slot-class machine, not ten.

Run in this order:

```text
A. hardware inventory
B. DIMM/NUMA topology
C. STREAM per socket
D. aggregate memory bandwidth
E. ISA feature inventory
F. Transit INT4×INT8 kernel per socket
G. aggregate Gweights/s
H. wall power
I. 10 GbE-class NIC
J. remote activation -> local resident-weight compute -> result
K. one real K3 expert/shard
```

Then fill one row in a reproducible benchmark record:

```text
server model
CPU configuration
DIMM configuration
measured GB/s
measured Gweights/s
weight-path tok/s equivalent
network GB/s
wall W
J/token-equivalent
purchase price
```

Only measured rows should drive quantity decisions.

## 22. Exact missing discoveries

The project should keep searching for any surplus object that collapses one of these problems:

### Discovery target A — ideal DDR3 tile

```text
PCIe endpoint
+ programmable compute
+ 4–8 independent DDR3 channels
+ large local capacity
+ documented toolchain
+ scrap price
```

### Discovery target B — absurdly cheap complete memory server

```text
48–64+ DIMM sockets
4–8 sockets/NUMA nodes
all CPU/memory modules present
fast-NIC-capable PCIe
Linux support
near-zero market value
```

### Discovery target C — documented server memory board

```text
multi-DIMM riser/backplane
+ public buffer/protocol documentation
+ known pinout/power
+ controllable from FPGA
```

### Discovery target D — obsolete FPGA/emulation appliance

```text
multiple FPGAs
large external DDR memory
host interface
public/obtainable toolchain
used price far below dev-board value
```

### Discovery target E — cheap fabric

```text
10/25/40 GbE or QDR/FDR IB NICs + switch + DACs
Linux-supported
bulk surplus pricing
```

## 23. Rule for future ideas

Every new hardware idea must answer five questions before it is considered a Transit candidate:

```text
1. Where do the K3 weights physically reside?
2. What reads those weights every token?
3. Where does the dot/MVM/expert compute happen?
4. What bytes cross the slowest interconnect per token?
5. What measured quantity can falsify the idea cheaply?
```

If those five answers are not concrete, the idea stays in research notes and does not become a purchase plan.