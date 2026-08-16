# 11 — DDR3 Tile Architecture

This document describes the current system-level Transit design.

## 1. Design statement

The current target is:

```text
1 × Dell PowerEdge R920
38 × logical Transit memory-compute tiles
8 × independent DDR3 channels per tile
--------------------------------------
304 × DDR3 channels total outside/alongside the host memory system
```

The R920 is the host/orchestrator. It is not expected to provide 304 DDR3 channels directly through its motherboard.

The 304 channels belong to active external tiles attached through a PCIe switch/fan-out fabric.

## 2. Why a tile exists

A tile solves four problems at once:

1. it owns local DDR3 PHY/controllers;
2. it stores a fixed subset of model weights;
3. it computes on those weights locally;
4. it exposes one host-facing endpoint instead of many DDR buses.

The abstraction is:

```text
                    one PCIe endpoint
                           |
                     command engine
                           |
            +--------------+--------------+
            |                             |
      activation buffer              result buffer
            |                             ^
            v                             |
       local compute / reduction engine---+
            |
   +--------+--------+-------- ... -------+
   |        |        |                   |
 DDR ch0  DDR ch1  DDR ch2             DDR ch7
   |        |        |                   |
 weights  weights  weights             weights
```

A tile should look like **one device** to the R920 even if it contains multiple DDR controllers and multiple internal compute engines.

## 3. Why 38 endpoints instead of 304 endpoints

A literal design with one PCIe card/controller per DDR3 channel creates:

- 304 PCIe endpoints;
- large PCIe bus-number/MMIO pressure;
- huge cable/riser count;
- excessive power-conversion duplication;
- many FPGAs/controllers;
- difficult software enumeration;
- unnecessary cost.

Grouping eight memory channels behind one endpoint reduces the system to about 38 endpoints while preserving local parallel memory paths.

This is the architectural meaning of:

```text
38 × 8 = 304
```

## 4. R920 role

The R920 should perform functions that are naturally centralized:

- boot and discover tiles;
- load/verify the model atlas;
- stage weights from NVMe/network into tile DDR3;
- choose placements/replicas;
- run or coordinate the K3 router;
- send activations and commands;
- collect tile outputs;
- perform final cross-tile reduction where needed;
- maintain KV/state if the chosen partition keeps it host-side;
- telemetry/error handling;
- checkpoint reload/recovery.

The R920 should **not** read all routed expert weights into CPU RAM and then resend them to the tiles for each token.

## 5. Host-memory correction

The R920 has many DIMM sockets and memory risers, but physical DIMM/riser topology is not equivalent to 32 or 96 freely independent full-bandwidth DDR channels.

For Transit planning:

- treat the R920's internal RAM as a measured NUMA memory system;
- bind host workloads to local NUMA nodes;
- use internal RAM for checkpoint staging, cache, KV/state and host computation;
- do not add a simplistic `number of visible DIMM paths × DDR3-1600 bandwidth` to the external tile roofline.

The external 304-channel target is separate and explicit.

## 6. PCIe fabric

A practical topology is a switch tree rather than trying to plug 38 cards directly into ten physical server slots.

Conceptually:

```text
R920 PCIe root ports
   |
   +-- switch A
   |     +-- tile 00
   |     +-- tile 01
   |     +-- ...
   |
   +-- switch B
   |     +-- tile 10
   |     +-- tile 11
   |     +-- ...
   |
   +-- switch C
         +-- ...
```

Requirements for the switch/fan-out layer:

- transparent PCIe switching where possible;
- enough downstream ports;
- sufficient upstream aggregate bandwidth for activation/result traffic;
- stable enumeration under Linux;
- sufficient bus numbers/MMIO space;
- ACS/IOMMU configuration understood rather than accidental;
- hotplug is optional for the first machine;
- powered risers/cables must preserve signal integrity.

## 7. Mining risers: correct use

Cheap mining risers are not memory expanders and not lane multipliers.

Their valid Transit use is:

```text
PCIe downstream port
      |
small x1 edge adapter
      |
USB-style cable used as PCIe physical transport by the riser
      |
powered x16 mechanical socket
      |
Transit endpoint card
```

The endpoint still gets the link width/speed negotiated by the riser path, often x1.

This can still be useful because local DDR traffic remains on the tile.

Before using them in quantity, test:

- Gen1/Gen2/Gen3 stability;
- error counters;
- sustained DMA;
- simultaneous multi-riser operation;
- 12 V/3.3 V quality;
- ground/reference integrity;
- cable length and EMI.

For final high-reliability hardware, direct switch backplanes/cabled PCIe may replace hobby mining risers while preserving the same logical topology.

## 8. PCIe bandwidth budget

The PCIe uplink must carry:

- activation vectors;
- command descriptors;
- router/expert IDs;
- block-scale metadata if not resident;
- result vectors;
- telemetry.

It should not carry active weight matrices per token.

Using 7168 as a rough hidden-vector size:

```text
8-bit activation  ~7 KiB
16-bit result     ~14 KiB
```

Even if an activation is sent to multiple selected experts, aggregate communication can remain far below the multi-terabyte/s local weight-read path.

The exact requirement must be generated from a K3 execution trace.

The important design rule is:

> Size PCIe from activation/result traffic, not from local DDR bandwidth.

## 9. Eight-channel tile bandwidth

Nominal local payload per eight-channel tile:

```text
DDR3-1600  8 × 12.8  = 102.4 GB/s
DDR3-1866  8 × 14.93 = 119.44 GB/s
DDR3-2133  8 × 17.07 = 136.56 GB/s
```

These are ideal bus payload numbers. Real sustained reads will be lower because of:

- controller efficiency;
- refresh;
- row/bank conflicts;
- burst shape;
- ECC width/overhead depending topology;
- FPGA clock-domain boundaries;
- compute backpressure;
- layout and DIMM timing.

The first tile must report measured sustained sequential and model-shaped traffic.

## 10. Eight-channel tile capacity

Example with one DIMM per channel:

```text
8 × 8 GB  = 64 GB/tile
8 × 16 GB = 128 GB/tile
8 × 32 GB = 256 GB/tile
```

Across 38 tiles:

```text
64 GB/tile  -> 2.432 TB
128 GB/tile -> 4.864 TB
256 GB/tile -> 9.728 TB
```

A tile may support more than one DIMM/channel if electrically designed for it, but additional DIMMs on a channel should be treated as capacity expansion, not new bandwidth channels.

## 11. Weight placement

The Weight Atlas assigns every resident shard to:

```text
tile_id
channel_id
local_address
length
format
scale_metadata
checksum
```

For MoE experts, the primary optimization objective is to minimize expensive cross-tile partial reductions.

Ideal case:

```text
selected expert E
  -> all/most E weight shards live on one tile
  -> tile computes full expert output locally
  -> host receives one reduced output vector
```

If an expert is larger than one tile's preferred bandwidth/capacity slice, shard it across a small fixed group of tiles and reduce within that group before returning to the host if possible.

## 12. Tile command protocol

A minimal command descriptor can contain:

```text
opcode
sequence_id
layer_id
expert_id
tensor/shard_id
activation_buffer_address
result_buffer_address
activation_length
result_length
format/scaling flags
checksum/version
```

Suggested command classes:

```text
RESET
IDENTIFY
LOAD_WEIGHT_BLOCK
VERIFY_WEIGHT_BLOCK
RUN_DOT/MVM
RUN_EXPERT
READ_COUNTERS
BARRIER
ABORT
```

For the first prototype, simplicity and deterministic debugging matter more than clever queue semantics.

## 13. Persistent model load

Startup sequence:

```text
1. R920 reads K3 atlas
2. enumerate all tiles
3. verify tile firmware/version/capacity
4. assign expert/tensor shards
5. DMA weights once into local DDR3
6. tile computes checksum
7. mark shard resident
8. begin inference
```

During inference, the normal path should not reload those weights unless:

- model changes;
- a tile resets;
- an expert is dynamically rebalanced;
- a replica is created/removed.

## 14. Token execution sketch

For a routed expert layer:

```text
host computes/receives activation
      |
router chooses 16 experts
      |
Weight Atlas maps experts -> tiles
      |
activation DMA to selected tile(s)
      |
RUN_EXPERT command
      |
selected tiles read local DDR3 in parallel
      |
local bitplane/MXFP compute
      |
local expert output + routing weight
      |
result DMA
      |
host/tile-group reduction
      |
next model operation
```

Inactive tiles should not read their expert weights.

## 15. Reduction hierarchy

Preferred hierarchy:

```text
inside compute lane
   -> inside DDR channel engine
      -> inside tile
         -> optional tile group
            -> R920
```

Every level should reduce data before sending it upward where mathematically legal.

This is one of the central differences between Transit and a remote-memory architecture.

## 16. Tile telemetry

Every tile should expose counters from day one:

```text
commands completed/errors
PCIe bytes RX/TX
DDR bytes read/written
DDR useful payload GB/s
weights processed
compute active cycles
DDR stall cycles
PCIe input stall cycles
result/output stall cycles
ECC/errors if available
clock/temperature/voltage if available
```

Host software should log these per command so end-to-end bottlenecks can be attributed.

## 17. Failure handling

Because weights are static, recovery can be straightforward:

```text
tile failure
   -> mark resident shards unavailable
   -> use replica if present
   -> otherwise reload shard on spare tile
   -> update atlas placement
```

Checksums should be verified after initial load and after any suspected memory/link error.

## 18. Prototype vs final tile

### Laboratory tile

Current candidate: YPCB-00338-1P1.

Purpose:

- prove PCIe endpoint path;
- prove FPGA local-DDR read;
- port bitplane arithmetic;
- measure compute/DDR balance;
- validate host command protocol.

It has only two local DDR3 channels/banks and limited capacity.

### Final logical tile

Target:

```text
1 host-facing PCIe endpoint
8 independent DDR3 channels
DIMM-friendly or otherwise very cheap large local capacity
programmable/local compute
local reduction
open/documented enough to maintain
```

The final physical implementation could be:

- one large FPGA with enough I/O banks;
- multiple DDR PHY FPGAs behind one local controller/endpoint;
- a documented enterprise memory board plus programmable front end;
- a surplus accelerator with many memory channels;
- a custom PCB only if surplus solutions fail economically.

## 19. Non-negotiable architecture rule

The project should reject any proposal that silently turns this back into:

```text
external DDR3 -> PCIe -> R920 RAM/CPU -> PCIe -> accelerator
```

for every active weight every token.

The weight path must remain local to the memory-compute tile.
