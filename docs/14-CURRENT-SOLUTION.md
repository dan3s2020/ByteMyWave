# 14 — Current Transit Solution

This is the current integrated answer to the project: what we build, why it works, what code runs where, and what must be proven next.

## 1. The machine

The target machine is not a conventional multi-GPU server and not 20 old servers connected together.

It is one host plus many cheap local-memory compute islands:

```text
                         Dell R920
              scheduler / router / reducer
                model atlas / state / NVMe
                             |
                      PCIe root ports
                             |
                     PCIe switch tree
                             |
       +---------------------+---------------------+
       |                     |                     |
   Transit tile 0        Transit tile 1       ... tile 37
   one endpoint          one endpoint              one endpoint
       |                     |                         |
   8 DDR3 channels       8 DDR3 channels          8 DDR3 channels
       |                     |                         |
   local weights         local weights             local weights
       |                     |                         |
   local compute         local compute             local compute
       +---------------------+-------------------------+
                             |
                        reduced results
```

Scale target:

```text
38 tiles × 8 independent DDR3 channels = 304 channels
```

The number `304` is a bandwidth-sizing target, not a requirement to put 304 DIMMs in the R920 motherboard.

## 2. The idea in one sentence

> Broadcast small dynamic activations to the memory that already owns the selected weights, compute there, and return only the reduced result.

This reverses the conventional problem.

Bad path:

```text
weights -> central host -> PCIe -> accelerator -> repeat every token
```

Transit path:

```text
weights loaded once -> local tile DDR3 -> stay resident
                                  ^
                                  |
                         tiny dynamic activation
                                  |
                              local compute
                                  |
                              result vector
```

## 3. Why K3 makes this attractive

The working K3 model is a sparse MoE system:

```text
~2.8T total parameters
~104B active parameters/token
896 experts
16 selected experts/token
93 layers
```

The routing sparsity should exist physically:

```text
router chooses 16 experts
   |
   +-- wake/read only the tiles containing those experts
   +-- leave the other expert weights stationary and idle
```

At a Q4-equivalent 4 bits/weight, reading ~104B active weights once is roughly 52 GB/token. The 100 tok/s weight-path roofline is therefore ~5.2 TB/s.

304 ideal DDR3-2133 x64 channels are also about 5.19 TB/s aggregate. That is why the 38×8 topology is a useful starting point.

It is not a promise of 100 end-to-end tok/s. It gives the design enough cheap physical parallelism to make that target worth engineering toward.

## 4. What the R920 does

The R920 is the control plane and the stateful host.

It runs:

```text
K3 checkpoint parser
Weight Atlas
placement manager
router/scheduler
PCIe command queues
result collection/reduction
KV/state chosen to stay host-side
telemetry
failure recovery
NVMe staging
```

It does not need to see every expert weight on the hot path.

## 5. What each Transit tile does

A final logical tile contains:

```text
PCIe endpoint
command/completion queues
activation DMA buffer
8 independent DDR3 controllers/PHYs
8 local weight streams
bitplane or native-MXFP compute engine
local accumulators
scale/format stage
result DMA buffer
performance/error counters
```

The tile should enumerate as one PCIe device even if the physical implementation uses multiple internal memory-controller chips/FPGAs.

## 6. What mining risers are for

Mining risers are not RAM risers and not bandwidth multipliers.

For the lab, they are simply cheap powered PCIe extenders:

```text
PCIe switch downstream port
       |
     x1 link
       |
 powered mining riser
       |
 Transit endpoint card
```

A x1 link remains x1. That can still be enough because it carries dynamic data, not the local 100+ GB/s tile weight stream.

The final rack may use proper switch backplanes/cabled PCIe instead; the logical architecture is unchanged.

## 7. Weight storage

Before inference, the checkpoint is transformed into a machine-readable Weight Atlas.

A shard record contains at least:

```text
tensor/expert identity
shape
stored numerical format
byte length
block-scale metadata
checksum
tile ID
DDR channel ID
local DDR address
replica group
```

Boot/load sequence:

```text
R920 reads checkpoint + atlas
        |
enumerates tiles
        |
assigns expert/tensor shards
        |
DMA each shard to its destination tile once
        |
tile verifies checksum
        |
weights become RESIDENT
```

Inference starts only after the required resident set is verified.

## 8. Token/expert execution protocol

For one routed expert operation:

```text
1. host/router obtains input activation
2. router chooses expert IDs
3. atlas maps expert IDs to tile IDs
4. host DMA-sends activation to selected tiles
5. host posts RUN_EXPERT descriptors
6. tiles read resident local DDR3 in parallel
7. local kernel computes and accumulates
8. tile applies required scale/output conversion
9. tile DMA-returns reduced output vector
10. host or tile-group combines selected-expert outputs
11. execution continues
```

Inactive tiles do not stream their weights.

## 9. Command descriptor

The first protocol should be boring and fixed-size.

A 64-byte descriptor is enough for an initial implementation:

```text
magic/version
opcode/flags
sequence ID
layer ID
expert ID
shard ID
activation buffer offset/length
result buffer offset/length
format ID
scale/metadata ID
reserved/check fields
```

Commands:

```text
IDENTIFY
RESET
LOAD_WEIGHT_BLOCK
VERIFY_WEIGHT_BLOCK
RUN_DOT
RUN_MVM
RUN_EXPERT
READ_COUNTERS
BARRIER
ABORT
```

The tile completion record returns status plus timing/counter information.

## 10. Proven computation kernel

The hardware proof starts from the arithmetic already demonstrated exactly on x64.

Signed INT4 weight:

```text
q = b0 + 2*b1 + 4*b2 - 8*b3
```

Signed INT8 activation:

```text
x = a0 + 2*a1 + 4*a2 + 8*a3
  + 16*a4 + 32*a5 + 64*a6 - 128*a7
```

For each 64-bit block:

```text
P[i][j] = popcount(Wplane[i] & Aplane[j])
```

Then:

```text
dot = sum_i,j coeffW[i] * coeffA[j] * P[i][j]
```

The already-tested native assembly kernel uses the same identity and reached exact integer equality (`maxdiff=0`).

The FPGA implementation therefore does not begin with an untested mathematical invention. It begins by reproducing an already exact reference on different hardware.

## 11. FPGA kernel pipeline

Transit V1 hardware pipeline:

```text
activation arrives through PCIe
          |
activation buffer
          |
bitplane transpose/encoder
          |
          +------------------------------------------+
          |                                          |
DDR3 burst readers -> weight bitplane FIFOs          |
          |                                          |
parallel AND + popcount lanes <----------------------+
          |
fixed coefficient accumulation
          |
row/expert accumulator RAM
          |
scale/output stage
          |
result DMA
```

The first implementation may use fewer compute lanes than final hardware. The objective is to expose whether DDR, popcount, scaling or PCIe is actually the bottleneck.

## 12. Numerical bridge to real K3

The proven kernel is signed INT4×INT8. Published K3 routed-expert weights/activations use MXFP4/MXFP8-style representations.

We therefore implement and compare two explicit paths.

### Native-format path

```text
MXFP4/MXFP8 bytes
 -> decode significands/signs
 -> bitwise/integer accumulation where applicable
 -> apply block scales correctly
 -> compare to software MXFP reference
```

### Transit-format path

```text
K3 checkpoint
 -> one-time calibrated conversion
 -> signed INT4 weights + stored scales
 -> exact proven bitplane engine
```

The second path is only acceptable if real model-quality tests show the conversion is good enough. It is not silently assumed.

## 13. First physical board

The current lab candidate is YPCB-00338-1P1:

```text
PCIe
  |
Kintex-7 FPGA
  |
2 local DDR3 banks/channels
```

It is useful because public reverse-engineering/LiteX work already removes much of the board-bring-up uncertainty.

It is **not** the final tile:

```text
lab board:   2 DDR3 paths, soldered memory
final tile: ~8 DDR3 paths, large cheap capacity, one endpoint
```

We use one board to prove the complete logical pipeline before designing/buying 38 final tiles.

## 14. First physical proof

The first success test is deliberately small:

```text
host
  sends 2048-byte/encoded activation command
      |
PCIe
      |
FPGA
  reads real resident bitplane weights from local DDR3
  computes one matrix-vector slice
  returns int accumulator/result
      |
host compares to software reference
```

Acceptance:

```text
exact signed-INT proof: max_abs_diff == 0
DDR bytes counted correctly
PCIe bytes counted correctly
no weight bytes returned to host
```

Then increase tensor size until local DDR3 is the actual working set.

## 15. Performance counters are part of the design

Every command must make bottlenecks observable.

Per tile:

```text
cycles total
compute-active cycles
DDR-stall cycles
activation-input stall cycles
result-output stall cycles
DDR bytes read/written
PCIe bytes RX/TX
weight elements processed
commands/errors
ECC/link errors where available
```

Without these counters we would repeat the earlier mistake of guessing whether compute, SSD, RAM or communication is limiting us.

## 16. Host software modules

Minimal software architecture:

```text
host/atlas.py
  parse/query placement map

host/protocol.py
  command/completion binary structs

host/device.py
  mmap/BAR/DMA backend abstraction

host/loader.py
  load and checksum resident weight shards

host/router.py
  K3 expert selection integration

host/scheduler.py
  dispatch commands to selected tiles

host/reduce.py
  combine tile/expert results

host/telemetry.py
  counters, tracing, failure detection

host/cli.py
  discover/load/test/run
```

The initial implementation can use Python for orchestration and C/C++/Rust only where host overhead proves relevant. The hot weight computation is on the tile.

## 17. RTL modules

Logical RTL partition:

```text
rtl/pcie_endpoint/
  BAR/control + DMA queues

rtl/command/
  descriptor parser
  completion writer

rtl/ddr/
  channel wrappers
  burst schedulers

rtl/bitplane/
  activation encoder
  AND/popcount lanes
  coefficient reducer
  row/expert accumulators

rtl/format/
  scale/MXFP logic

rtl/counters/
  performance/error counters
```

The first generic bitplane core can be simulated independently of the board's PCIe/MIG plumbing.

## 18. Scaling from 2 DDR paths to 8

The lab board answers the computational question. The final hardware search answers the economic/physical question.

Final tile options, in preference order:

```text
A. surplus documented accelerator/memory board with ~8 DDR3 paths
B. documented enterprise memory board + programmable front end
C. several cheap 2-channel memory engines grouped behind one endpoint
D. one high-pin-count FPGA custom 8-channel PCB
```

Option C is important: `8 channels/tile` is a **logical** tile requirement. It does not force one enormous FPGA to terminate all eight DIMM buses if four cheap 2-channel sub-engines can share one upstream endpoint economically.

## 19. Final 38-tile physical organization

A possible rack layout:

```text
R920
 |
 +-- PCIe switch domain A -> tiles 00..09
 +-- PCIe switch domain B -> tiles 10..19
 +-- PCIe switch domain C -> tiles 20..28
 +-- PCIe switch domain D -> tiles 29..37

separate power distribution
forced-air cooling
service/JTAG access
stable tile IDs derived from PCIe topology/EEPROM
```

The exact switch count is chosen after measuring the real activation/result traffic and available root-port topology.

## 20. What happens after the first tile works

Do not optimize the laptop CPU again. Do this:

```text
1 tile
 -> exact real DDR resident-weight proof
2 tiles
 -> parallel dispatch/reduction
4 tiles
 -> PCIe switch behavior
8 tiles
 -> contention + scheduler behavior
1 real K3 expert
 -> numerical-format proof
1 K3 routed layer
 -> routing/placement proof
1 complete K3 token
 -> real end-to-end profile
38×8 fabric
 -> only after the measured bottleneck justifies it
```

## 21. What is known, unknown, and therefore actionable

### Known

- stationary weights are the key architectural direction;
- bitplane signed INT4×INT8 arithmetic is exact and measured;
- host RAM/SSD overlap is practical;
- K3 active-weight bandwidth is the dominant roofline-sized problem;
- a modular tile/fan-out architecture avoids putting hundreds of DIMMs/controllers on the R920 motherboard;
- PCIe need not carry local DDR weight traffic.

### Unknown but now testable

- sustained FPGA local-DDR bitplane rate;
- native MXFP4/MXFP8 hardware mapping;
- exact K3 activation/result traffic per tile;
- final 8-channel surplus hardware choice;
- PCIe switch topology at 38 endpoints;
- real end-to-end K3 token rate.

Those are no longer vague questions. Each one has a specific experiment in the roadmap.

## 22. The current build decision

If hardware is bought now, buy **one known-compatible lab FPGA card**, not 38 final boards and not hundreds of risers.

Once it runs this exact path:

```text
resident DDR weight
+ PCIe activation
+ local exact compute
+ reduced PCIe result
```

we have the essential Transit machine in miniature. Scaling then becomes a measured engineering/cost problem rather than a conceptual gamble.
