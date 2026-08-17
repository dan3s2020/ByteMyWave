# Assembly runbook — one node, then x10

Date: **2026-08-17**

This runbook describes the mechanical and low-voltage assembly of the proposed node. It intentionally does **not** prescribe building mains wiring. Facility electrical distribution and HVAC must be handled by qualified professionals.

## 1. Canonical per-node layout

```text
Dell PowerEdge R920

CPU/RAM:
    4 x E7-4890 v2-class sockets
    16 x 16GB DDR3 ECC RDIMM = 256GB/node

PCIe:
    slot 3 x8  -> ConnectX-3 FDR56 NIC
    slot 4 x16 -> RTX 3060 12GB via true x16 powered riser
    slot 5 x16 -> GTX 1060 worker #1 via true x16 powered riser
    slot 6 x16 -> GTX 1060 worker #2 via true x16 powered riser
    slot 7 x16 -> GTX 1060 worker #3 via true x16 powered riser
    slot 8 x16 -> GTX 1060 worker #4 via true x16 powered riser
    slot 9 x16 -> GTX 1060 worker #5 via true x16 powered riser

storage:
    1 x 240GB SATA SSD in correct Dell caddy

external:
    1 x six-GPU open frame
    1 x 1200W-class ATX PSU for external GPUs/risers
```

The six GPU links use slots 4-9 because this is the six-x16 topology retained from the R920 hardware documentation when four CPUs are installed. The network card is deliberately placed on x8 so it does not consume a worker link.

## 2. Pre-assembly inspection

Before installing anything, photograph and inventory each R920.

Required server-side proprietary pieces:

```text
all intended CPU sockets usable
all required CPU heatsinks
all eight R920 memory riser boards
fan modules present
PCIe expansion hardware present
server PSUs present and matched
PERC/storage path present or known alternative boot path
iDRAC working
no obvious burned connectors/corrosion
```

Do not accept a cheap R920 lot if missing memory riser boards convert it into a parts hunt that destroys the economics.

## 3. CPU and RAM

If the processors are not already installed, install four matching CPUs and correct heatsinks according to Dell service procedures.

Populate RAM symmetrically. First-stage target:

```text
CPU1: 4 x 16GB
CPU2: 4 x 16GB
CPU3: 4 x 16GB
CPU4: 4 x 16GB
```

Goal: one DIMM on each of the 16 native memory channels before adding second DIMMs/channel.

Rules:

```text
use ECC registered modules validated by R920
prefer identical part number/rank/voltage
never mix RDIMM and LRDIMM in one system
verify memory speed and channel population in BIOS/iDRAC
```

Cluster result at 16 DIMMs/node:

```text
256GB/node * 10 = 2.56TB nominal RAM
```

## 4. Boot storage

Install the SATA SSD in the compatible Dell caddy. Use it for:

```text
Linux OS
CUDA/driver stack
runtime binaries
logs/metrics
small caches/configuration
```

It is not the primary K3 weight-capacity device.

## 5. Network card

Install the ConnectX-3 FDR56 card in the chosen x8 slot with a proper full-height bracket.

Do not connect the whole ten-node fabric initially. On node 1 validate enumeration and firmware first. The second node enables the first real FDR point-to-point/switch test.

## 6. GPU risers

Use only true x16 electrical risers.

Rejected designs:

```text
PCIe x1 mining riser -> x16 physical slot
USB-cable mining riser
passive arrangement that relies on unsupported server slot power for six external consumer GPUs
```

Preferred riser requirements:

```text
host connector: PCIe x16
GPU connector: PCIe x16
all 16 lanes carried
PCIe Gen3 stable at required cable length
separate slot-power input
mechanically secured cable/connectors
```

For the first node buy six risers only. Do not bulk-order 60 until simultaneous-bandwidth validation passes.

## 7. External GPU frame

Mount externally:

```text
position 0: RTX 3060 12GB
position 1: GTX 1060 6GB worker #1
position 2: GTX 1060 6GB worker #2
position 3: GTX 1060 6GB worker #3
position 4: GTX 1060 6GB worker #4
position 5: GTX 1060 6GB worker #5
```

Leave enough spacing for direct airflow and cable bend radius. The GPU frame should be electrically/mechanically stable and should not hang PCIe cables by connector tension.

## 8. External GPU power

The external ATX PSU powers GPU auxiliary connectors and the powered risers according to their manufacturers' requirements.

Rules:

```text
use native PCIe 6/8-pin GPU leads for GPU auxiliary power
avoid SATA-to-GPU 8-pin adapters
avoid overloaded splitters
size each cable path for the connected board load
power risers from connectors explicitly supported by the riser design
```

The R920 remains responsible for its motherboard/CPUs/RAM/fans/storage/NIC. The external GPU PSU is not a substitute for Dell's internal server PSUs.

Startup synchronization between the server and external PSU must use a proper supported control method/adapter or technician-designed solution. Do not use improvised exposed-wire jumpers as the final cluster implementation.

## 9. First power-on

Power up with only the minimum hardware first:

```text
R920 + CPUs + RAM + SSD + NIC
```

Confirm:

```text
POST clean
iDRAC accessible
256GB detected
all CPU sockets detected
no memory channel errors
SSD available
NIC enumerated
```

Then add one GPU/riser at a time.

Sequence:

```text
RTX3060 only
+ worker 1
+ worker 2
+ worker 3
+ worker 4
+ worker 5
```

At every step verify PCIe link width/speed and stability before adding the next board.

## 10. OS/software bring-up

Recommended first-node stack conceptually:

```text
Linux
NVIDIA driver supporting Ampere + Pascal
CUDA toolchain capable of sm_61 and sm_86 builds
OFED/rdma-core/Mellanox tools as appropriate
numactl/hwloc
benchmark suite from ByteMyWave
telemetry: nvidia-smi, perf, lm-sensors/IPMI/iDRAC metrics
```

Build separate device kernels where necessary:

```text
Pascal worker path -> sm_61
RTX3060 fixed path -> sm_86
E7 CPU path        -> AVX1-compatible code
```

Do not assume modern MXFP4 Tensor Core kernels run on GTX1060. Pascal workers require a custom packed-low-bit decode/compute path appropriate to their ISA.

## 11. Cluster network assembly

After two nodes pass local tests:

```text
R920 node FDR NIC -> QSFP DAC -> SX6036
R920 node FDR NIC -> QSFP DAC -> SX6036
...
```

Run an InfiniBand subnet manager somewhere in the fabric. Keep iDRAC/management Ethernet physically/logically separate from the low-latency model fabric.

Target topology:

```text
                 SX6036 FDR56
      +------+------+------+------+
      |      |      |      |      |
    node0  node1  node2   ...   node9
```

## 12. Rack placement

Ten R920s consume about 40U. Two 1U switches bring the rack to 42U.

A 42U rack is therefore a zero-headroom fit. Prefer a 45U/47U rack if economically available.

External GPU frames are separate from the 42U server/switch calculation and require their own stable mounting/shelving/adjacent frame arrangement.

Keep FDR DAC lengths short and similar where possible.

## 13. Facility power and cooling

Planning load:

```text
CPU+GPU component estimate ~13.9kW
full wall planning envelope ~16-20kW
```

This is not a measured load.

Requirements handed to facility electrician/HVAC contractor:

```text
continuous IT load target: up to ~20kW
multiple rack PDUs/circuits
grounding/protection appropriate to local code
startup/inrush considerations
heat rejection approximately equal to electrical IT load
additional HVAC electrical demand beyond IT load
```

Do not energize a full ten-node build from household extension leads or a single ordinary branch circuit.

## 14. Scale-out procedure

Never assemble ten unvalidated identical nodes at once.

```text
NODE 1
  -> pass all local PCIe/NUMA/kernel gates

NODE 2
  -> pass FDR latency + one-request 1->2 scaling gate

NODES 3-4
  -> validate scaling curve and switch behavior

NODES 5-10
  -> only after architecture shows continued single-request benefit
```

The acceptance measurements are specified in `VALIDATION.md`.