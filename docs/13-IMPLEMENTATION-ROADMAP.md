# 13 — Implementation Roadmap

This document turns the current Transit GPU architecture into an ordered engineering plan. The rule is simple: **each phase must produce a measurable artifact before the next phase scales it.**

## 0. Freeze the project vocabulary

Use these terms consistently:

- **R920 host** — scheduler/router/reducer/staging machine.
- **Transit tile** — one host-facing endpoint with local memory and local compute.
- **channel** — one independent DDR3 data path owned by a tile controller; not merely one DIMM socket.
- **Weight Atlas** — mapping from model tensor/expert shards to tile/channel/local addresses.
- **resident weights** — weights loaded once into tile-local DDR3 and reused across tokens.
- **activation packet** — dynamic input data sent to a tile for one operation.
- **result packet** — reduced output returned by a tile.
- **weight-path tok/s equivalent** — bandwidth/weight processing roofline only.
- **end-to-end tok/s** — measured complete-model generation rate only.

## 1. Repository structure for implementation

Target structure:

```text
TensorWave/
  README.md
  docs/
  benchmarks/
    transit_ollama_ssd_test.py
    transit_bitplane_kernel.asm
    transit_asm_runner.py
    run_transit_asm.ps1
    transit_asm_v2.py
    transit_ddr5_bench_v3_lowram.py
    transit_maskedsum_v4_1_fixed.py
    transit_final_host_overlap.py
  host/
    atlas.py
    protocol.py
    device.py
    scheduler.py
    loader.py
    router.py
    reduce.py
    cli.py
  rtl/
    pcie_endpoint/
    ddr/
    bitplane/
    command/
    counters/
  firmware/
  tools/
    k3_atlas_builder.py
    quantize_transit.py
    verify_tile.py
  tests/
    reference_dot.py
    reference_expert.py
    protocol_vectors/
  hardware/
    ypcb_00338_1p1/
    final_8ch_tile/
```

The exact host/RTL code will grow only after the lab board is available. The already-demonstrated host benchmark files are preserved now so the evidence is reproducible.

## 2. Phase A — reproduce the existing host proof from the repository

Goal: make sure the historical evidence can be rerun after checkout.

Tasks:

1. run `transit_ollama_ssd_test.py` against a local Ollama GGUF;
2. build/run `transit_bitplane_kernel.asm` with `transit_asm_runner.py`;
3. verify `max_abs_diff == 0`;
4. rerun V2 and DDR5 V3 if needed on a known host;
5. record machine metadata with every benchmark.

Do **not** use this phase to invent V5/V6 CPU kernels unless a specific hardware-design question requires it.

Exit condition:

```text
bitplane signed INT4×INT8 reference == assembly result exactly
```

## 3. Phase B — acquire one YPCB-00338-1P1 lab card

Goal: establish a real PCIe + FPGA + local DDR3 development target.

Before purchase verify:

- exact silkscreen/part revision;
- FPGA part marking;
- board photos match the documented YPCB variant;
- JTAG/programming access;
- power requirements;
- seller return/test status where possible.

Do not bulk buy similar-looking YZCA/other revisions unless the same bitstream/pinout is proven.

## 4. Phase C — bring up the board without Transit compute

Goal: prove the boring infrastructure first.

### C1. FPGA programming

- install compatible Vivado/toolchain;
- build an existing known-good blink/system-test project;
- program via JTAG;
- confirm clocks/reset.

### C2. DDR3

- instantiate the known MIG/LiteDRAM configuration;
- run a memory test across both local DDR3 banks;
- verify long sequential reads/writes;
- record sustained bandwidth;
- record error count over a long soak.

### C3. PCIe

- enumerate board in the R920/lab PC;
- confirm BAR allocation;
- run host-to-card and card-to-host DMA;
- measure x1/x4/x8 link cases if available;
- validate operation through a powered mining riser separately from direct-slot operation.

Exit condition:

```text
known bitstream
+ stable local DDR3
+ stable PCIe DMA
```

## 5. Phase D — port the exact bitplane proof kernel to FPGA

Start with the proven **signed INT4×INT8** format, not K3 MXFP immediately.

Why:

- a reference implementation already exists;
- exact equality is possible;
- it isolates hardware correctness from K3 quantization complexity.

### D1. Data format

Weights in local DDR3:

```text
W0 bitplane
W1 bitplane
W2 bitplane
W3 bitplane
```

Activation buffer:

```text
signed INT8 values
```

The FPGA may transpose activations into 8 bitplanes after DMA, or the host may send pretransposed activation planes for the first prototype. Measure both later.

### D2. Minimum kernel

Process fixed-size blocks, e.g. 64 or 256 elements:

```text
for each weight plane i:
  for each activation plane j:
      count = popcount(W_i & A_j)
      acc += CW[i] * CA[j] * count
```

Implement several lanes in parallel.

### D3. Correctness ladder

1. 64-element synthetic vector;
2. one real tensor row from the existing GGUF experiment;
3. 640×2048 tensor slice;
4. repeated DRAM-scale stream.

Acceptance:

```text
max_abs_diff == 0
```

### D4. Required hardware counters

Add counters before optimizing:

```text
cycles_total
cycles_compute_active
cycles_ddr_stall
cycles_activation_stall
cycles_result_stall
ddr_bytes_read
pcie_bytes_rx
pcie_bytes_tx
weight_elements_processed
commands_completed
errors
```

Exit condition:

> One physical card reads resident local DDR3 weights, receives an activation through PCIe, computes the exact result locally, and returns only the result.

That is the first true Transit proof.

## 6. Phase E — prove why PCIe can be narrow

Run the same command with weights already resident.

Measure separately:

```text
activation bytes in
result bytes out
local DDR bytes read
compute time
PCIe time
```

The desired observation is:

```text
local DDR traffic >> PCIe traffic
```

This validates the architectural use of cheap fan-out/mining-riser links.

If PCIe unexpectedly dominates, fix the protocol/reduction granularity before scaling.

## 7. Phase F — build the Transit host protocol

The first software stack can be Python/C++ user space plus a simple DMA driver/backend. Do not begin with a large framework.

### F1. Device discovery

For every endpoint record:

```text
device_id
PCIe BDF
firmware version
DDR capacity per bank/channel
clock rates
capabilities
health
```

### F2. Command queue

Define a fixed binary descriptor, ideally 64 bytes or another cache-friendly power of two.

Candidate fields:

```text
magic/version
opcode
flags
sequence_id
layer_id
expert_id
shard_id
activation_offset
activation_bytes
result_offset
result_bytes
format_id
scale_id
```

### F3. Completion queue

Return:

```text
sequence_id
status
cycles
ddr_bytes
weight_elements
result_bytes
error flags
```

### F4. Buffer ownership

Keep persistent DMA buffers and ring slots. Avoid allocate/free on the token hot path.

## 8. Phase G — Weight Atlas v2 for K3

Build an actual parser, not a manually edited spreadsheet.

Output one machine-readable atlas record per tensor/shard:

```json
{
  "tensor": "...",
  "layer": 0,
  "expert": 0,
  "shape": [0, 0],
  "format": "mxfp4",
  "bytes": 0,
  "scale_layout": "...",
  "tile": null,
  "channel": null,
  "address": null,
  "sha256": "..."
}
```

The builder should calculate:

- total checkpoint bytes;
- bytes by layer;
- bytes by expert;
- non-expert bytes;
- scale/metadata bytes;
- candidate placement given tile capacities.

Exit condition:

> Every K3 weight used by inference has an exact atlas record and can be located without runtime searching.

## 9. Phase H — solve MXFP4/MXFP8 semantics

This is the numerical bridge between the proof kernel and K3.

Do both implementations in software first.

### H1. Native decode reference

Write a reference function that exactly decodes the K3 stored representation, including block scales and special-value behavior relevant to the checkpoint.

### H2. Transit internal format experiment

Optionally quantize one expert to the proven signed INT4/INT8-style internal format and compare output/error.

Measure:

- tensor-level error;
- expert-output error;
- layer-level error;
- eventually model quality/perplexity/tasks.

Decision rule:

- if native MXFP maps cleanly to FPGA, implement it;
- if Transit INT4 gives acceptable quality and dramatically simpler/faster hardware, keep it as an explicit alternate checkpoint format;
- never conflate the two.

## 10. Phase I — one real K3 expert on one tile

Select one routed expert and make it resident on a tile.

Flow:

```text
K3 checkpoint
  -> atlas
  -> exact expert bytes/scales
  -> tile local DDR
  -> activation from reference runtime
  -> local expert compute
  -> output
  -> compare to software reference
```

If one physical YPCB card lacks capacity/bandwidth to contain a complete expert path, use it for a representative shard but preserve the final command semantics.

Exit condition:

> A real K3 expert or expert shard executes through the same PCIe/local-memory path intended for the final machine.

## 11. Phase J — prototype multiple endpoints

Scale from 1 to 2, then 4, then 8 boards/endpoints.

Test:

- simultaneous command dispatch;
- independent local DDR traffic;
- PCIe switch contention;
- R920 NUMA affinity;
- completion ordering;
- failure/retry;
- fan-out riser stability;
- aggregate activation/result traffic.

Do not jump from one board to 38 without this ladder.

## 12. Phase K — final eight-channel tile decision

By this point measured data will tell us what the final tile actually needs.

Required spec should include:

```text
channels              8 target
sustained DDR GB/s     derived from prototype
compute Gweights/s     enough not to starve DDR
PCIe uplink            sized from measured dynamic traffic
capacity/channel       based on expert placement
FPGA/LUT/popcount      based on proven engine
DSP requirement        based on MXFP scaling path
power                  measured/projected
cost                   must preserve Transit economics
```

Then choose among:

1. documented surplus 8-channel accelerator/memory board;
2. multiple cheap DDR-controller modules behind one endpoint;
3. custom FPGA board;
4. server-memory riser + programmable front end if protocol is documented;
5. another surplus architecture discovered by the ongoing hardware search.

## 13. Phase L — 38-tile fabric

Only after the tile is final:

```text
38 tiles × 8 channels = 304 channels
```

Infrastructure tasks:

- PCIe switch tree/backplane;
- power distribution;
- cooling;
- physical mounting;
- reset/JTAG/service access;
- R920 BIOS/Linux enumeration tuning;
- stable device naming;
- topology-aware scheduler;
- model placement.

## 14. Phase M — K3 execution partition

Start with routed experts on Transit tiles.

Possible initial partition:

```text
R920 CPU / optional GPU:
  embeddings
  router
  attention/shared operations
  KV/state
  global reductions

Transit tiles:
  routed expert weight-heavy operations
```

Then profile. Move additional operations to tiles only when data proves it removes a bottleneck.

## 15. Phase N — first full token

The first full K3 token is more important than a simulated 100 tok/s claim.

Record:

```text
end-to-end token latency
layer-by-layer timing
router time
tile DDR utilization
tile compute utilization
PCIe traffic
host NUMA traffic
reduction time
KV/attention traffic
errors/retries
```

This becomes the real optimization baseline.

## 16. Phase O — optimize toward 100 tok/s

Only after end-to-end profiling decide which knobs matter:

- expert placement;
- replication of hot experts;
- more/larger DDR channels;
- tile popcount parallelism;
- MXFP decode/scaling pipelines;
- batching;
- activation multicast/tree distribution;
- tile-local expert aggregation;
- PCIe switch topology;
- host NUMA placement;
- future in-DRAM operations.

## 17. What not to buy yet

Until the first YPCB/physical tile works:

- do not buy 38 final tiles;
- do not buy hundreds of mystery server memory risers;
- do not buy high-density NAND for compute;
- do not design a huge custom PCB;
- do not buy many R920s;
- do not assume a similar-looking FPGA board revision is compatible.

## 18. Definition of 'plug and play' for Transit

A tile is genuinely plug-and-play only when:

```text
1. power connected
2. PCIe enumerates automatically
3. stable firmware loads from flash
4. host driver/runtime recognizes device
5. local DDR trains automatically
6. health check passes
7. atlas assigns resident shards
8. inference commands run without JTAG/manual intervention
```

The YPCB candidate is close to a **hardware development platform**, not yet this final state.

## 19. Immediate next actions

In order:

1. preserve all existing benchmark sources in this branch;
2. buy/obtain one exact YPCB-00338-1P1 or an even better documented candidate;
3. reproduce public PCIe + DDR3 board bring-up;
4. port the exact signed INT4×INT8 kernel;
5. create the K3 atlas parser;
6. implement the real numerical-format bridge;
7. execute one K3 expert/shard on the physical tile.

At that point Transit stops being an architecture proposal and becomes a hardware inference system.
