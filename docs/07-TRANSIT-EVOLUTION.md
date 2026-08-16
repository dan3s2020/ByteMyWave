# 07 — Transit GPU Design Evolution

This document records how TensorWave evolved into Transit GPU. It intentionally preserves failed ideas, wrong turns and corrections because they define the design constraints as much as the surviving architecture does.

## 1. Original economic constraint

The project did not begin with a preferred architecture. It began with the observation that conventional AI hardware gives a bad price/capacity tradeoff for very large models.

The initial searches explored:

- Chinese surplus/datacenter-pull markets;
- Tesla V100/P40/P100 class hardware;
- broken GPUs and donor boards;
- VRAM replacement/upgrades;
- old enterprise accelerators;
- cheap high-capacity RAM as an alternative to expensive VRAM.

The recurring project constraint became:

> Do not spend money on a merely functional but unproductive conventional rig. Find an architecture that exploits hardware the market undervalues.

This is why solutions that technically run a model but require many mediocre GPUs or many complete servers are not considered satisfactory.

## 2. TensorWave phase: RAM as the persistent model store

The first serious architecture was:

```text
large system RAM
      |
quantized model tiles
      |
pinned host queues / prefetch
      |
PCIe DMA
      |
small VRAM ring/cache
      |
GPU compute
```

Key ideas from this phase remain useful:

- model weights are read-only during inference and do not need to live permanently in VRAM;
- execution order is sufficiently predictable to precompute a schedule;
- weights can be represented as an addressable `Weight Atlas` rather than an opaque checkpoint blob;
- VRAM can be a window/scratchpad rather than the total model store;
- quantization should happen before inference so compressed bytes cross the slow link;
- transfer and compute must overlap;
- the important metric is uncovered transfer / accelerator starvation, not raw copy time by itself;
- hot/warm/cold placement and predictive prefetch can reduce exposed data movement.

These ideas are preserved in documents 00–06.

## 3. Weight Atlas / model-as-map idea

A persistent idea across the project is that the model should be decomposed into named, indexed, independently placeable pieces.

Conceptually:

```text
model
  layer
    attention tensors
    router
    experts
      expert 000
      expert 001
      ...
```

For each tensor/tile/expert shard the atlas can store:

- logical ID;
- original tensor name;
- layer/expert ownership;
- shape;
- dtype/quantization format;
- local storage address;
- tile/controller ID;
- hash/checksum;
- scale metadata;
- dependencies;
- execution order;
- replication policy;
- routing statistics.

The original question was how to know exactly what is missing if a chunk disappears. The same atlas becomes the placement map for a distributed Transit machine.

## 4. Why pure RAM→VRAM streaming stopped being the final answer

RAM backing remains useful, but K3-scale arithmetic makes pure host-to-GPU weight streaming unattractive.

If about 104 billion weights are active per token and they occupy approximately 4 bits each, the active weight traffic is roughly:

```text
104e9 weights × 0.5 byte ≈ 52 GB/token
```

At 100 tokens/s this corresponds to approximately:

```text
5.2 TB/s
```

No PCIe link in the R920 class can carry that as host-to-device traffic every token. Therefore the project pivoted from:

> move the weights efficiently

into:

> do not move the weights out of their local memory in the first place.

That pivot is the core of Transit GPU.

## 5. SSD/NVMe phase

The project then tested whether storage could participate directly in the weight path.

The original Python benchmark:

- located a real Ollama GGUF tensor;
- requantized it to signed INT4;
- packed it as Q4 and as four one-bit planes with exactly the same 4 bits/weight logical storage;
- verified integer equivalence;
- tested RAM and direct SSD paths.

The direct NVMe path later saturated around 6.1 GB/s on the laptop. This proved that ordinary NVMe can be a useful prefetch/backing store, but not the primary active-weight source for a K3 100 tok/s target.

Decision:

- SSD remains useful for checkpoint loading, staging and cold data;
- it is not counted as compute;
- `SSD Q4-equivalent Gweights/s` is transport capacity only and must never be added to CPU compute as if both calculate.

## 6. Bitplane arithmetic discovery

Signed INT4 weights were decomposed into bitplanes:

```text
q = b0 + 2*b1 + 4*b2 - 8*b3
```

Signed INT8 activation:

```text
x = a0 + 2*a1 + 4*a2 + 8*a3 + 16*a4 + 32*a5 + 64*a6 - 128*a7
```

Therefore an exact dot product can be reconstructed from intersections of bitplanes:

```text
dot(q, x) = sum_i sum_j cW[i] * cX[j] * popcount(W_i & X_j)
```

This converts the matrix datapath into:

- AND;
- POPCNT/reduction;
- shifts;
- add/sub.

The proof kernel does not require a general matrix-multiply instruction for this integer representation.

This was not left as theory: the x64 assembly implementation was bit-exact against a software reference.

## 7. V4 masked-activation-sum idea

A second exact formulation was tested:

```text
S_i = sum(x[k] where bit_i(weight[k]) = 1)
dot = S0 + 2*S1 + 4*S2 - 8*S3
```

The idea is attractive for hardware because a weight plane can act like a physical gate/mask over activations.

On the laptop CPU, however, V4 reached only about 0.70× the speed of V3 at the tested 16-worker point.

Decision:

- reject V4 as a CPU optimization;
- retain masked-sum/gating as a possible custom-hardware idea.

## 8. Compute-in-flash / NAND / analog directions

The project explored whether fixed inference weights could live in nonvolatile memory and participate directly in compute.

Directions discussed included:

- commodity NAND;
- NOR/SLC arrays;
- analog crosspoints;
- SuperFlash-style compute;
- Mythic/memBrain-like fixed-weight concepts;
- hand-built matrices of inexpensive memory ICs;
- optical/holographic fixed transformations.

Important correction:

> Commodity NAND devices do not expose the internal bitline currents/analog state required to simply turn a normal NAND package into an analog matrix engine through its standard interface.

A previous cost path that implied tens of thousands of discrete flash devices and hundreds of thousands of dollars was rejected. It violated the project's core economic constraint by turning a proof mechanism into a supposed final architecture.

Decision:

- keep compute-in-memory as research inspiration;
- do not base the near-term machine on unavailable proprietary analog interfaces or huge numbers of small flash chips.

## 9. In-DRAM logic direction

Research such as ComputeDRAM, PULSAR, SoftMC and DRAM Bender showed that commodity DRAM can sometimes be driven with nonstandard command sequences to perform bulk operations or expose internal behavior.

This is highly relevant because fixed bitplane weights could potentially be acted on without leaving the DRAM arrays.

But the project must not overclaim:

- standard DDR3 DIMMs do not natively expose a complete K3 dot-product/popcount engine;
- row-copy/AND/OR primitives are not the same as a full reduction datapath;
- arbitrary timing normally requires a custom/FPGA DRAM controller.

Decision:

- retain as a future acceleration layer;
- first build a reliable local-DDR3 + FPGA compute tile using standard reads;
- only then experiment with in-DRAM operations.

## 10. Why one R920 became interesting

The Dell PowerEdge R920 is attractive not because it is a modern AI server, but because it is cheap enterprise infrastructure with:

- four-socket NUMA;
- very large DDR3 capacity;
- many DIMM sockets via memory risers;
- multiple PCIe expansion slots;
- mature Linux support;
- inexpensive used parts.

The important architectural correction is that physical DIMM/riser paths must not be naively counted as hundreds of independent full-speed CPU memory channels. The R920's internal memory subsystem has its own Xeon E7/SMB/NUMA topology and must be treated according to measured local-socket bandwidth.

The R920 is therefore used as:

- scheduler;
- router;
- model/placement manager;
- reducer;
- boot/staging host;
- PCIe root complex.

It is not expected to provide the final multi-terabyte/s active-weight bandwidth by itself.

## 11. Why 'buy many R920s' was rejected

A roofline exercise can always divide 5.2 TB/s by a server's measured memory bandwidth and produce a large server count.

That is not the Transit design goal.

A solution requiring a rack of 16–25 complete R920-class machines recreates the conventional cost/power problem and ignores the opportunity to keep compute close to cheap memory.

Decision:

> Keep one R920 if possible. Multiply memory-compute paths, not complete servers.

## 12. DDR3 channel aggregation insight

A crucial conceptual correction was:

> Do not electrically combine 300 DDR3 buses into one giant bus. Unite them after local compute.

The scalable structure is a set of memory islands/tiles:

```text
activation / command broadcast
          |
          +--> tile A: local DDR3 -> local compute -> partial result
          +--> tile B: local DDR3 -> local compute -> partial result
          +--> tile C: local DDR3 -> local compute -> partial result
          ...
                         |
                         v
                    reduction
```

Capacity and bandwidth are separate. Putting more DIMMs behind the same channel increases capacity, not independent bandwidth. The desired object is therefore not '300 DIMMs' but roughly '300 independent DDR3 data paths' or enough local memory paths to reach the required effective bandwidth.

## 13. Open-source DDR3 controller search

Open-source controller IP was examined:

- LiteDRAM;
- UberDDR3;
- ultraembedded/core_ddr3_controller;
- research controllers such as SoftMC/DRAM Bender.

The RTL itself can cost $0. The real constraints are:

- FPGA I/O count;
- DDR PHY quality;
- package pin count;
- PCB signal integrity;
- clocks;
- power rails;
- DIMM sockets/routing.

An important correction from this phase: an ECP5-45 in a small TQFP144 package does not have enough I/O pins to magically drive eight independent x64 DDR3 channels. Controller logic can be small while PHY pins dominate.

Decision:

- prefer surplus hardware where difficult DDR3 routing/PHY work is already manufactured;
- if a custom tile is required, use as few sufficiently large-package controllers as practical, not one tiny FPGA per DIMM.

## 14. Memory riser insight

Server memory risers became interesting because they are cheap professionally routed high-speed memory PCBs.

Example examined: IBM POWER7 eight-slot DDR3 risers.

The lesson is nuanced:

- physically, an eight-slot riser is valuable infrastructure;
- it is not a generic passive `1 DIMM slot -> 8 independent channels` adapter;
- enterprise risers often contain memory buffers and proprietary host interfaces;
- without documentation or a compatible controller they are not plug-and-play on an R920.

Decision:

- search aggressively for documented/open/compatible memory risers and memory-controller boards;
- do not assume any random server riser can be wired into an R920 DIMM socket.

## 15. Mining-riser insight

PCIe mining risers were initially dismissed because a physical x16 slot fed by x1 remains electrically x1.

That dismissal missed the Transit use case.

If weights remain local to a tile, the PCIe link does **not** carry the tile's internal DDR bandwidth. It carries only:

- command descriptors;
- activations;
- router metadata;
- reduced outputs/status.

Therefore a cheap powered mining riser can be a perfectly useful **physical extender** for a low-bandwidth-control/high-local-bandwidth endpoint.

Important limitations:

- it does not create PCIe lanes;
- it does not directly connect to a RAM riser;
- the board at its end must be an active PCIe endpoint;
- power quality and link generation/stability must be validated.

## 16. YPCB-00338-1P1 discovery

A particularly useful surplus FPGA card was found:

- Kintex-7 XC7K480T class FPGA;
- PCIe interface;
- two local DDR3 memory channels / banks;
- public reverse-engineering work;
- Vivado board files/examples;
- LiteX board support;
- MIG/PCIe examples in community projects.

This is close to the desired primitive:

```text
PCIe endpoint + programmable compute + local DDR3
```

But it is not the final Transit tile because:

- it offers only two DDR3 channels, not eight;
- the DDR3 is soldered on the card rather than using the project's loose DIMM inventory;
- 4 GB-class local capacity is a laboratory scale, not a K3 placement unit.

Decision:

> Use YPCB as a laboratory bridge from proven CPU bitplane math to a physical PCIe+DDR3+FPGA tile. Do not buy hundreds until the exact board revision, programming path and sustained memory behavior are validated.

## 17. Current target topology

The current architecture is deliberately simple:

```text
1 × Dell R920
      |
PCIe switch/fan-out tree
      |
38 logical Transit tiles
      |
8 independent DDR3 channels per tile
      |
local bitplane/MXFP-compatible compute
```

Arithmetic:

```text
38 × 8 = 304 DDR3 channels
```

The reason for grouping eight channels behind one endpoint is that 38 endpoints are much easier to enumerate, power and schedule than 304 independent PCIe cards.

The exact final eight-channel tile is still the missing hardware piece. That is now a well-defined search/engineering problem rather than an undefined 'DIY GPU'.

## 18. What the project believes now

The surviving Transit principles are:

1. **Weights are stationary whenever possible.**
2. **Memory bandwidth should scale by adding local memory-compute tiles.**
3. **PCIe is a control/activation/result fabric, not the active-weight bus.**
4. **MoE sparsity must be exploited physically:** only selected experts should wake/read/compute.
5. **The R920 orchestrates; it should not serialize the entire weight path.**
6. **Use surplus/open/documented hardware before designing expensive custom boards.**
7. **Prove one tile completely before scaling.**
8. **Measured evidence overrides attractive theory.**
9. **Rejected ideas remain documented so the project does not repeat them.**
10. **The final cost target remains radically below conventional multi-GPU infrastructure.**
