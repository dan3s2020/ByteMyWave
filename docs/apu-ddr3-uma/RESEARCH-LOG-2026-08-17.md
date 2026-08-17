# Research Log — 2026-08-17 — Carrizo APU + DDR3 UMA

## Purpose

Preserve how the APU direction emerged, which earlier assumptions were corrected, which paths remain valid alternatives, and exactly why the current status is **conditional GO for a small POC rather than fleet procurement**.

This log records architecture evolution, not a claim that later ideas invalidate all earlier tracks.

---

## 1. Starting problem — CPU memory channels

The investigation began from the desire to expose hundreds of independent cheap DDR3 channels to Kimi K3.

A conventional server CPU can expose only a fixed number of physical memory channels. Adding DIMM slots, risers or more DIMMs on the same channel increases capacity but does not create independent full-bandwidth channels.

This led to the initial question:

> Can the CPU do much less work and let a GPU perform the arithmetic while DDR3 provides the capacity/bandwidth?

That division of labor is possible in principle, but with a **discrete GPU** the weight path remains constrained by the connection between CPU/system memory and GPU.

---

## 2. FPGA/local-compute track considered

PR #10 already develops the clean local-memory solution:

```text
DDR3 -> local controller/compute -> reduced result -> host
```

with FPGA-class logic as the first programmable implementation candidate.

This avoids moving active weights through host PCIe every token.

The user explicitly asked to investigate a path **without FPGA**, with the CPU doing mainly command/control work.

Therefore this log does not reject the FPGA tile. It records a second route that may use mass-produced APU silicon as both memory controller and local GPU compute engine.

---

## 3. Discrete GPU + lightly loaded CPU

We separated CPU **cores** from the CPU **memory controller**.

The CPU cores can be mostly idle while the integrated memory controller still services DDR3. However, with a discrete GPU the data path remains approximately:

```text
DDR3 -> CPU memory controller -> PCIe -> GPU
```

Increasing the number of host DDR channels above the GPU's incoming PCIe bandwidth does not translate directly into useful GPU weight bandwidth.

For an old PCIe 3.0-class platform, changing from one x16 GPU to multiple smaller-width links redistributes the same limited root-lane budget; it does not create a multi-terabyte/s weight path.

Conclusion:

> Reducing CPU arithmetic is not enough. To preserve cheap DDR bandwidth, compute must be closer to the memory controller than a discrete PCIe GPU.

---

## 4. Carrizo APU route discovered

AMD Carrizo combines:

```text
CPU cores
integrated Radeon GPU
DDR3 memory controller
shared/hUMA memory architecture
```

The A8-8600P reference part exposes two DDR3 channels and a Radeon R6 iGPU at a 15 W default TDP.

This creates the alternative tile:

```text
DDR3 ch A ----\
               Carrizo shared memory -> Radeon R6 compute
DDR3 ch B ----/                 |
                                CPU control/runtime/network
```

The key improvement over discrete GPU is architectural: the normal weight path no longer needs to be sized as `DDR3 -> discrete GPU PCIe link`.

This was enough evidence to justify deeper research.

---

## 5. Initial 320-channel thought experiment

With two channels/APU:

```text
160 APUs * 2 channels = 320 channels
```

At DDR3-1600 nominal payload:

```text
320 * 12.8 GB/s = 4096 GB/s = 4.096 TB/s
```

The first simplistic calculation reused the older Transit lower bound:

```text
104B active params * 0.5 byte ~= 52 GB/token
4.096 TB/s / 52 GB ~= 78.8 weight-path tok/s
```

This number was **overinterpreted in conversation** and required two corrections.

---

## 6. Correction #1 — MoE placement prevents automatic bandwidth summation

K3 has 896 routed experts but selects 16 per token in each MoE layer.

If whole experts are simply assigned to separate boards:

```text
selected expert owners -> work
most other boards       -> idle
```

Therefore:

```text
320 installed channels
!=
320 channels automatically active for one token
```

To use a much larger fraction of the fleet for a single autoregressive sequence, active expert/non-expert matrices must be tensor-sharded across multiple APUs.

That introduces repeated gather/reduce/collective operations across the 93 serial layer boundaries.

New central risk:

> **distributed synchronization latency can erase the benefit of aggregate local DDR bandwidth.**

This is why the network must be benchmarked at 7/14/28 KiB-class payloads and K3-like layer cadence, not only with a bulk throughput test.

---

## 7. Correction #2 — 52 GB/token is not the exact released K3 storage path

The earlier 52 GB number assumes every activated parameter is exactly four bits:

```text
104B * 0.5 byte = 52 GB/token
```

The official K3 config does not support that interpretation as an exact physical byte count. It declares MXFP4 packed quantization for target linear modules but also has an ignore list for component families.

PR #11 had already correctly documented:

- 52 GB/token as an optimistic absolute lower bound;
- ~58 GB/token as a checkpoint-average screening model;
- exact tensor accounting as a required future measurement/tool.

During this APU investigation a more conservative mixed-precision envelope was derived:

```text
routed experts active/token ~= 48.62B weights
MXFP4 + 1-byte scale/32 ~= 0.53125 byte/weight
routed bytes ~= 25.83 GB

remaining active ~= 55.38B params
if pessimistically BF16 -> ~=110.76 GB

combined conservative envelope ~=136.6 GB/token
```

A conversational statement initially called roughly 137 GB/token “the correct value.” That was too strong.

**Correction recorded here:** ~136.6 GB/token is a conservative envelope, not an exact count, because additional non-expert `Linear` tensors may also be quantized by the released configuration.

Therefore the branch now requires **Gate 0: exact checkpoint + forward-path byte inventory** before a fleet budget.

---

## 8. Revised DDR rooflines

For 320 DDR3-1600 channels:

```text
nominal BW = 4.096 TB/s
```

Weight-path ceilings under the three current models:

```text
52 GB/token lower bound        -> ~78.8 tok/s
58 GB/token screening          -> ~70.6 tok/s
136.6 GB/token conservative    -> ~30.0 tok/s
```

For DDR3-2133 nominal 5.46 TB/s:

```text
52 GB/token lower bound        -> ~105 tok/s
58 GB/token screening          -> ~94 tok/s
136.6 GB/token conservative    -> ~40 tok/s
```

None of these are end-to-end predictions. They ignore imperfect memory utilization, old-iGPU compute limitations, distributed collectives, KDA/state work, software and stragglers.

---

## 9. Software-stack audit

The Carrizo GPU is old, but it is not invisible to current Linux graphics tooling:

- LLVM still identifies Carrizo/A8-8600P under AMDGPU target `gfx801`;
- Mesa RADV provides the practical Vulkan compute path for GFX8-class hardware;
- modern supported ROCm matrices do not make Carrizo a normal supported compute target.

Conclusion:

> Do not buy Carrizo expecting modern ROCm/vLLM to install and run K3. The baseline POC assumes custom Vulkan/LLVM-oriented kernels/runtime for the critical path.

This is a major engineering cost, but not a physical contradiction.

---

## 10. Compute risk on Radeon R6

K3 deployment literature shows that small-batch expert GEMV can be memory-bandwidth dominated on modern accelerators. That helps the architectural case for cheap bandwidth.

However Radeon R6/GFX8 has none of the modern AI matrix/tensor hardware expected by current inference stacks.

Therefore the following must be measured independently:

```text
pure GPU-visible DDR read
MXFP4 packed read + dequant + GEMV
high-precision/non-routed linear work
KDA recurrent-state update
shared expert path
normalization/residual
kernel dispatch overhead
```

If MXFP4-shaped useful throughput is far below the pure DDR read rate, cheap DDR parallelism is not enough.

---

## 11. Network risk quantified

Useful vector scales from K3 dimensions:

```text
3584 BF16 values ~= 7 KiB
7168 BF16 values ~= 14 KiB
```

At 1 Gb/s, 14,336 bytes alone require roughly 115 microseconds of ideal wire time before protocol/software/switch/collective overhead.

For a target of 20 tok/s:

```text
50 ms/token / 93 layers ~= 0.538 ms/layer
```

That makes per-layer latency a much stricter design constraint than merely saying “activation traffic is small compared with weight traffic.”

Conclusion:

> NIC and switch selection must be based on measured small-message collectives, not nominal Ethernet Gb/s alone.

---

## 12. Prior art found

Three lines of prior art support mechanisms used by this track without proving the full machine:

1. shared-memory/HSA zero-copy GPGPU research — supports avoiding the classic discrete PCIe-copy path;
2. MoEShard — supports tensor sharding experts across devices to address load balance;
3. edge GPU + near-data-processing MoE research — supports tensor-parallel active expert work close to memory and identifies scheduling/communication as central.

No publication found demonstrates **Kimi K3 on a 160-node Carrizo DDR3 cluster**. That exact system is therefore an engineering experiment, not an already reproduced design.

---

## 13. Concrete hardware leads

Two POC classes emerged:

### Dell Inspiron 3656 / 0W6FD / A8-8600P board

Advantages:

- cheap bare motherboard leads exist;
- two DDR3L slots;
- Carrizo APU with integrated Radeon.

Risks:

- proprietary OEM power/front-panel dependencies;
- actual memory frequency/capacity;
- NIC/PCIe expansion;
- listing quantity/price volatility.

### HP EliteDesk 705 G2 Mini / A8-8600B

Advantages:

- complete, easy-to-power/cool POC box;
- closely related Carrizo Pro APU;
- two memory sockets in the mini-PC class.

Risks:

- RAM capacity and expansion can be limiting for fleet use;
- complete-box cost higher than bare-board lots.

Current stance:

> choose whatever gets Gate 1/2 measured fastest and cheapest; do not lock fleet hardware before 8-node scaling.

---

## 14. Current architecture status

### Supported enough to prototype

```text
Carrizo shared-memory concept     YES
2 DDR3 channels/APU              YES
Linux Vulkan path                YES, as a POC route
K3 memory-bound expert premise   supported as a relevant regime
expert tensor sharding mechanism supported in prior art
```

### Not yet demonstrated

```text
Radeon R6 K3-format kernel       NO
sustained dual-channel useful BW NO
KDA performance on Carrizo       NO
2-node K3-like collective        NO
8-node scaling                   NO
complete K3 layer                NO
full K3 end-to-end tok/s         NO
160-node economics               NO
```

---

## 15. Decision

Current project decision for this track:

```text
1-4 node proof-of-concept: CONDITIONAL GO / justified
160-node purchase:         NO-GO until gates pass
```

The next decisive information must come from physical measurements.

The exact ladder is frozen in `POC-VALIDATION-AND-PROCUREMENT-GATES.md` so future discussion does not silently move the success criteria after seeing benchmark results.