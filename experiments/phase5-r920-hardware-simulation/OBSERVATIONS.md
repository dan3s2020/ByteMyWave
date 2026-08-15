# Phase 5 — Observations, implications and proposed improvements

Date: **2026-08-15**

This note records what became visible only after mapping the existing TensorWave Phase-3/Phase-4 design onto the R920 + RTX 3060 profile.

## 1. Capacity and bandwidth separate cleanly

A 1 TiB R920 can hold the compressed representation of the generic model sizes currently used by the project with enormous capacity margin. That makes host capacity a weak constraint for the first experiments.

The simulation simultaneously shows that dense `M=1` remains dominated by repeated H2D traffic. This is useful because it prevents us from confusing “the model fits in RAM” with “the model is interactive.”

### Improvement

Keep capacity and feed-rate metrics separate in every future report:

```text
host model capacity
active bytes/step
resident bytes
physical H2D bytes/step
unhidden transfer
```

---

## 2. R920 NUMA topology should become part of TensorWave metadata

On a four-socket machine, “host RAM” is not one uniform bucket from the point of view of a GPU attached to a CPU root complex.

The current Weight Atlas is concerned with what tensor/tile is needed and where it lives in the packed representation. The hardware mapping suggests adding placement metadata later:

```text
preferred_host_numa
preferred_gpu
preferred_pcie_root
replica_or_shard_id
measured_local_h2d_gbps
measured_remote_h2d_gbps
```

### Improvement

Add a topology discovery/calibration stage before runtime plan generation. The scheduler should be able to reject or warn on a plan that feeds a GPU predominantly from remote NUMA memory.

---

## 3. The 12 GiB RTX 3060 changes the optimization priority

For the representative Phase-3 tile geometry, the transient ring is only tens of MiB even at large M.

That means most of the RTX 3060's 12 GiB is not required merely to implement the two-slot streaming proof.

### Improvement

Use spare VRAM for a **persistent compressed cache**, not simply larger transient slots.

The first policy does not need to be sophisticated. It can begin with static residency selected from the Weight Atlas, then evolve to measured hotness/expert reuse.

Required metrics:

```text
resident compressed bytes
cache hit bytes
cache miss bytes
bytes avoided per step
evictions
starvation delta
wall-time delta
```

The reference sensitivity calculation shows why this is worth doing: 8 GiB of persistent compressed residency moves the ideal 70B crossover from roughly M=260 to M=209 under the same timing assumptions.

---

## 4. Current code is naturally multi-worker, not naturally model-parallel

The current proof has one device, one copy stream, one compute stream and one pair of compressed slots. The least invasive multi-GPU extension is therefore to replicate that state per GPU.

### Improvement

Implement a worker abstraction:

```text
GPUWorker
  device_id
  numa_node
  copy_stream
  compute_stream
  q4_slots[2]
  dequant_workspace
  persistent_cache
  local_host_queue
  metrics
```

Then dispatch independent requests/batches to workers.

This gives real multi-GPU value without prematurely adding cross-GPU synchronization.

---

## 5. If we later shard one model, MoE/expert ownership may suit R920 better than fine tensor parallelism

The R920 has multiple CPU/root complexes and RTX 3060 has no NVLink. Fine-grained tensor parallelism can introduce frequent cross-device communication that competes with the exact PCIe resource TensorWave is already trying to optimize.

### Improvement

Evaluate distributed strategies in this order:

```text
1. expert ownership / MoE routing
2. pipeline or layer-range ownership
3. fine-grained tensor parallelism
```

This is a hypothesis to measure, not a fixed architectural law.

---

## 6. GPU count must be represented by two different numbers

The R920 can expose multiple x16 electrical links, but a normal RTX 3060 is a long dual-slot powered card. Electrical link count and stock-chassis GPU capacity are different constraints.

### Improvement

Hardware profiles should expose both:

```text
electrical_gpu_links
validated_physical_gpu_positions
```

Never infer the second from the first.

---

## 7. The simulator needs calibration, not more invented precision

The most important simulator inputs are currently assumptions:

```text
H2D GB/s
effective GEMM TFLOP/s
dequant time
```

Adding more analytical detail before measuring these on the target hardware risks false precision.

### Improvement

The next code step should be an importer that consumes Phase-4 calibration JSON and generates the R920 simulation automatically with measured values.

Desired flow:

```text
Phase-4 measured run
 -> feasibility calibration JSON
 -> hardware/topology calibration JSON
 -> R920 simulator
 -> predicted vs measured comparison
```

---

## 8. Add local-vs-remote NUMA as an explicit experiment axis

Phase 4 currently varies M, tile geometry, wire format, residency and active fraction. On R920 there is another critical axis:

```text
host placement = local | remote-1-hop | remote-other
```

### Improvement

Add a NUMA H2D benchmark that keeps the GPU and CUDA path fixed while changing only the host allocation node. Record bandwidth and latency distributions, not only averages.

---

## 9. The first purchase/bring-up gate is one GPU, not multiple GPUs

Buying multiple RTX 3060 cards before verifying fit/power/cooling on the actual R920 would mix software feasibility with an avoidable mechanical risk.

### Improvement

Bring-up order:

```text
R920 + balanced RAM
-> one RTX 3060
-> verify PCIe/thermals/power
-> local/remote NUMA calibration
-> Phase-3/4 hardware run
-> persistent-cache experiment
-> second GPU on another root
-> concurrent independent workers
-> only then distributed model execution
```

---

## 10. Strongest current interpretation

The R920 is not valuable because it makes PCIe fast. It is valuable because it gives TensorWave a cheap laboratory in which the variables we care about are physically separated:

```text
huge persistent host store
four memory domains
multiple CPU PCIe roots
small GPU working windows
```

That makes failures informative. If the project fails on this platform, the measurements can tell us whether the reason is host-memory placement, PCIe, GPU compute, cache residency, scheduling, or chassis integration rather than merely “not enough RAM.”
