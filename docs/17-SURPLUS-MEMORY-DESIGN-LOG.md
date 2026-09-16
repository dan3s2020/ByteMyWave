# 17 — Surplus Memory Design Log

This document preserves the reasoning path that led from the original TensorWave RAM-streaming idea to the current DDR3-riser/FPGA and whole-server DDR2 alternatives. It exists so later work does not forget why apparently strange hardware was investigated.

## 1. Original constraint

The project is not trying to build a normal expensive inference server.

The search objective became:

> Find hardware whose **market value is far below the value of its already-manufactured memory/control infrastructure**.

Historical working acquisition target for the experimental memory fabric was roughly 1000–3000 RON where possible. This forced the search away from normal GPU/FPGA retail pricing and toward datacenter scrap, obsolete servers, memory risers, mining infrastructure and old research hardware.

## 2. Why the R920 was attractive

The Dell R920 offered:

- very large DDR3 host capacity;
- many DIMM sockets;
- multiple PCIe slots;
- four-socket NUMA;
- cheap used enterprise parts.

The important correction was that 96 DIMM sockets do not equal 96 independent memory channels and cannot simply be multiplied into a multi-TB/s bandwidth figure.

That moved the project toward external local-memory compute.

## 3. The `38 × 8 = 304` riser/tile idea

The project wanted approximately 300 independent cheap DDR3 data paths to approach the ~5.2 TB/s K3 active-weight roofline at ~100 tok/s.

Instead of 304 independent PCIe cards, group memory paths:

```text
38 active endpoints
× 8 DDR3 channels/endpoint
= 304 DDR3 channels
```

Each endpoint would:

- own its DDR3 controllers;
- hold resident expert shards;
- receive only activations/commands;
- compute locally;
- reduce locally;
- return small results.

This is why an 8-slot server memory riser looked so valuable: the expensive high-speed PCB routing already existed.

## 4. Why a random 8-slot riser was not enough

The first tempting mental model was effectively:

```text
one memory connection
-> riser
-> eight useful independent DDR paths
```

That is usually false.

Enterprise risers often contain:

- memory buffers;
- OEM-specific clocking;
- proprietary host-side links;
- firmware/training assumptions.

The IBM POWER7 eight-slot DDR3 riser became the canonical example: physically valuable, but not usable until its buffer/protocol/front-end path is understood.

## 5. Mining-riser correction

PCIe mining risers were first treated as almost useless because an x1 riser remains x1 electrically.

The correction was important:

> Transit does not need the mining riser to carry local DDR bandwidth.

If an active endpoint holds weights and computes locally, the upstream link carries only dynamic traffic.

Therefore cheap powered mining risers are valid **physical endpoint extenders** for prototypes, although they are not memory controllers or lane multipliers.

## 6. FPGA-controller search

The project searched for a controller cheap enough to sit near the memory.

Directions included:

- small Lattice ECP5 parts;
- open DDR3 controller RTL;
- old FPGA boards;
- old emulation systems such as BEE2/BEE3-class hardware.

The key correction was that controller RTL can be free while DDR PHY I/O is not. A small TQFP FPGA cannot physically terminate eight independent x64 DDR3 channels.

This shifted attention toward **already-manufactured large-package FPGA boards**.

## 7. YPCB-00338-1P1 discovery

The surplus YPCB-00338-1P1 card was important because it combines:

```text
Kintex-7 FPGA
+ PCIe endpoint
+ local DDR3
+ public reverse engineering/tooling
```

It is not the final tile because it exposes only two local DDR3 banks/channels and limited capacity, but it is enough to prove the Transit principle on real hardware.

## 8. The `why are we building the memory controller?` pivot

The next conceptual step was to look at the server scrap market and ask:

> If an obsolete complete server is nearly worthless, why reverse engineer its memory board instead of buying the machine that already knows how to drive it?

This created the whole-server DDR2 path.

The target stopped being only `a cheap riser` and became:

```text
cheap complete server
+ many DIMM sockets
+ many NUMA memory controllers
+ CPUs already attached to those controllers
+ Linux
+ PCIe for fast network
```

## 9. Named DDR2 server candidates

Two server families became concrete enough to preserve:

### HP ProLiant DL785 G5/G6

Project working candidate:

```text
8-socket class
64-DIMM-class configuration
DDR2 generation
```

HPE documentation verifies the 8-socket class and DDR2-generation memory options. G6 documentation also confirms memory-speed reductions under heavy DIMM population.

### Sun Fire X4640

Oracle documentation verifies:

```text
8 CPU modules maximum
8 DDR2 DIMM slots per CPU module
64 DIMMs total
512 GB maximum
```

This made it one of the cleanest examples of an obsolete server acting as eight prebuilt NUMA memory islands.

## 10. Example mixed-fleet thought experiment

A concrete discussion example was:

```text
5 × HP DL785 G5/G6
5 × Sun Fire X4640
```

The value of this thought experiment was not brand purity. It demonstrated that Transit software must tolerate heterogeneous nodes and schedule by measured properties.

A 10-server fleet at 512 GB/server class has a raw maximum-capacity order of ~5.12 TB, far above the project's current ~1.56 TB K3 checkpoint working size.

Again, this solves **capacity**, not throughput.

## 11. The 160-slot / 300-slot discussion

The project repeatedly used physical slot count as a rough acquisition shorthand. It must be translated into capacity before it is treated as sufficient.

Raw capacity examples:

| Slots | 4 GB DIMM | 8 GB DIMM | 16 GB DIMM |
|---:|---:|---:|---:|
| 160 | 640 GB | 1.28 TB | 2.56 TB |
| 300 | 1.20 TB | 2.40 TB | 4.80 TB |
| 320 | 1.28 TB | 2.56 TB | 5.12 TB |
| 640 | 2.56 TB | 5.12 TB | 10.24 TB |

For the named DDR2 server candidates, **8 GB/module is the safe high-density planning point from the documented 512 GB / 64-DIMM class**. Therefore 160 slots × 8 GB is only ~1.28 TB and does not by itself cover the current ~1.56 TB K3 checkpoint working size.

This is a useful correction to the casual statement that “160 slots is enough.” It can still become enough if:

- the actual resident representation is smaller than the current checkpoint working size;
- additional host/local storage participates in cold staging rather than active weights;
- a higher-density compatible DIMM exists for the exact platform;
- or the cluster has more than 160 populated slots.

The runtime should compute this from the exact atlas and inventory rather than rely on slot count.

## 12. Why `40 pieces` appeared in the earlier search

Some riser/controller concepts required dozens of repeated boards to reach the desired number of independent memory paths.

The problem was not only unit price. It was **availability in quantity**.

A $10–$30 board is not a solution if only three exist and the architecture needs 40.

The whole-server pivot partially solves this because one complete 64-slot machine replaces many individual passive boards from an acquisition/logistics standpoint, even though it does not create 64 independent channels.

## 13. DDR1 search

DDR1 was considered because the project explicitly searches for commercially useless memory infrastructure.

No DDR1 platform was promoted because:

- density is too low;
- bandwidth is much lower;
- CPUs are even weaker;
- server count/power increases rapidly.

Keep DDR1 only as an extreme scrap-price search branch.

## 14. How multiple DDR2 servers are connected

The correct architecture is **not distributed shared memory pretending to be one giant NUMA machine** for the first implementation.

Use explicit message passing:

```text
head/router
  -> activation + expert command
server owning expert
  -> local resident-weight compute
  -> reduced result
head/reducer
```

Start with persistent TCP over fast Ethernet.

Move to RDMA/InfiniBand only after measurement.

This makes failure handling, topology and ownership explicit and prevents accidental remote-weight traffic.

## 15. Where K3 actually runs

Logically K3 runs as one distributed runtime.

Physically:

```text
head node
  tokenizer
  generation loop
  scheduler
  router
  attention/shared path initially
  KV/state initially
  reductions

DDR2 server nodes
  routed expert shards
  local CPU kernels

DDR3/FPGA tiles when available
  fast/hot expert shards
  local FPGA kernels
```

The exact partition moves only after profiling.

## 16. How an agent framework uses it

The user should never have to manually log into 10 servers to chat with the model.

The head exposes a single OpenAI-compatible API.

```text
agent framework
  -> Transit base_url
    -> tokenizer/generation loop
      -> distributed scheduler
        -> server/tile workers
```

From the agent's point of view it is one model endpoint.

## 17. Can K3 be split across 10–20 servers?

The model bytes and operations can be partitioned. The project does not treat `can be partitioned` as equivalent to `will be fast`.

Three separate proof obligations exist:

```text
1. capacity proof
2. numerical correctness proof
3. throughput/latency proof
```

The named 64-DIMM-class servers make capacity plausible with a modest node count. The unresolved proof is old-CPU compute throughput per resident weight byte.

## 18. Performance expectation that survives scrutiny

Current K3 working roofline:

```text
~52 GB active 4-bit-equivalent weights/token
```

Illustrative fully populated 8-socket DDR2-533 server roofline used in document 14:

```text
~68.3 GB/s theoretical aggregate
~1.31 ideal weight-path tok/s/server
```

Therefore:

```text
10 servers -> ~13.1 ideal weight-path tok/s
20 servers -> ~26.3 ideal weight-path tok/s
```

A 60–75% memory-efficiency planning range gives roughly:

```text
10 servers -> ~7.9–9.8 weight-path tok/s
20 servers -> ~15.8–19.7 weight-path tok/s
```

But these numbers survive only if CPU kernel throughput matches memory throughput. The first server benchmark can reduce them dramatically.

## 19. Why 100 tok/s remains a DDR3/custom-tile target

To reach the current 100 tok/s active-weight roofline requires approximately 5.2 TB/s of useful weight-path bandwidth.

A small number of DDR2 servers cannot provide that.

The custom 304-channel DDR3 architecture was created precisely because its ideal aggregate bus bandwidth can approach that number.

Therefore the two branches have different expected roles:

```text
DDR2 whole servers
  -> fastest route to cheap capacity + distributed correctness
  -> possible single/low-double-digit tok/s class only if CPU results are strong

DDR3/FPGA tiles
  -> harder hardware problem
  -> current path intended to chase the ~100 tok/s class roofline
```

## 20. What we must find next

### If pursuing whole DDR2 servers

Find **one**, not ten, of:

```text
HP DL785 G5/G6 complete 8-socket configuration
Sun Fire X4640 complete 8-module configuration
or another 48–64+ DIMM complete obsolete server
```

Then benchmark it.

Also find:

```text
matching cheap 8 GB DDR2 ECC DIMMs
10 GbE or faster Linux-supported NIC
cheap switch/DAC fabric for multi-node stage
```

### If pursuing the riser/FPGA path

Find one of:

```text
documented 4–8-channel DDR3 FPGA/accelerator board
server memory riser with known buffer protocol/pinout
obsolete FPGA appliance with multiple DDR channels and usable host interface
```

### If pursuing hybrid

Acquire one server node and one YPCB-class FPGA node and make the same Transit command protocol run on both.

## 21. The next irreversible purchase rule

Before any quantity purchase, demand a one-unit measurement proving the exact claimed role.

For a server:

```text
GB/s
Gweights/s
W
network
correctness
```

For a riser/FPGA endpoint:

```text
programmable
DDR stable
PCIe stable
local compute correct
GB/s/Gweights/s measured
```

Only then multiply quantity.

## 22. Design principle preserved across every pivot

The project has changed hardware candidates many times, but one principle survived every correction:

> **Do not move the large static weights through the slowest shared fabric every token. Move the small dynamic activation to the memory, compute where the weights live, reduce, and return the small result.**

Everything else — R920, DDR3 tiles, risers, FPGA boards, DDR2 servers, networking — is replaceable infrastructure around that principle.