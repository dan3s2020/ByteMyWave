# 06 — Collaboration Model for 5 Concurrent Contributors

The repository must be understandable when five people are working on different parts at the same time.

The project therefore needs explicit ownership, current-state tracking, experiment records and rules that prevent two contributors from silently changing the same assumptions.

## Shared source of truth

The following files define current project meaning:

- `README.md` — one-page orientation;
- `docs/00-PROJECT-INTENT.md` — why the project exists and what must remain true;
- `docs/01-MODEL-MEMORY.md` — shared terminology for model/working memory;
- `docs/02-WEIGHT-ATLAS.md` — data addressing model;
- `docs/03-STREAMING-RUNTIME.md` — execution model;
- `docs/04-COMPRESSION.md` — representation/quantization strategy;
- `docs/05-OPEN-QUESTIONS.md` — hypotheses not yet demonstrated;
- `docs/TRANSCRIPT.md` — original discussion/context.

No contributor should infer a new project goal from only one implementation file.

## Proposed 5 workstreams

These are organizational lanes, not permanent job titles.

### Contributor A — Model / graph analysis

Owns:

- H3 graph inspection;
- tensor inventory;
- shapes and dependencies;
- identifying legal tile boundaries;
- generation of Weight Atlas metadata.

### Contributor B — Host memory / transfer engine

Owns:

- RAM layout;
- pinned memory;
- H2D throughput tests;
- DMA queue;
- double/triple buffering;
- transfer/compute overlap measurements.

### Contributor C — GPU execution / kernels

Owns:

- tile kernels;
- quantized GEMM integration;
- GPU-side dequantization;
- VRAM fixed-slot allocator;
- events and synchronization;
- GPU utilization/starvation instrumentation.

### Contributor D — Compression / representation

Owns:

- checkpoint conversion;
- Q8/Q6/Q4/Q3/Q2 experiments;
- mixed quantization policies;
- shared-basis/codebook experiments later;
- correctness/quality comparisons.

### Contributor E — Integration / benchmark / reproducibility

Owns:

- end-to-end harness;
- deterministic test cases;
- metrics schema;
- regression checks;
- environment setup documentation;
- integration across workstreams.

## Every modification must answer four questions

Any non-trivial change should make it possible for the other four contributors to determine:

1. **What changed?**
2. **Why did it change?**
3. **Which assumption or metric does it affect?**
4. **How was it verified?**

## Branch discipline

Use narrowly scoped branches, for example:

```text
model/h3-atlas-v1
transfer/pinned-ring
runtime/fixed-vram-slots
quant/q4-baseline
bench/h2d-overlap
```

Avoid one long-lived branch containing unrelated experiments.

## Experiment records

When implementation starts, each benchmark should create a record under a future structure such as:

```text
experiments/
  YYYY-MM-DD-short-name/
    README.md
    config.json
    results.json
    raw/
```

The record should include:

```text
hardware
OS/driver/CUDA
commit SHA
model/checkpoint identity
quantization
RAM type/capacity
PCIe generation + lane width
VRAM cap
buffer sizes
tile dimensions
measured H2D bandwidth
compute time
uncovered transfer time
GPU starvation
GPU utilization
correctness/output comparison
```

## No silent replacement of hypotheses with facts

Until an experiment verifies something, wording should remain:

```text
hypothesis
proposal
expected
estimated
```

After verification, record:

```text
measured
reproduced
hardware/configuration
commit SHA
```

## Architecture Decision Records

When implementation starts, important decisions should receive a short ADR rather than being buried in chat or commit history.

Suggested structure:

```text
docs/adr/
  0001-fixed-vram-ring.md
  0002-weight-atlas-format.md
  0003-host-pinning-strategy.md
```

Each ADR should contain:

```text
Context
Decision
Alternatives considered
Measured evidence
Consequences
Status
```

## Conflict avoidance

Before changing a shared interface, contributor should update or open the relevant design note first.

Examples of shared interfaces:

- Weight Atlas schema;
- execution-plan entry format;
- VRAM slot contract;
- quantized tile format;
- benchmark metric names.

This lets all five workstreams remain compatible.

## Current phase rule

At repository initialization, this is a **documentation-only phase**.

Do not treat code/proof as already existing. The next implementation stage begins only after the project description and original conversation have been captured.
