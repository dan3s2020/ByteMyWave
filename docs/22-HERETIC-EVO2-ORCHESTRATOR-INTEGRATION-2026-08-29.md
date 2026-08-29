# 22 — Heretic + Evo2 Orchestrator Integration — 2026-08-29

## Purpose

This branch adds **Heretic** as a pinned upstream research dependency for the LLM-orchestrator layer used by Transit/TensorWave workflows that call **Evo2** as a specialized genomics tool.

The integration boundary is deliberate:

```text
base GLM/Kimi orchestrator model
        |
        | offline model-research / evaluation stage
        v
Heretic candidate model artifact
        |
        v
Transit LLM serving runtime
        |
        v
agent/tool router
        |
        +------> Evo2 local service / API
        |             |
        |             v
        +<----- structured Evo2 result
        |
        v
final LLM interpretation / answer
```

Heretic is **not** inserted into the token-by-token Transit hot path. Evo2 is **not** modified by Heretic. The output of the offline model-research stage is a separate candidate orchestrator artifact that can be accepted or rejected after benchmarking.

## Upstream pin

The repository vendors Heretic as a git submodule:

```text
third_party/heretic
```

Pinned source of record:

```text
upstream: https://github.com/p-e-w/heretic.git
tag:      v1.4.0
commit:   6ea3b8d778d047b4b3b7c5b843e21c5bea98ee8d
license:  AGPL-3.0-or-later
```

The release ZIP published for v1.4.0 reports:

```text
sha256: 02dcaec06c513aa353d4323ca4140650291bb0f6661439fbb283f8bf3f5b9057
```

Machine-readable provenance is kept in:

```text
integrations/heretic/heretic.lock.toml
```

The submodule pin, not a floating `master`, is the reproducibility boundary.

## Why this is a separate research track

The current ByteMyWave/Transit execution work is primarily about model placement, memory bandwidth, heterogeneous execution, routing and local-memory compute. Heretic solves a different problem: modifying an LLM artifact offline and evaluating how behavior changes.

Those concerns must remain separable:

```text
Transit performance path != Heretic optimization path
Evo2 execution path       != Heretic optimization path
```

This prevents an offline experiment from being confused with a throughput optimization and makes rollback trivial: serve the original model artifact instead of the candidate artifact.

## GLM 5.x gate

Do **not** assume that the current Heretic release supports the exact GLM checkpoint just because Heretic supports several MoE families.

Upstream issue #90 documents that GLM models did not originally work out of the box and that a community dynamic auto-registration modification was used successfully for a GLM-4.6V-Flash model:

- https://github.com/p-e-w/heretic/issues/90

Therefore the GLM path is gated:

### G0 — exact-checkpoint load test

Pin the exact GLM checkpoint/revision intended for Transit and prove that Heretic can load the architecture without silently substituting model classes or losing custom modules.

### G1 — architecture registration test

If upstream auto-detection fails, isolate the minimum model-registration adapter required for the GLM architecture. Keep that adapter local to this integration track and test it independently.

### G2 — artifact integrity test

For original and candidate artifacts, record:

- model revision;
- tokenizer revision;
- tensor inventory;
- tensor shapes/dtypes;
- changed tensor set;
- artifact hashes;
- load/serve success in the selected runtime.

### G3 — benign Evo2 workflow benchmark

Compare the original orchestrator and the candidate on the exact same benign Evo2 tool-use suite. A candidate is useful only if unnecessary refusals fall **without** materially damaging tool-call validity, successful tool completion or answer quality.

No GLM-5.3 compatibility claim is made until G0–G3 pass on the exact checkpoint.

## Kimi K3 gate

The Kimi path is more experimental.

Heretic upstream issue #221 records a systematic Kimi-K2.5 test where standard linear abliteration had little or no useful behavioral effect on the main benchmark, with the report pointing to MoE routing/attention or other deeper mechanisms as possible causes:

- https://github.com/p-e-w/heretic/issues/221

Kimi K3 therefore gets a stricter rule:

```text
Do not treat `heretic <Kimi checkpoint>` as a validated K3 solution.
```

The first K3 work is **measurement**, not modification:

1. prove exact K3 model loading and layer/module discovery;
2. collect benign workflow refusal/acceptance traces;
3. collect router/expert-selection traces where technically available;
4. compare those traces between ordinary successful requests and false-refusal cases;
5. only then decide whether standard Heretic is applicable or whether the architecture needs a Kimi-specific research adapter.

This keeps the K3 research consistent with ByteMyWave's existing evidence rule: an architectural hypothesis is not a measured end-to-end result.

## Evo2 boundary

Evo2 remains a separately versioned tool/service. The orchestrator should invoke it through a stable adapter rather than importing Evo2 internals into the LLM modification code.

Logical contract:

```text
LLM tool request
  -> validated Evo2 operation + input
  -> Evo2 execution
  -> structured result + provenance + errors
  -> LLM interpretation
```

The first benchmark suite is intentionally restricted to **benign/non-hazardous** operations such as toy-sequence comparison, embeddings, metadata queries, neutral sequence scoring and result interpretation. The purpose of this track is to catch false refusals in legitimate tool use, not to create an unrestricted biological-design pipeline.

Official Evo2 repository:

- https://github.com/ArcInstitute/evo2

## Benchmark contract

Files:

```text
benchmarks/heretic-evo2/README.md
benchmarks/heretic-evo2/benign-evo2-prompts.jsonl
tools/heretic_evo2_gate.py
```

For every prompt, each evaluated model produces a JSONL record with at least:

```json
{
  "id": "toy-sequence-compare-001",
  "refused": false,
  "tool_call_valid": true,
  "tool_completed": true,
  "quality_score": 1.0
}
```

The gate compares original vs candidate on:

- false-refusal rate;
- valid tool-call rate;
- successful tool-completion rate;
- mean quality score;
- per-case regressions.

A lower refusal count by itself is **not** sufficient. A candidate that stops refusing but produces broken tool calls or lower-quality scientific interpretation fails the gate.

## Artifact policy

Never overwrite the original model.

Use immutable identities such as:

```text
<model>-original-<revision>
<model>-heretic-candidate-<date>-<config-hash>
```

Store alongside the candidate:

```text
base revision
Heretic version/commit
configuration hash
prompt-dataset hash
reproduction metadata
changed-tensor inventory
model/tokenizer hashes
benchmark report
```

This makes every experiment reversible and auditable.

## Transit deployment implication

If a candidate passes the benchmark, Transit does not require a new hardware architecture. It simply serves a different compatible model artifact through the same model-serving interface.

Conceptually:

```text
artifact selection
      |
      +-- original GLM/Kimi
      |
      +-- validated candidate
              |
              v
       existing Transit runtime
              |
              v
          tool router -> Evo2
```

Therefore the expected **steady-state Heretic overhead is zero** in the inference loop, apart from any model-weight differences produced by the offline process. The expensive search/evaluation stage happens before deployment.

## Acceptance ladder

### H0 — dependency reproducibility

- submodule initializes;
- checked-out SHA equals the lock file;
- upstream license remains visible.

### H1 — small supported-model smoke test

Prove the pinned Heretic install itself works before debugging GLM/Kimi architecture support.

### H2 — exact GLM architecture load

Pass G0–G2.

### H3 — benign Evo2 baseline capture

Run the unmodified GLM orchestrator on the benchmark suite and preserve raw results.

### H4 — GLM candidate comparison

Candidate must reduce false refusals while preserving tool validity/completion and quality within the configured regression budget.

### H5 — Transit serving proof

Serve the accepted candidate through the exact runtime/API used by the 4×2 Transit setup and repeat the benchmark end-to-end.

### H6 — Kimi K3 research

Only after the GLM path is reproducible, begin K3 architecture/trace analysis. Upstream K2.5 evidence means standard linear abliteration is not assumed to transfer.

## What this branch claims

It **does** claim:

- Heretic v1.4.0 is pinned in ByteMyWave as an upstream submodule;
- the integration/reproducibility boundary is defined;
- GLM and Kimi have explicit compatibility gates;
- a benign Evo2 false-refusal/tool-use benchmark contract is defined;
- Heretic is kept out of the Transit hot inference path.

It **does not** claim:

- GLM-5.3 has already been successfully processed by Heretic;
- Kimi K3 standard abliteration works;
- Evo2 itself has been modified;
- any new end-to-end token/s result;
- any candidate model has passed the benchmark yet.

The next falsifiable step is H0/H1, followed by an exact GLM checkpoint load test.
