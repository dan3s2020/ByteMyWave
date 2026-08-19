# 16 — Transit Active Memory, Retriever Guards, and Chronicle Memory

This document captures the agentic-memory architecture discussed on 2026-08-19 and integrates it conceptually with Transit.

The core idea is not to make the main LLM repeatedly search a giant filesystem. Instead, the project is kept as a live memory substrate, cheap structural indexes eliminate irrelevant regions immediately, GPU-parallel retriever guards decide whether their assigned memory is relevant, and background chroniclers maintain evidence-backed local histories that can later be used during ambiguous retrieval.

This is a software/runtime layer. It does not change the raw token-generation rate of the main model by itself. Its objective is to reduce wall-clock task time by eliminating file-search, repeated reading, context reconstruction, and unnecessary prompt tokens.

---

## 1. Goal

For a very large code/project corpus, potentially hundreds of GB or ~1 TB, conventional agent behavior is wasteful:

```text
user task
  -> search filesystem
  -> grep
  -> open file
  -> search again
  -> open dependencies
  -> reconstruct context
  -> reason
  -> edit
```

Transit Active Memory changes the flow to:

```text
user task
  -> build task context
  -> query RAM-resident structural memory
  -> activate only relevant retriever guards
  -> verify candidate memories + journals + current source
  -> deliver exact source regions to main reasoner
  -> edit RAM working tree
  -> checkpoint dirty files to persistent storage
```

The desired user-visible effect is higher **task throughput** even if raw model decoding remains unchanged.

---

## 2. RAM as the active project substrate

The active project can be exposed as a normal filesystem while the working set is RAM-backed.

Conceptually:

```text
SSD / NVMe
   |
   | load / restore
   v
RAM working store
   |
   +-- read
   +-- write
   +-- modify
   +-- rename
   +-- create
   +-- delete
   |
   | dirty-file checkpoint
   v
SSD / NVMe
```

The agent still sees ordinary paths and files. The difference is that the hot working representation is memory-resident.

Suggested tiers:

```text
HOT   -> RAM resident
WARM  -> RAM cache + persistent backing
COLD  -> SSD/NVMe only until promoted
```

For each mutable file, track at minimum:

```text
path
original hash
current RAM version
dirty flag
diff/version
timestamp or generation ID
```

A project-wide writeback must not rewrite 1 TB. Only dirty files are persisted.

---

## 3. Crash safety and persistence

RAM is not persistent across power loss or reboot. Therefore the runtime should combine fast RAM mutation with durable incremental logging.

Recommended path:

```text
RAM edit
  -> append minimal WAL / journal record to NVMe
  -> continue operating in RAM
  -> periodic atomic checkpoint
  -> fsync / durable commit boundary
```

At boot:

```text
persistent snapshot
  + WAL tail
  -> reconstruct RAM working tree
  -> validate hashes / generations
  -> resume
```

The RAM layer is therefore the **active workspace**, while SSD/NVMe remains the durable source of recovery.

---

## 4. Keep more than raw source in RAM

The value of RAM is not merely faster `open()` calls.

The active memory fabric should retain multiple synchronized representations:

```text
raw/hot source
paths
file metadata
symbol tables
ASTs
imports
call graph
caller/callee graph
dependency graph
test-to-source relationships
git/version metadata
summaries
lexical indexes
semantic embeddings
retriever descriptors
chronicle memories
hot KV/prefix state where practical
```

The main LLM should receive only the smallest source/evidence packet needed for the current task.

---

## 5. Retriever guard model

Do not use one monolithic retriever that scans the complete project for every query.

Each logical memory region has a retriever guard that knows what it protects.

Examples of protected units:

```text
repository
module/package
directory
file
class
function
symbol
code chunk
historical episode
```

A guard receives the current task query plus its own static memory description and answers approximately:

```text
Is this task about memory I protect?

NO  -> terminate branch
MAYBE -> promote to deeper/stronger retrieval
YES -> return candidate + evidence + confidence
```

This is conceptually an **active filesystem**: relevant memories identify themselves instead of forcing the main agent to discover them sequentially.

---

## 6. Hierarchical early termination

The critical optimization is hierarchical pruning.

```text
TASK
 |
 v
LEVEL 0 — global/index guards
 |
 +-- NO -> branch dies
 |
 v
LEVEL 1 — repo/module guards
 |
 +-- NO -> branch dies
 |
 v
LEVEL 2 — file guards
 |
 +-- NO -> branch dies
 |
 v
LEVEL 3 — function/symbol guards
 |
 +-- NO -> branch dies
 |
 v
LEVEL 4 — code/chunk guards
 |
 +-- MATCH -> return exact source/evidence
```

A negative decision at a parent should prevent all descendants from running.

This matters more than simply placing millions of independent retrievers on GPUs.

---

## 7. GPU parallelism

The user architecture assumes substantial GPU compute and parallelism.

Implementation should not launch one OS/CUDA process per file. Instead, guards should be represented as batched logical cells.

Example:

```text
GPU 0 -> guard shard 0..99,999
GPU 1 -> guard shard 100,000..199,999
GPU 2 -> guard shard 200,000..299,999
...
```

A cheap first-stage output can be compact:

```text
0 = irrelevant
1 = possible
2 = strong candidate
```

Then only the surviving candidates receive more expensive semantic evaluation.

Illustrative conditional-compute cascade:

```text
10,000,000 chunks
      |
      v
RAM structural/index filter
      |
      v
20,000 guard candidates
      |
      v
600 file guards
      |
      v
37 function guards
      |
      v
4 exact source regions
      |
      v
main reasoner
```

The exact counts are workload-dependent and must be benchmarked; they are not performance claims.

---

## 8. Static guard state and prefix/KV reuse

A guard should not repeatedly reread a long identity prompt such as:

```text
I protect file X.
It defines A, B, C.
It imports D and E.
Its responsibility is F.
```

Where the model/runtime permits, keep the stable portion in reusable state:

```text
STATIC GUARD STATE
  file/symbol identity
  summaries
  relationships
  relevant historical digest
  cached prefix/KV representation where supported

+ NEW TASK CONTEXT
```

The variable input should be dominated by the new task, not the guard's repeated self-description.

---

## 9. Two retrieval paths

Not every query deserves GPU semantic work.

### Fast path

For exact or nearly exact questions:

```text
RAM path/symbol/index lookup
  -> exact candidate
  -> return
```

Examples:

- exact filename;
- exact symbol;
- known function name;
- exact caller/callee query;
- direct import relation.

### Deep path

For ambiguous semantic questions:

```text
RAM prefilter
  -> GPU guard batch
  -> semantic candidate set
  -> function/chunk guards
  -> evidence verification
  -> return
```

This prevents a simple lookup from consuming accelerator capacity.

---

## 10. Retriever as chronicler

Each retriever has a second role: **local chronicler**.

The retriever owns not only a current memory region, but also a compact evidence-backed history of what happened around that region.

The chronicler runs after useful work has completed and only at low priority when accelerator capacity is available.

Conceptual prompts:

```text
PROMPT/ROLE 1 — GUARD
"I protect memory X. Is the current task about me?
If yes, return relevant material, evidence and confidence.
If not, stop."

PROMPT/ROLE 2 — CHRONICLER
"Given the memory I protect, the task that just completed,
the verified result and the changes produced, update my journal."
```

The chronicler is intentionally asynchronous relative to the user's critical path in the scheduling sense: it is queued as background/low-priority work and preempted whenever a user task requires the GPU.

It must not delay interactive work.

---

## 11. Chronicle contents

A journal should not be an unverifiable stream of free-form internal thoughts.

It should store structured, contestable project memory:

```text
entity / memory ID
current purpose
observed facts
evidence references
historical events
verified findings
hypotheses / interpretations
confidence
open questions
related memories
version / commit validity range
superseded beliefs
reason for supersession
```

Example:

```text
ENTITY:
PaymentService.refund()

CURRENT PURPOSE:
Processes refunds and updates order state.

EVIDENCE:
- PaymentService.ts:144-231
- StripeAdapter.ts:80-127
- refund.test.ts:31-102

HISTORY:
#381
User investigated failed Stripe refunds remaining pending.
Missing transition found in handleRefundFailure().
Patch applied and tests verified.

#419
refund() was refactored.
Old belief that PaymentService writes directly to DB is obsolete.

CURRENT BELIEFS:
- Refund status is written through OrderRepository.
  confidence: 0.98
- Stripe webhook may invoke the refund path asynchronously.
  confidence: 0.94

OPEN QUESTIONS:
- retry path may duplicate events

VALID_FOR:
commit/generation ...
```

---

## 12. Facts and interpretations must be distinct

Chronicle memory should explicitly distinguish:

```text
FACT:
Function X currently calls Y.

INTERPRETATION:
The user was probably referring to X when discussing the old bug.
```

Facts can become strongly trusted when verified against current source/tests.

Interpretations are context-dependent and must be re-evaluated for every new task.

This prevents old memory from silently redefining what a new user request means.

---

## 13. Current task context is authoritative for retrieval intent

A new task must be inferred from current context, not from historical confidence alone.

The task-context builder should combine:

```text
CURRENT USER MESSAGE
        +
recent conversation context
        +
current repository/project state
        +
relevant historical memories
```

Historical journals provide evidence and continuity. They do not own the meaning of the new request.

This addresses cases where two retrievers both believe that an ambiguous user reference such as "codul 1" or "problema aia" refers to their own history.

---

## 14. Evidence voting and adjudication

Multiple retrievers may return competing claims.

Do not simply select the highest self-reported confidence.

A candidate can be scored across several dimensions:

```text
candidate        relevance   evidence   recency   consistency
--------------------------------------------------------------
Retriever A         .93         .99        .97         .95
Retriever B         .89         .72        .43         .81
Retriever C         .37         .91        .99         .90
```

A resolver may combine these scores, but contradictions should remain explicit until verified.

Example:

```text
A: "User means refund failure handler."
B: "User means webhook retry handler."

        |
        v
DO NOT COLLAPSE IMMEDIATELY
        |
        v
fetch current code + current task context + relevant history
        |
        v
resolve with current evidence
```

---

## 15. Evidence authority order

Recommended authority order:

```text
1. current source / runtime evidence
2. current tests / logs / measurements
3. current user task and recent context
4. verified historical chronicle
5. older summaries / hypotheses
```

A chronicler entry can be useful but must lose against changed source code or new runtime evidence.

Each memory item should therefore carry provenance and version information.

---

## 16. Scheduler priorities

The active-memory runtime should explicitly prioritize interactive work over memory maintenance.

Suggested classes:

```text
P0 user inference / coding
P1 retrieval required by current task
P2 reranking / verification / adjudication
P3 chronicler updates
P4 compaction / re-indexing / maintenance
```

When a P0/P1 task arrives:

```text
P3/P4 work running
   -> preempt / pause
   -> run interactive work
   -> resume maintenance when capacity is free
```

Background memory work must never be allowed to make the system feel slower.

---

## 17. Incremental updates after edits

Do not rebuild a 1 TB index after each patch.

After a task modifies files:

```text
changed files
  -> update raw RAM copies
  -> reparse affected AST nodes
  -> update symbol table
  -> update call/dependency edges
  -> update lexical/semantic representation
  -> invalidate affected guard state
  -> enqueue chronicler update
  -> checkpoint dirty data
```

Dependency propagation should be bounded to the affected graph region whenever possible.

---

## 18. Integration with Transit

This subsystem fits above/beside the existing Transit hardware runtime.

The combined conceptual stack becomes:

```text
USER / AGENTIC FRAMEWORK
          |
          v
TASK CONTEXT BUILDER
          |
          v
TRANSIT ACTIVE MEMORY
  RAM filesystem / source
  AST / symbols / graphs
  summaries / embeddings
  retriever guards
  chronicle memories
  evidence resolver
          |
          v
MAIN REASONER / EDITOR
          |
          +----------------------------+
          |                            |
          v                            v
software/file operations       model inference runtime
                                   |
                                   v
                         existing Transit compute fabric
```

The same project may therefore use Transit in two senses:

1. **model execution Transit** — resident model weights, local compute, distributed memory bandwidth;
2. **agentic active-memory Transit** — resident project/code state, parallel retrieval, evidence-backed histories, reduced filesystem/context overhead.

They should remain separately measurable.

---

## 19. Why this can feel like higher tokens/s

Raw decoder throughput may remain, for example, 20 tok/s.

But task latency includes non-decoding work:

```text
filesystem search
open/read cycles
context reconstruction
repeated tool calls
prompt/prefill of irrelevant code
reasoning
output generation
```

Illustrative conventional task:

```text
search files             20 s
open/grep/read           15 s
reconstruct context      25 s
reason/generate          40 s
-----------------------------
wall clock              100 s
```

Illustrative active-memory task:

```text
RAM/GPU retrieval       0.5-3 s
context resolution        1-3 s
reason/generate            40 s
------------------------------
wall clock              ~42-46 s
```

The raw 20 tok/s model did not become a 40 tok/s decoder. But the user sees roughly 2x+ faster task completion in this illustrative workload.

For search-heavy agentic tasks, the wall-clock improvement could be larger. For pure generation tasks, it will be smaller.

These values are examples, not measured ByteMyWave results.

---

## 20. Prefill reduction is a real compute saving

The system may also reduce actual model compute by avoiding irrelevant prompt tokens.

Example concept:

```text
without active memory:
120k-token context

with exact retrieval:
18k-token relevant context
```

That can reduce:

```text
prefill time
attention work
KV-cache footprint
memory pressure
context-window waste
```

Therefore the architecture can produce both:

1. apparent speedup from eliminating agent/tool overhead;
2. real compute reduction from smaller relevant prompts.

---

## 21. Metrics that matter

Do not evaluate this subsystem only by model decoding tokens/s.

Measure:

```text
raw generation tok/s
prefill tok/s / prefill latency
time to first useful action
retrieval latency
candidate count by stage
bytes read from SSD
bytes read from RAM
source tokens delivered to main reasoner
irrelevant-token ratio
tool calls per task
wall-clock task completion time
tasks/hour
useful output tokens / wall-clock second
GPU utilization by priority class
chronicler backlog
retrieval false-negative rate
retrieval false-positive rate
journal contradiction rate
journal staleness rate
```

The main success criterion is lower wall-clock task completion without reducing correctness.

---

## 22. Benchmark plan

A credible implementation must compare against a baseline agent using normal filesystem tools.

Prepare a corpus with known answers and tasks such as:

```text
find exact definition
find all callers
find writer of state X
trace bug across modules
identify files required for refactor
recover an old design decision
resolve ambiguous historical reference
apply multi-file patch
```

For each task record:

```text
baseline filesystem/tool latency
active-memory retrieval latency
main-model prompt tokens
number of opened files
correct target in top-1/top-5/top-20
final task correctness
wall-clock completion time
```

Run cold-cache and warm-memory modes separately.

---

## 23. Important caveats

### RAM alone is not the whole speedup

Windows/Linux already cache recently read files aggressively. Merely copying files to a RAM disk may not produce a dramatic difference for warm-cache reads.

The major architectural gain should come from **avoiding unnecessary search/read operations entirely**, not only from replacing SSD bandwidth with RAM bandwidth.

### Millions of full LLM retrievers would be wasteful

The design should use cheap hierarchical filters and batched guard computation, escalating only surviving candidates.

### Journals can be wrong

Chronicle memories are evidence-bearing hypotheses/history, not truth. They must be versioned and contestable.

### Current code wins

When a journal conflicts with the current code or current runtime evidence, the journal must be marked stale/superseded.

### Performance numbers require measurement

No wall-clock speedup should be promoted to a ByteMyWave measured result until benchmarked on the actual hardware/runtime.

---

## 24. Implementation sequence

A practical staged implementation:

```text
Stage 1
RAM-backed working tree + dirty-file checkpointing

Stage 2
path/symbol/AST/import/call-graph index

Stage 3
fast exact retrieval API

Stage 4
semantic file/function guards in large GPU batches

Stage 5
hierarchical early-termination scheduler

Stage 6
structured chronicle records + versioning/provenance

Stage 7
background chronicler queue with P0-P4 priorities

Stage 8
evidence voting/context resolver

Stage 9
incremental invalidation after edits

Stage 10
full baseline-vs-active-memory benchmark suite
```

Do not start by building millions of expensive retrievers. First prove that the memory/index/guard hierarchy reduces end-to-end task latency.

---

## 25. Working architectural summary

The intended system is:

```text
persistent NVMe source/checkpoints
             |
             v
       live RAM project
             |
   +---------+----------+
   |                    |
structural memory   chronicle memory
paths/AST/graph     evidence/history
   |                    |
   +---------+----------+
             |
       cheap prefilter
             |
       GPU guard batches
             |
      surviving memories
             |
    evidence adjudication
             |
      exact source packet
             |
        main reasoner
             |
      RAM edits + tests
             |
 incremental index/journal
             |
      dirty-file commit
             v
          NVMe/Git
```

The design principle is:

> Keep the project live in memory, let relevant memories raise their hand in parallel, preserve evidence-backed local history, and make the main LLM spend its compute on reasoning rather than rediscovering the project.
