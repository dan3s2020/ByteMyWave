# 17 — Current Transit Solution Addendum: Active Project Memory

This addendum extends [`14-CURRENT-SOLUTION.md`](./14-CURRENT-SOLUTION.md) with the agentic/project-memory subsystem documented in [`16-ACTIVE-MEMORY-RETRIEVER-CHRONICLES.md`](./16-ACTIVE-MEMORY-RETRIEVER-CHRONICLES.md).

It does **not** replace the existing Kimi K3 / DDR3 Transit compute-tile architecture. It adds a second, software-facing Transit layer whose purpose is to reduce agentic file/search/context overhead.

## Added subsystem

```text
                         USER TASK
                            |
                            v
                   Task Context Builder
                            |
                            v
                   Transit Active Memory
          +-----------------+------------------+
          |                                    |
     RAM working tree                  Structural memory
 source / dirty state           paths / AST / symbols / graph
          |                                    |
          +-----------------+------------------+
                            |
                      cheap prefilter
                            |
                      GPU guard batches
                            |
                 Chronicle / evidence vote
                            |
                      exact code packet
                            |
                       main reasoner
                            |
                 edit/test in RAM workspace
                            |
          incremental index + chronicle update
                            |
                  dirty-file checkpoint
                            v
                         NVMe / Git
```

## Relationship to existing Transit

The existing Transit architecture optimizes the model execution path:

```text
resident model weights
+ local memory bandwidth
+ local compute
+ small activation/result movement
```

The active-memory subsystem optimizes the agentic software path:

```text
resident project state
+ hierarchical retrieval
+ GPU-parallel relevance guards
+ evidence-backed history
+ minimal context delivery
```

They are complementary and must be benchmarked separately.

## New runtime responsibilities

A host running the agentic Transit layer should maintain:

```text
RAM-backed hot working tree
persistent NVMe backing/checkpoints
write-ahead log for crash recovery
path/symbol indexes
AST + import + call/dependency graph
semantic representations
retriever guard descriptors
chronicle records
context resolver / evidence adjudicator
priority scheduler for interactive vs background work
incremental invalidation after edits
```

## Scheduling rule

Interactive work always wins:

```text
P0 user inference / coding
P1 retrieval required now
P2 reranking / verification
P3 chronicler updates
P4 compaction / re-indexing
```

Chroniclers and compaction should run only when accelerator capacity is otherwise free and should be preempted when P0/P1 work appears.

## Authority rule

Chronicle memory is historical evidence, not ground truth.

Recommended authority order:

```text
1. current source/runtime evidence
2. current tests/logs/measurements
3. current user task + recent context
4. verified historical chronicle
5. old summaries/hypotheses
```

The meaning of a new ambiguous task must be inferred from the current task context, even when multiple retrievers have strong historical beliefs that the user is referring to them.

## Performance claim discipline

The subsystem may make a fixed-tok/s model feel much faster by reducing:

```text
filesystem search
repeated open/grep cycles
context reconstruction
irrelevant prompt tokens
prefill work
unnecessary tool calls
```

But until benchmarked, speedup numbers remain illustrative.

Track at minimum:

```text
raw generation tok/s
retrieval latency
prefill latency
time to first useful action
tool calls/task
files opened/task
prompt tokens/task
task wall-clock time
tasks/hour
retrieval top-k correctness
journal staleness/contradiction rate
```

## Implementation gate

Before scaling to millions of guards, prove the following sequence:

```text
RAM workspace
 -> exact structural retrieval
 -> hierarchical semantic guard batch
 -> correct top-k source retrieval
 -> incremental update after edit
 -> chronicler update while idle
 -> historical ambiguity resolution
 -> measurable task-latency improvement vs baseline
```

This addendum should be read together with `14-CURRENT-SOLUTION.md` and `16-ACTIVE-MEMORY-RETRIEVER-CHRONICLES.md` when implementing the current Transit concept.
