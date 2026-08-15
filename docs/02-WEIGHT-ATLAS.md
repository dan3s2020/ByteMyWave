# 02 — Weight Atlas / Model Map

The conversation introduced the idea that a large model should be represented not only as a checkpoint file, but as an **addressable atlas** of tensors and sub-tensor tiles.

The purpose of the atlas is to answer immediately:

- what piece of the model is this?
- where is it in RAM?
- how large is it?
- in what quantization/encoding is it stored?
- which operation consumes it?
- which outputs depend on it?
- what comes before and after it in the execution plan?
- is it currently in RAM, pinned host memory, VRAM, or absent?
- if it is missing/corrupt, can it be recovered exactly or approximated?

## Hierarchical representation

The model can be viewed hierarchically:

```text
MODEL
  |
  +-- COMPONENT
       |
       +-- BLOCK
            |
            +-- OPERATION
                 |
                 +-- TENSOR
                      |
                      +-- TILE
                           |
                           +-- QUANT BLOCK / BYTES
```

Example:

```text
H3
 |
 +-- Transformer
      |
      +-- Block 17
           |
           +-- Attention
           |    +-- Wq -> [Q00][Q01][Q02]...
           |    +-- Wk -> [K00][K01][K02]...
           |    +-- Wv -> [V00][V01][V02]...
           |    +-- Wo -> [O00][O01][O02]...
           |
           +-- FFN
                +-- UP   -> [U00][U01]...
                +-- GATE -> [G00][G01]...
                +-- DOWN -> [D00][D01]...
```

## Proposed tile metadata

Each tile should have enough metadata that runtime lookup becomes arithmetic rather than discovery.

Conceptual record:

```text
id: block17.attn.wq.tile[0,2]
ram_offset: 0x...
compressed_bytes: ...
logical_shape: ...
quantization: Q4
scale_metadata_offset: ...
operation_id: ...
execution_index: ...
output_region: ...
next_use_index: ...
reuse_count: ...
checksum/hash: ...
importance_class: hot/warm/cold
```

Optional research metadata can include:

```text
mean
variance
L2 norm
min/max
low-rank sketch
similarity-neighbor IDs
reconstruction confidence
```

## Exact missing-piece detection

The atlas can tell us **exactly which piece is missing** because structure, position, size and dependency information are explicit.

Example:

```text
Expected: block17.attn.wq.tile[0,2]
Status: missing
Required by: op #1842
Expected bytes: 91,422,720
Quantization: Q4
Source region: host model store offset ...
```

However, the map alone cannot recover arbitrary lost learned values. It identifies what is missing; it does not magically infer the exact original numbers.

## Exact recovery through redundancy

The discussion proposed adding redundancy explicitly.

Simple conceptual parity:

```text
A B C D
P = A XOR B XOR C XOR D
```

If `C` is lost:

```text
C = P XOR A XOR B XOR D
```

A practical system could instead use erasure coding across groups of compressed tiles, allowing exact recovery of one or more missing/corrupt chunks with controlled storage overhead.

This is optional for inference performance, but useful for the broader "data entanglement" idea.

## Approximate reconstruction / shared structure

A second idea is to analyze whether multiple tiles share structure:

```text
W1 ~= B * C1
W2 ~= B * C2
W3 ~= B * C3
W4 ~= B * C4
```

where `B` is a shared basis and `C1..C4` are smaller coefficient sets.

Possible relationships in a semantic weight graph:

- cosine similarity;
- correlated distributions;
- shared low-rank bases;
- shared singular-vector structure;
- same quantization/codebook family;
- sensitivity/importance similarity;
- cross-layer predictive relationships.

A missing tile could then have states such as:

```text
exact recovery available: yes/no
parity recovery available: yes/no
approximation candidates: [...]
estimated reconstruction error: ...
```

Approximation is a research direction and must never be confused with exact recovery.

## Visualization

A useful visualization is not a literal plot of billions of scalar weights. It is a zoomable structural map.

Example health view:

```text
Block 00  ███████████████████████
Block 01  ███████████████████████
Block 02  ██████████░████████████
                    ^ missing/cold/not resident
Block 03  ███████████████████████
```

The same atlas can visualize residency:

```text
VRAM resident      = hot
in pinned window   = ready
in normal RAM      = queued/cold
on disk only       = not staged
missing/corrupt    = error/recovery path
```

## Runtime role

The atlas is not only documentation. It can become the source from which we compile a deterministic `EXECUTION_PLAN`.

At runtime, the hot path should ideally reduce to something conceptually like:

```text
tile = execution_plan[i]
src  = host_model_base + tile.ram_offset
dst  = fixed_vram_slot[next_slot]
```

No filesystem search, no dynamic graph discovery and no expensive hash lookup should be required in the critical transfer path.
