# 07 — MiniMax H3 release gate

Status checked: **2026-08-15**.

## What is verified

MiniMax officially announced MiniMax H3 on 2026-07-31. The official announcement describes:

- unified multimodal context across text, image, video and audio;
- native stereo audio/video generation;
- up to 15 seconds at 2K;
- H3-VAE;
- H3-Omni Transformer;
- In-context Regeneration;
- an intent to open the model weights in the following days, subject to applicable laws and regulations.

Official announcement:

- https://minimaxi.com/blog/minimax-h3

Official MiniMax Hugging Face organization:

- https://huggingface.co/MiniMaxAI

## What is NOT yet treated as verified in TensorWave

At the time of this status check, an official downloadable H3 checkpoint is not discoverable in the public MiniMaxAI Hugging Face model listing available to us.

Therefore TensorWave must **not** hardcode or promote as fact any currently unverified claim about:

- exact H3 parameter count;
- exact number of Transformer blocks;
- exact hidden size / FFN dimensions / attention heads;
- exact text encoder identity;
- exact checkpoint shard names;
- exact tensor names;
- exact source dtype;
- exact quantized checkpoint sizes.

Those values become project facts only when they can be read from an official checkpoint, config, tech report, or official implementation.

## Engineering consequence

Phase 2 is intentionally model-agnostic.

The tooling works directly against the safetensors file format:

1. read shard headers only;
2. inventory exact tensor names, shapes, dtypes and byte offsets;
3. construct the Weight Atlas;
4. select a homogeneous family of real rank-2 tensors;
5. copy exact row slices into a streaming pack without numeric deserialization;
6. record provenance + SHA-256 for every tile;
7. feed those exact checkpoint bytes to the fixed-VRAM CUDA ring.

When an official H3 checkpoint is released, the required input should be only its local checkpoint directory:

```powershell
.\scripts\run-real-weight-proof.ps1 -ModelDir "D:\models\MiniMax-H3"
```

No H3-specific tensor names should be necessary for the first real-weight experiment.

## Current boundary

A successful Phase-2 result proves:

> Real neural-network checkpoint bytes, substantially larger than the resident VRAM weight ring, can be streamed from pinned host RAM into fixed VRAM slots while GEMM work proceeds, with measured starvation/overlap and exact sequential-vs-overlapped output comparison.

It does **not** yet prove:

> The selected storage-order tensor sequence is the true H3 inference execution order.

Graph-derived execution order is the next stage after the checkpoint is available.
