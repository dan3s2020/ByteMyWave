import os
import sys
import json
import numpy as np

from gguf import GGUFReader, dequantize

model = sys.argv[1]
outdir = sys.argv[2]

TENSOR = "blk.0.ffn_gate.weight"
SEED = 20260916
GEOMETRY_VERSION = 2

reader = GGUFReader(model, "r")

t = next(
    x for x in reader.tensors
    if x.name == TENSOR
)

if t.tensor_type.name != "Q4_K":
    raise RuntimeError("Tensor is not Q4_K")

if len(t.shape) != 2:
    raise RuntimeError(
        f"Expected a 2-D tensor, got GGUF shape {tuple(int(v) for v in t.shape)}"
    )

# IMPORTANT: GGUFReader.ReaderTensor.shape is stored in GGML dimension order:
#   shape[0] == ne[0] == contiguous row length / GEMV input columns
#   shape[1] == ne[1] == number of matrix rows / GEMV outputs
#
# GGUFReader separately reverses these dimensions when it reshapes t.data for
# NumPy.  The old V10 preparation code treated t.shape as NumPy (rows, cols),
# which reinterpreted the same Q4_K byte stream with the wrong row boundaries.
cols = int(t.shape[0])
rows = int(t.shape[1])

if cols % 256 != 0:
    raise RuntimeError(
        f"Q4_K row length must be divisible by 256, got cols={cols}"
    )

raw = np.asarray(t.data).tobytes()

if len(raw) != t.n_bytes:
    raise RuntimeError(
        f"Raw byte mismatch: {len(raw)} != {t.n_bytes}"
    )

expected_blocks = rows * (cols // 256)
expected_bytes = expected_blocks * 144

if expected_bytes != t.n_bytes:
    raise RuntimeError(
        f"Q4_K geometry/byte mismatch: expected {expected_bytes} bytes "
        f"from {rows}x{cols}, GGUF reports {t.n_bytes}"
    )

qpath = os.path.join(
    outdir,
    "tensor-q4k.bin"
)

with open(qpath, "wb") as f:
    f.write(raw)

rng = np.random.default_rng(SEED)

x = rng.standard_normal(
    cols
).astype(np.float32)

xpath = os.path.join(
    outdir,
    "activation-f32.bin"
)

x.tofile(xpath)

print("Building independent gguf-py reference with real GGML row geometry...")

W = dequantize(
    t.data,
    t.tensor_type
)

# t.data is already laid out by GGUFReader in reversed/NumPy dimension order,
# i.e. rows x cols for a 2-D GGML tensor.  Keep that real matrix geometry.
W = np.asarray(
    W,
    dtype=np.float32
).reshape(
    rows,
    cols
)

reference = (
    W @ x
).astype(np.float32)

rpath = os.path.join(
    outdir,
    "reference-f32.bin"
)

reference.tofile(rpath)

meta = {
    "geometry_version": GEOMETRY_VERSION,
    "tensor": TENSOR,
    "rows": rows,
    "cols": cols,
    "gguf_ne0": cols,
    "gguf_ne1": rows,
    "q4k_bytes": int(t.n_bytes),
    "q4k_blocks": int(t.n_elements // 256),
    "blocks_per_row": int(cols // 256),
    "activation_f32_count": int(cols),
    "reference_f32_count": int(rows),
}

with open(
    os.path.join(outdir, "meta.json"),
    "w"
) as f:
    json.dump(meta, f, indent=2)

print()
print("Prepared:")
print(f"  tensor       : {TENSOR}")
print(f"  GGUF ne[0]   : {cols:,}  (GEMV input columns / contiguous row length)")
print(f"  GGUF ne[1]   : {rows:,}  (GEMV output rows)")
print(f"  GEMV matrix  : {rows:,} rows x {cols:,} cols")
print(f"  blocks/row   : {cols // 256:,}")
print(f"  Q4_K blocks  : {expected_blocks:,}")
print(f"  Q4_K bytes   : {t.n_bytes:,}")
print(f"  raw          : {qpath}")
print(f"  activation   : {xpath} ({cols:,} FP32 values)")
print(f"  reference    : {rpath} ({rows:,} FP32 values)")
print(f"  geometry ver : {GEOMETRY_VERSION}")
