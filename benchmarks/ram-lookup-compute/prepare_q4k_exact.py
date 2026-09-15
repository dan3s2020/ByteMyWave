import os
import sys
import json
import numpy as np

from gguf import GGUFReader, dequantize

model = sys.argv[1]
outdir = sys.argv[2]

TENSOR = "blk.0.ffn_gate.weight"
SEED = 20260916

reader = GGUFReader(model, "r")

t = next(
    x for x in reader.tensors
    if x.name == TENSOR
)

if t.tensor_type.name != "Q4_K":
    raise RuntimeError("Tensor is not Q4_K")

rows = int(t.shape[0])
cols = int(t.shape[1])

raw = np.asarray(t.data).tobytes()

if len(raw) != t.n_bytes:
    raise RuntimeError(
        f"Raw byte mismatch: {len(raw)} != {t.n_bytes}"
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

print("Building independent gguf-py reference...")

W = dequantize(
    t.data,
    t.tensor_type
)

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
    "tensor": TENSOR,
    "rows": rows,
    "cols": cols,
    "q4k_bytes": int(t.n_bytes),
    "q4k_blocks": int(t.n_elements // 256)
}

with open(
    os.path.join(outdir, "meta.json"),
    "w"
) as f:
    json.dump(meta, f, indent=2)

print()
print("Prepared:")
print(f"  tensor     : {TENSOR}")
print(f"  shape      : {rows} x {cols}")
print(f"  Q4_K bytes : {t.n_bytes:,}")
print(f"  raw        : {qpath}")
print(f"  activation : {xpath}")
print(f"  reference  : {rpath}")
