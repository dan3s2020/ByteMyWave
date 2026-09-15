import sys
import time
import math
import numpy as np

from gguf import GGUFReader, dequantize
from sklearn.cluster import MiniBatchKMeans

TENSOR = "blk.0.ffn_gate.weight"
K = 256
GROUPS = [16, 8]

QUALITY_TESTS = 8
BENCH_TESTS = 20
SEED = 20260916

path = sys.argv[1]

print()
print("=" * 80)
print(" REAL QWEN - POSITION-SPECIFIC PRODUCT QUANTIZATION LUT")
print("=" * 80)

reader = GGUFReader(path, "r")

t = next(
    x for x in reader.tensors
    if x.name == TENSOR
)

flat = dequantize(
    t.data,
    t.tensor_type
)

flat = np.asarray(
    flat,
    dtype=np.float32
).reshape(-1)

rows = int(t.shape[0])
cols = int(t.shape[1])

W = flat.reshape(rows, cols)

rng = np.random.default_rng(SEED)

print(f"Tensor              : {TENSOR}")
print(f"Shape               : {rows:,} x {cols:,}")
print(f"Weights             : {flat.size:,}")
print(f"Original Q4_K       : {t.n_bytes/1024**2:.3f} MiB")
print()


# ============================================================
# FIXED TEST INPUTS
# ============================================================

quality_x = [
    rng.standard_normal(cols).astype(np.float32)
    for _ in range(QUALITY_TESTS)
]

quality_ref = [
    W @ x
    for x in quality_x
]

bench_x = [
    rng.standard_normal(cols).astype(np.float32)
    for _ in range(BENCH_TESTS)
]


# ============================================================
# FP32 BASELINE
# ============================================================

for x in bench_x[:3]:
    _ = W @ x

s = time.perf_counter()

checksum = 0.0

for x in bench_x:
    y = W @ x
    checksum += float(y[0])

e = time.perf_counter()

baseline_ms = (
    (e - s)
    / BENCH_TESTS
    * 1000
)

print(
    f"FP32 NumPy baseline : "
    f"{baseline_ms:.3f} ms/GEMV"
)

results = []


# ============================================================
# GROUP SWEEP
# ============================================================

for G in GROUPS:

    print()
    print("=" * 80)
    print(f" POSITION-SPECIFIC PQ -- GROUP {G}")
    print("=" * 80)

    if cols % G != 0:
        print("Not divisible, skipping.")
        continue

    subspaces = cols // G

    print(
        f"Subspaces           : "
        f"{subspaces:,}"
    )

    print(
        f"Vectors/codebook    : "
        f"{rows:,}"
    )

    print(
        f"Centroids/codebook  : "
        f"{K}"
    )

    # codes[row, subspace]
    codes = np.empty(
        (rows, subspaces),
        dtype=np.uint8
    )

    # Stored representation = FP16.
    centers16 = np.empty(
        (subspaces, K, G),
        dtype=np.float16
    )

    train_start = time.perf_counter()

    for sidx in range(subspaces):

        begin = sidx * G
        end = begin + G

        X = np.ascontiguousarray(
            W[:, begin:end],
            dtype=np.float32
        )

        km = MiniBatchKMeans(
            n_clusters=K,
            batch_size=1024,
            max_iter=60,
            n_init=1,
            init="k-means++",
            random_state=SEED + sidx + G*10000,
            reassignment_ratio=0.01,
            verbose=0
        )

        km.fit(X)

        c = np.ascontiguousarray(
            km.cluster_centers_,
            dtype=np.float32
        )

        code = km.predict(X).astype(
            np.uint8
        )

        centers16[sidx] = c.astype(
            np.float16
        )

        codes[:, sidx] = code

        if (
            sidx == 0
            or (sidx + 1) % 64 == 0
            or sidx + 1 == subspaces
        ):
            print(
                f"trained "
                f"{sidx+1:4d}/{subspaces}"
            )

    train_end = time.perf_counter()

    print(
        f"Training total      : "
        f"{train_end-train_start:.3f} s"
    )


    # ========================================================
    # MEASURE WEIGHT RECONSTRUCTION
    #
    # Use ACTUAL FP16 stored centers.
    # ========================================================

    weight_sse = 0.0
    weight_energy = 0.0
    weight_cross = 0.0
    recon_energy = 0.0

    for sidx in range(subspaces):

        begin = sidx * G
        end = begin + G

        orig = W[:, begin:end]

        c = centers16[sidx].astype(
            np.float32
        )

        recon = c[
            codes[:, sidx]
        ]

        o64 = orig.astype(
            np.float64
        )

        r64 = recon.astype(
            np.float64
        )

        d64 = o64 - r64

        weight_sse += float(
            np.sum(d64*d64)
        )

        weight_energy += float(
            np.sum(o64*o64)
        )

        recon_energy += float(
            np.sum(r64*r64)
        )

        weight_cross += float(
            np.sum(o64*r64)
        )

    weight_rel_l2 = math.sqrt(
        weight_sse /
        weight_energy
    )

    weight_cos = (
        weight_cross /
        math.sqrt(
            weight_energy *
            recon_energy
        )
    )

    weight_snr = (
        10.0 *
        math.log10(
            weight_energy /
            weight_sse
        )
    )

    print()
    print(
        f"Weight rel-L2       : "
        f"{weight_rel_l2:.6f}"
    )

    print(
        f"Weight cosine       : "
        f"{weight_cos:.6f}"
    )

    print(
        f"Weight SNR          : "
        f"{weight_snr:.2f} dB"
    )


    # ========================================================
    # RUNTIME REPRESENTATION
    # ========================================================

    # Convert once for NumPy compute.
    # Stored model remains FP16 centers.

    centers = centers16.astype(
        np.float32
    )

    sub_idx = np.arange(
        subspaces
    )[None, :]


    def pq_lut_gemv(x):

        xb = x.reshape(
            subspaces,
            G
        )

        # table[subspace, centroid]
        #
        # Shape:
        #
        # [S,G] x [S,K,G]
        # -> [S,K]
        #
        # Arithmetic:
        #
        # S*K*G
        #
        # = cols*K
        # = constant for every G.

        table = np.einsum(
            "sg,skg->sk",
            xb,
            centers,
            optimize=True
        )

        # For each matrix row and subspace:
        # select its precomputed centroid dot-product.

        selected = table[
            sub_idx,
            codes
        ]

        return np.sum(
            selected,
            axis=1,
            dtype=np.float32
        )


    # ========================================================
    # OUTPUT QUALITY
    # ========================================================

    cosines = []
    rel_errors = []
    max_errors = []

    for x, ref in zip(
        quality_x,
        quality_ref
    ):

        out = pq_lut_gemv(x)

        a = ref.astype(np.float64)
        b = out.astype(np.float64)

        diff = a - b

        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)

        cos = float(
            np.dot(a, b) /
            (na * nb)
        )

        rel = float(
            np.linalg.norm(diff) /
            na
        )

        mx = float(
            np.max(
                np.abs(diff)
            )
        )

        cosines.append(cos)
        rel_errors.append(rel)
        max_errors.append(mx)

    mean_cos = float(
        np.mean(cosines)
    )

    worst_cos = float(
        np.min(cosines)
    )

    mean_rel = float(
        np.mean(rel_errors)
    )

    worst_rel = float(
        np.max(rel_errors)
    )

    print()
    print(
        f"Output cosine mean  : "
        f"{mean_cos:.6f}"
    )

    print(
        f"Output cosine worst : "
        f"{worst_cos:.6f}"
    )

    print(
        f"Output rel-L2 mean  : "
        f"{mean_rel:.6f}"
    )

    print(
        f"Output rel-L2 worst : "
        f"{worst_rel:.6f}"
    )


    # ========================================================
    # STORAGE
    # ========================================================

    code_bytes = codes.nbytes
    center_bytes = centers16.nbytes

    total_bytes = (
        code_bytes +
        center_bytes
    )

    compression = (
        t.n_bytes /
        total_bytes
    )

    print()
    print(
        f"Codes              : "
        f"{code_bytes/1024**2:.3f} MiB"
    )

    print(
        f"FP16 codebooks     : "
        f"{center_bytes/1024**2:.3f} MiB"
    )

    print(
        f"Total PQ storage   : "
        f"{total_bytes/1024**2:.3f} MiB"
    )

    print(
        f"Compression/Q4_K   : "
        f"{compression:.2f}x"
    )


    # ========================================================
    # COMPUTE ACCOUNTING
    # ========================================================

    lut_build_macs = (
        subspaces *
        K *
        G
    )

    lookups = (
        rows *
        subspaces
    )

    heavy_reduction = (
        rows * cols /
        lut_build_macs
    )

    table_bytes = (
        subspaces *
        K *
        4
    )

    print()
    print(
        f"LUT-build MACs      : "
        f"{lut_build_macs:,}"
    )

    print(
        f"Runtime lookups     : "
        f"{lookups:,}"
    )

    print(
        f"Heavy MAC reduction : "
        f"{heavy_reduction:.2f}x"
    )

    print(
        f"Dynamic LUT size    : "
        f"{table_bytes/1024:.2f} KiB"
    )


    # ========================================================
    # NUMPY PERFORMANCE
    # ========================================================

    for x in bench_x[:3]:
        _ = pq_lut_gemv(x)

    start = time.perf_counter()

    cs = 0.0

    for x in bench_x:
        y = pq_lut_gemv(x)
        cs += float(y[0])

    end = time.perf_counter()

    pq_ms = (
        (end-start)
        / BENCH_TESTS
        * 1000
    )

    speedup = (
        baseline_ms /
        pq_ms
    )

    print()
    print(
        f"PQ-LUT GEMV         : "
        f"{pq_ms:.3f} ms"
    )

    print(
        f"Speed vs FP32       : "
        f"{speedup:.2f}x"
    )

    print(
        f"Checksum            : "
        f"{cs:.6f}"
    )

    results.append(
        (
            G,
            weight_rel_l2,
            weight_cos,
            mean_cos,
            mean_rel,
            total_bytes/1024**2,
            compression,
            table_bytes/1024,
            lookups,
            pq_ms,
            speedup
        )
    )


print()
print("=" * 80)
print(" FINAL POSITION-SPECIFIC PQ TABLE")
print("=" * 80)

print(
    " G | w_rel | w_cos | out_cos | out_rel | "
    "MiB  | comp | LUTKiB | lookups | ms | speed"
)

print("-" * 110)

for r in results:

    (
        G,
        wr,
        wc,
        oc,
        ore,
        mib,
        comp,
        lkb,
        lookups,
        ms,
        sp
    ) = r

    print(
        f"{G:2d} | "
        f"{wr:5.3f} | "
        f"{wc:5.3f} | "
        f"{oc:7.4f} | "
        f"{ore:7.4f} | "
        f"{mib:5.2f} | "
        f"{comp:4.2f}x | "
        f"{lkb:7.1f} | "
        f"{lookups/1e6:6.3f}M | "
        f"{ms:5.2f} | "
        f"{sp:5.2f}x"
    )

print()
print("=" * 80)
print(" DECISION")
print("=" * 80)

print()
print("If G16 reaches approximately:")
print("  output cosine >= 0.95")
print("  and relative L2 falls strongly,")
print()
print("then position-specific/product quantization")
print("is a much better representation than the global codebook.")
print()
print("If G16 is still weak, inspect G8.")
print()
print("Python timing is NOT a llama.cpp Q4_K comparison.")
print("Quality is the primary result of this experiment.")
print("=" * 80)
