#!/usr/bin/env python3
import os
import sys
import time
import ctypes
import statistics
import subprocess
from pathlib import Path

import numpy as np

ROWS = 640
COLS = 2048
NWEIGHTS = ROWS * COLS
WPLANE_BYTES = NWEIGHTS // 8
XPLANE_BYTES = COLS // 8
WFILE_BYTES = WPLANE_BYTES * 4

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "transit_ollama_test"
if not DEFAULT_DATA.exists():
    DEFAULT_DATA = HERE

ASM = HERE / "transit_bitplane_kernel.asm"
BIN = HERE / "transit_bitplane_kernel.bin"

def die(msg):
    print("\nFATAL:", msg)
    raise SystemExit(1)

def ensure_nasm():
    from shutil import which
    nasm = which("nasm")
    if nasm:
        return nasm
    print("[setup] NASM not found.")
    print("[setup] Installing NASM with winget...")
    p = subprocess.run(
        ["winget", "install", "-e", "--id", "NASM.NASM",
         "--accept-source-agreements", "--accept-package-agreements"],
        text=True
    )
    if p.returncode != 0:
        die("NASM install failed. Run: winget install -e --id NASM.NASM")
    # winget PATH updates may not enter this process. Search common location.
    candidates = []
    local = os.environ.get("LOCALAPPDATA")
    if local:
        root = Path(local) / "Microsoft" / "WinGet" / "Packages"
        if root.exists():
            candidates += list(root.rglob("nasm.exe"))
    if not candidates:
        die("NASM installed but nasm.exe is not visible yet. Re-open PowerShell and rerun.")
    return str(candidates[0])

def assemble():
    nasm = ensure_nasm()
    cmd = [nasm, "-f", "bin", "-O3", "-o", str(BIN), str(ASM)]
    print("[build]", " ".join(cmd))
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr)
        die("NASM assembly failed")
    print(f"[build] raw x64 kernel: {BIN} ({BIN.stat().st_size} bytes)")

def make_activation_planes(x):
    x = np.asarray(x, dtype=np.int8)
    if x.shape != (COLS,):
        die(f"activation shape must be ({COLS},), got {x.shape}")
    codes = (x.astype(np.int16) & 0xFF).astype(np.uint8)
    planes = []
    for bit in range(8):
        bits = ((codes >> bit) & 1).astype(np.uint8)
        planes.append(np.packbits(bits, bitorder="little"))
    out = np.concatenate(planes).astype(np.uint8, copy=False)
    assert out.nbytes == XPLANE_BYTES * 8
    return out

def unpack_q4(packed):
    p = np.frombuffer(packed, dtype=np.uint8)
    out = np.empty(p.size * 2, dtype=np.int8)
    lo = (p & 0x0F).astype(np.int8)
    hi = ((p >> 4) & 0x0F).astype(np.int8)
    lo = np.where(lo >= 8, lo - 16, lo).astype(np.int8)
    hi = np.where(hi >= 8, hi - 16, hi).astype(np.int8)
    out[0::2] = lo
    out[1::2] = hi
    return out[:NWEIGHTS].reshape(ROWS, COLS)

# Windows API
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
MEM_COMMIT = 0x1000
MEM_RESERVE = 0x2000
MEM_RELEASE = 0x8000
PAGE_READWRITE = 0x04
PAGE_EXECUTE_READWRITE = 0x40

kernel32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
kernel32.VirtualAlloc.restype = ctypes.c_void_p
kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
kernel32.VirtualFree.restype = ctypes.c_int

def valloc(size, executable=False):
    prot = PAGE_EXECUTE_READWRITE if executable else PAGE_READWRITE
    p = kernel32.VirtualAlloc(None, size, MEM_COMMIT | MEM_RESERVE, prot)
    if not p:
        raise OSError(ctypes.get_last_error(), "VirtualAlloc failed")
    return p

def load_kernel():
    code = BIN.read_bytes()
    addr = valloc(len(code), executable=True)
    ctypes.memmove(addr, code, len(code))
    FN = ctypes.WINFUNCTYPE(
        ctypes.c_int64,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
    )
    return FN(addr), addr, len(code)

def alloc_copy(blob):
    addr = valloc(len(blob), executable=False)
    ctypes.memmove(addr, blob, len(blob))
    return addr

# Direct SSD read support
GENERIC_READ = 0x80000000
FILE_SHARE_READ = 0x1
FILE_SHARE_WRITE = 0x2
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
FILE_BEGIN = 0
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

CreateFileW = kernel32.CreateFileW
CreateFileW.argtypes = [
    ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p
]
CreateFileW.restype = ctypes.c_void_p

ReadFile = kernel32.ReadFile
ReadFile.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p
]
ReadFile.restype = ctypes.c_int

SetFilePointerEx = kernel32.SetFilePointerEx
SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_uint32]
SetFilePointerEx.restype = ctypes.c_int

CloseHandle = kernel32.CloseHandle
CloseHandle.argtypes = [ctypes.c_void_p]

def open_direct(path):
    h = CreateFileW(
        str(Path(path).resolve()),
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        None,
        OPEN_EXISTING,
        FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN,
        None
    )
    if h == INVALID_HANDLE_VALUE or not h:
        raise OSError(ctypes.get_last_error(), "CreateFileW(NO_BUFFERING) failed")
    return h

def direct_read_exact(h, dst, size):
    if not SetFilePointerEx(h, 0, None, FILE_BEGIN):
        raise OSError(ctypes.get_last_error(), "SetFilePointerEx failed")
    got = ctypes.c_uint32(0)
    if not ReadFile(h, dst, size, ctypes.byref(got), None):
        raise OSError(ctypes.get_last_error(), "ReadFile failed")
    if got.value != size:
        raise IOError(f"short direct read: {got.value} / {size}")

def bench_call(fn, repeats):
    # Warmup
    for _ in range(20):
        fn()
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        fn()
        vals.append(time.perf_counter_ns() - t0)
    vals.sort()
    return {
        "best_ns": vals[0],
        "median_ns": int(statistics.median(vals)),
        "p10_ns": vals[max(0, len(vals)//10 - 1)],
        "p90_ns": vals[min(len(vals)-1, (len(vals)*9)//10)]
    }

def report(label, ns):
    sec = ns / 1e9
    gw = NWEIGHTS / sec / 1e9
    equiv_gbs = gw * 0.5
    print(f"{label:<34} {ns/1e6:9.4f} ms   {gw:8.3f} Gweights/s   {equiv_gbs:7.3f} GB/s-Q4")

def main():
    if os.name != "nt" or ctypes.sizeof(ctypes.c_void_p) != 8:
        die("This V0 runner is Windows x64 only.")

    data = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_DATA.resolve()
    wpath = data / "weights_bitplanes_all.bin"
    qpath = data / "weights_q4_packed.bin"
    xpath = data / "activation_int8.npy"

    print("=" * 78)
    print("TRANSIT DIRECT ASSEMBLY BITPLANE TEST V1")
    print("=" * 78)
    print("Data:", data)

    for p in (wpath, qpath, xpath, ASM):
        if not p.exists():
            die(f"missing: {p}")

    if wpath.stat().st_size != WFILE_BYTES:
        die(f"weight bitplane file must be {WFILE_BYTES} bytes, got {wpath.stat().st_size}")

    assemble()

    x = np.load(xpath).astype(np.int8, copy=False)
    xp = make_activation_planes(x)

    wblob = wpath.read_bytes()
    qblob = qpath.read_bytes()[:NWEIGHTS//2]

    print("[verify] building exact signed INT4 reference...")
    q = unpack_q4(qblob).astype(np.int32)
    ref = q @ x.astype(np.int32)

    waddr = alloc_copy(wblob)
    xaddr = alloc_copy(xp.tobytes())
    outaddr = valloc(ROWS * 4, executable=False)

    kernel, kaddr, ksize = load_kernel()

    rc = kernel(waddr, xaddr, outaddr)
    if rc != 0:
        die(f"assembly kernel returned {rc}")

    out = np.ctypeslib.as_array(
        (ctypes.c_int32 * ROWS).from_address(outaddr)
    ).copy()

    exact = bool(np.array_equal(out.astype(np.int64), ref.astype(np.int64)))
    maxdiff = int(np.max(np.abs(out.astype(np.int64) - ref.astype(np.int64))))
    print(f"[verify] exact_int_accum = {exact}")
    print(f"[verify] max_abs_diff    = {maxdiff}")
    if not exact:
        idx = int(np.argmax(np.abs(out.astype(np.int64) - ref.astype(np.int64))))
        print(f"[verify] worst row={idx}: asm={out[idx]} ref={ref[idx]}")
        die("assembly output mismatch")

    print("\n--- PURE ASSEMBLY COMPUTE (weights already in RAM) ---")
    b = bench_call(lambda: kernel(waddr, xaddr, outaddr), repeats=2000)
    report("ASM best", b["best_ns"])
    report("ASM median", b["median_ns"])

    print("\n--- PHYSICAL SSD READ + PURE ASSEMBLY ---")
    # VirtualAlloc is naturally aligned enough for unbuffered I/O.
    h = open_direct(wpath)
    try:
        # warmup direct I/O
        for _ in range(10):
            direct_read_exact(h, waddr, WFILE_BYTES)
            kernel(waddr, xaddr, outaddr)

        vals = []
        read_vals = []
        asm_vals = []
        for _ in range(200):
            t0 = time.perf_counter_ns()
            direct_read_exact(h, waddr, WFILE_BYTES)
            t1 = time.perf_counter_ns()
            kernel(waddr, xaddr, outaddr)
            t2 = time.perf_counter_ns()
            read_vals.append(t1 - t0)
            asm_vals.append(t2 - t1)
            vals.append(t2 - t0)
        vals.sort(); read_vals.sort(); asm_vals.sort()

        report("SSD+ASM best", vals[0])
        report("SSD+ASM median", int(statistics.median(vals)))

        read_med = int(statistics.median(read_vals))
        asm_med = int(statistics.median(asm_vals))
        print(f"Direct-read median:              {read_med/1e6:9.4f} ms   "
              f"{WFILE_BYTES/(read_med/1e9)/1e9:7.3f} GB/s")
        print(f"ASM-inside-pipeline median:      {asm_med/1e6:9.4f} ms")
    finally:
        CloseHandle(h)

    print("\n--- RESULT BLOCK ---")
    print(f"exact={exact}")
    print(f"maxdiff={maxdiff}")
    print(f"asm_best_ms={b['best_ns']/1e6:.6f}")
    print(f"asm_median_ms={b['median_ns']/1e6:.6f}")
    print(f"asm_best_Gweights_s={NWEIGHTS/(b['best_ns']/1e9)/1e9:.6f}")
    print("Copy this whole console output back to ChatGPT.")

if __name__ == "__main__":
    main()
