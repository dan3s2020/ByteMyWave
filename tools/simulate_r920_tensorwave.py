#!/usr/bin/env python3
"""TensorWave R920 + RTX 3060 analytical/discrete-event simulator.

This is NOT a hardware benchmark. It mirrors the contracts already present in
TensorWave Phase 3 and Phase 4 so that the proposed R920 topology can be
reasoned about before the physical server exists.

Phase 3 mirrored here:
- Q4_SYM_G32_F32S = 20 bytes / 32 weights = 0.625 B/parameter
- two compressed VRAM slots
- slot(i) = i % 2
- copy(i) waits compute(i-2) before reusing the same slot
- compute(i) waits copy(i)
- one copy stream + one compute stream
- one reusable FP16 dequantized tile

Phase 4 mirrored here:
- T_transfer = P*b*(1-r)/BW
- T_compute = 2*P*M/F
- M_cross = b*(1-r)*F/(2*BW)

Defaults deliberately use analytical assumptions, not measurements:
- effective pinned H2D = 12 GB/s
- effective dense-linear compute = 10 TFLOP/s
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

Q4_GROUP_SIZE = 32
Q4_GROUP_BYTES = 20
Q4_BYTES_PER_PARAM = Q4_GROUP_BYTES / Q4_GROUP_SIZE


@dataclass(frozen=True)
class R920:
    sockets: int = 4
    dimm_slots_total: int = 96
    dimm_slots_per_socket: int = 24
    memory_channels_per_cpu: int = 4
    max_memory_bandwidth_gbps_per_cpu: float = 85.0
    pcie_generation: int = 3
    x16_slots_cpu2: int = 2
    x16_slots_cpu3: int = 2
    x16_slots_cpu4: int = 2
    x8_slots_cpu1: int = 3


@dataclass(frozen=True)
class CPU:
    name: str = "Intel Xeon E7-4890 v2"
    cores: int = 15
    threads: int = 30
    base_ghz: float = 2.8
    turbo_ghz: float = 3.4
    tdp_w: int = 155
    pcie_lanes: int = 32
    memory_channels: int = 4
    max_memory_bandwidth_gbps: float = 85.0


@dataclass(frozen=True)
class GPU:
    name: str = "GeForce RTX 3060 12GB"
    vram_gib: float = 12.0
    cuda_cores: int = 3584
    board_power_w: int = 170
    assumed_effective_h2d_gbps: float = 12.0
    assumed_effective_tflops: float = 10.0


@dataclass(frozen=True)
class RooflineCell:
    m: int
    stream_gb_per_step: float
    transfer_ms: float
    compute_ms: float
    ideal_step_ms: float
    unhidden_transfer_ms: float
    hidden_transfer_pct: float
    starvation_pct_lower_bound: float
    rows_per_s: float
    regime: str


@dataclass(frozen=True)
class RingCell:
    m: int
    q4_tile_mib: float
    fp16_tile_mib: float
    fixed_vram_mib: float
    copy_ms_per_tile: float
    compute_ms_per_tile: float
    wall_ms_for_tiles: float
    steady_starvation_ms: float
    steady_starvation_pct: float
    hidden_transfer_pct: float


def classify(hidden_pct: float, compute_ms: float, transfer_ms: float) -> str:
    if compute_ms >= transfer_ms:
        return "COMPUTE_BOUND"
    if hidden_pct >= 80.0:
        return "NEAR_BALANCED"
    if hidden_pct >= 50.0:
        return "BALANCED"
    return "TRANSFER_BOUND"


def crossover_m(*, bytes_per_param: float, resident_fraction: float,
                h2d_gbps: float, effective_tflops: float) -> float:
    return (
        bytes_per_param
        * (1.0 - resident_fraction)
        * effective_tflops
        * 1.0e12
        / (2.0 * h2d_gbps * 1.0e9)
    )


def roofline(*, model_b: float, m: int, bytes_per_param: float,
             resident_fraction: float, h2d_gbps: float,
             effective_tflops: float, active_fraction: float = 1.0) -> RooflineCell:
    active_params = model_b * 1.0e9 * active_fraction
    stream_bytes = active_params * bytes_per_param * (1.0 - resident_fraction)
    transfer_s = stream_bytes / (h2d_gbps * 1.0e9)
    compute_s = 2.0 * active_params * m / (effective_tflops * 1.0e12)
    ideal_s = max(transfer_s, compute_s)
    unhidden_s = max(0.0, transfer_s - compute_s)
    hidden = 100.0 if transfer_s == 0 else min(100.0, 100.0 * compute_s / transfer_s)
    active_s = compute_s + unhidden_s
    starvation = 0.0 if active_s == 0 else 100.0 * unhidden_s / active_s
    return RooflineCell(
        m=m,
        stream_gb_per_step=stream_bytes / 1.0e9,
        transfer_ms=transfer_s * 1000.0,
        compute_ms=compute_s * 1000.0,
        ideal_step_ms=ideal_s * 1000.0,
        unhidden_transfer_ms=unhidden_s * 1000.0,
        hidden_transfer_pct=hidden,
        starvation_pct_lower_bound=starvation,
        rows_per_s=0.0 if ideal_s == 0 else m / ideal_s,
        regime=classify(hidden, compute_s * 1000.0, transfer_s * 1000.0),
    )


def ring_simulation(*, m: int, k: int, n: int, tiles: int,
                    h2d_gbps: float, effective_tflops: float,
                    dequant_us_per_tile: float) -> RingCell:
    elements = k * n
    if elements % Q4_GROUP_SIZE:
        raise ValueError("K*N must be divisible by Q4 group size 32")
    if tiles < 2:
        raise ValueError("tiles must be >= 2")

    q4_tile_bytes = elements * Q4_BYTES_PER_PARAM
    fp16_tile_bytes = elements * 2.0
    x_bytes = m * k * 2.0
    y_bytes = m * n * 4.0
    fixed_vram_bytes = x_bytes + 2.0 * q4_tile_bytes + fp16_tile_bytes + y_bytes

    copy_ms = q4_tile_bytes / (h2d_gbps * 1.0e9) * 1000.0
    gemm_ms = 2.0 * m * k * n / (effective_tflops * 1.0e12) * 1000.0
    compute_ms = gemm_ms + dequant_us_per_tile / 1000.0

    copy_end: list[float] = []
    compute_start: list[float] = []
    compute_end: list[float] = []

    for i in range(tiles):
        # One copy stream means copies serialize.
        start_copy = copy_end[-1] if copy_end else 0.0
        # Exact Phase-3 two-slot ownership: wait before overwriting slot i%2.
        if i >= 2:
            start_copy = max(start_copy, compute_end[i - 2])
        end_copy = start_copy + copy_ms

        # One compute stream, and compute(i) waits for copy(i).
        start_compute = max(compute_end[-1] if compute_end else 0.0, end_copy)
        end_compute = start_compute + compute_ms

        copy_end.append(end_copy)
        compute_start.append(start_compute)
        compute_end.append(end_compute)

    starvation_ms = sum(
        max(0.0, compute_start[i] - compute_end[i - 1])
        for i in range(1, tiles)
    )
    steady_copy_ms = copy_ms * (tiles - 1)
    hidden = max(0.0, min(100.0, 100.0 * (1.0 - starvation_ms / steady_copy_ms)))
    active_ms = tiles * compute_ms + starvation_ms
    starvation_pct = 0.0 if active_ms == 0 else 100.0 * starvation_ms / active_ms

    return RingCell(
        m=m,
        q4_tile_mib=q4_tile_bytes / 1024.0**2,
        fp16_tile_mib=fp16_tile_bytes / 1024.0**2,
        fixed_vram_mib=fixed_vram_bytes / 1024.0**2,
        copy_ms_per_tile=copy_ms,
        compute_ms_per_tile=compute_ms,
        wall_ms_for_tiles=compute_end[-1],
        steady_starvation_ms=starvation_ms,
        steady_starvation_pct=starvation_pct,
        hidden_transfer_pct=hidden,
    )


def parse_int_csv(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def build_report(args: argparse.Namespace) -> dict:
    r920 = R920()
    cpu = CPU()
    gpu = GPU(
        assumed_effective_h2d_gbps=args.h2d_gbps,
        assumed_effective_tflops=args.effective_tflops,
    )
    m_values = parse_int_csv(args.m_values)
    gpu_counts = parse_int_csv(args.gpu_counts)

    model_wire_bytes = args.model_b * 1.0e9 * Q4_BYTES_PER_PARAM
    cache_bytes = args.cache_gib_per_gpu * 1024.0**3
    resident_fraction = min(1.0, cache_bytes / model_wire_bytes)
    local_ram_gib = args.host_ram_gib / r920.sockets

    roof = [
        roofline(
            model_b=args.model_b,
            m=m,
            bytes_per_param=Q4_BYTES_PER_PARAM,
            resident_fraction=resident_fraction,
            h2d_gbps=args.h2d_gbps,
            effective_tflops=args.effective_tflops,
        )
        for m in m_values
    ]
    ring = [
        ring_simulation(
            m=m,
            k=args.k,
            n=args.n,
            tiles=args.tiles,
            h2d_gbps=args.h2d_gbps,
            effective_tflops=args.effective_tflops,
            dequant_us_per_tile=args.dequant_us_per_tile,
        )
        for m in m_values
    ]

    multi_gpu = []
    for count in gpu_counts:
        for cell in roof:
            multi_gpu.append({
                "gpus": count,
                "m": cell.m,
                "replicated_service_aggregate_rows_per_s": cell.rows_per_s * count,
                "ideal_equal_shard_step_ms_excluding_collectives": cell.ideal_step_ms / count,
                "ideal_equal_shard_rows_per_s_excluding_collectives": (
                    0.0 if cell.ideal_step_ms == 0 else cell.m / (cell.ideal_step_ms / count / 1000.0)
                ),
                "warning": "equal-shard mode is an optimistic lower bound; current TensorWave code does not implement it",
            })

    return {
        "schema": "tensorwave.r920-simulation.v1",
        "warning": "analytical/discrete-event simulation; replace timing inputs with measured Phase-4 calibration",
        "hardware": {
            "r920": asdict(r920),
            "cpu": asdict(cpu),
            "cpu_count": 4,
            "total_cpu_cores": cpu.cores * 4,
            "total_cpu_threads": cpu.threads * 4,
            "host_ram_gib": args.host_ram_gib,
            "local_ram_gib_per_socket_if_balanced": local_ram_gib,
            "gpu": asdict(gpu),
            "one_gpu_h2d_as_pct_of_one_cpu_max_memory_bw": 100.0 * args.h2d_gbps / cpu.max_memory_bandwidth_gbps,
            "two_gpu_h2d_as_pct_of_one_cpu_max_memory_bw": 200.0 * args.h2d_gbps / cpu.max_memory_bandwidth_gbps,
        },
        "workload": {
            "model_billion_params": args.model_b,
            "q4_bytes_per_parameter": Q4_BYTES_PER_PARAM,
            "q4_wire_gb": model_wire_bytes / 1.0e9,
            "cache_gib_per_gpu": args.cache_gib_per_gpu,
            "resident_fraction_from_cache": resident_fraction,
            "k": args.k,
            "n": args.n,
            "tiles": args.tiles,
            "dequant_us_per_tile": args.dequant_us_per_tile,
        },
        "crossover_m": crossover_m(
            bytes_per_param=Q4_BYTES_PER_PARAM,
            resident_fraction=resident_fraction,
            h2d_gbps=args.h2d_gbps,
            effective_tflops=args.effective_tflops,
        ),
        "roofline": [asdict(x) for x in roof],
        "phase3_ring_simulation": [asdict(x) for x in ring],
        "multi_gpu": multi_gpu,
        "observations": [
            "1 TiB host RAM removes model-capacity pressure but not repeated dense H2D traffic.",
            "The current Phase-3 transient ring is tiny versus 12 GiB, so unused VRAM is a strong persistent-cache candidate.",
            "NUMA locality should become part of the Weight Atlas/execution plan.",
            "Current code scales most safely as one independent worker per GPU before model sharding.",
            "Six electrical x16 links must not be interpreted as six stock-chassis RTX 3060 positions.",
        ],
    }


def write_markdown(path: Path, report: dict) -> None:
    hw = report["hardware"]
    w = report["workload"]
    lines = [
        "# R920 + RTX 3060 — TensorWave simulation result",
        "",
        "> **Simulation, not measurement.** Replace H2D/TFLOPS assumptions with real Phase-4 calibration.",
        "",
        "## Configuration",
        "",
        f"- server: Dell PowerEdge R920, {hw['cpu_count']} sockets",
        f"- CPU: {hw['cpu']['name']} × {hw['cpu_count']} ({hw['total_cpu_cores']} cores / {hw['total_cpu_threads']} threads)",
        f"- RAM: {hw['host_ram_gib']:.0f} GiB total, {hw['local_ram_gib_per_socket_if_balanced']:.0f} GiB/socket balanced",
        f"- GPU profile: {hw['gpu']['name']}",
        f"- assumed H2D: {hw['gpu']['assumed_effective_h2d_gbps']:.2f} GB/s",
        f"- assumed effective compute: {hw['gpu']['assumed_effective_tflops']:.2f} TFLOP/s",
        f"- dense reference model: {w['model_billion_params']:.0f}B",
        f"- Q4 wire model: {w['q4_wire_gb']:.2f} GB",
        f"- cache: {w['cache_gib_per_gpu']:.2f} GiB/GPU = {w['resident_fraction_from_cache']*100:.2f}% residency",
        f"- tile geometry: K={w['k']}, N={w['n']}, tiles={w['tiles']}",
        "",
        f"Predicted crossover: **M ≈ {report['crossover_m']:.1f}**.",
        "",
        "## Dense roofline",
        "",
        "| M | stream GB | H2D ms | compute ms | hidden % | starvation % | rows/s | regime |",
        "|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for c in report["roofline"]:
        lines.append(
            f"| {c['m']} | {c['stream_gb_per_step']:.3f} | {c['transfer_ms']:.2f} | {c['compute_ms']:.2f} | "
            f"{c['hidden_transfer_pct']:.2f} | {c['starvation_pct_lower_bound']:.2f} | {c['rows_per_s']:.2f} | {c['regime']} |"
        )
    lines += [
        "",
        "## Phase-3 two-slot ring",
        "",
        "| M | Q4 tile MiB | fixed VRAM MiB | copy ms/tile | compute ms/tile | wall ms | starvation % | hidden % |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in report["phase3_ring_simulation"]:
        lines.append(
            f"| {c['m']} | {c['q4_tile_mib']:.3f} | {c['fixed_vram_mib']:.3f} | {c['copy_ms_per_tile']:.4f} | "
            f"{c['compute_ms_per_tile']:.4f} | {c['wall_ms_for_tiles']:.3f} | {c['steady_starvation_pct']:.2f} | {c['hidden_transfer_pct']:.2f} |"
        )
    lines += [
        "",
        "## Multi-GPU interpretation",
        "",
        "Current-code-compatible scaling is **one independent TensorWave worker per GPU**, each with NUMA-local pinned RAM. The JSON also contains an ideal equal-model-shard lower bound, explicitly excluding collectives/activation exchange and explicitly not implemented.",
        "",
        "## Observations / proposals",
        "",
    ]
    lines += [f"{i+1}. {text}" for i, text in enumerate(report["observations"])]
    lines += [
        "6. Add explicit GPU↔NUMA affinity and topology telemetry.",
        "7. Use otherwise-free RTX 3060 VRAM for a persistent compressed hot-weight/expert cache.",
        "8. Measure local versus remote pinned H2D on the actual R920 before trusting multi-socket placement.",
        "9. Validate one RTX 3060 fit, 8-pin power path, thermals and negotiated PCIe link before buying multiple cards.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Simulate TensorWave on R920 + RTX 3060")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-b", type=float, default=70.0)
    p.add_argument("--host-ram-gib", type=float, default=1024.0)
    p.add_argument("--h2d-gbps", type=float, default=12.0)
    p.add_argument("--effective-tflops", type=float, default=10.0)
    p.add_argument("--cache-gib-per-gpu", type=float, default=0.0)
    p.add_argument("--k", type=int, default=8192)
    p.add_argument("--n", type=int, default=256)
    p.add_argument("--tiles", type=int, default=32)
    p.add_argument("--m-values", default="1,4,16,64,128,256,512,1024,2048")
    p.add_argument("--gpu-counts", default="1,2,3")
    p.add_argument("--dequant-us-per-tile", type=float, default=0.0)
    return p


def validate(args: argparse.Namespace) -> None:
    if args.model_b <= 0 or args.host_ram_gib <= 0 or args.h2d_gbps <= 0 or args.effective_tflops <= 0:
        raise ValueError("model/RAM/H2D/TFLOPS must be > 0")
    if args.cache_gib_per_gpu < 0 or args.dequant_us_per_tile < 0:
        raise ValueError("cache/dequant values must be >= 0")
    if args.k <= 0 or args.n <= 0 or args.tiles < 2:
        raise ValueError("K/N must be > 0 and tiles >= 2")


def main() -> int:
    args = parser().parse_args()
    validate(args)
    report = build_report(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "simulation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "SIMULATION-REPORT.md", report)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "crossover_m": report["crossover_m"],
        "model_q4_gb": report["workload"]["q4_wire_gb"],
        "local_ram_gib_per_socket": report["hardware"]["local_ram_gib_per_socket_if_balanced"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
