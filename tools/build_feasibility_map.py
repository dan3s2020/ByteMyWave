#!/usr/bin/env python3
"""Build the TensorWave analytical feasibility map.

This is deliberately a roofline-style lower-bound model, not a claim of observed
end-to-end LLM performance. Feed it *measured effective* H2D GB/s and GEMM TFLOP/s
from TensorWave runs whenever available.

Outputs:
  feasibility-map.json
  feasibility-map.csv
  FEASIBILITY-MAP.md
  feasibility-map.svg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_MODELS_B = [7.0, 13.0, 33.0, 70.0, 120.0]
DEFAULT_M = [1, 4, 16, 64, 128, 256, 512, 1024, 2048]


@dataclass(frozen=True)
class Inputs:
    pcie_gbps: float
    effective_tflops: float
    bytes_per_param: float
    resident_fraction: float
    active_fraction: float


@dataclass(frozen=True)
class Cell:
    model_b: float
    m: int
    active_params_b: float
    stream_gb: float
    transfer_ms: float
    compute_ms: float
    ideal_step_ms: float
    unhidden_transfer_ms: float
    hidden_transfer_pct: float
    starvation_pct_lower_bound: float
    aggregate_rows_per_s: float
    regime: str


def parse_csv_numbers(value: str, cast=float) -> list:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("list must not be empty")
    try:
        return [cast(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def crossover_m(inputs: Inputs) -> float:
    # 2*P*M/F = P*b*(1-r)/BW ; P cancels.
    numerator = (
        inputs.bytes_per_param
        * (1.0 - inputs.resident_fraction)
        * inputs.effective_tflops
        * 1.0e12
    )
    denominator = 2.0 * inputs.pcie_gbps * 1.0e9
    return numerator / denominator


def classify(hidden_pct: float, compute_ms: float, transfer_ms: float) -> str:
    if compute_ms >= transfer_ms:
        return "COMPUTE_BOUND"
    if hidden_pct >= 80.0:
        return "NEAR_BALANCED"
    if hidden_pct >= 50.0:
        return "BALANCED"
    return "TRANSFER_BOUND"


def make_cell(model_b: float, m: int, inputs: Inputs) -> Cell:
    params = model_b * 1.0e9
    active_params = params * inputs.active_fraction
    stream_bytes = (
        active_params
        * inputs.bytes_per_param
        * (1.0 - inputs.resident_fraction)
    )
    transfer_s = stream_bytes / (inputs.pcie_gbps * 1.0e9)

    # Dense-linear approximation: one multiply-add ~= 2 FLOPs per active
    # parameter and activation row. Attention/KV/dequant/launch costs are not
    # included here and must be measured separately.
    compute_flops = 2.0 * active_params * m
    compute_s = compute_flops / (inputs.effective_tflops * 1.0e12)

    ideal_step_s = max(transfer_s, compute_s)
    unhidden_s = max(0.0, transfer_s - compute_s)
    hidden_pct = 100.0 if transfer_s == 0 else min(100.0, 100.0 * compute_s / transfer_s)
    starvation_pct = 0.0
    if compute_s + unhidden_s > 0:
        starvation_pct = 100.0 * unhidden_s / (compute_s + unhidden_s)

    rows_s = 0.0 if ideal_step_s == 0 else m / ideal_step_s

    return Cell(
        model_b=model_b,
        m=m,
        active_params_b=active_params / 1.0e9,
        stream_gb=stream_bytes / 1.0e9,
        transfer_ms=transfer_s * 1000.0,
        compute_ms=compute_s * 1000.0,
        ideal_step_ms=ideal_step_s * 1000.0,
        unhidden_transfer_ms=unhidden_s * 1000.0,
        hidden_transfer_pct=hidden_pct,
        starvation_pct_lower_bound=starvation_pct,
        aggregate_rows_per_s=rows_s,
        regime=classify(hidden_pct, compute_s * 1000.0, transfer_s * 1000.0),
    )


def generate_cells(models_b: Iterable[float], m_values: Iterable[int], inputs: Inputs) -> list[Cell]:
    return [make_cell(model_b, m, inputs) for model_b in models_b for m in m_values]


def write_json(path: Path, inputs: Inputs, cells: list[Cell]) -> None:
    payload = {
        "schema": "tensorwave.feasibility-map.v1",
        "model": "ideal-overlap-roofline",
        "warning": (
            "Analytical lower bound only. Replace input bandwidth/TFLOPS with measured effective values. "
            "Does not include attention, KV-cache traffic, dequant overhead, launch/synchronization, or graph irregularity."
        ),
        "inputs": asdict(inputs),
        "crossover_m": crossover_m(inputs),
        "cells": [asdict(cell) for cell in cells],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, cells: list[Cell]) -> None:
    fieldnames = list(asdict(cells[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for cell in cells:
            writer.writerow(asdict(cell))


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def write_markdown(path: Path, inputs: Inputs, models_b: list[float], m_values: list[int], cells: list[Cell]) -> None:
    by_key = {(cell.model_b, cell.m): cell for cell in cells}
    cross = crossover_m(inputs)

    lines = [
        "# TensorWave Feasibility Map",
        "",
        "> Analytical roofline-style map. This is not an observed end-to-end benchmark.",
        "",
        "## Inputs",
        "",
        f"- effective H2D bandwidth: **{fmt(inputs.pcie_gbps)} GB/s**",
        f"- effective GPU dense-linear throughput: **{fmt(inputs.effective_tflops)} TFLOP/s**",
        f"- wire representation: **{fmt(inputs.bytes_per_param, 4)} bytes/parameter**",
        f"- persistent resident fraction: **{fmt(inputs.resident_fraction * 100)}%**",
        f"- active parameter fraction: **{fmt(inputs.active_fraction * 100)}%**",
        f"- predicted compute/transfer crossover: **M ~= {fmt(cross, 1)}**",
        "",
        "The crossover follows:",
        "",
        "```text",
        "M_cross = bytes_per_param * (1-resident_fraction) * effective_FLOPS",
        "          ---------------------------------------------------------",
        "                         2 * H2D_bandwidth",
        "```",
        "",
        "`active_fraction` changes absolute bytes/time but cancels from the ideal crossover when compute and streamed active weights scale together.",
        "",
        "## Regime map",
        "",
    ]

    header = "| model | " + " | ".join(f"M={m}" for m in m_values) + " |"
    sep = "|---:|" + "|".join("---:" for _ in m_values) + "|"
    lines.extend([header, sep])
    for model_b in models_b:
        values = []
        for m in m_values:
            cell = by_key[(model_b, m)]
            label = {
                "COMPUTE_BOUND": "C",
                "NEAR_BALANCED": "N",
                "BALANCED": "B",
                "TRANSFER_BOUND": "T",
            }[cell.regime]
            values.append(f"{label} {cell.hidden_transfer_pct:.0f}%")
        lines.append(f"| {model_b:g}B | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "Legend: `T` transfer-bound, `B` balanced, `N` near-balanced, `C` compute-bound. Percentage is the ideal fraction of transfer hideable by dense-linear compute.",
            "",
            "## Absolute lower bounds",
            "",
            "The following values assume perfect overlap and therefore represent optimistic lower bounds. `rows/s` is useful for comparing batch/prefill reuse; it is not automatically single-user decode tokens/s.",
            "",
            "| model | M | stream GB/step | H2D ms | compute ms | ideal step ms | unhidden ms | rows/s | regime |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
        ]
    )
    for cell in cells:
        lines.append(
            f"| {cell.model_b:g}B | {cell.m} | {cell.stream_gb:.3f} | {cell.transfer_ms:.2f} | "
            f"{cell.compute_ms:.2f} | {cell.ideal_step_ms:.2f} | {cell.unhidden_transfer_ms:.2f} | "
            f"{cell.aggregate_rows_per_s:.2f} | {cell.regime} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation rules",
            "",
            "1. Batch=1 dense decode (`M=1`) is expected to be the worst TensorWave regime.",
            "2. Prefill and batched serving move rightward on the map because the same weight tile is reused across more activation rows.",
            "3. MoE reduces absolute streamed bytes if only active experts are loaded, but routing/cache behavior must be modeled explicitly.",
            "4. Persistent hot-weight residency reduces the crossover in direct proportion to `(1-resident_fraction)`.",
            "5. A faster GPU with unchanged PCIe requires larger `M` to hide transfer because it consumes each tile sooner.",
            "6. Real measurements supersede this map. Calibrate with `tools/calibrate_feasibility_map.py`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def regime_color(hidden_pct: float, compute_ms: float, transfer_ms: float) -> str:
    if compute_ms >= transfer_ms:
        return "#4caf50"
    if hidden_pct >= 80.0:
        return "#8bc34a"
    if hidden_pct >= 50.0:
        return "#ffc107"
    return "#ef5350"


def write_svg(path: Path, inputs: Inputs, models_b: list[float], m_values: list[int], cells: list[Cell]) -> None:
    by_key = {(cell.model_b, cell.m): cell for cell in cells}
    left = 90
    top = 95
    cell_w = 92
    cell_h = 58
    width = left + cell_w * len(m_values) + 30
    height = top + cell_h * len(models_b) + 90

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="20" font-weight="bold">TensorWave Feasibility Map</text>',
        f'<text x="20" y="52" font-family="sans-serif" font-size="12">H2D {inputs.pcie_gbps:g} GB/s · effective compute {inputs.effective_tflops:g} TFLOP/s · {inputs.bytes_per_param:g} B/param · crossover M≈{crossover_m(inputs):.1f}</text>',
        '<text x="20" y="70" font-family="sans-serif" font-size="11">Cell text = ideal hidden-transfer %. Green means compute can hide transfer; red means strongly transfer-bound.</text>',
    ]

    for col, m in enumerate(m_values):
        x = left + col * cell_w + cell_w / 2
        parts.append(f'<text x="{x}" y="{top - 12}" text-anchor="middle" font-family="monospace" font-size="11">M={m}</text>')

    for row, model_b in enumerate(models_b):
        y = top + row * cell_h
        parts.append(f'<text x="{left - 12}" y="{y + cell_h/2 + 4}" text-anchor="end" font-family="sans-serif" font-size="12">{model_b:g}B</text>')
        for col, m in enumerate(m_values):
            cell = by_key[(model_b, m)]
            x = left + col * cell_w
            color = regime_color(cell.hidden_transfer_pct, cell.compute_ms, cell.transfer_ms)
            parts.append(f'<rect x="{x}" y="{y}" width="{cell_w-3}" height="{cell_h-3}" rx="4" fill="{color}" stroke="#333" stroke-width="0.5"/>')
            parts.append(f'<text x="{x + (cell_w-3)/2}" y="{y + 24}" text-anchor="middle" font-family="monospace" font-size="13" font-weight="bold">{cell.hidden_transfer_pct:.0f}%</text>')
            parts.append(f'<text x="{x + (cell_w-3)/2}" y="{y + 42}" text-anchor="middle" font-family="monospace" font-size="9">{cell.ideal_step_ms:.0f} ms</text>')

    legend_y = top + len(models_b) * cell_h + 26
    legend = [
        ("#ef5350", "transfer-bound"),
        ("#ffc107", "balanced"),
        ("#8bc34a", "near-balanced"),
        ("#4caf50", "compute-bound"),
    ]
    x = 20
    for color, label in legend:
        parts.append(f'<rect x="{x}" y="{legend_y}" width="14" height="14" fill="{color}" stroke="#333"/>')
        parts.append(f'<text x="{x+20}" y="{legend_y+12}" font-family="sans-serif" font-size="11">{label}</text>')
        x += 145

    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build TensorWave analytical feasibility map")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pcie-gbps", type=float, default=12.0, help="Measured effective H2D GB/s")
    parser.add_argument("--effective-tflops", type=float, default=10.0, help="Measured effective dense-linear TFLOP/s")
    parser.add_argument("--bytes-per-param", type=float, default=0.625, help="Physical wire bytes per active parameter")
    parser.add_argument("--resident-fraction", type=float, default=0.0, help="Fraction of active weights persistently resident in VRAM [0,1]")
    parser.add_argument("--active-fraction", type=float, default=1.0, help="Fraction of total model parameters active/streamed per step [0,1]")
    parser.add_argument("--models-b", default=",".join(str(v) for v in DEFAULT_MODELS_B))
    parser.add_argument("--m-values", default=",".join(str(v) for v in DEFAULT_M))
    return parser


def validate(inputs: Inputs, models_b: list[float], m_values: list[int]) -> None:
    if inputs.pcie_gbps <= 0 or inputs.effective_tflops <= 0 or inputs.bytes_per_param <= 0:
        raise ValueError("pcie-gbps, effective-tflops and bytes-per-param must be > 0")
    if not 0.0 <= inputs.resident_fraction < 1.0:
        raise ValueError("resident-fraction must be >= 0 and < 1")
    if not 0.0 < inputs.active_fraction <= 1.0:
        raise ValueError("active-fraction must be > 0 and <= 1")
    if any(model <= 0 for model in models_b):
        raise ValueError("all model sizes must be > 0")
    if any(m <= 0 for m in m_values):
        raise ValueError("all M values must be > 0")


def main() -> int:
    args = build_parser().parse_args()
    models_b = parse_csv_numbers(args.models_b, float)
    m_values = parse_csv_numbers(args.m_values, int)
    inputs = Inputs(
        pcie_gbps=args.pcie_gbps,
        effective_tflops=args.effective_tflops,
        bytes_per_param=args.bytes_per_param,
        resident_fraction=clamp01(args.resident_fraction),
        active_fraction=clamp01(args.active_fraction),
    )
    validate(inputs, models_b, m_values)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cells = generate_cells(models_b, m_values, inputs)

    write_json(output_dir / "feasibility-map.json", inputs, cells)
    write_csv(output_dir / "feasibility-map.csv", cells)
    write_markdown(output_dir / "FEASIBILITY-MAP.md", inputs, models_b, m_values, cells)
    write_svg(output_dir / "feasibility-map.svg", inputs, models_b, m_values, cells)

    print(f"TensorWave crossover M ~= {crossover_m(inputs):.2f}")
    print(f"Wrote map to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
