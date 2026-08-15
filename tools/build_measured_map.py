#!/usr/bin/env python3
"""Build a direct measured TensorWave map from one Phase-2/3 benchmark run.

Unlike build_feasibility_map.py, this file does not extrapolate from one effective
TFLOPS number. Every plotted M point is taken directly from benchmark JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("m-*.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        g = data.get("geometry") or data.get("config") or {}
        ovl = data.get("overlapped") or {}
        m = int(g.get("m", 0))
        k = int(g.get("k", 0))
        n = int(g.get("n", 0))
        tiles = int(g.get("tiles", g.get("tile_count", 0)))
        gemm_ms = float(ovl.get("gemm_ms", ovl.get("compute_ms", 0.0)))
        tflops = 0.0
        if min(m, k, n, tiles) > 0 and gemm_ms > 0:
            tflops = (2.0 * m * k * n * tiles) / (gemm_ms / 1000.0) / 1.0e12
        physical_h2d = float(ovl.get("compressed_h2d_gbps", ovl.get("h2d_gbps", 0.0)))
        rows.append(
            {
                "file": path.name,
                "experiment": data.get("experiment", "unknown"),
                "m": m,
                "k": k,
                "n": n,
                "tiles": tiles,
                "physical_h2d_gbps": physical_h2d,
                "source_equivalent_h2d_gbps": float(ovl.get("source_equivalent_h2d_gbps", physical_h2d)),
                "effective_gemm_tflops": tflops,
                "dequant_ms": float(ovl.get("dequant_ms", 0.0)),
                "gemm_ms": gemm_ms,
                "compute_ms": float(ovl.get("compute_ms", gemm_ms)),
                "starvation_ms": float(ovl.get("starvation_ms", 0.0)),
                "starvation_pct": float(ovl.get("steady_starvation_pct", 0.0)),
                "hidden_transfer_pct": float(ovl.get("steady_hidden_transfer_pct", 0.0)),
                "wall_ms": float(ovl.get("wall_ms", 0.0)),
                "correctness_ok": bool(data.get("correctness_ok", False)),
                "verdict": data.get("verdict", ""),
            }
        )
    if not rows:
        raise ValueError(f"no m-*.json files under {run_dir}")
    return sorted(rows, key=lambda row: row["m"])


def classify(row: dict[str, Any]) -> str:
    if not row["correctness_ok"]:
        return "INVALID"
    if row["starvation_pct"] <= 10.0 and row["hidden_transfer_pct"] >= 80.0:
        return "STRONG"
    if row["starvation_pct"] <= 25.0:
        return "NEAR"
    return "TRANSFER_BOUND"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys()) + ["class"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["class"] = classify(row)
            writer.writerow(item)


def write_markdown(path: Path, run_dir: Path, rows: list[dict[str, Any]]) -> None:
    valid = [row for row in rows if row["correctness_ok"]]
    strong = [row for row in valid if classify(row) == "STRONG"]
    first_strong = min(strong, key=lambda row: row["m"]) if strong else None
    best = min(valid, key=lambda row: row["starvation_pct"]) if valid else None

    lines = [
        "# TensorWave Direct Measured Map",
        "",
        f"Source run: `{run_dir}`",
        "",
        "This map contains **observed points only**. No model-size extrapolation is used in the table below.",
        "",
    ]
    if first_strong:
        lines.extend(
            [
                f"First strong measured M: **{first_strong['m']}**",
                "",
            ]
        )
    else:
        lines.extend(["First strong measured M: **none in sweep**", ""])
    if best:
        lines.extend(
            [
                f"Lowest starvation: **{best['starvation_pct']:.3f}% at M={best['m']}**",
                "",
            ]
        )

    lines.extend(
        [
            "| M | K | N | tiles | physical H2D GB/s | source-eq GB/s | GEMM TFLOP/s | dequant ms | GEMM ms | starvation % | hidden % | wall ms | correct | class |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['m']} | {row['k']} | {row['n']} | {row['tiles']} | "
            f"{row['physical_h2d_gbps']:.3f} | {row['source_equivalent_h2d_gbps']:.3f} | "
            f"{row['effective_gemm_tflops']:.3f} | {row['dequant_ms']:.3f} | {row['gemm_ms']:.3f} | "
            f"{row['starvation_pct']:.3f} | {row['hidden_transfer_pct']:.3f} | {row['wall_ms']:.3f} | "
            f"{'yes' if row['correctness_ok'] else 'NO'} | {classify(row)} |"
        )

    lines.extend(
        [
            "",
            "`STRONG` = correctness passes, starvation <=10%, hidden transfer >=80%.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def point_color(row: dict[str, Any]) -> str:
    cls = classify(row)
    return {
        "INVALID": "#616161",
        "TRANSFER_BOUND": "#ef5350",
        "NEAR": "#ffc107",
        "STRONG": "#4caf50",
    }[cls]


def write_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 920, 440
    left, right, top, bottom = 75, 30, 55, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_m = max(row["m"] for row in rows)

    def x_of(m: int) -> float:
        # log2 scale makes M=1..2048 readable.
        import math
        max_log = max(1.0, math.log2(max_m))
        return left + plot_w * math.log2(max(1, m)) / max_log

    def y_of(starvation: float) -> float:
        return top + plot_h * (1.0 - max(0.0, min(100.0, starvation)) / 100.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="28" font-family="sans-serif" font-size="20" font-weight="bold">TensorWave measured starvation map</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#333"/>',
    ]
    for pct in [0, 10, 25, 50, 75, 100]:
        y = y_of(pct)
        parts.append(f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#ddd"/>')
        parts.append(f'<text x="{left-10}" y="{y+4}" text-anchor="end" font-family="monospace" font-size="10">{pct}%</text>')

    points = []
    for row in rows:
        x, y = x_of(row["m"]), y_of(row["starvation_pct"])
        points.append(f"{x:.1f},{y:.1f}")
    if len(points) >= 2:
        parts.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#1565c0" stroke-width="2"/>')

    for row in rows:
        x, y = x_of(row["m"]), y_of(row["starvation_pct"])
        parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{point_color(row)}" stroke="#222"/>')
        parts.append(f'<text x="{x}" y="{top+plot_h+22}" text-anchor="middle" font-family="monospace" font-size="10">{row["m"]}</text>')

    parts.append(f'<text x="{left+plot_w/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="12">M / activation rows (log2 position)</text>')
    parts.append(f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="12">steady starvation %</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build direct measured TensorWave map")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = load_rows(run_dir)

    write_csv(out / "measured-map.csv", rows)
    write_markdown(out / "MEASURED-MAP.md", run_dir, rows)
    write_svg(out / "measured-starvation.svg", rows)
    (out / "measured-map.json").write_text(
        json.dumps({"schema": "tensorwave.measured-map.v1", "source_run": str(run_dir), "points": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote direct measured map to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
