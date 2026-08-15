#!/usr/bin/env python3
"""Aggregate TensorWave Phase-4 run directories into one Markdown/CSV report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_result_files(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    config_path = run_dir / "run-config.json"
    config = {}
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))

    for path in sorted(run_dir.glob("m-*.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        g = data.get("geometry") or data.get("config") or {}
        ovl = data.get("overlapped") or {}
        seq = data.get("sequential") or {}
        rows.append(
            {
                "run_dir": str(run_dir),
                "source_model_dir": config.get("model_dir", ""),
                "experiment": data.get("experiment", "unknown"),
                "dtype": data.get("dtype", config.get("q4_format", "")),
                "m": int(g.get("m", 0)),
                "k": int(g.get("k", 0)),
                "n": int(g.get("n", 0)),
                "tiles": int(g.get("tiles", g.get("tile_count", 0))),
                "h2d_gbps": float(ovl.get("h2d_gbps", 0.0)),
                "starvation_pct": float(ovl.get("steady_starvation_pct", 0.0)),
                "hidden_pct": float(ovl.get("steady_hidden_transfer_pct", 0.0)),
                "wall_ms": float(ovl.get("wall_ms", 0.0)),
                "sequential_wall_ms": float(seq.get("wall_ms", 0.0)),
                "gemm_ms": float(ovl.get("gemm_ms", ovl.get("compute_ms", 0.0))),
                "dequant_ms": float(ovl.get("dequant_ms", 0.0)),
                "correctness_ok": bool(data.get("correctness_ok", False)),
                "speedup": float(data.get("speedup", 0.0)),
            }
        )
    return rows


def classify(row: dict[str, Any]) -> str:
    if not row["correctness_ok"]:
        return "INVALID"
    if row["starvation_pct"] <= 10.0 and row["hidden_pct"] >= 80.0:
        return "STRONG"
    if row["starvation_pct"] <= 25.0:
        return "NEAR"
    return "TRANSFER_BOUND"


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate TensorWave feasibility runs")
    parser.add_argument("--run-dir", action="append", type=Path, required=True, help="Repeat for each Phase-2/3 run directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for run_dir in args.run_dir:
        rows.extend(load_result_files(run_dir.resolve()))
    if not rows:
        raise ValueError("no m-*.json benchmark results found")

    for row in rows:
        row["class"] = classify(row)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    fields = list(rows[0].keys())
    with (out / "feasibility-runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    correct = [row for row in rows if row["correctness_ok"]]
    strong = [row for row in correct if row["class"] == "STRONG"]
    best = min(correct, key=lambda r: r["starvation_pct"]) if correct else None

    lines = [
        "# TensorWave Phase-4 measured feasibility aggregation",
        "",
        f"Result points: **{len(rows)}**",
        f"Correctness-passing points: **{len(correct)}**",
        f"Strong points: **{len(strong)}**",
        "",
    ]
    if best:
        lines.extend(
            [
                "## Lowest starvation point",
                "",
                f"- experiment: `{best['experiment']}`",
                f"- M/K/N: `{best['m']}/{best['k']}/{best['n']}`",
                f"- tiles: `{best['tiles']}`",
                f"- H2D: **{best['h2d_gbps']:.3f} GB/s**",
                f"- starvation: **{best['starvation_pct']:.3f}%**",
                f"- hidden transfer: **{best['hidden_pct']:.3f}%**",
                f"- wall: **{best['wall_ms']:.3f} ms**",
                "",
            ]
        )

    lines.extend(
        [
            "## All points",
            "",
            "| exp | M | K | N | tiles | H2D GB/s | starvation % | hidden % | gemm ms | dequant ms | wall ms | correct | class |",
            "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---|",
        ]
    )
    for row in sorted(rows, key=lambda r: (r["experiment"], r["n"], r["m"])):
        lines.append(
            f"| {row['experiment']} | {row['m']} | {row['k']} | {row['n']} | {row['tiles']} | "
            f"{row['h2d_gbps']:.3f} | {row['starvation_pct']:.3f} | {row['hidden_pct']:.3f} | "
            f"{row['gemm_ms']:.3f} | {row['dequant_ms']:.3f} | {row['wall_ms']:.3f} | "
            f"{'yes' if row['correctness_ok'] else 'NO'} | {row['class']} |"
        )

    lines.extend(
        [
            "",
            "`STRONG` means correctness passes, starvation <=10%, and hidden transfer >=80%.",
            "",
        ]
    )
    (out / "MEASURED-FEASIBILITY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Aggregated {len(rows)} points into {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
