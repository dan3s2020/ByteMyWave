#!/usr/bin/env python3
"""Summarize TensorWave Phase-1/Phase-2 JSON sweeps into Markdown + CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize TensorWave m-*.json results.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--csv", dest="csv_path", type=Path)
    return parser.parse_args()


def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.glob("m-*.json")):
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        geometry = data.get("geometry") or data.get("config") or {}
        sequential = data.get("sequential") or {}
        overlapped = data.get("overlapped") or {}

        starvation = float(overlapped.get("steady_starvation_pct", 0.0))
        hidden = float(overlapped.get("steady_hidden_transfer_pct", 0.0))
        correctness = bool(data.get("correctness_ok", False))
        supported = correctness and starvation <= 10.0 and hidden >= 80.0

        rows.append(
            {
                "file": path.name,
                "experiment": data.get("experiment", "unknown"),
                "dtype": data.get("dtype", "FP16-synthetic"),
                "m": int(geometry.get("m", 0)),
                "k": int(geometry.get("k", 0)),
                "n": int(geometry.get("n", 0)),
                "tiles": int(geometry.get("tiles", geometry.get("tile_count", 0))),
                "seq_wall_ms": float(sequential.get("wall_ms", 0.0)),
                "ovl_wall_ms": float(overlapped.get("wall_ms", 0.0)),
                "h2d_gbps": float(overlapped.get("h2d_gbps", 0.0)),
                "compute_ms": float(overlapped.get("compute_ms", 0.0)),
                "starvation_ms": float(overlapped.get("starvation_ms", 0.0)),
                "starvation_pct": starvation,
                "hidden_pct": hidden,
                "speedup": float(data.get("speedup", 0.0)),
                "correctness": correctness,
                "supported": supported,
                "verdict": data.get("verdict", ""),
            }
        )
    if not rows:
        raise ValueError(f"no m-*.json result files found in {run_dir}")
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    correct_rows = [row for row in rows if row["correctness"]]
    best = min(correct_rows, key=lambda r: r["starvation_pct"]) if correct_rows else None
    supported = [row for row in rows if row["supported"]]

    lines = [
        "# TensorWave benchmark summary",
        "",
        f"Result files: **{len(rows)}**",
        f"Correctness-passing shapes: **{len(correct_rows)}**",
        f"Strong-support shapes: **{len(supported)}**",
        "",
    ]

    if best is not None:
        lines.extend(
            [
                "## Lowest measured starvation among correct shapes",
                "",
                f"- M/K/N: `{best['m']}/{best['k']}/{best['n']}`",
                f"- starvation: **{fmt(best['starvation_pct'])}%**",
                f"- hidden transfer estimate: **{fmt(best['hidden_pct'])}%**",
                f"- H2D: **{fmt(best['h2d_gbps'])} GB/s**",
                f"- speedup: **{fmt(best['speedup'])}x**",
                f"- verdict: `{best['verdict']}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Sweep",
            "",
            "| M | K | N | tiles | dtype | H2D GB/s | starvation % | hidden % | speedup | correct | strong |",
            "|---:|---:|---:|---:|:---|---:|---:|---:|---:|:---:|:---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {m} | {k} | {n} | {tiles} | {dtype} | {h2d} | {starve} | {hidden} | {speedup} | {correct} | {strong} |".format(
                m=row["m"],
                k=row["k"],
                n=row["n"],
                tiles=row["tiles"],
                dtype=row["dtype"],
                h2d=fmt(row["h2d_gbps"]),
                starve=fmt(row["starvation_pct"]),
                hidden=fmt(row["hidden_pct"]),
                speedup=fmt(row["speedup"]),
                correct="yes" if row["correctness"] else "NO",
                strong="YES" if row["supported"] else "no",
            )
        )

    lines.extend(
        [
            "",
            "Strong support is defined as correctness passing, steady starvation <= 10%, and hidden transfer estimate >= 80%.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    rows = load_rows(run_dir)

    markdown = args.markdown or (run_dir / "SUMMARY.md")
    csv_path = args.csv_path or (run_dir / "summary.csv")
    write_markdown(markdown, rows)
    write_csv(csv_path, rows)

    strong = sum(bool(row["supported"]) for row in rows)
    print(f"Summarized {len(rows)} shapes")
    print(f"Strong-support shapes: {strong}")
    print(f"Markdown: {markdown}")
    print(f"CSV:      {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
