#!/usr/bin/env python3
"""Calibrate the analytical feasibility map from TensorWave benchmark JSON files.

The calibrator intentionally uses *effective measured* values rather than GPU peak
specifications. It supports Phase-1/2 style JSON and Phase-3 Q4 JSON defensively.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def nested_number(obj: dict[str, Any], *paths: tuple[str, ...]) -> float | None:
    for path in paths:
        current: Any = obj
        ok = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                ok = False
                break
            current = current[key]
        if ok:
            try:
                return float(current)
            except (TypeError, ValueError):
                pass
    return None


def load_results(run_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(run_dir.glob("m-*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"failed to read {path}: {exc}") from exc
        data["__path"] = str(path)
        results.append(data)
    if not results:
        raise ValueError(f"no m-*.json files found under {run_dir}")
    return results


def geometry(data: dict[str, Any]) -> tuple[int, int, int, int]:
    g = data.get("geometry") or data.get("config") or {}
    return (
        int(g.get("m", 0)),
        int(g.get("k", 0)),
        int(g.get("n", 0)),
        int(g.get("tiles", g.get("tile_count", 0))),
    )


def effective_tflops(data: dict[str, Any]) -> float | None:
    m, k, n, tiles = geometry(data)
    if min(m, k, n, tiles) <= 0:
        return None

    # Prefer GEMM-only time when Phase 3 reports it. Fall back to compute time.
    ms = nested_number(
        data,
        ("overlapped", "gemm_ms"),
        ("overlapped", "gemm_time_ms"),
        ("overlapped", "compute_ms"),
    )
    if ms is None or ms <= 0:
        return None

    flops = 2.0 * m * k * n * tiles
    return flops / (ms / 1000.0) / 1.0e12


def correctness(data: dict[str, Any]) -> bool:
    value = data.get("correctness_ok")
    if value is None:
        # Old result schemas without a correctness field are not used for calibration.
        return False
    return bool(value)


def collect(run_dir: Path) -> dict[str, Any]:
    results = load_results(run_dir)
    rows = []
    h2d_values: list[float] = []
    tflops_values: list[float] = []

    for data in results:
        m, k, n, tiles = geometry(data)
        h2d = nested_number(data, ("overlapped", "h2d_gbps"))
        tflops = effective_tflops(data)
        ok = correctness(data)
        if ok and h2d is not None and h2d > 0:
            h2d_values.append(h2d)
        if ok and tflops is not None and tflops > 0:
            tflops_values.append(tflops)

        rows.append(
            {
                "file": data["__path"],
                "m": m,
                "k": k,
                "n": n,
                "tiles": tiles,
                "correctness_ok": ok,
                "h2d_gbps": h2d,
                "effective_tflops": tflops,
                "starvation_pct": nested_number(data, ("overlapped", "steady_starvation_pct")),
                "hidden_transfer_pct": nested_number(data, ("overlapped", "steady_hidden_transfer_pct")),
            }
        )

    if not h2d_values:
        raise ValueError("no correctness-passing result contains positive overlapped.h2d_gbps")
    if not tflops_values:
        raise ValueError("no correctness-passing result has enough geometry/time data to estimate effective TFLOP/s")

    return {
        "schema": "tensorwave.feasibility-calibration.v1",
        "source_run_dir": str(run_dir.resolve()),
        "samples": rows,
        "effective_h2d_gbps_median": statistics.median(h2d_values),
        "effective_h2d_gbps_min": min(h2d_values),
        "effective_h2d_gbps_max": max(h2d_values),
        "effective_tflops_median": statistics.median(tflops_values),
        "effective_tflops_min": min(tflops_values),
        "effective_tflops_max": max(tflops_values),
        "correct_h2d_samples": len(h2d_values),
        "correct_compute_samples": len(tflops_values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate TensorWave feasibility map from a benchmark run")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = collect(args.run_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Measured H2D median: {payload['effective_h2d_gbps_median']:.3f} GB/s")
    print(f"Measured effective compute median: {payload['effective_tflops_median']:.3f} TFLOP/s")
    print("Use these as --pcie-gbps and --effective-tflops for build_feasibility_map.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
