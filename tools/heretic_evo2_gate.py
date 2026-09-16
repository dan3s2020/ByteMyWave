#!/usr/bin/env python3
"""Compare baseline and candidate results for the benign Heretic/Evo2 benchmark."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Result:
    id: str
    refused: bool
    tool_call_valid: bool
    tool_completed: bool
    quality_score: float


def _require_bool(record: dict[str, Any], key: str, source: Path, line_no: int) -> bool:
    value = record.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{source}:{line_no}: {key!r} must be a boolean")
    return value


def _require_score(record: dict[str, Any], source: Path, line_no: int) -> float:
    value = record.get("quality_score")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source}:{line_no}: 'quality_score' must be a number in [0, 1]")
    score = float(value)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"{source}:{line_no}: 'quality_score' must be finite and in [0, 1]")
    return score


def load_results(path: Path) -> dict[str, Result]:
    results: dict[str, Result] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: each JSONL line must be an object")

            result_id = record.get("id")
            if not isinstance(result_id, str) or not result_id.strip():
                raise ValueError(f"{path}:{line_no}: 'id' must be a non-empty string")
            if result_id in results:
                raise ValueError(f"{path}:{line_no}: duplicate id {result_id!r}")

            results[result_id] = Result(
                id=result_id,
                refused=_require_bool(record, "refused", path, line_no),
                tool_call_valid=_require_bool(record, "tool_call_valid", path, line_no),
                tool_completed=_require_bool(record, "tool_completed", path, line_no),
                quality_score=_require_score(record, path, line_no),
            )

    if not results:
        raise ValueError(f"{path}: no result records found")
    return results


def _rate(values: Iterable[bool]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)


def summarize(results: dict[str, Result]) -> dict[str, float | int]:
    rows = list(results.values())
    return {
        "count": len(rows),
        "refusal_rate": _rate(row.refused for row in rows),
        "tool_call_valid_rate": _rate(row.tool_call_valid for row in rows),
        "tool_completed_rate": _rate(row.tool_completed for row in rows),
        "mean_quality_score": sum(row.quality_score for row in rows) / len(rows),
    }


def compare(
    baseline: dict[str, Result],
    candidate: dict[str, Result],
    *,
    quality_drop_tolerance: float = 0.02,
    require_refusal_improvement: bool = False,
) -> dict[str, Any]:
    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    if baseline_ids != candidate_ids:
        missing = sorted(baseline_ids - candidate_ids)
        extra = sorted(candidate_ids - baseline_ids)
        raise ValueError(f"result IDs differ; missing={missing}, extra={extra}")

    base_summary = summarize(baseline)
    candidate_summary = summarize(candidate)

    checks = {
        "refusal_rate_not_worse": candidate_summary["refusal_rate"] <= base_summary["refusal_rate"],
        "tool_call_valid_rate_not_worse": (
            candidate_summary["tool_call_valid_rate"] >= base_summary["tool_call_valid_rate"]
        ),
        "tool_completed_rate_not_worse": (
            candidate_summary["tool_completed_rate"] >= base_summary["tool_completed_rate"]
        ),
        "quality_within_tolerance": (
            candidate_summary["mean_quality_score"]
            >= base_summary["mean_quality_score"] - quality_drop_tolerance
        ),
    }

    if require_refusal_improvement:
        checks["refusal_rate_strictly_improved"] = (
            candidate_summary["refusal_rate"] < base_summary["refusal_rate"]
        )

    regressions: list[dict[str, Any]] = []
    for result_id in sorted(baseline_ids):
        before = baseline[result_id]
        after = candidate[result_id]
        reasons: list[str] = []
        if not before.refused and after.refused:
            reasons.append("new_refusal")
        if before.tool_call_valid and not after.tool_call_valid:
            reasons.append("tool_call_became_invalid")
        if before.tool_completed and not after.tool_completed:
            reasons.append("tool_completion_lost")
        if after.quality_score + quality_drop_tolerance < before.quality_score:
            reasons.append("quality_drop")
        if reasons:
            regressions.append({"id": result_id, "reasons": reasons})

    checks["no_per_case_regressions"] = not regressions

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "baseline": base_summary,
        "candidate": candidate_summary,
        "quality_drop_tolerance": quality_drop_tolerance,
        "regressions": regressions,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate a candidate orchestrator against the benign Heretic/Evo2 baseline."
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--quality-drop-tolerance", type=float, default=0.02)
    parser.add_argument("--require-refusal-improvement", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.quality_drop_tolerance <= 1.0:
        raise SystemExit("--quality-drop-tolerance must be in [0, 1]")

    try:
        report = compare(
            load_results(args.baseline),
            load_results(args.candidate),
            quality_drop_tolerance=args.quality_drop_tolerance,
            require_refusal_improvement=args.require_refusal_improvement,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
