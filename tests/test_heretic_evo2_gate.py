from tools.heretic_evo2_gate import Result, compare


def _row(
    result_id: str,
    *,
    refused: bool,
    tool_call_valid: bool = True,
    tool_completed: bool = True,
    quality_score: float = 1.0,
) -> Result:
    return Result(
        id=result_id,
        refused=refused,
        tool_call_valid=tool_call_valid,
        tool_completed=tool_completed,
        quality_score=quality_score,
    )


def test_candidate_with_fewer_false_refusals_passes() -> None:
    baseline = {
        "a": _row("a", refused=True, tool_call_valid=False, tool_completed=False, quality_score=0.0),
        "b": _row("b", refused=False),
    }
    candidate = {
        "a": _row("a", refused=False),
        "b": _row("b", refused=False),
    }

    report = compare(baseline, candidate, require_refusal_improvement=True)

    assert report["passed"] is True
    assert report["candidate"]["refusal_rate"] < report["baseline"]["refusal_rate"]


def test_tool_regression_fails_even_if_refusal_rate_improves() -> None:
    baseline = {
        "a": _row("a", refused=True, tool_call_valid=False, tool_completed=False, quality_score=0.0),
        "b": _row("b", refused=False),
    }
    candidate = {
        "a": _row("a", refused=False),
        "b": _row("b", refused=False, tool_call_valid=False, tool_completed=False),
    }

    report = compare(baseline, candidate, require_refusal_improvement=True)

    assert report["passed"] is False
    assert report["checks"]["tool_call_valid_rate_not_worse"] is False
    assert report["checks"]["tool_completed_rate_not_worse"] is False


def test_quality_drop_beyond_tolerance_fails() -> None:
    baseline = {"a": _row("a", refused=False, quality_score=1.0)}
    candidate = {"a": _row("a", refused=False, quality_score=0.90)}

    report = compare(baseline, candidate, quality_drop_tolerance=0.02)

    assert report["passed"] is False
    assert report["checks"]["quality_within_tolerance"] is False
