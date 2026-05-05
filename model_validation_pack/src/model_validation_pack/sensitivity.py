"""Scenario-style sensitivity analysis for score functions."""

from __future__ import annotations

from collections.abc import Callable, Sequence

DEFAULT_SHOCKS = {
    "mild_recession": {"dti": 0.05, "ltv": 0.05, "days_past_due": 10.0},
    "severe_recession": {"dti": 0.10, "ltv": 0.10, "days_past_due": 30.0},
    "rate_relief": {"dti": -0.03, "ltv": -0.02, "days_past_due": -5.0},
}


def _apply_shock(features: dict[str, float], deltas: dict[str, float]) -> dict[str, float]:
    shocked = dict(features)
    for feature_name, delta in deltas.items():
        shocked[feature_name] = shocked.get(feature_name, 0.0) + delta
    return shocked


def run_sensitivity_analysis(
    scoring_function: Callable[[dict[str, float]], float] | None,
    feature_samples: Sequence[dict[str, float]],
    shocks: dict[str, dict[str, float]] | None = None,
) -> dict[str, object]:
    """Run macro-style shocked sensitivity analysis on a score function.

    Summary
    -------
    Apply a small library of stressed feature shocks to a scoring function and
    report the average change in predicted score.

    Method
    ------
    The function evaluates the supplied scoring function on a baseline sample of
    feature dictionaries, then re-evaluates it under each named shock scenario.
    The resulting average score differences form a simple sensitivity summary.

    Parameters
    ----------
    scoring_function:
        Callable mapping a feature dictionary to a probability-like score.
    feature_samples:
        Baseline feature sample used for the sensitivity exercise.
    shocks:
        Optional mapping of shock names to additive feature deltas.

    Returns
    -------
    dict[str, object]
        Sensitivity payload containing scenario-level shocked average scores.

    Raises
    ------
    None
        If no scoring function is provided, the function returns a structured
        `not_run` result.

    Notes
    -----
    This routine is intentionally generic so the validation pack can remain
    reusable outside of credit modelling.

    Edge Cases
    ----------
    Empty feature samples produce a `not_run` result because there is nothing to
    shock.

    References
    ----------
    - ECB Banking Supervision, "Internal models."
      https://www.bankingsupervision.europa.eu/activities/internal_models/html/index.en.html
    """

    if scoring_function is None or not feature_samples:
        return {"status": "not_run", "results": []}

    shocks = shocks or DEFAULT_SHOCKS
    baseline_average = sum(scoring_function(sample) for sample in feature_samples) / len(feature_samples)
    rows = []
    for shock_name, deltas in shocks.items():
        shocked_average = sum(scoring_function(_apply_shock(sample, deltas)) for sample in feature_samples) / len(feature_samples)
        rows.append(
            {
                "shock": shock_name,
                "baseline_avg_score": round(baseline_average, 6),
                "shocked_avg_score": round(shocked_average, 6),
                "avg_change": round(shocked_average - baseline_average, 6),
            }
        )

    return {"status": "completed", "results": rows}
