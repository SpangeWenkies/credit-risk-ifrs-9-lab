"""Stability diagnostics across time periods and segments."""

from __future__ import annotations

from collections import defaultdict

from .types import ScoredObservation


def run_stability_tests(observations: list[ScoredObservation]) -> dict[str, object]:
    """Summarise score stability across periods and segments.

    Summary
    -------
    Aggregate model scores and realised outcomes by period and by segment to
    assess whether the model behaves consistently across common reporting cuts.

    Method
    ------
    The function groups the observations by reporting period and segment, then
    computes mean score and default-rate summaries. It also reports the maximum
    period-to-period shift in the mean score as a simple stability statistic.

    Parameters
    ----------
    observations:
        Scored observations for the validation sample.

    Returns
    -------
    dict[str, object]
        Period and segment stability summaries plus a compact shift metric.

    Raises
    ------
    None
        Empty inputs return empty summaries.

    Notes
    -----
    Stability tables are especially useful in a model validation memo because
    they reveal whether apparent overall performance is hiding specific pockets
    of instability.

    Edge Cases
    ----------
    If all observations belong to one period, the maximum mean-score shift is
    zero by construction.

    References
    ----------
    - ECB Banking Supervision, "Internal models."
      https://www.bankingsupervision.europa.eu/activities/internal_models/html/index.en.html
    """

    by_period: dict[str, list[ScoredObservation]] = defaultdict(list)
    by_segment: dict[str, list[ScoredObservation]] = defaultdict(list)
    for observation in observations:
        by_period[observation.period].append(observation)
        by_segment[observation.segment].append(observation)

    period_summary = []
    for period in sorted(by_period):
        group = by_period[period]
        period_summary.append(
            {
                "period": period,
                "count": len(group),
                "mean_score": round(sum(obs.score for obs in group) / len(group), 6),
                "default_rate": round(sum(obs.outcome for obs in group) / len(group), 6),
            }
        )

    segment_summary = []
    for segment in sorted(by_segment):
        group = by_segment[segment]
        segment_summary.append(
            {
                "segment": segment,
                "count": len(group),
                "mean_score": round(sum(obs.score for obs in group) / len(group), 6),
                "default_rate": round(sum(obs.outcome for obs in group) / len(group), 6),
            }
        )

    mean_scores = [row["mean_score"] for row in period_summary]
    max_shift = max(mean_scores) - min(mean_scores) if mean_scores else 0.0
    return {
        "period_summary": period_summary,
        "segment_summary": segment_summary,
        "max_mean_score_shift": round(max_shift, 6),
        "period_count": len(period_summary),
    }
