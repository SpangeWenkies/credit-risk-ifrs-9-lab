"""Backtesting and calibration diagnostics for probability forecasts.

This module summarises predictive performance for binary event probability
models. The primary metric is the Brier score, supplemented with calibration
tables and simple rank-ordering diagnostics by score band.

Assumptions
-----------
- Each `ScoredObservation` contains a probability score and a binary realised
  outcome.
- Score bands are quantile-based for compactness and readability.

Primary references
------------------
- Brier, G. W. (1950), "Verification of Forecasts Expressed in Terms of
  Probability." https://cir.nii.ac.jp/crid/1361981468554183168
- Board of Governors of the Federal Reserve System and OCC, "Supervisory
  Guidance on Model Risk Management (SR 11-7)."
  https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107a1.pdf

Simplifications for this portfolio project
------------------------------------------
- The module focuses on Brier-style diagnostics rather than a full validation
  stack with Hosmer-Lemeshow tests, ROC/AUC, and multiple confidence intervals.
"""

from __future__ import annotations

import math

from .types import ScoredObservation


def _band_rows(observations: list[ScoredObservation], bands: int) -> list[dict[str, object]]:
    ordered = sorted(observations, key=lambda obs: obs.score)
    rows: list[dict[str, object]] = []
    for band in range(bands):
        start = round(band * len(ordered) / bands)
        end = round((band + 1) * len(ordered) / bands)
        group = ordered[start:end]
        if not group:
            continue
        avg_score = sum(obs.score for obs in group) / len(group)
        default_rate = sum(obs.outcome for obs in group) / len(group)
        rows.append(
            {
                "band": band + 1,
                "count": len(group),
                "avg_score": round(avg_score, 6),
                "default_rate": round(default_rate, 6),
                "calibration_gap": round(avg_score - default_rate, 6),
            }
        )
    return rows


def run_backtest(observations: list[ScoredObservation], bands: int = 5) -> dict[str, object]:
    """Run compact probability backtesting diagnostics on scored observations.

    Summary
    -------
    Compute Brier-style backtest metrics, score-band calibration rows, and a
    rank-ordering summary from binary probability forecasts.

    Method
    ------
    The function evaluates the mean squared error of the probability forecasts
    using the Brier score, compares the mean predicted rate with the observed
    event rate, and forms quantile score bands to inspect calibration and rank
    ordering.

    Parameters
    ----------
    observations:
        Validation sample consisting of probability forecasts and realised
        binary outcomes.
    bands:
        Number of score bands used for calibration and rank-ordering summaries.

    Returns
    -------
    dict[str, object]
        Backtest payload containing top-line metrics, calibration rows, and a
        simple monotonic rank-ordering flag.

    Raises
    ------
    ValueError
        Raised if `bands` is less than one.

    Notes
    -----
    The Brier score is appropriate here because the validation pack is designed
    around probability forecasts rather than raw binary classifications.

    Edge Cases
    ----------
    Empty inputs return a zeroed structure instead of raising so the wider
    validation pipeline can decide how to report "not enough data."

    References
    ----------
    - Brier, G. W. (1950), "Verification of Forecasts Expressed in Terms of
      Probability." https://cir.nii.ac.jp/crid/1361981468554183168
    """

    if bands < 1:
        raise ValueError("bands must be at least 1")
    if not observations:
        return {
            "observation_count": 0,
            "brier_score": 0.0,
            "rmse": 0.0,
            "mean_score": 0.0,
            "observed_default_rate": 0.0,
            "calibration_gap": 0.0,
            "bands": [],
            "rank_ordering": [],
            "monotonic_rank_ordering": True,
        }

    brier_score = sum((obs.score - obs.outcome) ** 2 for obs in observations) / len(observations)
    mean_score = sum(obs.score for obs in observations) / len(observations)
    observed_default_rate = sum(obs.outcome for obs in observations) / len(observations)
    band_rows = _band_rows(observations, bands)
    rank_ordering = list(reversed(band_rows))
    monotonic = all(
        left["default_rate"] >= right["default_rate"]
        for left, right in zip(rank_ordering, rank_ordering[1:], strict=False)
    )
    return {
        "observation_count": len(observations),
        "brier_score": round(brier_score, 6),
        "rmse": round(math.sqrt(brier_score), 6),
        "mean_score": round(mean_score, 6),
        "observed_default_rate": round(observed_default_rate, 6),
        "calibration_gap": round(mean_score - observed_default_rate, 6),
        "bands": band_rows,
        "rank_ordering": rank_ordering,
        "monotonic_rank_ordering": monotonic,
    }
