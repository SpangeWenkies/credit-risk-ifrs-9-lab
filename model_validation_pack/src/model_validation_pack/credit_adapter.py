"""Credit-risk adapter for the generic validation pack.

This module converts scored credit portfolio snapshots into the generic
`ScoredObservation` structures used by the validation toolkit. The adapter keeps
the package split-friendly: it understands plain data frames and column names
rather than importing root-repo dataclasses directly.
"""

from __future__ import annotations

import pandas as pd

from .types import ScoredObservation, ValidationBundle

DEFAULT_FEATURE_COLUMNS = ["ltv", "dti", "days_past_due", "utilization", "rating_rank"]


def _frame_to_observations(
    frame: pd.DataFrame,
    score_column: str,
    outcome_column: str,
    feature_columns: list[str],
) -> list[ScoredObservation]:
    observations: list[ScoredObservation] = []
    for row in frame.itertuples(index=False):
        features = {
            column: float(getattr(row, column))
            for column in feature_columns
            if hasattr(row, column) and pd.notna(getattr(row, column))
        }
        observations.append(
            ScoredObservation(
                observation_id=str(getattr(row, "loan_id")),
                score=float(getattr(row, score_column)),
                outcome=int(getattr(row, outcome_column)),
                period=str(getattr(row, "snapshot_date").date()) if hasattr(getattr(row, "snapshot_date"), "date") else str(getattr(row, "snapshot_date")),
                segment=str(getattr(row, "segment", "all")),
                features=features,
            )
        )
    return observations


def build_credit_validation_bundle(
    snapshot_scores: pd.DataFrame,
    previous_snapshot_scores: pd.DataFrame | None = None,
    benchmark_scores: list[float] | None = None,
    score_column: str = "pd_12m",
    outcome_column: str = "observed_default_12m",
    feature_columns: list[str] | None = None,
) -> ValidationBundle:
    """Build a generic validation bundle from scored credit snapshot frames.

    Summary
    -------
    Convert scored credit portfolio snapshots into the generic observation and
    benchmark payload expected by the validation pack API.

    Method
    ------
    The adapter maps each loan-level scored row to a `ScoredObservation`,
    preserves the reporting period and segment fields, and extracts a small set
    of numeric risk drivers for drift and sensitivity analysis. The current
    snapshot becomes the primary validation sample and the previous snapshot, if
    available, becomes the reference sample.

    Parameters
    ----------
    snapshot_scores:
        Current scored snapshot produced by the root repo.
    previous_snapshot_scores:
        Optional previous scored snapshot used as the drift reference sample.
    benchmark_scores:
        Optional benchmark or challenger score vector aligned one-for-one with
        `snapshot_scores`.
    score_column:
        Column to use as the primary model score.
    outcome_column:
        Column to use as the realised binary event.
    feature_columns:
        Optional explicit list of numeric features to retain for drift and
        sensitivity analysis.

    Returns
    -------
    ValidationBundle
        Generic bundle containing current observations, reference observations,
        benchmark scores, sensitivity samples, and simple metadata.

    Raises
    ------
    KeyError
        Raised if the requested score or outcome column is missing.
    ValueError
        Raised if benchmark scores are provided with the wrong length.

    Notes
    -----
    The adapter intentionally speaks plain `pandas` instead of importing the
    root package. That makes the validation pack easier to split into a separate
    repository later.

    Edge Cases
    ----------
    If no previous snapshot is provided, the reference sample is returned empty
    and the validation API will fall back to its own default reference logic.

    References
    ----------
    - ECB Banking Supervision, "Internal models."
      https://www.bankingsupervision.europa.eu/activities/internal_models/html/index.en.html
    """

    feature_columns = feature_columns or [column for column in DEFAULT_FEATURE_COLUMNS if column in snapshot_scores.columns]
    required_columns = {"loan_id", "snapshot_date", "segment", score_column, outcome_column}
    missing = sorted(column for column in required_columns if column not in snapshot_scores.columns)
    if missing:
        raise KeyError(f"Current snapshot is missing required columns: {missing}")
    if benchmark_scores is not None and len(benchmark_scores) != len(snapshot_scores):
        raise ValueError("Benchmark scores must align one-for-one with the current scored snapshot.")

    observations = _frame_to_observations(snapshot_scores, score_column, outcome_column, feature_columns)
    reference_observations = []
    if previous_snapshot_scores is not None and not previous_snapshot_scores.empty:
        previous_missing = sorted(column for column in required_columns if column not in previous_snapshot_scores.columns)
        if previous_missing:
            raise KeyError(f"Previous snapshot is missing required columns: {previous_missing}")
        reference_observations = _frame_to_observations(previous_snapshot_scores, score_column, outcome_column, feature_columns)

    sensitivity_samples = [observation.features for observation in observations if observation.features]
    metadata = {
        "score_column": score_column,
        "outcome_column": outcome_column,
        "feature_columns": feature_columns,
    }
    return ValidationBundle(
        observations=observations,
        reference_observations=reference_observations,
        benchmark_scores=benchmark_scores,
        sensitivity_samples=sensitivity_samples,
        metadata=metadata,
    )
