"""Data-quality and drift monitoring for scored credit portfolios.

This module provides a lightweight monitoring layer that checks schema health,
missingness, range violations, cohort consistency, and distribution shift
between two scored portfolio snapshots. The governance framing is aligned with
EU banking supervision expectations around credit monitoring, internal models,
and ongoing model oversight.

Assumptions
-----------
- Monitoring is performed between two comparable scored snapshots, typically the
  current and previous reporting dates.
- PSI and first Wasserstein distance are sufficient summary metrics for a
  portfolio project; they complement rather than replace more comprehensive
  model-performance monitoring.
- Range checks use transparent business-style bounds rather than institution-
  specific validation rules.

Primary references
------------------
- EBA, "Guidelines on loan origination and monitoring."
  https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-loan-origination-and-monitoring
- ECB Banking Supervision, "Internal models."
  https://www.bankingsupervision.europa.eu/activities/internal_models/html/index.en.html

Simplifications for this portfolio project
------------------------------------------
- No external monitoring store or alerting framework is used.
- PSI bins are derived from the combined reference/current sample for
  comparability.
- Thresholds are intentionally simple so they can be explained quickly in an
  interview.
"""

from __future__ import annotations

from math import isclose, log

import numpy as np
import pandas as pd

from .models import MonitoringReport

DEFAULT_MONITOR_COLUMNS = ["balance", "utilization", "ltv", "dti", "days_past_due", "pd_12m", "pd_lifetime"]
RANGE_RULES = {
    "balance": (0.0, 1_000_000.0),
    "utilization": (0.0, 1.1),
    "ltv": (0.0, 2.5),
    "dti": (0.0, 1.5),
    "days_past_due": (0.0, 180.0),
    "pd_12m": (0.0, 1.0),
    "pd_lifetime": (0.0, 1.0),
}


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Compute the population stability index on a common support.

    Summary
    -------
    Measure the bucketed distribution shift between a reference sample and a
    current sample using the population stability index.

    Method
    ------
    The function forms histogram buckets on the combined support of the two
    samples so both distributions are compared on the same partition. Each
    bucket contributes `(current - reference) * log(current / reference)` after
    small-value smoothing to avoid singularities.

    Parameters
    ----------
    reference:
        Reference numeric sample.
    current:
        Current numeric sample.
    bins:
        Number of equally spaced buckets used to discretise the combined sample.

    Returns
    -------
    float
        PSI value rounded to six decimals.

    Raises
    ------
    ValueError
        Raised if `bins` is less than one.

    Notes
    -----
    PSI is widely used by risk teams for score and feature stability checks, but
    it is only one summary statistic and should be interpreted with business
    context.

    Edge Cases
    ----------
    Identical empty samples return `0.0`. If one sample is empty and the other
    is not, the function returns a large finite PSI driven by the smoothing
    floor.

    References
    ----------
    - EBA, "Guidelines on loan origination and monitoring."
      https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-loan-origination-and-monitoring
    """

    if bins < 1:
        raise ValueError("bins must be at least 1")
    reference = reference.dropna().astype(float)
    current = current.dropna().astype(float)
    if reference.empty and current.empty:
        return 0.0

    combined = pd.concat([reference, current], ignore_index=True)
    low = float(combined.min())
    high = float(combined.max())
    if isclose(low, high):
        return 0.0

    edges = np.linspace(low, high, bins + 1)
    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)
    ref_share = np.clip(ref_hist / max(ref_hist.sum(), 1), 1e-6, None)
    cur_share = np.clip(cur_hist / max(cur_hist.sum(), 1), 1e-6, None)
    psi = np.sum((cur_share - ref_share) * np.log(cur_share / ref_share))
    return round(float(psi), 6)


def wasserstein_distance_1d(reference: pd.Series, current: pd.Series) -> float:
    """Compute the one-dimensional first Wasserstein distance.

    Summary
    -------
    Measure the average transport distance between two one-dimensional numeric
    samples.

    Method
    ------
    The function uses the closed-form one-dimensional Wasserstein implementation
    from `scipy.stats`, which compares the empirical distributions through their
    quantile structure rather than discrete bins.

    Parameters
    ----------
    reference:
        Reference numeric sample.
    current:
        Current numeric sample.

    Returns
    -------
    float
        Wasserstein distance rounded to six decimals.

    Raises
    ------
    None
        This helper does not raise custom exceptions.

    Notes
    -----
    Unlike PSI, Wasserstein distance is scale-sensitive. It is therefore useful
    as a companion metric when a bucketed statistic may hide the magnitude of a
    shift.

    Edge Cases
    ----------
    Two empty samples return `0.0`. If one sample is empty and the other is not,
    the function returns `1.0` as a sentinel value for a fully broken
    comparison.

    References
    ----------
    - ECB Banking Supervision, "Internal models."
      https://www.bankingsupervision.europa.eu/activities/internal_models/html/index.en.html
    """

    from scipy.stats import wasserstein_distance

    reference = reference.dropna().astype(float)
    current = current.dropna().astype(float)
    if reference.empty and current.empty:
        return 0.0
    if reference.empty or current.empty:
        return 1.0
    return round(float(wasserstein_distance(reference, current)), 6)


def run_monitoring(
    reference_snapshot: pd.DataFrame,
    current_snapshot: pd.DataFrame,
    reference_scores: pd.DataFrame,
    current_scores: pd.DataFrame,
) -> MonitoringReport:
    """Run data-quality and drift checks between two scored snapshots.

    Summary
    -------
    Compare a reference and current scored snapshot to produce data-quality
    tables, feature drift metrics, score drift metrics, and a compact summary.

    Method
    ------
    The function first validates schema coverage and missingness, then applies
    numeric range rules and cohort consistency checks, and finally measures
    feature and score drift with PSI and Wasserstein distance. A simple status
    summary counts the number of alerts triggered by the chosen thresholds.

    Parameters
    ----------
    reference_snapshot:
        Raw or scored reference snapshot used for cohort and feature checks.
    current_snapshot:
        Raw or scored current snapshot used for cohort and feature checks.
    reference_scores:
        Reference score frame, typically `previous_snapshot_scores`.
    current_scores:
        Current score frame, typically `snapshot_scores`.

    Returns
    -------
    MonitoringReport
        Dataclass containing data-quality tables, drift tables, and a summary
        dictionary.

    Raises
    ------
    None
        The function returns empty tables rather than raising when one side of
        the comparison is empty.

    Notes
    -----
    This function is designed for transparency. The thresholds are small and
    explicit so they are easy to defend in a portfolio interview.

    Edge Cases
    ----------
    If either snapshot is empty, schema, missingness, and range checks still run
    on the available data while drift statistics degrade gracefully.

    References
    ----------
    - EBA, "Guidelines on loan origination and monitoring."
      https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-loan-origination-and-monitoring
    """

    required_columns = sorted(set(DEFAULT_MONITOR_COLUMNS + ["loan_id", "segment", "snapshot_date"]))
    schema_rows = []
    for column in required_columns:
        schema_rows.append(
            {
                "column": column,
                "reference_present": int(column in reference_snapshot.columns),
                "current_present": int(column in current_snapshot.columns),
                "status": "ok" if column in reference_snapshot.columns and column in current_snapshot.columns else "missing",
            }
        )
    schema_checks = pd.DataFrame(schema_rows)

    shared_columns = [column for column in current_snapshot.columns if column in reference_snapshot.columns]
    missingness = pd.DataFrame(
        [
            {
                "column": column,
                "reference_missing_rate": round(float(reference_snapshot[column].isna().mean()), 6),
                "current_missing_rate": round(float(current_snapshot[column].isna().mean()), 6),
            }
            for column in shared_columns
        ]
    )

    range_rows = []
    for column, (lower, upper) in RANGE_RULES.items():
        if column not in current_snapshot.columns:
            continue
        series = pd.to_numeric(current_snapshot[column], errors="coerce")
        out_of_range = int(((series < lower) | (series > upper)).fillna(False).sum())
        range_rows.append(
            {
                "column": column,
                "lower_bound": lower,
                "upper_bound": upper,
                "out_of_range_count": out_of_range,
                "status": "alert" if out_of_range > 0 else "ok",
            }
        )
    range_checks = pd.DataFrame(range_rows)

    cohort_checks = pd.DataFrame(
        [
            {
                "metric": "reference_duplicate_loans",
                "value": int(reference_snapshot["loan_id"].duplicated().sum()) if "loan_id" in reference_snapshot.columns else np.nan,
            },
            {
                "metric": "current_duplicate_loans",
                "value": int(current_snapshot["loan_id"].duplicated().sum()) if "loan_id" in current_snapshot.columns else np.nan,
            },
            {
                "metric": "loan_overlap_ratio",
                "value": round(
                    float(
                        len(set(reference_snapshot.get("loan_id", pd.Series(dtype=str))) & set(current_snapshot.get("loan_id", pd.Series(dtype=str))))
                        / max(len(set(reference_snapshot.get("loan_id", pd.Series(dtype=str)))), 1)
                    ),
                    6,
                ),
            },
        ]
    )

    feature_drift_rows = []
    for column in DEFAULT_MONITOR_COLUMNS:
        if column not in reference_snapshot.columns or column not in current_snapshot.columns:
            continue
        psi = population_stability_index(reference_snapshot[column], current_snapshot[column])
        wasserstein = wasserstein_distance_1d(reference_snapshot[column], current_snapshot[column])
        feature_drift_rows.append(
            {
                "feature": column,
                "psi": psi,
                "wasserstein": wasserstein,
                "status": "alert" if psi > 0.25 or wasserstein > 0.15 else "watch" if psi > 0.10 else "stable",
            }
        )
    feature_drift = pd.DataFrame(feature_drift_rows)

    score_drift_rows = []
    for column in ("pd_12m", "pd_lifetime"):
        if column not in reference_scores.columns or column not in current_scores.columns:
            continue
        psi = population_stability_index(reference_scores[column], current_scores[column])
        wasserstein = wasserstein_distance_1d(reference_scores[column], current_scores[column])
        score_drift_rows.append(
            {
                "score": column,
                "psi": psi,
                "wasserstein": wasserstein,
                "status": "alert" if psi > 0.25 or wasserstein > 0.10 else "watch" if psi > 0.10 else "stable",
            }
        )
    score_drift = pd.DataFrame(score_drift_rows)

    summary = {
        "schema_missing": int(schema_checks["status"].eq("missing").sum()) if not schema_checks.empty else 0,
        "range_alerts": int(range_checks["status"].eq("alert").sum()) if not range_checks.empty else 0,
        "feature_drift_alerts": int(feature_drift["status"].eq("alert").sum()) if not feature_drift.empty else 0,
        "score_drift_alerts": int(score_drift["status"].eq("alert").sum()) if not score_drift.empty else 0,
    }

    return MonitoringReport(
        schema_checks=schema_checks,
        missingness=missingness,
        range_checks=range_checks,
        cohort_checks=cohort_checks,
        feature_drift=feature_drift,
        score_drift=score_drift,
        summary=summary,
    )
