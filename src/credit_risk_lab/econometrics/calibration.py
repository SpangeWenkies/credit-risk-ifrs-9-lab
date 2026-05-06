"""Econometric calibration utilities for PD models.

This module implements practical calibration tools for predicted default
probabilities: bin-level calibration tables, Brier score decomposition, and an
intercept-shift recalibration layer. These methods connect probability forecast
evaluation to the credit-risk requirement that predicted PDs should align with
observed default rates.

Assumptions
-----------
- Inputs are loan-level predicted probabilities and realised binary default
  outcomes over the same validation horizon.
- Recalibration is performed with a single logit-scale intercept shift, which
  preserves rank ordering while aligning the portfolio mean PD to a target or
  observed default rate.
- Calibration is diagnostic and governance-oriented; it does not replace model
  refitting when misspecification is severe.

Primary references
------------------
- Brier, G. W. (1950), "Verification of Forecasts Expressed in Terms of
  Probability."
- Cox, D. R. (1958), "Two Further Applications of a Model for Binary
  Regression."
- Van Calster et al. (2019), "Calibration: the Achilles heel of predictive
  analytics."

Simplifications for this portfolio project
------------------------------------------
- Bin calibration uses quantile or fixed-width buckets rather than smoother
  reliability curves.
- The intercept shift uses a one-dimensional binary search instead of fitting a
  full recalibration model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_table(
    frame: pd.DataFrame,
    prediction_column: str,
    observed_column: str,
    n_bins: int = 10,
    strategy: str = "quantile",
) -> pd.DataFrame:
    """Build a calibration table for probability forecasts.

    Summary
    -------
    Compare predicted default probabilities with observed default rates across
    probability bands.

    Method
    ------
    The function bins observations either by forecast quantiles or fixed
    probability intervals. For each bin it computes count, mean predicted PD,
    observed default rate, calibration error, and total expected versus observed
    defaults.

    Parameters
    ----------
    frame:
        Input validation data.
    prediction_column:
        Column containing predicted probabilities.
    observed_column:
        Column containing binary realised defaults.
    n_bins:
        Number of calibration bins.
    strategy:
        Either `"quantile"` for equal-count bins or `"uniform"` for fixed
        probability-width bins.

    Returns
    -------
    pandas.DataFrame
        Calibration table with one row per non-empty bin.

    Raises
    ------
    KeyError
        Raised when required columns are missing.
    ValueError
        Raised when `n_bins` is below two or `strategy` is unsupported.

    Notes
    -----
    Calibration tables are a standard validation artefact for PD models because
    IFRS 9 and model-governance work care about probability quality, not only
    rank ordering.

    Edge Cases
    ----------
    Duplicate predictions can collapse quantile bins; empty bins are dropped
    rather than reported with zero exposure.

    References
    ----------
    - Brier, G. W. (1950), "Verification of Forecasts Expressed in Terms of
      Probability."
    - Van Calster et al. (2019), "Calibration: the Achilles heel of predictive
      analytics."
    """

    missing = [column for column in (prediction_column, observed_column) if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing calibration columns: {missing}")
    if n_bins < 2:
        raise ValueError("n_bins must be at least two.")
    if strategy not in {"quantile", "uniform"}:
        raise ValueError("strategy must be either 'quantile' or 'uniform'.")

    data = frame[[prediction_column, observed_column]].dropna().copy()
    data[prediction_column] = data[prediction_column].astype(float).clip(1e-8, 1 - 1e-8)
    data[observed_column] = data[observed_column].astype(int)
    if data.empty:
        return pd.DataFrame(
            columns=["bin", "count", "mean_prediction", "observed_rate", "calibration_error", "expected_defaults", "observed_defaults"]
        )

    if strategy == "quantile":
        data["bin"] = pd.qcut(data[prediction_column], q=n_bins, duplicates="drop")
    else:
        data["bin"] = pd.cut(data[prediction_column], bins=np.linspace(0.0, 1.0, n_bins + 1), include_lowest=True)

    table = (
        data.groupby("bin", observed=True)
        .agg(
            count=(observed_column, "size"),
            mean_prediction=(prediction_column, "mean"),
            observed_rate=(observed_column, "mean"),
            expected_defaults=(prediction_column, "sum"),
            observed_defaults=(observed_column, "sum"),
        )
        .reset_index()
    )
    table["bin"] = table["bin"].astype(str)
    table["calibration_error"] = table["observed_rate"] - table["mean_prediction"]
    return table.round(
        {
            "mean_prediction": 6,
            "observed_rate": 6,
            "calibration_error": 6,
            "expected_defaults": 4,
            "observed_defaults": 4,
        }
    )


def brier_score(frame: pd.DataFrame, prediction_column: str, observed_column: str) -> float:
    """Calculate the Brier score for default probability forecasts.

    Summary
    -------
    Measure mean squared error between predicted PDs and realised binary default
    indicators.

    Method
    ------
    The Brier score is `mean((p_i - y_i)^2)`, where `p_i` is the predicted PD
    and `y_i` is the observed default indicator.

    Parameters
    ----------
    frame:
        Input validation data.
    prediction_column:
        Column containing predicted probabilities.
    observed_column:
        Column containing binary realised defaults.

    Returns
    -------
    float
        Brier score; lower is better.

    Raises
    ------
    KeyError
        Raised when required columns are missing.
    ValueError
        Raised when no complete observations are available.

    Notes
    -----
    The Brier score is a proper scoring rule for probability forecasts and is
    already used in the validation pack.

    Edge Cases
    ----------
    Predictions are clipped to `[0, 1]` before scoring to protect against
    numerical noise.

    References
    ----------
    - Brier, G. W. (1950), "Verification of Forecasts Expressed in Terms of
      Probability."
    """

    missing = [column for column in (prediction_column, observed_column) if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing Brier score columns: {missing}")
    data = frame[[prediction_column, observed_column]].dropna()
    if data.empty:
        raise ValueError("No complete observations are available for Brier scoring.")
    prediction = data[prediction_column].astype(float).clip(0.0, 1.0)
    observed = data[observed_column].astype(int)
    return float(np.mean((prediction - observed) ** 2))


def intercept_shift_recalibration(
    predictions: pd.Series,
    target_default_rate: float,
    tolerance: float = 1e-8,
    max_iter: int = 100,
) -> tuple[pd.Series, float]:
    """Recalibrate probabilities with a logit intercept shift.

    Summary
    -------
    Align the average predicted PD to a target default rate while preserving
    rank ordering.

    Method
    ------
    Predicted probabilities are transformed to logits, a scalar intercept shift
    is added, and probabilities are transformed back with the logistic function.
    A binary search finds the shift that makes the mean recalibrated PD match
    the target rate.

    Parameters
    ----------
    predictions:
        Raw probability forecasts.
    target_default_rate:
        Target mean PD, usually a realised default rate or long-run average.
    tolerance:
        Absolute tolerance for the mean PD match.
    max_iter:
        Maximum binary-search iterations.

    Returns
    -------
    tuple[pandas.Series, float]
        Recalibrated probability series and fitted intercept shift.

    Raises
    ------
    ValueError
        Raised when the target rate is not in `(0, 1)` or predictions are empty.

    Notes
    -----
    This is a transparent calibration overlay. It does not change score ranking,
    making it useful when rank ordering is accepted but portfolio-level
    calibration is off.

    Edge Cases
    ----------
    Extreme probabilities are clipped before applying the logit transform.

    References
    ----------
    - Cox, D. R. (1958), "Two Further Applications of a Model for Binary
      Regression."
    - Van Calster et al. (2019), "Calibration: the Achilles heel of predictive
      analytics."
    """

    if not 0.0 < target_default_rate < 1.0:
        raise ValueError("target_default_rate must be between zero and one.")
    if predictions.empty:
        raise ValueError("predictions must not be empty.")

    raw = predictions.astype(float).clip(1e-8, 1 - 1e-8)
    logits = np.log(raw / (1.0 - raw))
    low, high = -30.0, 30.0
    shift = 0.0
    calibrated = raw.copy()
    for _ in range(max_iter):
        shift = (low + high) / 2.0
        calibrated = pd.Series(1.0 / (1.0 + np.exp(-(logits + shift))), index=predictions.index)
        gap = float(calibrated.mean() - target_default_rate)
        if abs(gap) <= tolerance:
            break
        if gap > 0:
            high = shift
        else:
            low = shift
    return calibrated.clip(1e-8, 1 - 1e-8), float(shift)
