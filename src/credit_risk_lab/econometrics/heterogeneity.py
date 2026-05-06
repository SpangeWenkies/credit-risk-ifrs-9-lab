"""Heterogeneity diagnostics for credit-risk models.

This module implements segment-level performance diagnostics and coefficient
comparison helpers. It makes borrower and product heterogeneity visible before
moving to random-effects or hierarchical models.

Assumptions
-----------
- Segments are observed categories such as product type, region, or vintage.
- Segment-specific models are only fitted for groups with both target classes.
- The module is diagnostic and challenger-oriented.

Primary references
------------------
- Wooldridge, "Econometric Analysis of Cross Section and Panel Data."
- Gelman and Hill, "Data Analysis Using Regression and Multilevel/Hierarchical
  Models."

Simplifications for this portfolio project
------------------------------------------
- The module does not fit a full hierarchical Bayesian model in v1.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from .limited_dep import fit_binary_logit


def segment_performance_table(
    frame: pd.DataFrame,
    segment_column: str,
    prediction_column: str,
    observed_column: str,
) -> pd.DataFrame:
    """Summarise model performance by segment.

    Summary
    -------
    Compare mean predicted PDs and observed default rates across heterogeneous
    groups.

    Method
    ------
    The function groups data by segment and reports exposure count, mean
    prediction, observed event rate, and calibration gap.

    Parameters
    ----------
    frame:
        Scored validation data.
    segment_column:
        Segment identifier.
    prediction_column:
        Predicted probability column.
    observed_column:
        Binary observed outcome column.

    Returns
    -------
    pandas.DataFrame
        Segment-level calibration and event-rate table.

    Raises
    ------
    KeyError
        Raised when required columns are missing.

    Notes
    -----
    Segment performance tables are a practical first check for heterogeneity
    before fitting separate or hierarchical models.

    Edge Cases
    ----------
    Small segments are retained so they can be flagged by downstream governance
    review.

    References
    ----------
    - Gelman and Hill, "Data Analysis Using Regression and Multilevel/
      Hierarchical Models."
    """

    required = [segment_column, prediction_column, observed_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing segment performance columns: {missing}")
    result = (
        frame.groupby(segment_column, as_index=False)
        .agg(
            count=(observed_column, "size"),
            mean_prediction=(prediction_column, "mean"),
            observed_rate=(observed_column, "mean"),
            observed_defaults=(observed_column, "sum"),
        )
        .rename(columns={segment_column: "segment"})
    )
    result["calibration_gap"] = result["observed_rate"] - result["mean_prediction"]
    return result.sort_values("segment").reset_index(drop=True)


def fit_segment_binary_logits(
    frame: pd.DataFrame,
    segment_column: str,
    target_column: str,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str] | None = None,
    min_rows: int = 30,
) -> dict[str, object]:
    """Fit separate binary logit challengers by segment.

    Summary
    -------
    Estimate segment-specific binary response models where data volume permits.

    Method
    ------
    For each segment, the function checks minimum row count and target class
    variation, then fits a binary logit using the shared predictor set.

    Parameters
    ----------
    frame:
        Input model frame.
    segment_column:
        Segment identifier.
    target_column:
        Binary target column.
    numeric_columns:
        Numeric predictors.
    categorical_columns:
        Optional categorical predictors.
    min_rows:
        Minimum rows required for fitting a segment model.

    Returns
    -------
    dict[str, object]
        Mapping from segment label to fitted statsmodels result.

    Raises
    ------
    KeyError
        Raised when the segment column is missing.
    ValueError
        Raised when `min_rows` is below two.

    Notes
    -----
    Separate segment models are not automatically better than pooled models.
    They are useful for testing coefficient stability and heterogeneity.

    Edge Cases
    ----------
    Segments with only one target class are skipped instead of failing the full
    fit.

    References
    ----------
    - Wooldridge, "Econometric Analysis of Cross Section and Panel Data."
    """

    if segment_column not in frame.columns:
        raise KeyError(f"Missing segment column: {segment_column}")
    if min_rows < 2:
        raise ValueError("min_rows must be at least two.")
    fitted: dict[str, object] = {}
    for segment, group in frame.groupby(segment_column):
        if len(group) < min_rows or group[target_column].nunique() < 2:
            continue
        fitted[str(segment)] = fit_binary_logit(group, target_column, numeric_columns, categorical_columns)
    return fitted


def coefficient_stability_table(segment_models: dict[str, object]) -> pd.DataFrame:
    """Compare coefficients across segment-specific models.

    Summary
    -------
    Turn fitted segment model coefficients into a stability table.

    Method
    ------
    The function concatenates parameter vectors by segment and computes
    coefficient ranges across segments.

    Parameters
    ----------
    segment_models:
        Mapping from segment label to fitted statsmodels result.

    Returns
    -------
    pandas.DataFrame
        Long-form coefficient table with segment, feature, coefficient, and
        cross-segment range.

    Raises
    ------
    None
        Empty input returns an empty table.

    Notes
    -----
    Coefficient instability can indicate genuine heterogeneity, sparse data, or
    model misspecification.

    Edge Cases
    ----------
    Features absent from a segment model are omitted for that segment.

    References
    ----------
    - Gelman and Hill, "Data Analysis Using Regression and Multilevel/
      Hierarchical Models."
    """

    rows = []
    for segment, result in segment_models.items():
        for feature, coefficient in result.params.items():
            rows.append({"segment": segment, "feature": feature, "coefficient": float(coefficient)})
    table = pd.DataFrame(rows)
    if table.empty:
        return pd.DataFrame(columns=["segment", "feature", "coefficient", "coefficient_range"])
    ranges = table.groupby("feature")["coefficient"].agg(lambda values: float(values.max() - values.min())).rename("coefficient_range")
    return table.merge(ranges, on="feature", how="left").sort_values(["feature", "segment"]).reset_index(drop=True)
