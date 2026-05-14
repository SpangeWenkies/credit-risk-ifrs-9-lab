"""Panel-data helpers for loan-quarter credit modelling.

This module contains practical panel econometrics utilities: fixed-effect
demeaning, cohort summaries, and transition-rate panels. These functions make
the repo's loan-quarter structure visible as an econometric panel rather than a
flat classification table.

Assumptions
-----------
- Rows represent repeated observations of loans, borrowers, or segments.
- Entity and time identifiers are available as columns.
- Fixed-effect demeaning is used as a diagnostic transformation rather than as a
  full nonlinear fixed-effects PD estimator.

Primary references
------------------
- Wooldridge, "Econometric Analysis of Cross Section and Panel Data."
- Arellano, "Panel Data Econometrics."

Simplifications for this lab
------------------------------------------
- The module provides transformations and summaries, not a full panel estimator
  for nonlinear binary response models.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def add_vintage_and_age_columns(
    panel: pd.DataFrame,
    origination_column: str = "origination_date",
    snapshot_column: str = "snapshot_date",
) -> pd.DataFrame:
    """Add origination vintage and panel age variables.

    Summary
    -------
    Enrich loan-panel rows with cohort and age identifiers.

    Method
    ------
    The function converts origination and snapshot columns to timestamps, then
    creates quarterly vintage labels and months-on-book / quarters-on-book
    variables.

    Parameters
    ----------
    panel:
        Loan-level panel data.
    origination_column:
        Origination date column.
    snapshot_column:
        Observation date column.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with `origination_vintage`, `months_on_book_panel`,
        and `quarters_on_book_panel`.

    Raises
    ------
    KeyError
        Raised when required date columns are missing.

    Notes
    -----
    Vintage effects are important in credit risk because underwriting standards
    and macro conditions vary across origination cohorts.

    Edge Cases
    ----------
    Negative ages are clipped to zero to avoid invalid cohort calculations when
    synthetic dates are imperfect.

    References
    ----------
    - Wooldridge, "Econometric Analysis of Cross Section and Panel Data."
    """

    missing = [column for column in (origination_column, snapshot_column) if column not in panel.columns]
    if missing:
        raise KeyError(f"Missing panel date columns: {missing}")
    result = panel.copy()
    origination = pd.to_datetime(result[origination_column])
    snapshot = pd.to_datetime(result[snapshot_column])
    result["origination_vintage"] = origination.dt.to_period("Q").astype(str)
    months = (snapshot.dt.year - origination.dt.year) * 12 + (snapshot.dt.month - origination.dt.month)
    result["months_on_book_panel"] = months.clip(lower=0)
    result["quarters_on_book_panel"] = (result["months_on_book_panel"] // 3).astype(int)
    return result


def within_transform(
    panel: pd.DataFrame,
    columns: Sequence[str],
    entity_column: str,
) -> pd.DataFrame:
    """Apply entity fixed-effect demeaning to numeric columns.

    Summary
    -------
    Remove entity-level means from selected variables.

    Method
    ------
    For each requested column, the function subtracts the within-entity mean and
    returns demeaned columns with the suffix `_within`.

    Parameters
    ----------
    panel:
        Panel data frame.
    columns:
        Numeric columns to transform.
    entity_column:
        Entity identifier column.

    Returns
    -------
    pandas.DataFrame
        Data frame containing entity id and demeaned variables.

    Raises
    ------
    KeyError
        Raised when requested columns are missing.

    Notes
    -----
    This is useful for showing what variation remains after controlling for
    borrower or loan fixed effects.

    Edge Cases
    ----------
    Entities with one observation receive zero within-transformed values.

    References
    ----------
    - Wooldridge, "Econometric Analysis of Cross Section and Panel Data."
    """

    required = [entity_column, *columns]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"Missing within-transform columns: {missing}")
    result = panel[[entity_column]].copy()
    grouped = panel.groupby(entity_column)
    for column in columns:
        result[f"{column}_within"] = panel[column].astype(float) - grouped[column].transform("mean").astype(float)
    return result


def cohort_performance_table(
    panel: pd.DataFrame,
    cohort_column: str = "origination_vintage",
    time_column: str = "snapshot_date",
    default_column: str = "default_next_period",
    balance_column: str = "balance",
) -> pd.DataFrame:
    """Build a cohort-level performance panel.

    Summary
    -------
    Summarise active exposure, default rate, and balance by origination cohort
    and reporting date.

    Method
    ------
    The function groups by cohort and time, counts active loans, sums balance,
    and computes the interval default rate.

    Parameters
    ----------
    panel:
        Loan-quarter panel data.
    cohort_column:
        Cohort or vintage label.
    time_column:
        Reporting date column.
    default_column:
        Binary default indicator.
    balance_column:
        Outstanding balance column.

    Returns
    -------
    pandas.DataFrame
        Cohort-time performance table.

    Raises
    ------
    KeyError
        Raised when required columns are missing.

    Notes
    -----
    Cohort summaries help separate seasoning effects from origination-vintage
    effects.

    Edge Cases
    ----------
    If the requested cohort column is missing but date columns exist, create it
    first with `add_vintage_and_age_columns`.

    References
    ----------
    - Arellano, "Panel Data Econometrics."
    """

    required = [cohort_column, time_column, default_column, balance_column]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"Missing cohort performance columns: {missing}")
    table = (
        panel.groupby([cohort_column, time_column], as_index=False)
        .agg(active_loans=(default_column, "size"), defaults=(default_column, "sum"), exposure=(balance_column, "sum"))
        .sort_values([cohort_column, time_column])
    )
    table["default_rate"] = table["defaults"] / table["active_loans"].clip(lower=1)
    return table


def add_interaction_terms(
    frame: pd.DataFrame,
    left_columns: Sequence[str],
    right_columns: Sequence[str],
    separator: str = "_x_",
) -> pd.DataFrame:
    """Add numeric interaction terms to a modelling frame.

    Summary
    -------
    Create optional interaction features for testing macro-by-segment,
    borrower-by-macro, or nonlinear panel specifications.

    Method
    ------
    Every requested left column is multiplied by every requested right column.
    New columns are named `{left}{separator}{right}` and appended to a copy of
    the input frame.

    Parameters
    ----------
    frame:
        Input data frame.
    left_columns:
        First set of numeric columns.
    right_columns:
        Second set of numeric columns.
    separator:
        String used in generated feature names.

    Returns
    -------
    pandas.DataFrame
        Copy of `frame` with interaction columns.

    Raises
    ------
    KeyError
        Raised when requested columns are missing.

    Notes
    -----
    Interactions should be compared against the baseline rather than added by
    default. They are useful when diagnostics show that macro or behavioural
    effects differ by another continuous risk driver.

    Edge Cases
    ----------
    Duplicate generated names overwrite earlier generated names in the returned
    copy.

    References
    ----------
    - Wooldridge, "Econometric Analysis of Cross Section and Panel Data."
    """

    required = list(left_columns) + list(right_columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing interaction columns: {missing}")
    result = frame.copy()
    for left in left_columns:
        for right in right_columns:
            result[f"{left}{separator}{right}"] = result[left].astype(float) * result[right].astype(float)
    return result


def add_polynomial_terms(
    frame: pd.DataFrame,
    columns: Sequence[str],
    degree: int = 2,
) -> pd.DataFrame:
    """Add polynomial terms for nonlinear effect checks.

    Summary
    -------
    Create optional powers of numeric predictors to test whether a linear
    specification is too restrictive.

    Method
    ------
    For each selected column, the function adds powers from two through
    `degree`, named `{column}_pow_{power}`.

    Parameters
    ----------
    frame:
        Input modelling frame.
    columns:
        Numeric columns to expand.
    degree:
        Maximum polynomial degree.

    Returns
    -------
    pandas.DataFrame
        Copy of `frame` with polynomial features.

    Raises
    ------
    KeyError
        Raised when requested columns are missing.
    ValueError
        Raised when `degree < 2`.

    Notes
    -----
    Polynomial terms can reveal nonlinear LTV, DTI, or seasoning effects, but
    they can also add noise. Compare them with out-of-sample metrics.

    Edge Cases
    ----------
    Missing numeric values remain missing in generated powers.

    References
    ----------
    - Wooldridge, "Econometric Analysis of Cross Section and Panel Data."
    """

    if degree < 2:
        raise ValueError("degree must be at least two.")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing polynomial columns: {missing}")
    result = frame.copy()
    for column in columns:
        values = result[column].astype(float)
        for power in range(2, degree + 1):
            result[f"{column}_pow_{power}"] = np.power(values, power)
    return result


def poolability_diagnostic(
    frame: pd.DataFrame,
    segment_column: str,
    observed_column: str,
    prediction_column: str | None = None,
) -> pd.DataFrame:
    """Summarise whether a pooled model hides segment heterogeneity.

    Summary
    -------
    Compare segment event rates, optional prediction means, and deviations from
    the portfolio average.

    Method
    ------
    The function computes segment counts and observed rates. When predictions
    are supplied, it also computes calibration gaps. Deviations from the
    portfolio event rate give a compact poolability diagnostic.

    Parameters
    ----------
    frame:
        Input validation or modelling frame.
    segment_column:
        Segment identifier.
    observed_column:
        Binary observed outcome column.
    prediction_column:
        Optional predicted probability column.

    Returns
    -------
    pandas.DataFrame
        Segment-level poolability diagnostic.

    Raises
    ------
    KeyError
        Raised when required columns are missing.

    Notes
    -----
    Large segment deviations do not automatically require separate models; they
    indicate where interaction terms, segment effects, or hierarchical
    shrinkage should be tested.

    Edge Cases
    ----------
    Small segments are retained so sparse-data risk remains visible.

    References
    ----------
    - Wooldridge, "Econometric Analysis of Cross Section and Panel Data."
    - Gelman and Hill, "Data Analysis Using Regression and Multilevel/
      Hierarchical Models."
    """

    required = [segment_column, observed_column]
    if prediction_column is not None:
        required.append(prediction_column)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing poolability columns: {missing}")
    portfolio_rate = float(frame[observed_column].astype(int).mean())
    aggregations = {"count": (observed_column, "size"), "observed_rate": (observed_column, "mean")}
    if prediction_column is not None:
        aggregations["mean_prediction"] = (prediction_column, "mean")
    table = frame.groupby(segment_column, as_index=False).agg(**aggregations).rename(columns={segment_column: "segment"})
    table["rate_minus_portfolio"] = table["observed_rate"] - portfolio_rate
    if prediction_column is not None:
        table["calibration_gap"] = table["observed_rate"] - table["mean_prediction"]
    return table.sort_values("segment").reset_index(drop=True)
