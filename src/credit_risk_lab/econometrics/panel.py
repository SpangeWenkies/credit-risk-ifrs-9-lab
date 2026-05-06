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

Simplifications for this portfolio project
------------------------------------------
- The module provides transformations and summaries, not a full panel estimator
  for nonlinear binary response models.
"""

from __future__ import annotations

from collections.abc import Sequence

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
