"""Duration and competing-risk summaries for credit panels.

This module adds heavier survival-analysis diagnostics around seasoning,
default, and prepayment. It is intentionally non-parametric in v1, making it a
clear complement to the pooled-logit hazard model.

Assumptions
-----------
- Each row is one loan-period interval.
- `quarters_on_book` indexes duration since origination.
- Default and prepayment are competing exits from the active loan book.

Primary references
------------------
- Singer and Willett (1993), "It's About Time: Using Discrete-Time Survival
  Analysis to Study Duration and the Timing of Events."
- Fine and Gray (1999), "A Proportional Hazards Model for the Subdistribution
  of a Competing Risk."

Simplifications for this lab
------------------------------------------
- The module reports empirical hazards and cumulative incidence, not a full
  competing-risks regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .limited_dep import fit_binary_logit


def duration_hazard_table(
    panel: pd.DataFrame,
    duration_column: str = "quarters_on_book",
    default_column: str = "default_next_period",
    prepay_column: str = "prepayment_flag",
) -> pd.DataFrame:
    """Estimate empirical duration-specific default and prepayment hazards.

    Summary
    -------
    Summarise how default and prepayment rates vary with loan seasoning.

    Method
    ------
    The function groups active panel rows by duration, counts rows at risk,
    default events, and prepayment events, then computes empirical hazards and
    a Kaplan-Meier style active survival curve.

    Parameters
    ----------
    panel:
        Loan-period performance panel.
    duration_column:
        Duration or seasoning column.
    default_column:
        Binary default indicator.
    prepay_column:
        Binary prepayment indicator.

    Returns
    -------
    pandas.DataFrame
        Duration-level risk table.

    Raises
    ------
    KeyError
        Raised when required columns are missing.

    Notes
    -----
    Duration tables help reveal whether default risk is front-loaded, seasoned,
    or concentrated around maturity.

    Edge Cases
    ----------
    Empty inputs return an empty table with expected columns.

    References
    ----------
    - Singer, J. D., and Willett, J. B. (1993), "It's About Time: Using
      Discrete-Time Survival Analysis to Study Duration and the Timing of
      Events."
    """

    required = [duration_column, default_column, prepay_column]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"Missing duration columns: {missing}")
    if panel.empty:
        return pd.DataFrame(
            columns=[
                "duration",
                "at_risk",
                "defaults",
                "prepayments",
                "default_hazard",
                "prepay_hazard",
                "active_survival",
            ]
        )

    table = (
        panel.groupby(duration_column, as_index=False)
        .agg(at_risk=(default_column, "size"), defaults=(default_column, "sum"), prepayments=(prepay_column, "sum"))
        .rename(columns={duration_column: "duration"})
        .sort_values("duration")
    )
    table["default_hazard"] = table["defaults"] / table["at_risk"].clip(lower=1)
    table["prepay_hazard"] = table["prepayments"] / table["at_risk"].clip(lower=1)
    exit_hazard = (table["default_hazard"] + table["prepay_hazard"]).clip(0.0, 1.0)
    table["active_survival"] = (1.0 - exit_hazard).cumprod()
    return table


def competing_risk_cumulative_incidence(hazard_table: pd.DataFrame) -> pd.DataFrame:
    """Convert duration hazards into cumulative incidence curves.

    Summary
    -------
    Estimate cumulative default and prepayment incidence from empirical hazards.

    Method
    ------
    At each duration, the event-specific incidence increment is prior active
    survival multiplied by the event-specific hazard. These increments are then
    cumulatively summed.

    Parameters
    ----------
    hazard_table:
        Output of `duration_hazard_table`.

    Returns
    -------
    pandas.DataFrame
        Duration table with cumulative default and prepayment incidence.

    Raises
    ------
    KeyError
        Raised when required hazard columns are missing.

    Notes
    -----
    Competing-risk incidence is more informative than a default-only curve when
    prepayment or maturity removes loans from the population at risk.

    Edge Cases
    ----------
    Empty inputs return an empty table with added incidence columns.

    References
    ----------
    - Fine, J. P., and Gray, R. J. (1999), "A Proportional Hazards Model for the
      Subdistribution of a Competing Risk."
    """

    required = ["duration", "default_hazard", "prepay_hazard"]
    missing = [column for column in required if column not in hazard_table.columns]
    if missing:
        raise KeyError(f"Missing hazard table columns: {missing}")
    result = hazard_table.copy().sort_values("duration")
    if result.empty:
        result["default_cumulative_incidence"] = []
        result["prepay_cumulative_incidence"] = []
        return result

    total_exit = (result["default_hazard"] + result["prepay_hazard"]).clip(0.0, 1.0)
    survival_before = (1.0 - total_exit).cumprod().shift(1, fill_value=1.0)
    result["default_cumulative_incidence"] = (survival_before * result["default_hazard"]).cumsum()
    result["prepay_cumulative_incidence"] = (survival_before * result["prepay_hazard"]).cumsum()
    result["remaining_active_probability"] = 1.0 - result["default_cumulative_incidence"] - result["prepay_cumulative_incidence"]
    result["remaining_active_probability"] = result["remaining_active_probability"].clip(0.0, 1.0)
    return result


def baseline_hazard_by_segment(
    panel: pd.DataFrame,
    segment_column: str = "segment",
    duration_column: str = "quarters_on_book",
    default_column: str = "default_next_period",
) -> pd.DataFrame:
    """Estimate segment-specific empirical baseline default hazards.

    Summary
    -------
    Produce a non-parametric view of duration dependence by product or borrower
    segment.

    Method
    ------
    The function groups by segment and duration, counts at-risk rows and default
    events, and computes an empirical default hazard.

    Parameters
    ----------
    panel:
        Loan-period performance panel.
    segment_column:
        Segment grouping column.
    duration_column:
        Duration or seasoning column.
    default_column:
        Binary default indicator.

    Returns
    -------
    pandas.DataFrame
        Segment-duration hazard table.

    Raises
    ------
    KeyError
        Raised when required columns are missing.

    Notes
    -----
    This table is a simple diagnostic before fitting segment-specific or
    random-effects survival models.

    Edge Cases
    ----------
    Segment-duration cells with one exposure are retained but should be
    interpreted cautiously.

    References
    ----------
    - Singer, J. D., and Willett, J. B. (1993), "It's About Time: Using
      Discrete-Time Survival Analysis to Study Duration and the Timing of
      Events."
    """

    required = [segment_column, duration_column, default_column]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"Missing baseline hazard columns: {missing}")
    table = (
        panel.groupby([segment_column, duration_column], as_index=False)
        .agg(at_risk=(default_column, "size"), defaults=(default_column, "sum"))
        .rename(columns={segment_column: "segment", duration_column: "duration"})
        .sort_values(["segment", "duration"])
    )
    table["default_hazard"] = table["defaults"] / table["at_risk"].clip(lower=1)
    table["log_hazard"] = np.log(table["default_hazard"].clip(lower=1e-8))
    return table


def cure_relapse_table(
    panel: pd.DataFrame,
    loan_id_column: str = "loan_id",
    date_column: str = "snapshot_date",
    dpd_column: str = "days_past_due",
) -> pd.DataFrame:
    """Summarise cure and relapse movements in delinquency history.

    Summary
    -------
    Track transitions between current and delinquent status for each loan.

    Method
    ------
    The panel is sorted by loan and date. A row is delinquent when DPD is
    positive. A cure is a transition from delinquent to current, and a relapse
    is a transition from current to delinquent after a prior delinquency episode.

    Parameters
    ----------
    panel:
        Loan-period panel.
    loan_id_column:
        Loan identifier.
    date_column:
        Observation date.
    dpd_column:
        Days-past-due column.

    Returns
    -------
    pandas.DataFrame
        Loan-level cure and relapse counts.

    Raises
    ------
    KeyError
        Raised when required columns are missing.

    Notes
    -----
    Cure and relapse diagnostics help decide whether a simple first-default
    survival model misses recurrent delinquency dynamics.

    Edge Cases
    ----------
    Loans with one observation have zero cure and relapse counts.

    References
    ----------
    - Aalen, Borgan, and Gjessing, "Survival and Event History Analysis."
    """

    required = [loan_id_column, date_column, dpd_column]
    missing = [column for column in required if column not in panel.columns]
    if missing:
        raise KeyError(f"Missing cure/relapse columns: {missing}")
    ordered = panel.sort_values([loan_id_column, date_column]).copy()
    ordered["_delinquent"] = ordered[dpd_column].astype(float).gt(0)
    ordered["_prior_delinquent"] = ordered.groupby(loan_id_column)["_delinquent"].shift(1).fillna(False).astype(bool)
    ever_delinquent = ordered.groupby(loan_id_column)["_delinquent"].cummax()
    ordered["_ever_prior_delinquent"] = ever_delinquent.groupby(ordered[loan_id_column]).shift(1).fillna(False).astype(bool)
    ordered["_cure"] = ordered["_prior_delinquent"] & ~ordered["_delinquent"]
    ordered["_relapse"] = ordered["_ever_prior_delinquent"] & ~ordered["_prior_delinquent"] & ordered["_delinquent"]
    return (
        ordered.groupby(loan_id_column, as_index=False)
        .agg(cures=("_cure", "sum"), relapses=("_relapse", "sum"), observations=("_delinquent", "size"))
        .sort_values(loan_id_column)
        .reset_index(drop=True)
    )


def fit_competing_risk_logits(
    panel: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str] | None = None,
    default_column: str = "default_next_period",
    prepay_column: str = "prepayment_flag",
) -> dict[str, object]:
    """Fit separate binary logits for default and prepayment exits.

    Summary
    -------
    Estimate simple cause-specific challenger models for competing exits.

    Method
    ------
    The function fits one binary logit for default and one binary logit for
    prepayment using the same feature set. The two models are intentionally
    separate so each event type can have different drivers.

    Parameters
    ----------
    panel:
        Loan-period panel.
    numeric_columns:
        Numeric predictors.
    categorical_columns:
        Optional categorical predictors.
    default_column:
        Binary default event column.
    prepay_column:
        Binary prepayment event column.

    Returns
    -------
    dict[str, object]
        Fitted statsmodels results keyed by `default` and `prepayment` when
        class variation permits.

    Raises
    ------
    KeyError
        Raised when event columns are missing.

    Notes
    -----
    This is a practical competing-risk regression layer. It does not estimate a
    Fine-Gray subdistribution hazard; it gives interpretable cause-specific
    checks before using heavier models.

    Edge Cases
    ----------
    If an event type has only one observed class, that model is skipped.

    References
    ----------
    - Fine, J. P., and Gray, R. J. (1999), "A Proportional Hazards Model for the
      Subdistribution of a Competing Risk."
    """

    missing = [column for column in (default_column, prepay_column) if column not in panel.columns]
    if missing:
        raise KeyError(f"Missing competing-risk columns: {missing}")
    models: dict[str, object] = {}
    if panel[default_column].nunique() > 1:
        models["default"] = fit_binary_logit(panel, default_column, numeric_columns, categorical_columns)
    if panel[prepay_column].nunique() > 1:
        models["prepayment"] = fit_binary_logit(panel, prepay_column, numeric_columns, categorical_columns)
    return models
