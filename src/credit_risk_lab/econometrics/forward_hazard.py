"""Forward hazard-path modelling for lifetime PD term structures.

This module removes the constant-hazard approximation used by the first PD
scoring layer. Instead of converting a single quarterly hazard with
`1 - (1 - h)^n`, it builds a future loan-quarter panel, projects core loan and
macro variables, scores a hazard path, and compounds survival through time.

Assumptions
-----------
- The reporting snapshot contains the same model features used by the fitted
  survival model.
- Future balances amortise linearly unless the product is revolving.
- Macro paths are scenario tables indexed by future horizon quarter.
- Lifetime PD is built from a path of conditional hazards:
  `1 - product_t(1 - h_t)`.

Primary references
------------------
- Singer and Willett (1993), "It's About Time: Using Discrete-Time Survival
  Analysis to Study Duration and the Timing of Events."
- IFRS Foundation, "IFRS 9 Financial Instruments."

Simplifications for this portfolio project
------------------------------------------
- Feature projection is deterministic and intentionally transparent.
- Borrower income and behavioural features are not simulated with a full
  macroeconometric satellite model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from credit_risk_lab.models import SurvivalPDModel
from credit_risk_lab.survival import _prepare_design_matrix


def cumulative_pd_from_hazard_path(hazards: Sequence[float] | pd.Series) -> float:
    """Convert a hazard path into cumulative default probability.

    Summary
    -------
    Compound conditional default hazards into a multi-period PD.

    Method
    ------
    If `h_t` is the conditional probability of default in future period `t`
    given survival to the start of that period, survival is
    `product_t(1 - h_t)` and cumulative PD is one minus that survival
    probability.

    Parameters
    ----------
    hazards:
        Sequence of conditional period default probabilities.

    Returns
    -------
    float
        Cumulative PD over the supplied hazard path.

    Raises
    ------
    ValueError
        Raised when a hazard lies outside `[0, 1]`.

    Notes
    -----
    This is the main replacement for the constant-hazard formula in the first
    version of the repo.

    Edge Cases
    ----------
    An empty path returns zero because no future interval is being projected.

    References
    ----------
    - Singer, J. D., and Willett, J. B. (1993), "It's About Time: Using
      Discrete-Time Survival Analysis to Study Duration and the Timing of
      Events."
    """

    values = pd.Series(hazards, dtype=float)
    if values.empty:
        return 0.0
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("Hazards must lie between zero and one.")
    survival = float(np.prod(1.0 - values.clip(1e-8, 1 - 1e-8)))
    return float(np.clip(1.0 - survival, 0.0, 1.0))


def build_forward_panel(
    snapshot: pd.DataFrame,
    macro_path: pd.DataFrame,
    max_horizon_quarters: int | None = None,
) -> pd.DataFrame:
    """Project future loan-quarter rows from a reporting snapshot.

    Summary
    -------
    Build the future panel needed to score non-constant hazard paths.

    Method
    ------
    For each active loan, the function creates one row for each future quarter
    up to remaining maturity or `max_horizon_quarters`. It increments seasoning,
    decreases remaining term, amortises non-revolving balances, updates LTV
    using the scenario house-price path, and copies macro variables into the
    projected feature rows.

    Parameters
    ----------
    snapshot:
        Reporting-date loan-level score or performance frame.
    macro_path:
        Scenario macro path with `horizon_quarter` and optional macro columns.
    max_horizon_quarters:
        Optional cap on projected quarters.

    Returns
    -------
    pandas.DataFrame
        Forward loan-quarter panel with one row per loan and future horizon.

    Raises
    ------
    KeyError
        Raised when required snapshot or macro columns are missing.
    ValueError
        Raised when `max_horizon_quarters` is not positive.

    Notes
    -----
    This is an engineering bridge between current point-in-time scoring and a
    true lifetime PD term structure.

    Edge Cases
    ----------
    Loans with zero remaining term are omitted. If the macro path is shorter
    than a loan horizon, the last macro row is carried forward.

    References
    ----------
    - IFRS Foundation, "IFRS 9 Financial Instruments."
    - Singer, J. D., and Willett, J. B. (1993), "It's About Time: Using
      Discrete-Time Survival Analysis to Study Duration and the Timing of
      Events."
    """

    required_snapshot = ["loan_id", "remaining_term_quarters", "quarters_on_book", "balance", "collateral_value", "ltv", "dti"]
    missing_snapshot = [column for column in required_snapshot if column not in snapshot.columns]
    if missing_snapshot:
        raise KeyError(f"Snapshot is missing required columns: {missing_snapshot}")
    if "horizon_quarter" not in macro_path.columns:
        raise KeyError("macro_path must contain a 'horizon_quarter' column.")
    if macro_path.empty:
        raise ValueError("macro_path must contain at least one projected row.")
    if max_horizon_quarters is not None and max_horizon_quarters <= 0:
        raise ValueError("max_horizon_quarters must be positive when supplied.")

    macro = macro_path.sort_values("horizon_quarter").reset_index(drop=True).copy()
    rows: list[dict[str, object]] = []
    for _, loan in snapshot.iterrows():
        remaining = int(max(loan["remaining_term_quarters"], 0))
        horizon = remaining if max_horizon_quarters is None else min(remaining, max_horizon_quarters)
        if horizon <= 0:
            continue
        start_balance = float(loan["balance"])
        start_collateral = float(loan["collateral_value"])
        segment = str(loan.get("segment", ""))
        is_revolving = segment == "credit_card"
        for k in range(1, horizon + 1):
            macro_row = macro.loc[macro["horizon_quarter"].le(k)].tail(1)
            if macro_row.empty:
                macro_values = macro.iloc[0].to_dict()
            else:
                macro_values = macro_row.iloc[0].to_dict()
            projected = loan.to_dict()
            projected["horizon_quarter"] = k
            projected["quarters_on_book"] = int(loan["quarters_on_book"]) + k
            projected["remaining_term_quarters"] = max(remaining - k + 1, 1)
            if is_revolving:
                projected_balance = start_balance
            else:
                projected_balance = max(start_balance * (1.0 - k / max(remaining, 1)), 0.0)
            projected["balance"] = projected_balance
            hpi_growth = float(macro_values.get("cumulative_house_price_growth", macro_values.get("house_price_growth", 0.0)))
            projected_collateral = max(start_collateral * (1.0 + hpi_growth), 0.0)
            projected["collateral_value"] = projected_collateral
            projected["ltv"] = 1.0 if projected_collateral <= 0 else projected_balance / max(projected_collateral, 1.0)
            projected["dti"] = float(loan["dti"]) + 0.35 * max(float(macro_values.get("unemployment_rate", 0.0)) - 0.045, 0.0)
            for column, value in macro_values.items():
                if column != "horizon_quarter":
                    projected[column] = value
            rows.append(projected)
    return pd.DataFrame(rows).reset_index(drop=True)


def score_forward_pd_paths(
    snapshot: pd.DataFrame,
    model: SurvivalPDModel,
    macro_path: pd.DataFrame,
    max_horizon_quarters: int | None = None,
) -> pd.DataFrame:
    """Score non-constant forward hazard paths and cumulative PDs.

    Summary
    -------
    Produce 12-month and lifetime PDs from projected future hazards instead of
    from a flat current hazard.

    Method
    ------
    The function builds a forward panel, scores each projected row with the
    fitted survival model, then compounds loan-level hazard paths into 12-month
    and lifetime cumulative PDs.

    Parameters
    ----------
    snapshot:
        Current reporting-date loan snapshot.
    model:
        Fitted survival PD model.
    macro_path:
        Scenario macro path used to project future features.
    max_horizon_quarters:
        Optional cap on projected lifetime horizon.

    Returns
    -------
    pandas.DataFrame
        Loan-level table with path-based 12-month and lifetime PD outputs.

    Raises
    ------
    KeyError
        Raised when required projection or model feature columns are missing.

    Notes
    -----
    If the survival model feature specification includes macro columns, scenario
    macro paths enter the PD channel directly. If the model does not include
    macro features, the path still removes constant seasoning, term, balance,
    and LTV assumptions.

    Edge Cases
    ----------
    Loans without projected future rows are absent from the output.

    References
    ----------
    - Singer, J. D., and Willett, J. B. (1993), "It's About Time: Using
      Discrete-Time Survival Analysis to Study Duration and the Timing of
      Events."
    - IFRS Foundation, "IFRS 9 Financial Instruments."
    """

    forward_panel = build_forward_panel(snapshot, macro_path, max_horizon_quarters=max_horizon_quarters)
    if forward_panel.empty:
        return pd.DataFrame(columns=["loan_id", "pd_12m_forward", "pd_lifetime_forward", "mean_forward_hazard", "horizon_quarters"])

    design = _prepare_design_matrix(forward_panel, model.feature_spec, model.design_columns)
    forward_panel["forward_hazard"] = pd.Series(model.result.predict(design), index=forward_panel.index).clip(1e-8, 0.90)

    rows = []
    for loan_id, group in forward_panel.sort_values(["loan_id", "horizon_quarter"]).groupby("loan_id"):
        hazards = group["forward_hazard"]
        rows.append(
            {
                "loan_id": loan_id,
                "pd_12m_forward": cumulative_pd_from_hazard_path(hazards.iloc[:4]),
                "pd_lifetime_forward": cumulative_pd_from_hazard_path(hazards),
                "mean_forward_hazard": float(hazards.mean()),
                "horizon_quarters": int(group["horizon_quarter"].max()),
            }
        )
    return pd.DataFrame(rows)
