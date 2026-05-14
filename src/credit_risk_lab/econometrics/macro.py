"""Macroeconometric scenario-path utilities.

This module provides a lightweight time-series layer for the credit-risk lab:
AR(1) macro projections and scenario-path construction. It creates a route for
feeding macro variables directly into PD hazard paths rather than applying a
second opaque PD multiplier inside the ECL layer.

Assumptions
-----------
- Macro variables are observed at the same quarterly frequency as the synthetic
  portfolio.
- Each macro series is projected with a univariate AR(1) model in this first
  version.
- Scenario shocks are transparent level or growth shifts layered on top of the
  statistical forecast.

Primary references
------------------
- Box, Jenkins, Reinsel, and Ljung, "Time Series Analysis: Forecasting and
  Control."
- IFRS Foundation, "IFRS 9 Financial Instruments", for forward-looking
  scenario use in expected credit loss.

Simplifications for this lab
------------------------------------------
- Cross-variable dynamics are not modelled with a VAR in v1.
- Scenario paths are deterministic conditional on the fitted AR(1) projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


@dataclass(slots=True)
class MacroAR1Model:
    """Container for independent AR(1) macro forecasts."""

    coefficients: pd.DataFrame
    columns: tuple[str, ...]


@dataclass(slots=True)
class MacroVARModel:
    """Container for a first-order vector autoregression."""

    intercept: pd.Series
    coefficients: pd.DataFrame
    columns: tuple[str, ...]
    last_values: pd.Series


def fit_ar1_macro_model(macro_history: pd.DataFrame, columns: Sequence[str]) -> MacroAR1Model:
    """Fit independent AR(1) models to macro time series.

    Summary
    -------
    Estimate one autoregressive coefficient set per macro variable.

    Method
    ------
    For each series `x_t`, ordinary least squares estimates
    `x_t = alpha + phi * x_(t-1) + error_t`. The fitted coefficients are stored
    in a compact table for deterministic forecasting.

    Parameters
    ----------
    macro_history:
        Macro history ordered or orderable by `snapshot_date`.
    columns:
        Macro columns to model.

    Returns
    -------
    MacroAR1Model
        Fitted AR(1) coefficients and modelled column names.

    Raises
    ------
    KeyError
        Raised when requested macro columns are missing.
    ValueError
        Raised when fewer than three observations are available.

    Notes
    -----
    AR(1) is not the final macro model. It is a transparent first time-series
    layer that makes the repo ready for macro-sensitive PD paths.

    Edge Cases
    ----------
    A constant series receives `phi=0` and forecasts at its latest level.

    References
    ----------
    - Box, Jenkins, Reinsel, and Ljung, "Time Series Analysis: Forecasting and
      Control."
    """

    missing = [column for column in columns if column not in macro_history.columns]
    if missing:
        raise KeyError(f"Missing macro columns: {missing}")
    if len(macro_history) < 3:
        raise ValueError("At least three macro observations are required for AR(1) fitting.")

    ordered = macro_history.sort_values("snapshot_date") if "snapshot_date" in macro_history.columns else macro_history.copy()
    rows = []
    for column in columns:
        series = ordered[column].astype(float).reset_index(drop=True)
        y = series.iloc[1:].to_numpy()
        x_lag = series.iloc[:-1].to_numpy()
        if np.isclose(np.var(x_lag), 0.0):
            alpha = float(series.iloc[-1])
            phi = 0.0
            sigma = float(np.std(y - alpha)) if len(y) else 0.0
        else:
            design = np.column_stack([np.ones_like(x_lag), x_lag])
            alpha, phi = np.linalg.lstsq(design, y, rcond=None)[0]
            sigma = float(np.std(y - (alpha + phi * x_lag), ddof=1)) if len(y) > 2 else 0.0
        rows.append({"variable": column, "alpha": float(alpha), "phi": float(phi), "sigma": sigma, "last_value": float(series.iloc[-1])})
    return MacroAR1Model(coefficients=pd.DataFrame(rows), columns=tuple(columns))


def forecast_macro_path(model: MacroAR1Model, horizon_quarters: int) -> pd.DataFrame:
    """Forecast a baseline macro path from an AR(1) model.

    Summary
    -------
    Project modelled macro variables over a future quarterly horizon.

    Method
    ------
    Starting from each series' last observed value, the AR(1) recursion
    `x_(t+1) = alpha + phi * x_t` is iterated forward.

    Parameters
    ----------
    model:
        Fitted `MacroAR1Model`.
    horizon_quarters:
        Number of future quarters to forecast.

    Returns
    -------
    pandas.DataFrame
        Baseline forecast path with `horizon_quarter`.

    Raises
    ------
    ValueError
        Raised when `horizon_quarters` is not positive.

    Notes
    -----
    Forecast uncertainty is deliberately not simulated here; the scenario layer
    handles deterministic upside/downside shifts.

    Edge Cases
    ----------
    Explosive fitted `phi` values are clipped to `[-0.99, 0.99]` for forecast
    stability in this portfolio implementation.

    References
    ----------
    - Box, Jenkins, Reinsel, and Ljung, "Time Series Analysis: Forecasting and
      Control."
    """

    if horizon_quarters <= 0:
        raise ValueError("horizon_quarters must be positive.")
    coeffs = model.coefficients.set_index("variable")
    state = coeffs["last_value"].to_dict()
    rows = []
    for horizon in range(1, horizon_quarters + 1):
        row: dict[str, float | int] = {"horizon_quarter": horizon}
        for column in model.columns:
            alpha = float(coeffs.loc[column, "alpha"])
            phi = float(np.clip(coeffs.loc[column, "phi"], -0.99, 0.99))
            state[column] = alpha + phi * state[column]
            row[column] = float(state[column])
        rows.append(row)
    return pd.DataFrame(rows)


def build_macro_scenario_paths(
    baseline_path: pd.DataFrame,
    shocks: Mapping[str, Mapping[str, float]],
) -> dict[str, pd.DataFrame]:
    """Create named macro scenario paths from a baseline path.

    Summary
    -------
    Add transparent scenario shocks to a baseline macro forecast.

    Method
    ------
    For each scenario, the function copies the baseline path and adds the
    provided column-level shocks. It also creates `cumulative_house_price_growth`
    when `house_price_growth` is available, which is useful for collateral/LTV
    projections.

    Parameters
    ----------
    baseline_path:
        Baseline forecast path with `horizon_quarter`.
    shocks:
        Mapping from scenario name to `{column: additive_shift}`.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Named scenario paths.

    Raises
    ------
    KeyError
        Raised when `baseline_path` lacks `horizon_quarter`.

    Notes
    -----
    This keeps one primary macro channel for PD: future macro variables can be
    scored through the PD model instead of multiplying PD again in ECL.

    Edge Cases
    ----------
    Shocks for columns absent from the baseline create new constant-shift
    columns, allowing simple management overlays to be represented explicitly.

    References
    ----------
    - IFRS Foundation, "IFRS 9 Financial Instruments."
    """

    if "horizon_quarter" not in baseline_path.columns:
        raise KeyError("baseline_path must contain 'horizon_quarter'.")
    paths: dict[str, pd.DataFrame] = {}
    for scenario, scenario_shocks in shocks.items():
        path = baseline_path.copy()
        for column, shift in scenario_shocks.items():
            base = path[column] if column in path.columns else 0.0
            path[column] = base + float(shift)
        if "house_price_growth" in path.columns:
            path["cumulative_house_price_growth"] = (1.0 + path["house_price_growth"].astype(float)).cumprod() - 1.0
        path["scenario"] = scenario
        paths[scenario] = path
    return paths


def fit_var_macro_model(macro_history: pd.DataFrame, columns: Sequence[str]) -> MacroVARModel:
    """Fit a first-order vector autoregression to macro variables.

    Summary
    -------
    Estimate a transparent VAR(1) model so macro variables can influence each
    other's scenario paths.

    Method
    ------
    The model estimates `x_t = c + A x_(t-1) + e_t` by ordinary least squares
    across the selected macro variables. Coefficients are stored in a matrix
    whose rows are target variables and columns are lagged predictors.

    Parameters
    ----------
    macro_history:
        Macro history ordered or orderable by `snapshot_date`.
    columns:
        Macro variables to include in the VAR.

    Returns
    -------
    MacroVARModel
        Fitted intercept vector, coefficient matrix, columns, and latest state.

    Raises
    ------
    KeyError
        Raised when requested columns are missing.
    ValueError
        Raised when fewer than three observations are available.

    Notes
    -----
    A VAR is useful when unemployment, rates, GDP, and collateral values are not
    plausibly independent. It should be compared against simpler AR(1) paths
    because small samples can make VAR estimates unstable.

    Edge Cases
    ----------
    Least-squares estimation is used even if the system is over-parameterised;
    callers should inspect forecast stability.

    References
    ----------
    - Lutkepohl, "New Introduction to Multiple Time Series Analysis."
    - Box, Jenkins, Reinsel, and Ljung, "Time Series Analysis: Forecasting and
      Control."
    """

    missing = [column for column in columns if column not in macro_history.columns]
    if missing:
        raise KeyError(f"Missing macro columns: {missing}")
    if len(macro_history) < 3:
        raise ValueError("At least three macro observations are required for VAR(1) fitting.")
    ordered = macro_history.sort_values("snapshot_date") if "snapshot_date" in macro_history.columns else macro_history.copy()
    matrix = ordered[list(columns)].astype(float).to_numpy()
    y = matrix[1:, :]
    x_lag = matrix[:-1, :]
    design = np.column_stack([np.ones(len(x_lag)), x_lag])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    intercept = pd.Series(beta[0, :], index=columns, name="intercept")
    coefficients = pd.DataFrame(beta[1:, :].T, index=columns, columns=columns)
    last_values = pd.Series(matrix[-1, :], index=columns, name="last_values")
    return MacroVARModel(intercept=intercept, coefficients=coefficients, columns=tuple(columns), last_values=last_values)


def forecast_var_macro_path(model: MacroVARModel, horizon_quarters: int, coefficient_clip: float = 1.25) -> pd.DataFrame:
    """Forecast a macro path from a VAR(1) model.

    Summary
    -------
    Project interacting macro variables over a future quarterly horizon.

    Method
    ------
    Starting from the last observed macro vector, the function iterates
    `x_(t+1) = c + A x_t`. Coefficients are clipped elementwise for numerical
    stability in small synthetic samples.

    Parameters
    ----------
    model:
        Fitted VAR(1) model.
    horizon_quarters:
        Number of quarters to forecast.
    coefficient_clip:
        Absolute coefficient cap used for stable projections.

    Returns
    -------
    pandas.DataFrame
        Baseline VAR forecast path with `horizon_quarter`.

    Raises
    ------
    ValueError
        Raised when horizon or coefficient cap is invalid.

    Notes
    -----
    The VAR layer is optional. If cross-macro dynamics are weak or unstable, the
    simpler AR(1) path may be preferable.

    Edge Cases
    ----------
    Horizon one returns a single projected row.

    References
    ----------
    - Lutkepohl, "New Introduction to Multiple Time Series Analysis."
    """

    if horizon_quarters <= 0:
        raise ValueError("horizon_quarters must be positive.")
    if coefficient_clip <= 0:
        raise ValueError("coefficient_clip must be positive.")
    state = model.last_values.astype(float).to_numpy()
    intercept = model.intercept.astype(float).to_numpy()
    coefficients = model.coefficients.astype(float).clip(-coefficient_clip, coefficient_clip).to_numpy()
    rows: list[dict[str, float | int]] = []
    for horizon in range(1, horizon_quarters + 1):
        state = intercept + coefficients @ state
        row: dict[str, float | int] = {"horizon_quarter": horizon}
        row.update({column: float(value) for column, value in zip(model.columns, state, strict=True)})
        rows.append(row)
    return pd.DataFrame(rows)
