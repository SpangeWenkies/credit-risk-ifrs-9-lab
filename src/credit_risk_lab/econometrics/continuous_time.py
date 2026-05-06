"""Continuous-time default intensity and counting-process utilities.

This module creates a practical bridge from quarterly credit panels to
continuous-time default modelling. It estimates piecewise-constant default
intensities, builds default counting-process tables, and computes compensated
default processes of the form `M_t = N_t - A_t`, where `A_t` is cumulative
integrated intensity.

Assumptions
-----------
- Exposure time is measured in quarters unless the caller rescales it.
- Intensities are piecewise constant within a requested segment or portfolio
  group.
- The compensated process is a diagnostic approximation because the synthetic
  portfolio does not contain exact intraperiod default times.

Primary references
------------------
- Andersen and Gill (1982), "Cox's Regression Model for Counting Processes."
- Aalen, Borgan, and Gjessing, "Survival and Event History Analysis."
- Lando (1998), "On Cox Processes and Credit Risky Securities."

Simplifications for this portfolio project
------------------------------------------
- The module estimates exposure-count intensities rather than fitting a full
  Cox process.
- Cumulative intensity is aggregated at reporting dates.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def estimate_piecewise_default_intensity(
    performance: pd.DataFrame,
    group_columns: Sequence[str] | None = None,
    event_column: str = "default_next_period",
    exposure_time: float = 1.0,
) -> pd.DataFrame:
    """Estimate piecewise-constant default intensities.

    Summary
    -------
    Estimate default event rates per unit exposure time for portfolio groups.

    Method
    ------
    The function counts defaults and exposure rows within each group. Intensity
    is estimated as `events / exposure_time_at_risk`, with exposure time equal
    to row count times `exposure_time`.

    Parameters
    ----------
    performance:
        Loan-period performance panel.
    group_columns:
        Optional grouping columns such as `segment` or `rating_rank`.
    event_column:
        Binary indicator for default in the next interval.
    exposure_time:
        Time represented by one row, in quarters or years.

    Returns
    -------
    pandas.DataFrame
        Group-level event counts, exposure time, and intensity estimates.

    Raises
    ------
    KeyError
        Raised when required columns are missing.
    ValueError
        Raised when `exposure_time <= 0`.

    Notes
    -----
    This is the simplest reduced-form intensity estimator and is useful for
    explaining how a discrete panel can approximate continuous-time default
    modelling.

    Edge Cases
    ----------
    If no groups are supplied, a single portfolio-level intensity is returned.

    References
    ----------
    - Andersen, P. K., and Gill, R. D. (1982), "Cox's Regression Model for
      Counting Processes."
    - Lando, D. (1998), "On Cox Processes and Credit Risky Securities."
    """

    if exposure_time <= 0:
        raise ValueError("exposure_time must be positive.")
    group_columns = list(group_columns or [])
    missing = [column for column in [event_column, *group_columns] if column not in performance.columns]
    if missing:
        raise KeyError(f"Missing intensity columns: {missing}")

    data = performance.copy()
    data["_exposure_time"] = float(exposure_time)
    data["_event"] = data[event_column].astype(int)
    if group_columns:
        result = (
            data.groupby(group_columns, as_index=False)
            .agg(events=("_event", "sum"), observations=("_event", "size"), exposure_time=("_exposure_time", "sum"))
        )
    else:
        result = pd.DataFrame(
            [
                {
                    "events": int(data["_event"].sum()),
                    "observations": int(len(data)),
                    "exposure_time": float(data["_exposure_time"].sum()),
                }
            ]
        )
    result["intensity"] = result["events"] / result["exposure_time"].clip(lower=1e-12)
    return result


def build_default_counting_process(
    performance: pd.DataFrame,
    intensity: float,
    date_column: str = "snapshot_date",
    event_column: str = "default_next_period",
    exposure_time: float = 1.0,
) -> pd.DataFrame:
    """Build a portfolio default counting-process diagnostic table.

    Summary
    -------
    Construct observed defaults, cumulative defaults, cumulative compensator,
    and compensated default process over reporting dates.

    Method
    ------
    At each reporting date, the function counts at-risk rows and default events.
    The compensator increment is `intensity * exposure_count * exposure_time`.
    The compensated process is cumulative defaults minus cumulative compensator.

    Parameters
    ----------
    performance:
        Loan-period performance panel.
    intensity:
        Piecewise-constant default intensity per exposure-time unit.
    date_column:
        Reporting date column.
    event_column:
        Binary default indicator.
    exposure_time:
        Time represented by one row.

    Returns
    -------
    pandas.DataFrame
        Date-level counting-process diagnostic table.

    Raises
    ------
    KeyError
        Raised when required columns are missing.
    ValueError
        Raised when `intensity` is negative or `exposure_time <= 0`.

    Notes
    -----
    In a true intensity model, the compensated process is a martingale under the
    model filtration. Here it is used as a portfolio diagnostic because default
    times are observed only at interval level.

    Edge Cases
    ----------
    Empty input returns an empty table with the expected columns.

    References
    ----------
    - Andersen, P. K., and Gill, R. D. (1982), "Cox's Regression Model for
      Counting Processes."
    - Aalen, Borgan, and Gjessing, "Survival and Event History Analysis."
    """

    if intensity < 0:
        raise ValueError("intensity must be non-negative.")
    if exposure_time <= 0:
        raise ValueError("exposure_time must be positive.")
    missing = [column for column in (date_column, event_column) if column not in performance.columns]
    if missing:
        raise KeyError(f"Missing counting-process columns: {missing}")
    if performance.empty:
        return pd.DataFrame(
            columns=[
                "snapshot_date",
                "exposure_count",
                "defaults",
                "cumulative_defaults",
                "compensator",
                "cumulative_compensator",
                "compensated_process",
            ]
        )

    grouped = (
        performance.groupby(date_column, as_index=False)
        .agg(exposure_count=(event_column, "size"), defaults=(event_column, "sum"))
        .sort_values(date_column)
    )
    grouped["compensator"] = float(intensity) * grouped["exposure_count"] * float(exposure_time)
    grouped["cumulative_defaults"] = grouped["defaults"].cumsum()
    grouped["cumulative_compensator"] = grouped["compensator"].cumsum()
    grouped["compensated_process"] = grouped["cumulative_defaults"] - grouped["cumulative_compensator"]
    return grouped.rename(columns={date_column: "snapshot_date"})


def survival_probability_from_intensity(intensity_path: Sequence[float] | pd.Series, step_length: float = 1.0) -> float:
    """Convert an intensity path into survival probability.

    Summary
    -------
    Compute continuous-time survival under a piecewise-constant intensity path.

    Method
    ------
    For default intensity `lambda_t`, survival is
    `exp(-integral lambda_t dt)`. With piecewise-constant hazards this becomes
    `exp(-sum(lambda_k * step_length))`.

    Parameters
    ----------
    intensity_path:
        Sequence of non-negative intensities.
    step_length:
        Length of each intensity interval.

    Returns
    -------
    float
        Survival probability over the path.

    Raises
    ------
    ValueError
        Raised when intensities are negative or `step_length <= 0`.

    Notes
    -----
    This is the continuous-time analogue of compounding discrete survival
    probabilities.

    Edge Cases
    ----------
    An empty path returns one because no time at risk has elapsed.

    References
    ----------
    - Lando, D. (1998), "On Cox Processes and Credit Risky Securities."
    """

    if step_length <= 0:
        raise ValueError("step_length must be positive.")
    intensities = pd.Series(intensity_path, dtype=float)
    if intensities.empty:
        return 1.0
    if (intensities < 0).any():
        raise ValueError("Intensities must be non-negative.")
    return float(np.exp(-float(intensities.sum()) * float(step_length)))
