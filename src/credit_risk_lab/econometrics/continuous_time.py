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

Simplifications for this lab
------------------------------------------
- The module estimates exposure-count intensities rather than fitting a full
  Cox process.
- Cumulative intensity is aggregated at reporting dates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .markov import ABSORBING_STATES, CREDIT_STATES


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


def _duration_in_time_units(start: pd.Series, end: pd.Series, time_unit_days: float) -> pd.Series:
    if np.issubdtype(start.dtype, np.datetime64) or np.issubdtype(end.dtype, np.datetime64):
        duration_days = (pd.to_datetime(end) - pd.to_datetime(start)).dt.total_seconds() / (24.0 * 60.0 * 60.0)
        return duration_days / float(time_unit_days)
    return end.astype(float) - start.astype(float)


def estimate_default_intensity_from_intervals(
    intervals: pd.DataFrame,
    start_time_column: str,
    end_time_column: str,
    event_column: str = "default_event",
    group_columns: Sequence[str] | None = None,
    time_unit_days: float = 365.25,
) -> pd.DataFrame:
    """Estimate default intensity from exact or interval event-time data.

    Summary
    -------
    Estimate default events per unit exposure time when loans have start and
    end times rather than only quarter-end panel rows.

    Method
    ------
    The function converts each observation interval into a positive duration,
    sums exposure time within optional groups, counts default events, and
    estimates intensity as `events / exposure_time`.

    Parameters
    ----------
    intervals:
        Loan interval table with start time, end time, and default indicator.
    start_time_column:
        Interval start time. Can be numeric or datetime-like.
    end_time_column:
        Interval end time. Can be numeric or datetime-like.
    event_column:
        Binary default event indicator at the interval end.
    group_columns:
        Optional segment or state grouping columns.
    time_unit_days:
        Number of calendar days per time unit when datetime columns are used.

    Returns
    -------
    pandas.DataFrame
        Group-level events, exposure time, and intensity estimates.

    Raises
    ------
    KeyError
        Raised when required columns are missing.
    ValueError
        Raised when durations are negative, zero in aggregate, or
        `time_unit_days <= 0`.

    Notes
    -----
    This is the practical continuous-time upgrade over quarterly exposure
    counts. It is appropriate when origination, delinquency, cure, or default
    dates are observed more finely than quarter-end reporting.

    Edge Cases
    ----------
    Zero-duration rows are ignored because they add no exposure time.

    References
    ----------
    - Andersen, P. K., and Gill, R. D. (1982), "Cox's Regression Model for
      Counting Processes."
    - Lando, D. (1998), "On Cox Processes and Credit Risky Securities."
    """

    if time_unit_days <= 0:
        raise ValueError("time_unit_days must be positive.")
    group_columns = list(group_columns or [])
    required = [start_time_column, end_time_column, event_column, *group_columns]
    missing = [column for column in required if column not in intervals.columns]
    if missing:
        raise KeyError(f"Missing interval intensity columns: {missing}")
    data = intervals.copy()
    data["_duration"] = _duration_in_time_units(data[start_time_column], data[end_time_column], time_unit_days)
    if (data["_duration"] < 0).any():
        raise ValueError("Interval durations must be non-negative.")
    data = data.loc[data["_duration"] > 0].copy()
    data["_event"] = data[event_column].astype(int)
    if group_columns:
        result = data.groupby(group_columns, as_index=False).agg(
            events=("_event", "sum"),
            observations=("_event", "size"),
            exposure_time=("_duration", "sum"),
        )
    else:
        result = pd.DataFrame(
            [
                {
                    "events": int(data["_event"].sum()),
                    "observations": int(len(data)),
                    "exposure_time": float(data["_duration"].sum()),
                }
            ]
        )
    if (result["exposure_time"] <= 0).any():
        raise ValueError("Exposure time must be positive after removing zero-duration rows.")
    result["intensity"] = result["events"] / result["exposure_time"]
    return result


def estimate_ctmc_generator_from_durations(
    transitions: pd.DataFrame,
    duration_column: str,
    origin_state_column: str = "state",
    destination_state_column: str = "next_state",
    states: Sequence[str] = CREDIT_STATES,
    absorbing_states: Sequence[str] = ABSORBING_STATES,
    smoothing: float = 0.0,
) -> pd.DataFrame:
    """Estimate a continuous-time Markov generator from observed durations.

    Summary
    -------
    Fit transition intensities `q_ij` using exact time-at-risk observations for
    credit-state migration.

    Method
    ------
    For each origin state, the maximum-likelihood estimator of an off-diagonal
    CTMC intensity is `q_ij = N_ij / T_i`, where `N_ij` is the number of observed
    jumps from state `i` to state `j` and `T_i` is total exposure time spent in
    state `i`. Diagonals are set to the negative row exit rate. Absorbing states
    are forced to zero rows.

    Parameters
    ----------
    transitions:
        Event-time transition table with origin state, destination state, and
        duration in origin state.
    duration_column:
        Time spent in the origin state before the transition or censoring.
    origin_state_column:
        Origin state column.
    destination_state_column:
        Destination state column.
    states:
        Ordered CTMC states.
    absorbing_states:
        Absorbing terminal states.
    smoothing:
        Optional additive smoothing for transition counts.

    Returns
    -------
    pandas.DataFrame
        Valid CTMC generator with non-negative off-diagonal rates and zero row
        sums.

    Raises
    ------
    KeyError
        Raised when required columns are missing.
    ValueError
        Raised when durations or smoothing are invalid.

    Notes
    -----
    This function implements the exact-event-time version of the Markov
    migration extension. It should be preferred over matrix logarithms when
    transition durations are observed.

    Edge Cases
    ----------
    Non-absorbing states with no exposure receive zero rows, making them
    absorbing for diagnostic purposes rather than fabricating intensities.

    References
    ----------
    - Lando, D., and Skodeberg, T. M. (2002), "Analyzing Rating Transitions and
      Rating Drift with Continuous Observations."
    - Andersen, P. K., and Gill, R. D. (1982), "Cox's Regression Model for
      Counting Processes."
    """

    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")
    required = [duration_column, origin_state_column, destination_state_column]
    missing = [column for column in required if column not in transitions.columns]
    if missing:
        raise KeyError(f"Missing CTMC duration columns: {missing}")
    state_tuple = tuple(states)
    absorbing_tuple = tuple(absorbing_states)
    data = transitions.copy()
    data[duration_column] = data[duration_column].astype(float)
    if (data[duration_column] < 0).any():
        raise ValueError("Durations must be non-negative.")

    generator = pd.DataFrame(0.0, index=state_tuple, columns=state_tuple)
    for state in state_tuple:
        if state in absorbing_tuple:
            continue
        state_data = data.loc[data[origin_state_column].eq(state)]
        exposure = float(state_data[duration_column].sum())
        if exposure <= 0:
            continue
        counts = state_data[destination_state_column].value_counts().reindex(state_tuple, fill_value=0).astype(float)
        counts.loc[state] = 0.0
        if smoothing > 0:
            counts = counts + float(smoothing)
            counts.loc[state] = 0.0
        rates = counts / exposure
        generator.loc[state] = rates
        generator.loc[state, state] = -float(rates.sum())
    return generator


def build_compensated_process_from_intervals(
    intervals: pd.DataFrame,
    intensity: float | Mapping[str, float],
    start_time_column: str,
    end_time_column: str,
    event_column: str = "default_event",
    group_column: str | None = None,
    time_unit_days: float = 365.25,
) -> pd.DataFrame:
    """Build a compensated default process from event-time intervals.

    Summary
    -------
    Compute cumulative defaults, cumulative integrated intensity, and
    compensated defaults using exact or interval-level event timing.

    Method
    ------
    Each interval contributes duration times intensity to the compensator. When
    `intensity` is a mapping, the value is selected by `group_column`; otherwise
    a single portfolio intensity is used. Rows are ordered by interval end time
    before cumulative sums are computed.

    Parameters
    ----------
    intervals:
        Loan interval table.
    intensity:
        Single intensity or mapping from group label to intensity.
    start_time_column:
        Interval start time.
    end_time_column:
        Interval end time.
    event_column:
        Binary default event indicator.
    group_column:
        Optional group column used when intensity is a mapping.
    time_unit_days:
        Number of calendar days per time unit for datetime columns.

    Returns
    -------
    pandas.DataFrame
        Ordered table with defaults, compensator increments, cumulative values,
        and compensated process.

    Raises
    ------
    KeyError
        Raised when required columns are missing.
    ValueError
        Raised when intensities or durations are invalid.

    Notes
    -----
    In an intensity model, `N_t - A_t` should fluctuate around zero if the
    compensator is well calibrated. Persistent drift is evidence that the
    intensity specification misses time, segment, or macro structure.

    Edge Cases
    ----------
    Empty input returns an empty table with the expected output columns.

    References
    ----------
    - Andersen, P. K., and Gill, R. D. (1982), "Cox's Regression Model for
      Counting Processes."
    - Lando, D. (1998), "On Cox Processes and Credit Risky Securities."
    """

    if time_unit_days <= 0:
        raise ValueError("time_unit_days must be positive.")
    required = [start_time_column, end_time_column, event_column]
    if isinstance(intensity, Mapping):
        if group_column is None:
            raise ValueError("group_column is required when intensity is a mapping.")
        required.append(group_column)
        if any(float(value) < 0 for value in intensity.values()):
            raise ValueError("Intensities must be non-negative.")
    elif float(intensity) < 0:
        raise ValueError("Intensity must be non-negative.")
    missing = [column for column in required if column not in intervals.columns]
    if missing:
        raise KeyError(f"Missing compensated-process columns: {missing}")
    if intervals.empty:
        return pd.DataFrame(
            columns=[
                "event_time",
                "duration",
                "default_event",
                "compensator_increment",
                "cumulative_defaults",
                "cumulative_compensator",
                "compensated_process",
            ]
        )

    data = intervals.copy()
    data["duration"] = _duration_in_time_units(data[start_time_column], data[end_time_column], time_unit_days)
    if (data["duration"] < 0).any():
        raise ValueError("Interval durations must be non-negative.")
    if isinstance(intensity, Mapping):
        mapped = data[group_column].astype(str).map({str(key): float(value) for key, value in intensity.items()})
        if mapped.isna().any():
            missing_groups = data.loc[mapped.isna(), group_column].astype(str).unique().tolist()
            raise ValueError(f"Missing intensity for groups: {missing_groups}")
        data["_intensity"] = mapped
    else:
        data["_intensity"] = float(intensity)
    data["default_event"] = data[event_column].astype(int)
    data["compensator_increment"] = data["_intensity"] * data["duration"].astype(float)
    data = data.sort_values(end_time_column).reset_index(drop=True)
    data["cumulative_defaults"] = data["default_event"].cumsum()
    data["cumulative_compensator"] = data["compensator_increment"].cumsum()
    data["compensated_process"] = data["cumulative_defaults"] - data["cumulative_compensator"]
    data["event_time"] = data[end_time_column]
    return data[
        [
            "event_time",
            "duration",
            "default_event",
            "compensator_increment",
            "cumulative_defaults",
            "cumulative_compensator",
            "compensated_process",
        ]
    ]
