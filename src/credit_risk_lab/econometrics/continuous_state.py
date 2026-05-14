"""Continuous-state credit-quality process diagnostics.

This module implements a practical continuous-state extension of the Markov
credit migration lab. The state variable is a latent credit-quality coordinate:
higher values mean healthier credit quality, and a lower boundary represents
default. The implementation supports a finite-difference generator for a
diffusion-like credit-quality process, optional jump-to-default and jump-down
mechanisms, semigroup default probabilities, and Beurling-Deny-style diagnostic
decomposition into local, jump, and killing energy components.

Assumptions
-----------
- The latent credit-quality state is one-dimensional and observed only through
  a modelling grid.
- The generator is a diagnostic approximation for studying continuous-state
  ideas, not a production structural credit model.
- Default is represented as an absorbing cemetery state.
- Capacity, polar-set, Cheeger-energy, and regularity language is used as a
  finite-grid proxy so the repo can connect course theory to credit practice
  without overclaiming infinite-dimensional results.

Primary references
------------------
- Fukushima, Oshima, and Takeda, "Dirichlet Forms and Symmetric Markov
  Processes", 2nd revised and extended edition, 2011.
- Ma and Roeckner, "Introduction to the Theory of (Non-Symmetric) Dirichlet
  Forms", 1992.
- Black and Cox (1976), "Valuing Corporate Securities: Some Effects of Bond
  Indenture Provisions."
- Lando (1998), "On Cox Processes and Credit Risky Securities."
- Duffie and Singleton (1999), "Modeling Term Structures of Defaultable Bonds."

Simplifications for this lab
------------------------------------------
- Local diffusion is represented by a nearest-neighbour upwind finite-difference
  CTMC generator on a grid.
- Capacity and Cheeger energy are reported as finite-grid proxies rather than
  theorem-level analytic quantities.
- The Beurling-Deny decomposition is used as a comparison vocabulary for model
  components: local movement, jumps, and killing/default.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import expm


DEFAULT_STATE = "default"


@dataclass(slots=True)
class ContinuousCreditStateGrid:
    """One-dimensional latent credit-quality grid."""

    grid: np.ndarray
    active_grid: np.ndarray
    default_boundary: float
    dx: float
    default_state: str = DEFAULT_STATE


def build_continuous_credit_state_grid(
    x_min: float = -4.0,
    x_max: float = 4.0,
    n_points: int = 101,
    default_boundary: float = -2.0,
    default_state: str = DEFAULT_STATE,
) -> ContinuousCreditStateGrid:
    """Build a latent continuous credit-quality grid.

    Summary
    -------
    Create a one-dimensional state grid for a continuous-state credit-quality
    process with a default boundary.

    Method
    ------
    The function creates an evenly spaced grid, keeps active states strictly
    above the default boundary, and records the grid spacing for finite
    differences. Points at or below the boundary are represented by one
    absorbing cemetery state.

    Parameters
    ----------
    x_min:
        Lower grid endpoint.
    x_max:
        Upper grid endpoint.
    n_points:
        Number of grid points before the default boundary collapse.
    default_boundary:
        Latent state threshold at or below which default occurs.
    default_state:
        Label for the absorbing default state.

    Returns
    -------
    ContinuousCreditStateGrid
        Grid metadata with active states and spacing.

    Raises
    ------
    ValueError
        Raised when grid endpoints, point count, or boundary are invalid.

    Notes
    -----
    The grid is a practical bridge from continuous credit theory to code. It
    lets the repo study local diffusion, jump, and killing components without
    pretending that synthetic quarterly data identify a full continuous process.

    Edge Cases
    ----------
    If the boundary is outside the grid, a `ValueError` is raised because there
    would be no meaningful active/default split.

    References
    ----------
    - Black, F., and Cox, J. C. (1976), "Valuing Corporate Securities: Some
      Effects of Bond Indenture Provisions."
    - Fukushima, Oshima, and Takeda, "Dirichlet Forms and Symmetric Markov
      Processes", 2nd revised and extended edition, 2011.
    """

    if n_points < 5:
        raise ValueError("n_points must be at least 5.")
    if x_min >= x_max:
        raise ValueError("Require x_min < x_max.")
    if not (x_min < default_boundary < x_max):
        raise ValueError("default_boundary must lie strictly inside the grid range.")
    grid = np.linspace(float(x_min), float(x_max), int(n_points))
    dx = float(grid[1] - grid[0])
    active = grid[grid > float(default_boundary)]
    if len(active) < 3:
        raise ValueError("At least three active grid points are required.")
    return ContinuousCreditStateGrid(grid=grid, active_grid=active, default_boundary=float(default_boundary), dx=dx, default_state=default_state)


def _state_label(value: float) -> str:
    return f"x={value:.6g}"


def ou_credit_quality_generator(
    grid: ContinuousCreditStateGrid,
    mean_reversion: float = 0.35,
    long_run_mean: float = 0.5,
    volatility: float = 0.45,
    killing_intensity: float = 0.0,
    downward_jump_intensity: float = 0.0,
    downward_jump_size: float = 0.75,
) -> pd.DataFrame:
    """Construct a finite-difference generator for latent credit quality.

    Summary
    -------
    Build a continuous-time generator for a latent credit-quality process with
    local diffusion, optional downward jumps, and optional killing/default.

    Method
    ------
    The active latent state follows an Ornstein-Uhlenbeck-style drift
    `kappa(theta - x)` and volatility `sigma`. A monotone upwind finite
    difference scheme converts the local diffusion into nearest-neighbour
    transition rates. Optional downward jumps move the process to a lower grid
    point, and killing intensity sends mass directly to default. The default
    state is absorbing.

    Parameters
    ----------
    grid:
        Continuous credit-quality grid.
    mean_reversion:
        Speed of pull toward `long_run_mean`.
    long_run_mean:
        Long-run latent credit quality.
    volatility:
        Local diffusion volatility.
    killing_intensity:
        Direct jump-to-default intensity.
    downward_jump_intensity:
        Intensity of non-default downward jumps.
    downward_jump_size:
        Size of downward jumps in latent-state units.

    Returns
    -------
    pandas.DataFrame
        CTMC generator over active grid labels plus the absorbing default state.

    Raises
    ------
    ValueError
        Raised when model parameters are outside valid ranges.

    Notes
    -----
    This generator is useful for comparing credit processes by component: local
    deterioration/improvement, sudden jumps, and default killing. It is not a
    fitted structural model.

    Edge Cases
    ----------
    If a local or jump move crosses the default boundary, its rate is assigned
    to the absorbing default state.

    References
    ----------
    - Black, F., and Cox, J. C. (1976), "Valuing Corporate Securities: Some
      Effects of Bond Indenture Provisions."
    - Lando, D. (1998), "On Cox Processes and Credit Risky Securities."
    - Duffie, D., and Singleton, K. J. (1999), "Modeling Term Structures of
      Defaultable Bonds."
    """

    if mean_reversion < 0:
        raise ValueError("mean_reversion must be non-negative.")
    if volatility < 0:
        raise ValueError("volatility must be non-negative.")
    if killing_intensity < 0 or downward_jump_intensity < 0:
        raise ValueError("Intensities must be non-negative.")
    if downward_jump_size < 0:
        raise ValueError("downward_jump_size must be non-negative.")

    active = grid.active_grid
    labels = [_state_label(value) for value in active] + [grid.default_state]
    generator = pd.DataFrame(0.0, index=labels, columns=labels)
    diffusion_rate = 0.5 * float(volatility) ** 2 / (grid.dx**2)

    for idx, x_value in enumerate(active):
        label = _state_label(x_value)
        drift = float(mean_reversion) * (float(long_run_mean) - float(x_value))
        up_rate = diffusion_rate + max(drift, 0.0) / grid.dx
        down_rate = diffusion_rate + max(-drift, 0.0) / grid.dx

        if idx + 1 < len(active):
            generator.loc[label, _state_label(active[idx + 1])] += up_rate
        else:
            generator.loc[label, label] += up_rate

        if idx - 1 >= 0:
            generator.loc[label, _state_label(active[idx - 1])] += down_rate
        else:
            generator.loc[label, grid.default_state] += down_rate

        if downward_jump_intensity > 0 and downward_jump_size > 0:
            target = float(x_value) - float(downward_jump_size)
            if target <= grid.default_boundary:
                generator.loc[label, grid.default_state] += float(downward_jump_intensity)
            else:
                target_idx = int(np.searchsorted(active, target, side="right") - 1)
                target_idx = max(0, min(target_idx, len(active) - 1))
                generator.loc[label, _state_label(active[target_idx])] += float(downward_jump_intensity)

        generator.loc[label, grid.default_state] += float(killing_intensity)
        off_diagonal_sum = float(generator.loc[label].sum())
        generator.loc[label, label] -= off_diagonal_sum

    return generator


def default_probability_from_generator(
    generator: pd.DataFrame,
    start_state: str,
    horizons: Sequence[float],
    default_state: str = DEFAULT_STATE,
) -> pd.DataFrame:
    """Compute default probabilities from a continuous-time generator.

    Summary
    -------
    Use the CTMC semigroup `exp(tQ)` to obtain default probabilities over
    continuous horizons.

    Method
    ------
    For each horizon, the function computes `P(t) = exp(tQ)` and reads the
    transition probability from `start_state` to `default_state`.

    Parameters
    ----------
    generator:
        Valid CTMC generator.
    start_state:
        Starting latent grid state.
    horizons:
        Non-negative time horizons.
    default_state:
        Absorbing default state label.

    Returns
    -------
    pandas.DataFrame
        Horizon-level default probabilities and survival probabilities.

    Raises
    ------
    KeyError
        Raised when states are absent.
    ValueError
        Raised when horizons are negative or the generator labels are invalid.

    Notes
    -----
    This is the continuous-time analogue of raising a discrete transition
    matrix to integer powers.

    Edge Cases
    ----------
    Horizon zero returns current-state probabilities.

    References
    ----------
    - Lando, D. (1998), "On Cox Processes and Credit Risky Securities."
    - Duffie, D., and Singleton, K. J. (1999), "Modeling Term Structures of
      Defaultable Bonds."
    """

    if list(generator.index) != list(generator.columns):
        raise ValueError("Generator index and columns must match.")
    if start_state not in generator.index:
        raise KeyError(f"Unknown start_state: {start_state}")
    if default_state not in generator.columns:
        raise KeyError(f"Unknown default_state: {default_state}")
    horizon_values = [float(horizon) for horizon in horizons]
    if any(horizon < 0 for horizon in horizon_values):
        raise ValueError("Horizons must be non-negative.")
    q = generator.to_numpy(dtype=float)
    start_idx = list(generator.index).index(start_state)
    default_idx = list(generator.columns).index(default_state)
    rows: list[dict[str, float]] = []
    for horizon in horizon_values:
        transition = expm(q * horizon)
        default_probability = float(np.clip(transition[start_idx, default_idx], 0.0, 1.0))
        rows.append(
            {
                "horizon": horizon,
                "default_probability": default_probability,
                "survival_probability": 1.0 - default_probability,
            }
        )
    return pd.DataFrame(rows)


def beurling_deny_credit_decomposition(
    generator: pd.DataFrame,
    state_values: Mapping[str, float] | pd.Series,
    reference_measure: Mapping[str, float] | pd.Series | None = None,
    default_state: str = DEFAULT_STATE,
    cemetery_value: float = 0.0,
) -> dict[str, float]:
    """Decompose finite-grid credit energy into local, jump, and killing parts.

    Summary
    -------
    Report Beurling-Deny-style components for a credit score defined on a
    continuous-state generator grid.

    Method
    ------
    Off-diagonal rates between adjacent numeric grid labels are treated as the
    local diffusion proxy. Non-adjacent active-state rates are treated as jump
    energy. Rates into the default cemetery state are treated as killing energy
    with cemetery value zero by default.

    Parameters
    ----------
    generator:
        CTMC generator over numeric grid-state labels and a default state.
    state_values:
        Scalar score or risk surface defined on active states. A default-state
        value is optional and defaults to `cemetery_value`.
    reference_measure:
        Optional state weights. Uniform active-state weights are used when
        omitted.
    default_state:
        Absorbing default state label.
    cemetery_value:
        Score value assigned to the cemetery state for killing energy.

    Returns
    -------
    dict[str, float]
        Local, jump, killing, and total energy components.

    Raises
    ------
    ValueError
        Raised when generator labels, scores, or weights are invalid.

    Notes
    -----
    This is a finite-grid diagnostic inspired by the Beurling-Deny
    decomposition. It helps compare models: a diffusion-dominated model should
    have most energy in the local part, while a sudden-deterioration model has
    material jump or killing energy.

    Edge Cases
    ----------
    If there are no non-adjacent jumps or no killing rates, the corresponding
    component is zero.

    References
    ----------
    - Beurling and Deny, "Dirichlet Spaces", 1958-1959.
    - Fukushima, Oshima, and Takeda, "Dirichlet Forms and Symmetric Markov
      Processes", 2nd revised and extended edition, 2011.
    """

    if list(generator.index) != list(generator.columns):
        raise ValueError("Generator index and columns must match.")
    if default_state not in generator.index:
        raise ValueError(f"Missing default state: {default_state}")
    active_states = [state for state in generator.index if state != default_state]
    values = pd.Series(state_values, dtype=float).reindex(active_states)
    if values.isna().any():
        missing = values.index[values.isna()].tolist()
        raise ValueError(f"Missing state values for: {missing}")
    full_values = values.to_dict()
    full_values[default_state] = float(pd.Series(state_values, dtype=float).get(default_state, cemetery_value))

    if reference_measure is None:
        weights = pd.Series(1.0 / len(active_states), index=active_states, dtype=float)
    else:
        weights = pd.Series(reference_measure, dtype=float).reindex(active_states)
        if weights.isna().any():
            missing = weights.index[weights.isna()].tolist()
            raise ValueError(f"Missing reference weights for: {missing}")
        if (weights < 0).any() or float(weights.sum()) <= 0:
            raise ValueError("Reference weights must be non-negative and have positive mass.")
        weights = weights / float(weights.sum())

    local = 0.0
    jump = 0.0
    killing = 0.0
    ordered = active_states
    index_lookup = {state: idx for idx, state in enumerate(ordered)}
    for origin in active_states:
        for destination in generator.columns:
            if origin == destination:
                continue
            rate = float(generator.loc[origin, destination])
            if rate <= 0:
                continue
            diff = float(full_values[origin] - full_values[destination])
            contribution = 0.5 * float(weights.loc[origin]) * rate * diff * diff
            if destination == default_state:
                killing += contribution
            elif abs(index_lookup[origin] - index_lookup[destination]) == 1:
                local += contribution
            else:
                jump += contribution
    return {
        "local_energy": float(local),
        "jump_energy": float(jump),
        "killing_energy": float(killing),
        "total_energy": float(local + jump + killing),
    }


def default_boundary_capacity_proxy(
    generator: pd.DataFrame,
    reference_measure: Mapping[str, float] | pd.Series | None = None,
    default_state: str = DEFAULT_STATE,
) -> dict[str, float]:
    """Compute a finite-grid proxy for default-boundary capacity.

    Summary
    -------
    Measure how dynamically visible the default boundary is from nearby active
    states.

    Method
    ------
    The proxy sums reference-weighted direct rates into the default cemetery
    state. In continuous Dirichlet-form theory, capacity measures whether a set
    is seen by the process; on this grid, direct default conductance is the
    practical observable analogue.

    Parameters
    ----------
    generator:
        CTMC generator with an absorbing default state.
    reference_measure:
        Optional active-state weights.
    default_state:
        Default cemetery label.

    Returns
    -------
    dict[str, float]
        Capacity proxy, number of states connected to default, and total direct
        default rate.

    Raises
    ------
    ValueError
        Raised when generator or weights are invalid.

    Notes
    -----
    A near-zero proxy says the boundary is almost invisible in the chosen
    generator. A high proxy says default can be reached directly from material
    active states.

    Edge Cases
    ----------
    The proxy is zero when no active state has a direct default rate.

    References
    ----------
    - Fukushima, Oshima, and Takeda, "Dirichlet Forms and Symmetric Markov
      Processes", 2nd revised and extended edition, 2011.
    """

    if default_state not in generator.columns:
        raise ValueError(f"Missing default state: {default_state}")
    active_states = [state for state in generator.index if state != default_state]
    if reference_measure is None:
        weights = pd.Series(1.0 / len(active_states), index=active_states, dtype=float)
    else:
        weights = pd.Series(reference_measure, dtype=float).reindex(active_states)
        if weights.isna().any():
            missing = weights.index[weights.isna()].tolist()
            raise ValueError(f"Missing reference weights for: {missing}")
        if (weights < 0).any() or float(weights.sum()) <= 0:
            raise ValueError("Reference weights must be non-negative and have positive mass.")
        weights = weights / float(weights.sum())
    rates = generator.loc[active_states, default_state].astype(float).clip(lower=0.0)
    return {
        "capacity_proxy": float((weights * rates).sum()),
        "connected_active_states": int((rates > 0).sum()),
        "total_direct_default_rate": float(rates.sum()),
    }


def polar_state_diagnostic(
    generator: pd.DataFrame,
    target_state: str,
    reference_measure: Mapping[str, float] | pd.Series | None = None,
    threshold: float = 1e-10,
) -> dict[str, object]:
    """Flag whether a state is polar-like on the finite credit grid.

    Summary
    -------
    Check whether a target state receives negligible incoming transition mass.

    Method
    ------
    The diagnostic sums incoming active-state rates into the target state. With
    optional weights, it also computes weighted incoming conductance. A state is
    labelled polar-like when both are below the threshold.

    Parameters
    ----------
    generator:
        CTMC generator.
    target_state:
        State to test.
    reference_measure:
        Optional active-state weights.
    threshold:
        Numerical threshold for polar-like classification.

    Returns
    -------
    dict[str, object]
        Incoming rate totals and polar-like flag.

    Raises
    ------
    KeyError
        Raised when `target_state` is absent.
    ValueError
        Raised when threshold or weights are invalid.

    Notes
    -----
    In the full theory, polar sets are hit with probability zero. On a finite
    grid, the relevant practical question is whether the fitted dynamics ever
    move into the state.

    Edge Cases
    ----------
    Absorbing states can have zero outgoing rates but still not be polar-like if
    there are incoming default rates.

    References
    ----------
    - Fukushima, Oshima, and Takeda, "Dirichlet Forms and Symmetric Markov
      Processes", 2nd revised and extended edition, 2011.
    """

    if threshold < 0:
        raise ValueError("threshold must be non-negative.")
    if target_state not in generator.columns:
        raise KeyError(f"Unknown target_state: {target_state}")
    origin_states = [state for state in generator.index if state != target_state]
    incoming = generator.loc[origin_states, target_state].astype(float).clip(lower=0.0)
    if reference_measure is None:
        weights = pd.Series(1.0 / len(origin_states), index=origin_states, dtype=float)
    else:
        weights = pd.Series(reference_measure, dtype=float).reindex(origin_states).fillna(0.0)
        if (weights < 0).any() or float(weights.sum()) <= 0:
            raise ValueError("Reference weights must be non-negative and have positive mass.")
        weights = weights / float(weights.sum())
    weighted = float((weights * incoming).sum())
    total = float(incoming.sum())
    return {
        "target_state": target_state,
        "incoming_rate": total,
        "weighted_incoming_rate": weighted,
        "polar_like": bool(total <= threshold and weighted <= threshold),
    }


def cheeger_energy_credit_proxy(
    grid_points: Sequence[float],
    values: Sequence[float],
) -> float:
    """Compute a one-dimensional Cheeger-energy proxy for a score surface.

    Summary
    -------
    Measure squared slope of a score or risk surface over an ordered latent
    credit-quality coordinate.

    Method
    ------
    The proxy computes finite-difference slopes between adjacent grid points and
    returns `0.5 * mean(slope^2)`. This mirrors the role of Cheeger energy as a
    relaxed squared-gradient energy, but is intentionally finite-dimensional.

    Parameters
    ----------
    grid_points:
        Ordered latent-state coordinates.
    values:
        Score or risk values on the same grid.

    Returns
    -------
    float
        Non-negative smoothness energy.

    Raises
    ------
    ValueError
        Raised when lengths mismatch, there are fewer than two points, or grid
        spacing is non-positive.

    Notes
    -----
    This is useful when comparing whether a PD surface, stage scale, or
    management overlay is unnecessarily jagged over a continuous latent state.

    Edge Cases
    ----------
    Constant values return zero.

    References
    ----------
    - Ambrosio, Gigli, and Savare, "Gradient Flows in Metric Spaces and in the
      Space of Probability Measures", 2008.
    """

    x = np.asarray(grid_points, dtype=float)
    y = np.asarray(values, dtype=float)
    if x.shape != y.shape:
        raise ValueError("grid_points and values must have the same length.")
    if len(x) < 2:
        raise ValueError("At least two points are required.")
    dx = np.diff(x)
    if (dx <= 0).any():
        raise ValueError("grid_points must be strictly increasing.")
    slopes = np.diff(y) / dx
    return 0.5 * float(np.mean(slopes * slopes))


def regularity_diagnostics_for_grid(grid: ContinuousCreditStateGrid) -> dict[str, object]:
    """Summarise which continuous-state regularity issues matter on the grid.

    Summary
    -------
    Translate regularity and quasi-regularity language into a finite-grid
    implementation checklist.

    Method
    ------
    The diagnostic reports whether the active grid is finite, ordered, locally
    compact in the practical numerical sense, has an explicit cemetery state,
    and has a represented default boundary.

    Parameters
    ----------
    grid:
        Continuous credit-quality grid.

    Returns
    -------
    dict[str, object]
        Practical regularity checklist for the finite-grid approximation.

    Raises
    ------
    None
        This diagnostic does not raise custom exceptions for a valid dataclass.

    Notes
    -----
    The full regular/quasi-regular distinction matters when constructing a
    process from an analytic form on a delicate topological space. On this
    finite grid, the gain is governance clarity: state space, boundary, and
    cemetery behaviour are explicit.

    Edge Cases
    ----------
    A grid with no active points cannot be created by
    `build_continuous_credit_state_grid`, so this function assumes basic grid
    validity.

    References
    ----------
    - Ma, Z.-M., and Roeckner, M. (1992), "Introduction to the Theory of
      (Non-Symmetric) Dirichlet Forms."
    """

    active = np.asarray(grid.active_grid, dtype=float)
    return {
        "finite_grid": bool(len(active) > 0),
        "ordered_active_state": bool(np.all(np.diff(active) > 0)),
        "explicit_default_boundary": grid.default_boundary,
        "explicit_cemetery_state": grid.default_state,
        "regularity_issue_on_finite_grid": "none_material",
        "quasi_regular_extension_needed": False,
        "practical_gain": "state space, default boundary, and cemetery behaviour are explicit before modelling.",
    }
