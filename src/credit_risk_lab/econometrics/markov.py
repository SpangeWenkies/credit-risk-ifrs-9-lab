"""Finite-state Markov migration tools for the credit-risk lab.

This module implements the practical part of the Markov-chain extension
suggested by the Dirichlet-form course notes: a credit migration process can be
viewed through equivalent finite-dimensional objects:

- a transition kernel, represented by a row-stochastic transition matrix,
- a discrete-time semigroup, represented by powers of that matrix,
- a generator, represented by a quarterly jump-rate approximation,
- an energy form, represented by transition-weighted squared differences of
  functions on credit states.

Assumptions
-----------
- The state space is finite and consists of observable delinquency or terminal
  states.
- The fitted matrix is a discrete-time quarterly transition model estimated from
  the synthetic performance panel.
- Default and prepayment/maturity are treated as absorbing cemetery-style states
  so the model is conservative after terminal outcomes are added.
- Credit migration is generally directional and non-reversible, so the
  Dirichlet-energy diagnostic uses a symmetrised jump form by default rather
  than claiming the fitted migration chain is reversible.

Primary references
------------------
- Fukushima, Oshima, and Takeda, "Dirichlet Forms and Symmetric Markov
  Processes", 2nd revised and extended edition, 2011.
- Ma and Roeckner, "Introduction to the Theory of (Non-Symmetric) Dirichlet
  Forms", 1992.
- Jarrow, Lando, and Turnbull (1997), "A Markov Model for the Term Structure of
  Credit Risk Spreads."
- Lando and Skodeberg (2002), "Analyzing Rating Transitions and Rating Drift
  with Continuous Observations."

Simplifications for this portfolio project
------------------------------------------
- The model estimates unconditional transition probabilities before adding
  covariates. A later extension can replace this with multinomial logit or a
  continuous-time intensity model.
- The generator uses `(P - I) / dt` as a transparent first-order approximation.
  It is not a matrix-logarithm embedding test.
- The energy diagnostic is used for interpretation and smoothness checks of
  scores or state labels. It is not a full production implementation of
  non-symmetric Dirichlet-form theory.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

CREDIT_STATES: tuple[str, ...] = (
    "current",
    "dpd_1_29",
    "dpd_30_89",
    "default",
    "prepay_mature",
)
ABSORBING_STATES: tuple[str, ...] = ("default", "prepay_mature")


@dataclass(slots=True)
class MarkovTransitionModel:
    """Fitted finite-state credit migration model."""

    states: tuple[str, ...]
    absorbing_states: tuple[str, ...]
    transition_counts: pd.DataFrame
    transition_matrix: pd.DataFrame
    transition_panel: pd.DataFrame
    smoothing: float


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    """Validate that a frame contains required columns.

    Summary
    -------
    Check input schema before transition-panel construction.

    Method
    ------
    The helper compares requested column names with `frame.columns` and raises a
    single `KeyError` listing all missing fields.

    Parameters
    ----------
    frame:
        Input data frame.
    columns:
        Required column names.

    Returns
    -------
    None
        The function returns nothing when validation passes.

    Raises
    ------
    KeyError
        Raised when one or more required columns are absent.

    Notes
    -----
    Centralising this check keeps public functions consistent and makes schema
    failures easier to diagnose in tests.

    Edge Cases
    ----------
    An empty frame passes if it has the required columns.

    References
    ----------
    - EBA, "Guidelines on loan origination and monitoring", for the general
      governance principle that data inputs should be controlled before model
      use.
    """

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")


def _as_square_transition_matrix(transition_matrix: pd.DataFrame) -> pd.DataFrame:
    """Validate and cast a transition matrix.

    Summary
    -------
    Ensure that a candidate transition matrix can be treated as a finite-state
    Markov kernel.

    Method
    ------
    The helper checks that row and column labels match, the matrix is non-empty,
    probabilities are non-negative, and each row sums to one within numerical
    tolerance. It then casts entries to float.

    Parameters
    ----------
    transition_matrix:
        Candidate row-stochastic matrix.

    Returns
    -------
    pandas.DataFrame
        Float-valued transition matrix with original labels.

    Raises
    ------
    ValueError
        Raised when labels do not match, the matrix is empty, entries are
        negative, or rows do not sum to one.

    Notes
    -----
    In the finite-state setting, this matrix is the transition kernel. Matrix
    validity is therefore the minimum condition for using semigroup, generator,
    and absorption calculations.

    Edge Cases
    ----------
    Small negative values below numerical tolerance are tolerated only through
    the explicit threshold check.

    References
    ----------
    - Fukushima, Oshima, and Takeda, "Dirichlet Forms and Symmetric Markov
      Processes", 2nd revised and extended edition, 2011.
    """

    if list(transition_matrix.index) != list(transition_matrix.columns):
        raise ValueError("Transition matrix index and columns must contain the same states in the same order.")
    if transition_matrix.empty:
        raise ValueError("Transition matrix must not be empty.")
    matrix = transition_matrix.astype(float)
    if (matrix.to_numpy() < -1e-12).any():
        raise ValueError("Transition probabilities must be non-negative.")
    row_sums = matrix.sum(axis=1).to_numpy()
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        raise ValueError("Transition matrix rows must sum to one.")
    return matrix


def assign_delinquency_state(frame: pd.DataFrame, dpd_column: str = "days_past_due") -> pd.Series:
    """Map days-past-due observations to finite credit states.

    Summary
    -------
    Convert a loan performance snapshot into the state labels used by the
    Markov migration model.

    Method
    ------
    The mapping uses standard delinquency buckets: zero DPD is current, 1-29 DPD
    is early arrears, 30-89 DPD is serious arrears, and 90+ DPD is treated as a
    default-like state. Terminal default and prepayment/maturity states are
    assigned later when transitions are built.

    Parameters
    ----------
    frame:
        Loan-level performance frame.
    dpd_column:
        Name of the days-past-due column.

    Returns
    -------
    pandas.Series
        State labels aligned to `frame.index`.

    Raises
    ------
    KeyError
        Raised when `dpd_column` is missing.

    Notes
    -----
    The state mapping is intentionally simple because this module is the first
    Markov challenger. It can later be replaced by rating states, IFRS 9 stages,
    or a richer multi-state survival state space.

    Edge Cases
    ----------
    Negative DPD values are clipped to current. Missing DPD values are treated as
    current because the synthetic portfolio has no explicit unknown-delinquency
    state.

    References
    ----------
    - Jarrow, Lando, and Turnbull (1997), "A Markov Model for the Term Structure
      of Credit Risk Spreads."
    - Lando and Skodeberg (2002), "Analyzing Rating Transitions and Rating Drift
      with Continuous Observations."
    """

    _require_columns(frame, [dpd_column])
    dpd = frame[dpd_column].fillna(0).astype(float).clip(lower=0)
    states = np.select(
        [dpd.eq(0), dpd.lt(30), dpd.lt(90)],
        ["current", "dpd_1_29", "dpd_30_89"],
        default="default",
    )
    return pd.Series(states, index=frame.index, name="state")


def build_transition_panel(
    performance: pd.DataFrame,
    loan_id_column: str = "loan_id",
    date_column: str = "snapshot_date",
    dpd_column: str = "days_past_due",
    default_column: str = "default_next_period",
    prepay_column: str = "prepayment_flag",
    remaining_term_column: str = "remaining_term_quarters",
    terminal_balance_threshold: float = 250.0,
) -> pd.DataFrame:
    """Build observed one-step transitions from a loan performance panel.

    Summary
    -------
    Transform the synthetic loan-quarter panel into origin-state and next-state
    observations for a finite-state Markov migration model.

    Method
    ------
    The function sorts observations by loan and reporting date, maps each row to
    a delinquency state, and uses the next observed row for the same loan as the
    next state. If a loan exits after the current row, the terminal state is
    inferred from default, prepayment, maturity, or low balance indicators.
    Right-censored final observations with no observed terminal event are
    excluded from estimation.

    Parameters
    ----------
    performance:
        Loan-quarter performance panel from `generate_portfolio_timeseries`.
    loan_id_column:
        Loan identifier column.
    date_column:
        Snapshot date column.
    dpd_column:
        Days-past-due column.
    default_column:
        Indicator that default occurs before the next observation.
    prepay_column:
        Indicator that the loan prepays before the next observation.
    remaining_term_column:
        Remaining contractual term in quarters.
    terminal_balance_threshold:
        Balance threshold below which an exited loan is treated as matured or
        prepaid when no explicit default is observed.

    Returns
    -------
    pandas.DataFrame
        Transition panel with `loan_id`, `snapshot_date`, `state`, and
        `next_state`, plus any useful retained segmentation columns.

    Raises
    ------
    KeyError
        Raised when required input columns are missing.

    Notes
    -----
    Adding explicit absorbing states mirrors the cemetery-point construction
    from Markov process theory: the active-state chain may lose mass, but the
    augmented chain remains row-stochastic and easier to reason about.

    Edge Cases
    ----------
    Empty input returns an empty transition panel. Final observations at the end
    of the synthetic sample are omitted unless the row records a terminal event,
    because their next state is not observed.

    References
    ----------
    - Fukushima, Oshima, and Takeda, "Dirichlet Forms and Symmetric Markov
      Processes", 2nd revised and extended edition, 2011.
    - Lando and Skodeberg (2002), "Analyzing Rating Transitions and Rating Drift
      with Continuous Observations."
    """

    required = [loan_id_column, date_column, dpd_column, default_column, prepay_column, remaining_term_column]
    _require_columns(performance, required)
    if performance.empty:
        return pd.DataFrame(columns=[loan_id_column, date_column, "state", "next_state"])

    ordered = performance.sort_values([loan_id_column, date_column]).reset_index(drop=True).copy()
    ordered["state"] = assign_delinquency_state(ordered, dpd_column)
    ordered["_next_loan"] = ordered[loan_id_column].shift(-1)
    ordered["_next_state"] = ordered["state"].shift(-1)
    has_next_observation = ordered[loan_id_column].eq(ordered["_next_loan"])

    default_exit = ordered[default_column].astype(int).eq(1) | ordered["state"].eq("default")
    prepay_or_mature_exit = (
        ordered[prepay_column].astype(int).eq(1)
        | ordered[remaining_term_column].astype(float).le(1)
        | ordered.get("balance", pd.Series(np.inf, index=ordered.index)).astype(float).le(terminal_balance_threshold)
    )

    ordered["next_state"] = np.where(has_next_observation, ordered["_next_state"], pd.NA)
    ordered.loc[~has_next_observation & default_exit, "next_state"] = "default"
    ordered.loc[~has_next_observation & ~default_exit & prepay_or_mature_exit, "next_state"] = "prepay_mature"

    retained = [loan_id_column, date_column, "state", "next_state"]
    for optional in ("segment", "region", "rating_rank", "forborne", "prepayment_flag"):
        if optional in ordered.columns:
            retained.append(optional)

    transitions = ordered.loc[ordered["next_state"].notna(), retained].copy()
    transitions = transitions.rename(columns={loan_id_column: "loan_id", date_column: "snapshot_date"})
    return transitions.reset_index(drop=True)


def fit_markov_transition_model(
    performance: pd.DataFrame,
    states: Sequence[str] = CREDIT_STATES,
    absorbing_states: Sequence[str] = ABSORBING_STATES,
    smoothing: float = 0.5,
) -> MarkovTransitionModel:
    """Estimate a finite-state credit transition matrix.

    Summary
    -------
    Fit a simple Markov challenger that estimates
    `P(next_state | current_state)` from the synthetic performance panel.

    Method
    ------
    The function builds the observed transition panel, counts transitions by
    origin and destination state, adds optional additive smoothing for observed
    non-absorbing origin states, and normalises rows to obtain a row-stochastic
    transition matrix. Absorbing states are forced to self-transition with
    probability one.

    Parameters
    ----------
    performance:
        Loan-quarter performance panel.
    states:
        Ordered state labels to use as matrix rows and columns.
    absorbing_states:
        States that should be forced to absorb.
    smoothing:
        Non-negative additive smoothing applied to rows with observed
        transitions. Use zero for raw empirical frequencies.

    Returns
    -------
    MarkovTransitionModel
        Fitted transition counts, transition matrix, transition panel, state
        metadata, and smoothing value.

    Raises
    ------
    ValueError
        Raised when `smoothing` is negative or an absorbing state is not present
        in `states`.
    KeyError
        Raised when the performance panel is missing required columns.

    Notes
    -----
    This is the natural Markov challenger to the survival-logit PD model: the
    survival model asks for the next-period default hazard, while this model
    asks how loans move through the whole delinquency state graph.

    Edge Cases
    ----------
    If a non-absorbing state has no observed origin rows, it receives a
    self-transition probability of one rather than a smoothed artificial row.

    References
    ----------
    - Jarrow, Lando, and Turnbull (1997), "A Markov Model for the Term Structure
      of Credit Risk Spreads."
    - Lando and Skodeberg (2002), "Analyzing Rating Transitions and Rating Drift
      with Continuous Observations."
    """

    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")
    state_tuple = tuple(states)
    absorbing_tuple = tuple(absorbing_states)
    missing_absorbing = [state for state in absorbing_tuple if state not in state_tuple]
    if missing_absorbing:
        raise ValueError(f"Absorbing states must appear in states: {missing_absorbing}")

    transition_panel = build_transition_panel(performance)
    counts = pd.crosstab(transition_panel["state"], transition_panel["next_state"])
    counts = counts.reindex(index=state_tuple, columns=state_tuple, fill_value=0).astype(float)

    probabilities = pd.DataFrame(0.0, index=state_tuple, columns=state_tuple)
    for state in state_tuple:
        if state in absorbing_tuple:
            probabilities.loc[state, state] = 1.0
            continue

        row = counts.loc[state].copy()
        observed = float(row.sum())
        if observed <= 0:
            probabilities.loc[state, state] = 1.0
            continue

        if smoothing > 0:
            row = row + smoothing
        probabilities.loc[state] = row / float(row.sum())

    probabilities = _as_square_transition_matrix(probabilities)
    return MarkovTransitionModel(
        states=state_tuple,
        absorbing_states=absorbing_tuple,
        transition_counts=counts,
        transition_matrix=probabilities,
        transition_panel=transition_panel,
        smoothing=float(smoothing),
    )


def transition_matrices_by_group(
    transition_panel: pd.DataFrame,
    group_column: str,
    states: Sequence[str] = CREDIT_STATES,
    absorbing_states: Sequence[str] = ABSORBING_STATES,
    smoothing: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Estimate conditional transition matrices by a grouping variable.

    Summary
    -------
    Build a simple covariate-conditioned Markov extension by estimating one
    transition matrix per segment, rating group, macro regime, or other
    discrete grouping variable.

    Method
    ------
    The function filters the transition panel by group, counts
    `state -> next_state` transitions, applies additive smoothing to observed
    non-absorbing rows, and forces absorbing states to self-transition.

    Parameters
    ----------
    transition_panel:
        Output of `build_transition_panel`.
    group_column:
        Column used to condition the transition matrix.
    states:
        Ordered state labels.
    absorbing_states:
        Absorbing state labels.
    smoothing:
        Non-negative additive smoothing for observed non-absorbing rows.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Mapping from group label to transition matrix.

    Raises
    ------
    KeyError
        Raised when required transition-panel columns are missing.
    ValueError
        Raised when `smoothing` is negative.

    Notes
    -----
    This is the first practical step toward
    `P(next_state | current_state, borrower features, macro variables)`. It is
    deliberately discrete and explainable before moving to multinomial logits.

    Edge Cases
    ----------
    Groups with no transitions are omitted. Non-absorbing rows with no observed
    origins receive self-transition probability one.

    References
    ----------
    - Jarrow, Lando, and Turnbull (1997), "A Markov Model for the Term Structure
      of Credit Risk Spreads."
    - Lando and Skodeberg (2002), "Analyzing Rating Transitions and Rating Drift
      with Continuous Observations."
    """

    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")
    required = ["state", "next_state", group_column]
    missing = [column for column in required if column not in transition_panel.columns]
    if missing:
        raise KeyError(f"Missing grouped transition columns: {missing}")

    matrices: dict[str, pd.DataFrame] = {}
    state_tuple = tuple(states)
    absorbing_tuple = tuple(absorbing_states)
    for group, data in transition_panel.groupby(group_column):
        counts = pd.crosstab(data["state"], data["next_state"]).reindex(index=state_tuple, columns=state_tuple, fill_value=0).astype(float)
        probabilities = pd.DataFrame(0.0, index=state_tuple, columns=state_tuple)
        for state in state_tuple:
            if state in absorbing_tuple:
                probabilities.loc[state, state] = 1.0
                continue
            row = counts.loc[state].copy()
            if float(row.sum()) <= 0:
                probabilities.loc[state, state] = 1.0
            else:
                if smoothing > 0:
                    row = row + smoothing
                probabilities.loc[state] = row / float(row.sum())
        matrices[str(group)] = _as_square_transition_matrix(probabilities)
    return matrices


def markov_implied_default_pd(
    transition_matrix: pd.DataFrame,
    initial_state: str,
    horizon_steps: int,
    default_state: str = "default",
) -> float:
    """Compute Markov-implied default probability for one starting state.

    Summary
    -------
    Read default probability from a multi-period transition matrix.

    Method
    ------
    The transition matrix is raised to `horizon_steps`. The entry in the
    starting-state row and default-state column is the probability of being in
    default at that horizon when default is absorbing.

    Parameters
    ----------
    transition_matrix:
        Row-stochastic transition matrix.
    initial_state:
        Starting credit state.
    horizon_steps:
        Number of periods to project.
    default_state:
        Absorbing default state.

    Returns
    -------
    float
        Markov-implied default probability.

    Raises
    ------
    KeyError
        Raised when `initial_state` or `default_state` is absent.
    ValueError
        Raised when `horizon_steps` is negative or the matrix is invalid.

    Notes
    -----
    This is the clean comparison point against survival-logit PDs.

    Edge Cases
    ----------
    `horizon_steps=0` returns one if the loan starts in default and zero
    otherwise.

    References
    ----------
    - Jarrow, Lando, and Turnbull (1997), "A Markov Model for the Term Structure
      of Credit Risk Spreads."
    """

    projected = n_step_transition_matrix(transition_matrix, horizon_steps)
    if initial_state not in projected.index:
        raise KeyError(f"Unknown initial_state: {initial_state}")
    if default_state not in projected.columns:
        raise KeyError(f"Unknown default_state: {default_state}")
    return float(projected.loc[initial_state, default_state])


def n_step_transition_matrix(transition_matrix: pd.DataFrame, n_steps: int) -> pd.DataFrame:
    """Raise a transition matrix to a multi-period horizon.

    Summary
    -------
    Compute the Markov semigroup value `P^n` for an integer number of reporting
    periods.

    Method
    ------
    The function validates that the input is a square row-stochastic transition
    matrix and applies ordinary matrix powers. In finite state space, powers of
    the transition matrix are the discrete-time semigroup.

    Parameters
    ----------
    transition_matrix:
        Row-stochastic transition matrix with matching index and columns.
    n_steps:
        Number of periods. Zero returns the identity matrix.

    Returns
    -------
    pandas.DataFrame
        Multi-period transition matrix with the same labels as the input.

    Raises
    ------
    ValueError
        Raised when `n_steps` is negative or the matrix is not valid.

    Notes
    -----
    This directly mirrors the semigroup property from the notes:
    `P^(s+t) = P^s P^t` in discrete time.

    Edge Cases
    ----------
    `n_steps=0` returns the identity transition matrix, representing no elapsed
    reporting periods.

    References
    ----------
    - Fukushima, Oshima, and Takeda, "Dirichlet Forms and Symmetric Markov
      Processes", 2nd revised and extended edition, 2011.
    """

    if n_steps < 0:
        raise ValueError("n_steps must be non-negative.")
    matrix = _as_square_transition_matrix(transition_matrix)
    powered = np.linalg.matrix_power(matrix.to_numpy(dtype=float), n_steps)
    return pd.DataFrame(powered, index=matrix.index, columns=matrix.columns)


def transition_generator(transition_matrix: pd.DataFrame, step_length: float = 1.0) -> pd.DataFrame:
    """Approximate a continuous-time generator from a quarterly matrix.

    Summary
    -------
    Convert a one-period migration matrix into a transparent jump-rate
    generator approximation.

    Method
    ------
    The function uses the first-order relationship `P(dt) ≈ I + dt Q`, giving
    `Q = (P - I) / dt`. Off-diagonal entries are transition intensities per
    period and each row sums to zero.

    Parameters
    ----------
    transition_matrix:
        Row-stochastic transition matrix.
    step_length:
        Length of one observation interval. Use `1.0` for quarters or `0.25` for
        years when the matrix represents one quarter.

    Returns
    -------
    pandas.DataFrame
        Generator-like matrix with non-negative off-diagonal entries and
        approximately zero row sums.

    Raises
    ------
    ValueError
        Raised when the matrix is invalid or `step_length <= 0`.

    Notes
    -----
    This is a bridge toward continuous-time default modelling. It is deliberately
    not a matrix-logarithm estimator, because not every empirical transition
    matrix is embeddable in a valid continuous-time chain.

    Edge Cases
    ----------
    Absorbing rows become zero generator rows. That is the generator analogue of
    a cemetery state.

    References
    ----------
    - Ma and Roeckner, "Introduction to the Theory of (Non-Symmetric) Dirichlet
      Forms", 1992.
    - Lando and Skodeberg (2002), "Analyzing Rating Transitions and Rating Drift
      with Continuous Observations."
    """

    if step_length <= 0:
        raise ValueError("step_length must be positive.")
    matrix = _as_square_transition_matrix(transition_matrix)
    identity = np.eye(len(matrix), dtype=float)
    generator = (matrix.to_numpy(dtype=float) - identity) / float(step_length)
    return pd.DataFrame(generator, index=matrix.index, columns=matrix.columns)


def absorption_summary(
    transition_matrix: pd.DataFrame,
    absorbing_states: Sequence[str] = ABSORBING_STATES,
) -> pd.DataFrame:
    """Summarise default and terminal absorption from a transition matrix.

    Summary
    -------
    Compute eventual absorption probabilities and expected time to absorption
    for each non-absorbing credit state.

    Method
    ------
    The matrix is partitioned into transient states `T` and absorbing states
    `A`. The fundamental matrix `N = (I - P_TT)^-1` is the discrete Green
    operator for expected visits before absorption. Multiplying `N` by `P_TA`
    gives eventual absorption probabilities.

    Parameters
    ----------
    transition_matrix:
        Row-stochastic transition matrix.
    absorbing_states:
        Absorbing state labels, typically default and prepayment/maturity.

    Returns
    -------
    pandas.DataFrame
        One row per transient state with expected periods to absorption and
        eventual absorption probabilities.

    Raises
    ------
    ValueError
        Raised when an absorbing state is absent or the transient block is not
        invertible.

    Notes
    -----
    This is the most direct practical use of the Green/resolvent intuition from
    the course notes in a credit portfolio: it turns migration dynamics into
    expected time spent before terminal credit outcomes.

    Edge Cases
    ----------
    If every state is absorbing, the returned frame is empty because there are
    no transient origin states to summarise.

    References
    ----------
    - Fukushima, Oshima, and Takeda, "Dirichlet Forms and Symmetric Markov
      Processes", 2nd revised and extended edition, 2011.
    - Kemeny and Snell, "Finite Markov Chains", 1976.
    """

    matrix = _as_square_transition_matrix(transition_matrix)
    absorbing = list(absorbing_states)
    missing = [state for state in absorbing if state not in matrix.index]
    if missing:
        raise ValueError(f"Absorbing states missing from transition matrix: {missing}")

    transient = [state for state in matrix.index if state not in absorbing]
    if not transient:
        return pd.DataFrame(columns=["state", "expected_steps_to_absorption"])

    q_block = matrix.loc[transient, transient].to_numpy(dtype=float)
    r_block = matrix.loc[transient, absorbing].to_numpy(dtype=float)
    identity = np.eye(len(transient), dtype=float)
    try:
        fundamental = np.linalg.inv(identity - q_block)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Transient block is singular; absorption is not guaranteed from all transient states.") from exc

    absorption_probs = fundamental @ r_block
    expected_steps = fundamental.sum(axis=1)
    result = pd.DataFrame({"state": transient, "expected_steps_to_absorption": expected_steps})
    for idx, state in enumerate(absorbing):
        result[f"prob_absorb_{state}"] = absorption_probs[:, idx]
    return result


def dirichlet_transition_energy(
    transition_matrix: pd.DataFrame,
    state_values: Mapping[str, float] | pd.Series,
    weights: Mapping[str, float] | pd.Series | None = None,
    symmetrize: bool = True,
) -> float:
    """Measure roughness of a state function over the migration graph.

    Summary
    -------
    Compute a finite-state Dirichlet-style energy for a scalar function defined
    on credit states.

    Method
    ------
    For state values `f_i`, transition probabilities `P_ij`, and reference
    weights `m_i`, the unsymmetrised diagnostic is
    `0.5 * sum_i m_i sum_j P_ij (f_i - f_j)^2`. With `symmetrize=True`, the
    conductance is replaced by `0.5 * (m_i P_ij + m_j P_ji)` before applying the
    same squared-difference penalty.

    Parameters
    ----------
    transition_matrix:
        Row-stochastic transition matrix.
    state_values:
        Mapping from state label to numeric score, grade, stage, or loss value.
    weights:
        Optional reference measure over states. When omitted, uniform weights
        are used.
    symmetrize:
        Whether to use a symmetric jump conductance. This is the recommended
        setting for this portfolio project because credit migration is usually
        non-reversible.

    Returns
    -------
    float
        Non-negative transition energy.

    Raises
    ------
    ValueError
        Raised when matrix labels, values, or weights are invalid.

    Notes
    -----
    This diagnostic answers a credit-risk question in functional-analytic
    language: does a score, stage label, or provision proxy vary smoothly over
    the migration graph, or does it jump sharply across commonly observed
    transitions?

    Edge Cases
    ----------
    Constant `state_values` return zero energy. Absorbing states contribute only
    through transitions into or out of them; self-loops have zero contribution.

    References
    ----------
    - Fukushima, Oshima, and Takeda, "Dirichlet Forms and Symmetric Markov
      Processes", 2nd revised and extended edition, 2011.
    - Beurling and Deny, "Dirichlet Spaces", 1958-1959.
    """

    matrix = _as_square_transition_matrix(transition_matrix)
    states = list(matrix.index)
    values = pd.Series(state_values, dtype=float).reindex(states)
    if values.isna().any():
        missing = values.index[values.isna()].tolist()
        raise ValueError(f"Missing state values for: {missing}")

    if weights is None:
        reference = pd.Series(1.0 / len(states), index=states, dtype=float)
    else:
        reference = pd.Series(weights, dtype=float).reindex(states)
        if reference.isna().any():
            missing = reference.index[reference.isna()].tolist()
            raise ValueError(f"Missing state weights for: {missing}")
        if (reference < 0).any() or float(reference.sum()) <= 0:
            raise ValueError("State weights must be non-negative and have positive total mass.")
        reference = reference / float(reference.sum())

    p = matrix.to_numpy(dtype=float)
    m = reference.to_numpy(dtype=float)
    f = values.to_numpy(dtype=float)
    diffs = f[:, None] - f[None, :]

    if symmetrize:
        conductance = 0.5 * (m[:, None] * p + m[None, :] * p.T)
    else:
        conductance = m[:, None] * p
    energy = 0.5 * float(np.sum(conductance * diffs * diffs))
    return energy
