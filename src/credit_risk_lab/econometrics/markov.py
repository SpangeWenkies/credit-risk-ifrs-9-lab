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

Simplifications for this lab
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
from scipy.linalg import expm, logm
import statsmodels.api as sm

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


@dataclass(slots=True)
class GeneratorEmbeddingDiagnostics:
    """Matrix-logarithm generator candidate and embeddability diagnostics."""

    generator: pd.DataFrame
    is_valid_generator: bool
    max_imaginary_part: float
    min_off_diagonal: float
    max_row_sum_abs: float
    reconstruction_error: float
    message: str


@dataclass(slots=True)
class CovariateTransitionModel:
    """One-vs-rest covariate-dependent Markov transition model."""

    states: tuple[str, ...]
    absorbing_states: tuple[str, ...]
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    design_columns: tuple[str, ...]
    origin_destination_models: dict[str, dict[str, object]]
    origin_base_probabilities: dict[str, pd.Series]
    min_rows: int


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


def _prepare_covariate_design(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    design_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build a stable design matrix for covariate transition logits.

    Summary
    -------
    Convert borrower, loan, and macro covariates into the deterministic design
    matrix used by the covariate-dependent Markov transition model.

    Method
    ------
    Numeric covariates are cast to float and missing values are imputed with
    zero. Categorical covariates are one-hot encoded with a dropped reference
    level. A constant is added and columns are sorted. During scoring, the
    matrix is reindexed to the fitted `design_columns` so missing dummy levels
    become zeros and unexpected levels are ignored.

    Parameters
    ----------
    frame:
        Transition-panel or scoring frame containing covariates.
    feature_columns:
        Numeric covariates to include.
    categorical_columns:
        Categorical covariates to dummy encode.
    design_columns:
        Optional fitted design-column order.

    Returns
    -------
    pandas.DataFrame
        Numeric design matrix with a constant term.

    Raises
    ------
    KeyError
        Raised when a requested feature column is missing.

    Notes
    -----
    This helper is deliberately local to the Markov module because transition
    logits have a different target structure from the primary survival-logit PD
    model.

    Edge Cases
    ----------
    Empty inputs return an empty matrix with the fitted columns when
    `design_columns` are supplied.

    References
    ----------
    - Lando, D., and Skodeberg, T. M. (2002), "Analyzing Rating Transitions and
      Rating Drift with Continuous Observations."
    """

    required = list(feature_columns) + list(categorical_columns)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing covariate transition columns: {missing}")

    numeric = frame[list(feature_columns)].astype(float).fillna(0.0) if feature_columns else pd.DataFrame(index=frame.index)
    if categorical_columns:
        categoricals = pd.get_dummies(frame[list(categorical_columns)].astype("category"), drop_first=True, dtype=float)
    else:
        categoricals = pd.DataFrame(index=frame.index)
    design = pd.concat([numeric, categoricals], axis=1)
    design = sm.add_constant(design, has_constant="add")
    design = design.reindex(sorted(design.columns), axis=1)
    if design_columns is not None:
        design = design.reindex(list(design_columns), axis=1, fill_value=0.0)
    return design.astype(float)


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
    for optional in (
        "segment",
        "region",
        "rating_rank",
        "forborne",
        "prepayment_flag",
        "quarters_on_book",
        "remaining_term_quarters",
        "ltv",
        "dti",
        "days_past_due",
        "unemployment_rate",
        "policy_rate",
        "house_price_growth",
        "gdp_growth",
    ):
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


def fit_covariate_transition_model(
    transition_panel: pd.DataFrame,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str] | None = None,
    states: Sequence[str] = CREDIT_STATES,
    absorbing_states: Sequence[str] = ABSORBING_STATES,
    min_rows: int = 30,
    smoothing: float = 0.5,
) -> CovariateTransitionModel:
    """Fit a covariate-dependent Markov transition model.

    Summary
    -------
    Estimate `P(next_state | current_state, borrower features, macro variables)`
    using separate one-vs-rest transition logits by origin state.

    Method
    ------
    For each non-absorbing origin state, the function estimates one binary GLM
    per possible destination state. The resulting one-vs-rest probabilities are
    clipped and normalised to sum to one at prediction time. When a destination
    has no event variation or a state has too few rows, empirical smoothed
    transition frequencies are retained as a robust fallback.

    Parameters
    ----------
    transition_panel:
        Observed transition panel with `state`, `next_state`, and covariates.
    feature_columns:
        Numeric borrower, loan, behavioural, or macro covariates.
    categorical_columns:
        Optional categorical covariates such as segment or region.
    states:
        Ordered Markov states.
    absorbing_states:
        States forced to self-transition.
    min_rows:
        Minimum origin-state row count required before fitting logits.
    smoothing:
        Additive smoothing for empirical fallback probabilities.

    Returns
    -------
    CovariateTransitionModel
        Fitted one-vs-rest transition model with empirical fallbacks.

    Raises
    ------
    KeyError
        Raised when required transition or covariate columns are missing.
    ValueError
        Raised when `min_rows <= 0` or `smoothing < 0`.

    Notes
    -----
    This intentionally avoids forcing every transition through a fragile full
    multinomial model. Credit migration panels are often sparse in severe
    delinquency states, so robust fallbacks are more valuable than a single
    elegant but brittle likelihood.

    Edge Cases
    ----------
    Absorbing states are stored with deterministic self-transition
    probabilities and no fitted logits.

    References
    ----------
    - Lando, D., and Skodeberg, T. M. (2002), "Analyzing Rating Transitions and
      Rating Drift with Continuous Observations."
    - Jarrow, R. A., Lando, D., and Turnbull, S. M. (1997), "A Markov Model for
      the Term Structure of Credit Risk Spreads."
    """

    if min_rows <= 0:
        raise ValueError("min_rows must be positive.")
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative.")
    categorical_columns = list(categorical_columns or [])
    state_tuple = tuple(states)
    absorbing_tuple = tuple(absorbing_states)
    required = ["state", "next_state", *feature_columns, *categorical_columns]
    missing = [column for column in required if column not in transition_panel.columns]
    if missing:
        raise KeyError(f"Missing covariate transition columns: {missing}")

    design = _prepare_covariate_design(transition_panel, feature_columns, categorical_columns)
    design_columns = tuple(design.columns)
    origin_models: dict[str, dict[str, object]] = {}
    base_probabilities: dict[str, pd.Series] = {}

    for origin_state in state_tuple:
        subset = transition_panel.loc[transition_panel["state"].eq(origin_state)].copy()
        if origin_state in absorbing_tuple:
            base = pd.Series(0.0, index=state_tuple, dtype=float)
            base.loc[origin_state] = 1.0
            base_probabilities[origin_state] = base
            origin_models[origin_state] = {}
            continue

        counts = subset["next_state"].value_counts().reindex(state_tuple, fill_value=0).astype(float)
        if float(counts.sum()) > 0:
            base = (counts + float(smoothing)) / float((counts + float(smoothing)).sum())
        else:
            base = pd.Series(0.0, index=state_tuple, dtype=float)
            base.loc[origin_state] = 1.0
        base_probabilities[origin_state] = base

        fitted: dict[str, object] = {}
        if len(subset) < min_rows:
            origin_models[origin_state] = fitted
            continue
        origin_design = design.loc[subset.index]
        for destination_state in state_tuple:
            target = subset["next_state"].eq(destination_state).astype(int)
            if target.nunique() < 2:
                continue
            try:
                fitted[destination_state] = sm.GLM(target, origin_design, family=sm.families.Binomial()).fit()
            except Exception:
                continue
        origin_models[origin_state] = fitted

    return CovariateTransitionModel(
        states=state_tuple,
        absorbing_states=absorbing_tuple,
        feature_columns=tuple(feature_columns),
        categorical_columns=tuple(categorical_columns),
        design_columns=design_columns,
        origin_destination_models=origin_models,
        origin_base_probabilities=base_probabilities,
        min_rows=int(min_rows),
    )


def predict_covariate_transition_probabilities(
    model: CovariateTransitionModel,
    frame: pd.DataFrame,
    state_column: str = "state",
) -> pd.DataFrame:
    """Predict row-level Markov transition probabilities.

    Summary
    -------
    Produce a long-form table of `P(next_state | current_state, X)` for each
    scoring row and destination state.

    Method
    ------
    The function builds the fitted design matrix, evaluates any available
    destination-specific logits for the row's origin state, fills missing logits
    with empirical fallback probabilities, clips scores to valid probability
    ranges, and normalises each row across destinations.

    Parameters
    ----------
    model:
        Fitted covariate transition model.
    frame:
        Scoring frame containing a current state and the fitted covariates.
    state_column:
        Column holding the row's origin state.

    Returns
    -------
    pandas.DataFrame
        Long-form probabilities with columns `row_id`, `state`, `next_state`,
        and `probability`.

    Raises
    ------
    KeyError
        Raised when `state_column` or model covariates are missing.

    Notes
    -----
    Normalising one-vs-rest probabilities gives a practical transition
    distribution while retaining robust binary-logit fallbacks for sparse
    migration states.

    Edge Cases
    ----------
    Unknown origin states fall back to a deterministic self-transition if the
    state is in the model state list, otherwise a `KeyError` is raised.

    References
    ----------
    - Lando, D., and Skodeberg, T. M. (2002), "Analyzing Rating Transitions and
      Rating Drift with Continuous Observations."
    """

    if state_column not in frame.columns:
        raise KeyError(f"Missing state column: {state_column}")
    design = _prepare_covariate_design(frame, model.feature_columns, model.categorical_columns, model.design_columns)
    rows: list[dict[str, object]] = []
    for row_id, row in frame.iterrows():
        origin_state = str(row[state_column])
        if origin_state not in model.states:
            raise KeyError(f"Unknown origin state: {origin_state}")
        if origin_state in model.absorbing_states:
            probabilities = pd.Series(0.0, index=model.states, dtype=float)
            probabilities.loc[origin_state] = 1.0
        else:
            probabilities = model.origin_base_probabilities[origin_state].copy()
            fitted = model.origin_destination_models.get(origin_state, {})
            row_design = design.loc[[row_id]]
            for destination_state, result in fitted.items():
                probabilities.loc[destination_state] = float(result.predict(row_design).iloc[0])
            probabilities = probabilities.clip(lower=1e-9, upper=1.0)
            probabilities = probabilities / float(probabilities.sum())
        for destination_state, probability in probabilities.items():
            rows.append(
                {
                    "row_id": row_id,
                    "state": origin_state,
                    "next_state": destination_state,
                    "probability": float(probability),
                }
            )
    return pd.DataFrame(rows)


def covariate_markov_default_pd(
    model: CovariateTransitionModel,
    frame: pd.DataFrame,
    horizon_steps: int,
    state_column: str = "state",
    default_state: str = "default",
) -> pd.Series:
    """Compute row-level default PDs from covariate transition probabilities.

    Summary
    -------
    Convert fitted covariate-dependent one-step transitions into horizon
    default probabilities for comparison with survival-logit PDs.

    Method
    ------
    For each row, the predicted one-step transition distribution is used as the
    row corresponding to the current state, while other rows fall back to their
    empirical transition probabilities. The resulting matrix is raised to the
    requested horizon and the default-state entry is read from the origin row.

    Parameters
    ----------
    model:
        Fitted covariate transition model.
    frame:
        Scoring frame with current states and covariates.
    horizon_steps:
        Number of periods to project.
    state_column:
        Origin-state column.
    default_state:
        Absorbing default state.

    Returns
    -------
    pandas.Series
        Markov-implied default PDs aligned to `frame.index`.

    Raises
    ------
    ValueError
        Raised when `horizon_steps < 0`.
    KeyError
        Raised when the default state is absent.

    Notes
    -----
    This is a transparent challenger comparison rather than a full dynamic
    covariate path model. Future covariates are held fixed for the projection.

    Edge Cases
    ----------
    `horizon_steps=0` returns one for rows already in default and zero
    otherwise.

    References
    ----------
    - Jarrow, R. A., Lando, D., and Turnbull, S. M. (1997), "A Markov Model for
      the Term Structure of Credit Risk Spreads."
    """

    if horizon_steps < 0:
        raise ValueError("horizon_steps must be non-negative.")
    if default_state not in model.states:
        raise KeyError(f"Unknown default_state: {default_state}")

    predicted = predict_covariate_transition_probabilities(model, frame, state_column=state_column)
    result = pd.Series(index=frame.index, dtype=float)
    for row_id, row_probs in predicted.groupby("row_id"):
        origin_state = str(frame.loc[row_id, state_column])
        matrix = pd.DataFrame(
            [model.origin_base_probabilities[state].reindex(model.states).to_numpy(dtype=float) for state in model.states],
            index=model.states,
            columns=model.states,
        )
        row_vector = row_probs.set_index("next_state")["probability"].reindex(model.states).to_numpy(dtype=float)
        matrix.loc[origin_state] = row_vector
        result.loc[row_id] = markov_implied_default_pd(matrix, origin_state, horizon_steps, default_state)
    return result


def compare_markov_to_survival_pd(
    frame: pd.DataFrame,
    survival_pd_column: str,
    horizon_steps: int,
    transition_matrix: pd.DataFrame | None = None,
    covariate_model: CovariateTransitionModel | None = None,
    state_column: str = "state",
    default_state: str = "default",
) -> pd.DataFrame:
    """Compare Markov-implied default risk with survival-model PDs.

    Summary
    -------
    Create a challenger comparison table between a Markov migration model and
    the primary pooled-logit survival PD model.

    Method
    ------
    If `covariate_model` is supplied, row-level Markov PDs are computed from
    covariate transition probabilities. Otherwise, the unconditional transition
    matrix is used by starting state. The result is aligned with the supplied
    survival PD column and includes absolute differences.

    Parameters
    ----------
    frame:
        Scored frame containing survival PDs and current states.
    survival_pd_column:
        Column containing the primary survival-model PD.
    horizon_steps:
        Number of Markov periods to project.
    transition_matrix:
        Unconditional transition matrix used when no covariate model is given.
    covariate_model:
        Optional fitted covariate transition model.
    state_column:
        Column containing current Markov states.
    default_state:
        Absorbing default state.

    Returns
    -------
    pandas.DataFrame
        Comparison table with survival PD, Markov PD, and differences.

    Raises
    ------
    KeyError
        Raised when required columns are missing.
    ValueError
        Raised when neither `transition_matrix` nor `covariate_model` is given.

    Notes
    -----
    The comparison supports the lab design where survival logit remains the
    primary PD model and Markov migration is a challenger or explanatory model.

    Edge Cases
    ----------
    Rows with missing survival PD are retained with missing differences.

    References
    ----------
    - Brier, G. W. (1950), "Verification of Forecasts Expressed in Terms of
      Probability."
    - Jarrow, R. A., Lando, D., and Turnbull, S. M. (1997), "A Markov Model for
      the Term Structure of Credit Risk Spreads."
    """

    missing = [column for column in (survival_pd_column, state_column) if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing comparison columns: {missing}")
    if transition_matrix is None and covariate_model is None:
        raise ValueError("Either transition_matrix or covariate_model must be supplied.")

    if covariate_model is not None:
        markov_pd = covariate_markov_default_pd(covariate_model, frame, horizon_steps, state_column, default_state)
    else:
        matrix = _as_square_transition_matrix(transition_matrix)  # type: ignore[arg-type]
        markov_pd = frame[state_column].map(lambda state: markov_implied_default_pd(matrix, str(state), horizon_steps, default_state))
    output = frame[[state_column, survival_pd_column]].copy()
    output["markov_pd"] = markov_pd.astype(float)
    output["pd_difference"] = output["markov_pd"] - output[survival_pd_column].astype(float)
    output["absolute_pd_difference"] = output["pd_difference"].abs()
    return output


def assign_macro_regime(
    frame: pd.DataFrame,
    macro_column: str = "unemployment_rate",
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
    output_column: str = "macro_regime",
) -> pd.DataFrame:
    """Assign benign, baseline, and stress regimes from a macro variable.

    Summary
    -------
    Create a transparent macro-regime label that can condition Markov migration
    matrices without hard-coding a full macro satellite model.

    Method
    ------
    The selected macro variable is split at two empirical quantiles. Values at
    or below the low threshold are labelled `benign`, values at or above the
    high threshold are labelled `stress`, and middle observations are labelled
    `baseline`.

    Parameters
    ----------
    frame:
        Data frame containing the macro variable.
    macro_column:
        Macro variable used for regime assignment.
    low_quantile:
        Lower quantile cut.
    high_quantile:
        Upper quantile cut.
    output_column:
        Name of the regime label column to create.

    Returns
    -------
    pandas.DataFrame
        Copy of `frame` with the added regime column.

    Raises
    ------
    KeyError
        Raised when `macro_column` is missing.
    ValueError
        Raised when quantile settings are invalid.

    Notes
    -----
    The regime label is intentionally coarse. It lets the lab compare migration
    behaviour across macro environments while avoiding double-counting macro
    effects in both PD and ECL layers.

    Edge Cases
    ----------
    If all macro values are equal, every row is assigned `baseline`.

    References
    ----------
    - Lando, D., and Skodeberg, T. M. (2002), "Analyzing Rating Transitions and
      Rating Drift with Continuous Observations."
    """

    if macro_column not in frame.columns:
        raise KeyError(f"Missing macro column: {macro_column}")
    if not (0.0 < low_quantile < high_quantile < 1.0):
        raise ValueError("Require 0 < low_quantile < high_quantile < 1.")
    data = frame.copy()
    values = data[macro_column].astype(float)
    low = float(values.quantile(low_quantile))
    high = float(values.quantile(high_quantile))
    if np.isclose(low, high):
        data[output_column] = "baseline"
        return data
    data[output_column] = np.select([values <= low, values >= high], ["benign", "stress"], default="baseline")
    return data


def fit_macro_regime_transition_matrices(
    transition_panel: pd.DataFrame,
    macro_column: str = "unemployment_rate",
    regime_column: str = "macro_regime",
    states: Sequence[str] = CREDIT_STATES,
    absorbing_states: Sequence[str] = ABSORBING_STATES,
    smoothing: float = 0.5,
) -> dict[str, pd.DataFrame]:
    """Estimate Markov transition matrices by macro regime.

    Summary
    -------
    Fit baseline, benign, and stress migration matrices for scenario analysis.

    Method
    ------
    The transition panel is labelled with macro regimes when the regime column
    is absent. The function then reuses grouped transition estimation to produce
    one row-stochastic matrix per regime, with absorbing terminal states forced
    to self-transition.

    Parameters
    ----------
    transition_panel:
        Transition panel with `state`, `next_state`, and a macro variable.
    macro_column:
        Macro variable used when regime labels need to be created.
    regime_column:
        Regime column to group on.
    states:
        Ordered Markov states.
    absorbing_states:
        Absorbing terminal states.
    smoothing:
        Additive smoothing for transition rows.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Regime-to-transition-matrix mapping.

    Raises
    ------
    KeyError
        Raised when required transition or macro columns are missing.

    Notes
    -----
    This is the V3 macro-sensitive Markov layer. It should be used as a primary
    migration channel or as a challenger, not stacked blindly on top of a
    macro-sensitive PD model.

    Edge Cases
    ----------
    Regimes with no observations are absent from the returned mapping.

    References
    ----------
    - Jarrow, R. A., Lando, D., and Turnbull, S. M. (1997), "A Markov Model for
      the Term Structure of Credit Risk Spreads."
    """

    panel = transition_panel.copy()
    if regime_column not in panel.columns:
        panel = assign_macro_regime(panel, macro_column=macro_column, output_column=regime_column)
    return transition_matrices_by_group(
        panel,
        group_column=regime_column,
        states=states,
        absorbing_states=absorbing_states,
        smoothing=smoothing,
    )


def build_markov_scenario_matrices(
    regime_matrices: Mapping[str, pd.DataFrame],
    fallback_matrix: pd.DataFrame,
    scenario_regime_map: Mapping[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Map macro regimes to baseline, downside, and upside scenario matrices.

    Summary
    -------
    Convert regime-conditioned Markov matrices into scenario-conditioned
    matrices for migration-based ECL or challenger analysis.

    Method
    ------
    The default mapping uses `baseline -> baseline`, `downside -> stress`, and
    `upside -> benign`. Missing regimes fall back to the supplied portfolio
    matrix so scenario analysis remains runnable on small samples.

    Parameters
    ----------
    regime_matrices:
        Regime-to-transition-matrix mapping.
    fallback_matrix:
        Portfolio-level transition matrix used when a regime is missing.
    scenario_regime_map:
        Optional explicit scenario-to-regime mapping.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Scenario-to-transition-matrix mapping.

    Raises
    ------
    ValueError
        Raised when `fallback_matrix` is not row-stochastic.

    Notes
    -----
    The function makes the macro channel explicit. If PD already includes macro
    covariates, these scenario matrices should be used for migration explanation
    or challenger comparison rather than as an additional PD multiplier.

    Edge Cases
    ----------
    Empty `regime_matrices` returns every scenario mapped to the fallback
    matrix.

    References
    ----------
    - IFRS Foundation, "IFRS 9 Financial Instruments."
    - Lando, D., and Skodeberg, T. M. (2002), "Analyzing Rating Transitions and
      Rating Drift with Continuous Observations."
    """

    fallback = _as_square_transition_matrix(fallback_matrix)
    mapping = dict(scenario_regime_map or {"baseline": "baseline", "downside": "stress", "upside": "benign"})
    return {scenario: _as_square_transition_matrix(regime_matrices.get(regime, fallback)) for scenario, regime in mapping.items()}


def aggregate_transition_matrix(
    transition_matrix: pd.DataFrame,
    state_to_bucket: Mapping[str, str],
) -> pd.DataFrame:
    """Aggregate a state transition matrix to coarser buckets.

    Summary
    -------
    Translate detailed delinquency-state migration into stage, bucket, or other
    coarse transition probabilities.

    Method
    ------
    Rows and columns are mapped from detailed states to coarse buckets. Row mass
    is averaged equally across detailed origin states in the same bucket and
    destination probabilities are summed into destination buckets.

    Parameters
    ----------
    transition_matrix:
        Row-stochastic detailed transition matrix.
    state_to_bucket:
        Mapping from every detailed state to a coarse bucket label.

    Returns
    -------
    pandas.DataFrame
        Row-stochastic bucket-level transition matrix.

    Raises
    ------
    ValueError
        Raised when one or more states lack bucket mappings.

    Notes
    -----
    Equal weighting is a transparent default. For production use, replace it
    with exposure-weighted or state-occupancy-weighted aggregation.

    Edge Cases
    ----------
    Buckets containing one detailed state are copied directly.

    References
    ----------
    - IFRS Foundation, "IFRS 9 Financial Instruments."
    - Kemeny, J. G., and Snell, J. L. (1976), *Finite Markov Chains*.
    """

    matrix = _as_square_transition_matrix(transition_matrix)
    missing = [state for state in matrix.index if state not in state_to_bucket]
    if missing:
        raise ValueError(f"Missing bucket mapping for states: {missing}")
    buckets = tuple(dict.fromkeys(state_to_bucket[state] for state in matrix.index))
    aggregated = pd.DataFrame(0.0, index=buckets, columns=buckets)
    for origin_bucket in buckets:
        origin_states = [state for state in matrix.index if state_to_bucket[state] == origin_bucket]
        row = matrix.loc[origin_states].mean(axis=0)
        for destination_state, probability in row.items():
            aggregated.loc[origin_bucket, state_to_bucket[destination_state]] += float(probability)
    aggregated = aggregated.div(aggregated.sum(axis=1).replace(0.0, 1.0), axis=0)
    return aggregated


def stage_transition_matrix(
    transition_matrix: pd.DataFrame,
    stage_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Aggregate delinquency migration to IFRS-style stage movement.

    Summary
    -------
    Explain stage migration through a Markov transition matrix rather than only
    through realised reporting summaries.

    Method
    ------
    The default map treats current loans as Stage 1, early and serious arrears
    as Stage 2, default as Stage 3, and prepay/maturity as closed. The detailed
    transition matrix is aggregated to these buckets.

    Parameters
    ----------
    transition_matrix:
        Detailed Markov transition matrix.
    stage_map:
        Optional custom state-to-stage mapping.

    Returns
    -------
    pandas.DataFrame
        Stage-level transition matrix.

    Raises
    ------
    ValueError
        Raised when required mappings are missing.

    Notes
    -----
    This is a modelling explanation of migration pressure, not an accounting
    policy. Actual IFRS 9 stage allocation also uses origination deterioration,
    forbearance, and other policy criteria.

    Edge Cases
    ----------
    Custom maps can exclude a separate closed bucket by mapping prepayment into
    another terminal label.

    References
    ----------
    - IFRS Foundation, "IFRS 9 Financial Instruments."
    - Jarrow, R. A., Lando, D., and Turnbull, S. M. (1997), "A Markov Model for
      the Term Structure of Credit Risk Spreads."
    """

    mapping = dict(
        stage_map
        or {
            "current": "stage_1",
            "dpd_1_29": "stage_2",
            "dpd_30_89": "stage_2",
            "default": "stage_3",
            "prepay_mature": "closed",
        }
    )
    return aggregate_transition_matrix(transition_matrix, mapping)


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


def matrix_log_generator(
    transition_matrix: pd.DataFrame,
    step_length: float = 1.0,
    tolerance: float = 1e-8,
) -> GeneratorEmbeddingDiagnostics:
    """Estimate a continuous-time generator candidate using a matrix logarithm.

    Summary
    -------
    Compute `Q = log(P) / dt` and report whether the result is a valid
    continuous-time Markov-chain generator.

    Method
    ------
    A continuous-time Markov chain satisfies `P(dt) = exp(dt Q)`. The formal
    inverse is therefore `Q = log(P(dt)) / dt`. The function computes the
    principal matrix logarithm, drops negligible imaginary parts, reconstructs
    `exp(dt Q)`, and checks the generator constraints: rows sum to zero,
    off-diagonal entries are non-negative transition intensities, and diagonal
    entries are non-positive exit rates.

    Parameters
    ----------
    transition_matrix:
        Empirical or modelled row-stochastic one-period transition matrix.
    step_length:
        Length of the period represented by `transition_matrix`.
    tolerance:
        Numerical tolerance for imaginary parts, row sums, and rate signs.

    Returns
    -------
    GeneratorEmbeddingDiagnostics
        Candidate generator, validity flag, reconstruction error, and diagnostic
        messages explaining whether the empirical matrix appears embeddable.

    Raises
    ------
    ValueError
        Raised when the transition matrix is invalid, `step_length <= 0`, or
        `tolerance <= 0`.

    Notes
    -----
    This is intentionally diagnostic rather than the default generator
    estimator. Empirical credit matrices can contain sampling noise, absorbing
    rows, or zero entries that make the matrix logarithm produce negative
    off-diagonal rates. The simple `(P - I) / dt` generator remains the stable
    baseline in this lab; the matrix logarithm is useful when the fitted matrix
    passes the generator checks.

    Edge Cases
    ----------
    Absorbing states can produce singular transition matrices. `scipy.logm`
    may still return a candidate, but validity is determined by the generator
    checks rather than by successful numerical evaluation alone.

    References
    ----------
    - Israel, R. B., Rosenthal, J. S., and Wei, J. Z. (2001), "Finding
      Generators for Markov Chains via Empirical Transition Matrices, with
      Applications to Credit Ratings."
    - Lando, D., and Skodeberg, T. M. (2002), "Analyzing Rating Transitions and
      Rating Drift with Continuous Observations."
    """

    if step_length <= 0:
        raise ValueError("step_length must be positive.")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive.")

    matrix = _as_square_transition_matrix(transition_matrix)
    raw_log = logm(matrix.to_numpy(dtype=float)) / float(step_length)
    max_imaginary = float(np.max(np.abs(np.imag(raw_log)))) if raw_log.size else 0.0
    generator_array = np.real(raw_log)

    off_diagonal = generator_array.copy()
    np.fill_diagonal(off_diagonal, np.nan)
    min_off_diagonal = float(np.nanmin(off_diagonal))
    row_sum_abs = float(np.max(np.abs(generator_array.sum(axis=1))))
    diagonal = np.diag(generator_array)
    reconstruction = expm(generator_array * float(step_length))
    reconstruction_error = float(np.max(np.abs(reconstruction - matrix.to_numpy(dtype=float))))

    is_valid = (
        max_imaginary <= tolerance
        and min_off_diagonal >= -tolerance
        and row_sum_abs <= tolerance
        and bool((diagonal <= tolerance).all())
    )
    message = (
        "matrix logarithm is a valid CTMC generator within tolerance"
        if is_valid
        else "matrix logarithm candidate fails CTMC generator checks; use as diagnostic only"
    )
    generator = pd.DataFrame(generator_array, index=matrix.index, columns=matrix.columns)
    return GeneratorEmbeddingDiagnostics(
        generator=generator,
        is_valid_generator=is_valid,
        max_imaginary_part=max_imaginary,
        min_off_diagonal=min_off_diagonal,
        max_row_sum_abs=row_sum_abs,
        reconstruction_error=reconstruction_error,
        message=message,
    )


def project_generator_to_valid_rates(generator: pd.DataFrame, tolerance: float = 1e-12) -> pd.DataFrame:
    """Project a generator-like matrix onto valid transition-rate signs.

    Summary
    -------
    Convert a noisy generator candidate into a conservative rate matrix by
    clipping negative off-diagonal entries and resetting diagonals to row exits.

    Method
    ------
    Off-diagonal entries below zero are clipped to zero. Each diagonal is then
    set to the negative sum of the off-diagonal row rates, enforcing zero row
    sums. This is not a statistical estimator; it is a pragmatic repair step
    for diagnostics and sensitivity tests.

    Parameters
    ----------
    generator:
        Square generator-like matrix with matching labels.
    tolerance:
        Values above `-tolerance` are treated as numerical zero.

    Returns
    -------
    pandas.DataFrame
        Valid CTMC rate matrix with non-negative off-diagonal entries and zero
        row sums.

    Raises
    ------
    ValueError
        Raised when labels do not match or `tolerance < 0`.

    Notes
    -----
    Production credit migration work should estimate a generator directly from
    event-time data or use constrained optimisation. This helper only prevents
    a failed matrix logarithm from contaminating later diagnostics.

    Edge Cases
    ----------
    Rows with no positive off-diagonal rates become absorbing rows.

    References
    ----------
    - Israel, R. B., Rosenthal, J. S., and Wei, J. Z. (2001), "Finding
      Generators for Markov Chains via Empirical Transition Matrices, with
      Applications to Credit Ratings."
    """

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative.")
    if list(generator.index) != list(generator.columns):
        raise ValueError("Generator index and columns must contain the same states in the same order.")
    rates = generator.astype(float).copy()
    values = rates.to_numpy(dtype=float)
    for idx in range(values.shape[0]):
        for jdx in range(values.shape[1]):
            if idx == jdx:
                continue
            if values[idx, jdx] < 0 and values[idx, jdx] >= -tolerance:
                values[idx, jdx] = 0.0
            elif values[idx, jdx] < 0:
                values[idx, jdx] = 0.0
        values[idx, idx] = -float(values[idx, np.arange(values.shape[1]) != idx].sum())
    return pd.DataFrame(values, index=rates.index, columns=rates.columns)


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
        setting for this lab because credit migration is usually
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


def reversibility_diagnostics(
    transition_matrix: pd.DataFrame,
    weights: Mapping[str, float] | pd.Series | None = None,
) -> pd.DataFrame:
    """Diagnose detailed-balance failures in a finite credit migration matrix.

    Summary
    -------
    Compare forward and reverse probability flows across pairs of credit states.

    Method
    ------
    For a reference state distribution `m`, detailed balance requires
    `m_i P_ij = m_j P_ji` for every state pair. The function computes these two
    flows, their absolute imbalance, and a relative imbalance ratio for each
    unordered state pair.

    Parameters
    ----------
    transition_matrix:
        Row-stochastic transition matrix.
    weights:
        Optional reference distribution over states. Uniform weights are used
        when omitted.

    Returns
    -------
    pandas.DataFrame
        Pair-level reversibility diagnostics.

    Raises
    ------
    ValueError
        Raised when the matrix or weights are invalid.

    Notes
    -----
    Credit migration is often non-reversible because deterioration and repair
    are not mirror-image economic mechanisms. This diagnostic makes that
    asymmetry visible before a symmetric energy is used for score smoothness.

    Edge Cases
    ----------
    Absorbing states generally create large balance failures unless the
    reference distribution already puts all mass in terminal states.

    References
    ----------
    - Lando, D., and Skodeberg, T. M. (2002), "Analyzing Rating Transitions and
      Rating Drift with Continuous Observations."
    - Fukushima, Oshima, and Takeda, "Dirichlet Forms and Symmetric Markov
      Processes", 2nd revised and extended edition, 2011.
    """

    matrix = _as_square_transition_matrix(transition_matrix)
    states = list(matrix.index)
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

    rows: list[dict[str, object]] = []
    for i, origin in enumerate(states):
        for destination in states[i + 1 :]:
            forward_flow = float(reference.loc[origin] * matrix.loc[origin, destination])
            reverse_flow = float(reference.loc[destination] * matrix.loc[destination, origin])
            denominator = max(abs(forward_flow) + abs(reverse_flow), 1e-12)
            rows.append(
                {
                    "state_i": origin,
                    "state_j": destination,
                    "flow_i_to_j": forward_flow,
                    "flow_j_to_i": reverse_flow,
                    "absolute_imbalance": abs(forward_flow - reverse_flow),
                    "relative_imbalance": abs(forward_flow - reverse_flow) / denominator,
                }
            )
    return pd.DataFrame(rows).sort_values("absolute_imbalance", ascending=False).reset_index(drop=True)


def score_smoothness_diagnostics(
    transition_matrix: pd.DataFrame,
    state_values: Mapping[str, float] | pd.Series,
    weights: Mapping[str, float] | pd.Series | None = None,
    symmetrize: bool = True,
    top_n: int = 10,
) -> dict[str, object]:
    """Explain which migration edges drive score roughness.

    Summary
    -------
    Decompose the finite-state Dirichlet energy into edge-level contributions.

    Method
    ------
    The function builds the same conductance used by
    `dirichlet_transition_energy`, computes squared score differences across
    edges, and reports the largest contributors. High-contribution edges show
    where a score or stage scale changes sharply across commonly observed
    migrations.

    Parameters
    ----------
    transition_matrix:
        Row-stochastic transition matrix.
    state_values:
        Numeric score, stage, or risk scale by state.
    weights:
        Optional state reference distribution.
    symmetrize:
        Whether to use symmetric conductance.
    top_n:
        Number of largest edge contributions to return.

    Returns
    -------
    dict[str, object]
        Total energy, edge table, and a plain-language interpretation string.

    Raises
    ------
    ValueError
        Raised when matrix, scores, weights, or `top_n` are invalid.

    Notes
    -----
    This turns the abstract energy into a model-review question: are the biggest
    score jumps located where observed borrower movements justify them?

    Edge Cases
    ----------
    Constant scores return zero energy and an edge table with zero
    contributions.

    References
    ----------
    - Fukushima, Oshima, and Takeda, "Dirichlet Forms and Symmetric Markov
      Processes", 2nd revised and extended edition, 2011.
    - Beurling and Deny, "Dirichlet Spaces", 1958-1959.
    """

    if top_n <= 0:
        raise ValueError("top_n must be positive.")
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
    conductance = 0.5 * (m[:, None] * p + m[None, :] * p.T) if symmetrize else m[:, None] * p
    rows: list[dict[str, object]] = []
    for i, origin in enumerate(states):
        for j, destination in enumerate(states):
            if i == j:
                continue
            contribution = 0.5 * float(conductance[i, j] * (f[i] - f[j]) ** 2)
            if contribution <= 0:
                continue
            rows.append(
                {
                    "origin_state": origin,
                    "destination_state": destination,
                    "conductance": float(conductance[i, j]),
                    "score_difference": float(f[i] - f[j]),
                    "energy_contribution": contribution,
                }
            )
    edge_table = pd.DataFrame(rows)
    if not edge_table.empty:
        edge_table = edge_table.sort_values("energy_contribution", ascending=False).head(top_n).reset_index(drop=True)
    total_energy = dirichlet_transition_energy(matrix, values, reference, symmetrize=symmetrize)
    return {
        "total_energy": total_energy,
        "top_edges": edge_table,
        "interpretation": "Large contributions indicate score jumps across migration edges that exchange material probability mass.",
    }


def regularize_state_scores(
    transition_matrix: pd.DataFrame,
    state_values: Mapping[str, float] | pd.Series,
    alpha: float = 1.0,
    weights: Mapping[str, float] | pd.Series | None = None,
) -> pd.Series:
    """Smooth a state score over the observed migration graph.

    Summary
    -------
    Penalise excessive roughness in a rating, stage, or credit-quality scale
    while staying close to the original state values.

    Method
    ------
    The function builds a symmetric transition conductance and graph Laplacian
    `L`. It then solves `(W + alpha L) f_smooth = W f_raw`, where `W` is the
    diagonal reference-measure matrix. The solution is the score closest to the
    original scale after penalising large changes across commonly observed
    migration edges.

    Parameters
    ----------
    transition_matrix:
        Row-stochastic transition matrix.
    state_values:
        Raw numeric state scores.
    alpha:
        Non-negative smoothness penalty. `0` returns the original scores.
    weights:
        Optional state reference distribution.

    Returns
    -------
    pandas.Series
        Smoothed scores by state.

    Raises
    ------
    ValueError
        Raised when inputs are invalid or `alpha < 0`.

    Notes
    -----
    This is the V5 Dirichlet-form regularisation layer. It is a diagnostic
    model-design tool, not a claim that the true credit-risk process is
    reversible or symmetric.

    Edge Cases
    ----------
    If `alpha=0`, no smoothing is applied. If all raw scores are constant, the
    result remains constant for every `alpha`.

    References
    ----------
    - Fukushima, Oshima, and Takeda, "Dirichlet Forms and Symmetric Markov
      Processes", 2nd revised and extended edition, 2011.
    - Beurling and Deny, "Dirichlet Spaces", 1958-1959.
    """

    if alpha < 0:
        raise ValueError("alpha must be non-negative.")
    matrix = _as_square_transition_matrix(transition_matrix)
    states = list(matrix.index)
    values = pd.Series(state_values, dtype=float).reindex(states)
    if values.isna().any():
        missing = values.index[values.isna()].tolist()
        raise ValueError(f"Missing state values for: {missing}")
    if alpha == 0:
        return values.copy()
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
    conductance = 0.5 * (m[:, None] * p + m[None, :] * p.T)
    laplacian = np.diag(conductance.sum(axis=1)) - conductance
    weight_matrix = np.diag(m)
    lhs = weight_matrix + float(alpha) * laplacian
    rhs = weight_matrix @ values.to_numpy(dtype=float)
    smooth = np.linalg.solve(lhs, rhs)
    return pd.Series(smooth, index=states, name="smoothed_score")
