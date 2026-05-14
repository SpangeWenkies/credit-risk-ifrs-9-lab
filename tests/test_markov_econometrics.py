from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_lab import PortfolioConfig, generate_portfolio_timeseries  # noqa: E402
from credit_risk_lab.econometrics.markov import (  # noqa: E402
    ABSORBING_STATES,
    CREDIT_STATES,
    absorption_summary,
    assign_macro_regime,
    build_transition_panel,
    build_markov_scenario_matrices,
    compare_markov_to_survival_pd,
    covariate_markov_default_pd,
    dirichlet_transition_energy,
    fit_covariate_transition_model,
    fit_macro_regime_transition_matrices,
    fit_markov_transition_model,
    matrix_log_generator,
    n_step_transition_matrix,
    predict_covariate_transition_probabilities,
    regularize_state_scores,
    reversibility_diagnostics,
    score_smoothness_diagnostics,
    stage_transition_matrix,
    transition_generator,
)


def test_transition_panel_infers_terminal_states_and_excludes_censoring() -> None:
    performance = pd.DataFrame(
        [
            {
                "loan_id": "L1",
                "snapshot_date": pd.Timestamp("2024-03-31"),
                "days_past_due": 0,
                "default_next_period": 0,
                "prepayment_flag": 0,
                "remaining_term_quarters": 2,
                "balance": 100.0,
            },
            {
                "loan_id": "L1",
                "snapshot_date": pd.Timestamp("2024-06-30"),
                "days_past_due": 45,
                "default_next_period": 1,
                "prepayment_flag": 0,
                "remaining_term_quarters": 1,
                "balance": 90.0,
            },
            {
                "loan_id": "L2",
                "snapshot_date": pd.Timestamp("2024-03-31"),
                "days_past_due": 0,
                "default_next_period": 0,
                "prepayment_flag": 1,
                "remaining_term_quarters": 8,
                "balance": 1_000.0,
            },
            {
                "loan_id": "L3",
                "snapshot_date": pd.Timestamp("2024-03-31"),
                "days_past_due": 0,
                "default_next_period": 0,
                "prepayment_flag": 0,
                "remaining_term_quarters": 8,
                "balance": 1_000.0,
            },
        ]
    )

    transitions = build_transition_panel(performance)

    assert len(transitions) == 3
    observed = transitions.set_index(["loan_id", "snapshot_date"])["next_state"].to_dict()
    assert observed[("L1", pd.Timestamp("2024-03-31"))] == "dpd_30_89"
    assert observed[("L1", pd.Timestamp("2024-06-30"))] == "default"
    assert observed[("L2", pd.Timestamp("2024-03-31"))] == "prepay_mature"
    assert "L3" not in set(transitions["loan_id"])


def test_markov_transition_model_is_row_stochastic_and_absorbing() -> None:
    dataset = generate_portfolio_timeseries(PortfolioConfig(random_seed=29, periods=10, num_borrowers=90, num_loans=150))
    model = fit_markov_transition_model(dataset.performance, smoothing=0.1)

    assert not model.transition_panel.empty
    assert list(model.transition_matrix.index) == list(CREDIT_STATES)
    np.testing.assert_allclose(model.transition_matrix.sum(axis=1).to_numpy(), np.ones(len(CREDIT_STATES)))
    for state in ABSORBING_STATES:
        assert model.transition_matrix.loc[state, state] == 1.0

    generator = transition_generator(model.transition_matrix)
    np.testing.assert_allclose(generator.sum(axis=1).to_numpy(), np.zeros(len(CREDIT_STATES)), atol=1e-10)
    off_diagonal = generator.to_numpy().copy()
    np.fill_diagonal(off_diagonal, 0.0)
    assert (off_diagonal >= -1e-12).all()


def test_semigroup_absorption_and_dirichlet_energy_diagnostics() -> None:
    transition_matrix = pd.DataFrame(
        [
            [0.70, 0.15, 0.05, 0.04, 0.06],
            [0.20, 0.45, 0.20, 0.10, 0.05],
            [0.05, 0.15, 0.35, 0.35, 0.10],
            [0.00, 0.00, 0.00, 1.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 1.00],
        ],
        index=CREDIT_STATES,
        columns=CREDIT_STATES,
    )

    four_step = n_step_transition_matrix(transition_matrix, 4)
    np.testing.assert_allclose(four_step.sum(axis=1).to_numpy(), np.ones(len(CREDIT_STATES)))
    assert four_step.loc["dpd_30_89", "default"] > transition_matrix.loc["dpd_30_89", "default"]

    summary = absorption_summary(transition_matrix)
    assert set(summary["state"]) == {"current", "dpd_1_29", "dpd_30_89"}
    assert (summary["expected_steps_to_absorption"] > 0).all()
    absorb_cols = ["prob_absorb_default", "prob_absorb_prepay_mature"]
    np.testing.assert_allclose(summary[absorb_cols].sum(axis=1).to_numpy(), np.ones(3))

    constant_energy = dirichlet_transition_energy(transition_matrix, {state: 1.0 for state in CREDIT_STATES})
    ordered_energy = dirichlet_transition_energy(
        transition_matrix,
        {"current": 1.0, "dpd_1_29": 2.0, "dpd_30_89": 3.0, "default": 4.0, "prepay_mature": 0.0},
    )
    assert constant_energy == 0.0
    assert ordered_energy > 0.0

    stage_matrix = stage_transition_matrix(transition_matrix)
    np.testing.assert_allclose(stage_matrix.sum(axis=1).to_numpy(), np.ones(len(stage_matrix)))
    smooth = regularize_state_scores(
        transition_matrix,
        {"current": 1.0, "dpd_1_29": 3.0, "dpd_30_89": 4.0, "default": 6.0, "prepay_mature": 0.0},
        alpha=0.5,
    )
    assert smooth.loc["dpd_1_29"] < 3.0
    diagnostics = score_smoothness_diagnostics(transition_matrix, smooth)
    assert diagnostics["total_energy"] >= 0.0
    reversibility = reversibility_diagnostics(transition_matrix)
    assert not reversibility.empty


def test_matrix_log_generator_and_macro_scenario_matrices() -> None:
    q = pd.DataFrame(
        [
            [-0.30, 0.20, 0.05, 0.05, 0.00],
            [0.10, -0.35, 0.15, 0.10, 0.00],
            [0.02, 0.08, -0.45, 0.30, 0.05],
            [0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00],
        ],
        index=CREDIT_STATES,
        columns=CREDIT_STATES,
    )
    p = pd.DataFrame(np.asarray(expm(q.to_numpy())), index=CREDIT_STATES, columns=CREDIT_STATES)
    diagnostics = matrix_log_generator(p)
    assert diagnostics.is_valid_generator
    np.testing.assert_allclose(diagnostics.generator.sum(axis=1).to_numpy(), np.zeros(len(CREDIT_STATES)), atol=1e-7)

    dataset = generate_portfolio_timeseries(PortfolioConfig(random_seed=61, periods=9, num_borrowers=75, num_loans=130))
    markov_model = fit_markov_transition_model(dataset.performance, smoothing=0.1)
    labelled = assign_macro_regime(markov_model.transition_panel, "unemployment_rate")
    regime_matrices = fit_macro_regime_transition_matrices(labelled, smoothing=0.1)
    scenarios = build_markov_scenario_matrices(regime_matrices, markov_model.transition_matrix)
    assert set(scenarios) == {"baseline", "downside", "upside"}
    for matrix in scenarios.values():
        np.testing.assert_allclose(matrix.sum(axis=1).to_numpy(), np.ones(len(CREDIT_STATES)))


def test_covariate_markov_model_predicts_and_compares_to_survival_pd() -> None:
    dataset = generate_portfolio_timeseries(PortfolioConfig(random_seed=67, periods=10, num_borrowers=90, num_loans=160))
    markov_model = fit_markov_transition_model(dataset.performance, smoothing=0.1)
    transition_panel = markov_model.transition_panel.copy()
    model = fit_covariate_transition_model(
        transition_panel,
        feature_columns=["rating_rank", "days_past_due", "ltv", "unemployment_rate"],
        categorical_columns=["segment"],
        min_rows=5,
        smoothing=0.1,
    )
    sample = transition_panel.head(15)
    probabilities = predict_covariate_transition_probabilities(model, sample)
    assert not probabilities.empty
    row_sums = probabilities.groupby("row_id")["probability"].sum()
    np.testing.assert_allclose(row_sums.to_numpy(), np.ones(len(row_sums)))

    default_pd = covariate_markov_default_pd(model, sample, horizon_steps=4)
    assert default_pd.between(0.0, 1.0).all()
    comparison = compare_markov_to_survival_pd(sample.assign(pd_12m=0.05), "pd_12m", 4, covariate_model=model)
    assert {"markov_pd", "pd_difference", "absolute_pd_difference"}.issubset(comparison.columns)
