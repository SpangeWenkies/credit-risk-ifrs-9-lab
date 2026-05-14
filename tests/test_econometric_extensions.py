from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_lab import PortfolioConfig, fit_survival_pd_model, generate_portfolio_timeseries, score_portfolio  # noqa: E402
from credit_risk_lab.econometrics.calibration import calibration_table, intercept_shift_recalibration, isotonic_recalibration, segment_calibration_drift  # noqa: E402
from credit_risk_lab.econometrics.causal import omitted_variable_sensitivity, policy_shock_sensitivity, variable_identification_audit  # noqa: E402
from credit_risk_lab.econometrics.continuous_time import (  # noqa: E402
    build_compensated_process_from_intervals,
    build_default_counting_process,
    estimate_ctmc_generator_from_durations,
    estimate_default_intensity_from_intervals,
    estimate_piecewise_default_intensity,
    survival_probability_from_intensity,
)
from credit_risk_lab.econometrics.continuous_state import (  # noqa: E402
    beurling_deny_credit_decomposition,
    build_continuous_credit_state_grid,
    cheeger_energy_credit_proxy,
    default_boundary_capacity_proxy,
    default_probability_from_generator,
    ou_credit_quality_generator,
    polar_state_diagnostic,
    regularity_diagnostics_for_grid,
)
from credit_risk_lab.econometrics.duration import competing_risk_cumulative_incidence, cure_relapse_table, duration_hazard_table, fit_competing_risk_logits  # noqa: E402
from credit_risk_lab.econometrics.forward_hazard import (  # noqa: E402
    cumulative_pd_from_hazard_path,
    score_forward_pd_paths,
)
from credit_risk_lab.econometrics.heterogeneity import empirical_bayes_segment_shrinkage, segment_performance_table  # noqa: E402
from credit_risk_lab.econometrics.limited_dep import fit_binary_logit, predict_binary_logit  # noqa: E402
from credit_risk_lab.econometrics.macro import build_macro_scenario_paths, fit_ar1_macro_model, fit_var_macro_model, forecast_macro_path, forecast_var_macro_path  # noqa: E402
from credit_risk_lab.econometrics.markov import fit_markov_transition_model, markov_implied_default_pd, transition_matrices_by_group  # noqa: E402
from credit_risk_lab.econometrics.measurement_error import (  # noqa: E402
    inject_numeric_measurement_error,
    missingness_by_segment,
    prediction_noise_sensitivity,
)
from credit_risk_lab.econometrics.model_selection import CandidateSpec, compare_binary_model_specs, select_best_model  # noqa: E402
from credit_risk_lab.econometrics.panel import add_interaction_terms, add_polynomial_terms, add_vintage_and_age_columns, cohort_performance_table, poolability_diagnostic, within_transform  # noqa: E402


def test_forward_hazard_and_macro_paths_remove_flat_hazard_assumption() -> None:
    dataset = generate_portfolio_timeseries(PortfolioConfig(random_seed=41, periods=10, num_borrowers=80, num_loans=140))
    model = fit_survival_pd_model(dataset.performance)
    scores = score_portfolio(dataset, model, dataset.performance["snapshot_date"].max())

    macro_model = fit_ar1_macro_model(dataset.macro, ["unemployment_rate", "policy_rate", "house_price_growth"])
    baseline = forecast_macro_path(macro_model, horizon_quarters=6)
    var_model = fit_var_macro_model(dataset.macro, ["unemployment_rate", "policy_rate", "house_price_growth"])
    var_path = forecast_var_macro_path(var_model, horizon_quarters=3)
    paths = build_macro_scenario_paths(
        baseline,
        {
            "baseline": {},
            "downside": {"unemployment_rate": 0.015, "house_price_growth": -0.025},
        },
    )

    forward = score_forward_pd_paths(scores.snapshot_scores.head(10), model, paths["downside"], max_horizon_quarters=6)

    assert not forward.empty
    assert len(var_path) == 3
    assert (forward["pd_lifetime_forward"] >= forward["pd_12m_forward"]).all()
    assert np.isclose(cumulative_pd_from_hazard_path([0.1, 0.2]), 0.28)


def test_calibration_duration_and_continuous_time_outputs_reconcile() -> None:
    dataset = generate_portfolio_timeseries(PortfolioConfig(random_seed=43, periods=9, num_borrowers=80, num_loans=140))
    model = fit_survival_pd_model(dataset.performance)
    scores = score_portfolio(dataset, model, dataset.performance["snapshot_date"].max())
    history = scores.scored_history

    table = calibration_table(history, "pd_12m", "observed_default_12m", n_bins=5)
    recalibrated, shift = intercept_shift_recalibration(history["pd_12m"], target_default_rate=0.08)
    isotonic = isotonic_recalibration(history["pd_12m"], history["observed_default_12m"])
    drift = segment_calibration_drift(history.head(40), history.tail(40), "segment", "pd_12m", "observed_default_12m")
    assert len(table) <= 5
    assert abs(float(recalibrated.mean()) - 0.08) < 1e-5
    assert isinstance(shift, float)
    assert isotonic.between(0.0, 1.0).all()
    assert "calibration_gap_change" in drift.columns

    hazard = duration_hazard_table(dataset.performance)
    incidence = competing_risk_cumulative_incidence(hazard)
    cure_relapse = cure_relapse_table(dataset.performance)
    competing_models = fit_competing_risk_logits(
        dataset.performance,
        numeric_columns=["ltv", "dti", "days_past_due", "rating_rank"],
        categorical_columns=["segment"],
    )
    assert (incidence["remaining_active_probability"].between(0.0, 1.0)).all()
    assert {"cures", "relapses"}.issubset(cure_relapse.columns)
    assert competing_models

    intensity = estimate_piecewise_default_intensity(dataset.performance, group_columns=["segment"])
    portfolio_intensity = estimate_piecewise_default_intensity(dataset.performance)["intensity"].iloc[0]
    counting = build_default_counting_process(dataset.performance, intensity=portfolio_intensity)
    assert not intensity.empty
    assert np.isclose(
        counting["compensated_process"].iloc[-1],
        counting["cumulative_defaults"].iloc[-1] - counting["cumulative_compensator"].iloc[-1],
    )
    assert 0.0 < survival_probability_from_intensity([portfolio_intensity] * 4) <= 1.0

    intervals = pd.DataFrame(
        {
            "loan_id": ["A", "B", "C", "D"],
            "start": [0.0, 0.0, 0.0, 1.0],
            "end": [1.0, 2.0, 1.5, 2.5],
            "state": ["current", "current", "dpd_1_29", "dpd_1_29"],
            "next_state": ["dpd_1_29", "default", "current", "default"],
            "default_event": [0, 1, 0, 1],
            "segment": ["cards", "cards", "loans", "loans"],
        }
    )
    exact_intensity = estimate_default_intensity_from_intervals(intervals, "start", "end", group_columns=["segment"])
    assert (exact_intensity["intensity"] >= 0.0).all()
    ctmc = estimate_ctmc_generator_from_durations(intervals, "end", smoothing=0.0)
    np.testing.assert_allclose(ctmc.sum(axis=1).to_numpy(), np.zeros(len(ctmc)), atol=1e-12)
    compensated = build_compensated_process_from_intervals(intervals, intensity=0.25, start_time_column="start", end_time_column="end")
    assert np.isclose(
        compensated["compensated_process"].iloc[-1],
        compensated["cumulative_defaults"].iloc[-1] - compensated["cumulative_compensator"].iloc[-1],
    )


def test_continuous_state_credit_process_components_are_available() -> None:
    grid = build_continuous_credit_state_grid(x_min=-3.0, x_max=3.0, n_points=41, default_boundary=-1.5)
    generator = ou_credit_quality_generator(
        grid,
        mean_reversion=0.25,
        long_run_mean=0.5,
        volatility=0.35,
        killing_intensity=0.01,
        downward_jump_intensity=0.02,
        downward_jump_size=0.75,
    )
    np.testing.assert_allclose(generator.sum(axis=1).to_numpy(), np.zeros(len(generator)), atol=1e-10)
    start_state = generator.index[len(generator.index) // 2]
    probs = default_probability_from_generator(generator, start_state, [0.0, 1.0, 2.0])
    assert probs["default_probability"].is_monotonic_increasing

    active_states = [state for state in generator.index if state != "default"]
    values = pd.Series(np.linspace(0.0, 1.0, len(active_states)), index=active_states)
    decomposition = beurling_deny_credit_decomposition(generator, values)
    assert decomposition["total_energy"] >= decomposition["local_energy"] >= 0.0
    capacity = default_boundary_capacity_proxy(generator)
    assert capacity["capacity_proxy"] >= 0.0
    polar = polar_state_diagnostic(generator, target_state=start_state)
    assert "polar_like" in polar
    cheeger = cheeger_energy_credit_proxy(grid.active_grid, np.linspace(0.0, 1.0, len(grid.active_grid)))
    assert cheeger >= 0.0
    regularity = regularity_diagnostics_for_grid(grid)
    assert regularity["finite_grid"]


def test_panel_limited_dep_heterogeneity_measurement_error_and_selection() -> None:
    dataset = generate_portfolio_timeseries(PortfolioConfig(random_seed=47, periods=8, num_borrowers=70, num_loans=120))
    panel = add_vintage_and_age_columns(dataset.performance)
    within = within_transform(panel, ["dti", "ltv"], entity_column="loan_id")
    nonlinear = add_polynomial_terms(panel.head(10), ["dti"], degree=3)
    interacted = add_interaction_terms(panel.head(10), ["dti"], ["ltv"])
    cohort = cohort_performance_table(panel)
    poolability = poolability_diagnostic(panel.assign(prediction=0.05), "segment", "default_next_period", "prediction")
    assert {"dti_within", "ltv_within"}.issubset(within.columns)
    assert {"dti_pow_2", "dti_pow_3"}.issubset(nonlinear.columns)
    assert "dti_x_ltv" in interacted.columns
    assert not cohort.empty
    assert not poolability.empty

    sample = panel.copy()
    result = fit_binary_logit(
        sample,
        "default_next_period",
        numeric_columns=["ltv", "dti", "days_past_due", "rating_rank"],
        categorical_columns=["segment"],
    )
    predictions = predict_binary_logit(result, sample.head(20), ["ltv", "dti", "days_past_due", "rating_rank"], ["segment"])
    assert predictions.between(0.0, 1.0).all()

    scored = sample.assign(prediction=predict_binary_logit(result, sample, ["ltv", "dti", "days_past_due", "rating_rank"], ["segment"]))
    segment_table = segment_performance_table(scored, "segment", "prediction", "default_next_period")
    shrinkage = empirical_bayes_segment_shrinkage(scored, "segment", "default_next_period", prior_strength=10.0)
    assert not segment_table.empty
    assert "shrunk_rate" in shrinkage.columns

    missing = missingness_by_segment(sample, ["dti", "ltv"], segment_column="segment")
    noisy = inject_numeric_measurement_error(sample.head(20), {"dti": 0.01}, random_seed=1)
    sensitivity = prediction_noise_sensitivity(sample.head(20)["dti"], noisy["dti"])
    assert not missing.empty
    assert sensitivity["mean_absolute_change"] > 0.0

    comparison = compare_binary_model_specs(
        sample,
        "default_next_period",
        [
            CandidateSpec("small", ("ltv", "dti")),
            CandidateSpec("behavioural", ("ltv", "dti", "days_past_due", "rating_rank"), ("segment",)),
        ],
    )
    best = select_best_model(comparison, "aic")
    assert best["status"] == "fit"
    full = fit_binary_logit(sample, "default_next_period", ["ltv", "dti", "days_past_due", "rating_rank"], ["segment"])
    restricted = fit_binary_logit(sample, "default_next_period", ["ltv", "dti"], ["segment"])
    omitted = omitted_variable_sensitivity(full, restricted, "ltv")
    assert "absolute_change" in omitted


def test_causal_and_grouped_markov_extensions_are_available() -> None:
    dataset = generate_portfolio_timeseries(PortfolioConfig(random_seed=53, periods=8, num_borrowers=70, num_loans=120))
    audit = variable_identification_audit(
        ["unemployment_rate", "days_past_due"],
        roles={"unemployment_rate": "macro", "days_past_due": "post_treatment"},
    )
    assert set(audit["variable"]) == {"unemployment_rate", "days_past_due"}

    frame = pd.DataFrame({"policy_rate": [0.02, 0.03], "base": [1.0, 2.0]})
    shocked = policy_shock_sensitivity(frame, "policy_rate", 0.01, lambda df: df["base"] + 10 * df["policy_rate"])
    assert (shocked["delta_score"] > 0).all()

    markov_model = fit_markov_transition_model(dataset.performance, smoothing=0.1)
    grouped = transition_matrices_by_group(markov_model.transition_panel, "segment", smoothing=0.1)
    assert grouped
    pd_4q = markov_implied_default_pd(markov_model.transition_matrix, "current", 4)
    assert 0.0 <= pd_4q <= 1.0
