"""Econometric extension points for the credit risk lab.

This package contains the first implemented layer of the repo's econometrics
extensions: calibration, panel diagnostics, duration tables, forward hazard
paths, macro scenario paths, Markov migration, continuous-time intensity
diagnostics, measurement-error sensitivity, causal audits, heterogeneity
checks, and model-selection helpers.
"""

from .calibration import brier_score, calibration_table, intercept_shift_recalibration
from .causal import policy_shock_sensitivity, variable_identification_audit
from .continuous_time import (
    build_default_counting_process,
    estimate_piecewise_default_intensity,
    survival_probability_from_intensity,
)
from .duration import baseline_hazard_by_segment, competing_risk_cumulative_incidence, duration_hazard_table
from .forward_hazard import build_forward_panel, cumulative_pd_from_hazard_path, score_forward_pd_paths
from .heterogeneity import coefficient_stability_table, fit_segment_binary_logits, segment_performance_table
from .limited_dep import fit_binary_logit, predict_binary_logit, prepare_regression_design
from .macro import MacroAR1Model, build_macro_scenario_paths, fit_ar1_macro_model, forecast_macro_path
from .markov import (
    ABSORBING_STATES,
    CREDIT_STATES,
    MarkovTransitionModel,
    absorption_summary,
    build_transition_panel,
    dirichlet_transition_energy,
    fit_markov_transition_model,
    markov_implied_default_pd,
    n_step_transition_matrix,
    transition_generator,
    transition_matrices_by_group,
)
from .measurement_error import inject_numeric_measurement_error, missingness_by_segment, prediction_noise_sensitivity
from .model_selection import CandidateSpec, compare_binary_model_specs, select_best_model
from .panel import add_vintage_and_age_columns, cohort_performance_table, within_transform

__all__ = [
    "ABSORBING_STATES",
    "CREDIT_STATES",
    "CandidateSpec",
    "MacroAR1Model",
    "MarkovTransitionModel",
    "absorption_summary",
    "add_vintage_and_age_columns",
    "baseline_hazard_by_segment",
    "brier_score",
    "build_default_counting_process",
    "build_forward_panel",
    "build_macro_scenario_paths",
    "build_transition_panel",
    "calibration_table",
    "coefficient_stability_table",
    "cohort_performance_table",
    "compare_binary_model_specs",
    "competing_risk_cumulative_incidence",
    "cumulative_pd_from_hazard_path",
    "dirichlet_transition_energy",
    "duration_hazard_table",
    "estimate_piecewise_default_intensity",
    "fit_ar1_macro_model",
    "fit_binary_logit",
    "fit_markov_transition_model",
    "fit_segment_binary_logits",
    "forecast_macro_path",
    "inject_numeric_measurement_error",
    "intercept_shift_recalibration",
    "markov_implied_default_pd",
    "missingness_by_segment",
    "n_step_transition_matrix",
    "policy_shock_sensitivity",
    "prediction_noise_sensitivity",
    "predict_binary_logit",
    "prepare_regression_design",
    "score_forward_pd_paths",
    "segment_performance_table",
    "select_best_model",
    "survival_probability_from_intensity",
    "transition_generator",
    "transition_matrices_by_group",
    "variable_identification_audit",
    "within_transform",
]
