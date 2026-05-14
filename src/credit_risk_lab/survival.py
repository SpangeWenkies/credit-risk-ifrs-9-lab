"""Discrete-time survival modelling for point-in-time PD estimation.

This module implements a pooled-logit discrete-time survival model on quarterly
account-performance panels. The project uses the Singer and Willett framework
because it is transparent, naturally aligns with panelised retail credit data,
and produces probabilities that can be rolled into both 12-month and lifetime
IFRS 9 horizons.

Assumptions
-----------
- Each performance row represents one active exposure observed at the start of a
  quarter, with `default_next_period` indicating default before the next
  quarter-end.
- The conditional hazard predicted for the current snapshot is held constant
  across future quarters when converting a one-quarter hazard into a 12-month or
  lifetime PD. This is a baseline simplification.
- The model is pooled across products, with segment and region effects handled
  through categorical regressors rather than separate sub-models.

Primary references
------------------
- Singer, J. D., and Willett, J. B. (1993), "It's About Time: Using
  Discrete-Time Survival Analysis to Study Duration and the Timing of Events."
  https://journals.sagepub.com/doi/10.3102/10769986018002155

Simplifications for this lab
------------------------------------------
- The implementation uses a straightforward logistic GLM from `statsmodels`
  rather than production regularisation, calibration overlays, or macro-varying
  future hazards.
- Feature engineering is intentionally transparent so the mapping from raw
  panel fields to model inputs can be audited.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .models import FeatureSpec, PDScoreFrame, PortfolioDataset, SurvivalPDModel

DEFAULT_FEATURE_SPEC = FeatureSpec(
    numeric_columns=[
        "quarters_on_book",
        "remaining_term_quarters",
        "ltv",
        "dti",
        "days_past_due",
        "rating_rank",
    ],
    categorical_columns=["segment"],
)


def _resolve_feature_spec(feature_spec: FeatureSpec | None) -> FeatureSpec:
    return feature_spec or DEFAULT_FEATURE_SPEC


def _prepare_design_matrix(
    frame: pd.DataFrame,
    feature_spec: FeatureSpec,
    design_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Create the model design matrix with stable dummy columns.

    Summary
    -------
    Convert raw panel features into the exact design matrix required by the
    pooled-logit survival model, including dummy encoding and a constant term.

    Method
    ------
    Numeric predictors are cast to float and missing values are imputed with
    zero for this synthetic project. Categorical predictors are expanded with
    one-hot encoding using `drop_first=True` to avoid a full dummy trap. When
    `design_columns` are supplied, the matrix is reindexed so scoring uses the
    same column set as model fitting.

    Parameters
    ----------
    frame:
        Input panel containing raw model features.
    feature_spec:
        Numeric, categorical, and target-column metadata.
    design_columns:
        Optional fixed design columns from a previously fitted model. When
        provided, missing columns are backfilled with zeros and extra columns are
        dropped.

    Returns
    -------
    pandas.DataFrame
        Design matrix with a `const` column and deterministic column ordering.

    Raises
    ------
    KeyError
        Raised if the input frame is missing one or more requested feature
        columns.

    Notes
    -----
    Holding the design matrix shape stable between fitting and scoring is
    essential when using pooled categorical regressors. Without reindexing, a
    scoring sample that happens to miss a segment or region would not line up
    with the fitted coefficient vector.

    Edge Cases
    ----------
    Empty frames are allowed and return an empty matrix with the requested
    columns. This keeps downstream scoring code simple when a previous snapshot
    does not exist.

    References
    ----------
    - Singer, J. D., and Willett, J. B. (1993), "It's About Time: Using
      Discrete-Time Survival Analysis to Study Duration and the Timing of
      Events." https://journals.sagepub.com/doi/10.3102/10769986018002155
    """

    required = feature_spec.numeric_columns + feature_spec.categorical_columns
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")

    numeric = frame[feature_spec.numeric_columns].astype(float).fillna(0.0)
    categoricals = pd.get_dummies(
        frame[feature_spec.categorical_columns].astype("category"),
        drop_first=True,
        dtype=float,
    )
    design = pd.concat([numeric, categoricals], axis=1)
    design = sm.add_constant(design, has_constant="add")
    design = design.reindex(sorted(design.columns), axis=1)

    if design_columns is not None:
        design = design.reindex(list(design_columns), axis=1, fill_value=0.0)
    return design


def _interval_to_cumulative_pd(hazard: pd.Series, horizon_quarters: pd.Series) -> pd.Series:
    capped_hazard = hazard.clip(lower=1e-6, upper=0.80)
    capped_horizon = horizon_quarters.clip(lower=1).astype(int)
    return 1.0 - np.power(1.0 - capped_hazard, capped_horizon)


def fit_survival_pd_model(
    panel: pd.DataFrame,
    feature_spec: FeatureSpec | None = None,
) -> SurvivalPDModel:
    """Fit a pooled-logit discrete-time survival model on quarterly panel data.

    Summary
    -------
    Estimate a point-in-time quarterly default hazard from account-level panel
    observations and return a fitted model object with reusable design metadata.

    Method
    ------
    The function follows the discrete-time hazard approach described by Singer
    and Willett (1993): each panel row is treated as one interval in which the
    event indicator equals one if default occurs before the next observation.
    A pooled logistic GLM estimates the conditional interval hazard as a
    function of borrower, loan, delinquency, and macro covariates.

    Parameters
    ----------
    panel:
        Performance panel containing model features and the
        `default_next_period` target.
    feature_spec:
        Optional explicit feature specification. If omitted, the package default
        specification is used.

    Returns
    -------
    SurvivalPDModel
        Dataclass containing the fitted `statsmodels` result object, the feature
        specification, the design columns, and a coefficient summary table.

    Raises
    ------
    KeyError
        Raised if the requested target column or one of the feature columns is
        missing from the panel.
    ValueError
        Raised if the target column has fewer than two classes or the panel is
        empty after filtering.

    Notes
    -----
    The model is intentionally simple and unregularised. That makes coefficient
    signs easy to inspect in the baseline credit-risk workflow.

    Edge Cases
    ----------
    A panel with only defaults or only non-defaults is rejected because the
    logistic GLM cannot learn a meaningful hazard under complete separation in
    this simplified setup.

    References
    ----------
    - Singer, J. D., and Willett, J. B. (1993), "It's About Time: Using
      Discrete-Time Survival Analysis to Study Duration and the Timing of
      Events." https://journals.sagepub.com/doi/10.3102/10769986018002155
    """

    feature_spec = _resolve_feature_spec(feature_spec)
    if feature_spec.target_column not in panel.columns:
        raise KeyError(f"Panel is missing target column '{feature_spec.target_column}'.")

    fit_panel = panel.copy()
    fit_panel = fit_panel.loc[fit_panel["remaining_term_quarters"] > 0].reset_index(drop=True)
    if fit_panel.empty:
        raise ValueError("Panel is empty after filtering active observations.")

    target = fit_panel[feature_spec.target_column].astype(int)
    if target.nunique() < 2:
        raise ValueError("Target column must contain both default and non-default observations.")

    design = _prepare_design_matrix(fit_panel, feature_spec)
    result = sm.GLM(target, design, family=sm.families.Binomial()).fit()

    coefficient_table = pd.DataFrame(
        {
            "feature": result.params.index,
            "coefficient": result.params.values,
            "std_error": result.bse.values,
            "z_score": result.tvalues.values,
        }
    )
    return SurvivalPDModel(
        result=result,
        feature_spec=feature_spec,
        design_columns=list(design.columns),
        coefficient_table=coefficient_table,
    )


def score_portfolio(
    dataset: PortfolioDataset,
    model: SurvivalPDModel,
    as_of_date: str | pd.Timestamp,
) -> PDScoreFrame:
    """Score one reporting snapshot and expose current, previous, and history views.

    Summary
    -------
    Apply the fitted survival model to the portfolio history and return a score
    object tailored to IFRS 9, validation, and monitoring use cases.

    Method
    ------
    The function predicts a one-quarter hazard for each active panel row and
    converts that interval hazard into a 12-month PD and a simplified lifetime
    PD by assuming the current hazard persists over future quarters. It then
    extracts both the requested reporting date and the immediately previous
    snapshot so downstream modules can compute stage migration, roll-forwards,
    and drift comparisons.

    Parameters
    ----------
    dataset:
        Synthetic portfolio dataset returned by
        :func:`credit_risk_lab.portfolio.generate_portfolio_timeseries`.
    model:
        Fitted survival model created by
        :func:`credit_risk_lab.survival.fit_survival_pd_model`.
    as_of_date:
        Reporting date to score. Any `pandas.Timestamp`-compatible value is
        accepted.

    Returns
    -------
    PDScoreFrame
        Dataclass containing the scored current snapshot, previous snapshot, and
        the full scored history.

    Raises
    ------
    ValueError
        Raised if the requested reporting date does not exist in the performance
        panel.

    Notes
    -----
    Observed 12-month and lifetime default indicators are derived from the
    synthetic defaults table so the validation pack can backtest the scores
    without fitting another model.

    Edge Cases
    ----------
    If the requested reporting date is the first available snapshot, the
    `previous_snapshot_scores` frame is returned empty rather than raising an
    exception.

    References
    ----------
    - Singer, J. D., and Willett, J. B. (1993), "It's About Time: Using
      Discrete-Time Survival Analysis to Study Duration and the Timing of
      Events." https://journals.sagepub.com/doi/10.3102/10769986018002155
    """

    as_of_timestamp = pd.Timestamp(as_of_date)
    performance = dataset.performance.copy()
    available_dates = sorted(performance["snapshot_date"].unique())
    if as_of_timestamp not in available_dates:
        raise ValueError(f"Reporting date {as_of_timestamp.date()} is not available in the performance panel.")

    design = _prepare_design_matrix(performance, model.feature_spec, model.design_columns)
    hazard = pd.Series(model.result.predict(design), index=performance.index, name="quarterly_hazard").clip(lower=1e-6, upper=0.80)
    scored_history = performance.copy()
    scored_history["quarterly_hazard"] = hazard
    scored_history["pd_12m"] = _interval_to_cumulative_pd(hazard, scored_history["remaining_term_quarters"].clip(upper=4))
    scored_history["pd_lifetime"] = _interval_to_cumulative_pd(hazard, scored_history["remaining_term_quarters"])
    scored_history["pd_band"] = pd.cut(
        scored_history["pd_12m"],
        bins=[-np.inf, 0.01, 0.03, 0.07, 0.15, np.inf],
        labels=["A", "B", "C", "D", "E"],
    ).astype(str)

    default_lookup = dataset.defaults.set_index("loan_id")["default_date"] if not dataset.defaults.empty else pd.Series(dtype="datetime64[ns]")
    scored_history["default_date"] = scored_history["loan_id"].map(default_lookup).astype("datetime64[ns]")
    scored_history["observed_default_12m"] = (
        scored_history["default_date"].notna()
        & (scored_history["default_date"] > scored_history["snapshot_date"])
        & (scored_history["default_date"] <= scored_history["snapshot_date"] + pd.offsets.QuarterEnd(4))
    ).astype(int)
    scored_history["observed_default_lifetime"] = scored_history.apply(
        lambda row: int(
            pd.notna(row["default_date"])
            and row["default_date"] > row["snapshot_date"]
            and row["default_date"] <= row["snapshot_date"] + pd.offsets.QuarterEnd(int(row["remaining_term_quarters"]))
        ),
        axis=1,
    )

    snapshot_scores = scored_history.loc[scored_history["snapshot_date"] == as_of_timestamp].reset_index(drop=True)
    prior_dates = [timestamp for timestamp in available_dates if timestamp < as_of_timestamp]
    previous_snapshot_scores = (
        scored_history.loc[scored_history["snapshot_date"] == prior_dates[-1]].reset_index(drop=True)
        if prior_dates
        else scored_history.iloc[0:0].copy()
    )
    return PDScoreFrame(
        as_of_date=as_of_timestamp,
        snapshot_scores=snapshot_scores,
        previous_snapshot_scores=previous_snapshot_scores,
        scored_history=scored_history.reset_index(drop=True),
    )
