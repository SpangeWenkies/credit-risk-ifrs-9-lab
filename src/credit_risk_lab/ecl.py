"""IFRS 9 staging and expected credit loss calculations.

This module translates point-in-time PD outputs into a simplified IFRS 9
impairment workflow. It applies stage 1 / 2 / 3 logic, scenario-weighted
expected credit loss calculations, stage migration summaries, and a basic
provision roll-forward that mirrors the kind of artefacts a junior credit-risk
or actuarial candidate may be asked to explain.

Assumptions
-----------
- Stage allocation uses delinquency, forbearance, and deterioration versus
  origination as the main signals. This is a simplified interpretation of IFRS
  9 significant increase in credit risk logic.
- Scenario effects are applied as transparent multiplicative overlays to the
  quarterly hazard, collateral recovery, and conversion-factor assumptions.
- The current implementation treats stage 3 as high-delinquency/default-like
  exposure and assigns a PD of one in the simplified ECL formula.

Primary references
------------------
- IFRS Foundation, "IFRS 9 Financial Instruments."
  https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf

Simplifications for this portfolio project
------------------------------------------
- The scenario engine is illustrative rather than an econometric satellite
  model.
- Lifetime PD is derived from the current quarterly hazard instead of a full
  forward term structure.
- Provision roll-forward components are designed to reconcile mechanically and
  explain intuition, not replicate accounting general-ledger entries.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .models import ECLReport, MacroScenario, PDScoreFrame, PortfolioDataset

DEFAULT_EAD_MODEL = {
    "mortgage": 0.00,
    "auto": 0.00,
    "personal_loan": 0.00,
    "credit_card": 0.75,
}
DEFAULT_LGD_MODEL = {
    "mortgage": {"floor": 0.12, "collateral_haircut": 0.18},
    "auto": {"floor": 0.22, "collateral_haircut": 0.25},
    "personal_loan": {"floor": 0.58, "collateral_haircut": 1.00},
    "credit_card": {"floor": 0.85, "collateral_haircut": 1.00},
}


def default_macro_scenarios() -> list[MacroScenario]:
    """Return the default baseline/downside/upside macro scenario set.

    Summary
    -------
    Provide a compact scenario set that can be reused in examples, tests, and
    recruiter-facing reports without external data dependencies.

    Method
    ------
    The scenario set applies small, interpretable shifts to unemployment, house
    prices, and policy rates. These shifts are later translated into PD, LGD,
    and EAD overlays inside the IFRS 9 engine.

    Parameters
    ----------
    None
        This function does not take runtime parameters.

    Returns
    -------
    list[MacroScenario]
        Baseline, downside, and upside scenarios with weights summing to one.

    Raises
    ------
    None
        This function does not raise custom exceptions.

    Notes
    -----
    The scenario set is intentionally small because the project prioritises
    explainability and portability over a rich macro-satellite framework.

    Edge Cases
    ----------
    Weights are hard-coded to sum to one; downstream functions do not renormalise
    them.

    References
    ----------
    - IFRS Foundation, "IFRS 9 Financial Instruments."
      https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf
    """

    return [
        MacroScenario("baseline", unemployment_shift=0.00, house_price_shift=0.00, policy_rate_shift=0.00, weight=0.60),
        MacroScenario("downside", unemployment_shift=0.020, house_price_shift=-0.10, policy_rate_shift=0.012, weight=0.30),
        MacroScenario("upside", unemployment_shift=-0.008, house_price_shift=0.04, policy_rate_shift=-0.008, weight=0.10),
    ]


def _classify_stage(row: pd.Series) -> int:
    """Assign a simplified IFRS 9 stage to one scored exposure.

    Summary
    -------
    Map one scored loan snapshot into stage 1, 2, or 3.

    Method
    ------
    The function treats 90+ DPD as stage 3, then flags stage 2 on 30+ DPD,
    forbearance, rating deterioration versus origination, or material PD
    deterioration versus the origination anchor. Otherwise the exposure remains
    in stage 1.

    Parameters
    ----------
    row:
        Loan-level scored snapshot row.

    Returns
    -------
    int
        IFRS 9 stage code.

    Raises
    ------
    None
        This helper does not raise custom exceptions.

    Notes
    -----
    The stage logic is intentionally compact and interpretable; it should be
    read as a teaching approximation rather than a bank policy document.

    Edge Cases
    ----------
    Multiple stage 2 triggers are treated identically; the function returns the
    highest applicable stage only.

    References
    ----------
    - IFRS Foundation, "IFRS 9 Financial Instruments."
      https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf
    """

    if row["days_past_due"] >= 90:
        return 3
    if row["days_past_due"] >= 30:
        return 2
    if int(row["forborne"]) == 1:
        return 2
    if int(row["rating_rank"] - row["origination_rating_rank"]) >= 2:
        return 2
    if float(row["pd_12m"]) / max(float(row["origination_pd_anchor"]), 1e-4) >= 2.0:
        return 2
    if float(row["pd_lifetime"]) / max(float(row["origination_pd_anchor"]), 1e-4) >= 3.0:
        return 2
    return 1


def _scenario_multiplier(scenario: MacroScenario) -> float:
    """Translate macro scenario shifts into a quarterly hazard multiplier.

    Summary
    -------
    Convert scenario-level macro shifts into a multiplicative overlay on the
    base quarterly hazard.

    Method
    ------
    The function exponentiates a linear combination of unemployment, house-
    price, and policy-rate shifts, then clips the result to a conservative
    interval so stressed scenarios remain numerically stable.

    Parameters
    ----------
    scenario:
        Macro scenario to translate.

    Returns
    -------
    float
        Hazard multiplier used inside the ECL engine.

    Raises
    ------
    None
        This helper does not raise custom exceptions.

    Notes
    -----
    The weights are judgemental overlays rather than econometrically estimated
    satellite coefficients.

    Edge Cases
    ----------
    Very severe scenario inputs are clipped to avoid explosive hazards.

    References
    ----------
    - IFRS Foundation, "IFRS 9 Financial Instruments."
      https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf
    """

    return float(
        np.clip(
            np.exp(
                7.0 * scenario.unemployment_shift
                - 2.5 * scenario.house_price_shift
                + 4.0 * scenario.policy_rate_shift
            ),
            0.60,
            3.50,
        )
    )


def _scenario_lgd(row: pd.Series, scenario: MacroScenario, lgd_model: dict[str, dict[str, float]]) -> float:
    """Apply a scenario-specific LGD overlay to one exposure.

    Summary
    -------
    Convert collateral and macro information into a stressed loss-given-default
    estimate for one exposure.

    Method
    ------
    The function shocks collateral values by the scenario house-price shift,
    applies a segment-level haircut, derives the recoverable share of exposure,
    and enforces a segment floor. Unemployment stress then scales the result
    upward.

    Parameters
    ----------
    row:
        Loan-level scored snapshot row.
    scenario:
        Macro scenario used for the stress overlay.
    lgd_model:
        Segment-level LGD settings containing floors and collateral haircuts.

    Returns
    -------
    float
        Scenario-specific LGD estimate.

    Raises
    ------
    KeyError
        Raised if the segment is missing from the LGD settings.

    Notes
    -----
    This is intentionally an explainable downturn-LGD style overlay rather than
    a cashflow recovery model.

    Edge Cases
    ----------
    Unsecured segments with zero collateral fall back immediately to the segment
    floor logic.

    References
    ----------
    - IFRS Foundation, "IFRS 9 Financial Instruments."
      https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf
    """

    settings = lgd_model[str(row["segment"])]
    stressed_collateral = max(float(row["collateral_value"]) * (1.0 + scenario.house_price_shift), 0.0)
    haircut = settings["collateral_haircut"]
    recoverable = 0.70 * stressed_collateral * max(1.0 - haircut, 0.0) / max(float(row["balance"]), 1.0)
    lgd = max(1.0 - recoverable, settings["floor"])
    lgd *= 1.0 + 1.8 * max(scenario.unemployment_shift, 0.0)
    return float(np.clip(lgd, 0.05, 0.99))


def _scenario_ead(row: pd.Series, scenario: MacroScenario, ead_model: dict[str, float]) -> float:
    """Apply a scenario-specific EAD overlay to one exposure.

    Summary
    -------
    Convert current balance and undrawn commitment into a stressed exposure at
    default.

    Method
    ------
    The function applies a segment-level credit conversion factor to the
    undrawn balance and increases that factor modestly under worse unemployment
    or policy-rate shifts.

    Parameters
    ----------
    row:
        Loan-level scored snapshot row.
    scenario:
        Macro scenario used for the stress overlay.
    ead_model:
        Segment-level base credit conversion factors.

    Returns
    -------
    float
        Scenario-specific EAD estimate.

    Raises
    ------
    KeyError
        Raised if the segment is missing from the EAD settings.

    Notes
    -----
    The overlay is most relevant for revolving exposures such as credit cards.

    Edge Cases
    ----------
    For fully amortising products with no undrawn amount, stressed EAD equals
    current balance.

    References
    ----------
    - IFRS Foundation, "IFRS 9 Financial Instruments."
      https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf
    """

    ccf = ead_model[str(row["segment"])] + 0.10 * max(scenario.unemployment_shift, 0.0) + 0.05 * max(scenario.policy_rate_shift, 0.0)
    ccf = float(np.clip(ccf, 0.0, 1.0))
    return float(row["balance"] + ccf * row["undrawn_amount"])


def _score_snapshot_for_ecl(
    snapshot_scores: pd.DataFrame,
    scenarios: Sequence[MacroScenario],
    lgd_model: dict[str, dict[str, float]],
    ead_model: dict[str, float],
) -> pd.DataFrame:
    """Calculate stage, PD/LGD/EAD overlays, and weighted ECL for one snapshot.

    Summary
    -------
    Transform one scored snapshot into a loan-level ECL table under multiple
    macro scenarios.

    Method
    ------
    The function assigns stages, converts the quarterly hazard into the relevant
    stage horizon, applies scenario-specific LGD and EAD overlays, discounts the
    expected loss cashflow, and aggregates the scenario-weighted ECL.

    Parameters
    ----------
    snapshot_scores:
        Loan-level score frame for one reporting date.
    scenarios:
        Macro scenarios with weights.
    lgd_model:
        Segment-level LGD settings.
    ead_model:
        Segment-level EAD conversion settings.

    Returns
    -------
    pandas.DataFrame
        Loan-level ECL table with scenario-specific and weighted outputs.

    Raises
    ------
    None
        Empty snapshots return an empty frame.

    Notes
    -----
    Each scenario column is kept on the output frame so the notebook and report
    artefacts can show exactly where the weighted ECL comes from.

    Edge Cases
    ----------
    Stage 3 exposures are forced to a PD of one and a slightly more conservative
    LGD overlay.

    References
    ----------
    - IFRS Foundation, "IFRS 9 Financial Instruments."
      https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf
    """

    if snapshot_scores.empty:
        return snapshot_scores.copy()

    result = snapshot_scores.copy()
    result["stage"] = result.apply(_classify_stage, axis=1)
    result["ead_baseline"] = result["balance"]
    result["weighted_ecl"] = 0.0
    result["weighted_lgd"] = 0.0

    for scenario in scenarios:
        hazard_multiplier = _scenario_multiplier(scenario)
        scenario_hazard = (result["quarterly_hazard"] * hazard_multiplier).clip(lower=1e-6, upper=0.90)
        stage_horizon = np.where(result["stage"].eq(1), np.minimum(result["remaining_term_quarters"], 4), result["remaining_term_quarters"])
        scenario_pd = 1.0 - np.power(1.0 - scenario_hazard, stage_horizon.astype(int))
        scenario_pd = np.where(result["stage"].eq(3), 1.0, scenario_pd)
        scenario_lgd = result.apply(lambda row: _scenario_lgd(row, scenario, lgd_model), axis=1)
        scenario_lgd = np.where(result["stage"].eq(3), np.clip(scenario_lgd + 0.08, 0.05, 0.99), scenario_lgd)
        scenario_ead = result.apply(lambda row: _scenario_ead(row, scenario, ead_model), axis=1)
        discount = 1.0 / np.power(
            1.0 + np.clip(result["interest_rate"] + scenario.policy_rate_shift, 0.0, None) / 4.0,
            np.maximum(stage_horizon.astype(int), 1),
        )
        scenario_ecl = scenario.weight * scenario_pd * scenario_lgd * scenario_ead * discount

        result[f"{scenario.name}_pd"] = np.round(scenario_pd, 6)
        result[f"{scenario.name}_lgd"] = np.round(scenario_lgd, 6)
        result[f"{scenario.name}_ead"] = np.round(scenario_ead, 2)
        result[f"{scenario.name}_ecl"] = np.round(scenario_ecl, 2)
        result["weighted_ecl"] += scenario_ecl
        result["weighted_lgd"] += scenario.weight * scenario_lgd

    result["weighted_ecl"] = result["weighted_ecl"].round(2)
    result["weighted_lgd"] = result["weighted_lgd"].round(6)
    return result


def _build_roll_forward(
    current_results: pd.DataFrame,
    previous_results: pd.DataFrame,
    defaults: pd.DataFrame,
    previous_date: pd.Timestamp | None,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    """Reconcile opening and closing allowance into intuitive components.

    Summary
    -------
    Build a simple provision roll-forward between two reporting dates.

    Method
    ------
    The function partitions the ECL change into opening allowance, new
    originations, maturities/prepayments, write-offs, stage migration, and
    remeasurement for continuing exposures, then returns the closing allowance.

    Parameters
    ----------
    current_results:
        Current loan-level ECL results.
    previous_results:
        Previous loan-level ECL results.
    defaults:
        Defaults table used to distinguish write-offs from non-default exits.
    previous_date:
        Previous reporting date, if available.
    as_of_date:
        Current reporting date.

    Returns
    -------
    pandas.DataFrame
        Roll-forward table with one row per component.

    Raises
    ------
    None
        The helper returns a new-origination style roll-forward when no opening
        snapshot exists.

    Notes
    -----
    This roll-forward is designed to reconcile numerically and be easy to
    explain. It is not a substitute for accounting-ledger reporting.

    Edge Cases
    ----------
    If no previous snapshot exists, the closing allowance is treated as entirely
    new origination allowance.

    References
    ----------
    - IFRS Foundation, "IFRS 9 Financial Instruments."
      https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf
    """

    current_total = float(current_results["weighted_ecl"].sum()) if not current_results.empty else 0.0
    previous_total = float(previous_results["weighted_ecl"].sum()) if not previous_results.empty else 0.0
    if previous_results.empty or previous_date is None:
        return pd.DataFrame(
            [
                {"component": "opening_allowance", "amount": 0.0},
                {"component": "new_originations", "amount": round(current_total, 2)},
                {"component": "maturities_and_prepayments", "amount": 0.0},
                {"component": "write_offs", "amount": 0.0},
                {"component": "stage_migration", "amount": 0.0},
                {"component": "remeasurement", "amount": 0.0},
                {"component": "closing_allowance", "amount": round(current_total, 2)},
            ]
        )

    merged = previous_results[["loan_id", "stage", "weighted_ecl"]].merge(
        current_results[["loan_id", "stage", "weighted_ecl"]],
        on="loan_id",
        how="outer",
        suffixes=("_prev", "_cur"),
        indicator=True,
    )
    new_originations = float(merged.loc[merged["_merge"].eq("right_only"), "weighted_ecl_cur"].sum())
    exited_ids = merged.loc[merged["_merge"].eq("left_only"), "loan_id"]
    defaulted_ids = set(
        defaults.loc[
            (defaults["default_date"] > previous_date) & (defaults["default_date"] <= as_of_date),
            "loan_id",
        ].tolist()
    )
    write_offs = float(merged.loc[merged["loan_id"].isin(defaulted_ids), "weighted_ecl_prev"].sum()) * -1.0
    maturities_and_prepayments = float(
        merged.loc[exited_ids.index.difference(merged.loc[merged["loan_id"].isin(defaulted_ids)].index), "weighted_ecl_prev"].sum()
    ) * -1.0
    overlap = merged.loc[merged["_merge"].eq("both")].copy()
    overlap["delta"] = overlap["weighted_ecl_cur"] - overlap["weighted_ecl_prev"]
    stage_migration = float(overlap.loc[overlap["stage_prev"] != overlap["stage_cur"], "delta"].sum())
    remeasurement = float(overlap.loc[overlap["stage_prev"] == overlap["stage_cur"], "delta"].sum())

    roll_forward = pd.DataFrame(
        [
            {"component": "opening_allowance", "amount": round(previous_total, 2)},
            {"component": "new_originations", "amount": round(new_originations, 2)},
            {"component": "maturities_and_prepayments", "amount": round(maturities_and_prepayments, 2)},
            {"component": "write_offs", "amount": round(write_offs, 2)},
            {"component": "stage_migration", "amount": round(stage_migration, 2)},
            {"component": "remeasurement", "amount": round(remeasurement, 2)},
            {"component": "closing_allowance", "amount": round(current_total, 2)},
        ]
    )
    return roll_forward


def run_ifrs9_pipeline(
    dataset: PortfolioDataset,
    pd_scores: PDScoreFrame,
    lgd_model: dict[str, dict[str, float]] | None = None,
    ead_model: dict[str, float] | None = None,
    scenarios: Sequence[MacroScenario] | None = None,
) -> ECLReport:
    """Run simplified IFRS 9 staging, scenario ECL, and governance summaries.

    Summary
    -------
    Convert portfolio PD scores into a reporting-date IFRS 9 ECL report with
    stage classifications, scenario-weighted loan-level ECLs, stage migration,
    and a reconciled provision roll-forward.

    Method
    ------
    The pipeline first classifies exposures into stage 1, 2, or 3 using a
    simplified significant increase in credit risk logic inspired by IFRS 9.
    It then applies scenario overlays to the quarterly hazard, LGD, and EAD
    components and aggregates the weighted scenario losses. Finally, it compares
    the current and previous snapshots to produce stage migration and roll-
    forward artefacts.

    Parameters
    ----------
    dataset:
        Root synthetic dataset used to retrieve default timing for the
        roll-forward.
    pd_scores:
        Score object returned by :func:`credit_risk_lab.survival.score_portfolio`.
    lgd_model:
        Optional segment-level LGD settings. When omitted, transparent defaults
        are used.
    ead_model:
        Optional segment-level credit-conversion factors. When omitted,
        transparent defaults are used.
    scenarios:
        Optional macro scenario collection. When omitted, the default baseline /
        downside / upside set is used.

    Returns
    -------
    ECLReport
        Reporting-date IFRS 9 artefact bundle including loan-level results,
        stage and scenario summaries, migration matrix, and provision roll-
        forward.

    Raises
    ------
    ValueError
        Raised if the scenario weights do not sum approximately to one.

    Notes
    -----
    The stage allocation and macro overlays are intentionally interpretable. The
    goal is to communicate modelling judgment clearly, not to mimic every
    institution-specific impairment policy.

    Edge Cases
    ----------
    If the current reporting date has no prior snapshot, the migration matrix is
    returned empty and the roll-forward treats the entire balance as new
    origination allowance.

    References
    ----------
    - IFRS Foundation, "IFRS 9 Financial Instruments."
      https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf
    """

    lgd_model = lgd_model or DEFAULT_LGD_MODEL
    ead_model = ead_model or DEFAULT_EAD_MODEL
    scenarios = list(scenarios or default_macro_scenarios())
    if not np.isclose(sum(scenario.weight for scenario in scenarios), 1.0, atol=1e-6):
        raise ValueError("Scenario weights must sum to 1.0.")

    current_results = _score_snapshot_for_ecl(pd_scores.snapshot_scores, scenarios, lgd_model, ead_model)
    previous_results = _score_snapshot_for_ecl(pd_scores.previous_snapshot_scores, scenarios, lgd_model, ead_model)

    stage_summary = (
        current_results.groupby("stage", as_index=False)
        .agg(
            loan_count=("loan_id", "nunique"),
            exposure=("balance", "sum"),
            mean_pd_12m=("pd_12m", "mean"),
            mean_lgd=("weighted_lgd", "mean"),
            allowance=("weighted_ecl", "sum"),
        )
        .round({"exposure": 2, "mean_pd_12m": 6, "mean_lgd": 6, "allowance": 2})
    )

    scenario_rows = []
    for scenario in scenarios:
        scenario_rows.append(
            {
                "scenario": scenario.name,
                "weight": scenario.weight,
                "total_ecl": round(float(current_results[f"{scenario.name}_ecl"].sum()), 2),
            }
        )
    scenario_summary = pd.DataFrame(scenario_rows)

    if previous_results.empty:
        migration_matrix = pd.DataFrame()
    else:
        migration_matrix = (
            previous_results[["loan_id", "stage"]]
            .merge(current_results[["loan_id", "stage"]], on="loan_id", how="inner", suffixes=("_previous", "_current"))
            .pivot_table(
                index="stage_previous",
                columns="stage_current",
                values="loan_id",
                aggfunc="count",
                fill_value=0,
            )
            .sort_index(axis=0)
            .sort_index(axis=1)
        )

    previous_date = None if pd_scores.previous_snapshot_scores.empty else pd.Timestamp(pd_scores.previous_snapshot_scores["snapshot_date"].iloc[0])
    provision_roll_forward = _build_roll_forward(
        current_results=current_results,
        previous_results=previous_results,
        defaults=dataset.defaults,
        previous_date=previous_date,
        as_of_date=pd_scores.as_of_date,
    )

    return ECLReport(
        as_of_date=pd_scores.as_of_date,
        loan_results=current_results,
        stage_summary=stage_summary,
        scenario_summary=scenario_summary,
        migration_matrix=migration_matrix,
        provision_roll_forward=provision_roll_forward,
    )
