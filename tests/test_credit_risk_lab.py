from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pandas.testing as pdt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_lab import (  # noqa: E402
    PDScoreFrame,
    PortfolioConfig,
    PortfolioDataset,
    build_portfolio_report,
    fit_survival_pd_model,
    generate_portfolio_timeseries,
    run_ifrs9_pipeline,
    run_monitoring,
    score_portfolio,
)
from credit_risk_lab.models import ECLReport  # noqa: E402
from credit_risk_lab.survival import _prepare_design_matrix  # noqa: E402


def test_portfolio_generation_is_reproducible_and_has_expected_tables() -> None:
    config = PortfolioConfig(random_seed=7, periods=10, num_borrowers=50, num_loans=80)
    first = generate_portfolio_timeseries(config)
    second = generate_portfolio_timeseries(config)

    expected_tables = ("borrowers", "loans", "performance", "defaults", "recoveries", "macro", "snapshots")
    for table_name in expected_tables:
        assert hasattr(first, table_name)
    pdt.assert_frame_equal(first.borrowers, second.borrowers)
    pdt.assert_frame_equal(first.loans, second.loans)
    pdt.assert_frame_equal(first.performance, second.performance)
    assert {"loan_id", "snapshot_date", "default_next_period", "ltv", "dti"}.issubset(first.performance.columns)


def test_portfolio_transitions_are_quarterly_and_defaults_stop_active_rows() -> None:
    dataset = generate_portfolio_timeseries(PortfolioConfig(random_seed=13, periods=11, num_borrowers=60, num_loans=90))
    grouped = dataset.performance.groupby("loan_id")
    for _, history in grouped:
        dates = list(history["snapshot_date"])
        assert dates == sorted(dates)
        remaining = history["remaining_term_quarters"].tolist()
        assert remaining == sorted(remaining, reverse=True)

    if not dataset.defaults.empty:
        merged = dataset.performance.merge(dataset.defaults[["loan_id", "default_date"]], on="loan_id", how="left")
        active_after_default = merged.loc[merged["default_date"].notna() & (merged["snapshot_date"] >= merged["default_date"])]
        assert active_after_default.empty


def test_survival_scores_are_ordered_for_better_and_worse_risk_profiles() -> None:
    dataset = generate_portfolio_timeseries(PortfolioConfig(random_seed=5, periods=12, num_borrowers=120, num_loans=220))
    model = fit_survival_pd_model(dataset.performance)
    as_of = dataset.performance["snapshot_date"].max()
    scores = score_portfolio(dataset, model, as_of)

    assert (scores.snapshot_scores["pd_lifetime"] >= scores.snapshot_scores["pd_12m"]).all()

    base_row = scores.snapshot_scores.iloc[[0]].copy()
    worse_row = base_row.copy()
    worse_row["rating_rank"] = base_row["rating_rank"] + 2
    worse_row["days_past_due"] = base_row["days_past_due"] + 60
    worse_row["ltv"] = base_row["ltv"] + 0.20
    worse_row["dti"] = base_row["dti"] + 0.15

    base_design = _prepare_design_matrix(base_row, model.feature_spec, model.design_columns)
    worse_design = _prepare_design_matrix(worse_row, model.feature_spec, model.design_columns)
    base_hazard = float(model.result.predict(base_design).iloc[0])
    worse_hazard = float(model.result.predict(worse_design).iloc[0])
    assert worse_hazard > base_hazard


def test_ifrs9_pipeline_assigns_stages_and_roll_forward_reconciles() -> None:
    snapshot_date = pd.Timestamp("2024-12-31")
    previous_date = pd.Timestamp("2024-09-30")
    current_snapshot = pd.DataFrame(
        [
            {
                "loan_id": "L1",
                "segment": "mortgage",
                "snapshot_date": snapshot_date,
                "balance": 100_000.0,
                "undrawn_amount": 0.0,
                "quarterly_hazard": 0.01,
                "remaining_term_quarters": 16,
                "interest_rate": 0.03,
                "collateral_value": 150_000.0,
                "days_past_due": 0,
                "forborne": 0,
                "rating_rank": 2,
                "origination_rating_rank": 2,
                "pd_12m": 0.03,
                "pd_lifetime": 0.08,
                "origination_pd_anchor": 0.04,
            },
            {
                "loan_id": "L2",
                "segment": "auto",
                "snapshot_date": snapshot_date,
                "balance": 20_000.0,
                "undrawn_amount": 0.0,
                "quarterly_hazard": 0.05,
                "remaining_term_quarters": 8,
                "interest_rate": 0.05,
                "collateral_value": 18_000.0,
                "days_past_due": 35,
                "forborne": 0,
                "rating_rank": 4,
                "origination_rating_rank": 2,
                "pd_12m": 0.20,
                "pd_lifetime": 0.35,
                "origination_pd_anchor": 0.04,
            },
            {
                "loan_id": "L3",
                "segment": "personal_loan",
                "snapshot_date": snapshot_date,
                "balance": 8_000.0,
                "undrawn_amount": 0.0,
                "quarterly_hazard": 0.15,
                "remaining_term_quarters": 4,
                "interest_rate": 0.07,
                "collateral_value": 0.0,
                "days_past_due": 95,
                "forborne": 1,
                "rating_rank": 5,
                "origination_rating_rank": 3,
                "pd_12m": 0.55,
                "pd_lifetime": 0.70,
                "origination_pd_anchor": 0.08,
            },
        ]
    )
    previous_snapshot = current_snapshot.copy()
    previous_snapshot["snapshot_date"] = previous_date
    previous_snapshot["pd_12m"] = previous_snapshot["pd_12m"] * 0.85
    previous_snapshot["pd_lifetime"] = previous_snapshot["pd_lifetime"] * 0.85
    previous_snapshot.loc[previous_snapshot["loan_id"] == "L2", "days_past_due"] = 5
    previous_snapshot.loc[previous_snapshot["loan_id"] == "L2", "rating_rank"] = 2
    previous_snapshot.loc[previous_snapshot["loan_id"] == "L2", "forborne"] = 0
    previous_snapshot.loc[previous_snapshot["loan_id"] == "L3", "days_past_due"] = 65

    pd_scores = PDScoreFrame(
        as_of_date=snapshot_date,
        snapshot_scores=current_snapshot,
        previous_snapshot_scores=previous_snapshot,
        scored_history=pd.concat([previous_snapshot, current_snapshot], ignore_index=True),
    )
    dataset = PortfolioDataset(
        borrowers=pd.DataFrame(),
        loans=pd.DataFrame(),
        performance=pd.DataFrame(),
        defaults=pd.DataFrame(columns=["loan_id", "default_date"]),
        recoveries=pd.DataFrame(),
        macro=pd.DataFrame(),
        snapshots=pd.DataFrame(),
    )

    report = run_ifrs9_pipeline(dataset, pd_scores)
    stages = report.loan_results.set_index("loan_id")["stage"].to_dict()
    assert stages == {"L1": 1, "L2": 2, "L3": 3}
    assert (report.loan_results["weighted_ecl"] >= 0.0).all()

    roll_forward = report.provision_roll_forward.set_index("component")["amount"]
    reconciled = (
        roll_forward["opening_allowance"]
        + roll_forward["new_originations"]
        + roll_forward["maturities_and_prepayments"]
        + roll_forward["write_offs"]
        + roll_forward["stage_migration"]
        + roll_forward["remeasurement"]
    )
    assert round(reconciled, 2) == round(roll_forward["closing_allowance"], 2)


def test_monitoring_flags_shifted_scores_and_report_builder_returns_markdown() -> None:
    dataset = generate_portfolio_timeseries(PortfolioConfig(random_seed=21, periods=12, num_borrowers=80, num_loans=140))
    model = fit_survival_pd_model(dataset.performance)
    as_of = dataset.performance["snapshot_date"].max()
    scores = score_portfolio(dataset, model, as_of)
    monitoring = run_monitoring(
        reference_snapshot=scores.previous_snapshot_scores,
        current_snapshot=scores.snapshot_scores,
        reference_scores=scores.previous_snapshot_scores,
        current_scores=scores.snapshot_scores,
    )
    assert not monitoring.feature_drift.empty
    assert not monitoring.score_drift.empty

    ecl_report = run_ifrs9_pipeline(dataset, scores)
    markdown = build_portfolio_report(dataset, scores, ecl_report, monitoring)
    assert markdown.startswith("# Credit Risk & IFRS 9 Summary")
    assert "Provision Roll-Forward" in markdown
