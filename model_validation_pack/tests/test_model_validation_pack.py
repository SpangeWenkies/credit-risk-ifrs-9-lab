from __future__ import annotations

import builtins
import sys
from math import exp
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ROOT_SRC = ROOT / "src"
VALIDATION_SRC = ROOT / "model_validation_pack" / "src"
for path in (ROOT_SRC, VALIDATION_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from credit_risk_lab import (  # noqa: E402
    PortfolioConfig,
    generate_portfolio_timeseries,
    fit_survival_pd_model,
    score_challenger_model,
    score_portfolio,
)
from model_validation_pack import (  # noqa: E402
    build_credit_validation_bundle,
    run_backtest,
    run_drift_tests,
    run_validation_pack,
)


def sensitivity_score(features: dict[str, float]) -> float:
    linear = -5.0
    linear += 0.55 * features.get("rating_rank", 2.0)
    linear += 0.020 * features.get("days_past_due", 0.0)
    linear += 1.20 * max(features.get("ltv", 0.8) - 0.85, 0.0)
    linear += 1.10 * max(features.get("dti", 0.35) - 0.35, 0.0)
    linear += 0.65 * max(features.get("utilization", 0.60) - 0.70, 0.0)
    return 1.0 / (1.0 + exp(-linear))


def _build_bundle():
    dataset = generate_portfolio_timeseries(PortfolioConfig(random_seed=17, periods=12, num_borrowers=100, num_loans=180))
    model = fit_survival_pd_model(dataset.performance)
    as_of = dataset.performance["snapshot_date"].max()
    scores = score_portfolio(dataset, model, as_of)
    challenger = score_challenger_model(scores.snapshot_scores).tolist()
    bundle = build_credit_validation_bundle(scores.snapshot_scores, scores.previous_snapshot_scores, challenger)
    return bundle


def test_credit_adapter_builds_observations_and_validates_benchmark_alignment() -> None:
    bundle = _build_bundle()
    assert bundle.observations
    assert bundle.reference_observations
    assert len(bundle.observations) == len(bundle.benchmark_scores or [])
    assert bundle.observations[0].features


def test_backtest_returns_brier_score_and_rank_ordering() -> None:
    bundle = _build_bundle()
    result = run_backtest(bundle.observations)
    assert result["brier_score"] >= 0.0
    assert result["bands"]
    assert isinstance(result["monotonic_rank_ordering"], bool)


def test_drift_is_zero_for_identical_samples_and_sinkhorn_can_be_skipped(monkeypatch) -> None:
    bundle = _build_bundle()
    identical = run_drift_tests(bundle.observations, bundle.observations)
    assert identical["score_psi"] == 0.0
    assert identical["score_wasserstein"] == 0.0

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "ot":
            raise ModuleNotFoundError("simulated missing pot")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    without_ot = run_drift_tests(bundle.reference_observations, bundle.observations)
    assert without_ot["score_sinkhorn"]["status"] == "not_run"


def test_validation_pack_returns_all_sections_and_expanded_memo() -> None:
    bundle = _build_bundle()
    result = run_validation_pack(
        model_name="survival_pd_model",
        observations=bundle.observations,
        reference_observations=bundle.reference_observations,
        benchmark_scores=bundle.benchmark_scores,
        scoring_function=sensitivity_score,
        sensitivity_samples=bundle.sensitivity_samples,
        metadata={"domain": "credit_risk", **bundle.metadata},
    )
    assert {"backtest", "stability", "sensitivity", "drift", "benchmark", "memo"}.issubset(result.keys())
    assert "## Scope" in result["memo"]
    assert "## Findings" in result["memo"]
    assert "## Remediation" in result["memo"]
