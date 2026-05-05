from __future__ import annotations

import os
import sys
from math import exp
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mplconfig"))

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
from model_validation_pack import build_credit_validation_bundle, run_validation_pack  # noqa: E402


def sensitivity_score(features: dict[str, float]) -> float:
    linear = -5.0
    linear += 0.55 * features.get("rating_rank", 2.0)
    linear += 0.020 * features.get("days_past_due", 0.0)
    linear += 1.20 * max(features.get("ltv", 0.8) - 0.85, 0.0)
    linear += 1.10 * max(features.get("dti", 0.35) - 0.35, 0.0)
    linear += 0.65 * max(features.get("utilization", 0.60) - 0.70, 0.0)
    return max(min(1.0 / (1.0 + exp(-linear)), 0.95), 0.001)


def main() -> None:
    dataset = generate_portfolio_timeseries(PortfolioConfig(random_seed=9, periods=12, num_borrowers=150, num_loans=260))
    model = fit_survival_pd_model(dataset.performance)
    as_of_date = dataset.performance["snapshot_date"].max()
    scores = score_portfolio(dataset, model, as_of_date)
    challenger_scores = score_challenger_model(scores.snapshot_scores).tolist()

    bundle = build_credit_validation_bundle(
        snapshot_scores=scores.snapshot_scores,
        previous_snapshot_scores=scores.previous_snapshot_scores,
        benchmark_scores=challenger_scores,
        feature_columns=["ltv", "dti", "days_past_due", "utilization", "rating_rank"],
    )
    result = run_validation_pack(
        model_name="survival_pd_model",
        observations=bundle.observations,
        reference_observations=bundle.reference_observations,
        benchmark_scores=bundle.benchmark_scores,
        scoring_function=sensitivity_score,
        sensitivity_samples=bundle.sensitivity_samples,
        metadata={"domain": "credit_risk", **bundle.metadata},
    )

    print("Backtest")
    print(result["backtest"])
    print("\nStability")
    print(result["stability"])
    print("\nDrift")
    print(result["drift"])
    print("\nBenchmark")
    print(result["benchmark"])
    print("\nMemo")
    print(result["memo"])


if __name__ == "__main__":
    main()
