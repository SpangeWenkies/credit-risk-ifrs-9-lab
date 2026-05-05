from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".mplconfig"))

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from credit_risk_lab import (  # noqa: E402
    PortfolioConfig,
    build_portfolio_report,
    fit_survival_pd_model,
    generate_portfolio_timeseries,
    run_ifrs9_pipeline,
    run_monitoring,
    score_portfolio,
)


def main() -> None:
    dataset = generate_portfolio_timeseries(PortfolioConfig(random_seed=11, periods=12, num_borrowers=160, num_loans=280))
    model = fit_survival_pd_model(dataset.performance)
    as_of_date = dataset.performance["snapshot_date"].max()
    scores = score_portfolio(dataset, model, as_of_date)
    ecl_report = run_ifrs9_pipeline(dataset, scores)
    monitoring_report = run_monitoring(
        reference_snapshot=scores.previous_snapshot_scores,
        current_snapshot=scores.snapshot_scores,
        reference_scores=scores.previous_snapshot_scores,
        current_scores=scores.snapshot_scores,
    )
    print(build_portfolio_report(dataset, scores, ecl_report, monitoring_report))


if __name__ == "__main__":
    main()
