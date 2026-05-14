"""Reporting helpers for credit risk lab outputs."""

from __future__ import annotations

import pandas as pd

from .models import ECLReport, MonitoringReport, PDScoreFrame, PortfolioDataset


def format_table(frame: pd.DataFrame, max_rows: int = 12) -> str:
    """Render a small `pandas` table as aligned plain text."""

    if frame.empty:
        return "(no rows)"
    preview = frame.head(max_rows).copy()
    preview = preview.round(6)
    return preview.to_string(index=False)


def build_portfolio_report(
    dataset: PortfolioDataset,
    pd_scores: PDScoreFrame,
    ecl_report: ECLReport,
    monitoring_report: MonitoringReport,
) -> str:
    """Build a concise Markdown-style portfolio summary."""

    latest_snapshot = dataset.snapshots.loc[dataset.snapshots["snapshot_date"] == pd_scores.as_of_date].iloc[0]
    stage_table = format_table(ecl_report.stage_summary)
    scenario_table = format_table(ecl_report.scenario_summary)
    score_drift = format_table(monitoring_report.score_drift)
    roll_forward = format_table(ecl_report.provision_roll_forward)

    return f"""# Credit Risk & IFRS 9 Summary

Reporting date: {pd_scores.as_of_date.date()}

## Portfolio

- active loans: {int(latest_snapshot['active_loans'])}
- total balance: {latest_snapshot['total_balance']:.2f}
- average DPD: {latest_snapshot['avg_dpd']:.2f}
- average DTI: {latest_snapshot['avg_dti']:.4f}

## Stage Summary

```text
{stage_table}
```

## Scenario Summary

```text
{scenario_table}
```

## Provision Roll-Forward

```text
{roll_forward}
```

## Score Drift

```text
{score_drift}
```
"""
