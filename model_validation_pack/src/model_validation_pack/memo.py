"""Memo rendering for validation outputs."""

from __future__ import annotations

from pathlib import Path


TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "templates" / "validation_memo.md"


def _rows_to_lines(rows: list[dict[str, object]], label_key: str) -> str:
    if not rows:
        return "- no results"
    return "\n".join(
        f"- {row[label_key]}: " + ", ".join(f"{key}={value}" for key, value in row.items() if key != label_key)
        for row in rows
    )


def _findings(backtest: dict[str, object], drift: dict[str, object], benchmark: dict[str, object]) -> list[str]:
    findings: list[str] = []
    if abs(float(backtest.get("calibration_gap", 0.0))) > 0.02:
        findings.append(f"Calibration gap is material at {backtest.get('calibration_gap')}.")
    if float(drift.get("score_psi", 0.0)) > 0.10:
        findings.append(f"Score PSI indicates drift at {drift.get('score_psi')}.")
    sinkhorn = drift.get("score_sinkhorn", {})
    if isinstance(sinkhorn, dict) and sinkhorn.get("status") == "completed" and float(sinkhorn.get("value") or 0.0) > 0.001:
        findings.append(f"Sinkhorn divergence is non-trivial at {sinkhorn.get('value')}.")
    if isinstance(benchmark, dict) and benchmark.get("brier_delta") is not None and float(benchmark.get("brier_delta", 0.0)) > 0.0:
        findings.append("Benchmark outperforms the primary model on Brier score.")
    if not findings:
        findings.append("No material adverse findings were detected in the compact validation checks.")
    return findings


def render_validation_memo(
    model_name: str,
    backtest: dict[str, object],
    stability: dict[str, object],
    sensitivity: dict[str, object],
    drift: dict[str, object],
    benchmark: dict[str, object],
    metadata: dict[str, object] | None = None,
) -> str:
    """Render a markdown validation memo from validation outputs.

    Summary
    -------
    Convert validation results into a memo format that looks closer to real
    model governance output than a raw dictionary dump.

    Method
    ------
    The renderer fills a markdown template with scope, methodology, backtest,
    stability, drift, benchmark, findings, limitations, and remediation
    sections. Findings and conclusion text are derived from a small ruleset so
    the memo remains deterministic and reproducible.

    Parameters
    ----------
    model_name:
        Name of the model under review.
    backtest:
        Backtest payload returned by `run_backtest`.
    stability:
        Stability payload returned by `run_stability_tests`.
    sensitivity:
        Sensitivity payload returned by `run_sensitivity_analysis`.
    drift:
        Drift payload returned by `run_drift_tests`.
    benchmark:
        Benchmark payload or a `not_run` structure.
    metadata:
        Optional extra context, such as the score column or domain example.

    Returns
    -------
    str
        Markdown validation memo.

    Raises
    ------
    FileNotFoundError
        Raised if the memo template is missing.

    Notes
    -----
    The memo is intentionally concise. It should read like a strong first-pass
    validation note rather than a full internal committee pack.

    Edge Cases
    ----------
    If optional sections were not run, the memo marks them explicitly instead of
    pretending a result exists.

    References
    ----------
    - Board of Governors of the Federal Reserve System and OCC, "Supervisory
      Guidance on Model Risk Management (SR 11-7)."
      https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107a1.pdf
    """

    metadata = metadata or {}
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    findings = _findings(backtest, drift, benchmark)
    conclusion = (
        "Model is acceptable for continued use with routine monitoring."
        if findings == ["No material adverse findings were detected in the compact validation checks."]
        else "Model requires targeted remediation or heightened monitoring before broader use."
    )

    sinkhorn = drift.get("score_sinkhorn", {})
    sinkhorn_summary = (
        f"status={sinkhorn.get('status')}, value={sinkhorn.get('value')}, reason={sinkhorn.get('reason')}"
        if isinstance(sinkhorn, dict)
        else str(sinkhorn)
    )

    return template.format(
        model_name=model_name,
        scope_summary=f"- Domain example: {metadata.get('domain', 'generic')}\n- Observation count: {backtest.get('observation_count', 0)}",
        data_summary=f"- Score column: {metadata.get('score_column', 'score')}\n- Outcome column: {metadata.get('outcome_column', 'outcome')}",
        methodology_summary=(
            "- Backtest metric: Brier score.\n"
            "- Stability cuts: period and segment summaries.\n"
            "- Drift metrics: PSI, Wasserstein distance, optional Sinkhorn divergence.\n"
            f"- Sensitivity status: {sensitivity.get('status', 'not_run')}"
        ),
        backtest_summary=_rows_to_lines(backtest.get("bands", []), "band"),
        stability_summary=_rows_to_lines(stability.get("period_summary", []), "period") + "\n" + _rows_to_lines(stability.get("segment_summary", []), "segment"),
        drift_summary=_rows_to_lines(drift.get("feature_drift", []), "feature") + f"\n- score_sinkhorn: {sinkhorn_summary}",
        benchmark_summary="\n".join(f"- {key}: {value}" for key, value in benchmark.items()) if benchmark else "- not_run",
        findings_summary="\n".join(f"- {line}" for line in findings),
        limitations_summary=(
            "- Compact validation metrics only; no confidence-interval estimation.\n"
            "- Synthetic or simplified data may understate production complexity.\n"
            "- Scenario sensitivity is additive and illustrative rather than structural."
        ),
        remediation_summary=(
            "- Investigate material drift before relying on historical calibration.\n"
            "- Benchmark any adverse Brier delta against a challenger refresh.\n"
            "- Extend stability analysis with longer out-of-time samples when available."
        ),
        conclusion=conclusion,
    )
