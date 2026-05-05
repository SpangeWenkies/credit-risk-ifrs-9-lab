"""Benchmark comparison metrics for validation exercises."""

from __future__ import annotations

from .types import ScoredObservation


def _pearson_correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 1.0
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((l - left_mean) * (r - right_mean) for l, r in zip(left, right, strict=True))
    left_denominator = sum((l - left_mean) ** 2 for l in left) ** 0.5
    right_denominator = sum((r - right_mean) ** 2 for r in right) ** 0.5
    if left_denominator == 0.0 or right_denominator == 0.0:
        return 1.0
    return numerator / (left_denominator * right_denominator)


def compare_with_benchmark(
    observations: list[ScoredObservation],
    benchmark_scores: list[float],
) -> dict[str, object]:
    """Compare primary probability scores with a benchmark score vector.

    Summary
    -------
    Measure whether the primary model outperforms a benchmark or challenger
    score vector on the same validation sample.

    Method
    ------
    The function computes Brier scores for the primary and benchmark scores on
    the same realised outcomes, then reports the Brier delta, score
    correlation, and mean absolute score gap.

    Parameters
    ----------
    observations:
        Primary scored observations with realised outcomes.
    benchmark_scores:
        Benchmark or challenger score vector aligned one-for-one with the
        observations.

    Returns
    -------
    dict[str, object]
        Comparison payload containing Brier metrics and score alignment
        diagnostics.

    Raises
    ------
    ValueError
        Raised if the benchmark vector length does not match the number of
        observations.

    Notes
    -----
    A validation benchmark need not dominate the primary model. Its value is to
    provide an independent reference point under EU-style internal-model
    governance.

    Edge Cases
    ----------
    Small samples can yield unstable correlations; the function still returns a
    numeric result for transparency.

    References
    ----------
    - Brier, G. W. (1950), "Verification of Forecasts Expressed in Terms of
      Probability." https://cir.nii.ac.jp/crid/1361981468554183168
    - EBA Model Validation.
      https://www.eba.europa.eu/regulation-and-policy/model-validation
    """

    if len(observations) != len(benchmark_scores):
        raise ValueError("Benchmark scores must align one-for-one with observations.")
    if not observations:
        return {
            "primary_brier": 0.0,
            "benchmark_brier": 0.0,
            "brier_delta": 0.0,
            "score_correlation": 1.0,
            "avg_score_gap": 0.0,
        }

    primary_brier = sum((observation.score - observation.outcome) ** 2 for observation in observations) / len(observations)
    benchmark_brier = sum(
        (benchmark_score - observation.outcome) ** 2
        for benchmark_score, observation in zip(benchmark_scores, observations, strict=True)
    ) / len(observations)
    primary_scores = [observation.score for observation in observations]
    return {
        "primary_brier": round(primary_brier, 6),
        "benchmark_brier": round(benchmark_brier, 6),
        "brier_delta": round(primary_brier - benchmark_brier, 6),
        "score_correlation": round(_pearson_correlation(primary_scores, benchmark_scores), 6),
        "avg_score_gap": round(
            sum(abs(primary - benchmark) for primary, benchmark in zip(primary_scores, benchmark_scores, strict=True)) / len(observations),
            6,
        ),
    }
