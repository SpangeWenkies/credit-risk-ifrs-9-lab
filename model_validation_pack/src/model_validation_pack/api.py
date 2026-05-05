"""Top-level orchestration API for the validation pack."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .backtest import run_backtest
from .benchmark import compare_with_benchmark
from .drift import run_drift_tests
from .memo import render_validation_memo
from .sensitivity import run_sensitivity_analysis
from .stability import run_stability_tests
from .types import ScoredObservation


def run_validation_pack(
    model_name: str,
    observations: Sequence[ScoredObservation],
    reference_observations: Sequence[ScoredObservation] | None = None,
    benchmark_scores: Sequence[float] | None = None,
    scoring_function: Callable[[dict[str, float]], float] | None = None,
    sensitivity_samples: Sequence[dict[str, float]] | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    """Run the full compact validation workflow on one scored model sample.

    Summary
    -------
    Execute the validation pack end-to-end: backtest, stability, sensitivity,
    drift, benchmark comparison, and memo rendering.

    Method
    ------
    The function materialises the primary and reference validation samples,
    applies each validation component in sequence, and then renders a markdown
    memo. If optional inputs such as benchmark scores or a sensitivity scoring
    function are missing, the corresponding section returns a structured
    `not_run` payload instead of failing the entire validation run.

    Parameters
    ----------
    model_name:
        Name of the model being validated.
    observations:
        Primary validation sample.
    reference_observations:
        Optional reference sample used for drift checks. If omitted, the
        function splits the primary sample into a reference half and a current
        half.
    benchmark_scores:
        Optional benchmark or challenger score vector aligned with the primary
        sample.
    scoring_function:
        Optional callable used for sensitivity analysis.
    sensitivity_samples:
        Optional explicit feature sample for sensitivity analysis.
    metadata:
        Optional extra context forwarded to the memo renderer.

    Returns
    -------
    dict[str, object]
        Full validation payload with all component outputs and a rendered memo.

    Raises
    ------
    ValueError
        Propagated from benchmark comparison if benchmark alignment is invalid.

    Notes
    -----
    This function is designed to remain generic so the package can later be
    split from the credit-risk lab without code changes.

    Edge Cases
    ----------
    If no explicit reference sample is provided and the primary sample is very
    small, the same observations may partially serve both as reference and
    current slices.

    References
    ----------
    - EBA Model Validation.
      https://www.eba.europa.eu/regulation-and-policy/model-validation
    """

    observation_list = list(observations)
    metadata = metadata or {}
    if reference_observations is None:
        midpoint = max(1, len(observation_list) // 2)
        reference_list = observation_list[:midpoint]
        current_list = observation_list[midpoint:] or observation_list[:midpoint]
    else:
        reference_list = list(reference_observations)
        current_list = observation_list

    backtest = run_backtest(observation_list)
    stability = run_stability_tests(observation_list)
    drift = run_drift_tests(reference_list, current_list)
    if benchmark_scores is None:
        benchmark = {"status": "not_run"}
    else:
        benchmark = compare_with_benchmark(observation_list, list(benchmark_scores))

    if sensitivity_samples is None:
        sensitivity_samples = [observation.features for observation in observation_list if observation.features]
    sensitivity = run_sensitivity_analysis(scoring_function, list(sensitivity_samples))
    memo = render_validation_memo(
        model_name=model_name,
        backtest=backtest,
        stability=stability,
        sensitivity=sensitivity,
        drift=drift,
        benchmark=benchmark,
        metadata=metadata,
    )
    return {
        "backtest": backtest,
        "stability": stability,
        "sensitivity": sensitivity,
        "drift": drift,
        "benchmark": benchmark,
        "memo": memo,
    }
