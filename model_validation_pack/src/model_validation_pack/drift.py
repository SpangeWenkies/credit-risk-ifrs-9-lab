"""Distribution-shift diagnostics for validation samples.

This module pairs classic score stability metrics such as PSI and Wasserstein
distance with an optional Sinkhorn divergence implementation. The Sinkhorn path
is deliberately optional so the validation pack remains usable even when the OT
dependency is not installed.

Assumptions
-----------
- Comparisons are performed on one-dimensional scores and numeric feature
  distributions.
- Sinkhorn is computed on histogram supports rather than raw empirical optimal
  transport plans to keep the implementation simple and portable.

Primary references
------------------
- EBA Model Validation.
  https://www.eba.europa.eu/regulation-and-policy/model-validation
- Cuturi, M. (2013), "Sinkhorn Distances: Lightspeed Computation of Optimal
  Transport." https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport

Simplifications for this portfolio project
------------------------------------------
- Sinkhorn uses a one-dimensional histogram cost matrix.
- No attempt is made to choose the transport regularisation parameter
  adaptively.
"""

from __future__ import annotations

from math import isclose, log

import numpy as np

from .types import ScoredObservation


def _histogram(values: list[float], bins: int, low: float, high: float) -> np.ndarray:
    if not values:
        return np.zeros(bins, dtype=float)
    if isclose(low, high):
        counts = np.zeros(bins, dtype=float)
        counts[0] = len(values)
        return counts / max(counts.sum(), 1.0)
    edges = np.linspace(low, high, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    return counts / max(counts.sum(), 1.0)


def _psi(reference: list[float], current: list[float], bins: int = 10) -> float:
    if not reference and not current:
        return 0.0
    combined = reference + current
    low = min(combined)
    high = max(combined)
    ref = np.clip(_histogram(reference, bins, low, high), 1e-6, None)
    cur = np.clip(_histogram(current, bins, low, high), 1e-6, None)
    return round(float(np.sum((cur - ref) * np.log(cur / ref))), 6)


def _quantile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _wasserstein(reference: list[float], current: list[float], grid_size: int = 25) -> float:
    if not reference and not current:
        return 0.0
    if not reference or not current:
        return 1.0
    distance = 0.0
    for step in range(grid_size + 1):
        quantile = step / grid_size
        distance += abs(_quantile(reference, quantile) - _quantile(current, quantile))
    return round(distance / (grid_size + 1), 6)


def _sinkhorn(reference: list[float], current: list[float], bins: int = 20, reg: float = 0.01) -> dict[str, object]:
    try:
        import ot
    except ModuleNotFoundError:
        return {"status": "not_run", "reason": "Optional dependency 'pot' is not installed.", "value": None}

    if not reference and not current:
        return {"status": "completed", "reason": "", "value": 0.0}
    if not reference or not current:
        return {"status": "not_run", "reason": "One of the compared samples is empty.", "value": None}

    combined = reference + current
    low = min(combined)
    high = max(combined)
    ref = np.clip(_histogram(reference, bins, low, high), 1e-6, None)
    cur = np.clip(_histogram(current, bins, low, high), 1e-6, None)
    ref = ref / ref.sum()
    cur = cur / cur.sum()
    centers = np.linspace(low, high, bins)
    cost = (centers[:, None] - centers[None, :]) ** 2
    value = float(ot.sinkhorn2(ref, cur, cost, reg=reg))
    return {"status": "completed", "reason": "", "value": round(value, 6)}


def run_drift_tests(
    reference_observations: list[ScoredObservation],
    current_observations: list[ScoredObservation],
) -> dict[str, object]:
    """Run score and feature drift checks between two observation samples.

    Summary
    -------
    Compare a reference validation sample with a current sample using PSI,
    Wasserstein distance, optional Sinkhorn divergence, and simple feature mean
    shifts.

    Method
    ------
    Score drift is measured on the one-dimensional model scores. Feature drift is
    summarised through per-feature reference/current means plus PSI and
    Wasserstein distance on any numeric features shared by both samples. If the
    optional OT dependency is available, the score comparison also includes a
    Sinkhorn divergence.

    Parameters
    ----------
    reference_observations:
        Reference validation sample.
    current_observations:
        Current validation sample.

    Returns
    -------
    dict[str, object]
        Drift payload with score-level metrics, optional Sinkhorn results, and
        per-feature shift summaries.

    Raises
    ------
    None
        This function degrades gracefully and reports `not_run` when Sinkhorn
        cannot be computed.

    Notes
    -----
    Using both bucketed and transport-based metrics gives a better portfolio
    story than using a single stability statistic.

    Edge Cases
    ----------
    Empty comparisons return finite placeholders rather than raising.

    References
    ----------
    - Cuturi, M. (2013), "Sinkhorn Distances: Lightspeed Computation of Optimal
      Transport." https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport
    - EBA Model Validation.
      https://www.eba.europa.eu/regulation-and-policy/model-validation
    """

    reference_scores = [observation.score for observation in reference_observations]
    current_scores = [observation.score for observation in current_observations]
    if any(observation.features for observation in reference_observations + current_observations):
        common_features = set.intersection(
            *(set(observation.features.keys()) for observation in reference_observations + current_observations if observation.features)
        )
    else:
        common_features = set()

    feature_rows = []
    for feature_name in sorted(common_features):
        reference_feature = [observation.features[feature_name] for observation in reference_observations if feature_name in observation.features]
        current_feature = [observation.features[feature_name] for observation in current_observations if feature_name in observation.features]
        feature_rows.append(
            {
                "feature": feature_name,
                "reference_mean": round(sum(reference_feature) / max(len(reference_feature), 1), 6),
                "current_mean": round(sum(current_feature) / max(len(current_feature), 1), 6),
                "shift": round((sum(current_feature) / max(len(current_feature), 1)) - (sum(reference_feature) / max(len(reference_feature), 1)), 6),
                "psi": _psi(reference_feature, current_feature),
                "wasserstein": _wasserstein(reference_feature, current_feature),
            }
        )

    return {
        "score_psi": _psi(reference_scores, current_scores),
        "score_wasserstein": _wasserstein(reference_scores, current_scores),
        "score_sinkhorn": _sinkhorn(reference_scores, current_scores),
        "feature_drift": feature_rows,
    }
