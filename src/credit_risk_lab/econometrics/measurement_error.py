"""Measurement-error and data-quality robustness utilities.

This module adds econometric diagnostics for missingness, noisy covariates, and
reporting bias. These tools complement the monitoring layer by focusing on how
imperfect measurement can affect model inputs and outputs.

Assumptions
-----------
- Missingness and noise are analysed column by column.
- Robustness checks are deterministic when a random seed is supplied.
- The utilities are designed for sensitivity analysis, not for automatic data
  repair.

Primary references
------------------
- Fuller, "Measurement Error Models."
- Little and Rubin, "Statistical Analysis with Missing Data."

Simplifications for this portfolio project
------------------------------------------
- Noise injection is additive Gaussian noise for numeric variables.
- Missingness summaries are descriptive rather than formal missingness tests.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def missingness_by_segment(frame: pd.DataFrame, columns: Sequence[str], segment_column: str | None = None) -> pd.DataFrame:
    """Summarise missingness overall or by segment.

    Summary
    -------
    Report missing-value rates for selected model inputs.

    Method
    ------
    The function computes the fraction of null values for each selected column,
    optionally within each segment.

    Parameters
    ----------
    frame:
        Input data.
    columns:
        Columns to audit.
    segment_column:
        Optional segment column.

    Returns
    -------
    pandas.DataFrame
        Missingness table.

    Raises
    ------
    KeyError
        Raised when requested columns are missing.

    Notes
    -----
    Measurement error often begins as data availability bias, so missingness is
    treated as a modelling issue rather than only a data-engineering issue.

    Edge Cases
    ----------
    Empty frames return missing rates as `NaN`.

    References
    ----------
    - Little, R. J. A., and Rubin, D. B., "Statistical Analysis with Missing
      Data."
    """

    required = list(columns) + ([segment_column] if segment_column else [])
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing missingness-audit columns: {missing}")
    if segment_column is None:
        return pd.DataFrame(
            [{"segment": "all", "column": column, "missing_rate": float(frame[column].isna().mean())} for column in columns]
        )
    rows = []
    for segment, group in frame.groupby(segment_column):
        for column in columns:
            rows.append({"segment": segment, "column": column, "missing_rate": float(group[column].isna().mean())})
    return pd.DataFrame(rows)


def inject_numeric_measurement_error(
    frame: pd.DataFrame,
    noise_scales: Mapping[str, float],
    random_seed: int = 42,
) -> pd.DataFrame:
    """Inject additive numeric measurement error.

    Summary
    -------
    Create a noisy copy of selected numeric inputs for robustness experiments.

    Method
    ------
    For each selected column, independent Gaussian noise with the requested
    standard deviation is added to the original value.

    Parameters
    ----------
    frame:
        Input data.
    noise_scales:
        Mapping from column name to additive noise standard deviation.
    random_seed:
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Noisy copy of the input frame.

    Raises
    ------
    KeyError
        Raised when a selected column is missing.
    ValueError
        Raised when a noise scale is negative.

    Notes
    -----
    This is useful for asking whether PD outputs are fragile to plausible input
    measurement error such as noisy DTI or collateral values.

    Edge Cases
    ----------
    Zero noise leaves a column unchanged.

    References
    ----------
    - Fuller, W. A., "Measurement Error Models."
    """

    missing = [column for column in noise_scales if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing noisy columns: {missing}")
    negative = [column for column, scale in noise_scales.items() if scale < 0]
    if negative:
        raise ValueError(f"Noise scales must be non-negative: {negative}")
    rng = np.random.default_rng(random_seed)
    result = frame.copy()
    for column, scale in noise_scales.items():
        result[column] = result[column].astype(float) + rng.normal(0.0, float(scale), size=len(result))
    return result


def prediction_noise_sensitivity(
    base_scores: pd.Series,
    stressed_scores: pd.Series,
) -> dict[str, float]:
    """Measure score sensitivity to noisy inputs.

    Summary
    -------
    Compare baseline and stressed model scores after a measurement-error
    experiment.

    Method
    ------
    The function aligns two score series and reports mean absolute change, max
    absolute change, and rank correlation.

    Parameters
    ----------
    base_scores:
        Baseline predicted probabilities.
    stressed_scores:
        Stressed predicted probabilities.

    Returns
    -------
    dict[str, float]
        Sensitivity metrics.

    Raises
    ------
    ValueError
        Raised when no aligned scores are available.

    Notes
    -----
    Large score shifts under small input noise are a warning sign for model
    robustness and validation review.

    Edge Cases
    ----------
    If rank correlation is undefined because one series is constant, it is
    returned as `NaN`.

    References
    ----------
    - Fuller, W. A., "Measurement Error Models."
    """

    aligned = pd.concat([base_scores.rename("base"), stressed_scores.rename("stressed")], axis=1).dropna()
    if aligned.empty:
        raise ValueError("No aligned scores are available.")
    diff = aligned["stressed"].astype(float) - aligned["base"].astype(float)
    return {
        "mean_absolute_change": float(diff.abs().mean()),
        "max_absolute_change": float(diff.abs().max()),
        "rank_correlation": float(aligned["base"].corr(aligned["stressed"], method="spearman")),
    }
