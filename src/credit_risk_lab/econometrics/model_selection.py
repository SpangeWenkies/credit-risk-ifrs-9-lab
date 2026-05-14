"""Model-selection helpers for credit-risk challenger models.

This module implements information-criterion and holdout-style comparison
utilities for transparent binary-response challenger models.

Assumptions
-----------
- Candidate models are fitted on the same target and comparable samples.
- Information criteria are used for model selection diagnostics, not as the sole
  governance decision.

Primary references
------------------
- Akaike (1974), "A New Look at the Statistical Model Identification."
- Schwarz (1978), "Estimating the Dimension of a Model."
- Brier (1950), "Verification of Forecasts Expressed in Terms of Probability."

Simplifications for this lab
------------------------------------------
- Candidate fitting uses binary GLMs from the limited-dependent-variable module.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import pandas as pd

from .calibration import brier_score
from .limited_dep import fit_binary_logit, predict_binary_logit


@dataclass(slots=True)
class CandidateSpec:
    """Specification for one binary-response challenger model."""

    name: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...] = ()


def compare_binary_model_specs(
    frame: pd.DataFrame,
    target_column: str,
    candidates: Sequence[CandidateSpec],
) -> pd.DataFrame:
    """Compare binary logit candidate specifications.

    Summary
    -------
    Fit multiple candidate binary models and compare fit statistics.

    Method
    ------
    Each candidate is fitted on the same frame. The function reports log
    likelihood, AIC, BIC, parameter count, and in-sample Brier score.

    Parameters
    ----------
    frame:
        Input model frame.
    target_column:
        Binary target column.
    candidates:
        Candidate model specifications.

    Returns
    -------
    pandas.DataFrame
        One row per fitted candidate.

    Raises
    ------
    ValueError
        Raised when no candidates are supplied.
    KeyError
        Raised when target or feature columns are missing.

    Notes
    -----
    AIC/BIC reward fit while penalising complexity. They should be combined with
    out-of-sample and calibration checks.

    Edge Cases
    ----------
    Candidates that fail to fit are reported with `status='failed'` and the
    exception message.

    References
    ----------
    - Akaike, H. (1974), "A New Look at the Statistical Model Identification."
    - Schwarz, G. (1978), "Estimating the Dimension of a Model."
    - Brier, G. W. (1950), "Verification of Forecasts Expressed in Terms of
      Probability."
    """

    if not candidates:
        raise ValueError("At least one candidate specification is required.")
    rows = []
    for candidate in candidates:
        try:
            result = fit_binary_logit(frame, target_column, candidate.numeric_columns, candidate.categorical_columns)
            scored = frame.copy()
            scored["_prediction"] = predict_binary_logit(result, frame, candidate.numeric_columns, candidate.categorical_columns)
            rows.append(
                {
                    "model": candidate.name,
                    "status": "fit",
                    "nobs": int(result.nobs),
                    "n_params": int(len(result.params)),
                    "log_likelihood": float(result.llf),
                    "aic": float(result.aic),
                    "bic": float(result.bic),
                    "brier_score": brier_score(scored, "_prediction", target_column),
                    "error": "",
                }
            )
        except Exception as exc:  # pragma: no cover - exercised through result table in exploratory use
            rows.append(
                {
                    "model": candidate.name,
                    "status": "failed",
                    "nobs": 0,
                    "n_params": 0,
                    "log_likelihood": float("nan"),
                    "aic": float("nan"),
                    "bic": float("nan"),
                    "brier_score": float("nan"),
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows).sort_values(["status", "aic"], na_position="last").reset_index(drop=True)


def select_best_model(comparison_table: pd.DataFrame, criterion: str = "aic") -> pd.Series:
    """Select the best fitted model by a comparison criterion.

    Summary
    -------
    Return the best candidate row from a model-comparison table.

    Method
    ------
    The function filters fitted models and selects the row with the minimum
    requested criterion.

    Parameters
    ----------
    comparison_table:
        Output of `compare_binary_model_specs`.
    criterion:
        Column to minimise, typically `aic`, `bic`, or `brier_score`.

    Returns
    -------
    pandas.Series
        Best model row.

    Raises
    ------
    KeyError
        Raised when required columns are missing.
    ValueError
        Raised when no fitted models are available.

    Notes
    -----
    This utility is intentionally simple so model selection remains auditable.

    Edge Cases
    ----------
    Ties are resolved by the existing row order.

    References
    ----------
    - Akaike, H. (1974), "A New Look at the Statistical Model Identification."
    - Schwarz, G. (1978), "Estimating the Dimension of a Model."
    """

    required = ["status", criterion]
    missing = [column for column in required if column not in comparison_table.columns]
    if missing:
        raise KeyError(f"Missing model-selection columns: {missing}")
    fitted = comparison_table.loc[comparison_table["status"].eq("fit")].dropna(subset=[criterion])
    if fitted.empty:
        raise ValueError("No fitted models are available for selection.")
    return fitted.sort_values(criterion).iloc[0]
