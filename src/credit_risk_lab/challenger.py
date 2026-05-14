"""Simple challenger model for portfolio benchmarking.

This module provides a deliberately transparent challenger score so the
validation pack can compare a primary survival model with a simpler benchmark.
The challenger is not meant to be superior; it is meant to create a realistic
"base versus challenger" validation workflow.

Assumptions
-----------
- The challenger is a cross-sectional logistic-style score built directly from
  current snapshot features.
- The same synthetic risk drivers used by the main model should directionally
  increase the challenger score.

Primary references
------------------
- EBA, "Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures."
  https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/model-validation/guidelines-pd-estimation-lgd
- ECB Banking Supervision, "Internal models."
  https://www.bankingsupervision.europa.eu/activities/internal_models/html/index.en.html

Simplifications for this lab
------------------------------------------
- No separate fitting step is performed.
- The formula is intentionally hand-crafted so differences versus the primary
  model can be discussed in plain language.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def score_challenger_model(snapshot_scores: pd.DataFrame) -> pd.Series:
    """Generate a simple challenger PD score from snapshot features.

    Summary
    -------
    Produce a benchmark probability score that can be used in model validation
    comparisons against the primary survival model.

    Method
    ------
    The challenger applies a fixed logistic formula to rating, delinquency,
    leverage, indebtedness, utilization, and macro stress variables. This keeps
    the benchmark interpretable while remaining different enough from the main
    survival specification to support a meaningful comparison.

    Parameters
    ----------
    snapshot_scores:
        Current or historical scored snapshot containing the required features.

    Returns
    -------
    pandas.Series
        Challenger 12-month PD values indexed like the input frame.

    Raises
    ------
    KeyError
        Raised if one of the required input columns is missing.

    Notes
    -----
    The challenger is intentionally simpler than the main model. In EU internal-
    model governance, a benchmark does not need to be a perfect alternative; it
    needs to be a credible point of comparison.

    Edge Cases
    ----------
    Missing values are filled with zero before scoring because the synthetic
    pipeline already keeps key fields populated.

    References
    ----------
    - EBA, "Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures."
      https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/model-validation/guidelines-pd-estimation-lgd
    """

    required = ["rating_rank", "days_past_due", "ltv", "dti", "utilization", "unemployment_rate"]
    missing = [column for column in required if column not in snapshot_scores.columns]
    if missing:
        raise KeyError(f"Missing challenger features: {missing}")

    frame = snapshot_scores[required].astype(float).fillna(0.0)
    linear = (
        -5.0
        + 0.58 * frame["rating_rank"]
        + 0.021 * frame["days_past_due"]
        + 1.20 * np.maximum(frame["ltv"] - 0.85, 0.0)
        + 1.10 * np.maximum(frame["dti"] - 0.35, 0.0)
        + 0.65 * np.maximum(frame["utilization"] - 0.70, 0.0)
        + 6.0 * np.maximum(frame["unemployment_rate"] - 0.045, 0.0)
    )
    return pd.Series(np.clip(1.0 / (1.0 + np.exp(-linear)), 0.001, 0.95), index=snapshot_scores.index, name="challenger_pd_12m")
