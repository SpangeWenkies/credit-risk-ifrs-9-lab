"""Causal and identification-oriented diagnostics for credit models.

This module does not claim causal identification for the synthetic PD model.
Instead, it makes the predictive-versus-structural distinction explicit and
provides utilities for documenting endogeneity, post-treatment variables, and
policy-shock sensitivity.

Assumptions
-----------
- The base repo is predictive and governance-focused.
- Causal statements require additional identification assumptions that are not
  supplied by the synthetic data generator.
- Variable audits are structured documentation artefacts.

Primary references
------------------
- Angrist and Pischke, "Mostly Harmless Econometrics."
- Pearl, "Causality."
- Imbens and Rubin, "Causal Inference for Statistics, Social, and Biomedical
  Sciences."

Simplifications for this portfolio project
------------------------------------------
- The module creates audit tables and deterministic stress comparisons rather
  than estimating causal treatment effects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd


def variable_identification_audit(
    variables: Sequence[str],
    roles: Mapping[str, str] | None = None,
    concerns: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Create a predictive-versus-causal variable audit table.

    Summary
    -------
    Document how model variables should be interpreted in credit-risk analysis.

    Method
    ------
    Each variable is assigned a role such as `predictive`, `policy`, `macro`,
    `post_treatment`, or `potentially_endogenous`, plus a short concern note.

    Parameters
    ----------
    variables:
        Variables to audit.
    roles:
        Optional mapping from variable to role.
    concerns:
        Optional mapping from variable to concern text.

    Returns
    -------
    pandas.DataFrame
        Variable audit table.

    Raises
    ------
    None
        This function does not raise custom exceptions.

    Notes
    -----
    This is useful interview material: it shows that macro and behavioural
    predictors can be valid for prediction without being structural causal
    effects.

    Edge Cases
    ----------
    Variables without supplied metadata default to `predictive` with a generic
    caution.

    References
    ----------
    - Angrist, J. D., and Pischke, J.-S., "Mostly Harmless Econometrics."
    - Pearl, J., "Causality."
    """

    roles = dict(roles or {})
    concerns = dict(concerns or {})
    rows = []
    for variable in variables:
        rows.append(
            {
                "variable": variable,
                "role": roles.get(variable, "predictive"),
                "causal_interpretation": roles.get(variable, "predictive") not in {"post_treatment", "potentially_endogenous"},
                "concern": concerns.get(variable, "Predictive use is plausible; causal interpretation needs separate identification."),
            }
        )
    return pd.DataFrame(rows)


def policy_shock_sensitivity(
    frame: pd.DataFrame,
    shock_column: str,
    shock_size: float,
    score_function,
    output_column: str = "score",
) -> pd.DataFrame:
    """Apply a deterministic policy or macro shock and rescore.

    Summary
    -------
    Compare baseline and shocked model outputs without claiming causal
    identification.

    Method
    ------
    The function scores the original frame, adds `shock_size` to one variable,
    scores the shocked frame, and returns row-level and aggregate deltas.

    Parameters
    ----------
    frame:
        Input scoring frame.
    shock_column:
        Variable to shock.
    shock_size:
        Additive shock amount.
    score_function:
        Callable that accepts a frame and returns a score series.
    output_column:
        Name to use for baseline and shocked score columns.

    Returns
    -------
    pandas.DataFrame
        Row-level baseline, shocked, and delta scores.

    Raises
    ------
    KeyError
        Raised when `shock_column` is missing.

    Notes
    -----
    This is a sensitivity exercise. A shocked score difference should be framed
    as model response, not causal effect, unless a separate identification
    design is supplied.

    Edge Cases
    ----------
    If the scoring function returns an unnamed series, alignment is still based
    on index.

    References
    ----------
    - Imbens, G. W., and Rubin, D. B., "Causal Inference for Statistics, Social,
      and Biomedical Sciences."
    """

    if shock_column not in frame.columns:
        raise KeyError(f"Missing shock column: {shock_column}")
    baseline = pd.Series(score_function(frame), index=frame.index, name=f"baseline_{output_column}")
    shocked_frame = frame.copy()
    shocked_frame[shock_column] = shocked_frame[shock_column].astype(float) + float(shock_size)
    shocked = pd.Series(score_function(shocked_frame), index=frame.index, name=f"shocked_{output_column}")
    result = pd.concat([baseline, shocked], axis=1)
    result[f"delta_{output_column}"] = result[f"shocked_{output_column}"] - result[f"baseline_{output_column}"]
    return result
