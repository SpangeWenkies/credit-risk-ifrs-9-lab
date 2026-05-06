"""Limited dependent variable models for credit outcomes.

This module provides compact binary-logit and multinomial-logit helpers for
default, delinquency, cure, and prepayment experiments. It keeps model fitting
transparent and reusable without changing the root survival PD API.

Assumptions
-----------
- Binary outcomes are coded as 0/1.
- Multinomial outcomes are categorical state labels.
- The functions are intended for challenger and educational analysis, not
  production model deployment.

Primary references
------------------
- McFadden (1974), "Conditional Logit Analysis of Qualitative Choice Behavior."
- Cox (1958), "Two Further Applications of a Model for Binary Regression."

Simplifications for this portfolio project
------------------------------------------
- The module uses `statsmodels` GLM/Logit-style estimators without advanced
  regularisation.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import statsmodels.api as sm


def prepare_regression_design(
    frame: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str] | None = None,
    design_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Prepare a stable regression design matrix.

    Summary
    -------
    Convert numeric and categorical predictors into a model-ready design matrix.

    Method
    ------
    Numeric columns are cast to float, categorical columns are one-hot encoded
    with a dropped baseline, and a constant is added. Optional `design_columns`
    reindex the matrix for scoring with previously fitted coefficients.

    Parameters
    ----------
    frame:
        Input data.
    numeric_columns:
        Numeric predictor names.
    categorical_columns:
        Optional categorical predictor names.
    design_columns:
        Optional fixed design column order.

    Returns
    -------
    pandas.DataFrame
        Design matrix.

    Raises
    ------
    KeyError
        Raised when requested predictors are missing.

    Notes
    -----
    This helper mirrors the survival-model design preparation while keeping the
    limited-dependent-variable module independent.

    Edge Cases
    ----------
    Missing values are imputed with zero for synthetic demonstration purposes.

    References
    ----------
    - Cox, D. R. (1958), "Two Further Applications of a Model for Binary
      Regression."
    """

    categorical_columns = list(categorical_columns or [])
    required = [*numeric_columns, *categorical_columns]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing regression design columns: {missing}")
    numeric = frame[list(numeric_columns)].astype(float).fillna(0.0)
    if categorical_columns:
        categoricals = pd.get_dummies(frame[categorical_columns].astype("category"), drop_first=True, dtype=float)
        design = pd.concat([numeric, categoricals], axis=1)
    else:
        design = numeric
    design = sm.add_constant(design, has_constant="add")
    design = design.reindex(sorted(design.columns), axis=1)
    if design_columns is not None:
        design = design.reindex(list(design_columns), axis=1, fill_value=0.0)
    return design


def fit_binary_logit(
    frame: pd.DataFrame,
    target_column: str,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str] | None = None,
):
    """Fit a transparent binary logit model.

    Summary
    -------
    Estimate a binary outcome model for credit events such as default,
    prepayment, cure, or delinquency.

    Method
    ------
    The function prepares a design matrix and fits a binomial GLM with logit
    link using `statsmodels`.

    Parameters
    ----------
    frame:
        Input model frame.
    target_column:
        Binary target column.
    numeric_columns:
        Numeric predictors.
    categorical_columns:
        Optional categorical predictors.

    Returns
    -------
    statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        Fitted binary logit result.

    Raises
    ------
    KeyError
        Raised when the target or predictors are missing.
    ValueError
        Raised when the target has fewer than two classes.

    Notes
    -----
    This helper is useful for quick challenger models and limited-dependent
    variable demonstrations.

    Edge Cases
    ----------
    Complete separation can still cause unstable coefficients; the function
    does not hide those model diagnostics.

    References
    ----------
    - Cox, D. R. (1958), "Two Further Applications of a Model for Binary
      Regression."
    """

    if target_column not in frame.columns:
        raise KeyError(f"Missing target column: {target_column}")
    target = frame[target_column].astype(int)
    if target.nunique() < 2:
        raise ValueError("Binary target must contain both classes.")
    design = prepare_regression_design(frame, numeric_columns, categorical_columns)
    return sm.GLM(target, design, family=sm.families.Binomial()).fit()


def predict_binary_logit(result, frame: pd.DataFrame, numeric_columns: Sequence[str], categorical_columns: Sequence[str] | None = None) -> pd.Series:
    """Predict probabilities from a fitted binary logit model.

    Summary
    -------
    Score a frame with a fitted limited-dependent-variable model.

    Method
    ------
    The function rebuilds the design matrix using the fitted coefficient index
    and applies the model's `predict` method.

    Parameters
    ----------
    result:
        Fitted statsmodels binary logit result.
    frame:
        Scoring frame.
    numeric_columns:
        Numeric predictors.
    categorical_columns:
        Optional categorical predictors.

    Returns
    -------
    pandas.Series
        Predicted probabilities.

    Raises
    ------
    KeyError
        Raised when scoring features are missing.

    Notes
    -----
    Stable dummy-column alignment prevents category mismatch errors at scoring
    time.

    Edge Cases
    ----------
    Categories unseen in fitting are dropped by reindexing to fitted columns.

    References
    ----------
    - Cox, D. R. (1958), "Two Further Applications of a Model for Binary
      Regression."
    """

    design = prepare_regression_design(frame, numeric_columns, categorical_columns, design_columns=result.params.index)
    return pd.Series(result.predict(design), index=frame.index, name="predicted_probability").clip(1e-8, 1 - 1e-8)
