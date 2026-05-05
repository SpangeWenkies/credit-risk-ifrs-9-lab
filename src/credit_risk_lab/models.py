"""Core types for the credit risk and IFRS 9 lab.

This module defines the dataclasses that connect the synthetic portfolio
generator, the discrete-time survival model, the IFRS 9 engine, and the
monitoring layer. The project treats these types as the stable public surface
of the root package so that the downstream validation pack can consume outputs
without knowing implementation details.

Assumptions
-----------
- Portfolio data is represented with :class:`pandas.DataFrame` objects rather
  than custom row classes so the project resembles common risk-modelling
  workflows.
- Discrete-time survival modelling is performed on quarterly panel data rather
  than monthly panels to keep the synthetic portfolio readable while preserving
  the core hazard-modelling logic discussed by Singer and Willett.
- IFRS 9 outputs are simplified portfolio artefacts intended for a portfolio
  project rather than production accounting books.

Primary references
------------------
- Singer, J. D., and Willett, J. B. (1993), "It's About Time: Using
  Discrete-Time Survival Analysis to Study Duration and the Timing of Events."
  https://journals.sagepub.com/doi/10.3102/10769986018002155
- IFRS Foundation, "IFRS 9 Financial Instruments."
  https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf

Simplifications for this portfolio project
------------------------------------------
- The dataclasses deliberately store rich intermediate tables so the notebook,
  example scripts, and validation pack can inspect internal calculations.
- Type annotations prefer clarity over very strict runtime protocols because
  the repository is educational and recruiter-facing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(slots=True)
class PortfolioConfig:
    """Configuration for synthetic multi-period portfolio generation."""

    random_seed: int = 42
    start_date: str = "2022-03-31"
    periods: int = 12
    num_borrowers: int = 350
    num_loans: int = 500


@dataclass(slots=True)
class PortfolioDataset:
    """Container for the synthetic portfolio tables used throughout the repo."""

    borrowers: pd.DataFrame
    loans: pd.DataFrame
    performance: pd.DataFrame
    defaults: pd.DataFrame
    recoveries: pd.DataFrame
    macro: pd.DataFrame
    snapshots: pd.DataFrame


@dataclass(slots=True)
class FeatureSpec:
    """Feature specification for the pooled-logit survival model."""

    numeric_columns: list[str]
    categorical_columns: list[str]
    target_column: str = "default_next_period"


@dataclass(slots=True)
class SurvivalPDModel:
    """Fitted pooled-logit survival model and its design metadata."""

    result: Any
    feature_spec: FeatureSpec
    design_columns: list[str]
    interval_name: str = "quarter"
    coefficient_table: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(slots=True)
class PDScoreFrame:
    """Model scores for the current and previous portfolio snapshots."""

    as_of_date: pd.Timestamp
    snapshot_scores: pd.DataFrame
    previous_snapshot_scores: pd.DataFrame
    scored_history: pd.DataFrame


@dataclass(slots=True)
class MacroScenario:
    """Simple macro scenario path used by the IFRS 9 ECL engine."""

    name: str
    unemployment_shift: float
    house_price_shift: float
    policy_rate_shift: float
    weight: float


@dataclass(slots=True)
class ECLReport:
    """IFRS 9 outputs for one reporting date."""

    as_of_date: pd.Timestamp
    loan_results: pd.DataFrame
    stage_summary: pd.DataFrame
    scenario_summary: pd.DataFrame
    migration_matrix: pd.DataFrame
    provision_roll_forward: pd.DataFrame


@dataclass(slots=True)
class MonitoringReport:
    """Data-quality and drift-monitoring outputs for one comparison window."""

    schema_checks: pd.DataFrame
    missingness: pd.DataFrame
    range_checks: pd.DataFrame
    cohort_checks: pd.DataFrame
    feature_drift: pd.DataFrame
    score_drift: pd.DataFrame
    summary: dict[str, Any]
