"""Synthetic multi-period retail loan portfolio generation.

This module simulates a quarterly retail credit portfolio with borrower,
account, performance, default, recovery, and macro tables. The generated data
is intentionally synthetic-first so the repository remains reproducible and
shareable while still supporting a realistic modelling workflow.

Assumptions
-----------
- Quarterly snapshots are sufficient for the baseline lab workflow that wants
  to demonstrate discrete-time survival modelling and IFRS 9 staging without
  introducing monthly data volume or unnecessary complexity.
- The simulation imposes intuitive monotonic relationships between worsening
  borrower conditions and higher delinquency/default pressure, but it does not
  claim to be a structural credit-economy model.
- Recovery cashflows are produced deterministically from the simulated default
  severity assumptions so downstream LGD logic can be explained clearly.

Primary references
------------------
- IFRS Foundation, "IFRS 9 Financial Instruments."
  https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf

Simplifications for this lab
------------------------------------------
- The portfolio combines mortgages, auto loans, personal loans, and revolving
  credit in a single synthetic generator.
- Macro variables are simulated rather than sourced externally.
- Default timing, utilization, and recovery patterns are stylised so the code
  remains inspectable and reproducible.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .models import PortfolioConfig, PortfolioDataset

REGIONS = ("north", "south", "east", "west")
SEGMENTS = ("mortgage", "auto", "personal_loan", "credit_card")
RATINGS = ("A", "B", "C", "D", "E")
RATING_TO_RANK = {rating: idx + 1 for idx, rating in enumerate(RATINGS)}
SEGMENT_TERM_RANGES = {
    "mortgage": (20, 28),
    "auto": (8, 16),
    "personal_loan": (6, 12),
    "credit_card": (6, 16),
}
REVOLVING_SEGMENTS = {"credit_card"}


def _logistic(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-value))


def _simulate_macro_table(snapshot_dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Simulate a simple quarterly macro path for the synthetic portfolio.

    Summary
    -------
    Generate unemployment, policy-rate, house-price, and GDP-growth series for
    the quarterly portfolio horizon.

    Method
    ------
    The function evolves each macro variable with a small cyclical component and
    Gaussian noise, then derives house-price growth and GDP growth from those
    simulated levels. The resulting path is deliberately smooth enough for
    scenario storytelling while still moving enough to create variation in the
    synthetic panel.

    Parameters
    ----------
    snapshot_dates:
        Quarterly reporting dates used by the portfolio panel.
    rng:
        NumPy random generator seeded by the portfolio configuration.

    Returns
    -------
    pandas.DataFrame
        Quarterly macro table keyed by `snapshot_date`.

    Raises
    ------
    None
        This helper does not raise custom exceptions.

    Notes
    -----
    The macro path is not intended to reproduce any specific economy. Its job is
    to create coherent directional pressure for the synthetic credit panel.

    Edge Cases
    ----------
    Very short horizons still produce a valid table; the cyclical component just
    has less time to evolve.

    References
    ----------
    - IFRS Foundation, "IFRS 9 Financial Instruments."
      https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf
    """
    unemployment = []
    policy_rate = []
    house_price_index = []
    gdp_growth = []

    current_unemployment = 0.045
    current_policy_rate = 0.020
    current_house_price = 100.0

    for step, snapshot_date in enumerate(snapshot_dates):
        cycle = np.sin(step / 2.5)
        current_unemployment = np.clip(current_unemployment + 0.0020 * cycle + rng.normal(0.0, 0.0015), 0.030, 0.095)
        current_policy_rate = np.clip(current_policy_rate + 0.0015 * np.cos(step / 3.0) + rng.normal(0.0, 0.0012), 0.005, 0.060)
        house_growth = 0.006 - 0.20 * max(current_unemployment - 0.05, 0.0) + rng.normal(0.0, 0.004)
        current_house_price = max(current_house_price * (1.0 + house_growth), 80.0)
        gdp = 0.005 + 0.60 * house_growth - 0.12 * max(current_unemployment - 0.05, 0.0) + rng.normal(0.0, 0.003)

        unemployment.append(current_unemployment)
        policy_rate.append(current_policy_rate)
        house_price_index.append(current_house_price)
        gdp_growth.append(gdp)

    macro = pd.DataFrame(
        {
            "snapshot_date": snapshot_dates,
            "unemployment_rate": np.round(unemployment, 6),
            "policy_rate": np.round(policy_rate, 6),
            "house_price_index": np.round(house_price_index, 4),
            "gdp_growth": np.round(gdp_growth, 6),
        }
    )
    macro["house_price_growth"] = macro["house_price_index"].pct_change().fillna(0.0)
    return macro


def _generate_borrowers(config: PortfolioConfig, rng: np.random.Generator) -> pd.DataFrame:
    borrowers = pd.DataFrame(
        {
            "borrower_id": [f"B{idx + 1:05d}" for idx in range(config.num_borrowers)],
            "region": rng.choice(REGIONS, size=config.num_borrowers, replace=True),
            "annual_income": np.round(rng.lognormal(mean=11.0, sigma=0.35, size=config.num_borrowers), 2),
            "base_dti": np.round(np.clip(rng.normal(0.34, 0.10, size=config.num_borrowers), 0.12, 0.80), 4),
            "employment_stability": np.round(np.clip(rng.normal(0.70, 0.12, size=config.num_borrowers), 0.20, 0.98), 4),
            "unemployment_sensitivity": np.round(np.clip(rng.normal(0.95, 0.25, size=config.num_borrowers), 0.35, 1.60), 4),
            "forbearance_propensity": np.round(np.clip(rng.normal(0.18, 0.08, size=config.num_borrowers), 0.02, 0.55), 4),
        }
    )
    return borrowers


def _sample_origination_balance(segment: str, rng: np.random.Generator) -> tuple[float, float, float]:
    if segment == "mortgage":
        balance = rng.uniform(120_000.0, 420_000.0)
        collateral = balance / rng.uniform(0.60, 0.92)
        limit_amount = balance
    elif segment == "auto":
        balance = rng.uniform(9_000.0, 35_000.0)
        collateral = balance / rng.uniform(0.65, 0.98)
        limit_amount = balance
    elif segment == "personal_loan":
        balance = rng.uniform(4_000.0, 28_000.0)
        collateral = 0.0
        limit_amount = balance
    else:
        balance = rng.uniform(1_500.0, 16_000.0)
        collateral = 0.0
        limit_amount = balance / rng.uniform(0.35, 0.80)
    return balance, collateral, limit_amount


def _generate_loans(
    config: PortfolioConfig,
    borrowers: pd.DataFrame,
    snapshot_dates: pd.DatetimeIndex,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create the static loan master table for the synthetic portfolio.

    Summary
    -------
    Sample product type, origination timing, term, balance, collateral, and
    origination risk markers for each synthetic loan.

    Method
    ------
    The function draws borrowers with replacement, samples a segment mix, assigns
    origination dates within the synthetic history window, and derives
    origination-level balances, collateral, and a simple PD anchor from rating,
    DTI, and LTV.

    Parameters
    ----------
    config:
        Portfolio generation configuration.
    borrowers:
        Borrower master table.
    snapshot_dates:
        Quarterly reporting dates available for origination.
    rng:
        NumPy random generator seeded by the portfolio configuration.

    Returns
    -------
    pandas.DataFrame
        Static loan master table.

    Raises
    ------
    None
        This helper does not raise custom exceptions.

    Notes
    -----
    The origination PD anchor is used later as a deterioration benchmark in the
    simplified IFRS 9 stage-allocation logic.

    Edge Cases
    ----------
    Revolving products receive a credit limit above the current balance while
    amortising loans set `limit_amount == origination_balance`.

    References
    ----------
    - IFRS Foundation, "IFRS 9 Financial Instruments."
      https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf
    """
    borrower_indices = rng.choice(borrowers.index, size=config.num_loans, replace=True)
    segments = rng.choice(SEGMENTS, size=config.num_loans, p=[0.35, 0.22, 0.23, 0.20])
    origination_indices = rng.integers(low=0, high=max(2, len(snapshot_dates) - 4), size=config.num_loans)
    ratings = rng.choice(RATINGS, size=config.num_loans, p=[0.18, 0.30, 0.28, 0.17, 0.07])

    records: list[dict[str, object]] = []
    for loan_idx in range(config.num_loans):
        segment = str(segments[loan_idx])
        balance, collateral, limit_amount = _sample_origination_balance(segment, rng)
        origination_rating = str(ratings[loan_idx])
        term_low, term_high = SEGMENT_TERM_RANGES[segment]
        term_quarters = int(rng.integers(term_low, term_high + 1))
        interest_rate = float(np.clip(rng.normal(0.045, 0.015), 0.015, 0.140))
        borrower = borrowers.iloc[int(borrower_indices[loan_idx])]

        origination_dti = float(np.clip(borrower["base_dti"] + rng.normal(0.0, 0.04), 0.10, 0.90))
        origination_ltv = 1.0 if collateral == 0.0 else balance / collateral
        origination_pd_anchor = float(
            np.clip(
                _logistic(
                    -4.8
                    + 0.75 * RATING_TO_RANK[origination_rating]
                    + 0.95 * max(origination_dti - 0.35, 0.0)
                    + 0.80 * max(origination_ltv - 0.80, 0.0)
                ),
                0.002,
                0.220,
            )
        )

        records.append(
            {
                "loan_id": f"L{loan_idx + 1:05d}",
                "borrower_id": borrower["borrower_id"],
                "segment": segment,
                "region": borrower["region"],
                "origination_date": snapshot_dates[int(origination_indices[loan_idx])],
                "term_quarters": term_quarters,
                "origination_balance": round(balance, 2),
                "limit_amount": round(limit_amount, 2),
                "secured": segment in {"mortgage", "auto"},
                "collateral_value_origination": round(collateral, 2),
                "origination_rating": origination_rating,
                "origination_rating_rank": RATING_TO_RANK[origination_rating],
                "interest_rate": round(interest_rate, 6),
                "origination_dti": round(origination_dti, 4),
                "origination_ltv": round(origination_ltv, 4),
                "origination_pd_anchor": round(origination_pd_anchor, 6),
            }
        )

    loans = pd.DataFrame.from_records(records).sort_values(["origination_date", "loan_id"]).reset_index(drop=True)
    return loans


def generate_portfolio_timeseries(config: PortfolioConfig | None = None) -> PortfolioDataset:
    """Generate a synthetic multi-period retail credit portfolio.

    Summary
    -------
    Build a reproducible quarterly loan-book dataset containing borrower,
    account, performance, default, recovery, macro, and snapshot summary tables.

    Method
    ------
    The generator first simulates a quarterly macro path, then creates borrower
    and loan master tables, and finally rolls each active loan forward through
    quarterly performance states until maturity or default. Default pressure is
    modelled with a stylised logistic mechanism that increases with weaker
    borrower quality, higher delinquency, higher leverage, and a worse macro
    backdrop. This simulation is designed to support discrete-time survival
    modelling rather than to recover structural credit parameters.

    Parameters
    ----------
    config:
        Optional configuration controlling sample size, reporting horizon, and
        random seed. When omitted, :class:`PortfolioConfig` defaults are used.

    Returns
    -------
    PortfolioDataset
        Dataclass whose fields are `pandas` tables named `borrowers`, `loans`,
        `performance`, `defaults`, `recoveries`, `macro`, and `snapshots`.

    Raises
    ------
    ValueError
        Raised if the requested number of periods is too small to create both
        origination history and out-of-time validation windows.

    Notes
    -----
    The current implementation uses quarterly panels. This choice keeps the
    dataset compact while still matching the discrete-time survival framing in
    Singer and Willett (1993) and providing enough history for simple stage
    migration and roll-forward reporting.

    Edge Cases
    ----------
    Loans that default stop generating active performance rows after the default
    quarter. Recoveries are then emitted as separate rows in the `recoveries`
    table. Revolving products retain a credit limit and undrawn amount, while
    amortising products keep `limit_amount == balance`.

    References
    ----------
    - Singer, J. D., and Willett, J. B. (1993), "It's About Time: Using
      Discrete-Time Survival Analysis to Study Duration and the Timing of
      Events." https://journals.sagepub.com/doi/10.3102/10769986018002155
    - IFRS Foundation, "IFRS 9 Financial Instruments."
      https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf
    """

    config = config or PortfolioConfig()
    if config.periods < 8:
        raise ValueError("Portfolio generation requires at least 8 quarterly periods.")

    rng = np.random.default_rng(config.random_seed)
    snapshot_dates = pd.date_range(config.start_date, periods=config.periods, freq="QE-DEC")
    macro = _simulate_macro_table(snapshot_dates, rng)
    borrowers = _generate_borrowers(config, rng)
    loans = _generate_loans(config, borrowers, snapshot_dates, rng)

    borrower_lookup = borrowers.set_index("borrower_id")
    macro_lookup = macro.set_index("snapshot_date")

    performance_rows: list[dict[str, object]] = []
    default_rows: list[dict[str, object]] = []
    recovery_rows: list[dict[str, object]] = []

    for loan in loans.itertuples(index=False):
        borrower = borrower_lookup.loc[loan.borrower_id]
        origination_idx = snapshot_dates.get_loc(pd.Timestamp(loan.origination_date))
        balance = float(loan.origination_balance)
        limit_amount = float(loan.limit_amount)
        collateral_value = float(loan.collateral_value_origination)
        rating_rank = int(loan.origination_rating_rank)
        prior_dpd_bucket = 0.0
        defaulted = False

        for step_offset, snapshot_date in enumerate(snapshot_dates[origination_idx:]):
            remaining_term_quarters = int(loan.term_quarters - step_offset)
            if remaining_term_quarters <= 0 or balance <= 250.0 or defaulted:
                break

            macro_row = macro_lookup.loc[pd.Timestamp(snapshot_date)]
            utilization = 1.0 if loan.segment not in REVOLVING_SEGMENTS else float(np.clip(rng.normal(0.58, 0.12), 0.15, 0.98))
            undrawn_amount = max(limit_amount - balance, 0.0)
            if loan.segment in REVOLVING_SEGMENTS:
                undrawn_amount = max(limit_amount * (1.0 - utilization), 0.0)
                balance = limit_amount - undrawn_amount

            dti = float(
                np.clip(
                    borrower["base_dti"]
                    + borrower["unemployment_sensitivity"] * 0.80 * (macro_row["unemployment_rate"] - 0.045)
                    + 0.18 * max(utilization - 0.60, 0.0)
                    + rng.normal(0.0, 0.03),
                    0.10,
                    1.20,
                )
            )
            if collateral_value > 0.0:
                collateral_value = max(
                    collateral_value * (1.0 + macro_row["house_price_growth"] + rng.normal(0.0, 0.01)),
                    balance * 0.45,
                )
            ltv = 1.0 if collateral_value <= 0.0 else float(np.clip(balance / collateral_value, 0.20, 2.50))

            delinquency_score = (
                -2.9
                + 0.55 * rating_rank
                + 1.35 * max(dti - 0.35, 0.0)
                + 0.95 * max(ltv - 0.85, 0.0)
                + 6.50 * (macro_row["unemployment_rate"] - 0.045)
                + 0.012 * prior_dpd_bucket
                + rng.normal(0.0, 0.25)
            )
            delinquency_probability = float(np.clip(_logistic(delinquency_score), 0.01, 0.85))
            if delinquency_probability < 0.12:
                days_past_due = 0
            elif delinquency_probability < 0.22:
                days_past_due = 5
            elif delinquency_probability < 0.35:
                days_past_due = 15
            elif delinquency_probability < 0.55:
                days_past_due = 30
            elif delinquency_probability < 0.72:
                days_past_due = 60
            elif delinquency_probability < 0.86:
                days_past_due = 90
            else:
                days_past_due = 120

            forborne = bool(
                days_past_due >= 30
                and rng.random() < float(np.clip(borrower["forbearance_propensity"] + 0.05 * (days_past_due >= 60), 0.02, 0.80))
            )
            default_probability = float(
                np.clip(
                    _logistic(
                        -6.2
                        + 0.60 * rating_rank
                        + 0.028 * days_past_due
                        + 1.50 * max(dti - 0.35, 0.0)
                        + 1.15 * max(ltv - 0.85, 0.0)
                        + 0.25 * int(forborne)
                        + 8.50 * (macro_row["unemployment_rate"] - 0.045)
                        + 3.00 * max(macro_row["policy_rate"] - 0.02, 0.0)
                        + rng.normal(0.0, 0.18)
                    ),
                    0.001,
                    0.65,
                )
            )
            default_next_period = int(rng.random() < default_probability)
            if days_past_due >= 90:
                rating_rank = min(rating_rank + 1, 5)
            elif days_past_due >= 30 and rng.random() < 0.35:
                rating_rank = min(rating_rank + 1, 5)
            elif days_past_due == 0 and rng.random() < 0.15:
                rating_rank = max(rating_rank - 1, 1)

            scheduled_principal = 0.0
            prepayment_flag = False
            if loan.segment in REVOLVING_SEGMENTS:
                scheduled_principal = 0.0
            else:
                scheduled_principal = max(balance / max(remaining_term_quarters, 1) * rng.uniform(0.80, 1.10), 0.0)
                prepayment_probability = float(np.clip(0.015 + 0.10 * max(loan.interest_rate - macro_row["policy_rate"], 0.0), 0.005, 0.18))
                prepayment_flag = bool(rng.random() < prepayment_probability)
                if prepayment_flag:
                    scheduled_principal = balance

            performance_rows.append(
                {
                    "loan_id": loan.loan_id,
                    "borrower_id": loan.borrower_id,
                    "segment": loan.segment,
                    "region": loan.region,
                    "snapshot_date": pd.Timestamp(snapshot_date),
                    "origination_date": pd.Timestamp(loan.origination_date),
                    "term_quarters": int(loan.term_quarters),
                    "quarters_on_book": int(step_offset),
                    "remaining_term_quarters": remaining_term_quarters,
                    "balance": round(balance, 2),
                    "limit_amount": round(limit_amount, 2),
                    "undrawn_amount": round(undrawn_amount, 2),
                    "utilization": round(utilization, 6),
                    "interest_rate": float(loan.interest_rate),
                    "collateral_value": round(collateral_value, 2),
                    "ltv": round(ltv, 6),
                    "dti": round(dti, 6),
                    "days_past_due": int(days_past_due),
                    "forborne": int(forborne),
                    "rating_rank": rating_rank,
                    "origination_rating_rank": int(loan.origination_rating_rank),
                    "origination_pd_anchor": float(loan.origination_pd_anchor),
                    "employment_stability": float(borrower["employment_stability"]),
                    "unemployment_sensitivity": float(borrower["unemployment_sensitivity"]),
                    "unemployment_rate": float(macro_row["unemployment_rate"]),
                    "policy_rate": float(macro_row["policy_rate"]),
                    "house_price_index": float(macro_row["house_price_index"]),
                    "house_price_growth": float(macro_row["house_price_growth"]),
                    "gdp_growth": float(macro_row["gdp_growth"]),
                    "default_next_period": default_next_period,
                    "prepayment_flag": int(prepayment_flag),
                }
            )

            prior_dpd_bucket = float(days_past_due)
            if default_next_period:
                default_date = pd.Timestamp(snapshot_date) + pd.offsets.QuarterEnd(1)
                ead = balance if loan.segment not in REVOLVING_SEGMENTS else balance + 0.75 * undrawn_amount
                recovery_rate = float(
                    np.clip(
                        0.75 * (collateral_value / max(ead, 1.0)) if collateral_value > 0.0 else 0.12,
                        0.05,
                        0.85,
                    )
                )
                lgd = 1.0 - recovery_rate
                default_rows.append(
                    {
                        "loan_id": loan.loan_id,
                        "default_date": default_date,
                        "segment": loan.segment,
                        "ead_at_default": round(ead, 2),
                        "balance_at_default": round(balance, 2),
                        "undrawn_at_default": round(undrawn_amount, 2),
                        "collateral_value_at_default": round(collateral_value, 2),
                        "recovery_rate_assumption": round(recovery_rate, 6),
                        "lgd_realised_assumption": round(lgd, 6),
                    }
                )
                recovery_total = ead * recovery_rate
                for lag_quarter, weight in enumerate((0.45, 0.30, 0.15, 0.10), start=1):
                    recovery_rows.append(
                        {
                            "loan_id": loan.loan_id,
                            "recovery_date": default_date + pd.offsets.QuarterEnd(lag_quarter),
                            "recovery_cashflow": round(recovery_total * weight, 2),
                            "months_since_default": lag_quarter * 3,
                            "recovery_weight": weight,
                        }
                    )
                defaulted = True
                break

            if loan.segment not in REVOLVING_SEGMENTS:
                balance = max(balance - scheduled_principal, 0.0)

    performance = pd.DataFrame.from_records(performance_rows).sort_values(["snapshot_date", "loan_id"]).reset_index(drop=True)
    defaults = pd.DataFrame.from_records(default_rows).sort_values(["default_date", "loan_id"]).reset_index(drop=True)
    recoveries = pd.DataFrame.from_records(recovery_rows).sort_values(["recovery_date", "loan_id"]).reset_index(drop=True)

    snapshots = (
        performance.groupby("snapshot_date", as_index=False)
        .agg(
            active_loans=("loan_id", "nunique"),
            total_balance=("balance", "sum"),
            avg_dpd=("days_past_due", "mean"),
            avg_dti=("dti", "mean"),
            avg_ltv=("ltv", "mean"),
        )
        .merge(
            defaults.groupby("default_date", as_index=False)
            .agg(default_count=("loan_id", "count"), default_ead=("ead_at_default", "sum"))
            .rename(columns={"default_date": "snapshot_date"}),
            on="snapshot_date",
            how="left",
        )
        .fillna({"default_count": 0, "default_ead": 0.0})
    )
    snapshots["default_rate"] = snapshots["default_count"] / snapshots["active_loans"].clip(lower=1)
    snapshots["config"] = str(asdict(config))

    return PortfolioDataset(
        borrowers=borrowers,
        loans=loans,
        performance=performance,
        defaults=defaults,
        recoveries=recoveries,
        macro=macro,
        snapshots=snapshots,
    )
