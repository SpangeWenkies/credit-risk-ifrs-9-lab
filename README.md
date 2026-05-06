# Credit Risk & IFRS 9 Lab

Portfolio project for junior credit risk modellers and junior actuaries, built to apply econometric training to a credit-risk domain.

This repository now demonstrates a full synthetic-first workflow:

- multi-period retail loan portfolio generation,
- pooled-logit survival-based PD estimation,
- IFRS 9 stage 1 / 2 / 3 logic,
- scenario-weighted ECL with interpretable LGD/EAD,
- data-quality and drift monitoring,
- a split-friendly validation subrepo for backtesting and governance.

The validation toolkit lives in [`model_validation_pack`](model_validation_pack) and is designed so it can later become its own repository with minimal cleanup.

## Why This Repo Exists

The intention of this repository is not only to show interest in credit risk. It is to show that I already understand the econometric structure behind many credit-risk problems and that I am using this project to transfer that advantage into a concrete finance domain.

The core story is:

> I understand binary outcomes, panel data, survival logic, calibration, and out-of-sample validation. What I am adding here is the domain layer: how those econometric tools become IFRS 9 provisions, monitoring packs, validation evidence, and credit-risk decision support.

So this repo should be read as:

- econometrics first,
- credit risk as the application domain,
- IFRS 9 and validation as the institutional layer.

## Repo Layout

```text
.
├── README.md
├── data/
├── docs/
│   ├── econometric_extensions.md
│   ├── dirichlet_markov_credit_bridge.md
│   ├── methodology_notes.md
│   └── portfolio_story.md
├── examples/
│   └── run_lab_demo.py
├── model_validation_pack/
├── notebooks/
│   └── credit_risk_ifrs9_walkthrough.ipynb
├── reports/
│   └── credit_risk_lab_walkthrough.md
├── src/credit_risk_lab/
│   ├── econometrics/
│   ├── ecl.py
│   ├── monitoring.py
│   ├── portfolio.py
│   └── survival.py
└── tests/
```

## Econometrics To Credit Risk

The current repo already maps standard econometric ideas into credit-risk tasks:

| Econometrics skill | Credit-risk use | Current location | Planned extension location |
| --- | --- | --- | --- |
| Logistic / limited dependent variable models | PD and delinquency modelling | [`survival.py`](src/credit_risk_lab/survival.py), [`challenger.py`](src/credit_risk_lab/challenger.py) | [`src/credit_risk_lab/econometrics/limited_dep.py`](src/credit_risk_lab/econometrics/limited_dep.py) |
| Panel data | Loan-quarter performance panels | [`portfolio.py`](src/credit_risk_lab/portfolio.py) | [`src/credit_risk_lab/econometrics/panel.py`](src/credit_risk_lab/econometrics/panel.py) |
| Survival / duration analysis | Time-to-default, prepayment, cure | [`survival.py`](src/credit_risk_lab/survival.py) | [`src/credit_risk_lab/econometrics/duration.py`](src/credit_risk_lab/econometrics/duration.py) |
| Time series | Macro paths and scenario overlays | [`portfolio.py`](src/credit_risk_lab/portfolio.py), [`ecl.py`](src/credit_risk_lab/ecl.py) | [`src/credit_risk_lab/econometrics/macro.py`](src/credit_risk_lab/econometrics/macro.py) |
| Non-constant hazard term structures | Forward PD paths and lifetime PD without flat hazard | current constant-hazard simplification in [`survival.py`](src/credit_risk_lab/survival.py) | [`src/credit_risk_lab/econometrics/forward_hazard.py`](src/credit_risk_lab/econometrics/forward_hazard.py) |
| Markov / state-transition models | Delinquency, stage migration, cure, prepay | migration summaries in [`ecl.py`](src/credit_risk_lab/ecl.py), finite-state utilities in [`markov.py`](src/credit_risk_lab/econometrics/markov.py) | [`src/credit_risk_lab/econometrics/markov.py`](src/credit_risk_lab/econometrics/markov.py) |
| Continuous-time default modelling | Intensities, counting processes, default times between observation dates | [`continuous_time.py`](src/credit_risk_lab/econometrics/continuous_time.py) | extend with exact event-time models |
| Calibration | Aligning predicted PDs to observed default rates | [`calibration.py`](src/credit_risk_lab/econometrics/calibration.py), [`model_validation_pack/backtest.py`](model_validation_pack/src/model_validation_pack/backtest.py) | extend with smoother calibration models |
| Out-of-sample validation | Backtesting, challenger comparison, drift | [`model_validation_pack/`](model_validation_pack) | keep extending `model_validation_pack/` |
| Causal / structural thinking | Endogeneity, policy effects, macro transmission | [`causal.py`](src/credit_risk_lab/econometrics/causal.py) | extend with explicit identification designs |
| Measurement error | Missingness, reporting bias, data quality | [`monitoring.py`](src/credit_risk_lab/monitoring.py), [`measurement_error.py`](src/credit_risk_lab/econometrics/measurement_error.py) | extend with formal error models |
| Heterogeneity | Segment effects, borrower/product differences | [`heterogeneity.py`](src/credit_risk_lab/econometrics/heterogeneity.py) | extend with hierarchical models |
| Model selection | Feature choice and challenger models | [`model_selection.py`](src/credit_risk_lab/econometrics/model_selection.py), validation pack | extend with cross-validation workflows |

The fuller roadmap is in [docs/econometric_extensions.md](docs/econometric_extensions.md). The Markov-chain extension has a separate functional-analysis bridge note in [docs/dirichlet_markov_credit_bridge.md](docs/dirichlet_markov_credit_bridge.md).

## Econometrics-Heavy Extension Slots

The repo is intentionally built so it can grow into heavier econometric work later without changing its core story.

### 1. Remove the constant-hazard assumption

Implemented location:
- [`src/credit_risk_lab/econometrics/forward_hazard.py`](src/credit_risk_lab/econometrics/forward_hazard.py)

Implemented first layer:
- build forward loan-quarter panels,
- project future borrower, balance, and macro features,
- score a hazard path `h1, h2, ..., hn`,
- compute lifetime PD from `1 - Π(1 - h_t)` instead of `1 - (1 - h)^n`.

### 2. Put macro variables directly into the PD channel

Implemented locations:
- [`src/credit_risk_lab/econometrics/macro.py`](src/credit_risk_lab/econometrics/macro.py)
- [`src/credit_risk_lab/econometrics/forward_hazard.py`](src/credit_risk_lab/econometrics/forward_hazard.py)

Implemented first layer:
- fit transparent AR(1) macro paths,
- create baseline/downside/upside macro scenario paths,
- feed baseline/downside/upside macro paths through the PD model,
- keep the intended design clear: overlays mainly for LGD and EAD,
- avoid double-counting macro effects on PD.

### 3. Add a Markov challenger for migration logic

Current state:
- the repo already produces stage summaries, migration matrices, and provision roll-forwards,
- `src/credit_risk_lab/econometrics/markov.py` now adds the first finite-state transition model scaffold and utilities.

Planned location:
- [`src/credit_risk_lab/econometrics/markov.py`](src/credit_risk_lab/econometrics/markov.py)

Implemented first layer:
- build observed one-step transition panels,
- estimate a row-stochastic transition matrix,
- treat default and prepay/maturity as absorbing cemetery-style states,
- compute multi-period transition matrices,
- approximate a quarterly generator,
- compute absorption probabilities and expected time to absorption,
- compute a Dirichlet-style transition energy diagnostic for state scores,
- estimate grouped transition matrices for segment or macro-regime style conditioning,
- compute Markov-implied default probabilities for comparison against survival-logit PDs.

Intended next upgrade:
- estimate transition probabilities across `current`, `1-29 DPD`, `30-89 DPD`, `90+ / default`, `cure`, and `prepay / mature`,
- use a simple transition matrix as a transparent challenger,
- later extend to covariate-dependent transition models,
- compare Markov-implied default probabilities with survival-logit PDs.

### 4. Add continuous-time default modelling

Implemented location:
- [`src/credit_risk_lab/econometrics/continuous_time.py`](src/credit_risk_lab/econometrics/continuous_time.py)

Implemented first layer:
- estimate piecewise-constant default intensities,
- build default counting-process diagnostic tables,
- compute compensated default processes `N_t - A_t`,
- convert intensity paths into continuous-time survival probabilities.

### 5. Econometric diagnostics pack

Implemented location:
- [`src/credit_risk_lab/econometrics/`](src/credit_risk_lab/econometrics/)

Implemented first layer:
- calibration tables, Brier score, and intercept-shift recalibration,
- panel vintage, within-transform, and cohort performance tables,
- duration hazard and competing-risk cumulative incidence tables,
- binary logit helpers for limited dependent variable models,
- segment performance, segment logits, and coefficient stability checks,
- measurement-error missingness and noise sensitivity diagnostics,
- causal variable audits and policy-shock sensitivity without overclaiming causality,
- AIC/BIC/Brier challenger model selection.

## Public Root APIs

- `generate_portfolio_timeseries(config) -> PortfolioDataset`
- `fit_survival_pd_model(panel, feature_spec) -> SurvivalPDModel`
- `score_portfolio(dataset, model, as_of_date) -> PDScoreFrame`
- `run_ifrs9_pipeline(dataset, pd_scores, lgd_model, ead_model, scenarios) -> ECLReport`
- `run_monitoring(reference_snapshot, current_snapshot, reference_scores, current_scores) -> MonitoringReport`

## Scientific Stack

The root project depends on:

- `numpy`
- `pandas`
- `scipy`
- `statsmodels`
- `matplotlib`
- `pytest`
- `jupyter`

## Quick Start

```bash
.venv/bin/python examples/run_lab_demo.py
.venv/bin/python model_validation_pack/examples/run_validation_demo.py
.venv/bin/pytest -q tests/test_credit_risk_lab.py
.venv/bin/pytest -q tests/test_econometric_extensions.py
.venv/bin/pytest -q model_validation_pack/tests/test_model_validation_pack.py
```

## Core Methods And References

- IFRS 9 impairment logic:
  [IFRS 9 Financial Instruments](https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf)
- Discrete-time survival modelling:
  [Singer & Willett (1993)](https://journals.sagepub.com/doi/10.3102/10769986018002155)
- EU banking model governance:
  [EBA Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/model-validation/guidelines-pd-estimation-lgd)
- EU banking monitoring and governance:
  [EBA Guidelines on loan origination and monitoring](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-loan-origination-and-monitoring)
- ECB / Dutch supervisory context for internal models:
  [ECB Internal Models](https://www.bankingsupervision.europa.eu/activities/internal_models/html/index.en.html)
- Dutch / actuarial prudential context:
  [DNB Solvency II general notes](https://www.dnb.nl/en/sector-information/open-book-supervision/open-book-supervision-sectors/insurers/law-and-regulations-insurers/solvency-ii-general-notes/)
  and [EIOPA Solvency II](https://www.eiopa.europa.eu/browse/regulation-and-policy/solvency-ii_en)
- Markov migration and functional-analysis bridge:
  [Fukushima, Oshima, and Takeda (2011)](https://www.degruyterbrill.com/document/doi/10.1515/9783110218091/html),
  [Ma and Roeckner (1992)](https://link.springer.com/book/10.1007/978-3-642-77739-4),
  [Jarrow-Lando-Turnbull (1997)](https://academic.oup.com/rfs/article/10/2/481/1589160),
  and [Lando-Skodeberg (2002)](https://www.sciencedirect.com/science/article/pii/S037842660100228X).
- Probability forecast backtesting:
  [Brier (1950)](https://cir.nii.ac.jp/crid/1361981468554183168)
- Optional Sinkhorn divergence:
  [Cuturi (2013)](https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport)

## Talking Points

- Why this is best understood as an econometrics portfolio project applied to credit risk, not only a credit-risk demo.
- Why a synthetic multi-period panel is enough to demonstrate survival modelling and validation logic.
- Why IFRS 9 staging and provision roll-forwards make the repo more relevant than a generic classification project.
- Why Dutch/EU banking roles care more about `IFRS 9 + EBA/ECB/DNB` than U.S. SR guidance.
- Why the actuarial angle should be framed through `Solvency II / DNB / EIOPA` rather than U.S. banking model-risk language.
