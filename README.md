# Credit Risk & IFRS 9 Lab

Synthetic-first credit risk modelling lab for applying econometric methods to PD, migration, IFRS 9 ECL, monitoring, and validation workflows.

This repository now demonstrates a full synthetic-first workflow:

- multi-period retail loan portfolio generation,
- pooled-logit survival-based PD estimation,
- IFRS 9 stage 1 / 2 / 3 logic,
- scenario-weighted ECL with interpretable LGD/EAD,
- data-quality and drift monitoring,
- a split-friendly validation subrepo for backtesting and governance.

The validation toolkit lives in [`model_validation_pack`](model_validation_pack) and is designed so it can later become its own repository with minimal cleanup.

## Technical Aim

The repository is organised as a credit-risk lab rather than one single maximal model. Each econometric component can be tested, compared, and switched on only when the data support it:

- binary outcome and limited-dependent-variable models for default and delinquency,
- panel and duration tools for loan-quarter behaviour,
- survival-logit PD as the primary transparent baseline,
- Markov migration as a challenger and stage-migration explanation,
- continuous-time and continuous-state modules when event timing or latent-state modelling is justified,
- validation and monitoring outputs that separate performance, stability, drift, and governance evidence.

## Repo Layout

```text
.
├── README.md
├── data/
├── docs/
│   ├── econometric_extensions.md
│   ├── dirichlet_markov_credit_bridge.md
│   ├── methodology_notes.md
│   └── regulatory_scope.md
├── examples/
│   └── run_lab_demo.py
├── model_validation_pack/
├── notebooks/
│   ├── credit_risk_ifrs9_walkthrough.ipynb
│   └── econometric_toolbox_walkthrough.ipynb
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

| Econometrics skill | Credit-risk use | Current location | Planned extension |
| --- | --- | --- | --- |
| Logistic / limited dependent variable models | PD and delinquency modelling | [`survival.py`](src/credit_risk_lab/survival.py), [`challenger.py`](src/credit_risk_lab/challenger.py) | [`src/credit_risk_lab/econometrics/limited_dep.py`](src/credit_risk_lab/econometrics/limited_dep.py) |
| Panel data | Loan-quarter performance panels | [`portfolio.py`](src/credit_risk_lab/portfolio.py) | [`src/credit_risk_lab/econometrics/panel.py`](src/credit_risk_lab/econometrics/panel.py) |
| Survival / duration analysis | Time-to-default, prepayment, cure | [`survival.py`](src/credit_risk_lab/survival.py) | [`src/credit_risk_lab/econometrics/duration.py`](src/credit_risk_lab/econometrics/duration.py) |
| Time series | Macro paths and scenario overlays | [`portfolio.py`](src/credit_risk_lab/portfolio.py), [`ecl.py`](src/credit_risk_lab/ecl.py) | [`src/credit_risk_lab/econometrics/macro.py`](src/credit_risk_lab/econometrics/macro.py) |
| Non-constant hazard term structures | Forward PD paths and lifetime PD without flat hazard | current constant-hazard simplification in [`survival.py`](src/credit_risk_lab/survival.py) | [`src/credit_risk_lab/econometrics/forward_hazard.py`](src/credit_risk_lab/econometrics/forward_hazard.py) |
| Markov / state-transition models | Delinquency, stage migration, cure, prepay | migration summaries in [`ecl.py`](src/credit_risk_lab/ecl.py), finite-state utilities in [`markov.py`](src/credit_risk_lab/econometrics/markov.py) | [`src/credit_risk_lab/econometrics/markov.py`](src/credit_risk_lab/econometrics/markov.py) |
| Continuous-time default modelling | Intensities, counting processes, default times between observation dates | [`continuous_time.py`](src/credit_risk_lab/econometrics/continuous_time.py) | exact event-time intensity models |
| Continuous-state credit quality | Diffusion, jump, killing/default, Beurling-Deny diagnostics | [`continuous_state.py`](src/credit_risk_lab/econometrics/continuous_state.py) | fitted latent-state models |
| Calibration | Aligning predicted PDs to observed default rates | [`calibration.py`](src/credit_risk_lab/econometrics/calibration.py), [`model_validation_pack/backtest.py`](model_validation_pack/src/model_validation_pack/backtest.py) | extend with smoother calibration models |
| Out-of-sample validation | Backtesting, challenger comparison, drift | [`model_validation_pack/`](model_validation_pack) | keep extending `model_validation_pack/` |
| Causal / structural thinking | Endogeneity, policy effects, macro transmission | [`causal.py`](src/credit_risk_lab/econometrics/causal.py) | extend with explicit identification designs |
| Measurement error | Missingness, reporting bias, data quality | [`monitoring.py`](src/credit_risk_lab/monitoring.py), [`measurement_error.py`](src/credit_risk_lab/econometrics/measurement_error.py) | extend with formal error models |
| Heterogeneity | Segment effects, borrower/product differences | [`heterogeneity.py`](src/credit_risk_lab/econometrics/heterogeneity.py) | extend with hierarchical models |
| Model selection | Feature choice and challenger models | [`model_selection.py`](src/credit_risk_lab/econometrics/model_selection.py), validation pack | extend with cross-validation workflows |

The fuller plan is in [docs/econometric_extensions.md](docs/econometric_extensions.md). The Markov-chain extension has a separate functional-analysis bridge note in [docs/dirichlet_markov_credit_bridge.md](docs/dirichlet_markov_credit_bridge.md), to make use of my knowledge from a PhD course I took.

## Econometrics-Heavy Extension Slots

The repo is built so it can grow into heavier econometric work later

### 1. Remove the constant-hazard assumption

Implemented location:
- [`src/credit_risk_lab/econometrics/forward_hazard.py`](src/credit_risk_lab/econometrics/forward_hazard.py)

Implemented first layer:
- build forward loan-quarter panels,
- project future borrower, balance, and macro features,
- score a hazard path `h1, h2, ..., hn`,
- compute lifetime PD from `1 - Π(1 - h_t)` instead of `1 - (1 - h)^n`.

### 2. Put macro variables directly into the PD part instead of ECL later

Implemented locations:
- [`src/credit_risk_lab/econometrics/macro.py`](src/credit_risk_lab/econometrics/macro.py)
- [`src/credit_risk_lab/econometrics/forward_hazard.py`](src/credit_risk_lab/econometrics/forward_hazard.py)

Implemented first layer:
- fit transparent AR(1) macro paths,
- create baseline/downside/upside macro scenario paths,
- feed baseline/downside/upside macro paths through the PD model,
- keep the intended design clear: overlays mainly for LGD and EAD,
- avoid double-counting macro effects on PD.

### 3. Add a Markov "challenger" model for migration logic

Implemented location:
- [`src/credit_risk_lab/econometrics/markov.py`](src/credit_risk_lab/econometrics/markov.py)

Implemented layer:
- build observed one-step transition panels,
- estimate a row-stochastic transition matrix,
- treat default and prepay/maturity as absorbing cemetery-style states,
- compute multi-period transition matrices,
- approximate a quarterly generator,
- compute a matrix-log generator candidate with embeddability diagnostics,
- compute absorption probabilities and expected time to absorption,
- compute a Dirichlet-style transition energy diagnostic for state scores,
- estimate grouped transition matrices for segment or macro-regime style conditioning,
- estimate covariate-dependent transition probabilities,
- produce macro-regime scenario transition matrices,
- aggregate delinquency migration to stage migration,
- compute Markov-implied default probabilities for comparison against survival-logit PDs,
- run score smoothness, reversibility, and Dirichlet regularisation diagnostics.

### 4. Add continuous-time default modelling

Implemented location:
- [`src/credit_risk_lab/econometrics/continuous_time.py`](src/credit_risk_lab/econometrics/continuous_time.py)

Implemented first layer:
- estimate piecewise-constant default intensities,
- estimate default intensities from exact start/end event intervals,
- estimate CTMC generators from state durations when transition times are observed,
- build default counting-process diagnostic tables,
- compute compensated default processes `N_t - A_t`,
- convert intensity paths into continuous-time survival probabilities.

### 4b. Add continuous-state credit-quality modelling

Implemented location:
- [`src/credit_risk_lab/econometrics/continuous_state.py`](src/credit_risk_lab/econometrics/continuous_state.py)

Implemented layer:
- build a latent credit-quality grid with a default boundary,
- construct a finite-difference continuous-time generator with local diffusion, jumps, and killing/default,
- compute default probabilities from `exp(tQ)`,
- decompose credit-state energy into local, jump, and killing components,
- run finite-grid proxies for capacity, polar-like states, Cheeger energy, and regularity diagnostics.

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

## Dependencies

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
- Regulatory default definition:
  [EBA Guidelines on the application of the definition of default](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-application-definition)
  and [CRR Article 178](https://www.eba.europa.eu/regulation-and-policy/single-rulebook/interactive-single-rulebook/16022)
- ECB / Dutch supervisory context for internal models:
  [ECB Internal Models](https://www.bankingsupervision.europa.eu/activities/internal_models/html/index.en.html)
- Prudential capital background only:
  [EBA CRR3/CRD6 dashboard](https://www.eba.europa.eu/risk-and-data-analysis/risk-analysis/risk-monitoring/crr3-crd6-dashboard)
  and [BIS Basel III final reforms](https://www.bis.org/bcbs/publ/d424.htm)
- Markov migration and functional-analysis bridge:
  [Fukushima, Oshima, and Takeda (2011)](https://www.degruyterbrill.com/document/doi/10.1515/9783110218091/html),
  [Ma and Roeckner (1992)](https://link.springer.com/book/10.1007/978-3-642-77739-4),
  [Jarrow-Lando-Turnbull (1997)](https://academic.oup.com/rfs/article/10/2/481/1589160),
  and [Lando-Skodeberg (2002)](https://www.sciencedirect.com/science/article/pii/S037842660100228X).
- Continuous-time and structural/reduced-form credit risk:
  [Black-Cox (1976)](https://www.jstor.org/stable/2326758),
  [Lando (1998)](https://doi.org/10.1017/S0269964898173055),
  and [Duffie-Singleton (1999)](https://academic.oup.com/rfs/article-abstract/12/4/687/1599653).
- Matrix-log generator and embeddability diagnostics:
  [Israel-Rosenthal-Wei (2001)](https://doi.org/10.1080/713665550)
- Probability forecast backtesting:
  [Brier (1950)](https://cir.nii.ac.jp/crid/1361981468554183168)
- Optional Sinkhorn divergence:
  [Cuturi (2013)](https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport)

The regulatory boundary is documented in [docs/regulatory_scope.md](docs/regulatory_scope.md). The short version is: IFRS 9, EBA Loan Origination and Monitoring, Definition of Default, and model-governance references are in scope; Basel final reforms and CRR3/CRD6 are background capital context, not implemented as a capital calculator.
