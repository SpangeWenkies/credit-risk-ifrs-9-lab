# Econometric Extensions Map

This document makes explicit where heavier econometric work fits into the repository.

## Positioning

The repo should communicate this message clearly:

> The statistical structure is already familiar to me from econometrics. What I am doing here is using that toolkit in a credit-risk setting, where the outputs must become PD, LGD, EAD, IFRS 9, monitoring, and validation artefacts.

That means the repo is not only:

- "I want to learn credit risk."

It is also:

- "I already have econometric tools and I am applying them to credit risk."

## Direct Mapping

| Econometrics theme | Credit-risk use | Current implementation | Best extension slot |
| --- | --- | --- | --- |
| Logistic regression | PD / default modelling | [`src/credit_risk_lab/survival.py`](../src/credit_risk_lab/survival.py) | `src/credit_risk_lab/econometrics/limited_dep.py` |
| Panel data | Loan-quarter performance panels | [`src/credit_risk_lab/portfolio.py`](../src/credit_risk_lab/portfolio.py) | `src/credit_risk_lab/econometrics/panel.py` |
| Survival / duration | Default timing, prepayment, cure | [`src/credit_risk_lab/survival.py`](../src/credit_risk_lab/survival.py) | `src/credit_risk_lab/econometrics/duration.py` |
| Time series | Macro scenarios, overlays, forecasting | [`src/credit_risk_lab/portfolio.py`](../src/credit_risk_lab/portfolio.py), [`src/credit_risk_lab/ecl.py`](../src/credit_risk_lab/ecl.py) | `src/credit_risk_lab/econometrics/macro.py` |
| Forward hazard paths | Lifetime PD without flat hazard | constant-hazard conversion in [`src/credit_risk_lab/survival.py`](../src/credit_risk_lab/survival.py) | `src/credit_risk_lab/econometrics/forward_hazard.py` |
| Markov chains | State migration, cure, prepay, stage movement | migration summaries in [`src/credit_risk_lab/ecl.py`](../src/credit_risk_lab/ecl.py), finite-state utilities in [`src/credit_risk_lab/econometrics/markov.py`](../src/credit_risk_lab/econometrics/markov.py) | `src/credit_risk_lab/econometrics/markov.py` |
| Continuous-time default modelling | Intensities and event timing beyond quarter-end panels | [`src/credit_risk_lab/econometrics/continuous_time.py`](../src/credit_risk_lab/econometrics/continuous_time.py) | exact event-time intensity models |
| Limited dependent variable models | Delinquency, cure, prepayment | [`src/credit_risk_lab/econometrics/limited_dep.py`](../src/credit_risk_lab/econometrics/limited_dep.py) | ordered / multinomial variants |
| Model selection | Feature choice, challenger models, regularisation | [`src/credit_risk_lab/econometrics/model_selection.py`](../src/credit_risk_lab/econometrics/model_selection.py) | cross-validation workflows |
| Calibration | Aligning PDs to observed defaults | [`src/credit_risk_lab/econometrics/calibration.py`](../src/credit_risk_lab/econometrics/calibration.py), [`model_validation_pack/src/model_validation_pack/backtest.py`](../model_validation_pack/src/model_validation_pack/backtest.py) | smoother calibration models |
| Out-of-sample testing | Backtesting PD/LGD/EAD models | `model_validation_pack/` | keep extending `model_validation_pack/` |
| Causal thinking | Predictive vs endogenous vs policy-driven variables | [`src/credit_risk_lab/econometrics/causal.py`](../src/credit_risk_lab/econometrics/causal.py) | explicit identification designs |
| Stress testing | Scenario design, macro overlays, sensitivities | [`src/credit_risk_lab/ecl.py`](../src/credit_risk_lab/ecl.py), [`src/credit_risk_lab/econometrics/macro.py`](../src/credit_risk_lab/econometrics/macro.py) | richer macro satellite models |
| Measurement error | Data quality, missingness, reporting bias | [`src/credit_risk_lab/monitoring.py`](../src/credit_risk_lab/monitoring.py), [`src/credit_risk_lab/econometrics/measurement_error.py`](../src/credit_risk_lab/econometrics/measurement_error.py) | formal error models |
| Heterogeneity | Segment-specific models, fixed/random effects | [`src/credit_risk_lab/econometrics/heterogeneity.py`](../src/credit_risk_lab/econometrics/heterogeneity.py) | hierarchical/random-effects models |

## Best Next Econometric Upgrades

### 1. Panel-data upgrade

Best location:
- `src/credit_risk_lab/econometrics/panel.py`

Implemented first layer:
- borrower fixed effects,
- cohort effects,
- panel-summary tables for migration and attrition.

Still open:
- macro-by-segment interaction terms,
- nonlinear fixed-effects PD estimators.

### 2. Duration-model upgrade

Best location:
- `src/credit_risk_lab/econometrics/duration.py`

Implemented first layer:
- discrete-time hazard variants,
- competing risks for default vs prepayment,
- duration dependence terms,
- baseline hazard visualisation.

Still open:
- cure / relapse logic,
- competing-risks regression.

### 2b. Forward hazard / lifetime-PD upgrade

Best location:
- `src/credit_risk_lab/econometrics/forward_hazard.py`

Why it matters:
- the current repo explicitly assumes a constant quarterly hazard when turning a one-period PD into `12m PD` and `lifetime PD`,
- that is fine for a first version, but it should be easy to replace with a projected hazard path later.

Implemented first layer:
- future loan-quarter panel generation,
- amortisation-aware forward balance projection,
- scenario-driven future LTV/DTI paths,
- macro-sensitive future hazards,
- cumulative PD from `1 - Π(1 - h_t)`.

### 3. Calibration upgrade

Best location:
- `src/credit_risk_lab/econometrics/calibration.py`

Implemented first layer:
- calibration curves,
- Platt / intercept-shift style recalibration,
- long-run average default rate alignment,
- Brier score support.

Still open:
- segment-wise calibration drift,
- isotonic and spline calibration.

### 4. Causal / identification upgrade

Best location:
- `src/credit_risk_lab/econometrics/causal.py`

Implemented first layer:
- endogeneity notes for macro covariates,
- policy shock thought experiments,
- treatment of predictive versus structural interpretation.

Still open:
- omitted-variable experiments,
- actual identification designs.

### 5. Heterogeneity upgrade

Best location:
- `src/credit_risk_lab/econometrics/heterogeneity.py`

Implemented first layer:
- segment-specific PD models,
- borrower-type splits,
- product-specific coefficients and stability checks.

Still open:
- random-effects style experiments,
- hierarchical models.

### 6. Markov-chain challenger and migration upgrade

Best location:
- `src/credit_risk_lab/econometrics/markov.py`

Implemented first layer:
- transition panel construction from loan-quarter performance data,
- simple `P(next_state | current_state)` transition matrix,
- absorbing terminal states for default and prepay/maturity,
- multi-period transition matrix powers,
- quarterly generator approximation,
- absorption probabilities and expected time to absorption,
- Dirichlet-style transition energy diagnostic for functions defined on credit states,
- grouped transition matrices for covariate or regime splits,
- Markov-implied default PDs for challenger comparison.

Good next additions:
- transition matrix for a richer state space: `current`, `1-29 DPD`, `30-89 DPD`, `90+ / default`, `cure`, `prepay / mature`,
- full multinomial covariate-dependent transition probabilities,
- stage-migration explanation for IFRS 9,
- comparison of Markov-implied default risk versus survival-logit PDs,
- macro-regime-specific transition matrices.

Recommended role:
- keep the survival-logit as the primary PD model,
- use the Markov model as a challenger, migration model, or educational comparison,
- evolve it later into a multi-state survival style extension.

Interview line:

> The survival logit is my primary PD model. A Markov model is a natural challenger because credit deterioration is state-based and it explains stage migration more directly than a binary default model.

Functional-analysis bridge:
- the transition matrix is the finite-state kernel,
- matrix powers are the semigroup,
- the generator approximation is the bridge to continuous-time intensities,
- default and prepay/maturity are cemetery-style absorbing states,
- the absorbing-chain fundamental matrix is the finite-state Green/resolvent analogue,
- transition energy measures how rough a score or stage function is over the migration graph.

Detailed note:
- [`docs/dirichlet_markov_credit_bridge.md`](dirichlet_markov_credit_bridge.md)

### 7. Continuous-time default and counting-process upgrade

Best location:
- `src/credit_risk_lab/econometrics/continuous_time.py`

Implemented first layer:
- reduced-form intensity modelling,
- compensated default counting process table,
- stochastic-calculus notes translating econometric training into credit-risk language,
- bridge between discrete-time panel approximations and continuous-time default intensities.

Still open:
- exact default times between reporting dates,
- Cox-process or counting-process regression.

Why it belongs here:
- this repo is quarterly and discrete-time now because that is the clearest version for a portfolio project,
- but the scaffold should already show that it can grow into continuous-time default modelling when richer event timing is available.

### 8. Macro-sensitive PD path upgrade

Best locations:
- `src/credit_risk_lab/econometrics/macro.py`
- `src/credit_risk_lab/econometrics/forward_hazard.py`

Implemented first layer:
- AR(1) macro forecasting layer,
- baseline/downside/upside macro path construction,
- score baseline/downside/upside macro paths directly through the PD model,
- keep scenario overlays mainly for LGD and EAD,
- use PD overlays only as explicitly labelled management overlays if needed.

Still open:
- VAR or richer macro satellite model,
- direct refit of the primary PD model with macro variables in the default feature spec.

Recommended design:
- one primary macro channel for PD,
- avoid putting unemployment into the PD model and then multiplying PD again in ECL for the same effect unless that second layer is clearly a judgemental overlay.

## What To Say In Interviews

- "This repo is where I translate econometrics into credit risk."
- "The panel, binary-outcome, calibration, and survival logic are econometric strengths I already had."
- "The value of the project is showing how those tools become provisions, stress tests, and validation evidence in a regulated credit-risk setting."
- "The current repo starts with a transparent discrete-time hazard model, but it is intentionally scaffolded for forward hazard paths, macro-sensitive PD term structures, Markov migration models, and continuous-time default modelling."
