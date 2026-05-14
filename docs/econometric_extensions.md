# Econometric Extensions Map

This document makes explicit where econometric work fits into the repository.

## Positioning

The repo should communicate this message clearly:

> The statistical structure is already familiar to me from econometrics. What I am doing here is using that toolkit in a credit-risk setting, where the outputs must become PD, LGD, EAD, IFRS 9, monitoring, and validation artefacts.

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
| Continuous-state credit quality | Latent credit-quality diffusion, jumps, and killing/default | [`src/credit_risk_lab/econometrics/continuous_state.py`](../src/credit_risk_lab/econometrics/continuous_state.py) | fitted latent-state models |
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
- macro-by-segment interaction terms,
- polynomial feature expansion for nonlinear-effect checks,
- poolability diagnostics before choosing pooled, segmented, or hierarchical specifications.

### 2. Duration-model upgrade

Best location:
- `src/credit_risk_lab/econometrics/duration.py`

Implemented first layer:
- discrete-time hazard variants,
- competing risks for default vs prepayment,
- duration dependence terms,
- baseline hazard visualisation.
- cure / relapse logic,
- cause-specific competing-risk logits.

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
- segment-wise calibration drift,
- isotonic calibration.

### 4. Causal / identification upgrade

Best location:
- `src/credit_risk_lab/econometrics/causal.py`

Implemented first layer:
- endogeneity notes for macro covariates,
- policy shock thought experiments,
- treatment of predictive versus structural interpretation.
- omitted-variable experiments,
- coefficient stability sensitivity between restricted and full specifications.

### 5. Heterogeneity upgrade

Best location:
- `src/credit_risk_lab/econometrics/heterogeneity.py`

Implemented first layer:
- segment-specific PD models,
- borrower-type splits,
- product-specific coefficients and stability checks,
- empirical-Bayes segment shrinkage as a lightweight hierarchical diagnostic.

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

Implemented extended layer:
- covariate-dependent one-vs-rest transition logits with empirical fallback,
- row-level transition probability scoring,
- comparison of Markov-implied default risk versus survival-logit PDs,
- macro-regime-specific transition matrices,
- baseline/downside/upside scenario migration matrices,
- stage-migration aggregation,
- matrix-log generator diagnostics and generator repair helpers,
- reversibility diagnostics,
- score smoothness decomposition,
- graph-Laplacian score regularisation.

Recommended role:
- keep the survival-logit as the primary PD model unless model comparison supports a change,
- use the Markov model as a challenger, migration model, or state-transition explanation,
- evolve it later into a multi-state survival style extension.

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
- exact interval default-intensity estimation,
- CTMC generator estimation from observed state durations,
- compensated process construction from exact intervals,
- bridge between discrete-time panel approximations and continuous-time default intensities.

Still open:
- Cox-process or counting-process regression with covariates,
- recurrent delinquency event processes,
- censoring-aware estimation for incomplete event histories.

Why it belongs here:
- this repo starts quarterly and discrete-time because that matches the synthetic reporting panel,
- but the scaffold should already show that it can grow into continuous-time default modelling when richer event timing is available.

### 7b. Continuous-state latent credit-quality upgrade

Best location:
- `src/credit_risk_lab/econometrics/continuous_state.py`

Implemented layer:
- finite latent credit-quality grid with explicit default boundary,
- OU-style finite-difference generator with local diffusion,
- optional downward jumps and killing/default intensity,
- default probabilities from `exp(tQ)`,
- Beurling-Deny-style split into local, jump, and killing energy,
- finite-grid capacity, polar-like state, Cheeger-energy, and regularity diagnostics.

Use this layer when:
- event timing or latent credit quality matters more than quarter-end buckets,
- the model needs to distinguish gradual deterioration from sudden jumps,
- score surfaces or management overlays should be checked for smoothness over a metric state space.

Do not use this layer automatically:
- quarterly bucket data alone may not identify a continuous-state process,
- a non-constant hazard path can add noise if the empirical hazard is actually close to flat,
- Dirichlet regularisation is a diagnostic and model-design tool, not proof that the true credit process is symmetric.

### 8. Macro-sensitive PD path upgrade

Best locations:
- `src/credit_risk_lab/econometrics/macro.py`
- `src/credit_risk_lab/econometrics/forward_hazard.py`

Implemented first layer:
- AR(1) macro forecasting layer,
- VAR(1) macro forecasting layer,
- baseline/downside/upside macro path construction,
- score baseline/downside/upside macro paths directly through the PD model,
- keep scenario overlays mainly for LGD and EAD,
- use PD overlays only as explicitly labelled management overlays if needed.

Still open:
- direct refit of the primary PD model with macro variables in the default feature spec.

Recommended design:
- one primary macro channel for PD,
- avoid putting unemployment into the PD model and then multiplying PD again in ECL for the same effect unless that second layer is clearly a judgemental overlay.

## Toolbox Principle

The modules are designed as a lab. A nonlinear term, heterogeneity correction,
macro-sensitive hazard path, Markov challenger, continuous-time generator, or
Dirichlet regulariser should be used only after a diagnostic or model-comparison
reason supports it. The intended workflow is:

- fit the transparent baseline,
- run diagnostics for calibration, heterogeneity, macro sensitivity, drift, and
  migration coherence,
- add one extension at a time,
- compare against the baseline,
- keep the simpler model when the extension adds noise or weakens validation
  evidence.

## Regulatory Boundary

The econometric toolbox is aligned with IFRS 9 ECL, EBA Loan Origination and
Monitoring, Definition of Default, and model-governance concepts. CRR3/CRD6 and
Basel final reforms are background capital context only. The repo should not
mix IFRS 9 provisioning with a capital/RWA engine unless a separate module is
explicitly added for that purpose.

Detailed note:
- [`docs/regulatory_scope.md`](regulatory_scope.md)
