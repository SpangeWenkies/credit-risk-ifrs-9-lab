# Methodology Notes

This document records the modelling choices behind the lab. The repo is a
toolbox: start with transparent baseline models, run diagnostics, then add
extensions only when the data and validation evidence justify them.

## Discrete-Time Survival PD

The baseline PD model is a pooled-logit discrete-time hazard model. This fits
quarterly account panels naturally: each loan-quarter row is an interval and
`default_next_period` is the event indicator.

Reference:
- Singer, J. D., and Willett, J. B. (1993), "It's About Time: Using Discrete-Time Survival Analysis to Study Duration and the Timing of Events."
  https://journals.sagepub.com/doi/10.3102/10769986018002155

Current baseline:
- estimate a one-quarter hazard,
- convert it to 12-month and lifetime PD with a constant-hazard approximation.

Available upgrade:
- use `src/credit_risk_lab/econometrics/forward_hazard.py` to construct future
  loan-quarter rows and combine a non-constant hazard path with
  `1 - product(1 - h_t)`.

Model-choice rule:
- keep the constant-hazard baseline if the forward path is effectively flat or
  less stable out of sample,
- use the forward path when seasoning, balance, LTV, DTI, or macro projections
  materially change future hazards.

## IFRS 9 Staging And ECL

The impairment engine uses simplified significant-increase-in-credit-risk logic
based on delinquency, forbearance, and deterioration versus origination. It is a
portfolio-level modelling representation, not a production accounting policy.

Reference:
- IFRS Foundation, "IFRS 9 Financial Instruments."
  https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf

Design choice:
- macro effects can enter PD through the model or through a management overlay,
  but the repo should not apply the same macro deterioration twice without
  explicitly labelling the second effect as judgemental.

## Validation And Governance

The validation pack separates performance, stability, drift, sensitivity, and
benchmark comparison. This is broader than predictive accuracy and aligns with
credit model-governance expectations.

References:
- EBA, "Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures."
  https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/model-validation/guidelines-pd-estimation-lgd
- ECB Banking Supervision, "Internal models."
  https://www.bankingsupervision.europa.eu/activities/internal_models/html/index.en.html
- EBA, "Guidelines on loan origination and monitoring."
  https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-loan-origination-and-monitoring

## Definition Of Default And Monitoring Scope

The synthetic generator and Markov module use `90+ DPD / default` as a clear
default-like state. The relevant EU regulatory anchor is CRR Article 178 and
the EBA Guidelines on the application of the definition of default. For this
repo, the regulation is used to structure default definitions, cure/return
logic, and monitoring evidence. It is not implemented as a full regulatory
default policy engine.

References:
- EBA, "Guidelines on the application of the definition of default."
  https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-application-definition
- EBA Interactive Single Rulebook, CRR Article 178.
  https://www.eba.europa.eu/regulation-and-policy/single-rulebook/interactive-single-rulebook/16022

## Prudential Capital Boundary

Basel final reforms, CRR3, and CRD6 are useful background because they explain
the wider prudential capital environment around credit risk models. They should
not drive this repo's implementation, which is focused on IFRS 9 ECL,
monitoring, and validation rather than capital/RWA calculation.

References:
- EBA, "CRR3/CRD6 dashboard."
  https://www.eba.europa.eu/risk-and-data-analysis/risk-analysis/risk-monitoring/crr3-crd6-dashboard
- BIS Basel Committee, "Basel III: Finalising post-crisis reforms."
  https://www.bis.org/bcbs/publ/d424.htm

## Backtesting

Probability forecast quality is summarised with calibration tables, observed
default rates by band, and Brier score.

Reference:
- Brier, G. W. (1950), "Verification of Forecasts Expressed in Terms of Probability."
  https://cir.nii.ac.jp/crid/1361981468554183168

## Drift And Sinkhorn

The validation pack computes PSI and Wasserstein distance. Sinkhorn divergence
is optional and only runs when the `pot` dependency is installed; otherwise the
API returns a structured `not_run` result.

Reference:
- Cuturi, M. (2013), "Sinkhorn Distances: Lightspeed Computation of Optimal Transport."
  https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport

## Markov Migration

Finite-state Markov migration is useful when the question is not only "will the
loan default?" but "how does the loan move between credit states?" The baseline
states are current, early arrears, serious arrears, default, and prepay/mature.

References:
- Jarrow, R. A., Lando, D., and Turnbull, S. M. (1997), "A Markov Model for the Term Structure of Credit Risk Spreads."
  https://academic.oup.com/rfs/article/10/2/481/1589160
- Lando, D., and Skodeberg, T. M. (2002), "Analyzing Rating Transitions and Rating Drift with Continuous Observations."
  https://www.sciencedirect.com/science/article/pii/S037842660100228X

Implemented methods:
- unconditional transition matrices,
- grouped and macro-regime matrices,
- covariate-dependent transition logits,
- Markov-implied PD comparison,
- stage-transition aggregation,
- reversibility and score-smoothness diagnostics.

Model-choice rule:
- use Markov migration as a challenger or migration explanation,
- use covariate Markov migration when transition probabilities differ
  materially by borrower, product, or macro state,
- keep the simpler unconditional matrix when sparse transition data make
  covariate estimates unstable.

## Continuous-Time Modelling

Continuous-time modelling is preferable when exact event timing is available.
If the data contain default dates, cure dates, delinquency-entry dates, or
account-level durations, then intensities and CTMC generators are more natural
than quarter-end approximations.

References:
- Andersen, P. K., and Gill, R. D. (1982), "Cox's Regression Model for Counting Processes."
- Lando, D. (1998), "On Cox Processes and Credit Risky Securities."
  https://doi.org/10.1017/S0269964898173055
- Duffie, D., and Singleton, K. J. (1999), "Modeling Term Structures of Defaultable Bonds."
  https://academic.oup.com/rfs/article-abstract/12/4/687/1599653

Implemented methods:
- exposure-count default intensities,
- exact-interval default intensities,
- compensated default counting-process diagnostics,
- CTMC generator estimation from observed durations,
- survival from intensity paths.

Model-choice rule:
- use quarterly discrete-time survival when observations are quarterly,
- use exact-event-time intensities when event dates are observed,
- use matrix-log generators only as diagnostics unless validity checks pass.

## Matrix Logarithms And Embeddability

A discrete transition matrix `P` can be generated by a continuous-time Markov
chain only if there exists a valid generator `Q` such that `P = exp(Q dt)`.
The matrix logarithm `log(P) / dt` is the formal inverse, but empirical credit
matrices can fail the generator constraints.

Reference:
- Israel, R. B., Rosenthal, J. S., and Wei, J. Z. (2001), "Finding Generators for Markov Chains via Empirical Transition Matrices, with Applications to Credit Ratings."
  https://doi.org/10.1080/713665550

Implemented methods:
- `transition_generator(P)` for the stable first-order approximation,
- `matrix_log_generator(P)` for embeddability diagnostics,
- `project_generator_to_valid_rates(Q)` for diagnostic repair experiments.

## Continuous-State Credit Quality

Continuous-state modelling is useful when credit quality is represented as a
latent coordinate rather than a finite delinquency bucket. The process can
combine gradual deterioration, sudden jumps, and killing/default.

References:
- Black, F., and Cox, J. C. (1976), "Valuing Corporate Securities: Some Effects of Bond Indenture Provisions."
  https://www.jstor.org/stable/2326758
- Fukushima, M., Oshima, Y., and Takeda, M. (2011), "Dirichlet Forms and Symmetric Markov Processes."
  https://www.degruyterbrill.com/document/doi/10.1515/9783110218091/html
- Ma, Z.-M., and Roeckner, M. (1992), "Introduction to the Theory of (Non-Symmetric) Dirichlet Forms."
  https://link.springer.com/book/10.1007/978-3-642-77739-4

Implemented methods:
- latent state grid with default boundary,
- OU-style finite-difference generator,
- jump and killing/default intensities,
- default probabilities from `exp(tQ)`,
- Beurling-Deny-style local, jump, and killing decomposition,
- finite-grid proxies for capacity, polar-like states, Cheeger energy, and
  regularity.

Model-choice rule:
- use continuous-state diagnostics to compare process assumptions,
- avoid claiming the synthetic quarterly data identify a continuous-state model
  without additional event-time or latent-score evidence.
