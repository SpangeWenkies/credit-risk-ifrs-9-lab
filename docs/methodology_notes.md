# Methodology Notes

## Why These Methods

### Discrete-time survival PD

The root repo uses a pooled-logit discrete-time hazard model because it fits naturally with quarterly account panels and is much easier to explain in a junior-role interview than a black-box approach.

Reference:
- Singer, J. D., and Willett, J. B. (1993), "It's About Time: Using Discrete-Time Survival Analysis to Study Duration and the Timing of Events."
  https://journals.sagepub.com/doi/10.3102/10769986018002155

Current simplification:
- the current version converts a one-quarter hazard into `12m PD` and `lifetime PD` using a constant-hazard approximation.

Planned econometric upgrade:
- use `src/credit_risk_lab/econometrics/forward_hazard.py` to build a forward hazard path, where future balance, seasoning, and macro variables evolve over time and lifetime PD is built from the full hazard term structure rather than a flat hazard.

Best extension slots:
- `src/credit_risk_lab/econometrics/forward_hazard.py`
- `src/credit_risk_lab/econometrics/macro.py`

### IFRS 9 staging and ECL

The impairment engine uses a simplified significant-increase-in-credit-risk framework based on delinquency, forbearance, and deterioration versus origination. This is not a production accounting policy, but it is a credible portfolio-project representation of the logic required by IFRS 9.

Reference:
- IFRS Foundation, "IFRS 9 Financial Instruments."
  https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf

### Validation and governance

The validation pack is organised around model-risk governance rather than only predictive accuracy. That is why it includes benchmark comparison, drift diagnostics, sensitivity analysis, and memo rendering.

Reference:
- EBA, "Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures."
  https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/model-validation/guidelines-pd-estimation-lgd
- ECB Banking Supervision, "Internal models."
  https://www.bankingsupervision.europa.eu/activities/internal_models/html/index.en.html

### Dutch / EU relevance

For Dutch banking roles, the repo should be read through `IFRS 9`, `EBA`, `ECB`, and `DNB` rather than U.S. supervisory letters. That is especially true for PD/LGD/EAD, monitoring, and internal-model governance.

References:
- EBA, "Guidelines on loan origination and monitoring."
  https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-loan-origination-and-monitoring
- ECB Banking Supervision, "Internal models."
  https://www.bankingsupervision.europa.eu/activities/internal_models/html/index.en.html

### Actuarial angle

The actuarial relevance of the repo comes from reserve logic, stress/sensitivity thinking, governance, and internal-model discussion. For Dutch actuarial and insurer roles, `Solvency II` is the right prudential anchor.

References:
- DNB, "Solvency II: General notes."
  https://www.dnb.nl/en/sector-information/open-book-supervision/open-book-supervision-sectors/insurers/law-and-regulations-insurers/solvency-ii-general-notes/
- DNB, "Pillar 1: Internal models."
  https://www.dnb.nl/en/sector-information/open-book-supervision/open-book-supervision-sectors/insurers/solvency-ii-request-overview/pillar-1-internal-models/
- EIOPA, "Solvency II."
  https://www.eiopa.europa.eu/browse/regulation-and-policy/solvency-ii_en

### Backtesting

Probability forecast quality is summarised with the Brier score because the project validates PD-type probabilities rather than raw classifications.

Reference:
- Brier, G. W. (1950), "Verification of Forecasts Expressed in Terms of Probability."
  https://cir.nii.ac.jp/crid/1361981468554183168

### Sinkhorn divergence

The validation pack treats Sinkhorn divergence as optional. It is useful for distribution-shift analysis, but it should not make the generic validation workflow unusable when the OT dependency is absent.

Reference:
- Cuturi, M. (2013), "Sinkhorn Distances: Lightspeed Computation of Optimal Transport."
  https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport

### International comparison

U.S. SR guidance can still be mentioned as an international comparison point in interviews, but it should not be the primary regulatory anchor for Dutch applications.

### Markov and continuous-time room

The repo is also intentionally scaffolded for two more advanced directions:

- `Markov / state-transition modelling`, which is especially natural for delinquency migration, cure, prepayment, and IFRS 9 stage movement. The first implementation lives in `src/credit_risk_lab/econometrics/markov.py` and treats the transition matrix as the finite-state kernel, matrix powers as the semigroup, default/prepayment as absorbing cemetery-style states, and the absorbing-chain fundamental matrix as a Green/resolvent analogue.
- `Continuous-time default modelling`, which becomes relevant when default timing is observed more finely than just quarter-end panels and where counting-process / intensity ideas become useful. The first implementation lives in `src/credit_risk_lab/econometrics/continuous_time.py` and includes piecewise intensity estimates, counting-process tables, compensated default diagnostics, and survival from intensity paths.

Best extension slots:
- `src/credit_risk_lab/econometrics/markov.py`
- `src/credit_risk_lab/econometrics/continuous_time.py`

Detailed bridge note:
- `docs/dirichlet_markov_credit_bridge.md`

References:
- Fukushima, M., Oshima, Y., and Takeda, M. (2011), "Dirichlet Forms and Symmetric Markov Processes."
  https://www.degruyterbrill.com/document/doi/10.1515/9783110218091/html
- Jarrow, R. A., Lando, D., and Turnbull, S. M. (1997), "A Markov Model for the Term Structure of Credit Risk Spreads."
  https://academic.oup.com/rfs/article/10/2/481/1589160
- Lando, D., and Skodeberg, T. M. (2002), "Analyzing Rating Transitions and Rating Drift with Continuous Observations."
  https://www.sciencedirect.com/science/article/pii/S037842660100228X
