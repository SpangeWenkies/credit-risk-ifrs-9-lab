# Methodology Notes

## Why These Methods

### Discrete-time survival PD

The root repo uses a pooled-logit discrete-time hazard model because it fits naturally with quarterly account panels and is much easier to explain in a junior-role interview than a black-box approach.

Reference:
- Singer, J. D., and Willett, J. B. (1993), "It's About Time: Using Discrete-Time Survival Analysis to Study Duration and the Timing of Events."
  https://journals.sagepub.com/doi/10.3102/10769986018002155

### IFRS 9 staging and ECL

The impairment engine uses a simplified significant-increase-in-credit-risk framework based on delinquency, forbearance, and deterioration versus origination. This is not a production accounting policy, but it is a credible portfolio-project representation of the logic required by IFRS 9.

Reference:
- IFRS Foundation, "IFRS 9 Financial Instruments."
  https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf

### Validation and governance

The validation pack is organised around model-risk governance rather than only predictive accuracy. That is why it includes benchmark comparison, drift diagnostics, sensitivity analysis, and memo rendering.

Reference:
- Board of Governors of the Federal Reserve System and OCC, "Supervisory Guidance on Model Risk Management (SR 11-7)."
  https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107a1.pdf

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
