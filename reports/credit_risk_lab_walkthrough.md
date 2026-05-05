# Credit Risk Lab Walkthrough

## Objective

Show an end-to-end credit-risk workflow that is relevant to junior modeller and junior actuarial applications:

- synthetic loan-book generation,
- survival-based PD estimation,
- IFRS 9 staging and ECL,
- monitoring,
- validation and governance.

## Why The Design Is Strong For Hiring

- It is clearly credit-focused rather than a generic ML project.
- It demonstrates that model output must become provisions, monitoring, and validation evidence.
- It is reproducible without private data.
- The validation logic is isolated in a separate package, which makes the engineering story stronger.

## Method References

- Discrete-time survival modelling:
  [Singer & Willett (1993)](https://journals.sagepub.com/doi/10.3102/10769986018002155)
- IFRS 9 impairment logic:
  [IFRS 9 Financial Instruments](https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf)
- EU banking governance:
  [EBA Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/model-validation/guidelines-pd-estimation-lgd)
- EU banking supervisory context:
  [ECB Internal Models](https://www.bankingsupervision.europa.eu/activities/internal_models/html/index.en.html)
- Dutch / actuarial prudential context:
  [DNB Solvency II general notes](https://www.dnb.nl/en/sector-information/open-book-supervision/open-book-supervision-sectors/insurers/law-and-regulations-insurers/solvency-ii-general-notes/)
  and [EIOPA Solvency II](https://www.eiopa.europa.eu/browse/regulation-and-policy/solvency-ii_en)
- Probability forecast validation:
  [Brier (1950)](https://cir.nii.ac.jp/crid/1361981468554183168)
- Optional Sinkhorn divergence:
  [Cuturi (2013)](https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport)

## What To Demo In An Interview

1. Generate a synthetic multi-period loan book.
2. Fit the pooled-logit survival model and explain why quarterly panels work.
3. Score a reporting date and show `12m PD` versus `lifetime PD`.
4. Run the IFRS 9 engine and explain stage migration plus provision roll-forward.
5. Run the validation demo and explain what a benchmark/challenger result means.
