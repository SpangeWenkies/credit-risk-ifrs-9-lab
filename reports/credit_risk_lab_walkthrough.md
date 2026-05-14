# Credit Risk Lab Walkthrough

## Objective

Show an end-to-end credit-risk workflow:

- synthetic multi-period loan-book generation,
- pooled-logit survival-based PD estimation,
- IFRS 9 staging and ECL,
- scenario-weighted ECL with interpretable LGD/EAD,
- monitoring and drift diagnostics,
- validation and challenger comparison,
- Markov migration and continuous-time extension hooks.

## Baseline Pipeline

1. Generate a synthetic multi-period retail loan book.
2. Fit the pooled-logit survival PD model.
3. Score a reporting date and compare `12m PD` with `lifetime PD`.
4. Run the IFRS 9 ECL engine and reconcile stage summaries plus provision roll-forward.
5. Run monitoring and validation diagnostics.
6. Add one econometric extension at a time only when diagnostics justify it.

## Extension Checks

- Use forward hazard paths when lifetime risk is not well represented by a flat quarterly hazard.
- Use macro-sensitive PD or Markov migration as one primary macro channel, not as duplicated overlays.
- Use Markov transition tools when state migration and cure/default pathways matter.
- Use continuous-time intensity tools when exact event dates or state durations are available.
- Use continuous-state Dirichlet diagnostics when studying latent credit-quality diffusion, jumps, and killing/default.

## Method References

- Discrete-time survival modelling:
  [Singer & Willett (1993)](https://journals.sagepub.com/doi/10.3102/10769986018002155)
- IFRS 9 impairment logic:
  [IFRS 9 Financial Instruments](https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf)
- EU banking governance:
  [EBA Guidelines on PD estimation, LGD estimation and the treatment of defaulted exposures](https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/model-validation/guidelines-pd-estimation-lgd)
- EU banking supervisory context:
  [ECB Internal Models](https://www.bankingsupervision.europa.eu/activities/internal_models/html/index.en.html)
- Markov migration:
  [Jarrow-Lando-Turnbull (1997)](https://academic.oup.com/rfs/article/10/2/481/1589160)
  and [Lando-Skodeberg (2002)](https://www.sciencedirect.com/science/article/pii/S037842660100228X)
- Continuous-time credit risk:
  [Lando (1998)](https://doi.org/10.1017/S0269964898173055)
  and [Duffie-Singleton (1999)](https://academic.oup.com/rfs/article-abstract/12/4/687/1599653)
- Matrix-log generators:
  [Israel-Rosenthal-Wei (2001)](https://doi.org/10.1080/713665550)
- Probability forecast validation:
  [Brier (1950)](https://cir.nii.ac.jp/crid/1361981468554183168)
- Optional Sinkhorn divergence:
  [Cuturi (2013)](https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport)
