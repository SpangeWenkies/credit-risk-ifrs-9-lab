# Credit Risk & IFRS 9 Lab

Portfolio project for junior credit risk modellers and junior actuaries.

This repository now demonstrates a full synthetic-first workflow:

- multi-period retail loan portfolio generation,
- pooled-logit survival-based PD estimation,
- IFRS 9 stage 1 / 2 / 3 logic,
- scenario-weighted ECL with interpretable LGD/EAD,
- data-quality and drift monitoring,
- a split-friendly validation subrepo for backtesting and governance.

The validation toolkit lives in [`model_validation_pack`](model_validation_pack) and is designed so it can later become its own repository with minimal cleanup.

## Repo Layout

```text
.
├── README.md
├── data/
├── docs/
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
└── tests/
```

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
.venv/bin/pytest -q model_validation_pack/tests/test_model_validation_pack.py
```

## Core Methods And References

- IFRS 9 impairment logic:
  [IFRS 9 Financial Instruments](https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2021/issued/part-a/ifrs-9-financial-instruments.pdf)
- Discrete-time survival modelling:
  [Singer & Willett (1993)](https://journals.sagepub.com/doi/10.3102/10769986018002155)
- Validation governance framing:
  [SR 11-7](https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107a1.pdf)
- Probability forecast backtesting:
  [Brier (1950)](https://cir.nii.ac.jp/crid/1361981468554183168)
- Optional Sinkhorn divergence:
  [Cuturi (2013)](https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport)

## Talking Points

- Why a synthetic multi-period panel is enough to demonstrate survival modelling and validation logic.
- Why IFRS 9 staging and provision roll-forwards make the repo more relevant than a generic classification project.