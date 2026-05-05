# Model Validation Pack

Self-contained validation toolkit that can be split from the root repository later.

## Purpose

Given one model's scored observations, this package can produce:

- backtest and calibration output,
- period and segment stability summaries,
- sensitivity analysis,
- drift tests with PSI and Wasserstein distance,
- optional Sinkhorn divergence when `pot` is installed,
- benchmark comparison,
- a validation memo in Markdown.

## Generic API

- `run_validation_pack(...)`
- `run_backtest(...)`
- `run_stability_tests(...)`
- `run_sensitivity_analysis(...)`
- `run_drift_tests(...)`
- `compare_with_benchmark(...)`
- `render_validation_memo(...)`

## Credit Adapter

The package also includes:

- `build_credit_validation_bundle(...)`

This adapter converts scored credit-risk snapshots into the generic
`ScoredObservation` structures used by the validation API without importing
root-repo dataclasses directly.

## Dependencies

Core:

- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `pytest`

Optional:

- `pot` for Sinkhorn / entropic optimal transport via `model-validation-pack[ot]`

## Quick Start

```bash
.venv/bin/python model_validation_pack/examples/run_validation_demo.py
.venv/bin/pytest -q model_validation_pack/tests/test_model_validation_pack.py
```

## References

- EU banking model validation:
  [EBA Model Validation](https://www.eba.europa.eu/regulation-and-policy/model-validation)
- ECB supervisory context for internal models:
  [ECB Internal Models](https://www.bankingsupervision.europa.eu/activities/internal_models/html/index.en.html)
- Probability forecast backtesting:
  [Brier (1950)](https://cir.nii.ac.jp/crid/1361981468554183168)
- Optional Sinkhorn divergence:
  [Cuturi (2013)](https://papers.nips.cc/paper/4927-sinkhorn-distances-lightspeed-computation-of-optimal-transport)
