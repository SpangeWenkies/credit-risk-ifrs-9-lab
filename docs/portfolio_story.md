# Portfolio Story

## Positioning

Use this repository to present yourself as someone who can connect:

- econometrics,
- modelling,
- regulation,
- business interpretation,
- and model governance.

That combination is more valuable for junior risk roles than a generic machine-learning demo.

The strongest framing is:

- I already understand the econometric structure behind binary outcomes, panel data, survival logic, calibration, and validation.
- I am using this repo to apply that toolkit in a credit-risk setting.
- The new thing I am learning and demonstrating is the domain layer: how model outputs become IFRS 9, monitoring, and governance artefacts.

## Interview Narrative

Tell the project as a workflow:

1. I built a synthetic multi-period retail loan book so the work is reproducible and shareable.
2. I fitted a discrete-time survival model to quarterly account panels to estimate point-in-time PD, which is a direct econometric-to-credit-risk mapping.
3. I implemented stage 1 / 2 / 3 logic and scenario-weighted IFRS 9 ECL with interpretable LGD and EAD assumptions.
4. I added monitoring for data quality, PSI, and Wasserstein-based distribution shift.
5. I separated the validation tooling so it can become a standalone model validation pack with backtesting and memo generation.
6. I added first-pass econometric extension modules for panel methods, duration diagnostics, calibration, causal audit thinking, measurement-error sensitivity, model selection, and heterogeneity checks.
7. I also implemented first layers for non-constant forward hazard paths, Markov migration models, and continuous-time intensity / counting-process diagnostics.
8. I connected the Markov extension to my functional-analysis coursework by treating the finite-state credit chain through kernels, semigroups, generators, absorbing cemetery states, Green/fundamental matrices, and Dirichlet-style transition energy.

## What Recruiters Should Notice

- You understand that a good risk model is not only about prediction.
- You can map econometric tools directly to credit-risk use cases.
- You know how model outputs become provisions, governance packs, and validation evidence.
- You can write clean, modular Python that looks like the start of a real internal toolkit.
- You can explain which methods are simplified, why they are still defensible, and which references support them.

## Best Extensions

- add charts for stage migration and scenario ECL,
- deepen the econometric modules under `src/credit_risk_lab/econometrics/`,
- integrate forward hazard path outputs directly into the IFRS 9 ECL engine,
- expand Markov migration models into stage-based challengers,
- extend grouped Markov matrices into full covariate-dependent transition models,
- extend continuous-time default / intensity modelling when exact event timing is available,
- swap the single pooled-logit PD model for segment-specific variants,
- add confidence intervals and richer challenger testing in the validation pack,
- add documentation that maps each module to a real bank or insurer workflow.
