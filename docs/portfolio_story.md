# Portfolio Story

## Positioning

Use this repository to present yourself as someone who can connect:

- modelling,
- regulation,
- business interpretation,
- and model governance.

That combination is more valuable for junior risk roles than a generic machine-learning demo.

## Interview Narrative

Tell the project as a workflow:

1. I built a synthetic multi-period retail loan book so the work is reproducible and shareable.
2. I fitted a discrete-time survival model to quarterly account panels to estimate point-in-time PD.
3. I implemented stage 1 / 2 / 3 logic and scenario-weighted IFRS 9 ECL with interpretable LGD and EAD assumptions.
4. I added monitoring for data quality, PSI, and Wasserstein-based distribution shift.
5. I separated the validation tooling so it can become a standalone model validation pack with backtesting and memo generation.

## What Recruiters Should Notice

- You understand that a good risk model is not only about prediction.
- You know how model outputs become provisions, governance packs, and validation evidence.
- You can write clean, modular Python that looks like the start of a real internal toolkit.
- You can explain which methods are simplified, why they are still defensible, and which references support them.

## Best Extensions

- add charts for stage migration and scenario ECL,
- swap the single pooled-logit PD model for segment-specific variants,
- add confidence intervals and richer challenger testing in the validation pack,
- add documentation that maps each module to a real bank or insurer workflow.
