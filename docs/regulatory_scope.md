# Regulatory Scope

This repo is an IFRS 9 credit-risk modelling lab, not a regulatory capital
calculator. Regulation is included only where it clarifies model definitions,
monitoring expectations, validation evidence, or the boundary between IFRS 9
provisioning and prudential capital.

Current reference date: 2026-05-08.

## What Belongs In Scope

| Regulation or framework | Include in repo? | Role in this lab |
| --- | --- | --- |
| IFRS 9 | Core | Stage allocation, 12-month versus lifetime ECL, forward-looking scenarios. |
| EBA Loan Origination and Monitoring | Core governance context | Creditworthiness assessment, credit-granting governance, ongoing monitoring, data quality. |
| CRR Article 178 / EBA Definition of Default | Core modelling definition | Default identification, 90 days past due, unlikeliness-to-pay, cure/return-to-performing logic. |
| EBA PD/LGD Guidelines | Core validation context | PD/LGD estimation expectations, calibration, validation, conservatism, margin of conservatism. |
| ECB internal-model guidance | Context | Internal-model governance, monitoring, model-change and review expectations. |
| Basel III final reforms / CRR3-CRD6 | Background capital context | Explains prudential capital and RWA context, but not implemented as this repo’s main engine. |
| CRD IV / CRR | Historical/legal background | Useful because Article 178 CRR anchors Definition of Default, but the repo should not present CRD IV as the current main package. |

## Loan Origination And Monitoring

The EBA Guidelines on Loan Origination and Monitoring are directly relevant.
They specify internal governance for granting and monitoring credit facilities,
borrower creditworthiness assessment, data handling, and lifecycle monitoring.
This maps naturally to the repo’s synthetic loan-book design, data-quality
checks, monitoring layer, and validation pack.

Implementation implication:

- keep data-quality and monitoring outputs prominent,
- document borrower and loan features used in PD modelling,
- treat missingness, range checks, drift, and cohort consistency as model
  governance evidence,
- do not reduce the repo to a classification exercise.

Source:
- European Banking Authority, "Guidelines on loan origination and monitoring."
  https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-loan-origination-and-monitoring

## Definition Of Default

Definition of Default is directly relevant and should be part of the repo. The
current code already uses `90+ DPD` as default-like behaviour and default as an
absorbing Markov state. The regulatory reference adds the missing governance
language: days-past-due, materiality, unlikeliness-to-pay, return-to-non-default
conditions, external data treatment, group application, and retail-specific
aspects.

Implementation implication:

- keep `90+ DPD / default` as a clear default trigger in examples,
- add room for unlikeliness-to-pay indicators beyond DPD,
- treat cure and return-to-performing logic as a model extension,
- document that the synthetic version is simplified and not a full Article 178
  policy engine.

Source:
- European Banking Authority, "Guidelines on the application of the definition
  of default."
  https://www.eba.europa.eu/activities/single-rulebook/regulatory-activities/credit-risk/guidelines-application-definition
- EBA Interactive Single Rulebook, CRR Article 178, "Default of an obligor or
  credit facility."
  https://www.eba.europa.eu/regulation-and-policy/single-rulebook/interactive-single-rulebook/16022

Note:
- The EBA page shows the guidelines as final and translated into EU official
  languages. It also shows an amending-guidelines consultation that closed on
  15 October 2025. Until a new final version is issued, this repo should cite
  the current final guidelines and mention the review only as a live regulatory
  development.

## CRD IV, CRR, CRR3, CRD6, And Basel Final Reforms

CRD IV is not the best current headline for this repo. It is historically
important because the CRD IV / CRR package implemented the first elements of
Basel III in EU law and CRR Article 178 anchors the default definition.

For current EU banking context, the better language is:

- `CRR / CRD` for the EU prudential framework,
- `CRR3 / CRD6` for the EU Banking Package implementing the final Basel III
  reforms,
- `Basel III final reforms` rather than `Basel IV`, because Basel IV is market
  shorthand rather than the official Basel Committee name.

Implementation implication:

- include these as background context for capital, RWA, model landscape, and
  IRB relevance,
- do not build a Basel capital calculator into this IFRS 9 repo,
- keep capital/regulatory RWA as a separate future project if needed.

Sources:
- EBA, "The Basel framework: the global regulatory standards for banks."
  https://www.eba.europa.eu/activities/basel-framework-global-regulatory-standards-banks
- EBA, "CRR3/CRD6 dashboard."
  https://www.eba.europa.eu/risk-and-data-analysis/risk-analysis/risk-monitoring/crr3-crd6-dashboard
- BIS Basel Committee, "Basel III: Finalising post-crisis reforms."
  https://www.bis.org/bcbs/publ/d424.htm

## Recommended Repo Treatment

The public README should keep IFRS 9, EBA Loan Origination and Monitoring,
Definition of Default, and EBA/ECB model governance in the main reference list.
Basel final reforms and CRR3/CRD6 should appear in a short prudential-context
section only.

The implementation should not claim compliance with any of these frameworks.
It should state that the repo is a modelling lab that uses regulatory concepts
to structure definitions, monitoring, and validation evidence.
