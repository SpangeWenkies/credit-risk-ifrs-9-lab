# Dirichlet-Form And Markov Credit Migration Bridge

This note explains how Markov-chain and Dirichlet-form ideas are used in the
credit-risk lab. The purpose is practical: choose the right stochastic language
for the data frequency, make state migration interpretable, and add diagnostics
that compare discrete, continuous-time, and continuous-state credit processes.

## Practical Rule

Use the theory that matches the data and modelling object.

| Data or modelling object | Recommended theory | Implemented location |
| --- | --- | --- |
| Quarterly delinquency states | Discrete-time finite Markov chain | `src/credit_risk_lab/econometrics/markov.py` |
| Exact transition/default dates | Continuous-time Markov chain and counting process | `src/credit_risk_lab/econometrics/continuous_time.py` |
| Latent continuous credit quality | Continuous-state generator with local, jump, and killing components | `src/credit_risk_lab/econometrics/continuous_state.py` |
| Score or stage smoothness over observed migration | Finite-state Dirichlet energy diagnostic | `dirichlet_transition_energy`, `score_smoothness_diagnostics` |
| Smooth latent score surface | Finite-grid Cheeger-energy proxy | `cheeger_energy_credit_proxy` |

The discrete quarterly model should not be dressed up as continuous-time theory
unless it is explicitly used as an approximation. Conversely, if exact event
times are available, it is better to estimate continuous-time intensities
directly instead of forcing a quarterly panel model.

## Finite-State Credit Translation

| Course object | Credit-risk object | Repo implementation |
| --- | --- | --- |
| State space `X` | Delinquency states plus terminal states | `current`, `dpd_1_29`, `dpd_30_89`, `default`, `prepay_mature` |
| Kernel `p(x, dy)` | One-period migration probabilities | `fit_markov_transition_model` |
| Semigroup `T_t` | Multi-period transition matrix powers | `n_step_transition_matrix(P, n)` |
| Generator `Q` | Transition-rate matrix | `transition_generator`, `matrix_log_generator`, `estimate_ctmc_generator_from_durations` |
| Cemetery state | Default or prepay/maturity terminal state | absorbing rows |
| Green/resolvent intuition | Expected visits before absorption | `absorption_summary(P)` |
| Dirichlet form | Transition-weighted roughness of a score | `dirichlet_transition_energy(P, f)` |
| Detailed balance | Reversibility of state flows | `reversibility_diagnostics(P, m)` |
| Regularisation by energy | Smooth a rating scale over observed migration | `regularize_state_scores(P, f)` |

## Implemented Roadmap

### V1: Finite-State Migration Challenger

Implemented in `src/credit_risk_lab/econometrics/markov.py`.

- Build observed one-step transition panels.
- Estimate row-stochastic transition matrices.
- Force default and prepay/maturity as absorbing states.
- Compute multi-period matrix powers.
- Approximate a generator by `(P - I) / dt`.
- Compute absorption probabilities and expected time to absorption.
- Compute finite-state Dirichlet-style score roughness.
- Estimate grouped transition matrices.
- Compute Markov-implied default PDs.

### V2: Covariate-Dependent Markov Model

Implemented in `fit_covariate_transition_model`.

- Estimate `P(next_state | current_state, borrower features, macro variables)`.
- Use robust one-vs-rest transition logits with empirical fallback.
- Score row-level transition distributions.
- Compute covariate Markov default PDs.
- Compare Markov-implied default PDs to survival-logit PDs with `compare_markov_to_survival_pd`.

This is a challenger and explanation layer, not a replacement for the primary
survival PD model by default.

### V3: Macro-Sensitive Markov Scenario Engine

Implemented in `assign_macro_regime`, `fit_macro_regime_transition_matrices`,
and `build_markov_scenario_matrices`.

- Label benign, baseline, and stress regimes from a macro variable.
- Estimate regime-conditioned migration matrices.
- Map baseline/downside/upside scenarios to migration matrices.
- Avoid double-counting by using this either as the migration channel or as a
  challenger when the PD model already includes macro covariates.

### V4: Continuous-Time Default And Counting Processes

Implemented in `src/credit_risk_lab/econometrics/continuous_time.py`.

- Estimate piecewise default intensities from panel exposure counts.
- Estimate default intensities from exact start/end event intervals.
- Estimate CTMC generators from observed state durations.
- Build compensated default counting-process diagnostics.
- Convert intensity paths into continuous-time survival probabilities.

The continuous-time object is the counting process `N_t` and its compensator
`A_t = integral lambda_s ds`. If the intensity model is adequate, the
compensated process `N_t - A_t` should not show systematic drift.

### V5: Dirichlet-Form Regularisation And Continuous-State Diagnostics

Implemented in `markov.py` and `continuous_state.py`.

- Use transition energy to detect rough state scores.
- Decompose which migration edges drive roughness.
- Smooth state scores with a graph-Laplacian penalty.
- Diagnose reversibility before choosing symmetric or non-symmetric energy.
- Build a latent continuous credit-quality grid with default boundary.
- Construct a generator with local diffusion, downward jumps, and killing.
- Decompose energy into local, jump, and killing components.
- Run finite-grid proxies for capacity, polar-like states, Cheeger energy, and
  regularity.

## Simple Explanation Of The Matrix-Log Sentence

The old note said:

> The current implementation deliberately uses the simple approximation rather
> than a matrix logarithm, because empirical credit matrices are not always
> embeddable in a valid continuous-time chain.

Plain-language version:

A quarterly transition matrix says where borrowers ended up after one quarter.
For example, out of all loans that started current, some stayed current, some
moved into arrears, and some defaulted. That matrix is directly observed from
quarterly data.

A continuous-time model asks a harder question: what was happening inside the
quarter? Did borrowers deteriorate gradually, jump suddenly, or default at a
specific time? To answer that, we need transition rates, not just before/after
quarterly percentages.

The simple approximation says: if one quarter is short enough, then the
quarterly change is approximately equal to rate times time:

```text
Q approx (P - I) / dt
```

This is easy to inspect. Off-diagonal entries are approximate transition rates,
and diagonal entries are exit rates. It is not perfect, but it almost never
breaks the workflow.

## Why Use A Matrix Logarithm?

In a true continuous-time Markov chain:

```text
P(dt) = exp(dt Q)
```

The matrix exponential turns transition rates `Q` into a transition matrix over
time `dt`. Therefore, if we already have `P(dt)` and want the exact `Q`, the
formal inverse is:

```text
Q = log(P(dt)) / dt
```

That is why a matrix logarithm is mathematically attractive. It asks for the
continuous-time generator whose exponential reproduces the observed quarterly
matrix.

The repo now implements this as `matrix_log_generator(P)`. It returns the
candidate generator and diagnostics rather than silently accepting the result.

## Why Empirical Credit Matrices May Not Be Embeddable

An empirical credit matrix is a transition matrix estimated from observed data:
count movements from one state to another, divide by row totals, and obtain
quarterly probabilities. It contains sampling noise, sparse rows, absorbing
states, and sometimes zero cells.

Embeddable means there exists a continuous-time generator `Q` such that:

```text
P = exp(Q dt)
```

A valid continuous-time generator must satisfy three constraints:

- off-diagonal entries are non-negative transition intensities,
- rows sum to zero,
- diagonal entries are non-positive exit rates.

Some empirical matrices have no matrix logarithm satisfying these constraints.
The numerical logarithm can produce a candidate with negative off-diagonal
rates. A negative transition intensity is not a valid continuous-time Markov
chain. It would mean a borrower has a negative instantaneous rate of moving from
one credit state to another.

The practical implementation therefore uses:

- `(P - I) / dt` as a transparent approximation,
- `matrix_log_generator(P)` as an embeddability diagnostic,
- `estimate_ctmc_generator_from_durations(...)` as the preferred estimator when
  exact transition durations are available.

The matrix logarithm is useful after the continuous-time extension, but mainly
as a diagnostic or sensitivity check unless it passes generator validity tests.

## Reversibility And Credit Migration

Reversibility means detailed balance:

```text
m_i P_ij = m_j P_ji
```

In simple language, the amount of probability flow from state `i` to state `j`
equals the reverse flow from `j` back to `i` under a reference distribution
`m`.

Credit migration is usually non-reversible because deterioration and repair are
not mirror images. A loan moving from current to 30-89 DPD is not economically
the same as a loan curing from 30-89 DPD back to current. Default and
prepayment also create terminal mass that does not flow back. Lando and
Skodeberg (2002) explicitly study rating drift and transition dynamics using
continuous observations, which is exactly the setting where downgrade/upgrade
asymmetries and duration effects become visible.

When could migration be close to reversible?

- A closed, non-terminal rating system with stable long-run state occupancy.
- Similar upgrade and downgrade mechanisms after conditioning on the invariant
  distribution.
- No absorbing default or prepayment state in the analysed subchain.
- A deliberately symmetrised diagnostic graph rather than the true migration
  dynamics.

If reversibility is empirically plausible, use the invariant distribution `pi`
and the reversible conductance `pi_i P_ij` directly. If it is not plausible,
either use a symmetrised diagnostic only for roughness checks or move to
non-symmetric Dirichlet-form tools. The repo exposes
`reversibility_diagnostics(P, m)` so this choice is visible.

## Score Smoothness Diagnostic In Plain Language

A credit score or stage scale should not behave strangely relative to how loans
actually move. If two states frequently exchange borrowers, and the score jumps
massively between those states, the score scale may be too jagged or the state
definition may hide important information.

The diagnostic computes a penalty:

```text
large migration probability * large score jump^2
```

The questions arise naturally:

| Question | Why it matters | How the repo answers it |
| --- | --- | --- |
| Are score jumps aligned with observed migration? | A big score jump is reasonable for rare severe movement, but suspicious for common mild movement. | `score_smoothness_diagnostics` lists the edges with largest energy contribution. |
| Does the stage scale vary smoothly across common transitions? | IFRS-style stages should reflect deterioration, not arbitrary discontinuities. | `dirichlet_transition_energy` compares raw stage labels or scores. |
| Are there suspiciously large score discontinuities between states that frequently exchange mass? | Frequent two-way movement with huge score gaps suggests unstable bucketing or noisy policy thresholds. | `regularize_state_scores` shows how much smoothing the migration graph implies. |

These questions still arise if the process is reversible. The difference is
that in a reversible process the energy has a cleaner probabilistic
interpretation under detailed balance. In a non-reversible credit process, the
same calculation is a diagnostic unless a non-symmetric form is explicitly used.

## Continuous-State Dirichlet Translation

The continuous-state module is useful in practice when the analyst wants to
study a latent credit-quality process, exact default timing, or how default
arises from gradual deterioration versus sudden jumps.

| Dirichlet-form topic | Practical credit translation | What the repo gains |
| --- | --- | --- |
| Regularity | The latent state space, topology, and boundary are explicit enough to support a process. | `regularity_diagnostics_for_grid` checks the finite-grid modelling contract. |
| Quasi-regularity | Needed for delicate infinite-dimensional state spaces; not binding on the finite grid. | The doc states when the finite-grid proxy stops being enough. |
| Capacity | Measures whether the default boundary is visible to the process. | `default_boundary_capacity_proxy` reports how much direct default conductance reaches the cemetery state. |
| Polar sets | Sets the process almost never hits. | `polar_state_diagnostic` flags states with negligible incoming rate. |
| Quasi-homeomorphism | Lets one transfer quasi-everywhere statements between equivalent state representations. | Useful later for state compression; currently documented, not overclaimed. |
| Locality | Distinguishes continuous local movement from jumps. | `beurling_deny_credit_decomposition` separates local nearest-neighbour energy from jump energy. |
| LeJan / carré du champ | Local squared-gradient energy of a score surface. | Local component explains whether score variation is driven by gradual credit-quality movement. |
| Cheeger energy | Metric-space squared-gradient energy. | `cheeger_energy_credit_proxy` checks whether a latent PD or score surface is unnecessarily jagged. |
| Killing | Default as mass removed from active credit states. | Killing energy isolates direct jump-to-default or boundary default behaviour. |
| Beurling-Deny decomposition | Energy split into local, jump, and killing parts. | Different continuous credit models can be compared by whether risk comes from diffusion, sudden jumps, or default killing. |

## Continuous-Time Versus Continuous-State

Continuous-time does not automatically mean continuous-state.

| Model | State space | Time | Practical use |
| --- | --- | --- | --- |
| Quarterly survival logit | Loan rows and features | discrete quarters | baseline PD with panel data |
| Finite Markov chain | delinquency states | discrete quarters | migration challenger |
| CTMC migration | delinquency states | continuous time | exact transition dates and generator intensities |
| Intensity default model | alive/default counting process | continuous time | default time modelling |
| Latent diffusion/jump/killing | continuous credit quality | continuous time | structural diagnostics and local/jump/default decomposition |

The repo uses continuous-time theory only when a function explicitly says so:
`matrix_log_generator`, `estimate_ctmc_generator_from_durations`,
`build_compensated_process_from_intervals`, and functions in
`continuous_state.py`.

## Paper Map

| Paper or source | Why it is relevant here |
| --- | --- |
| Jarrow, Lando, and Turnbull (1997) | Markov-chain credit migration with default as a credit-risk state. |
| Lando and Skodeberg (2002) | Rating transitions with continuous observations, generator estimation, and rating drift. |
| Israel, Rosenthal, and Wei (2001) | Matrix-log and generator estimation problems for empirical credit rating transition matrices. |
| Black and Cox (1976) | Structural first-passage default with a default boundary. |
| Lando (1998) | Cox-process intensity modelling for defaultable securities. |
| Duffie and Singleton (1999) | Reduced-form defaultable bond modelling with default intensity and recovery assumptions. |
| Fukushima, Oshima, and Takeda (2011) | Symmetric Markov processes, Dirichlet forms, capacity, killing, and energy. |
| Ma and Roeckner (1992) | Non-symmetric Dirichlet forms, relevant because credit migration is generally non-reversible. |

## References

- Black, F., and Cox, J. C. (1976), ["Valuing Corporate Securities: Some Effects of Bond Indenture Provisions"](https://www.jstor.org/stable/2326758).
- Duffie, D., and Singleton, K. J. (1999), ["Modeling Term Structures of Defaultable Bonds"](https://academic.oup.com/rfs/article-abstract/12/4/687/1599653).
- Fukushima, M., Oshima, Y., and Takeda, M. (2011), [*Dirichlet Forms and Symmetric Markov Processes*](https://www.degruyterbrill.com/document/doi/10.1515/9783110218091/html).
- Israel, R. B., Rosenthal, J. S., and Wei, J. Z. (2001), ["Finding Generators for Markov Chains via Empirical Transition Matrices, with Applications to Credit Ratings"](https://doi.org/10.1080/713665550).
- Jarrow, R. A., Lando, D., and Turnbull, S. M. (1997), ["A Markov Model for the Term Structure of Credit Risk Spreads"](https://academic.oup.com/rfs/article/10/2/481/1589160).
- Lando, D. (1998), ["On Cox Processes and Credit Risky Securities"](https://doi.org/10.1017/S0269964898173055).
- Lando, D., and Skodeberg, T. M. (2002), ["Analyzing Rating Transitions and Rating Drift with Continuous Observations"](https://www.sciencedirect.com/science/article/pii/S037842660100228X).
- Ma, Z.-M., and Roeckner, M. (1992), [*Introduction to the Theory of (Non-Symmetric) Dirichlet Forms*](https://link.springer.com/book/10.1007/978-3-642-77739-4).
