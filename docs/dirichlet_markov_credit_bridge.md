# Dirichlet-Form Markov Theory In The Credit Migration Module

This note explains how part of the basics of the Markov-chain and Dirichlet-form material from my course on Dirichlet forms at Tor Vergata can be used in the credit-risk repo without forcing abstract theory where it does not belong.

## Bottom Line

The 1-to-1 correspondence from markov chains and functional analysis objects in the course is directly useful for the Markov-chain challenger model part of this repo, but at the right level:

- Use it strongly for finite-state credit migration, transition matrices, semigroups, generators, absorbing states, and Green/fundamental matrices.
- Use it carefully for Dirichlet-energy diagnostics on score or stage functions over the migration graph.
- Explicitly avoided is overclaiming regularity, quasi-regularity, capacity, polar sets, LeJan energy measures, or Cheeger energy in the first credit version. Those are better reserved for a later continuous-state or continuous-time extensions of the credit risk repo.

The practical credit translation is:

| Course object | Finite-state credit object | Repo implementation |
| --- | --- | --- |
| State space `X` | Delinquency states plus terminal states | `current`, `dpd_1_29`, `dpd_30_89`, `default`, `prepay_mature` |
| Kernel `p_t(x, dy)` | Transition probabilities | `transition_matrix` |
| Semigroup `T_t` | Multi-period transition matrix | `n_step_transition_matrix(P, n)` |
| Generator `L` / `Q` | Quarterly jump-rate approximation | `transition_generator(P)` |
| Cemetery state | Default or prepay/mature terminal state | absorbing rows |
| Green/resolvent intuition | Expected visits before absorption | `absorption_summary(P)` |
| Dirichlet form | Transition-weighted roughness of a state function | `dirichlet_transition_energy(P, f)` |
| Invariant sets | Closed communicating classes / absorbing components | default and prepay/mature states |

## What The Course Correspondence Adds

### 1. Transition Matrix As Kernel

In the notes, a Markov kernel maps a starting point to a probability measure over next states. In a finite credit state space, that kernel is simply a row of a transition matrix:

```text
P_ij = P(next_state = j | current_state = i)
```

This is exactly what the credit migration challenger model should estimate. The repo now has `fit_markov_transition_model(...)` in `src/credit_risk_lab/econometrics/markov.py`.

### 2. Matrix Powers As Semigroup

The course repeatedly uses the semigroup property:

```text
T_(s+t) = T_s T_t
```

For a quarterly migration matrix this becomes:

```text
P^(n+m) = P^n P^m
```

That gives a clean way to calculate multi-quarter migration probabilities and a Markov-implied default probability over 4 quarters, 8 quarters, or lifetime horizons.

### 3. Generator As Continuous-Time Bridge

The finite-state generator is the natural bridge to your intended continuous-time default modelling. For a one-quarter matrix, the first-order approximation is:

```text
Q = (P - I) / dt
```

The off-diagonal entries are transition intensities and rows sum to zero. This creates room for later continuous-time work:

- estimate transition intensities from event times,
- model default as an absorbing jump,
- compare quarterly matrix estimates to continuous-time generator estimates,
- connect default counting processes to reduced-form intensity modelling.

The current implementation deliberately uses the simple approximation rather than a matrix logarithm, because empirical credit matrices are not always embeddable in a valid continuous-time chain.

### 4. Absorbing States As Cemetery States

The notes use a cemetery state to make a killed process conservative after it leaves the active state space. The credit version is natural:

- active states are performing or delinquent loans,
- default and prepay/maturity remove the loan from active performance,
- adding `default` and `prepay_mature` as absorbing states keeps the augmented chain row-stochastic.

This is useful because it shows the same idea in credit language: active-book attrition is a loss of mass unless terminal outcomes are included.

### 5. Green/Fundamental Matrix For Expected Time Before Absorption

The notes introduce Green/resolvent ideas to study accumulated occupation before long-run outcomes. In finite absorbing Markov chains, the same idea appears as the fundamental matrix:

```text
N = (I - P_TT)^(-1)
```

where `P_TT` is the transient-to-transient block. In credit risk this gives:

- expected quarters before default/prepay/maturity,
- expected time spent in delinquency states,
- eventual probability of default versus prepayment/maturity.

This is a strong as it makes Markov challenger more usefull than just a transition-count table.

### 6. Dirichlet Energy As A Score Smoothness Diagnostic

The course interprets a Dirichlet form as an energy measuring how much a function varies under the dynamics. In a credit migration graph, a function can be:

- a numeric stage label,
- a risk score,
- a provision proxy,
- a manually assigned credit quality scale.

A finite-state energy diagnostic has the form:

```text
E(f) = 1/2 sum_i m_i sum_j P_ij (f_i - f_j)^2
```

Because credit migration is usually non-reversible, the repo symmetrises the jump conductance by default for this diagnostic.

This is a good econometrics-heavy extension because it lets you ask:

- Are score jumps aligned with observed migration?
- Does the stage scale vary smoothly across common transitions?
- Are there suspiciously large score discontinuities between states that frequently exchange mass?

## What Should Not Be Forced Into V1

Several lecture topics are mathematically important but too abstract for a first (discrete) credit-risk implementation.

### Regularity And Quasi-Regularity

Regular and quasi-regular Dirichlet forms matter when the state space has delicate topology or when one wants to construct a process from an analytic form. A finite delinquency-state chain has no such difficulty. The state space is finite, measurable, and fully explicit.

Possible future use:

- latent continuous credit-quality state,
- continuous-time diffusion/jump process for creditworthiness,
- point barriers or singular transition regions,
- topology-sensitive state compression.

### Capacity, Polar Sets, Quasi-Homeomorphism

These are not needed to estimate a finite transition matrix. They become relevant only if the repo later models a continuous latent risk process where certain boundaries or points are negligible for the process but still matter analytically.

Possible future use:

- continuous credit quality process with default boundary,
- singular borrower states that are rarely or never hit,
- quasi-everywhere statements about default boundaries.

### Locality, LeJan, Carre Du Champ, Cheeger Energy

These are mainly useful for continuous-state local dynamics. Credit migration between delinquency types is a jump process, so the Beurling-Deny jump part is the more relevant conceptual bridge. The continuous local part is not the right first object.

Possible future use:

- continuous latent creditworthiness diffusion,
- score dynamics with a metric structure,
- energy regularisation for smooth macro-credit state surfaces.

## Recommended Markov Roadmap

### V1: Finite-State Migration Challenger

Implemented location:

- `src/credit_risk_lab/econometrics/markov.py`

Current capabilities:

- build observed one-step transition panel,
- estimate row-stochastic transition matrix,
- force default and prepay/maturity as absorbing states,
- compute multi-period matrix powers,
- approximate a generator,
- compute absorption probabilities and expected time to absorption,
- compute a Dirichlet-style score roughness diagnostic,
- estimate grouped transition matrices for segment or macro-regime style conditioning,
- compute Markov-implied default PDs for challenger comparison.

### V2: Covariate-Dependent Markov Model

Next upgrade:

- estimate `P(next_state | current_state, borrower features, macro variables)`,
- use multinomial logit or separate transition logits,
- compare Markov-implied default probabilities to survival-logit PDs,
- explain stage migration through transition probabilities rather than only through realised summaries.

### V3: Macro-Sensitive Markov Scenario Engine

Planned upgrade:

- estimate transition matrices conditional on macro regimes,
- produce baseline/downside/upside migration matrices,
- avoid double-counting by choosing one primary macro channel for PD or migration.

### V4: Continuous-Time Default And Counting Processes

Planned upgrade:

- estimate continuous-time transition intensities when exact event dates are available,
- model default as an absorbing jump time,
- write the default indicator as a counting process,
- connect compensated default counting processes to intensity-based credit modelling.

### V5: Dirichlet-Form Regularisation Or Diagnostics

Planned upgrade:

- use transition energy to regularise state scores or rating scales,
- penalise excessive roughness over commonly observed migration edges,
- investigate whether a proposed credit score is dynamically coherent with observed borrower movement.

## Interview Framing

The clean interview version is:

> The survival logit is my primary PD model because it directly estimates a next-period default hazard from loan-quarter panel data. A Markov model is the natural challenger because credit deterioration is state-based. My Markov extension treats delinquency buckets as a finite state space, estimates the transition kernel, uses matrix powers as the semigroup, adds default and prepayment as absorbing cemetery states, and uses a generator as the bridge to later continuous-time intensity modelling.

A stronger econometrics version is:

> My Markov-chain coursework gave me a functional-analytic dictionary: kernel, semigroup, generator, process, and energy form. In this repo I use the finite-state version of that dictionary for credit migration. The abstract regularity and capacity theory is not needed for the first implementation, but it gives a roadmap for future continuous-state or continuous-time default models.

## References

- Fukushima, M., Oshima, Y., and Takeda, M. (2011), [*Dirichlet Forms and Symmetric Markov Processes*](https://www.degruyterbrill.com/document/doi/10.1515/9783110218091/html), 2nd revised and extended edition.
- Ma, Z.-M., and Roeckner, M. (1992), [*Introduction to the Theory of (Non-Symmetric) Dirichlet Forms*](https://link.springer.com/book/10.1007/978-3-642-77739-4).
- Jarrow, R. A., Lando, D., and Turnbull, S. M. (1997), ["A Markov Model for the Term Structure of Credit Risk Spreads"](https://academic.oup.com/rfs/article/10/2/481/1589160).
- Lando, D., and Skodeberg, T. M. (2002), ["Analyzing Rating Transitions and Rating Drift with Continuous Observations"](https://www.sciencedirect.com/science/article/pii/S037842660100228X).
- Kemeny, J. G., and Snell, J. L. (1976), *Finite Markov Chains*.
