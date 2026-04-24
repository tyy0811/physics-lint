# PH-SYM-003 — SO(2) Lie-derivative equivariance diagnostic (adapter-only, CRITICAL three-layer)

## Scope-separation discipline (read first)

PH-SYM-003 validates **an infinitesimal Lie-derivative equivariance
diagnostic under explicit SO(2) / smoothness / generator assumptions.
It does not prove global finite equivariance for arbitrary models.**

The v1.0 production rule (`ph_sym_003.py`) is **adapter-mode-only**.
In dump mode the rule emits `SKIPPED` because the Lie derivative
requires forward-mode automatic differentiation on a live callable
model, which a frozen dumped tensor cannot supply. In adapter mode,
the rule computes the single-generator Lie derivative

```
(L_A f)(x) := d/dθ |_{θ=0} f( R_θ · x )
```

via `torch.autograd.functional.jvp` on the map `θ ↦ f(R_θ · x)` at
`θ = 0` with tangent `v = 1`, then reports the per-point L² norm of
`L_A f` against a floor. `A = [[0,-1],[1,0]]` is the `so(2)`
generator; `R_θ = exp(θA)` is the rotation matrix.

This anchor applies the V1-stub / scope-narrower CRITICAL three-layer
pattern (Tasks 5, 7, 11 precedent —
`feedback_critical_rule_stub_three_layer_contract.md`,
`feedback_narrower_estimator_than_theorem.md`) plus the mathematical-
preflight-gate discipline added by the user's 2026-04-24 revised
Task 6 contract:

- **F1 Mathematical-legitimacy.** Hall 2015 §2.5 + §3.7 (section-
  level, ⚠) + Varadarajan 1984 §2.9–2.10 (section-level, ⚠) +
  Kondor-Trivedi 2018 compact-group equivariance theorem. F1
  separates the **finite ⇒ infinitesimal** direction (trivial:
  differentiate along a one-parameter subgroup) from the
  **infinitesimal ⇒ finite** direction (delicate: requires
  smoothness + connected group + generator coverage + exact
  constraint; only then does the identity-component theorem give
  finite equivariance). The empirical rule measures a **single-
  generator, pointwise-L², scalar-invariant** subset of the finite
  identity; F1 scopes the claim accordingly and does **not** claim
  the reverse implication for empirical data.
- **F2 harness-level (authoritative).** `_harness/symmetry.py` adds
  three SO(2) primitives: `so2_lie_derivative` (the `jvp` primitive
  the rule inherits), `radial_scalar` (an exactly SO(2)-invariant
  positive control), and `coord_dependent_scalar_2d` /
  `anisotropic_xx_minus_yy_2d` (non-equivariant negative controls).
  Three case-splits are exercised: (A) equivariant positive controls
  → Lie-derivative norm at roundoff floor; (B) non-equivariant
  negative controls → Lie-derivative norm clearly nonzero with
  closed-form expected value; (C) finite-vs-infinitesimal
  consistency → `||f(R_ε x) − f(x) − ε · L_A f(x)|| / ||ε · L_A
  f(x)||` scales as `O(ε)` on a negative control. The harness layer
  is the **authoritative** validation of F1 on the implemented
  quantity.
- **Rule-verdict contract.** `ph_sym_003.check()` V1 behavior is
  exercised on five SKIP paths (SO2-not-declared, dump mode,
  non-2D-grid, non-origin-centered grid, non-square domain), plus a
  live PASS path (a `CallableField` wrapping `radial_scalar` on an
  origin-centered square 2D grid → rule returns PASS with
  `lie_norm` below the 10 × tolerance × floor threshold), plus a
  live WARN / FAIL path (a `CallableField` wrapping an anisotropic
  non-equivariant map → rule returns WARN or FAIL).

**Wording discipline.** Required:

> PH-SYM-003 validates an infinitesimal Lie-derivative equivariance
> diagnostic under explicit SO(2) / smoothness / generator
> assumptions. It does not prove global finite equivariance for
> arbitrary models.

Avoid:

> PH-SYM-003 proves rotation equivariance.
> PH-SYM-003 tests that a model is SO(2)-equivariant.
> PH-SYM-003 certifies disconnected-group symmetries.

**Assumption statement** (required by the 2026-04-24 user-revised
contract). The infinitesimal Lie-derivative constraint `L_A f ≡ 0`
as tested here assumes:

- **G = SO(2)**, a connected compact abelian matrix Lie group with
  one-dimensional Lie algebra `so(2) ≅ R` and single generator `A`.
  Disconnected groups (e.g., O(2) including reflections) are out of
  V1 scope — the single-generator Lie-derivative diagnostic does
  not detect reflection-component violations.
- **Smoothness of f** on the 2D origin-centered grid, so
  differentiating `f(exp(θA) x)` at `θ = 0` is well-defined and the
  Taylor expansion to first order is exact up to `O(θ²)`.
- **Scalar-output invariance (`ρ_Y = identity`)**. The rule assumes
  the output transforms as an SO(2)-scalar. For non-trivial output
  representations (vector fields that rotate with the input),
  extending the rule to measure `ρ_Y_*(A) f(x) − ρ_X_*(A) · ∇f(x)`
  is V1.1 work; V1 tests scalar-invariance only.
- **Generator coverage.** For SO(2) (1-dim Lie algebra), the single
  generator `A` suffices. For higher-dim Lie groups (e.g., SO(3)
  with 3-dim Lie algebra), equivariance would require vanishing of
  `L_{A_i} f` for all three generators; single-generator evaluation
  is insufficient.
- **Exact constraint.** The identity `L_A f ≡ 0` holds everywhere
  in the domain, not merely at a finite sample. The rule's per-
  point L² average over the grid is a proxy; a model that satisfies
  the constraint at grid nodes but not between them would pass the
  rule yet fail finite equivariance at off-grid points.
- **2D origin-centered square domain.** The rotation `R_θ` is about
  the origin; a non-origin-centered grid would rotate points
  outside the sampled domain. The rule gates on grid center offset
  `≤ 1e-6 × max(|coord|, 1)` and square extent `|lx − ly| / max(lx,
  ly) ≤ 1e-6`.

## Function-labeled citation stack

Per complete-v1.0 plan §1.3 three-function labeling. Authored during
Task 6 of the complete-v1.0 plan on 2026-04-24 under the user's
revised stricter-preflight contract.

### Mathematical-legitimacy (Tier 1 structural-equivalence)

- **Primary — one-parameter subgroup.** Hall, B.C. (2015). *Lie
  Groups, Lie Algebras, and Representations*, 2nd ed. Springer GTM
  222. ISBN 978-3-319-13466-6. DOI 10.1007/978-3-319-13467-3. §2.5
  (one-parameter-subgroup theorem, section-level — theorem number
  pending primary-source verification per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠ after complete-v1.0 Task 0
  Step 3; secondary-source corroboration via Wikipedia "One-parameter
  group" and "Lie group–Lie algebra correspondence" articles). For
  SO(2), the one-parameter subgroup is `θ ↦ R_θ = exp(θA)` with `A =
  [[0,-1],[1,0]]`.
- **Primary — continuous-to-smooth.** Hall 2015 §3.7 (continuous-to-
  smooth for matrix Lie group homomorphisms, section-level — theorem
  number pending primary-source verification per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠; secondary-source
  corroboration via nLab "continuous homomorphisms of Lie groups are
  smooth"). Ensures that a continuous equivariant operator `f` is
  automatically smooth enough for the `L_A f` computation to be
  meaningful.
- **Primary — identity-component generation.** Varadarajan, V.S.
  (1984). *Lie Groups, Lie Algebras, and Their Representations.*
  Springer GTM 102. ISBN 978-0-387-90969-1. §2.9–2.10 (identity-
  component generation from a neighborhood of the identity, section-
  level — pending primary-source verification per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠). SO(2) is connected, so
  the identity component is SO(2) itself; a neighborhood of the
  identity generates the entire group via finite products of
  `exp(θA)` with arbitrary `θ`. This is the structural step that
  carries the infinitesimal equivariance condition to a finite
  equivariance condition — when `f` is smooth and the infinitesimal
  identity holds exactly.
- **Primary — compact-group equivariance.** Kondor, R. & Trivedi,
  S. (2018). *On the generalization of equivariance and convolution
  in neural networks to the action of compact groups.* ICML 2018,
  PMLR 80:2747–2755. [arXiv:1802.03690](https://arxiv.org/abs/1802.03690).
  Main theorem: a feedforward neural network layer is equivariant to
  a compact group action if and only if it is a generalized
  convolution with respect to that group. SO(2) is a connected
  compact abelian Lie group; its unitary dual is `Z` (Fourier
  series). The Kondor-Trivedi structure theorem is consumed here as
  F1 framing — it establishes **which operators are SO(2)-
  equivariant** (namely, the SO(2)-convolutions / Fourier
  multipliers indexed by Bessel-radial profiles); it does not supply
  the rule's empirical diagnostic, which is Hall + Varadarajan's
  infinitesimal ↔ finite story.
- **Structural-equivalence proof-sketch** (SO(2) Lie-derivative
  diagnostic on a 2D origin-centered square grid; section-level
  framing throughout — no tight theorem-number claims for Hall /
  Varadarajan per §6.4 of complete-v1.0 plan):

  1. **Group and Lie algebra.** SO(2) = `{R_θ : θ ∈ R}` with `R_θ
     = exp(θA)`, `A = [[0,-1],[1,0]]`. SO(2) is a one-dimensional
     connected compact abelian matrix Lie group with Lie algebra
     `so(2) = {θA : θ ∈ R} ≅ R`. Hall §2.5 (section-level) frames
     `θ ↦ R_θ` as the canonical one-parameter subgroup.

  2. **Finite equivariance (canonical).** A scalar-output map
     `f: R² → R` is finitely SO(2)-invariant iff `f(R_θ x) = f(x)`
     for all `θ ∈ R` and all `x` in the domain. The general
     equivariance condition is `f(ρ_X(g) x) = ρ_Y(g) f(x)`; here
     `ρ_Y = identity` gives invariance as the scalar-output case.

  3. **Finite ⇒ infinitesimal (trivial direction).** Differentiate
     `f(R_θ x) = f(x)` at `θ = 0` (assuming `f` is smooth in `x`).
     The right-hand side is `θ`-independent, so

     ```
     d/dθ |_{θ=0} f(R_θ x)  =  ∇f(x) · (A x)  =  (L_A f)(x)  =  0.
     ```

     This is the **pointwise identity** the rule evaluates. Any
     finitely invariant smooth map produces `L_A f ≡ 0`, so the
     rule returns `lie_norm = 0` up to roundoff.

  4. **Infinitesimal ⇒ finite (delicate direction).** The reverse
     implication requires four assumptions:

     (a) `f` is smooth (so Taylor expansion to all orders is valid,
         not just the first-order identity).

     (b) The group is connected (SO(2) is). Varadarajan §2.9–2.10
         (section-level) establishes that the identity component is
         generated by any neighborhood of the identity. For
         connected `G`, `L_A f ≡ 0` pointwise + smoothness ⇒ `f` is
         constant along each orbit of the exponential map; iterating
         infinitesimal steps gives `f(exp(θA) x) = f(x)` for all
         `θ`, i.e. finite invariance on the identity component.

     (c) Generator coverage. For 1-dim `so(2)` the single generator
         `A` suffices. For higher-dim Lie algebras (e.g., `so(3)`
         has 3 generators), the reverse direction requires `L_{A_i}
         f ≡ 0` for all generators, plus a compatibility argument
         via the Baker-Campbell-Hausdorff formula. V1 covers only
         `so(2)`.

     (d) The identity holds **exactly everywhere**, not just at a
         finite sample. A model that is `L_A f = 0` at the 64 × 64
         sampled grid nodes but not between them would pass the
         rule yet fail finite invariance at off-grid points. The
         rule's per-point L² average is a proxy, not a certificate.

     Under all four, Hall §2.5 + §3.7 + Varadarajan §2.9–2.10
     (all section-level ⚠) chain to: smooth `f` with `L_A f ≡ 0`
     on a connected matrix Lie group is finitely invariant on the
     identity component. Disconnected components (e.g., O(2) with
     reflections) require a separate per-component argument that
     the rule does **not** implement.

  5. **Rule's narrower-estimator scope.** The rule's emitted
     quantity is `||L_A f||_{per-point L²}` on a sampled grid. This
     is a **subset** of the full finite equivariance identity in
     four ways: (i) infinitesimal only (one `θ = 0` evaluation, not
     a finite `θ` sweep); (ii) scalar-invariant only (`ρ_Y =
     identity`); (iii) single generator (SO(2), not multi-generator
     Lie groups); (iv) sampled-grid only (not a pointwise-
     everywhere certificate). Per
     `feedback_narrower_estimator_than_theorem.md`, F1's claim is
     therefore scoped to **infinitesimal-LEE-diagnostic-of-scalar-
     SO(2)-invariance** and does **not** inherit the stronger
     guarantees of the Hall + Varadarajan + Kondor-Trivedi
     structure theorems.

  6. **Harness coverage.** The F2 layer exercises three case-splits
     (positive equivariant, negative non-equivariant, finite-vs-
     infinitesimal) that validate the implemented quantity on
     closed-form fixtures where the answer is derivable
     analytically. This is the authoritative validation; the rule-
     verdict contract verifies that `ph_sym_003.check()`'s live
     path on a `CallableField` wrapping a harness primitive
     reproduces the harness-level PASS / WARN / FAIL classification.

- **Verification status** (per `../_harness/TEXTBOOK_AVAILABILITY.md`):
  Hall 2015 §2.5 + §3.7 and Varadarajan 1984 §2.9–2.10 are all ⚠
  secondary-source-confirmed only as of Task 0 Step 3 (2026-04-23).
  Section-level framing is required; tight theorem-number framing is
  mechanically rejected by
  `scripts/check_theorem_number_framing.py`. Kondor-Trivedi 2018 is
  an arXiv preprint with DOI-identifiable mathematics; tight
  theorem-pointing is permitted for that citation.

### Correctness-fixture (CI-runnable, non-credibility-claiming)

Lives in `external_validation/PH-SYM-003/test_anchor.py` and consumes
`_harness/symmetry.py` primitives added in Task 6:

- **`so2_lie_derivative(f, grid)`** — infinitesimal Lie derivative of
  `f(R_θ x)` at `θ = 0`, computed via `torch.autograd.functional.jvp`
  with `v = 1`. This is the shared primitive both the rule and F2
  consume; they agree on the emitted quantity by construction.
- **`radial_scalar(phi)`** — SO(2)-invariant positive control.
  `f(x, y) = φ(√(x² + y²))` is exactly invariant under `R_θ` by
  construction; `L_A f ≡ 0` exactly.
- **`anisotropic_xx_minus_yy_2d`** — negative control. `f(x, y) =
  x² − y²` has `L_A f = ∇f · (A · (x, y)) = (2x, −2y) · (−y, x) =
  −2xy − 2xy = −4xy`, closed-form nonzero.
- **`coord_dependent_scalar_2d`** — negative control. `f(x, y) = x`
  has `L_A f = ∇f · (A · (x, y)) = (1, 0) · (−y, x) = −y`, closed-
  form nonzero.

**Case A — equivariant positive controls.** `identity` scalar map
and `radial_scalar` with `φ(r) = exp(−r²)`. Expected
`||L_A f||_{per-point-L²} ≤ float64 roundoff` (≤ 1e-12 on a 64 × 64
grid centered at origin).

**Case B — non-equivariant negative controls.**
`anisotropic_xx_minus_yy_2d` and `coord_dependent_scalar_2d`.
Expected `||L_A f||_{per-point-L²}` equal to the closed-form
analytical value (`||(−4xy)||_{L²}` and `||(−y)||_{L²}`
respectively) to `float64` accuracy.

**Case C — finite-vs-infinitesimal consistency.** For
`coord_dependent_scalar_2d` and `ε ∈ {1e-1, 1e-2, 1e-3, 1e-4}`,
measure

```
defect(ε) := ||f(R_ε x) − f(x) − ε · L_A f(x)||_2 / ||ε · L_A f(x)||_2.
```

Expected `defect(ε) = O(ε)` (Taylor remainder of the exponential
map); the ratio `defect(ε) / ε` is bounded above by a constant
independent of `ε`. Empirically measured in the test; used to
demonstrate that the infinitesimal quantity is the correct linear
approximation of the finite defect.

**Measured numbers** (64 × 64 origin-centered grid on `[-1, 1]²`,
float64 throughout; measured 2026-04-24 during Task 6 execution):

- Case A `||L_A f||_{per-point L²}` (identity scalar): **0.0 exactly**
  (constant function → `∇f ≡ 0`).
- Case A `||L_A f||_{per-point L²}` (radial Gaussian `exp(−r²)`, and
  `log(1 + r²)`, `r²`, `sinc(r²)`): **0.0 exactly** (radial scalars
  are SO(2)-invariant by construction → `L_A f ≡ 0` at the level of
  the jvp; no roundoff observed).
- Case B `||L_A f||_{per-point L²}` (`coord_x`, `f = x`):
  **0.5864429587908292**, matching the closed-form
  `||−y||_{per-point L²} = 0.5864429587908292` to all 16 float64
  digits.
- Case B `||L_A f||_{per-point L²}` (`anisotropic`, `f = x² − y²`):
  **1.3756613756613754**, matching the closed-form
  `||−4xy||_{per-point L²} = 1.3756613756613754` to all 16 float64
  digits.
- Case C finite-vs-infinitesimal remainder ratio on `coord_x` at
  `ε ∈ {1e-1, 1e-2, 1e-3, 1e-4}`: `4.999e-2, 5.000e-3, 5.000e-4,
  5.000e-5`. Coefficient `defect(ε) / ε` stays ≈ 0.5 (Taylor-series
  second-order coefficient of `cos(ε) − 1 ≈ −ε²/2`); successive
  ratios contract by exactly a factor of 10 (O(ε) scaling verified).

**Positive / negative distinguishability.** On the same fixture grid,
negative-to-positive ratio is `1.376 / 0.0 = ∞` (positive controls
literally hit 0.0); using the roundoff floor `2.221e-16`, the
negative control sits at `6.19e15 × floor`, many orders above the
rule's FAIL threshold (`100 × tolerance × floor = 6.66e-14`).
Distinguishability margin is enormous — PASS vs FAIL is decisive on
the V1 fixture set.

### Borrowed-credibility (external published reproduction layers)

**F3 absent with justification.** No CI-executable reproduction
target exists in V1 for PH-SYM-003's emitted quantity:

- **Modal / RotMNIST / escnn / e3nn infrastructure absent.**
  Codebase grep for `rotmnist|RotMNIST|modal|escnn|e3nn|gruver|
  lie-deriv` returns only a codespell ignore-list row, an unrelated
  Tier-A script, and the rule itself. `pyproject.toml` has no
  `equivariance` optional-dependency group; neither escnn nor e3nn
  is a dev dep. `.github/workflows/` has no Modal trigger or
  RotMNIST workflow.

- **Plan §14's two-layer RotMNIST policy cannot be built in V1
  budget.** The plan's PR-CI cached-checkpoint layer requires a
  pinned escnn C₈-steerable CNN checkpoint on a Modal volume; the
  pre-release validation layer requires Modal A100 provisioning
  plus ~2 h × 3 seeds of training. Neither piece of infrastructure
  exists; the combined build cost exceeds Task 6's 2.2 ED budget.

- **Gruver `lie-deriv` + ImageNet is opt-in and infrastructure-
  gated.** The Gruver paper's LEE baseline requires ImageNet-1k
  access (user-supplied, not bundled) + GPU infra to run on ~10⁶
  samples; physics-lint V1 has no ImageNet loader adapter and no
  GPU path to the rule's `CallableField` contract.

- **Resolution per user's 2026-04-24 revised F3 contract:** "If
  RotMNIST + escnn is CI-runnable in v1, keep it as optional
  borrowed credibility. If not, demote to Supplementary." All
  three F3 pins (RotMNIST two-layer, Gruver ImageNet, escnn / e3nn
  cross-library) are pre-demoted to Supplementary calibration
  context below. Per complete-v1.0 plan §1.2, F3-absent-is-
  structural for rules whose correctness layer is the analytical
  harness fixture and whose mathematical-legitimacy layer is the
  Tier-1 structural-equivalence proof-sketch above.

- **Author-measured SO(2) fixture numbers in Correctness-fixture
  above are NEW measurements, not reproductions of any published
  baseline.** Per the user's 2026-04-24 contract: "Do not describe
  author-measured SO(2) fixture numbers as published reproduction."

### Supplementary calibration context

**Theoretical framing only — NOT reproduction.** The following
references motivate why equivariance matters and calibrate the
scale on which the rule's diagnostic operates. They are **not**
reproduced in V1 CI.

- **Cohen, T.S. & Welling, M. (2016).** *Group Equivariant
  Convolutional Networks.* ICML 2016, arXiv:1602.07576. Canonical
  G-equivariant CNN reference; establishes that discrete-rotation
  equivariance reduces test error on RotMNIST to 2.28%. PH-SYM-003
  inherits the "equivariant networks are a real thing" motivation
  but does not test any P4CNN-family model in V1.

- **Weiler, M. & Cesa, G. (2019).** *General E(2)-Equivariant
  Steerable CNNs.* NeurIPS 2019, arXiv:1911.08251. Published
  RotMNIST test error 0.705 ± 0.025%. Establishes that continuous-
  rotation steerable layers further reduce error. The LEE-based
  diagnostic in PH-SYM-003 is the "is this particular model
  actually equivariant?" checker complementary to the "construct
  equivariant layers" program of Weiler-Cesa.

- **Gruver, N., Finzi, M., Goldblum, M., & Wilson, A.G. (2023).**
  *The Lie Derivative for Measuring Learned Equivariance.* ICLR
  2023, [arXiv:2210.02984](https://arxiv.org/abs/2210.02984). The
  direct source of the Lie-derivative equivariance-error metric
  (LEE). PH-SYM-003's emitted quantity — per-point L² norm of
  `d/dθ |_{θ=0} f(R_θ x)` — is the scalar-output SO(2) specialization
  of Gruver's LEE (their paper covers multi-dim Lie groups on
  ImageNet-scale classifiers). Physics-lint inherits the diagnostic
  but does not reproduce Gruver's ImageNet numbers; Gruver's LEE
  values on ImageNet classifiers are not directly comparable to
  physics-lint's LEE on author-constructed 64 × 64 fixtures.

- **Weiler-Forré-Verlinde-Welling (2025).** *Equivariant and
  Coordinate Independent Convolutional Networks.* World Scientific,
  DOI 10.1142/14143, ISBN 978-981-98-0662-1. Geometric-deep-
  learning backbone per complete-v1.0 plan §0.3; section-level ⚠.
  Background context for why Lie-derivative diagnostics sit at the
  core of the equivariance-verification program.

## Citation summary

- **Primary (mathematical-legitimacy)**: Hall 2015 §2.5 + §3.7
  (section-level ⚠); Varadarajan 1984 §2.9–2.10 (section-level ⚠);
  Kondor-Trivedi 2018 (arXiv:1802.03690).
- **Correctness-fixture (F2)**: `_harness/symmetry.py` primitives
  + three case-splits (positive, negative, finite-vs-infinitesimal)
  measured in `test_anchor.py`.
- **Borrowed-credibility (F3)**: absent with justification (F3-
  INFRA-GAP; plan §14 two-layer RotMNIST / escnn / Modal / Gruver
  infrastructure does not exist in V1).
- **Supplementary calibration context**: Cohen-Welling ICML 2016,
  Weiler-Cesa NeurIPS 2019, Gruver et al. ICLR 2023, Weiler-Forré-
  Verlinde-Welling 2025.
- **Pinned value**: verdict-based — PASS on `radial_scalar`
  (`||L_A f||` ≤ roundoff floor), WARN / FAIL on
  `anisotropic_xx_minus_yy_2d` (`||L_A f||` at the closed-form
  nonzero value). Exact numbers recorded in the Correctness-fixture
  subsection once measured.
- **Verification date**: 2026-04-24 (Task 6 of complete-v1.0 plan).
- **Verification protocol**: three-layer CRITICAL pattern (F1 + F2
  + rule-verdict contract) with mathematical-preflight-gate
  discipline (F1 proof sketch written before test implementation
  per 2026-04-24 user contract).

## Pre-execution audit

Per `docs/audits/2026-04-24-task-6-preflight.md`. Highlights:

- Rule V1 emitted quantity identified and scoped to infinitesimal
  scalar-invariant SO(2) diagnostic (narrower than finite-
  equivariance theorem).
- F1 / finite ↔ infinitesimal separation authored before any test
  code.
- F3 executability infrastructure audit: all F3 pins absent in V1
  → classified F3-INFRA-GAP → pre-demoted to F3-absent +
  Supplementary.
- CRITICAL three-layer pattern applied (Tasks 5, 7, 11 precedent);
  rule-verdict contract layer exercises the live PASS / WARN / FAIL
  path in addition to the five SKIP gates in `ph_sym_003.py:36-68`.

## Scope note

PH-SYM-003 covers the continuous SO(2) Lie-derivative diagnostic for
scalar-output invariant maps on 2D origin-centered square domains in
adapter mode. The following are explicitly **out of V1 scope**:

- **Finite-rotation equivariance tests** on non-infinitesimal `θ`
  values (would require a `θ` sweep; V1 only evaluates at `θ = 0`).
- **Non-scalar outputs** (vector / tensor fields with non-trivial
  `ρ_Y`). The rule would false-PASS on a correctly equivariant
  vector field; V1.1 work.
- **Higher-dimensional Lie groups** (SO(3), SE(3)). Would require
  multi-generator evaluation and BCH compatibility argument.
- **Disconnected groups** (O(2), E(2)). The single-generator Lie-
  derivative diagnostic does not detect reflection-component
  violations.
- **Non-origin-centered or non-square domains**. Rule gates SKIP.
- **Dump mode** (frozen tensor input). Rule gates SKIP.
