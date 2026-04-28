# PH-SYM-004 — Translation equivariance (CRITICAL V1-stub three-layer)

## Scope-separation discipline (read first)

PH-SYM-004 validates **the mathematical and harness-level translation-
equivariance contract for controlled operators**. The v1.0 production
rule validates only its implemented rule-verdict behavior — which, in
V1, is **a structural stub that always emits `SKIPPED`** once past its
declared-symmetry + periodicity gates (`ph_sym_004.py:36-52`). True
translation equivariance is a *model property* (`f(roll(x)) == roll(f(x))`
on a live callable) and requires adapter-mode plumbing that lands in
V1.1.

This anchor applies the V1-stub CRITICAL three-layer pattern (Task 5
precedent `feedback_critical_rule_stub_three_layer_contract.md`):

- **F1 Mathematical-legitimacy.** Classical group-theoretic equivariance
  plus the convolution theorem: for translation operator `T_a` and a
  convolution / Fourier-multiplier operator `K`, `K(T_a u) = T_a K(u)`
  on a periodic domain. Cited backbone: Kondor-Trivedi 2018 (compact-
  group equivariance theorem) + Li et al. 2021 FNO §2 convolution-theorem
  derivation.
- **F2 harness-level (authoritative).** `_harness/symmetry.py` adds four
  controlled operators: identity (trivially equivariant), circular
  convolution 1D/2D (equivariant by convolution theorem), Fourier
  multiplier 1D/2D (equivariant by Fourier-shift theorem), and a
  coordinate-dependent multiplication (deliberately *non*-equivariant
  negative control). `shift_commutation_error` measures
  `||K(T_s u) − T_s K(u)||_2 / ||K(T_s u)||_2`. Measured ≤ 3.75e-16
  across 100 random 2D trials on all three equivariant operators
  (float64 roundoff); 0.09–0.87 on the coord-dependent negative
  control. The harness layer is the **authoritative** F1 validation.
- **Rule-verdict contract.** Rule `ph_sym_004.check()` V1 behavior is
  exercised on all three SKIP-path branches (not-declared,
  non-periodic, V1-stub deferral); each branch returns `status="SKIPPED"`
  with the documented reason. The rule does **not** compute
  `shift_commutation_error` in V1 — any future V1.x tightening that
  wires an adapter-mode live-callable path must update this anchor's
  rule-verdict layer in the same commit.

**Wording discipline.** Do not write "PH-SYM-004 validates translation
equivariance of FNO layers." The production rule does not; it is a
structural stub. Write: "PH-SYM-004 validates the mathematical and
harness-level translation-equivariance contract for controlled
operators. The v1.0 production rule validates only its implemented
rule-verdict behavior."

**Assumption statement (required by the 2026-04-24 user-revised
contract).** The equivariance identity `K(T_a u) = T_a K(u)` as tested
here assumes:

- **Periodic domain** (grid is a discrete torus `Z_N` or `Z_{Nx} × Z_{Ny}`).
- **Grid-aligned shifts only** (integer `s`; continuous sub-grid shifts
  would require interpolation and are out of V1 scope).
- **Same input and output grid** (operator preserves shape and dtype).
- **Consistent translation action** (same `torch.roll(...)` convention
  on both sides of the identity).
- **No boundary artifacts** (periodic `torch.roll` has no boundary; any
  fixture that introduces Dirichlet/Neumann padding or wave-reflection
  would require a deliberately non-equivariant test, covered by the
  coord-dependent-multiplication negative control).

## Function-labeled citation stack

Per complete-v1.0 plan §1.3 three-function labeling. Authored during
Task 7 of the complete-v1.0 plan on 2026-04-24.

### Mathematical-legitimacy (Tier 1 structural-equivariance)

- **Primary — compact-group equivariance theorem.** Kondor, R. &
  Trivedi, S. (2018). "On the generalization of equivariance and
  convolution in neural networks to the action of compact groups."
  *Proceedings of the 35th International Conference on Machine
  Learning*, PMLR 80:2747–2755.
  [arXiv:1802.03690](https://arxiv.org/abs/1802.03690). Main theorem:
  a feedforward neural network layer is equivariant to a compact group
  action if and only if it is a generalized convolution with respect
  to that group. For the abelian compact group `Z_N` (1D) / `Z_N × Z_N`
  (2D), "generalized convolution" specializes to ordinary circular
  convolution, giving the well-known translation-equivariance of
  convolutional layers on periodic grids.
- **Primary — FNO convolution theorem.** Li, Z., Kovachki, N., Azizzadenesheli,
  K., Liu, B., Bhattacharya, K., Stuart, A. & Anandkumar, A. (2021).
  "Fourier Neural Operator for Parametric Partial Differential
  Equations." *ICLR 2021*.
  [arXiv:2010.08895](https://arxiv.org/abs/2010.08895). §2 derives the
  spectral convolution `K(u) = F^{-1}(R · F(u))` as a translation-
  equivariant linear operator on a periodic domain, via the convolution
  theorem `F(a * b) = F(a) · F(b)`. Physics-lint's Fourier-multiplier
  harness operator is the periodic-grid special case of FNO's spectral
  convolution (fixed rather than learned multiplier).
- **Structural-equivalence proof-sketch** (section-level framing — all
  cited references are either arXiv preprints with DOI-identifiable
  mathematics, so tight theorem-pointing is permitted here; no
  textbook `⚠` status applies):
  1. **Translation action on a periodic grid.** For `u ∈ R^N` (1D)
     or `u ∈ R^{Nx × Ny}` (2D), the shift `T_s u = torch.roll(u, s)` is
     a group action of `Z_N` (resp. `Z_{Nx} × Z_{Ny}`) on the signal
     space. This group is abelian and compact (finite cyclic).
  2. **Convolution theorem (Li §2).** The discrete Fourier transform
     `F: R^N → C^N` satisfies `F(a * b) = F(a) ⊙ F(b)` where `*` is
     circular convolution and `⊙` is elementwise product. Equivalently,
     multiplication in Fourier space is convolution in signal space.
  3. **Shift theorem.** `F(T_s u)[k] = e^{-2πi·k·s/N} · F(u)[k]`;
     translation in signal space becomes phase-rotation in Fourier
     space. Phase rotation commutes with elementwise multiplication by
     any fixed multiplier `m(k)`: `m(k) · e^{-2πi·k·s/N} · F(u)[k] =
     e^{-2πi·k·s/N} · m(k) · F(u)[k]`. Inverse-FFT of both sides gives
     `K(T_s u) = T_s K(u)` for `K = F^{-1}(m · F(·))`. This is
     Li §2 applied to fixed multipliers.
  4. **Kondor-Trivedi generalization.** The convolution-theorem
     argument in step 3 is the `G = Z_N` specialization of
     Kondor-Trivedi 2018's main theorem: for any compact group `G`,
     the `G`-equivariant linear maps between `G`-input and `G`-output
     feature spaces are exactly the generalized convolutions. On
     `Z_N` (abelian, unitary dual `= Z_N`), generalized convolution
     is ordinary circular convolution.
  5. **Harness coverage.** The F2 layer exercises three operator
     families — identity, circular convolution, Fourier multiplier —
     each of which satisfies the hypotheses of step 3 (fixed
     multiplier in Fourier space), so each commutes with shifts to
     within float-precision roundoff. The coordinate-dependent
     multiplication `K(u)(x) = w(x) · u(x)` with `w(x)` position-
     dependent violates the hypothesis ("multiplier must depend only
     on frequency, not on position") and therefore breaks
     equivariance by construction — the negative control validates
     the measurement framework.

### Correctness-fixture (CI-runnable, non-credibility-claiming)

**F2 harness-level authoritative.** `_harness/symmetry.py` extensions:

| Operator | Layer type | Expected behavior |
|----------|------------|-------------------|
| `identity_op` | equivariant (trivial) | error exactly 0 |
| `circular_convolution_1d(kernel)` | equivariant (convolution theorem) | error ≤ 1e-14 (float64 roundoff) |
| `circular_convolution_2d(kernel)` | equivariant | error ≤ 1e-14 |
| `fourier_multiplier_1d(n, seed)` | equivariant (Fourier-multiplier) | error ≤ 1e-14 |
| `fourier_multiplier_2d(nx, ny, seed)` | equivariant | error ≤ 1e-14 |
| `coord_dependent_multiply_1d(n)` | **non-equivariant** (negative control) | error > 0.05 |
| `coord_dependent_multiply_2d(nx, ny)` | **non-equivariant** | error > 0.05 |

**Measured values (2026-04-24, 100 random trials each on 2D `nx=ny=32`,
float64):**

| Operator | Max error | Min error (for non-equivariant) |
|----------|-----------|---------------------------------|
| identity | 0.00e+00  | — |
| circular_convolution_2d | 3.14e-16 | — |
| fourier_multiplier_2d | 3.75e-16 | — |
| coord_dependent_multiply_2d | 8.69e-01 | 9.17e-02 |

**Acceptance bands:**
- Equivariant ops: `shift_commutation_error ≤ 1e-14` across 100 random
  `(x, shift, seed)` trials in 1D and 2D. Tolerance ~30× safety over
  observed 3.75e-16 max; easily distinguished from non-equivariant
  regime (≥ 9.2e-02).
- Non-equivariant negative control: `shift_commutation_error > 0.05`
  across 100 random trials. Enforces that the measurement framework
  distinguishes equivariant from non-equivariant operators.

The plan's original tolerance `< 1e-5` (plan §15 step 3) was calibrated
for a random FNO-layer with learnable parameters in float32; it is
vastly looser than what the simpler controlled operators achieve in
float64 (`1e-14` with 20× safety over observed roundoff). Plan-diff 26
tightens tolerance to match the harness operators' actual floor.

**Rule-verdict contract.** `ph_sym_004.check(field, spec)` returns
`RuleResult(status="SKIPPED", ...)` with a reason string identifying
which SKIP branch fired:

| Spec | Expected SKIP reason substring |
|------|--------------------------------|
| No `translation_x` or `translation_y` in `spec.symmetries` | `"no translation_x or translation_y declared"` |
| `spec.periodic == False` + symmetry declared | `"periodic-only in V1"` |
| Periodic + symmetry declared + past gates | `"V1 structural stub"` |

The rule never returns PASS or WARN in V1 by design — the author of
`ph_sym_004.py` documented that offline field invariance
`||roll(u) − u|| / ||roll(u)||` is bounded above by 2.0 via the
triangle inequality and does not distinguish equivariant from
non-equivariant inputs, so the false-pass metric was removed rather
than shipped (`ph_sym_004.py:8-14`). V1.1 will replace the stub with
adapter-mode plumbing that compares `f(roll(x))` against `roll(f(x))`
on a live callable — the V1 SKIP reason points forward to that work.

### Borrowed-credibility (external published reproduction layers)

**F3 absent with justification.** Per plan §15 rationale and the
2026-04-24 user-revised Task 7 F3 contract: "If Helwig / FNO / equivariant-
operator reproduction is not CI-executable in v1, demote to
Supplementary."

No live external reproduction target is CI-executable for PH-SYM-004
in V1:

- **Helwig 2023** (GERNS / group-equivariant reaction networks) reports
  equivariance-error numbers on specific learned GERNS architectures
  trained on benchmarks; reproducing those numbers requires training
  or loading those specific models (not the rule's emitted quantity).
- **Li et al. 2021 (FNO)** Appendix reports PDE-solver RMSE numbers for
  FNO on Burgers, Darcy, and Navier-Stokes — none of which are
  equivariance-error metrics that physics-lint's rule or harness
  could reproduce. The FNO paper's §2 mathematical derivation is
  consumed in F1, not as a reproduction target.
- **Kondor-Trivedi 2018** is theoretical (main theorem on equivariant
  linear maps); no numerical table to reproduce.

Per user's revised contract: "Do not force borrowed credibility by
citing theory as if it were an executable reproduction." Helwig 2023
§2.2 Lemma 3.1 is moved to Supplementary calibration context (see
below). F3 is absent-with-justification.

No F3-INFRA-GAP risk (F3-absent is structural — the rule's emitted
quantity in V1 is `SKIPPED`, not a numerical equivariance-error, so no
published baseline could be comparable to it anyway).

### Supplementary calibration context

- **Helwig 2023 GERNS** — Helwig, J., Zhang, X., Fu, C., Kurtin, J.,
  Wojtas, S. & Ji, S. (2023). "Group equivariant Fourier neural
  operators for partial differential equations." *ICML 2023*.
  [arXiv:2306.05697](https://arxiv.org/abs/2306.05697). **§2.2 Lemma
  3.1** (equivariance of the Fourier convolutional layer under group
  actions including translation). **Flagged: theoretical framing, not
  reproduction.** Lemma 3.1 is a theoretical result; the paper does
  not ship a CI-reproducible equivariance-error benchmark table that
  physics-lint could match. Cited here for pedagogical continuity —
  the Helwig proof of translation equivariance of FNO-style layers
  specializes the same convolution-theorem argument that Kondor-Trivedi
  and Li §2 establish.
- **Cohen-Welling G-CNN** — Cohen, T.S. & Welling, M. (2016). "Group
  Equivariant Convolutional Networks." *ICML 2016*.
  [arXiv:1602.07576](https://arxiv.org/abs/1602.07576). Foundational
  G-CNN paper. **Flagged: pedagogical framing.** Relevant for the
  discrete-group case (the PH-SYM-001/002 anchors already consume this
  framing for C4/Z2); included here as backbone for the general
  discrete-abelian translation case (`Z_N`).

## Citation summary

- **Primary (mathematical-legitimacy, Tier 1)**: Kondor-Trivedi 2018
  main theorem (arXiv:1802.03690); Li et al. 2021 FNO §2 convolution
  theorem (arXiv:2010.08895). Five-step structural proof-sketch with
  explicit assumptions (periodic domain, grid-aligned shifts, same
  input/output grid, consistent translation action).
- **F2 harness-level**: `external_validation/_harness/symmetry.py`
  `shift_commutation_error` on seven operators (identity + circular
  conv 1D + circular conv 2D + Fourier mult 1D + Fourier mult 2D +
  coord-dep mul 1D + coord-dep mul 2D). 100-trial stability sweep in
  2D.
- **Rule-verdict contract**: `PH-SYM-004` V1 stub SKIP verification on
  all three SKIP-path branches.
- **Pinned values** (all measured 2026-04-24 on float64):
  - Identity: error = 0 exactly.
  - Circular conv 2D (5×5 kernel): max error 3.14e-16 over 100 trials.
  - Fourier multiplier 2D (32×32): max error 3.75e-16 over 100 trials.
  - Coord-dependent mul 2D (32×32): error ∈ [9.17e-02, 8.69e-01] over
    100 trials.
  - Rule SKIP on not-declared: `"no translation_x or translation_y
    declared"`.
  - Rule SKIP on non-periodic: `"periodic-only in V1"`.
  - Rule SKIP on V1-stub: `"V1 structural stub"`.
- **F3**: absent-with-justification per user's 2026-04-24 Task 7
  revised F3 contract.
- **Verification date**: 2026-04-24.
- **Verification protocol**: CRITICAL three-layer (F1 Kondor-Trivedi
  + Li §2 proof-sketch + F2 harness-authoritative shift-commutation +
  rule-verdict V1-stub SKIP contract + F3 absent-with-justification).

## Pre-execution audit

PH-SYM-004 is an equivariance-structural rule (discrete-group-action
commutation check). Per complete-v1.0 plan §6.2 Tier C enumerate-the-
splits allocation (0.1 d), the splits audited are:

- **Continuous shift vs grid-aligned shift.** Grid-aligned only in V1.
  Continuous shifts require sub-grid interpolation which introduces
  its own approximation error unrelated to the equivariance property;
  deferred to V2.
- **1D vs 2D.** Both in V1 F2 scope. 1D exercises the basic shift
  theorem; 2D exercises the Cartesian product `Z_{Nx} × Z_{Ny}`.
- **Periodic vs Dirichlet-padded.** Periodic only in V1. Dirichlet
  padding breaks periodic translation equivariance by construction
  (the padding values are fixed in space); V1.1 may add a "compatible
  padding convention" mode if the rule's adapter layer supports it.
- **Equivariant vs non-equivariant operators.** Both present in F2
  per user's 2026-04-24 contract requirement. Non-equivariant negative
  control validates the measurement framework.
- **Rule verdict scope.** V1 stub — SKIPPED on all code paths past
  the precondition gates. Rule-verdict layer verifies the three SKIP
  branches (not-declared / non-periodic / V1-stub). Per CRITICAL
  three-layer pattern (2026-04-24 Task 5 precedent).

Audit outcome: V1 F2 scope = controlled-operator harness with four
equivariant + one non-equivariant families, tested in both 1D and 2D
at 100 random trials. Rule-verdict contract covers all three V1 SKIP
branches. No reconfiguration needed beyond plan-diffs logged below.
Audit cost 0.1 d absorbed into Task 7 budget.

## Test design

- **Harness primitives** (extended in `_harness/symmetry.py`):
  `shift_commutation_error`, `identity_op`, `circular_convolution_1d`,
  `circular_convolution_2d`, `fourier_multiplier_1d`,
  `fourier_multiplier_2d`, `coord_dependent_multiply_1d`,
  `coord_dependent_multiply_2d`.
- **Trial counts**: 100 random `(x, shift, seed)` trials per 2D
  operator, per the plan's "100 random trials" acceptance spec;
  1D single-shift parametrized to keep runtime bounded.
- **Wall-time budget**: < 10 s (pure torch on CPU, no ML library).
- **Tests**: 36 total (each parametrized instance is a distinct pytest
  case)
  - 12 Case A 1D parametrized — 3 operators × 4 shifts; error ≤ 1e-14.
  - 12 Case A 2D parametrized — 3 operators × 4 (sx, sy) pairs; error
    ≤ 1e-14.
  - 1 Case A 2D stability — 100 random trials per operator, max
    ≤ 1e-14 on each.
  - 3 Case B 1D parametrized — coord-dep-mul × 3 shifts; error > 0.05.
  - 3 Case B 2D parametrized — coord-dep-mul × 3 (sx, sy) pairs;
    error > 0.05.
  - 1 Case B stability — 100 random trials in 2D, min error > 0.05.
  - 3 Rule-verdict contract — SKIP with "not declared" / "periodic-
    only in V1" / "V1 structural stub" reasons.
  - 1 Rule-verdict invariance — rule SKIPs regardless of whether
    input is equivariant or non-equivariant (V1 stub doesn't measure).

## Scope note

PH-SYM-004 V1 covers:

- **Mathematical-legitimacy (F1)**: Kondor-Trivedi 2018 + Li et al.
  2021 FNO §2 convolution-theorem framing for translation equivariance
  on periodic domains with grid-aligned shifts.
- **Harness-authoritative (F2)**: controlled operators — identity,
  circular convolution (1D, 2D), Fourier multiplier (1D, 2D) —
  verified equivariant at float64 roundoff; coord-dependent
  multiplication verified non-equivariant at 9e-02 to 9e-01
  magnitude.
- **Rule-verdict contract**: V1 structural stub SKIPs on all three
  gate branches (not-declared / non-periodic / V1-stub deferral).

Out of V1 scope:

- **Live FNO-layer equivariance check.** Plan §15 step 3 "random
  FNO-layer + random input + random shift" is replaced by controlled
  harness operators (identity / circular-conv / Fourier-multiplier)
  per the 2026-04-24 user-revised Task 7 F2 contract ("Use known
  translation-equivariant operators"). FNO-layer equivariance is
  structurally the same shift-commutation property but requires a
  `neuraloperator` or `pytorch_fno` library dependency that the
  harness avoids.
- **Continuous / sub-grid shifts.** Grid-aligned (integer) only in V1.
- **Dirichlet-padded or Neumann-padded non-periodic domains.**
  Periodic only in V1.
- **Adapter-mode live-callable equivariance check on the production
  rule.** The rule's V1 behavior is `SKIPPED`. V1.1 will replace with
  an adapter-mode implementation — this anchor's rule-verdict layer
  must be updated in that same commit.
- **Continuous-group equivariance** (full `R^2` translation, not
  `Z^2`). Physics-lint operates on discrete grids; continuous-group
  equivariance is ill-defined at the discretization level without
  interpolation.
