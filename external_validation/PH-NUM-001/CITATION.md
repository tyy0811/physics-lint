# PH-NUM-001 — FEM quadrature exactness (CRITICAL V1-stub three-layer)

## Scope-separation discipline (read first)

PH-NUM-001 validates **the mathematical and harness-level quadrature-error
contract for controlled weak-form fixtures**. The v1.0 production rule
validates only its implemented diagnostic behavior — which, in V1, is a
**structural stub that emits `PASS` with a pass-through baseline integral
and the reason string `"qorder convergence check is a stub until V1.1"`**
when given a `MeshField`, and `SKIPPED` otherwise (`ph_num_001.py:31-57`).
The rule does **not** compute a convergence rate, does **not** compare
quadrature at orders `q` vs `2q`, and does **not** measure any variational
crime in V1.

This anchor applies the V1-stub CRITICAL three-layer pattern (Task 5 / Task 7
precedents, `feedback_critical_rule_stub_three_layer_contract.md`):

- **F1 Mathematical-legitimacy.** Classical Gauss-Legendre quadrature
  exactness + variational-crime (Ciarlet / Strang / Brenner-Scott) framing.
  Separates the exact bilinear form `a(u, v) = ∫_Ω ∇u · ∇v dx` from the
  quadrature approximation `a_h(u, v) = Σ_K Σ_q w_q ∇u(x_q) · ∇v(x_q)`.
  Insufficient quadrature changes the residual or weak-form quantity
  (variational crime); sufficient quadrature preserves the FE convergence
  rate.
- **F2 harness-level (authoritative).** `_harness/quadrature.py` integrates
  monomials and product polynomials over a scikit-fem unit-square mesh at
  varying `intorder` and compares to closed-form analytical values. Three
  scoped cases per the 2026-04-24 user-revised contract:
  - **Case A (exact):** `degree ≤ intorder` → error at float64 roundoff.
  - **Case B (under-integrated):** `degree > intorder` (with gap ≥ 3) →
    error bounded away from 0.
  - **Case C (convergence):** fix polynomial degree, sweep `intorder` →
    error drops from under-integrated regime to roundoff once `intorder`
    matches or exceeds the polynomial degree. Measured drop factor
    `errs[0] / errs[-1] ≈ 7.8 × 10^{12}` across `intorders ∈ {2, 4, 6, 8, 10}`
    at `degree = 10`.
- **Rule-verdict contract.** Rule's V1 PASS-with-reason behavior is
  exercised on both code paths (MeshField PASS + non-MeshField SKIP);
  each branch returns the documented reason string. The rule's
  `raw_value` is `field.integrate()` — a pass-through baseline integral,
  not a convergence rate. The anchor explicitly states the rule does
  **not** catch quadrature pathologies in V1.

**Wording discipline.** Do not write "PH-NUM-001 validates FEM quadrature
convergence." The production rule does not; it is a structural stub that
emits PASS with a stub reason. Write: "PH-NUM-001 validates the
mathematical and harness-level quadrature-error contract for controlled
weak-form fixtures. The v1.0 production rule validates only its
implemented diagnostic behavior."

**Bilinear form separation (required by 2026-04-24 user contract).**

- **Exact bilinear form**: `a(u, v) = ∫_Ω ∇u · ∇v dx` (continuous Poisson
  problem). For polynomial test functions on a polygonal domain, `a(u, v)`
  reduces to sums of monomial integrals `∫_Ω x^p y^q dx` which have
  closed-form values `1/((p+1)(q+1))` on the unit square.
- **Quadrature approximation**: `a_h(u, v) = Σ_K Σ_q w_q^K ∇u(x_q^K) ·
  ∇v(x_q^K)`, where each element `K` carries a quadrature rule of
  prescribed exactness. The substitution `a → a_h` is a variational
  crime; its magnitude depends on the quadrature exactness relative to
  the integrand polynomial degree.
- **What the F2 harness measures**: the **quadrature-error contribution
  alone**, by integrating known polynomials of controlled degree against
  unit weight on a fixed mesh. This isolates the quadrature-exactness
  claim from the full `a(u, v) − a_h(u, v)` variational-crime bound
  (which also involves coefficient regularity + stability constants
  from Strang's lemma). A full variational-crime test would require an
  FEM solve with varying `intorder` and measurement of `||u − u_h||_{H¹}`;
  that test is **out of V1 scope** per user's 2026-04-24 preference for
  "simple polynomial integrals over a full FEM assembly first."

## Function-labeled citation stack

Per complete-v1.0 plan §1.3 three-function labeling. Authored during
Task 11 of the complete-v1.0 plan on 2026-04-24.

### Mathematical-legitimacy (Tier 2 multi-paper)

- **Primary — FEM convergence with quadrature (general theory).**
  Ciarlet, P.G. (2002). *The Finite Element Method for Elliptic
  Problems*. SIAM Classics 40. ISBN 978-0-89871-514-9.
  **§4.1, chapter-level** per `../_harness/TEXTBOOK_AVAILABILITY.md`
  ⚠. Establishes the optimal-order FE convergence bound under
  sufficient-quadrature assumptions: for conforming FE of polynomial
  order `p` on a smooth solution with quadrature exactness at least
  `2p − 2`, the global `||u − u_h||_{H¹} = O(h^p)` rate is preserved.
- **Primary — variational crimes.** Strang, G. (1972). "Variational
  crimes in the finite element method." In A.K. Aziz (ed.), *The
  Mathematical Foundations of the Finite Element Method*, Academic
  Press. ISBN 978-0-12-068650-6. **Chapter-level** per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠. Introduces the variational-
  crime taxonomy — quadrature, non-conforming elements, inexact boundary
  conditions, curved boundaries — and bounds the resulting error via
  Strang's first and second lemmas. The quadrature crime is exactly
  the `a → a_h` substitution this anchor probes at the monomial-
  integral level.
- **Secondary framing — quadrature-crime implementation.** Brenner,
  S.C. & Scott, L.R. (2008). *The Mathematical Theory of Finite
  Element Methods*, 3rd ed. Springer Texts in Applied Mathematics 15.
  ISBN 978-0-387-75933-3. **§10.3, chapter-level** per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠. Modern treatment of the
  variational-crime framework with explicit quadrature-error bounds.

**Structural proof-sketch** (section-level framing per §6.4, no tight
theorem-number claims):

1. **Exact bilinear form vs quadrature approximation.** For a conforming
   FE discretization of `-Δ u = f` on a 2D polygon Ω with P1/P2/P3
   triangular elements, the discrete problem is "find `u_h ∈ V_h` such
   that `a(u_h, v_h) = (f, v_h)` for all `v_h ∈ V_h`." In practice the
   integrals `a(·, ·)` and `(f, ·)` are replaced by quadrature-based
   approximations `a_h(·, ·)` and `(f, ·)_h`. This substitution is the
   variational crime (Strang 1972).
2. **Gauss-Legendre exactness property.** An n-point Gauss-Legendre
   quadrature rule on `[-1, 1]` integrates polynomials of degree
   `≤ 2n − 1` exactly. On a reference triangle with a prescribed
   quadrature rule of exactness `m`, monomials of degree `≤ m` integrate
   exactly (to float precision); monomials of degree `> m` integrate
   with error bounded above by the quadrature-error-bound theorem and
   bounded below away from 0 (the quadrature is genuinely not exact).
3. **scikit-fem's intorder.** The `intorder` keyword selects a
   quadrature rule whose exactness degree is at least `intorder`; the
   library rounds up to a standard rule, so `intorder=k` gives
   exactness ≥ `k` (often equal). Consequence: a polynomial of degree
   ≤ `intorder` integrates to float64 roundoff; degree > `intorder`
   integrates with nonzero error.
4. **Case A (exact) validates step 3 forward.** Integrating `x^d` for
   `d ≤ intorder` against unit weight on the unit square should give
   `1/(d + 1)` to float64 roundoff.
5. **Case B (under-integrated) validates step 3 contrapositive.**
   Integrating `x^d` for `d > intorder` (with gap `≥ 3`) should give
   error `≥ 1e-6` (well above roundoff), demonstrating the quadrature
   crime exists.
6. **Case C (convergence) validates step 2 + step 3 jointly.** Fix a
   polynomial of degree `d = 10`; sweep `intorder ∈ {2, 4, 6, 8, 10}`.
   The error drops from under-integrated (4.3e-4 at intorder=2) to
   roundoff (5.6e-17 at intorder=10) — a `7.8 × 10^{12}` reduction,
   validating that increasing quadrature order progressively eliminates
   the crime until full exactness is reached.
7. **Does NOT prove the full Ciarlet `p + 1` rate.** The monomial-
   exactness claim (steps 4–6) is a necessary ingredient of Ciarlet's
   FEM-convergence proof, but not sufficient to establish the full
   `O(h^p)` convergence rate of `u_h → u` under h-refinement. That
   proof additionally requires: coercivity / continuity of the
   bilinear form, interpolation-error estimates, Strang's lemma
   applied at the `a_h` level, and regularity of the manufactured
   solution. This anchor does **not** measure the full `O(h^p)` rate
   in V1; the rule does **not** emit that quantity. A V1.1 tightening
   that wires a full MMS h-refinement sweep into the rule would
   extend this anchor's F2 with a fourth case.

### Correctness-fixture (CI-runnable, non-credibility-claiming)

**F2 harness-level authoritative.** `_harness/quadrature.py`.

**Case A — exact quadrature (degree ≤ intorder).** Integrate `x^d`
against unit weight on the unit square at matched `(degree, intorder)`
pairs.

| `degree` | `intorder` | Numerical | Analytical (`1/(d+1)`) | Absolute error |
|----------|------------|-----------|------------------------|----------------|
| 0 | 0 | 1.0000000000 | 1.0000000000 | 0.000e+00 |
| 1 | 1 | 0.5000000000 | 0.5000000000 | 0.000e+00 |
| 3 | 3 | 0.2500000000 | 0.2500000000 | 5.551e-16 |
| 5 | 5 | 0.1666666667 | 0.1666666667 | 1.943e-16 |
| 7 | 7 | 0.1428571429 | 0.1428571429 | 2.914e-16 |

Also tested: 2D product monomials `x^dx · y^dy` at `intorder = dx + dy`
give roundoff error (`x^2 · y^3` at `intorder=5`: err 8.3e-17).

**Case A acceptance:** error ≤ `1e-14` across all (degree, intorder)
pairs with `degree ≤ intorder`. Measured max 5.6e-16; tolerance ~17×
safety over observed.

**Case B — under-integrated (degree > intorder, gap ≥ 3).**

| `degree` | `intorder` | gap | Absolute error |
|----------|------------|-----|----------------|
| 4 | 1 | 3 | 1.567e-05 |
| 6 | 2 | 4 | 7.746e-05 |
| 8 | 3 | 5 | 2.339e-04 |
| 10 | 4 | 6 | 2.859e-06 |
| 12 | 5 | 7 | 4.834e-06 |

2D cross-check: `x^5 · y^5` at `intorder=3` (total degree 10, gap 7):
err 1.057e-04.

**Case B acceptance:** error `≥ 1e-6` across all (degree, intorder)
pairs with gap `≥ 3`. Measured min 2.9e-6; tolerance leaves ~3×
safety over observed minimum.

**Case C — convergence (fixed degree = 10, sweep intorder).**

| `intorder` | Absolute error |
|------------|----------------|
| 2 | 4.334e-04 |
| 4 | 2.859e-06 |
| 6 | 2.765e-09 |
| 8 | 1.296e-13 |
| 10 | 5.551e-17 |

**Case C acceptance:**
- Strictly non-increasing (passes `is_non_increasing` with slack 1e-15).
- Final error `≤ 1e-14` once `intorder ≥ degree`.
- Drop factor `errs[0] / errs[-1] ≥ 1e6` (measured 7.8e+12).

**Rule-verdict contract.** `ph_num_001.check(field, spec)`:

| Input | Expected `status` | Expected `reason` substring |
|-------|-------------------|------------------------------|
| `MeshField` on any basis / any DOFs | `"PASS"` | `"qorder convergence check is a stub until V1.1"` |
| `GridField` (non-mesh) | `"SKIPPED"` | `"PH-NUM-001 requires MeshField"` |

The rule's `raw_value` equals `field.integrate()` — a pass-through FE
baseline integral approximating `∫_Ω u dx`. The rule does **not** emit
a convergence rate in V1; the V1 output is a scalar integral + a
forward-looking reason string. Any future V1.1 tightening that wires
a qorder sweep into the rule must update this anchor's rule-verdict
layer in the same commit.

### Borrowed-credibility (external published reproduction layers)

**F3 absent with justification.** Per plan §19 rationale, Task 0 Step 5
F3-hunt (`docs/audits/2026-04-22-f3-hunt-results.md:78-102`), and the
2026-04-24 user-revised Task 11 F3 contract: "If no live external FEM/
quadrature benchmark is CI-executable, demote to absent-with-
justification. Ciarlet / Strang / Brenner-Scott can support F1, not F3
reproduction."

No live external reproduction target is CI-executable for PH-NUM-001 in
V1:

- **Ciarlet, Strang, Brenner-Scott** publish quadrature-convergence
  theorems in their textbooks, but tabulated error/rate values under
  varying `intorder` are typically **examples within proofs** or
  illustrative figures, not systematic reproduction targets (Task 0
  Step 5 F3-hunt:78-102).
- **Ern-Guermond 2021 Finite Elements §8.3** gives quadrature-calibration
  discussion, but again no systematic `(p, intorder, MMS)` error table
  that physics-lint's rule's emitted quantity could reproduce.
- **MOOSE FEM-convergence tutorial** confirms the standard `p + 1`
  convergence-rate theorem under sufficient quadrature, but its
  tabulated values are for its own solver implementation, not a
  peer-reviewed reproduction target.
- Rule's V1 emitted quantity is a pass-through baseline integral —
  there is no numerical convergence-rate or error-table result from
  the rule that a paper could be compared against.

No F3-INFRA-GAP risk (F3-absent is structural — the rule's V1 emitted
quantity is a baseline integral, not a quadrature-error measurement,
so no published `(p, intorder)` error table could match it anyway).
Per user's contract: "Ciarlet / Strang / Brenner-Scott can support
F1, not F3 reproduction" — they are cited in F1 as theoretical
backbone + Supplementary calibration context as pedagogical framing.

### Supplementary calibration context

- **Ern-Guermond 2021 Finite Elements §8.3.** Ern, A. & Guermond,
  J.-L. (2021). *Finite Elements I–III*. Springer Texts in Applied
  Mathematics 72–74. ISBNs 978-3-030-56340-0, 978-3-030-56922-8,
  978-3-030-57347-8. **§8.3, chapter-level** per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠. Quadrature-calibration
  discussion; **flagged: pedagogical framing, not reproduction.**
- **MOOSE FEM-convergence tutorial.**
  [mooseframework.inl.gov/modules/fem/convergence.html](https://mooseframework.inl.gov/modules/fem/convergence.html).
  INL's MOOSE framework convergence-rate tutorial. **Flagged:
  methodology reference, not reproduction.** Confirms the standard
  `p + 1` rate under sufficient quadrature for its own solver; the
  numerical values are MOOSE-internal and not a peer-reviewed
  physics-lint match.
- **Strang-Fix 2008** *An Analysis of the Finite Element Method*
  (Wellesley-Cambridge Press, ISBN 978-0-9802327-0-8) and
  **Ern-Guermond 2004** *Theory and Practice of Finite Elements*
  (Springer Applied Math Sciences 159, ISBN 978-0-387-20574-8) —
  both in `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠. **Flagged:
  background / framing only.**

## Citation summary

- **Primary (mathematical-legitimacy, Tier 2)**: Ciarlet 2002 §4.1
  (chapter-level ⚠), Strang 1972 Variational Crimes (chapter-level ⚠),
  Brenner-Scott 2008 §10.3 (chapter-level ⚠). Seven-step structural
  proof-sketch with explicit bilinear-form separation (`a(u,v)` vs
  `a_h(u,v)`) and explicit "does NOT prove the full `p + 1` rate"
  disclaimer.
- **F2 harness-level**: `external_validation/_harness/quadrature.py`
  `integrate_monomial_1d`, `integrate_product_monomial`,
  `quadrature_error_monomial_1d`, `quadrature_error_product_monomial`,
  `convergence_sweep_over_intorder`, `is_non_increasing`. No full FEM
  assembly; no MMS h-refinement; no bilinear-form comparison — pure
  monomial quadrature exactness.
- **Rule-verdict contract**: `PH-NUM-001.check()` V1 stub PASS-with-
  reason on MeshField + SKIP on non-MeshField.
- **Pinned values** (all measured 2026-04-24 on scikit-fem 12.0.1,
  Python 3.11, float64):
  - Case A exact (5 pairs): max error 5.6e-16 (tolerance 1e-14).
  - Case B under-integrated (5 pairs, gap ≥ 3): min error 2.9e-6
    (tolerance 1e-6).
  - Case C convergence (degree 10, intorders {2, 4, 6, 8, 10}):
    errors (4.3e-4, 2.9e-6, 2.8e-9, 1.3e-13, 5.6e-17); strictly
    non-increasing; drop factor 7.8e+12.
  - Rule PASS on MeshField with reason `"qorder convergence check is
    a stub until V1.1"`; rule SKIP on GridField with reason
    `"PH-NUM-001 requires MeshField"`.
- **F3**: absent-with-justification per user's 2026-04-24 Task 11
  revised F3 contract.
- **Verification date**: 2026-04-24.
- **Verification protocol**: CRITICAL three-layer (F1 Gauss-Legendre +
  variational-crime proof-sketch + F2 harness-authoritative monomial
  quadrature across three regimes + rule-verdict V1-stub PASS/SKIP
  contract + F3 absent-with-justification).

## Pre-execution audit

PH-NUM-001 is a FEM-quadrature rule with V1 structural-stub scope. Per
complete-v1.0 plan §6.2 Tier A enumerate-the-splits allocation (0.2 d),
the splits audited are:

- **Linear vs quadratic vs cubic FE (P1/P2/P3).** The harness layer
  uses `ElementTriP2` uniformly; the choice of element is orthogonal
  to the monomial-quadrature-exactness test (which depends only on
  the quadrature rule's exactness degree, not the FE basis order).
  V1.1 quadrature-convergence rule wiring may exercise P1/P2/P3
  explicitly.
- **Sufficient vs under-integrated intorder.** Both tested per Cases A
  (sufficient) and B (under-integrated).
- **Smooth MMS vs sharp-gradient.** Smooth polynomial only in V1 F2.
  Sharp-gradient manufactured solutions would test the
  interpolation-error path of Ciarlet's `p + 1` proof; out of V1
  scope since the rule doesn't emit an interpolation-error quantity.
- **Monomial quadrature vs full bilinear-form `a(u,v) − a_h(u,v)`.**
  Monomial quadrature only in V1 per 2026-04-24 user-revised Task 11
  contract ("simple polynomial integrals over a full FEM assembly
  first"). Full bilinear-form variational-crime measurement is
  deferred to V1.1 when the rule has an actual `qorder` kwarg.
- **Rule-verdict scope.** V1 stub — PASS with reason on MeshField,
  SKIP on non-MeshField. Rule-verdict layer verifies both code
  paths per CRITICAL three-layer pattern (Task 5 / Task 7
  precedents).

Audit outcome: V1 F2 scope = monomial-quadrature exactness across
three regimes at `intorder ∈ {0, 1, 2, ..., 12}` and polynomial
degrees `d ∈ {0, ..., 12}` on the unit square. Rule-verdict contract
covers both V1 code paths. Audit cost 0.2 d absorbed into Task 11
budget.

## Test design

- **Harness primitives** (`_harness/quadrature.py`):
  `integrate_monomial_1d`, `integrate_product_monomial`,
  `quadrature_error_monomial_1d`, `quadrature_error_product_monomial`,
  `convergence_sweep_over_intorder`, `is_non_increasing`.
- **Trial counts**: parametrized pairs for Cases A/B; a 5-intorder
  sweep for Case C.
- **Wall-time budget**: < 15 s (dominated by the scikit-fem basis
  construction × ~30 invocations; each is small).
- **Tests**: ~20 total (each parametrized instance is a distinct
  pytest case)
  - 5 Case A 1D exact parametrized — degree = intorder pairs.
  - 2 Case A 2D product-monomial exact.
  - 5 Case B 1D under-integrated parametrized — degree > intorder
    with gap ≥ 3.
  - 1 Case B 2D product-monomial under-integrated.
  - 3 Case C convergence — drop factor ≥ 1e6; non-increasing;
    final error ≤ 1e-14.
  - 2 Rule-verdict PASS — MeshField on smooth DOFs returns
    PASS with stub reason; `raw_value` equals
    `field.integrate()`.
  - 2 Rule-verdict SKIP — GridField input returns SKIPPED.

## Scope note

PH-NUM-001 V1 covers:

- **Mathematical-legitimacy (F1)**: Ciarlet 2002 §4.1 + Strang 1972
  + Brenner-Scott 2008 §10.3 chapter-level framing for FE
  quadrature convergence + variational-crime taxonomy + explicit
  bilinear-form separation.
- **Harness-authoritative (F2)**: monomial quadrature exactness
  across three regimes (exact, under-integrated, convergence) on
  scikit-fem unit-square P2 mesh. No full FEM assembly, no MMS
  h-refinement, no full variational-crime bound measurement.
- **Rule-verdict contract**: V1 structural stub returns PASS with
  reason `"qorder convergence check is a stub until V1.1"` on
  MeshField; SKIP on non-MeshField. `raw_value = field.integrate()`
  as pass-through baseline integral.

Out of V1 scope:

- **Full `a(u, v) − a_h(u, v)` variational-crime bound.** Would
  require an FEM solve with varying intorder and measurement of
  `||u − u_h||_{H¹}`; deferred to V1.1 when the rule ships an actual
  `qorder` kwarg.
- **MMS h-refinement with Ciarlet `p + 1` rate.** Plan §19's original
  "convergence rate matches Ciarlet's predicted `p + 1` rate within
  10%" is not what the V1 rule emits; deferred to V1.1. Per
  2026-04-24 user-revised Task 11 contract: "do not imply production
  catches all quadrature pathologies unless the rule actually
  computes them."
- **Linear vs higher-order FE basis choice exercise.** Anchor fixes
  `ElementTriP2`.
- **Sharp-gradient manufactured solutions.** Anchor uses smooth
  polynomials only.
- **Single-paper numerical reproduction target.** F3 absent per
  Task 0 F3-hunt: quadrature-convergence theorems are textbook-
  method results, not systematic reproduction tables.
