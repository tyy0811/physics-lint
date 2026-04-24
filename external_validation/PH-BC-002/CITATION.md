# PH-BC-002 — Boundary flux imbalance (divergence theorem)

## Scope-separation discipline (read first)

PH-BC-002's external validation separates **(i) a harness-level Gauss-
Green correctness fixture** from **(ii) the production rule's currently
supported Laplace-scope verdict behavior**. This separation is applied
per the 2026-04-24 V1-stub CRITICAL-task pattern
(`feedback_critical_rule_stub_three_layer_contract.md`): the production
rule `src/physics_lint/rules/ph_bc_002.py` is Laplace-scope only (Week
1 scope; Poisson arm raises `NotImplementedError` until source-term
plumbing lands in Week 2), while the F1 mathematical anchor — the
general Gauss-Green / divergence theorem — covers arbitrary C¹ vector
fields on Lipschitz-boundary domains.

This document does not, and must not, claim the production rule
validates general vector-flux Gauss-Green. It does claim:

- (F1) the rule's mathematical legitimacy is anchored in the Gauss-
  Green theorem in its full generality (Evans App C.2 Thm 1 / Gilbarg-
  Trudinger §2.4);
- (F2 harness-level, authoritative) a free-standing Gauss-Green
  correctness fixture for `F = (x, y)` on the unit square verifies the
  identity `∫_Ω div F dV = ∫_{∂Ω} F·n dS = 2` to roundoff on both
  triangulation and quadrilateralization, independent of the rule;
- (rule-verdict contract) the rule itself emits `∫Δu dV ≈ 0` on
  Laplace-harmonic fixtures within its V1 emitted scope, with SKIP
  paths on Poisson (Week 2 wiring) and non-laplace/poisson PDEs.

## Function-labeled citation stack

Per complete-v1.0 plan §1.3 three-function labeling. Authored during
Task 5 of the complete-v1.0 plan on 2026-04-24. F3 is absent by
structure (Gauss-Green reproduction on MMS fixtures is tautological
under the theorem's stated preconditions).

### Mathematical-legitimacy (Tier 1 structural-equivalence)

- **Primary — Gauss-Green theorem, general form.** Evans, L.C. (2010).
  *Partial Differential Equations*, 2nd ed. Graduate Studies in
  Mathematics 19. AMS. ISBN 978-0-8218-4974-3. **Appendix C.2 Theorem
  1 (Gauss-Green), section-level** per `../_harness/TEXTBOOK_AVAILABILITY.md`
  ⚠ (theorem number pending primary-source verification per §6.4).
- **Secondary framing.** Gilbarg, D. & Trudinger, N.S. (2001).
  *Elliptic Partial Differential Equations of Second Order*, 2nd ed.
  Springer Classics in Mathematics. DOI 10.1007/978-3-642-61798-0.
  **§2.4, section-level** per `../_harness/TEXTBOOK_AVAILABILITY.md`
  ⚠.
- **Structural-equivalence proof-sketch** (section-level framing
  throughout — no tight theorem-number claims per §6.4):
  1. **Gauss-Green preconditions.** For a bounded open `Ω ⊂ ℝⁿ` with
     Lipschitz boundary and a C¹ vector field `F: Ω̄ → ℝⁿ`,
     `∫_Ω div F dV = ∫_{∂Ω} F·n dS` where `n` is the outward unit
     normal. Evans App C.2 establishes this at the theorem level;
     Gilbarg-Trudinger §2.4 provides the PDE-geared framing.
  2. **Unit-square fixture satisfies preconditions.** `[0, 1]²` has a
     Lipschitz boundary (piecewise-linear convex polygon). `F = (x, y)`
     is C¹ everywhere (in fact polynomial of degree 1). So Gauss-Green
     applies.
  3. **Closed-form evaluation.** `div F = ∂F_x/∂x + ∂F_y/∂y = 1 + 1 =
     2`. Volume integral = `∫_Ω 2 dV = 2 · 1 = 2`. Boundary flux splits
     across four edges of the unit square: bottom `(y=0, n=(0,-1))`,
     `F·n = -y = 0` → integral 0; top `(y=1, n=(0,+1))`, `F·n = y = 1`
     → integral 1; left `(x=0, n=(-1,0))`, `F·n = -x = 0` → integral 0;
     right `(x=1, n=(+1,0))`, `F·n = x = 1` → integral 1. Total flux
     = 2.
  4. **Rule-family specialization.** For a Laplace-harmonic field
     `u: Ω → ℝ` (`Δu ≡ 0`), applying Gauss-Green to `F = ∇u` gives
     `∫_Ω Δu dV = ∫_{∂Ω} ∂u/∂n dS`, and both sides equal 0. The
     production rule PH-BC-002's V1 emitted quantity (`∫Δu dV`)
     measures the left-hand side's numerical evaluation and compares
     to 0; deviation is the imbalance metric. This is a direct
     specialization of the theorem to the Laplace-scope the rule
     implements in V1.

### Correctness-fixture (CI-runnable, non-credibility-claiming)

**Layer F2 (harness-level, authoritative).** `F = (x, y)` on unit
square via `external_validation/_harness/divergence.py`'s
`gauss_green_on_unit_square(mesh_type, n_refine)`:

- **Triangulation (MeshTri + ElementTriP1).** At N ∈ {4, 8, 16}
  subdivisions, LHS = 2.000000000000000, RHS = 2.000000000000000
  (exact to float64 roundoff); `|LHS − RHS| = 0`.
- **Quadrilateralization (MeshQuad + ElementQuad1).** At N ∈ {4, 8,
  16}, LHS = 2.0, RHS = 2.0000000000000004 (diff ~4e-16 at N ≥ 8);
  well inside the 1e-12 tolerance.
- **Mesh-type and refinement-level invariance.** Both values hold
  across both mesh types and all refinement levels — Gauss-Green is
  exact for F of polynomial degree 1 + Gaussian quadrature of
  sufficient order, independent of refinement.

**Rule-verdict contract (Layer RVC).** Exercises the production
rule's ACTUAL V1 Laplace-scope emitted quantity (`∫Δu dV` for a Laplace
field). Does not claim arbitrary-F Gauss-Green coverage.

- **Degree-2 harmonic `u = x² − y²`.** FD4 `Δu = 0` exactly (stencil
  exact on polynomials of degree ≤ 4, both interior and boundary).
  Rule emits `raw_value = 0` at every N ∈ {16, 32, 64} → PASS.
- **Degree-5 harmonic `u = x⁵ − 10x³y² + 5xy⁴`** (Re of z⁵). FD4
  interior is exact (degree 5 second derivatives reduce to degree 3
  and 4 polynomials, stencil still exact), boundary FD4 is 2nd-order
  so dependent on `∂⁴u` which is nonzero on degree 5. Measured:
  - N=16: raw = −2.67e-2, ratio = 3.51e-2 → **WARN** (boundary error
    above 0.01 threshold at coarse grid; documented, not a rule bug).
  - N=32: raw = −3.02e-3, ratio = 4.02e-3 → PASS.
  - N=64: raw = −3.60e-4, ratio = 4.80e-4 → PASS.
  - Ratio ~ 9× decrease per doubling — consistent with boundary
    FD4 O(h²) convergence on a nontrivial 4th derivative.
- **SKIP paths (Category 8 semantic-compatibility).**
  - Poisson → SKIP with reason "PH-BC-002 for Poisson requires
    source integration; lands in Week 2." (Documented V1 scope.)
  - Heat / wave / other PDEs → SKIP with reason "PH-BC-002 applies
    to laplace/poisson only; got <pde>." (Scope guard.)

**Rule anchor assertions** (12 tests total in `test_anchor.py`):
- Layer F2 (harness-level): 6 parametrized + 1 invariance summary test.
- Layer RVC (rule verdict): 5 tests — degree-2 PASS, degree-5 PASS at
  N≥32, degree-5 WARN at N=16 (boundary-error-above-threshold
  documentation), Poisson SKIP with Week-2 reason, non-laplace SKIP
  with scope guard.

### Borrowed-credibility (external published reproduction layers)

**F3 absent with justification.** Per complete-v1.0 plan §13
rationale: Gauss-Green reproduction on MMS fixtures is **tautological**
under the theorem's stated preconditions — the theorem holds exactly,
so "reproducing" a published numerical value for this identity would
not be a borrowed-credibility claim but a repetition of the same
closed-form derivation. The Tier-1 structural-equivalence
reproduction (Gauss-Green on the F=(x,y) fixture + Laplace-scope rule-
verdict contract) carries the credibility here. Per plan §1.2, F3-
absent-is-structural for rules whose canonical published reproduction
target would be tautological; borrowed-credibility via published
numerical baseline is not applicable.

### Supplementary calibration context

- **LeVeque 2002 FVM §2.1 conservation-form pedagogical framing.**
  LeVeque, R.J. (2002). *Finite Volume Methods for Hyperbolic
  Problems.* Cambridge Texts in Applied Mathematics 31. Cambridge
  University Press. ISBN 978-0-521-81087-6. §2.1. **Pedagogical
  framing, not reproduction.** LeVeque §2.1 presents the divergence-
  theorem rearrangement that motivates finite-volume conservation
  schemes; it is context for why PH-BC-002's boundary-flux imbalance
  metric is the right target for PDE-surrogate conformance, not a
  reproduction of a numerical result.

## Citation summary

- **Primary (mathematical-legitimacy, Tier 1)**: Evans 2010 App C.2
  Gauss-Green (section-level per `../_harness/TEXTBOOK_AVAILABILITY.md`
  ⚠) + Gilbarg-Trudinger 2001 §2.4 (section-level ⚠).
- **F2 harness-level**: `external_validation/_harness/divergence.py`
  `gauss_green_on_unit_square(mesh_type, n_refine)`. Tested at
  N ∈ {4, 8, 16} on both triangulation and quadrilateralization.
- **Calibration (Supplementary)**: LeVeque 2002 FVM §2.1 (ISBN 978-
  0-521-81087-6).
- **Pinned values**: (Layer F2) LHS = RHS = 2.0 within float64
  roundoff (~4e-16) on both mesh types at all tested refinements;
  (Layer RVC) rule raw_value = 0 exactly on `u = x² − y²`; rule
  WARNs at N=16 and PASSes at N ≥ 32 on `u = x⁵ − 10x³y² + 5xy⁴`
  with boundary-FD4 O(h²) convergence.
- **Verification date**: 2026-04-24.
- **Verification protocol**: three-layer (F1 proof-sketch + F2
  harness-level Gauss-Green on F=(x,y) + rule-verdict contract on
  Laplace-harmonic fixtures) with explicit scope-separation language.

## Pre-execution audit

PH-BC-002 is a continuous-math rule (divergence-theorem imbalance).
Per complete-v1.0 plan §6.2 Tier C enumerate-the-splits allocation
(0.1 d), the splits audited are:

- **Convex vs non-convex polygon**: V1 scope restricts to convex
  (unit square). Non-convex polygons (L-shape, re-entrant corner) are
  scope-aligned with Task 10 (PH-CON-004) and deferred here.
- **Triangular vs quadrilateral mesh**: both tested in F2
  harness-level via `mesh_type ∈ {"tri", "quad"}`. Mesh-type
  invariance of Gauss-Green on F=(x,y) is verified.
- **F smooth vs discontinuous**: smooth only in V1 (F = (x, y) is
  polynomial, trivially smooth). Discontinuous F (piecewise constants,
  characteristic functions) breaks the Gauss-Green precondition
  (F no longer C¹) and is out of V1 scope.
- **Rule-scope vs F1-scope split** (V1-stub CRITICAL-task pattern):
  the production rule PH-BC-002 is Laplace-scope only (Week 1 scope,
  `ph_bc_002.py:8`), strictly narrower than the general Gauss-Green
  anchor. The harness-level F2 fixture (arbitrary-F Gauss-Green) and
  the rule-verdict contract (Laplace-harmonic) are separated so
  CITATION.md, README, and tests do not imply broader rule coverage
  than V1 provides.

Audit outcome: F2 is harness-level authoritative (plan-diff 6 vs plan
§13 Step 3); RVC layer added on Laplace-harmonic fixture. No other
splits require reconfiguration. Audit cost 0.1 d absorbed into Task 5
budget.

## Test design

- **Harness-level fixture (F2)**: `F = (x, y)` on unit square
  `[0, 1]²` via `external_validation/_harness/divergence.py`.
- **Mesh types**: both triangulation (`MeshTri` + `ElementTriP1`)
  and quadrilateralization (`MeshQuad` + `ElementQuad1`).
- **Refinement levels**: N ∈ {4, 8, 16} for parametrized testing;
  any N would work (F polynomial degree 1 → Gaussian quadrature
  exact).
- **Rule-verdict fixture (RVC)**: `u = x² − y²` (degree-2 harmonic)
  for the exact-zero case at all N ∈ {16, 32, 64}; `u = x⁵ − 10x³y²
  + 5xy⁴` (degree-5 harmonic, Re of z⁵) for the nontrivial boundary-
  FD4 convergence case at N ∈ {16, 32, 64}.
- **DomainSpec**: `pde="laplace"`, `grid_shape=[N, N]`,
  `domain={"x": [0, 1], "y": [0, 1]}`, `periodic=False`,
  `boundary_condition={"kind": "dirichlet_homogeneous"}`,
  `field={"type": "grid", "backend": "fd", "dump_path": "p.npz"}`.
- **Wall-time budget**: < 5 s (scikit-fem mesh assembly + FD4
  Laplacian on 64×64).
- **Tests**: 12 total (6 F2 parametrized + 1 F2 invariance + 5 RVC).

## Scope note

PH-BC-002 covers the V1 Laplace-harmonic case at the rule-verdict
layer. Poisson (source-term integration) lands in Week 2; non-
laplace/poisson PDEs are out of rule scope (Category 8 SKIP with
scope guard). The F2 harness-level layer verifies the general
Gauss-Green identity for polynomial C¹ fields on Lipschitz-boundary
convex polygons; extensions to non-convex polygons (Task 10
connection), 3D domains, and C¹ non-polynomial F are out of v1.0
scope.
