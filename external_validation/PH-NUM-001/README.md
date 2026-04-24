# PH-NUM-001 external-validation anchor

**Scope separation (read first):** PH-NUM-001 validates **the mathematical
and harness-level quadrature-error contract for controlled weak-form
fixtures**. The v1.0 production rule validates only its implemented
diagnostic behavior — which, in V1, is a **structural stub that emits
`PASS` with a pass-through baseline integral and the reason string
`"qorder convergence check is a stub until V1.1"`** when given a
`MeshField`, and `SKIPPED` otherwise (`ph_num_001.py:31-57`). The rule
does **not** compute a convergence rate, does **not** compare
quadrature at orders `q` vs `2q`, and does **not** measure any
variational crime in V1.

Per 2026-04-24 user-revised Task 11 contract (CRITICAL three-layer
pattern, Task 5 / Task 7 precedents):

- **F1 Mathematical-legitimacy:** Gauss-Legendre exactness + variational-
  crime framing. Classical result: an `n`-point Gauss rule integrates
  polynomials of degree `≤ 2n − 1` exactly. Variational crime (Strang
  1972) replaces exact `a(u, v) = ∫ ∇u·∇v dx` with quadrature
  `a_h(u, v) = Σ_K Σ_q w_q ∇u(x_q)·∇v(x_q)`. Ciarlet §4.1 +
  Brenner-Scott §10.3 (all chapter-level ⚠) give the convergence
  bounds under sufficient quadrature.
- **F2 harness-level (authoritative):** `_harness/quadrature.py`
  integrates monomials `x^d` and product polynomials `x^dx · y^dy`
  against unit weight on a scikit-fem unit-square P2 mesh, compares
  to closed-form analytical values, and demonstrates three regimes
  per user's recommended fixtures:
  - **Case A (exact):** `degree ≤ intorder` → error at float64
    roundoff. Measured max 5.6e-16 across 7 pairs; tolerance 1e-14.
  - **Case B (under-integrated):** `degree > intorder` with gap ≥ 3
    → error ≥ 1e-6 (measured min 2.9e-6 across 6 pairs).
  - **Case C (convergence):** fix `degree = 10`, sweep
    `intorder ∈ {2, 4, 6, 8, 10}` → errors drop from 4.3e-4 to
    5.6e-17 (factor 7.8 × 10^{12}), strictly non-increasing.
- **Rule-verdict contract:** rule returns `PASS` with the V1.1-stub
  reason on any `MeshField`; rule returns `SKIPPED` with the
  MeshField-required reason on `GridField`. `raw_value` is
  `field.integrate()` — pass-through baseline integral, not a
  convergence rate.
- **F3 Borrowed-credibility:** **absent with justification.** No live
  external FEM/quadrature benchmark is CI-executable; per user's
  2026-04-24 revised F3 contract: "Ciarlet / Strang / Brenner-Scott
  can support F1, not F3 reproduction." Task 0 F3-hunt confirmed
  (2026-04-22): quadrature-convergence theorems are textbook-method
  results, not systematic reproduction tables.
- **Supplementary calibration context:** Ern-Guermond 2021 §8.3 +
  MOOSE FEM-convergence tutorial, both flagged as pedagogical
  framing / methodology reference — not reproduction.

**Wording discipline.** Do not write "PH-NUM-001 validates FEM
quadrature convergence." The production rule does not; it is a
structural stub emitting PASS with a stub reason. Write: "PH-NUM-001
validates the mathematical and harness-level quadrature-error contract
for controlled weak-form fixtures. The v1.0 production rule validates
only its implemented diagnostic behavior."

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-NUM-001/ -v
```

Expected: 20 passed in < 15 s (5 Case A 1D exact parametrized + 2
Case A 2D product-monomial + 5 Case B 1D under-integrated parametrized
+ 1 Case B 2D product-monomial + 3 Case C convergence + 2 rule-verdict
PASS + 2 rule-verdict SKIP).

Requires scikit-fem 12.0.1+ (already in `requirements.txt`).

## Citation stack (three-function-labeled)

See `CITATION.md` for the full function-labeled citation stack.
Summary:

- **F1 Mathematical-legitimacy** (Tier 2 multi-paper): Ciarlet 2002
  §4.1 (chapter-level ⚠) — FEM convergence with quadrature;
  Strang 1972 Variational Crimes (chapter-level ⚠) — variational-
  crime taxonomy + Strang's lemma; Brenner-Scott 2008 §10.3
  (chapter-level ⚠) — modern treatment. Seven-step structural proof-
  sketch with explicit bilinear-form separation (`a(u, v)` vs
  `a_h(u, v)`) and explicit "does NOT prove the full `p + 1` rate"
  disclaimer.
- **F2 Correctness-fixture (harness-level, authoritative)**:
  `external_validation/_harness/quadrature.py` — monomial-quadrature
  exactness across three regimes (exact / under-integrated /
  convergence). No full FEM assembly, no MMS h-refinement, no full
  variational-crime measurement.
- **Rule-verdict contract**: `PH-NUM-001.check()` V1 stub PASS-with-
  reason on MeshField + SKIP on non-MeshField.
- **F3 Borrowed-credibility**: **absent with justification** per
  user's 2026-04-24 revised F3 contract. No CI-executable
  reproduction target.
- **Supplementary calibration context**: Ern-Guermond 2021 §8.3 +
  MOOSE FEM-convergence tutorial (flagged pedagogical / methodology).

## Bilinear-form separation (key F1 scope truth)

The anchor's F1 proof-sketch explicitly separates:

- **Exact bilinear form** `a(u, v) = ∫_Ω ∇u · ∇v dx` — closed-form
  value for polynomial test functions on the unit square.
- **Quadrature approximation** `a_h(u, v) = Σ_K Σ_q w_q^K ∇u(x_q^K)
  · ∇v(x_q^K)` — variational crime whose magnitude depends on
  quadrature exactness vs integrand polynomial degree.

**What F2 measures:** the **quadrature-error contribution alone**,
by integrating known polynomials of controlled degree against unit
weight on a fixed mesh. This isolates the quadrature-exactness claim
from the full `a(u, v) − a_h(u, v)` variational-crime bound (which
also involves coefficient regularity + stability constants from
Strang's lemma). A full variational-crime measurement is out of V1
scope per user's 2026-04-24 preference for "simple polynomial
integrals over a full FEM assembly first."

## V1 stub scope-truth observation

The rule's docstring (`ph_num_001.py:3-15`) explicitly documents why
the V1 qorder check is a stub: `MeshField.integrate` does not expose a
`qorder` kwarg in V1, so the rule cannot compare quadrature at orders
`q` and `2q`. It ships as a structural stub: the rule module exists,
the rule ID is in the registry, and the CLI surface is stable. V1.1
can plug in the real q-vs-2q check without breaking any public API.
In V1 the rule emits `PASS` with a pass-through `raw_value` equal to
the baseline integral — it does not fabricate a convergence claim.

V1.1 will replace the stub with an actual q-vs-2q comparison using
the same harness primitives this anchor's F2 layer exercises. When
that lands, this anchor's rule-verdict contract must be updated in
the same commit to switch from "rule PASSes with V1.1-stub reason"
to "rule emits convergence rate matching harness-level measurement."

## Plan-diffs (29 cumulative across Tier-B execution)

See `test_anchor.py` module docstring for diffs 27–29 (Task 11). Diffs
1–26 are from Tasks 2, 3, 4, 5, 7, 8, 9, 10, 12. Summary of Task 11
diffs:

27. Plan §19 step 4 "MMS fixture with analytical solution; vary
    `intorder ∈ {1, 2, 3, 4}`; measure convergence rate against
    element order `p ∈ {1, 2, 3}`; assert rate matches Ciarlet's
    theoretical prediction within 10%" → replaced with simpler
    polynomial-exactness fixtures (Gauss-Legendre exactness /
    under-integration / convergence sweep) per 2026-04-24 user-
    revised Task 11 contract ("I would prefer simple polynomial
    integrals over a full FEM assembly first"). Full MMS h-refinement
    with `p + 1` rate is deferred to V1.1 when the rule has a
    `qorder` kwarg. V1 anchor scope matches what the rule actually
    validates.
28. CRITICAL three-layer pattern applied (Task 5 / Task 7
    precedents): rule-verdict contract layer added to verify the V1-
    stub PASS-with-reason behavior on MeshField + SKIP on non-
    MeshField. Anchor docs explicitly state the rule does **not**
    compute a convergence rate in V1 and does **not** catch
    quadrature pathologies — the rule's V1 `raw_value` is a pass-
    through baseline integral.
29. Plan §19 F3 already-absent status reinforced per 2026-04-24 user-
    revised F3 contract: "Ciarlet / Strang / Brenner-Scott can
    support F1, not F3 reproduction." Ern-Guermond 2021 §8.3 + MOOSE
    FEM-convergence tutorial moved to Supplementary calibration
    context with explicit "pedagogical framing / methodology
    reference, not reproduction" flags. No attempt to promote
    textbook quadrature-convergence theorems to reproduction layer.
