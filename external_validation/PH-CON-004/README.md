# PH-CON-004 external-validation anchor

**Scope separation (read first):** PH-CON-004 validates a **v1.0 2D
mesh-based residual-indicator contract**. It does not claim full
adaptive finite-element solver validation or 3D tetrahedral coverage.

The rule emits `max_K / mean_K` of `∫_K (Δ_{L²-proj zero-trace} u)² dx`
over interior elements of a 2D triangulation. It is a
**conservation-defect localization indicator** (not a Verfürth
guaranteed-error-bound estimator — physics-lint's rule omits the
`||h f||²` source and `||h^{1/2} [∇u · n_e]||²` facet-jump terms that
classical residual estimators require).

The F2 fixture splits into two scoped layers:

- **F2 harness-level (authoritative localization):** on L-shape mesh
  + Gaussian-bump fixture `u = exp(−20 r²)` centered at the re-entrant
  corner, top-k hotspot elements concentrate within ~1.7 element-layers
  of the corner across refinements (refinement-invariant in layer units).
- **Rule-verdict contract:** PH-CON-004 PASSes on smooth fixtures
  (ratio 2.2–4.5); WARNs on concentrated-Laplacian fixtures (ratio
  8.7 → 10.8 → 75.1 monotonically as mesh refines).

**Fixture discovery (plan-diff 21, documented in CITATION.md):** the
canonical L-shape benchmark exact solution `u = r^(2/3) sin((2/3)θ)`
does **not** give corner localization under PH-CON-004's zero-trace
projection — it is not zero on the outer L-shape boundary, so projection
artifacts dominate over the corner-singularity signal. The Gaussian bump
`u = exp(−20 r²)` vanishes on the boundary at float64 precision
(`exp(−20·r²_max) ≈ 4e-18`) and gives the clean localization signal the
rule's interior-volumetric quantity is designed to detect.

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-CON-004/ -v
```

Expected: 13 passed in < 10 s (2 Case A localization + 1 Case A
layer-invariance + 5 Case B rule-verdict parametrized (2 bump WARN +
3 smooth PASS) + 1 Case B monotonicity + 1 Case B smooth-vs-singular
separation + 3 SKIP contracts).

Requires scikit-fem 12.0.1+ (already in `requirements.txt`).

## Citation stack (three-function-labeled)

See `CITATION.md` for the full function-labeled citation stack.
Summary:

- **F1 Mathematical-legitimacy** (Tier 2 multi-paper): Verfürth 2013
  Chs 1–4 (chapter-level per `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠)
  + Bangerth-Rannacher 2003 (chapter-level ⚠) + Ainsworth-Oden 2000
  (chapter-level ⚠). Five-step proof-sketch framing the rule as a
  narrower variant of the classical residual estimator, with explicit
  **no-guaranteed-error-bound** claim. F1 scopes the rule's
  legitimacy to **conservation-defect localization**, not full
  residual-estimator reliability.
- **F2 Correctness-fixture (harness-level, authoritative localization)**:
  `external_validation/_harness/aposteriori.py`. On L-shape +
  Gaussian-bump fixture, top-5 hotspot centroids land at
  `max(r)/h = 1.70 ± 0.01` element-layers across `n_refine ∈ {2, 3, 4}`
  — refinement-invariant in layer units. Tolerance 2.0 layers.
- **F2 Rule-verdict contract**: PH-CON-004 PASSes on smooth fixtures
  (ratios 2.17, 3.65, 4.47 at n_refine 2, 3, 4); WARNs on
  Gaussian-bump fixtures (ratios 8.66, 10.85, 75.06 — monotonic growth
  under refinement, transition PASS → WARN around threshold 10).
- **F3 Borrowed-credibility**: **absent with justification** — no
  single-paper L-shape benchmark pinned row exists across peer-reviewed
  sources because effectivity-index values depend on the estimator +
  marker + solver triple (Task 0 Step 5 F3-hunt,
  `docs/audits/2026-04-22-f3-hunt-results.md:57-65`). scikit-fem Example
  22 is not pip-importable from the installed package (examples ship in
  the repo, not the wheel); the anchor's L-shape fixture pattern is
  locally reimplemented in `_harness/aposteriori.py`. Per 2026-04-24
  user-revised F3 contract: "If scikit-fem Example 22 or related is
  implemented locally and CI-runnable, keep as F3; else demote to
  Supplementary or absent-with-justification." Demoted to
  absent-with-justification here.
- **Supplementary calibration context**: Becker-Rannacher 2001 DWR
  survey (*Acta Numerica* 10, 1–102, DOI
  10.1017/S0962492901000010, flagged: pedagogical framing, different
  estimator family); Verfürth 2013 + Ainsworth-Oden 2000 (pedagogical
  framing, re-cited from F1); scikit-fem Ex 22 pattern (methodology
  reuse, not a numerical reproduction target).

## Rule-vs-Verfürth-estimator contract (key scope truth)

PH-CON-004 emits interior `∫_K (Δ_{L²-proj zero-trace} u)² dx` per
element. Verfürth residual estimator is `η² = ||h f||² + Σ_e
||h^{1/2} [∇u · n_e]||²` per element. The rule implements **neither
the volumetric source term nor the facet-jump term**, and it does
**not h-weight**. Consequently:

- The rule does **not** provide a reliability estimate (upper bound on
  `||u − u_h||_{H¹}`).
- The rule does **not** provide an efficiency estimate (lower bound on
  `||u − u_h||`).
- The rule **does** provide conservation-defect localization: on
  fixtures where the zero-trace projection is clean, hotspots
  co-locate with regions of large second-derivative concentration.

The anchor's F1 proof-sketch step 2 explicitly flags this distinction.
Any future V1.x tightening of the rule toward a full residual estimator
would need to add source + facet-jump terms and revisit the F1 claim.

## Plan-diffs (22 cumulative across Tier-B execution)

See `test_anchor.py` module docstring for diffs 19–22 (Task 10). Diffs
1–18 are from Tasks 2, 3, 4, 5, 8, 9, 12. Summary of Task 10 diffs:

19. Plan §18 "Verfürth 2013 Thm 1.12 residual estimator upper/lower bound"
    → scoped to general residual-estimator theory. Physics-lint's rule
    implements only the interior volumetric `||Δ u||²` term, **not** the
    Verfürth estimator with `||hf||² + Σ_e ||h^{1/2}[∇u·n_e]||²`. F1
    claims **conservation-defect localization**, explicitly not
    guaranteed error bound, per 2026-04-24 user-revised Task 10
    contract.
20. Plan §18 step 5 "reproduce scikit-fem Example 22 adaptive-Poisson
    fixture" → scikit-fem's `docs/examples/ex22.py` is not importable
    from pip-installed scikit-fem. Replaced with locally-implemented
    L-shape + P2 + Gaussian-bump fixture in `_harness/aposteriori.py`
    (CI-runnable, tests same localization property).
21. Plan §18 step 5 "hotspots within 2 element-layers of the L-corner"
    on canonical exact `u = r^(2/3) sin((2/3)θ)` → exact solution does
    not vanish on the outer L-shape boundary, so zero-trace projection
    introduces boundary artifacts that mask the corner signal. Replaced
    with Gaussian-bump `u = exp(−20 r²)` fixture that vanishes on the
    boundary at float precision. Measured: top-5 hotspots within 1.7
    element-layers of origin across refined(2)–refined(4), acceptance
    tolerance 2.0 layers.
22. Plan §18 enumerate-the-splits item (c) "uniform refinement vs
    adaptive (both tested)" → uniform only in V1. Physics-lint does not
    ship an adaptive marker or refiner; full AFEM loop is out of V1
    scope per user's 2026-04-24 contract. Anchor tests localization
    under uniform refinement at `n_refine ∈ {2, 3, 4}`.
