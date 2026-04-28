# PH-CON-004 — Per-element conservation-defect hotspot (2D triangulated)

## Scope-separation discipline (read first)

PH-CON-004 validates a **v1.0 2D mesh-based residual-indicator contract**.
It does not claim full adaptive finite-element solver validation or 3D
tetrahedral coverage.

The rule's emitted quantity is **`max_K / mean_K` of ``∫_K (Δ_{L²-proj zero-trace} u)² dx``**
over interior elements of a 2D triangulation (`ph_con_004.py:84-164`). The
ratio is a **conservation-defect localization indicator**: a smooth field
gives a ratio ≈ 1–5 (PASS at threshold 10); a field whose Laplacian is
locally concentrated gives a large ratio and hotspots co-located with
the concentration. The rule is **not** a Verfürth-style guaranteed
a-posteriori error estimator (`η² = ||h f||² + Σ_e ||h^(1/2) [∇u_h · n_e]||²`):
physics-lint's rule implements **only** the interior volumetric
`||Δ u_{L²-proj}||²` term, without the volumetric source `||h f||²` or
facet jumps `||h^(1/2) [∇u_h · n_e]||²`. The `scripts/check_theorem_number_framing.py`
closeout script is consistent with this framing — Verfürth 2013 is cited
at chapter-level per `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠, no
specific theorem number.

F2 splits into two scoped layers, following the Task 9 / Task 12
harness-authoritative + rule-verdict precedent:

- **F2 harness-level (authoritative localization).** On a controlled
  L-shape fixture where the true Laplacian is concentrated at the
  re-entrant corner and u vanishes on the outer boundary (Gaussian bump
  `u = exp(−α r²)` with α = 20), the top-k interior hotspot elements
  concentrate within ~1.7 element layers of the corner across multiple
  refinements. This localization is refinement-invariant in
  element-layer units: `h` halves across refinements but
  `max_r_top5 / h ≈ 1.70` at each level.
- **Rule-verdict contract.** On the same L-shape fixtures, the rule's
  `max_K / mean_K` ratio: PASSes on smooth input (ratio 2.17–4.47 at
  refined(2)–refined(4)); grows monotonically on concentrated-Laplacian
  input (ratio 8.66 → 10.85 → 75.06 as mesh refines) and transitions
  from PASS → WARN around the threshold 10.

The anchor does **not** exercise an adaptive refinement loop or a
Dörfler marker: physics-lint does not ship a marker or refiner. The
anchor tests the rule's localization + monotonic behavior under
*uniform* refinement, which is sufficient to validate the
conservation-defect-localization contract. A full DWR adaptive-FEM
solver is out of V1 scope.

## Function-labeled citation stack

Per complete-v1.0 plan §1.3. Authored during Task 10 on 2026-04-24.

### Mathematical-legitimacy (Tier 2 multi-paper)

- **Primary — residual-based a-posteriori error estimation (general
  theory).** Verfürth, R. (2013). *A Posteriori Error Estimation Techniques
  for Finite Element Methods*. Oxford Numerical Mathematics and Scientific
  Computation. ISBN 978-0-19-967942-3. **Chapters 1–4, chapter-level**
  per `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠. Establishes the
  upper-/lower-bound structure of residual-based indicators on
  triangulations and the localization property: residual magnitude
  concentrates where the FE approximation has large local error.
- **Secondary framing — DWR adaptive methods.** Bangerth, W. &
  Rannacher, R. (2003). *Adaptive Finite Element Methods for Differential
  Equations*. Birkhäuser Lectures in Mathematics, ETH Zürich. ISBN
  978-3-7643-7009-1. **Chapter-level** per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠. Dual-weighted-residual
  framing; cited as theoretical backdrop for why per-element indicators
  are useful for adaptive refinement. Physics-lint does **not** ship a
  DWR solver or adaptive marker.
- **Secondary framing — error-estimator overview.** Ainsworth, M. &
  Oden, J.T. (2000). *A Posteriori Error Estimation in Finite Element
  Analysis*. Wiley. ISBN 978-0-471-29411-5. **Chapter-level** per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠. Survey of residual,
  recovery-type, and equilibrated-flux estimators. Establishes the
  taxonomy within which physics-lint's narrower indicator sits.

**Structural proof-sketch** (section-level framing per §6.4, no tight
theorem-number claims):

1. **Classical residual estimator (reference theorem, section-level).**
   Verfürth 2013 Ch. 1 establishes: for `-Δu = f` on a Lipschitz 2D
   polygon with homogeneous Dirichlet BC and P1 FE, the element-wise
   residual indicator `η(τ, u_h)² = ||h f||²_{0,τ} + Σ_{e ⊂ ∂τ ∩ Ω}
   ||h^{1/2} [∇u_h · n_e]||²_{0,e}` satisfies an upper bound
   `||u − u_h||_{H¹} ≤ C · (Σ_τ η(τ, u_h)²)^{1/2}` with efficiency
   constants independent of `h` under standard mesh-regularity.
2. **Physics-lint rule is a narrower variant.** PH-CON-004 emits
   `∫_K (Δ_{L²-proj zero-trace} u)² dx` per interior element
   (`ph_con_004.py:107-111`). It does **not** include `||h f||²_{0,τ}`
   (no source term enters the rule) or `||h^{1/2} [∇u_h·n_e]||²_{0,e}`
   (no facet-jump term). No h-weighting is applied. The quantity is an
   **interior volumetric conservation-defect proxy**, not the full
   Verfürth estimator. Consequently, no guaranteed error bound of the
   form `||u − u_h|| ≤ C · (Σ η²)^{1/2}` is claimed by the rule or the
   anchor.
3. **Localization sub-claim.** Where the L²-projected zero-trace
   Laplacian operator has large local values — i.e., where u has large
   second derivatives *and* u vanishes on the boundary so the
   zero-trace projection is clean — the per-element residual-sq
   concentrates in that region. On the L-shape with a Gaussian bump at
   the re-entrant corner, this gives hotspots within ~2 element layers
   of the corner, monotonically across refinements.
4. **Non-localization failure mode (documented).** If u does not vanish
   on the outer boundary (e.g., the canonical L-shape exact solution
   `u = r^(2/3) sin((2/3)θ)` is nonzero on the two edges at x = 1 and
   y = 1), the zero-trace projection introduces a boundary artifact
   that dominates the interior concentration signal. This is a
   consequence of the rule's choice of operator (not a bug), and it
   scopes the test fixtures the anchor can use: F2 fixtures must vanish
   on the full boundary of the L-shape for the localization claim to
   hold.
5. **Interior-only mask.** The rule structurally excludes
   boundary-touching elements (DOF-aware mask at `ph_con_004.py:121-126`)
   so the reported ratio reflects interior localization, not boundary
   artifacts. This is essential for Point 4 — the rule would be
   meaningless without the mask because the boundary band of the
   zero-trace projection always carries the largest residual.

### Correctness-fixture (CI-runnable, non-credibility-claiming)

**F2 layer A — harness-level authoritative localization.**
`_harness/aposteriori.py` `gaussian_bump_at_corner(basis, alpha=20)` on
`l_shape_mesh(n_refine)` with `ElementTriP2()` basis. Laplacian magnitude
peaks at origin; u vanishes on outer L-shape boundary at α=20
(`exp(−20·r²_max) = exp(−40) ≈ 4e-18`).

**Measured localization** (2026-04-24):

| n_refine | n_elements | n_interior | h       | top-5 r-to-origin (min, max) | max(r)/h |
|----------|------------|------------|---------|------------------------------|----------|
| 2        | 96         | 24         | 0.2500  | (0.3727, 0.4249)             | 1.70     |
| 3        | 384        | 176        | 0.1250  | (0.1863, 0.2125)             | 1.70     |
| 4        | 1536       | 880        | 0.0625  | (0.0932, 0.1062)             | 1.70     |

**Case A acceptance:** `max_layer_distance := max(top5_r) / h ≤ 2.0` at
refined(3) and refined(4). Tolerance 2.0 calibrated from precheck
observed 1.70 with ~0.3-layer margin. Refinement-invariant in layer
units — the physical distance halves with `h`, validating the rule's
conservation-defect-localization contract.

**F2 layer B — rule-verdict contract.** Same fixtures passed to
`PH-CON-004.check()`:

| Fixture                        | n_refine | rule status | ratio  |
|--------------------------------|----------|-------------|--------|
| gaussian bump (α=20) at origin | 2        | PASS        | 8.66   |
| gaussian bump (α=20) at origin | 3        | WARN        | 10.85  |
| gaussian bump (α=20) at origin | 4        | WARN        | 75.06  |
| smooth bubble                  | 2        | PASS        | 2.17   |
| smooth bubble                  | 3        | PASS        | 3.65   |
| smooth bubble                  | 4        | PASS        | 4.47   |

**Case B acceptance:**
- PH-CON-004 PASSes on the smooth fixture across `n_refine ∈ {2, 3, 4}`
  with `ratio < 5`; rule threshold is 10 so ample margin.
- PH-CON-004 WARNs on the Gaussian-bump fixture at `n_refine ∈ {3, 4}`
  with `ratio > 10`.
- Monotonic ratio growth on the Gaussian bump: `ratio(n+1) > ratio(n)`
  across `n ∈ {2, 3, 4}` — evidence the indicator responds to
  refinement (a non-responsive indicator would not detect
  pre-asymptotic concentration).

**SKIP-path contracts.**

- `GridField` input (non-Mesh) → rule SKIPs with reason
  `"PH-CON-004 requires MeshField (scikit-fem extra)"`
  (`ph_con_004.py:91-93`). Anchor tests the scope-boundary contract.
- Constant MeshField → interior residual below scale-aware numerical-zero
  floor → rule SKIPs with `"per-element residual is numerically zero"`
  (`ph_con_004.py:141-147`). Anchor tests the well-defined-path
  contract.

### Borrowed-credibility (external published reproduction layers)

**F3 absent with justification** (pre-recorded by Task 0 Step 5 F3-hunt,
`docs/audits/2026-04-22-f3-hunt-results.md:57-65`).

Per the F3-hunt audit: the L-shape benchmark is a canonical adaptive-FEM
benchmark with a well-defined exact solution, but **effectivity-index
values depend on (a) which residual-type estimator is used (classical
residual, recovery-type, equilibrated-flux), (b) which marker is used
(maximum, Dörfler, equidistribution), (c) mesh generation specifics.**
Different peer-reviewed sources report different effectivity values on
the "same" benchmark because the precise estimator + marker + solver
triple varies. There is no single-paper reproduction target that
physics-lint's PH-CON-004 output could map onto within its measurement
framework — and the rule's narrower quantity (interior volumetric
`||Δ u||²` only, no facet jumps) widens this incompatibility further.

Per user's 2026-04-24 Task 10 revised contract: "If scikit-fem Example 22
or related benchmark is implemented locally and CI-runnable, keep as F3.
If not, demote to Supplementary or absent-with-justification." Checked:
scikit-fem Example 22 (`docs/examples/ex22.py` in the scikit-fem repo)
is **not** importable from the pip-installed scikit-fem package
(examples are repo-only); importing `docs.examples.ex22` fails. Demoted
to absent-with-justification accordingly. The helper `_harness/aposteriori.py`
locally implements the L-shape + P2-basis + localization fixture
pattern (CI-runnable), but this is a *local methodology reproduction*
of the benchmark setup, not a reproduction of a published numerical
result.

No F3-INFRA-GAP risk — F3-absent is structural (no single-paper pinned
row exists across the estimator/marker/solver configuration space) and
consistent with the rule's narrower emitted quantity.

### Supplementary calibration context

- **Becker-Rannacher 2001 DWR survey.** Becker, R. & Rannacher, R.
  (2001). "An optimal control approach to a posteriori error estimation
  in finite element methods." *Acta Numerica* 10, 1–102. DOI
  [10.1017/S0962492901000010](https://doi.org/10.1017/S0962492901000010).
  **Flagged: pedagogical framing, not reproduction.** DWR dual-weighted
  residual is a different estimator family than physics-lint's interior
  volumetric indicator; cited as backdrop for the adaptive-FEM literature
  rather than as a direct methodology match.
- **Verfürth 2013 Chs 1–4** — already in F1 as primary theoretical
  backbone; re-cited here as Supplementary for the chapter-level
  pedagogical framing of the efficiency-reliability estimator pairing.
- **Ainsworth-Oden 2000** — already in F1 as survey framing;
  re-cited here for the taxonomy of estimator types (classical residual,
  recovery-type, equilibrated-flux) that contextualizes why a single
  "effectivity-index" reproduction target cannot exist.
- **scikit-fem Example 22** (adaptive Poisson on L-shape). The anchor
  helpers in `_harness/aposteriori.py` locally reimplement the L-shape
  + P2-basis + corner-singular-class fixture pattern that scikit-fem's
  `docs/examples/ex22.py` demonstrates. This is methodology reuse, not
  a numerical reproduction — scikit-fem's example does not ship a
  pinned per-element hotspot-ratio value.

## Citation summary

- **Primary (mathematical-legitimacy, Tier 2)**: Verfürth 2013 Chs 1–4
  (chapter-level ⚠); Bangerth-Rannacher 2003 (chapter-level ⚠);
  Ainsworth-Oden 2000 (chapter-level ⚠). Structural proof-sketch
  (5 steps) framing the rule as a narrower variant of the classical
  residual estimator, with explicit no-guaranteed-error-bound claim.
- **F2 harness-level**: `external_validation/_harness/aposteriori.py`
  `l_shape_mesh`, `p2_basis`, `gaussian_bump_at_corner`, `smooth_bubble`,
  `interior_element_mask`, `per_element_residual_sq`,
  `top_k_hotspot_centroids`, `corner_distance_layers`. Tested at
  `n_refine ∈ {2, 3, 4}`.
- **F2 rule-verdict**: `PH-CON-004.check()` PASS on smooth, WARN on
  Gaussian-bump at refined(3)+ with monotonic ratio growth
  8.66 → 10.85 → 75.06.
- **Pinned values** (all measured 2026-04-24 on scikit-fem 12.0.1):
  - `max_layer_distance` at refined(3), refined(4): 1.70 ± 0.01.
  - Gaussian-bump rule ratio at refined(2), refined(3), refined(4):
    8.66, 10.85, 75.06 (monotonic).
  - Smooth-bubble rule ratio at refined(2), refined(3), refined(4):
    2.17, 3.65, 4.47 (all PASS).
- **F3**: absent-with-justification (no single-paper reproduction target
  exists; scikit-fem Ex 22 is not pip-installable).
- **Verification date**: 2026-04-24.
- **Verification protocol**: three-layer (F1 structural proof-sketch +
  F2 harness-level localization + F2 rule-verdict PASS/WARN/monotonic +
  SKIP-path contracts).

## Pre-execution audit

PH-CON-004 is a continuous-math rule (a-posteriori residual indicator)
with scikit-fem integration. Per complete-v1.0 plan §6.2 Tier A
enumerate-the-splits allocation (0.2 d), the splits audited are:

- **2D vs 3D triangulation.** V1 is 2D only (`ph_con_004.py` uses
  `MeshField` which wraps any scikit-fem `Basis`, but the anchor
  fixtures use `MeshTri.init_lshaped()` — 2D only). 3D tetrahedral
  extension is deferred to v1.2 per plan §0.9. Anchor states 2D-only
  scope explicitly.
- **Linear vs higher-order FE.** Anchor uses `ElementTriP2()` (quadratic
  Lagrange) consistent with the rule's test suite in `tests/rules/
  test_ph_con_004.py`. P1 is out of V1 test scope for this anchor; the
  rule itself accepts any `Basis` so a P1 test would work but is not
  exercised in V1 to keep the fixture calibrated.
- **Uniform vs adaptive refinement.** **Uniform only in V1 per user's
  2026-04-24 Task 10 contract** ("It does not validate arbitrary
  meshes, 3D tets, or full DWR adaptivity unless implemented").
  Physics-lint does not ship an adaptive marker or refiner; a full AFEM
  loop is out of scope. The anchor tests localization under uniform
  refinement at `n_refine ∈ {2, 3, 4}` and verifies layer-unit
  invariance (max(top5_r) / h ≈ 1.70 is the load-bearing claim).
- **Smooth vs corner-singular-equivalent fixture.** Smooth: `smooth_bubble`
  (sin sin with boundary cutoff) PASSes. Corner-singular-equivalent:
  `gaussian_bump_at_corner` (α=20) WARNs with monotonic ratio growth
  and tight corner localization. The canonical L-shape exact
  `u = r^(2/3) sin((2/3)θ)` was considered and **rejected** as a V1
  fixture (plan-diff 21) because it does not vanish on the outer
  L-shape boundary and the zero-trace projection's boundary artifact
  swamps the corner concentration signal.
- **Rule-vs-Verfürth-estimator contract.** Rule implements interior
  `||Δ_{L²-proj zero-trace} u||²` per element, not the full Verfürth
  estimator `η² = ||hf||² + Σ_e ||h^{1/2}[∇u·n_e]||²`. F1 cites
  Verfürth at chapter-level as general residual-indicator theory, and
  scopes the rule's claim to conservation-defect localization (not
  guaranteed error bound) per user's 2026-04-24 contract.

Audit outcome: V1 F2 scope = 2D triangulated L-shape with
`gaussian_bump_at_corner` + `smooth_bubble` fixtures at
`n_refine ∈ {2, 3, 4}`; no reconfiguration needed beyond plan-diffs
logged below. Audit cost 0.2 d absorbed into Task 10 budget.

## Test design

- **Harness fixtures** (`test_anchor.py`):
  `l_shape_mesh(n_refine)` → `p2_basis(mesh)` → `gaussian_bump_at_corner`
  or `smooth_bubble` → per-element residuals + localization extraction.
- **DomainSpec**: `pde="poisson"`, `grid_shape=[32, 32]` (unused — rule
  is mesh-only), `domain={"x": [-1, 1], "y": [-1, 1]}`,
  `periodic=False`, `boundary_condition={"kind": "dirichlet_homogeneous"}`,
  `field={"type": "mesh", "backend": "fd", "adapter_path": "x"}`.
- **Tests**: 13 total
  - 2 Case A parametrized localization at `n_refine ∈ {3, 4}` — top-5
    within 2 element-layers of origin.
  - 1 Case A layer-invariance — max-layer across `n_refine ∈ {2, 3, 4}`
    stays within `[1.5, 2.0]`.
  - 2 Case B rule-verdict parametrized — Gaussian-bump rule WARNs at
    `n_refine ∈ {3, 4}` with ratio > 10.
  - 3 Case B rule-verdict parametrized — smooth-bubble rule PASSes
    with ratio < 5 at `n_refine ∈ {2, 3, 4}`.
  - 1 Case B monotonic-ratio-growth on Gaussian bump.
  - 1 Case B smooth vs singular separation — smooth ratio strictly
    below singular ratio at the same refinement.
  - 3 SKIP-path contracts — GridField, constant field, very coarse
    mesh (all-boundary elements).
- **Wall-time budget**: < 10 s (dominated by the refined(4) runs with
  1536 elements).

## Scope note

PH-CON-004 V1 covers:

- **2D triangulated L-shape mesh** via `MeshTri.init_lshaped()` +
  uniform refinement.
- **Conservation-defect localization** via interior volumetric
  `||Δ_{L²-proj zero-trace} u||²` per element.
- **Hotspot indicator** `max_K / mean_K` over interior elements, with
  threshold 10 (PASS/WARN).

Out of V1 scope:

- **3D tetrahedral meshes.** Deferred to v1.2 per §0.9 scope boundary.
- **Full Verfürth residual estimator with facet-jump and source
  terms.** Rule is a narrower interior-volumetric variant.
- **Adaptive refinement loop with Dörfler marker / maximum strategy.**
  Rule is a localization indicator, not an AFEM solver.
- **DWR dual-weighted-residual estimators.** Different estimator family
  (Becker-Rannacher 2001) cited only as Supplementary context.
- **Higher-order FE (P3+).** Not exercised in V1; anchor fixes
  `ElementTriP2()`.
- **L-shape exact solution `u = r^(2/3) sin((2/3)θ)` without boundary
  cutoff.** Does not vanish on outer L-shape boundary; rule's
  zero-trace projection artifact dominates the corner concentration
  signal. Gaussian-bump-at-corner fixture substitutes (localizes
  cleanly and validates the same semantic property).
- **Single-paper numerical reproduction target.** F3 absent per Task 0
  F3-hunt — effectivity-index values vary across estimator/marker/solver
  triples across peer-reviewed sources; no pinned row exists that the
  rule's narrower emitted quantity could reproduce.
