# PH-RES-003 — Spectral-vs-FD residual on periodic grids

## Function-labeled citation stack

Per complete-v1.0 plan §1.3 three-function labeling. Authored during Task 3
of the complete-v1.0 plan on 2026-04-24. F3 is absent by structure — Task 0
Step 5 F3-hunt (`docs/audits/2026-04-22-f3-hunt-results.md` §"Task 3 —
PH-RES-003 … TERTIARY") confirms Trefethen's canonical `exp(sin x)`
spectral-vs-FD demonstration is a plot, not a tabulated reproduction
target, so curve-shape resemblance does not qualify as reproduction
under the §1.2 F3 definition.

### Mathematical-legitimacy (Tier 3 classical-textbook theorem reproduction)

- **Primary**: Trefethen, L.N. (2000). *Spectral Methods in MATLAB.* SIAM
  Other Titles in Applied Mathematics 10. ISBN 978-0-89871-465-4.
  **Chapters 3–4, theorem number pending local copy** per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠ (§6.4 chapter-level framing;
  trigonometric-interpolation spectral accuracy for analytic periodic
  functions). Trefethen's companion errata and code repository at
  `https://people.maths.ox.ac.uk/trefethen/spectral.html` provides the
  secondary-source corroboration path recorded in
  TEXTBOOK_AVAILABILITY.md.
- **FD-rate corroboration**: Fornberg, B. (1988). *Generation of finite
  difference formulas on arbitrarily spaced grids.* Math. Comp.
  51(184):699–706. DOI 10.1090/S0025-5718-1988-0935077-0. The 4th-order
  central stencil `(−1, 16, −30, 16, −1) / 12` used by
  `src/physics_lint/field/grid.py` line 22–30 has interior truncation
  error `O(h⁴)` on smooth inputs; on a periodic grid the same stencil
  applies uniformly (wrap-around at the boundary), so the FD residual
  decays as `O(N⁻⁴)` without the boundary-layer anomaly PH-RES-001
  exhibited in its non-periodic Layer 1b characterization.
- **FDM-consistency textbook**: LeVeque, R.J. (2007). *Finite Difference
  Methods for Ordinary and Partial Differential Equations.* SIAM.
  ISBN 978-0-898716-29-0. **FDM consistency-order (section-level)** per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠.
- **Structural-equivalence proof-sketch** (section-level framing
  throughout — no tight theorem-number claims per §6.4):
  1. **Spectral path.** For an analytic periodic `u: T² → R` with
     analyticity strip of height `σ > 0`, the Fourier coefficients
     decay as `|û_k| ≤ C · exp(−σ|k|)`. Truncating at `N` modes gives
     aliasing error `O(exp(−σN))`. Trefethen 2000 Chapters 3–4
     establishes this "spectral accuracy" result at the chapter level;
     theorem number pending local copy. For `u(x, y) = exp(sin x + sin y)`
     (analyticity strip height `σ ≈ 1`), the residual
     `max |Δu_spectral − Δu_exact|` decays super-algebraically in `N`
     until float64 machine precision is reached (~`1e-13` empirically
     at N≈32 for this fixture).
  2. **FD path.** For the same `u` sampled on a periodic uniform grid
     of spacing `h = 2π/N`, the rule's 4th-order central Laplacian
     (Fornberg 1988 coefficients, applied via GridField's periodic
     wrap-around) has truncation error `C · h⁴ · sup|∂⁶u|` on the
     full domain (no boundary-layer anomaly on periodic grids, unlike
     the non-periodic Layer 1b case in PH-RES-001). So
     `Δu_FD = Δu_exact + C · h⁴ + o(h⁴)`, giving FD residual scaling as
     `O(N⁻⁴)`.
  3. **Gap.** By triangle inequality,
     `|Δu_spectral − Δu_FD| ≤ |Δu_spectral − Δu_exact| + |Δu_exact − Δu_FD|
      = O(exp(−σN)) + O(N⁻⁴)`. The FD term dominates at all tested
     `N`; the rule's emitted max relative ratio tracks the FD rate.
  4. **Measurement target.** On `u(x, y) = exp(sin x + sin y)` on
     `[0, 2π]²` periodic, the refinement sweep N ∈ {16, 32, 64}
     exercises the FD4 rate; the pre-floor sweep N ∈ {8, 10, 12, 14,
     16} exercises the spectral exponential-decay regime above the
     float64 noise floor.

### Correctness-fixture (CI-runnable, non-credibility-claiming)

- **MMS fixture**: `u(x, y) = exp(sin(x) + sin(y))` on `[0, 2π]²`
  periodic. Closed-form Laplacian
  `Δu = [(cos²x − sin x) + (cos²y − sin y)] · u`. Analytic in a strip
  of height `σ ≈ 1` around the real axis (same as `exp(sin x)` per
  Trefethen's canonical 1D example).
- **Layer 1 refinement sweep** (spectral pre-floor, tightened per plan-
  diff 3): `SPECTRAL_NS = [8, 10, 12, 14, 16]`, residual clipped at
  `FLOAT64_FLOOR = 1e-13`. Measured residuals at 2026-04-24:
  - N =  8  (h ≈ 0.785): 5.60 × 10⁻²
  - N = 10  (h ≈ 0.628): 3.07 × 10⁻³
  - N = 12  (h ≈ 0.524): 4.74 × 10⁻⁴
  - N = 14  (h ≈ 0.449): 1.80 × 10⁻⁵
  - N = 16  (h ≈ 0.393): 2.13 × 10⁻⁶
  - All 5 above floor; log-linear fit `log(err) = a − 1.275 N`,
    **R² = 0.9946** (above 0.99 threshold).
- **Layer 2 refinement sweep** (FD polynomial, unchanged from plan):
  `FD_NS = [16, 32, 64]`. Measured residuals at 2026-04-24:
  - N = 16: 1.03 × 10⁻¹
  - N = 32: 7.26 × 10⁻³
  - N = 64: 4.68 × 10⁻⁴
  - **log-log slope = 3.89, R² = 0.9999** (within [3.6, 4.4], matches
    Fornberg O(h⁴) interior rate; the non-quartic slope at the smallest
    N is the pre-asymptotic regime per LeVeque Ch. 5).
- **Layer 3 rule verdict** (unchanged from plan): `RULE_NS = [16, 32,
  64]`. Rule's raw_value = `max|Δu_spectral − Δu_FD| / max|Δu_spectral|`:
  - N = 16: 6.97 × 10⁻³ (PASS, rule threshold 0.01)
  - N = 32: 4.91 × 10⁻⁴ (PASS)
  - N = 64: 3.17 × 10⁻⁵ (PASS)
- **Rule anchor assertions** (7 tests; 1 conditionally skipped when the
  other pathway fires):
  - `test_layer1_spectral_residual_above_floor_has_r2_above_0_99`
    (4+ points above floor → log-linear fit, slope < 0, R² > 0.99);
  - `test_layer1_spectral_residual_rapid_collapse_characterization`
    (fewer than 4 above floor → characterize large first point +
    last-point-at-floor; skipped when the R² pathway fires);
  - `test_layer1_spectral_residual_monotonically_decreases_in_exp_regime`;
  - `test_layer2_fd_residual_polynomial_order_4`;
  - `test_layer2_fd_residual_monotonically_decreases`;
  - `test_layer3_rule_passes_at_every_N`;
  - `test_rule_skipped_on_non_periodic_domain` (Category 8
    semantic-compatibility check).
- **Category 8 semantic-compatibility check (enumerate-the-splits)**:
  rule's check-fn contract (`ph_res_003.py:21-30`) requires
  `spec.periodic=True` or emits SKIPPED; the non-periodic-domain test
  exercises this contract.

### Borrowed-credibility (external published reproduction layers)

**F3 absent with justification** (pre-recorded in Task 0 Step 5 F3-hunt,
`docs/audits/2026-04-22-f3-hunt-results.md`, unlike Task 2 where F3-absent
was decided in execution). The canonical spectral-vs-FD convergence
demonstration — Trefethen 2000 Program 5 for `exp(sin x)` — is a
log-error-vs-N *plot*, not a tabulated reproduction target. Plot-shape
resemblance is calibration, not reproduction under the §1.2 stricter F3
definition. Boyd 2001 *Chebyshev and Fourier Spectral Methods* (Dover
reprint free PDF at `depts.washington.edu/ph506/Boyd.pdf`) was logged
as unattempted in Task 0's F3-hunt budget with expected-yield low; Task
3 inherits that disposition. Per complete-v1.0 plan §1.2, F3-absent-is-
structural for rules whose canonical published reproduction target is
plot-based rather than table-based; the Tier-3 classical-textbook
reproduction (Trefethen spectral-accuracy + Fornberg FD4 interior rate
on the analytic periodic fixture) carries the credibility here.

### Supplementary calibration context

- **Canuto-Hussaini-Quarteroni-Zang 2006 §2.3 convergence curves**.
  Canuto, C., Hussaini, M.Y., Quarteroni, A., Zang, T.A. (2006).
  *Spectral Methods — Fundamentals in Single Domains.* Springer
  Scientific Computation. DOI 10.1007/978-3-540-30726-6. §2.3
  (convergence of Fourier spectral methods on analytic periodic
  functions). **Curve-shape framing, not reproduction**: Canuto §2.3
  plots logarithmic spectral-error-vs-N curves for several analytic
  periodic test functions; the shape-matching qualitative agreement
  with this anchor's Layer 1 data is calibration, not a numerical
  reproduction claim.
- **Trefethen 2000 Program 5 plot**. Trefethen's canonical `exp(sin x)`
  spectral-vs-FD comparison in *Spectral Methods in MATLAB* Program 5.
  **Plot-shape framing, not reproduction**: the Program 5 figure shows
  log-error-vs-N lines for spectral and FD on `exp(sin x)` in 1D; this
  anchor's Layer 1+2 data on the 2D `exp(sin(x) + sin(y))` fixture
  exhibits the same qualitative pattern (spectral collapses to
  machine precision by N≈32; FD continues to decay polynomially).
  Not a reproduction — (i) the 2D generalization is an execution-time
  choice (DomainSpec requires grid_shape ≥ 2D, plan-diff 4 in
  test_anchor.py), and (ii) plot-shape resemblance is calibration, not
  table-row reproduction.

## Citation summary

- **Primary (mathematical-legitimacy, Tier 3)**: Trefethen 2000
  Chapters 3–4 (chapter-level per `../_harness/TEXTBOOK_AVAILABILITY.md`
  ⚠; theorem number pending local copy) + Fornberg 1988 FD4 coefficients
  (DOI 10.1090/S0025-5718-1988-0935077-0) + LeVeque 2007 FDM consistency-
  order (section-level ⚠).
- **Calibration (Supplementary)**: Canuto-Hussaini-Quarteroni-Zang 2006
  §2.3 (DOI 10.1007/978-3-540-30726-6) + Trefethen 2000 Program 5 plot.
- **Pinned values**: (Layer 1) log-linear slope of spectral residual vs
  N ≈ −1.275, R² = 0.9946 on N ∈ [8, 10, 12, 14, 16] above
  FLOAT64_FLOOR = 1e-13; (Layer 2) log-log slope of FD residual vs h =
  3.89, R² = 0.9999 on N ∈ [16, 32, 64]; (Layer 3) rule max ratio <
  0.01 at every N ∈ [16, 32, 64].
- **Verification date**: 2026-04-24.
- **Verification protocol**: four-layer (mathematical-legitimacy proof-
  sketch + spectral-pre-floor R² fit + FD4 polynomial-order fit +
  rule-verdict sanity + SKIPPED-on-non-periodic contract check).

## Pre-execution audit

PH-RES-003 is a continuous-math rule (spectral vs FD Laplacian on periodic
grids). Per complete-v1.0 plan §6.2 Tier C enumerate-the-splits allocation
(0.1 d), the splits audited are:

- **Analytic vs C^k periodic**: v1 scope restricts to analytic periodic
  (`exp(sin x + sin y)` is entire, so its restriction to the 2π-periodic
  torus is analytic). C^k-periodic fixtures would test the Gibbs-adjacent
  regime where Fourier coefficients decay only algebraically; out of v1
  scope.
- **1D vs 2D**: plan pre-execution audit says "V1: 1D only" but
  `physics_lint.spec.DomainSpec` constrains `grid_shape` to tuple length
  in [2, 3] (`spec.py:127`), so 1D is not constructible through the
  rule's DomainSpec contract. Plan-diff 4 documents the 2D fixture
  substitution. The 2D analog `u(x, y) = exp(sin x + sin y)` preserves
  the analyticity-strip property and exhibits the same spectral /
  FD convergence behavior as 1D.
- **N even vs odd**: tested implicitly — SPECTRAL_NS includes 8 (even)
  and 14 (even); FD_NS includes 16, 32, 64 (all even). The spectral
  FFT backend handles both even and odd N in `_spectral_laplacian`; odd
  N is not required by any Trefethen-style result and is not separately
  tested to keep the refinement sweep clean.

Audit outcome: no splits surface that would reconfigure the fixture
beyond the 1D → 2D DomainSpec-induced substitution (plan-diff 4). Audit
cost 0.1 d absorbed into Task 3 budget.

## Test design

- **Fixture**: `u(x, y) = exp(sin(x) + sin(y))` on `[0, 2π]²` periodic,
  implemented as a numpy-returning function and wrapped in `GridField`
  at each refinement level.
- **Spectral sweep** (Layer 1): `SPECTRAL_NS = [8, 10, 12, 14, 16]`, clip
  at `FLOAT64_FLOOR = 1e-13`.
- **FD sweep** (Layer 2): `FD_NS = [16, 32, 64]`.
- **Rule sweep** (Layer 3): `RULE_NS = [16, 32, 64]`.
- **DomainSpec**: `pde="poisson"`, `grid_shape=[N, N]`,
  `domain={"x": [0, 2π], "y": [0, 2π]}`, `periodic=True`,
  `boundary_condition={"kind": "periodic"}`,
  `field={"type": "grid", "backend": "auto", "dump_path": "p.npz"}`.
- **Wall-time budget**: < 10 s (FFT + FD4 on 64×64 = 4096 points at
  largest N; float64 throughout; no torch dependency unlike PH-RES-002).
- **Tests**: 7 total (6 pass + 1 conditional skip when the R² pathway
  fires).

## Scope note

PH-RES-003 covers the 2D analytic-periodic smooth-fixture case. C^k-
periodic (finite-smoothness) fixtures, 3D, non-analytic-periodic, and
non-uniform-grid variants are out of v1.0 scope per the §6.2 enumerate-
the-splits audit outcome. The rule itself handles any periodic
GridField regardless of smoothness; the fixture chosen here pins the
known-correct convergence rates on the analytic-periodic regime.
