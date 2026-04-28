# PH-RES-002 — AD vs FD residual cross-check

## Function-labeled citation stack

Per complete-v1.0 plan §1.3 three-function labeling. Authored during Task 2
of the complete-v1.0 plan on 2026-04-23. F3 falls back to absent-with-
justification per plan §10 acceptance-criteria fallback path because Task 0
literature-pin pass (`docs/audits/2026-04-22-pdebench-hansen-pins.md` and
`docs/audits/2026-04-22-f3-hunt-results.md`) covered Tasks 4/8/9 (PDEBench
+ Hansen) and Tasks 10/11/3 (F3-hunt) but did not pin a directly-comparable
CAN-PINN Chiu 2022 CMAME row for PH-RES-002.

### Mathematical-legitimacy (Tier 2 theoretical-plus-multi-paper)

- **AD accuracy backbone**: Griewank, A. & Walther, A. (2008). *Evaluating
  Derivatives: Principles and Techniques of Algorithmic Differentiation*,
  2nd ed. SIAM. ISBN 978-0-89871-659-7. DOI 10.1137/1.9780898717761.
  **Chapter 3 (accuracy of reverse-mode AD derivatives), section-level**
  per `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠ (row added during Task 2 per
  §6.4 "new backbone textbook added at rule pre-execution"; proposition
  number pending local copy). Secondary-source corroboration: Baydin, A.G.,
  Pearlmutter, B.A., Radul, A.A., Siskind, J.M. (2018). *Automatic
  Differentiation in Machine Learning: A Survey.* JMLR 18(153):1–43.
  arXiv:1502.05767. §3 (reverse-mode AD evaluates the Jacobian-vector
  product to the precision of the underlying floating-point representation,
  bounded by a small constant times unit roundoff).
- **FD consistency-order backbone**: LeVeque, R.J. (2007). *Finite
  Difference Methods for Ordinary and Partial Differential Equations:
  Steady-State and Time-Dependent Problems.* SIAM. ISBN 978-0-898716-29-0.
  **FDM consistency-order theorem (section-level)** per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠. A 4th-order central FD stencil
  on a smooth function has truncation error O(h⁴) in the interior; the
  physics-lint FD4 stencil is the Fornberg 1988 canonical form
  `(−1, 16, −30, 16, −1) / 12` (applied in `src/physics_lint/field/grid.py`
  line 22–30; interior-only rate is 4th-order per design doc §3.2).
- **Structural-equivalence proof-sketch** (AD vs FD on a smooth callable,
  section-level framing throughout — no tight theorem-number claims per
  §6.4 of complete-v1.0 plan):
  1. **AD path.** For a smooth `u: R² → R` implemented as a torch callable,
     `torch.func.hessian ∘ vmap` computes the per-point Hessian; trace of
     the Hessian is `Δu`. Griewank-Walther 2008 Ch. 3 (section-level)
     establishes that reverse-mode AD on a smooth composition of
     differentiable primitives accumulates error at most proportional to
     unit roundoff ε ≈ 2.22 × 10⁻¹⁶ (float64). So `Δu_AD = Δu + O(ε)`.
  2. **FD path.** For the same `u` sampled on a uniform grid of spacing
     `h`, the rule materializes the field values and applies the 4th-order
     central Laplacian (Fornberg 1988 coefficients) on the interior band
     `[2:-2]` of each axis. LeVeque 2007 (section-level) establishes that
     the FD4 stencil has truncation error `C · h⁴ · sup|∂⁶u|` for a smooth
     function on the interior band. So `Δu_FD = Δu + C · h⁴ + o(h⁴)`.
  3. **Gap.** By the triangle inequality,
     `|Δu_AD − Δu_FD| ≤ |Δu_AD − Δu| + |Δu − Δu_FD| = O(ε) + C · h⁴`.
     The rule's emitted quantity — max interior
     `|Δu_FD − Δu_AD| / max(|Δu_FD|, |Δu_AD|, 1e-12)` — is the relative
     form of this gap. For `h > (ε/C)^{1/4} ≈ 10⁻⁴`, the FD term dominates
     and the ratio scales as O(h⁴); for finer grids the AD machine-precision
     floor dominates and the ratio plateaus near ε.
  4. **Measurement target.** On the MMS fixture `u(x,y) = sin(πx) sin(πy)`
     (smooth, infinitely differentiable, bounded sixth derivatives), the
     refinement sweep N ∈ {16, 32, 64, 128} traverses `h` from `1/15` to
     `1/127`. All four points lie above the noise floor, so the measured
     log-log slope should equal 4 within regression noise.

### Correctness-fixture (CI-runnable, non-credibility-claiming)

- **MMS fixture**: `u(x, y) = sin(π x) sin(π y)` on `[0, 1]²` Dirichlet
  homogeneous. Reuses `physics_lint.analytical.poisson.sin_sin_mms_square`'s
  analytical form via a parallel `torch.sin(π · x) torch.sin(π · y)` model
  wrapped in `CallableField`. The analytical Laplacian is
  `Δu = −2π² sin(π x) sin(π y)`; `|Δu|_{max} = 2π² ≈ 19.74`.
- **Refinement sweep**: N ∈ {16, 32, 64, 128}. Measured max interior
  AD-vs-FD relative discrepancy ratios (at 2026-04-23):
  - N = 16 (h = 0.0667): 2.13 × 10⁻⁵
  - N = 32 (h = 0.0323): 1.17 × 10⁻⁶
  - N = 64 (h = 0.0159): 6.87 × 10⁻⁸
  - N = 128 (h = 0.0079): 4.17 × 10⁻⁹
  - **log-log slope = 3.997, R² = 1.0000** (empirical O(h⁴), matching
    the LeVeque 4th-order theorem).
- **Rule anchor assertions**: `test_layer1_refinement_slope_is_4` (slope
  in [3.6, 4.4]); `test_layer1_refinement_monotonically_decreases`;
  `test_layer1_regression_r_squared_above_0_99`;
  `test_rule_passes_at_every_refinement_level` (max ratio < 0.01 at each N);
  `test_rule_skipped_on_non_callable_field` (dump-mode SKIPPED contract).
- **Category 8 semantic-compatibility check (enumerate-the-splits)**: rule
  `__input_modes__ = {"adapter"}` — `GridField` (dump mode) must emit
  SKIPPED, not silently PASS. Covered by the skipped-on-non-callable test.

### Borrowed-credibility (external published reproduction layers)

**F3 absent with justification.** PH-RES-002's emitted quantity is a max
interior relative discrepancy ratio between AD-computed and FD-computed
Laplacians on a physics-lint-constructed MMS fixture. The CAN-PINN family
(Chiu, Fuks, Oommen, Karniadakis 2022, CMAME arXiv:2110.14432; and
follow-ups) reports absolute-error reductions and training-time
improvements from combining AD and FD residuals, not a directly-comparable
"AD-vs-FD discrepancy at successive grid refinements" measurement. Task 0
literature-pin pass (`docs/audits/2026-04-22-pdebench-hansen-pins.md` and
`docs/audits/2026-04-22-f3-hunt-results.md`) covered Tasks 4/8/9 PDEBench
+ Hansen anchoring and Tasks 10/11/3 F3-hunt; it did not pin a CAN-PINN
row for PH-RES-002, so the plan §10 acceptance-criteria fallback path
("F3 demoted to absent-with-justification + CAN-PINN moved to
Supplementary calibration context") applies. Per complete-v1.0 plan §1.2,
F3-absent-is-structural for rules whose reproduction target would be
metric-incompatible with the rule's emitted quantity; the Tier-1
structural-equivalence reproduction (interior O(h⁴) from LeVeque + AD
machine-precision from Griewank-Walther) carries the credibility here.

### Supplementary calibration context

- **CAN-PINN**: Chiu, P.-H., Wong, J.C., Ooi, C., Dao, M.H., Ong, Y.-S.
  (2022). *CAN-PINN: A Fast Physics-Informed Neural Network Based on
  Coupled-Automatic-Numerical Differentiation Method.* CMAME 395:114909.
  arXiv:2110.14432. Framing context: motivates the practice of
  cross-checking AD-computed and FD-computed derivatives in PINNs and
  neural-PDE contexts. §4 Numerical Experiments reports CAN-PINN's
  accuracy and training-time improvements against pure AD-PINN baselines
  on 1D/2D viscous Burgers, convection-diffusion, and incompressible
  Navier–Stokes benchmarks. **Calibration-only: not a reproduction
  claim.** CAN-PINN reports final-solution error metrics (L∞, L², u-RMS)
  after convergence, not AD-vs-FD Laplacian discrepancy at refinement
  levels on a smooth MMS fixture; the two quantities measure different
  things on different problem classes.

## Citation summary

- **Primary (mathematical-legitimacy, Tier 2)**: Griewank-Walther 2008
  Ch. 3 (section-level per `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠) +
  LeVeque 2007 FDM consistency-order theorem (section-level ⚠).
- **Structural bridge**: Baydin-Pearlmutter-Radul-Siskind 2018 JMLR survey
  (arXiv:1502.05767) for AD accuracy statement content.
- **Calibration (Supplementary)**: Chiu et al. CMAME 2022 CAN-PINN
  (arXiv:2110.14432), §4 Numerical Experiments.
- **Pinned value**: refinement-based — log-log slope of max interior
  AD-vs-FD relative discrepancy vs h = 3.997, R² = 1.0000 on N ∈ {16, 32,
  64, 128} for the `sin(π x) sin(π y)` MMS fixture on `[0, 1]²`. All four
  grid levels classify as PASS (max ratio < 0.01 rule threshold).
- **Verification date**: 2026-04-23.
- **Verification protocol**: three-layer (mathematical-legitimacy proof-
  sketch + refinement sweep slope assertion + rule-verdict sanity +
  skipped-on-non-callable-field contract check).

## Pre-execution audit

PH-RES-002 is a continuous-math rule (AD vs FD residual gap). Per
complete-v1.0 plan §6.2 Tier B enumerate-the-splits allocation (0.15 d),
the splits audited are:

- **1D vs 2D vs 3D**: v1 scope restricts to 2D (the fixture uses a 2D
  `CallableField`; 1D is accessible via a reshape but is not tested here;
  3D is out of v1 scope). The LeVeque 4th-order stencil theorem and the
  Griewank-Walther AD accuracy statement are both dim-agnostic.
- **Smooth-source vs discontinuous-source**: smooth-source only per the
  plan (sin(π x) sin(π y) has bounded derivatives of all orders; its sixth
  derivative magnitude sets the FD4 truncation constant).
- **BC type (Dirichlet/Neumann/periodic)**: the rule uses the interior
  `[2:-2]` band only, so the outer boundary stencil (which differs across
  BC types) is excluded. Dirichlet homogeneous is sufficient to exercise
  the interior FD4 path.
- **Uniform vs non-uniform grid**: uniform grid only in v1. Non-uniform
  grids would require interpolation of the FD stencil; out of v1 scope.

Audit outcome: no splits surface that would reconfigure the fixture. Audit
cost 0.15 d absorbed into Task 2 budget.

## Test design

- **Fixture**: `u(x, y) = sin(π x) sin(π y)` on `[0, 1]²` Dirichlet
  homogeneous, implemented as a `torch.Tensor`-returning callable and
  wrapped in `CallableField`.
- **Grid sweep**: N ∈ {16, 32, 64, 128}, uniform spacing `h = 1/(N−1)`.
- **DomainSpec**: `pde="poisson"`, `periodic=False`,
  `boundary_condition={"kind": "dirichlet_homogeneous"}`,
  `field={"type": "callable", "backend": "fd", "dump_path": "p.npz"}`.
- **Wall-time budget**: < 10 s (torch hessian+vmap across 128×128 = 16384
  points at N=128 is the expensive step; float64 throughout).
- **Tests**: 5 total — slope-in-range (4 ± 0.4), monotone decrease,
  R² ≥ 0.99, PASS at every N, SKIPPED on non-callable dump-mode field.

## Scope note

PH-RES-002 covers the 2D smooth-source Dirichlet case. 1D, 3D,
discontinuous-source, non-uniform-grid, and Neumann/periodic BC variants
are out of v1.0 scope per the §6.2 enumerate-the-splits audit outcome.
The rule's interior-only FD4 check excludes boundary-stencil behavior; a
full-domain refinement characterization (like PH-RES-001's Layer 1b at
slope ~3.5) is not in scope here because the rule's `raw_value` is the
interior-cut max ratio by design.
