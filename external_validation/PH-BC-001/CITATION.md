# PH-BC-001 — Dirichlet boundary-trace violation

## Scope-separation discipline (read first)

PH-BC-001 validates **Dirichlet-type boundary trace behavior in the
production rule; Neumann/flux semantics are outside the production
validation scope for v1.0**. The rule's emitted quantity is
`||field.values_on_boundary() − boundary_target||`, a discrete-L²
comparison of the field's boundary trace against a caller-supplied
target (Dirichlet-type value on boundary, not Neumann normal-derivative
flux). Extending PH-BC-001 to Neumann flux semantics would require a
separate normal-derivative extraction path and is deferred.

This document does not, and must not, claim the production rule
validates general boundary-condition types. It claims:

- (F1) the rule's mathematical legitimacy is anchored in the trace
  theorem (Evans 2010 §5.5 Thm 1), applied to the Dirichlet-trace
  specialization;
- (F2) three analytic Dirichlet fixtures with known boundary values
  verify the rule's zero-on-exact-trace, perturbation-scaling, and
  absolute-vs-relative mode-branch behavior on the unit square;
- (F3) absent with justification — live PDEBench reproduction would
  require dataset-loader infrastructure not shipped in V1; Task 0's
  pinned PDEBench rows are retained in Supplementary calibration
  context with semantic-equivalence derivation.

## Function-labeled citation stack

Per complete-v1.0 plan §1.3. Authored during Task 4 on 2026-04-24.

### Mathematical-legitimacy (Tier 2 theoretical-plus-multi-paper)

- **Primary — trace theorem.** Evans, L.C. (2010). *Partial Differential
  Equations*, 2nd ed. Graduate Studies in Mathematics 19. AMS. ISBN
  978-0-8218-4974-3. **§5.5 Theorem 1, section-level** per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠ (theorem number pending
  primary-source verification per §6.4). The trace operator
  `γ: H¹(Ω) → H^{1/2}(∂Ω)` is bounded on bounded open `Ω ⊂ ℝⁿ` with
  Lipschitz boundary.
- **Structural-equivalence proof-sketch** (section-level framing per
  §6.4, no tight theorem-number claims):
  1. **Trace-theorem preconditions.** `Ω = [0, 1]²` is a Lipschitz-
     boundary convex polygon. For any `u ∈ H¹(Ω)` (in particular for
     the smooth analytic fixtures below), the trace `γ(u) = u|_{∂Ω}`
     is well-defined in `H^{1/2}(∂Ω)` and bounded by `||u||_{H¹(Ω)}`
     up to a constant depending only on the domain.
  2. **Discrete trace operator.** `GridField.values_on_boundary()`
     evaluates the pointwise trace at the grid's boundary sample
     points, ordered `[left | right | bottom | top]` (grid.py line
     150-158). This is the discrete analog of `γ(u)` on a uniform grid.
  3. **Dirichlet-trace mismatch.** PH-BC-001's emitted quantity is
     `||γ(u) − g||` where `g` is the caller-supplied Dirichlet target;
     `γ(u)` is the discrete trace above. A nonzero value indicates
     the field's boundary trace does not match the prescribed
     Dirichlet target — i.e., the field violates the Dirichlet BC
     at the discrete-L² level.
  4. **Mode-branch specialization.** For Dirichlet-homogeneous (zero
     Dirichlet) cases `g ≡ 0`, the relative-mode ratio `||u − g||/||g||`
     is ill-defined as `||g|| → 0`; the rule switches to an **absolute**
     binary PASS/FAIL against a 1e-3 tolerance (Rev 4.1 fix for the
     homogeneous-Dirichlet footgun, per `ph_bc_001.py:1-12`). For
     `||g|| > 1e-8`, relative mode applies with tri-state
     classification against the calibrated floor.

### Correctness-fixture (CI-runnable, non-credibility-claiming)

Three analytic Dirichlet fixtures on the unit square `[0, 1]²`,
covering the rule's absolute- and relative-mode branches. Boundary
targets built via `external_validation/_harness/trace.py`'s
`dirichlet_trace_on_unit_square_grid(u_fn, n)` (populated per plan §3
harness extension).

- **Fixture A — polynomial, relative mode.** `u(x, y) = x² − y²`.
  Analytical Dirichlet trace on the four edges (nonzero; e.g., left
  `u|_{x=0} = −y²`, top `u|_{y=1} = x² − 1`). `||g||` ≈ O(1), so rule
  enters **relative mode**. Measured: zero-violation `raw_value = 0`
  exactly at N ∈ {16, 32, 64}; PASS at every N.
- **Fixture B — trigonometric, absolute mode.** `u(x, y) = sin(πx)
  sin(πy)`. Analytical Dirichlet trace on all four edges is zero
  (sin(0) = sin(π) = 0). Measured `||g||` is at float64 roundoff
  (<1e-15), below the rule's 1e-8 absolute-mode threshold, so rule
  enters **absolute mode** with binary PASS/FAIL against 1e-3
  tolerance. Measured: zero-violation `raw_value = 0` exactly at
  N ∈ {16, 32, 64}; PASS at every N.
- **Fixture C — trigonometric, relative mode.** `u(x, y) = cos(πx)
  cos(πy)`. Analytical Dirichlet trace nonzero on all four edges.
  `||g||` ≈ O(1), rule enters **relative mode**. Measured:
  zero-violation `raw_value = 0` exactly at N ∈ {16, 32, 64}; PASS
  at every N.

**Rule liveness (perturbation-scaling test).** For Fixture A at N=32,
perturbing the left-edge boundary by `δ = 1e-3` produces
`raw_value ≈ 8.4e-4` (tested to lie in `[1e-4, 5e-3]`). This
demonstrates the rule is live — the numerical value scales with the
discrete-L² magnitude of the perturbation (theoretical estimate
`δ · sqrt(N_y) / sqrt(4N−4) = 1e-3 · sqrt(32) / sqrt(124) ≈ 5.08e-4`
for err_norm, multiplied by `1/g_norm` in relative mode gives
≈ 8e-4). Zero-violation tests alone would be consistent with a
no-op rule; the perturbation-scaling test rules that out.

**Rule-verdict contract.** 13 tests total in `test_anchor.py`:
- 9 F2 parametrized (3 fixtures × 3 refinement levels) cover
  zero-violation PASS + mode-branch behavior per fixture;
- 1 F2 perturbation-scaling test (rule detects known boundary delta);
- 3 RVC (rule-verdict contract): mode-branch threshold via ||g||,
  shape-mismatch ValueError (Category 8 API contract), absolute-mode
  FAIL when err_norm > abs_tol_fail=1e-3.

### Borrowed-credibility (external published reproduction layers)

**F3 absent with justification.** Task 0 literature-pin pass
(`docs/audits/2026-04-22-pdebench-hansen-pins.md` §"Task 4") identified
three PDEBench bRMSE rows as *semantically-equivalent* reproduction
candidates: Diffusion-sorption (Dirichlet-dominant, PDEBench Table 5),
2D diffusion-reaction (Neumann no-flow, Table 5), and 1D Advection
(periodic, Table 6). **Live CI-runnable reproduction of the U-Net /
FNO bRMSE numbers on these rows requires a PDEBench dataset loader**
— an adapter-mode plumbing between physics-lint's `DomainSpec` and
PDEBench's HDF5 dataset layout — which V1 physics-lint does not ship.
The pre-execution precheck (`docs/audits/2026-04-24-plan-smoke-
precheck.md`) missed this loader-infrastructure gap; it is identified
and deferred here rather than absorbed into Task 4's 1.5 ED budget
(loader plumbing + trained-model inference + CI provisioning would
multi-ply the task budget).

Per plan §1.2, F3-absent-is-structural for rules whose canonical
reproduction target requires external infrastructure not available
at V1. The Tier-2 theoretical-plus-multi-paper reproduction (trace
theorem + three analytic Dirichlet fixtures + perturbation-scaling
test) carries the credibility here. PDEBench rows move to
Supplementary calibration context with semantic-equivalence
derivation preserved.

### Supplementary calibration context

- **PDEBench Takamoto et al. 2022** `arXiv:2210.07182` Tables 5–6
  (Supplement §§D.3, D.7, D.9). **Semantic-equivalence derivation
  (from Task 0 pin audit):** PH-BC-001 measures
  `u_pred − u_BC_prescribed` at boundary points; PDEBench's bRMSE
  measures `u_pred − u_true` at boundary points where `u_true` is
  the high-fidelity numerical reference. For PDEBench datasets with
  strongly-imposed Dirichlet BCs (e.g., Diffusion-sorption), the
  reference `u_true|_{∂Ω}` is numerically identical to the prescribed
  BC, so the two metrics are semantically equivalent within solver
  roundoff. Pinned rows (U-Net and FNO bRMSE columns; PINN column
  omitted per Takamoto's own commentary on PINN noise):
  - **Diffusion-sorption** (Dirichlet-dominant): U-Net 6.1e-3,
    FNO 2.0e-3.
  - **2D diffusion-reaction** (Neumann no-flow): U-Net 7.8e-2,
    FNO 2.7e-2. — **flagged: Neumann semantics, not exercised by V1
    production rule scope.**
  - **1D Advection (β=0.1)** (periodic): U-Net 3.8e-2, FNO 4.9e-3.
    — **flagged: periodic has no boundary in the trace-theorem
    sense; calibration-adjacent only.**
  Reproduction tolerance from Task 0: "physics-lint's emitted bRMSE
  on the same dataset row must land within ±2× of the reported U-Net
  / FNO numerical values" (±2× envelope accommodates PDEBench's
  reported run-to-run seed variance). **Not a reproduction claim in
  V1** — pending PDEBench loader plumbing.

## Citation summary

- **Primary (mathematical-legitimacy, Tier 2)**: Evans 2010 §5.5
  Theorem 1 (section-level per `../_harness/TEXTBOOK_AVAILABILITY.md`
  ⚠; theorem number pending local copy).
- **F2 harness-level**: `external_validation/_harness/trace.py`
  `dirichlet_trace_on_unit_square_grid(u_fn, n)`. Tested at N ∈
  {16, 32, 64} on three Dirichlet fixtures covering both absolute
  and relative mode branches.
- **Calibration (Supplementary)**: PDEBench Takamoto et al. 2022
  arXiv:2210.07182 Tables 5–6 pinned rows with semantic-equivalence
  derivation; reproduction deferred pending V1.x PDEBench loader.
- **Pinned values**: (F2) `raw_value = 0` exactly on three analytic
  Dirichlet fixtures at all tested N; (F2 perturbation scaling)
  `raw_value ≈ 8.4e-4` on `u = x² − y²` at N=32 with `δ = 1e-3`
  left-edge perturbation; (RVC) absolute/relative mode branches on
  `||g|| < 1e-8` threshold verified; shape-mismatch ValueError
  raised.
- **Verification date**: 2026-04-24.
- **Verification protocol**: three-layer (F1 trace-theorem proof-
  sketch + F2 three-fixture Dirichlet correctness + perturbation
  scaling + rule-verdict contract) with scope separation enforced
  ("Dirichlet-trace only; Neumann outside scope" stated in F1 +
  F2 + Supplementary subsections).

## Pre-execution audit

PH-BC-001 is a continuous-math rule (Dirichlet-trace violation).
Per complete-v1.0 plan §6.2 Tier A enumerate-the-splits allocation
(0.2 d, trace-theorem preconditions subtle), the splits audited are:

- **BC type: Dirichlet / Neumann / periodic.** V1 rule scope is
  Dirichlet-trace only (rule emits `values_on_boundary()`-based
  value mismatch, not Neumann normal-derivative). Neumann and
  periodic deferred; fixtures scoped to Dirichlet. Plan-diff 7 from
  plan §12 "Dirichlet, Neumann, periodic" → "Dirichlet only."
- **Lipschitz vs non-Lipschitz domain boundary.** V1: Lipschitz only
  (unit square = convex polygon with piecewise-linear boundary).
  Non-Lipschitz (cusps, re-entrant corners) would break the trace
  theorem's precondition and is not tested.
- **2D vs 3D.** V1: 2D only (DomainSpec supports both but trace
  extraction order in 3D is more elaborate; 2D scope matches
  plan §12).
- **Smooth vs non-smooth BC function.** V1: smooth only. Polynomial
  (`x² − y²`) and trigonometric (`sin/cos(π·)`) fixtures are
  analytic, satisfying the H¹(Ω) trace-theorem precondition.

Audit outcome: scope reduced to Dirichlet-only per plan-diff 7;
three analytic fixtures cover the rule's mode-branch surface. Audit
cost 0.2 d absorbed into Task 4 budget.

## Test design

- **Harness-level fixture (F2)**: boundary-target construction via
  `_harness/trace.py` `dirichlet_trace_on_unit_square_grid(u_fn, n)`.
- **Analytic functions**: `u = x² − y²` (polynomial, relative mode),
  `u = sin(πx) sin(πy)` (trig zero-Dirichlet, absolute mode),
  `u = cos(πx) cos(πy)` (trig nonzero-Dirichlet, relative mode).
- **Refinement levels**: N ∈ {16, 32, 64}.
- **DomainSpec**: `pde="laplace"`, `grid_shape=[N, N]`,
  `domain={"x": [0, 1], "y": [0, 1]}`, `periodic=False`,
  `boundary_condition={"kind": "dirichlet_homogeneous"}`,
  `field={"type": "grid", "backend": "fd", "dump_path": "p.npz"}`.
- **Wall-time budget**: < 3 s (uniform-grid evaluation only; no mesh
  assembly or FD time-stepping).
- **Tests**: 13 total (9 F2 parametrized + 1 F2 perturbation-scaling
  + 3 rule-verdict-contract).

## Scope note

PH-BC-001 covers Dirichlet-trace value mismatch on Lipschitz-boundary
2D unit-square fixtures in V1. Neumann flux semantics, 3D domains,
non-Lipschitz boundaries, non-smooth BC functions, and live PDEBench
reproduction are out of v1.0 scope.
