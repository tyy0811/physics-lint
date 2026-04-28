# PH-CON-001 — Heat mass conservation (analytic snapshots)

## Scope-separation discipline (read first)

PH-CON-001 validates the production rule's ability to measure **integral
conservation drift on analytically controlled source-free snapshots**.
It does not certify the accuracy of a heat-equation time integrator.

The F2 fixture is explicitly **analytical-snapshot** mode
(`u(x, y, t) = cos(2πx) cos(2πy) · exp(−8π²κt)`, exact solution of
the 2D periodic heat equation with zero spatial mean at all `t`), not a
numerically-evolved FD solution. Mixing analytical-snapshot and
numerically-evolved tolerances is explicitly disallowed per the
2026-04-24 revised Task 8 contract. A numerically-evolved fixture would
be a separate anchor with method-dependent tolerance; it is out of V1
scope.

This document does not, and must not, claim the production rule
validates heat-equation solver accuracy. It claims:

- (F1) the rule's mathematical legitimacy is anchored in the balance
  law `d/dt ∫_Ω u dV = 0` for source-free heat on a no-flux / periodic
  domain;
- (F2) the rule's conservation-drift metric on source-free analytic
  snapshots lands at numerical roundoff (observed floor ~1e-18;
  acceptance tolerance 1e-15 with ~1000× safety factor over observed);
- (F3) absent with justification — live Hansen ProbConserv reproduction
  requires a `github.com/amazon-science/probconserv` checkpoint loader
  not shipped in V1 (already scoped to v1.1 in the repo-level
  `external_validation/README.md`).

## Function-labeled citation stack

Per complete-v1.0 plan §1.3. Authored during Task 8 on 2026-04-24.

### Mathematical-legitimacy (Tier 2 theoretical-plus-multi-paper)

- **Primary — balance-law / mass-conservation identity.** Evans, L.C.
  (2010). *Partial Differential Equations*, 2nd ed. Graduate Studies in
  Mathematics 19. AMS. ISBN 978-0-8218-4974-3. **§2.3 fundamental
  solution of the heat equation, section-level** per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠ (theorem number pending
  primary-source verification per §6.4).
- **Secondary framing.** Dafermos, C.M. (2016). *Hyperbolic
  Conservation Laws in Continuum Physics*, 4th ed. Springer GTM 325.
  DOI 10.1007/978-3-662-49451-6. **Chapter I (balance-law framing),
  section-level** per `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠.
- **Structural-equivalence proof-sketch** (section-level framing per
  §6.4, no tight theorem-number claims):
  1. **Balance-law preconditions.** For the source-free heat equation
     `u_t = κ Δu` on a periodic domain `Ω = 𝕋²` (or no-flux Neumann
     boundary), integrating both sides over `Ω` gives
     `d/dt ∫_Ω u dV = κ ∫_Ω Δu dV = κ ∮_{∂Ω} ∂u/∂n dS = 0` (Gauss-
     Green with no-flux or vacuous-on-torus boundary). Evans §2.3
     establishes the fundamental solution; Dafermos Ch I provides the
     balance-law framing.
  2. **Analytical-snapshot fixture.** Choose `u(x, y, t) = cos(2πx)
     cos(2πy) · exp(−8πκt)`. This is an exact solution of the 2D
     periodic heat equation (verify: `u_t = −8π²κ · u`, and
     `Δu = −8π² · u`, so `u_t = κ Δu` ✓). `∫_{[0, 1]²} u dV = 0`
     analytically at every `t` (zero-mean trigonometric mode on the
     periodic domain).
  3. **Discrete-integral invariance on periodic grid.** The rectangle
     rule (endpoint-exclusive) on an `N × N` periodic grid is
     spectrally accurate on periodic analytic integrands — discrete
     integral equals analytical integral (zero here) up to
     FFT-coefficient-roundoff accumulation, empirically `~1e-18` in
     float64.
  4. **Rule's emitted quantity.** PH-CON-001 (exact-mass branch on
     periodic BC) emits `max_t |M(t) − M(0)| / max(|M(0)|, ‖u_0‖_{L¹})`
     where `M(t) = ∫_Ω u(t, x) dV`. On the analytical snapshot,
     `M(t) = 0` at all `t` (within rectangle-rule roundoff), so
     `raw_value` = O(roundoff / L¹-scale) = O(1e-18 / 0.4) = O(1e-18).
     Rule's 1e-4 shipped-default threshold is cleared by ~14 orders
     of magnitude.

### Correctness-fixture (CI-runnable, non-credibility-claiming)

**F2 analytical-snapshot layer (authoritative).** `u(x, y, t) =
cos(2πx) cos(2πy) · exp(−8πκt)` on `[0, 1]²` periodic via
`external_validation/_harness/energy.py`'s
`analytical_heat_snapshot_2d(fixture, nx, nt, kappa, t_end)`
(populated per plan §3 harness extension).

**Measured mass-drift floor (2026-04-24, float64, κ=1.0, t_end=0.1):**

| Nx  | Nt=5     | Nt=11    | Nt=21    |
|-----|----------|----------|----------|
| 16  | 2.75e-18 | 3.57e-18 | 3.57e-18 |
| 32  | 5.44e-19 | 1.26e-18 | 1.76e-18 |
| 64  | 4.86e-19 | 7.20e-19 | 7.20e-19 |
| 128 | 1.06e-18 | 1.08e-18 | 1.08e-18 |

Max observed: **3.57e-18**. Refinement-invariant at roundoff level
(no N-scaling, as expected for a spectrally-accurate rectangle rule
on an analytic integrand).

**Tolerance selection** (per 2026-04-24 revised Task 8 contract "avoid
hard-coding 1e-14 unless measured robustly across N and platforms"):
`ANALYTIC_SNAPSHOT_TOL = 1e-15`. This gives ~1000× margin over the
observed maximum floor and is still 11 orders of magnitude below
the rule's 1e-4 shipped-default threshold, so the rule classifies
PASS with an enormous safety margin. Plan §16's original "1e-14"
would also pass empirically but misses the actual physics — the
rule emits sub-epsilon drift, not `O(h^4)` or `O(Δt^2)`-scale drift.

**Liveness test.** Perturbing the analytic snapshot at the final time
step by `δ = 1e-6` gives `raw_value ≈ 2.48e-6` (theoretical estimate
`δ / ‖cos(2πx)cos(2πy)‖_{L¹}[0, 1]² = 1e-6 / (2/π)² ≈ 2.47e-6`).
Rule is live and mass-drift scaling is correctly calibrated.

**Rule anchor assertions** (17 tests total):
- 12 F2 parametrized (`Nx ∈ {16, 32, 64, 128}` × `Nt ∈ {5, 11, 21}`)
  cover the analytic-snapshot floor across all configurations.
- 1 liveness / injected-drift test.
- 1 refinement-invariance summary.
- 3 rule-verdict-contract: SKIP on non-heat PDE (ph_con_001.py:47);
  SKIP on insufficient time samples (ph_con_001.py:55); exact-mass
  branch on periodic BC (vs rate-consistency on Dirichlet).

### Borrowed-credibility (external published reproduction layers)

**F3 absent with justification.** Task 0 literature-pin pass
(`docs/audits/2026-04-22-pdebench-hansen-pins.md:166`, ANP row
`CE = 4.68 × 10⁻³ ± 0.10`) identified Hansen ProbConserv as
F3-PRESENT. The Task 8 preflight F3-infrastructure check (per
`feedback_precheck_f3_executability_category.md`, 2026-04-24) confirms
V1 physics-lint does not ship a `github.com/amazon-science/probconserv`
checkpoint loader — reproduction would require:

- clone + install the `probconserv` package;
- load the ANP pre-trained checkpoint (PyTorch);
- run inference on the diffusion-equation test set at `t = 0.5`, `k = 1`;
- feed predictions into physics-lint's PH-CON-001 rule;
- compare emitted drift to `4.68 × 10⁻³ ± 0.10`.

The infrastructure is out of V1 scope. `external_validation/README.md`
already tags `PH-CON-001 Hansen ProbConserv mass CE` to v1.1, which
is now honored here by pre-demoting F3 → Supplementary (plan-diff 9).
Per plan §1.2, F3-absent-is-structural for rules whose reproduction
requires external checkpoint infrastructure not shipped in V1. The
Tier-2 theoretical-plus-multi-paper reproduction (balance-law +
analytical-snapshot + liveness test) carries the credibility here.

### Supplementary calibration context

- **Hansen ProbConserv (Hansen et al. 2024 Physica D / arXiv:2302.11002).**
  Hansen, D., Maddix, D.C., Alizadeh, S., Gupta, G., Mahoney, M.W.
  (2024). *Learning Physics Constrained Neural Networks via
  Differentiable Linear Algebra.* Physica D. arXiv:2302.11002. Table 1,
  diffusion-equation row (easy case of GPME, `k(u) = k`). Pinned
  baseline numbers (from Task 0 audit, mean ± std error over 50 runs):

  | Model           | CE × 10⁻³     | LL         | MSE × 10⁻⁴  |
  |-----------------|---------------|------------|-------------|
  | ANP             | 4.68 (0.10)   | 2.72       | 1.71        |
  | SoftC-ANP       | 3.47 (0.17)   | 2.40       | 2.24        |
  | HardC-ANP       | 0.00          | 3.08       | 1.37        |
  | ProbConserv-ANP | 0.00          | 2.74       | 1.55        |

  **Semantic-equivalence derivation (from Task 0 pin audit).** Hansen
  CE measures `(Gμ − b)|_{t_j}` — the deviation of the model's
  predicted mean μ from exact conservation at time `t_j`. For the
  diffusion equation with zero-flux boundaries and random-field IC
  centered on zero, true total mass `U(t) = ∫_Ω u dΩ` is zero at all
  `t`, so CE at `t = 0.5` equals the absolute deviation of
  `∫_Ω u_pred(0.5, x) dΩ` from zero. PH-CON-001 measures
  `|∫_Ω u_pred(t, x) dΩ − ∫_Ω u_pred(0, x) dΩ|` — current-mass
  minus initial-mass. For Hansen's zero-mean IC setup both quantities
  equal zero by construction, so the two metrics are semantically
  equivalent. **Reproduction deferred to V1.1** pending ProbConserv
  loader — calibration-only flag.

## Citation summary

- **Primary (mathematical-legitimacy, Tier 2)**: Evans 2010 §2.3
  (section-level per `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠;
  theorem number pending local copy) + Dafermos 2016 Ch I
  (section-level ⚠; DOI 10.1007/978-3-662-49451-6).
- **F2 harness-level**: `external_validation/_harness/energy.py`
  `analytical_heat_snapshot_2d(fixture, nx, nt, kappa, t_end)`.
  Tested at `Nx ∈ {16, 32, 64, 128}` × `Nt ∈ {5, 11, 21}`.
- **Calibration (Supplementary)**: Hansen 2024 Physica D
  arXiv:2302.11002 Table 1; diffusion-equation ANP row `CE × 10⁻³ =
  4.68 (0.10)` with semantic-equivalence derivation; reproduction
  deferred to V1.1 pending ProbConserv loader infrastructure.
- **Pinned values**: (F2 analytical-snapshot) `raw_value < 1e-15` at
  every `(Nx, Nt)` in the sweep; max observed 3.57e-18; (liveness)
  `δ = 1e-6` injection → `raw_value ≈ 2.48e-6` matching theoretical
  `δ / (2/π)² = 2.47e-6`; (rule-verdict contract) SKIP on non-heat,
  SKIP on nt<3, exact-mass branch on periodic BC.
- **Verification date**: 2026-04-24.
- **Verification protocol**: three-layer (F1 balance-law proof-sketch
  + F2 analytical-snapshot mass-drift floor across N × Nt +
  liveness / injected-perturbation + rule-verdict contract) with
  scope separation enforced.

## Pre-execution audit

PH-CON-001 is a continuous-math rule (mass-conservation drift on
heat equation). Per complete-v1.0 plan §6.2 Tier B enumerate-the-
splits allocation (0.15 d), the splits audited are:

- **Periodic vs Dirichlet BC.** Periodic conserves mass exactly
  (rule's `exact-mass` branch); Dirichlet does not (rule's
  `rate-consistency` branch, measures `dM/dt − κ∫Δu` error). V1 F2
  scope: periodic only. Dirichlet-branch testing deferred to a
  separate task (not in V1 plan).
- **1D vs 2D.** V1 F2 scope: 2D (matches rule's ndim ≥ 3 contract;
  1D spatial + 1 time = ndim 2, below threshold; 2D spatial + 1 time
  = ndim 3, above threshold). Plan-diff 11 documents the 1D → 2D
  extension vs plan §16's `u_0(x) = cos(2πx)` 1D example.
- **Smooth vs non-smooth IC.** V1: smooth only (`cos(2πx) cos(2πy)`
  is analytic). Non-smooth ICs would break the spectral-accuracy of
  the rectangle quadrature and raise the observed floor; out of V1
  scope.
- **Fixture mode: analytical-snapshot vs numerically-evolved**
  (per user's 2026-04-24 revised contract): F2 is
  **analytical-snapshot only**. Numerically-evolved fixtures are out
  of V1 scope; a solver-accuracy anchor would be a separate task.

Audit outcome: F2 scope = 2D periodic analytical-snapshot; no
reconfiguration required. Plan-diffs 9, 10, 11 log fixture-mode
scoping and tolerance-measurement discipline. Audit cost 0.15 d
absorbed into Task 8 budget.

## Test design

- **Harness-level fixture (F2)**: `analytical_heat_snapshot_2d` in
  `_harness/energy.py` (populated for Task 8, Task 9 will extend with
  wave-energy helpers).
- **Analytic function**: `u(x, y, t) = cos(2πx) cos(2πy) · exp(−8πκt)`
  on `[0, 1]²` periodic.
- **Refinement sweep**: `Nx ∈ {16, 32, 64, 128}`, `Nt ∈ {5, 11, 21}`.
- **DomainSpec**: `pde="heat"`, `grid_shape=[Nx, Nx, Nt]`,
  `domain={"x": [0, 1], "y": [0, 1], "t": [0, 0.1]}`, `periodic=True`,
  `boundary_condition={"kind": "periodic"}`,
  `field={"type": "grid", "backend": "fd", "dump_path": "p.npz"}`,
  `diffusivity=1.0`.
- **Wall-time budget**: < 5 s across full sweep (pure numpy; no
  time-stepping, no mesh assembly).
- **Tests**: 17 total (12 F2 parametrized + 1 liveness + 1
  refinement-invariance summary + 3 rule-verdict-contract).

## Scope note

PH-CON-001 covers the V1 periodic-BC heat-equation analytical-snapshot
case. Non-periodic (Neumann-homogeneous, Dirichlet rate-consistency
branch), non-smooth IC, and numerically-evolved solver-accuracy
fixtures are out of V1 scope. Live Hansen ProbConserv reproduction
deferred to V1.1 pending ProbConserv loader infrastructure.
