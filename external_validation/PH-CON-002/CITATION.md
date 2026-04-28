# PH-CON-002 — Wave energy conservation (analytic snapshots)

## Scope-separation discipline (read first)

PH-CON-002 validates **the production rule's ability to measure
wave-energy drift on analytically controlled conservative snapshots**.
It does not certify the accuracy of a wave-equation time integrator.

The F2 fixture splits into two distinct layers:

- **F2 harness-level (authoritative):** computes `E(t)` directly from
  analytical `u_t, u_x, u_y` field components. Drift across time
  snapshots is roundoff-only because no numerical derivative is
  taken. This is the primary validation of the F1 identity
  `dE/dt = 0` at float64 precision.
- **Rule-verdict contract:** feeds analytical `u` snapshots through
  the production rule. The rule computes `u_t` internally via
  2nd-order central FD (`ph_con_002.py:65`), introducing
  `O(Δt²)` truncation error. Tolerance is method-dependent. This is
  **not** a leapfrog time-stepper; per user's 2026-04-24 revised
  contract, any leapfrog-evolved fixture would be labeled
  supplementary liveness with method-dependent tolerance — none is
  included in V1.

This document does not, and must not, claim the production rule
validates wave-equation solver accuracy. It claims:

- (F1) the rule's mathematical legitimacy is anchored in the wave-
  energy identity `dE/dt = 0` for compatible BCs;
- (F2 harness-level) on analytical snapshots, the energy functional
  is constant at roundoff (~5e-16 relative); tolerance `1e-14`;
- (rule-verdict contract) on the same analytical snapshots, the rule
  emits `O(Δt²)` drift — measured log-log slope 1.94, PASS at
  `Δt ≤ π/25` on one full wave period;
- (F3) absent with justification — pre-recorded by Task 0 Step 4
  pin audit (PDEBench wave-energy rows are semantically incompatible;
  Hansen ProbConserv CE is first-order-in-time, not second-order
  energy functional).

## Function-labeled citation stack

Per complete-v1.0 plan §1.3. Authored during Task 9 on 2026-04-24.

### Mathematical-legitimacy (Tier 2 theoretical-plus-multi-paper)

- **Primary — wave energy identity.** Evans, L.C. (2010). *Partial
  Differential Equations*, 2nd ed. Graduate Studies in Mathematics 19.
  AMS. ISBN 978-0-8218-4974-3. **§2.4.3 energy identity,
  section-level** per `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠
  (theorem number pending primary-source verification per §6.4).
- **Secondary framing — wave equation in one spatial dimension.**
  Strauss, W.A. (2007). *Partial Differential Equations: An
  Introduction*, 2nd ed. Wiley. ISBN 978-0-470-05456-7. **§2.2
  conservation of energy, section-level** per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠.
- **Secondary framing — symplectic integrator conservation.**
  Hairer, E., Lubich, C., Wanner, G. (2006). *Geometric Numerical
  Integration: Structure-Preserving Algorithms for Ordinary
  Differential Equations*, 2nd ed. Springer. ISBN 978-3-540-30663-4.
  **Chapter IX, section-level** per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠. Hairer-Lubich-Wanner IX
  characterizes bounded `O(Δt²)` energy oscillation for symplectic
  integrators; physics-lint does not ship a leapfrog stepper but the
  reference establishes the theoretical backdrop for why analytical-
  snapshot validation (F2 layer) is preferred over numerically-evolved
  validation for rule-metric scoping.
- **Structural-equivalence proof-sketch** (section-level framing per
  §6.4, no tight theorem-number claims):
  1. **Energy-identity preconditions.** For the wave equation
     `u_tt = c² Δu` on `Ω ⊂ ℝⁿ` with compatible boundary conditions
     (periodic, homogeneous Dirichlet, or homogeneous Neumann),
     `E(t) = ½ ∫_Ω (u_t² + c²|∇u|²) dV` is time-invariant. Derivation
     (Evans §2.4.3 section-level): `dE/dt = ∫(u_t u_tt + c² ∇u · ∇u_t)
     = ∫(u_t · c² Δu) + c² ∫(∇u · ∇u_t)`; integration by parts on the
     second term gives `c² ∫∇u · ∇u_t = −c² ∫u_t Δu + c² ∮u_t ∂u/∂n
     dS`; the boundary integral vanishes under compatible BCs
     (periodic has no boundary; homogeneous Dirichlet makes `u_t = 0`
     on boundary; homogeneous Neumann makes `∂u/∂n = 0`). So `dE/dt
     = 0`.
  2. **Analytical-snapshot fixture.** Choose `u(x, y, t) = sin(kx)
     cos(ckt)` on `[0, 2π]²` periodic (y-independent). Verify:
     `u_tt = −(ck)² u`, `u_xx = −k² u`, `u_yy = 0`; `c² Δu = −c²k² u
     = u_tt` ✓. Energy density = `½(c²k² sin²(kx) sin²(ckt) +
     c²k² cos²(kx) cos²(ckt))` with y-factor 1. Integrated over
     `[0, 2π]²`: `E(t) = ½ c²k² [sin²(ckt) ∫₀^{2π}sin²(kx) dx ·
     2π + cos²(ckt) ∫₀^{2π}cos²(kx) dx · 2π] = ½ c²k² · π · 2π ·
     [sin²(ckt) + cos²(ckt)] = π²c²k²`. Constant in `t`. For
     `c = k = 1`: `E = π² ≈ 9.8696`.
  3. **Discrete-integral invariance on periodic grid.** Rectangle
     rule (endpoint-exclusive) is spectrally accurate on periodic
     analytic integrands — the harness-level energy computation from
     analytical `u_t, u_x, u_y` gives `E_discrete(t_k) = E_exact(t_k)
     + O(roundoff)` at every time snapshot.
  4. **Rule's emitted quantity.** PH-CON-002 emits `max_t
     |E(t) − E(0)| / |E(0)|`. When `u_t` is supplied analytically,
     the metric lands at roundoff. When the rule computes `u_t`
     internally via central FD (production path), the kinetic term
     acquires `O(Δt²)` error; rule drift scales as `α·Δt²` with
     `α ≈ (ck)²/3` for the sinusoidal fixture. The rule's
     classification is PASS when `Δt ≤ π/25` on one full wave
     period, matching the 0.01 shipped threshold.

### Correctness-fixture (CI-runnable, non-credibility-claiming)

**F2 layer A — harness-level authoritative.** `u(x, y, t) = sin(kx)
cos(ckt)` via `external_validation/_harness/energy.py`'s
`analytical_wave_snapshot_2d_yindep(fixture, nx, nt, c, k, t_end)`.
Energy computed via `wave_energy_from_analytical_fields(u_t, u_x,
u_y, c, h, periodic)` directly from analytical components.

**Measured energy-drift floor (2026-04-24, float64, c=k=1, one wave
period):**

| Nx  | Nt=11    | Nt=21    | Nt=51    |
|-----|----------|----------|----------|
| 16  | 3.60e-16 | 3.60e-16 | 3.60e-16 |
| 32  | 1.80e-16 | 1.80e-16 | 3.60e-16 |
| 64  | 1.80e-16 | 1.80e-16 | 5.40e-16 |

Max observed relative drift: **5.40e-16** — refinement-invariant
pure roundoff (no N or Nt scaling, confirming spectral accuracy of
rectangle rule + exact analytical fields). Tolerance
`HARNESS_ENERGY_TOL = 1e-14` gives ~20× safety over observed.
`E(0) = π² ≈ 9.8696` matches analytical value to float64 precision.

**Rule-verdict contract — method-dependent drift.** Rule drift on
analytical `u` snapshots (rule computes `u_t` via 2nd-order central
FD internally):

| Nt  | Δt    | raw_value | status |
|-----|-------|-----------|--------|
| 11  | 0.628 | 1.16e-1   | WARN   |
| 21  | 0.314 | 3.25e-2   | WARN   |
| 51  | 0.126 | 5.23e-3   | PASS   |
| 101 | 0.063 | 1.32e-3   | PASS   |

Log-log slope of drift vs Δt: **1.94** (theoretical 2 for central-FD
O(Δt²) truncation). Rule PASSes at `Δt ≤ π/25 ≈ 0.126`, matching
the shipped 0.01 threshold for c=k=1. Refinement-independent on
spatial `nx` (spread across `nx ∈ {16, 32, 64}` at fixed `nt = 51`
is below 1e-6, well inside the time-FD-dominated regime).

**Rule anchor assertions** (17 tests total):
- 9 F2 harness-level parametrized (`Nx × Nt` sweep) + 2 E(0)-value
  checks (c=k=1; c=2,k=1 / c=1,k=2 / c=2,k=2 scaling).
- 1 RVC PASS assertion at `nt = 51` with theoretical-drift range.
- 1 RVC log-log-slope-≈-2 convergence assertion across `nt ∈ {11,
  21, 51, 101}`.
- 1 RVC refinement-independence on spatial `nx`.
- 3 rule-verdict-contract SKIP paths: non-wave PDE, non-energy-
  conserving BC, `nt < 3`.

### Borrowed-credibility (external published reproduction layers)

**F3 absent with justification** (pre-recorded by Task 0 Step 4 pin
audit, `docs/audits/2026-04-22-pdebench-hansen-pins.md:207`).

- **PDEBench.** No standalone wave-equation dataset. Closest
  candidates: shallow-water equation (reports mass-conservation
  cRMSE, not wave-energy — semantically incompatible); compressible
  Navier-Stokes (mass row only; paper does not break out energy-
  conservation metrics). Both demoted from F3 to Supplementary
  per Task 0 audit.
- **Hansen ProbConserv.** CE metric is defined for first-order-in-
  time integral conservation laws (`∂_t u + div J = 0`), not
  second-order-in-time wave-energy functionals
  `E[u, u_t] = ½∫(u_t² + c²|∇u|²)`. Different mathematical object;
  reproduction not applicable.

Per plan §1.2, F3-absent-is-structural for rules whose canonical
reproduction target is mathematically incompatible with the rule's
emitted quantity. No F3-INFRA-GAP risk (F3-absent is structural, not
a loader-infrastructure gap). The Tier-2 theoretical-plus-multi-
paper reproduction (wave-energy identity + analytical-snapshot
harness-level E(t) at roundoff + rule-verdict O(Δt²) convergence
test) carries the credibility here.

### Supplementary calibration context

- **PDEBench shallow-water cRMSE (mass, not energy)** — Takamoto et
  al. 2022 `arXiv:2210.07182` Table 5/Supplement §D.6. **Flagged:
  different conservation functional**; physics-lint's PH-CON-002
  emitted quantity is second-order-in-time wave energy, PDEBench's
  is first-order mass. Not a reproduction.
- **Hansen ProbConserv CE (first-order, not second-order energy)**
  — Hansen et al. 2024 Physica D `arXiv:2302.11002` Table 1.
  **Flagged: CE metric defined for `∂_t u + div J = 0` conservation
  laws**; PH-CON-002's rule target is a different mathematical
  object. Not a reproduction.

## Citation summary

- **Primary (mathematical-legitimacy, Tier 2)**: Evans 2010 §2.4.3
  (section-level per `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠;
  theorem number pending local copy) + Strauss 2007 §2.2
  (section-level ⚠) + Hairer-Lubich-Wanner 2006 Ch IX (section-level
  ⚠).
- **F2 harness-level**: `external_validation/_harness/energy.py`
  `analytical_wave_snapshot_2d_yindep` + `wave_energy_from_
  analytical_fields`. Tested at `Nx ∈ {16, 32, 64}` × `Nt ∈ {11,
  21, 51}`.
- **Pinned values**: (F2 harness) max relative energy drift ≤
  5.40e-16 across sweep; `E(0) = π²` matches analytical;
  (rule-verdict) log-log slope = 1.94 vs theoretical 2; rule PASSes
  at `Δt ≤ π/25`.
- **F3**: absent-with-justification (pre-recorded by Task 0 Step 4
  pin audit; PDEBench and Hansen both in Supplementary with
  semantic-mismatch flags).
- **Verification date**: 2026-04-24.
- **Verification protocol**: three-layer (F1 energy-identity proof-
  sketch + F2 harness-level energy invariance at roundoff +
  rule-verdict O(Δt²) convergence + SKIP-path contracts).

## Pre-execution audit

PH-CON-002 is a continuous-math rule (wave-energy conservation drift).
Per complete-v1.0 plan §6.2 Tier C enumerate-the-splits allocation
(0.1 d), the splits audited are:

- **Symplectic vs non-symplectic time-stepper.** V1 F2 scope:
  analytical snapshots, no time-stepper. Rule computes `u_t` via
  central FD internally, not leapfrog. Symplectic vs non-symplectic
  distinction is moot at this layer.
- **1D vs 2D wave.** Rule requires `ndim ≥ 3`; F2 uses 2D
  y-independent extension of 1D fixture (plan-diff 13).
- **Reflective BC vs periodic.** V1 F2 periodic only. Homogeneous
  Dirichlet / Neumann also conserve energy per spec.py:82 but are
  not tested here (SKIP-path test covers the non-conserving BC
  branch via `dirichlet` / `neumann` inhomogeneous kinds).
- **Fixture mode: analytical-snapshot vs numerically-evolved**
  (2026-04-24 user-approved revised contract): analytical-snapshot
  only. Numerically-evolved (leapfrog or otherwise) supplementary
  liveness tests with method-dependent tolerance are explicitly
  out of V1 scope — plan-diff 12.

Audit outcome: F2 scope = 2D y-independent periodic analytical-
snapshot; no reconfiguration required. Plan-diffs 12, 13 log the
fixture-mode scoping + 1D→2D extension. Audit cost 0.1 d absorbed
into Task 9 budget.

## Test design

- **Harness-level fixture (F2 layer A)**:
  `analytical_wave_snapshot_2d_yindep` in `_harness/energy.py` —
  returns analytical `u, u_t, u_x, u_y` arrays and spacing tuple.
- **Analytic function**: `u(x, y, t) = sin(kx) cos(ckt)` on
  `[0, 2π]²` periodic (y-independent). `E = π²c²k²`.
- **Refinement sweep**: `Nx ∈ {16, 32, 64}`, `Nt ∈ {11, 21, 51}`
  for harness layer; `nt ∈ {11, 21, 51, 101}` for rule-verdict
  convergence.
- **DomainSpec**: `pde="wave"`, `grid_shape=[Nx, Nx, Nt]`,
  `domain={"x": [0, 2π], "y": [0, 2π], "t": [0, period]}`,
  `periodic=True`, `boundary_condition={"kind": "periodic"}`,
  `field={"type": "grid", "backend": "fd", "dump_path": "p.npz"}`,
  `wave_speed=c`.
- **Wall-time budget**: < 10 s across full sweep.
- **Tests**: 17 total (11 harness-level F2 incl. parametrized sweep
  and scaling + 3 rule-verdict method-dependent + 3 SKIP-path
  contracts).

## Scope note

PH-CON-002 covers the V1 periodic-BC wave-equation analytical-
snapshot case. Non-periodic BCs (homogeneous Dirichlet / Neumann
would also conserve energy but are not in F2 scope), non-smooth
solutions, numerically-evolved leapfrog fixtures, and all 3D cases
are out of V1. Live PDEBench and Hansen reproduction are
structurally-unavailable (semantically-incompatible functionals) —
no V1.x loader deferral applies here.
