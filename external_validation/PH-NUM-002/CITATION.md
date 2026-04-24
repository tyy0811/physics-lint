# PH-NUM-002 — Refinement convergence rate (observed-order `p_obs`)

## Scope-separation discipline (read first)

PH-NUM-002 validates **the production rule's ability to detect observed
convergence-order on explicitly declared manufactured-solution cases**. The
expected rate is **case-specific** and depends on PDE, backend, boundary
treatment, and asymptotic regime. The anchor does **not** certify
convergence for arbitrary PDE / backend / BC triples.

The F2 fixture splits into **three scoped cases** plus one authoritative
harness-level methodology anchor:

- **Case A — F2 harness-level (authoritative).** `_harness/mms.py`
  `mms_observed_order_fd2` computes `p_obs = log₂(r_h / r_{h/2})` on a
  pure-interior 2nd-order central-difference Laplacian applied to smooth
  harmonic fixtures. This is the textbook-clean observed-order anchor:
  `p_obs → 2` as `h → 0`, independent of the rule's FD4 stencil. The
  harness-level layer is the authoritative F1 validation of the Roy 2005
  observed-order algorithm.
- **Case B — rule-verdict, FD + non-periodic.** Rule `PH-NUM-002` with
  `backend="fd"`, `periodic=False`, Dirichlet BC. Shipped FD4 has a
  2nd-order boundary band (`ph_num_002.py:9-22`); on a 2D grid the
  `4N`-cell boundary-band residual dominates the `N²`-cell interior at
  `h^{2.5}` by area-weighted scaling (derivation below). Measured
  `p_obs ≈ 2.50`, asymptotically approached from above (2.62 at N=16→32
  trending to 2.51 at N=128→256).
- **Case C — rule-verdict, saturation floor.** Either spectral+periodic on
  a period-compatible harmonic (= constant by Liouville on T², see scope
  note below), or FD+non-periodic on a harmonic polynomial where 2nd-order
  FD is exact (e.g. `x²−y²`). Both residuals fall below the rule's
  `_SATURATION_FLOOR = 1e-11`; rule returns `rate = inf` and `PASS`. No
  algebraic rate is asserted in this regime — the anchor verifies the
  floor behavior only, per the 2026-04-24 revised contract.

V1 scope: the production rule is **Laplace-only** (`ph_num_002.py:92`);
Poisson / heat / wave SKIP explicitly. Case-contract tests cover the SKIP
paths as structural V1 scope boundaries, not as reproduction claims.

## Function-labeled citation stack

Per complete-v1.0 plan §1.3. Authored during Task 12 on 2026-04-24.

### Mathematical-legitimacy (Tier 2 theoretical-plus-multi-paper)

- **Primary — Lax equivalence.** Strikwerda, J.C. (2004). *Finite
  Difference Schemes and Partial Differential Equations*, 2nd ed. SIAM.
  ISBN 978-0-89871-567-5. **Chapter 10, section-level** per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠. Consistency (local truncation
  `O(h^p)`) + stability (uniform bound on the discrete operator) implies
  convergence at order `p` for well-posed linear IVP/BVPs.
- **Primary — observed-order (`p_obs`) methodology.** Roy, C.J. (2005).
  "Review of code and solution verification procedures for computational
  simulation." *Journal of Computational Physics* 205, 131–156. DOI
  [10.1016/j.jcp.2004.10.036](https://doi.org/10.1016/j.jcp.2004.10.036).
  Formalizes `p_obs = log₂(e_h / e_{h/2})` as the observed-order
  diagnostic for a discretization in its asymptotic regime.
- **Secondary framing — Céa's lemma (FE analog).** Ciarlet, P.G. (2002).
  *The Finite Element Method for Elliptic Problems*. SIAM Classics 40.
  ISBN 978-0-89871-514-9. **§3.2, section-level** per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠. For FE discretizations of
  elliptic problems, Céa bounds the error by the best-approximation error
  scaled by continuity / coercivity constants; together with
  interpolation-error estimates this gives `‖u − u_h‖ ≤ C h^p`. Not
  directly applicable to FD, but establishes the same consistency +
  stability → convergence logic.
- **Secondary framing — verification procedure.** Oberkampf, W.L. &
  Roy, C.J. (2010). *Verification and Validation in Scientific Computing*.
  Cambridge University Press. ISBN 978-0-521-11360-1. **Chapters 5–6,
  section-level** per `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠. Formal
  verification procedure including observed-order testing, Richardson
  extrapolation, and asymptotic-regime diagnostics.

**Structural-equivalence proof-sketch** (section-level framing per §6.4,
no tight theorem-number claims):

1. **Consistency.** For a linear discrete operator `L_h` approximating
   `L = −Δ` on a smooth `u ∈ C^{p+2}(Ω)`, the local truncation error is
   `e_trunc(x, h) = (L_h u − L u)(x) = C_L · h^p · D^{p+2} u(x) + O(h^{p+2})`
   where `C_L` depends on the stencil. For an interior 2nd-order central
   difference on a 2D Cartesian grid, `C_L = 1/12` and `p = 2`
   (`(L_h − L) u = (h²/12) (∂⁴u/∂x⁴ + ∂⁴u/∂y⁴) + O(h⁴)`).
2. **L² scaling on a 2D domain.** Writing `r = L_h u − L u` and summing
   over the grid with cell area `h²`:
   - Interior sum: `N²` cells × `h²` × `(C h^p)²` = `h² · h^{2p} · N²`;
     with `N ∼ 1/h` this gives `||r_int||_{L²}² = O(h^{2p})`, so
     `||r_int||_{L²} = O(h^p)`.
   - Boundary-band sum (FD4 outer-ring 2nd-order band): `≈ 4N` cells ×
     `h²` × `(C' h²)²` = `4 · h⁵`; with `N ∼ 1/h` and taking `√`,
     `||r_bdy||_{L²} = O(h^{5/2}) = O(h^{2.5})`.
   - Mixed regime: total `||r||_{L²} = max(||r_int||, ||r_bdy||)`.
3. **Case A (2nd-order interior-only).** `p = 2` interior, no boundary
   band (harness uses interior-only quadrature). `||r||_{L²} = O(h²)` →
   `p_obs = log₂(h_c² / h_f²) = 2` at `h_c = 2h_f`.
4. **Case B (FD4 + non-periodic).** Interior `p = 4` gives `||r_int|| =
   O(h⁴)`; boundary band gives `||r_bdy|| = O(h^{2.5})`. At finite N the
   boundary term dominates: `p_obs → 2.5` as `h → 0`. Measured asymptote
   2.50 (Case B precheck below), consistent within ≤0.12 at N=16→32.
5. **Case C (saturation).** When `u` is representable at float precision
   by the discrete operator (periodic harmonic = constant by Liouville;
   harmonic polynomial degree ≤ 2 on 2nd-order FD; bandlimited Fourier
   modes on spectral), `||r||_{L²}` is pure float64 roundoff
   (`~1e-13` or smaller on `[0,1]²`). Both coarse and fine residuals fall
   below `_SATURATION_FLOOR = 1e-11` and `rate = inf` by design
   (`ph_num_002.py:115-121`). The rule correctly PASSes; the anchor does
   not assert an algebraic rate in this regime.

**Liouville on T² (scope-truth observation).** The only harmonic
functions on a 2-torus are constants (maximum-principle argument or
Liouville on the periodic cover). Therefore **there is no non-constant
harmonic period-compatible fixture** for the rule's FD+periodic or
spectral+periodic path; the only reachable behavior is saturation
(constants) or rule-correct WARN (non-harmonic, where `Δu ≠ 0` and the
rule flags non-harmonicity). The rule's docstring mentions
"fd4 interior-dominated (periodic): ~4 per doubling" — this regime is
**structurally unreachable** with any non-constant harmonic fixture on a
periodic grid. The rule's shipped behavior is correct (saturation on
constants, WARN on non-harmonic inputs); the docstring is aspirational
for a regime that cannot be constructed.

### Correctness-fixture (CI-runnable, non-credibility-claiming)

**F2 layer A — harness-level authoritative** (`_harness/mms.py`
`mms_observed_order_fd2`). Interior-only 2nd-order central-difference
Laplacian on smooth harmonic fixtures:

| Fixture | N pair | `p_obs` | r_coarse | r_fine |
|---------|--------|---------|----------|--------|
| `exp(x)·cos(y)` on [0,1]² | 32→64   | 2.0196 | 2.550e-4 | 6.289e-5 |
| `exp(x)·cos(y)`           | 64→128  | 2.0100 | 6.289e-5 | 1.561e-5 |
| `exp(x)·cos(y)`           | 128→256 | 2.0050 | 1.561e-5 | 3.890e-6 |
| `exp(x)·cos(y)`           | 256→512 | 2.0025 | 3.890e-6 | 9.708e-7 |
| `sin(πx)·sinh(πy)`        | 32→64   | 2.0073 | 5.170e-2 | 1.286e-2 |
| `sin(πx)·sinh(πy)`        | 64→128  | 2.0041 | 1.286e-2 | 3.206e-3 |
| `sin(πx)·sinh(πy)`        | 128→256 | 2.0021 | 3.206e-3 | 8.002e-4 |

**Case A acceptance band:** `p_obs ∈ [1.9, 2.1]` across both fixtures at
N pairs `{32→64, 64→128, 128→256}`; tolerance `±0.1` calibrated from the
2026-04-24 precheck. `N=16→32` is mildly pre-asymptotic (2.037) but
still inside the band so it is included. Valid N range: `N ∈ {16, 32, 64,
128, 256, 512}`, pairs separated by a factor of 2.

**F2 layer B — rule-verdict, FD + non-periodic (boundary-dominated).**
Rule `PH-NUM-002.check()` on smooth harmonic fixtures with `backend="fd"`,
`periodic=False`, Dirichlet BC:

| Fixture | N pair | Rule `refinement_rate` | Rule status |
|---------|--------|------------------------|-------------|
| `exp(x)·cos(y)` | 16→32   | 2.6210 | PASS |
| `exp(x)·cos(y)` | 32→64   | 2.5585 | PASS |
| `exp(x)·cos(y)` | 64→128  | 2.5288 | PASS |
| `exp(x)·cos(y)` | 128→256 | 2.5143 | PASS |
| `sin(πx)·sinh(πy)` | 16→32   | 2.4805 | PASS |
| `sin(πx)·sinh(πy)` | 32→64   | 2.4830 | PASS |
| `sin(πx)·sinh(πy)` | 64→128  | 2.4903 | PASS |
| `sin(πx)·sinh(πy)` | 128→256 | 2.4949 | PASS |

**Case B acceptance band:** `refinement_rate ∈ [2.3, 2.8]` across both
fixtures at N pairs `{16→32, 32→64, 64→128, 128→256}`; tolerance `±0.25`
from a center of 2.55, calibrated from precheck. Rule threshold 1.8 →
all pairs PASS. Measured asymptote 2.50 matches the theoretical boundary-
dominance scaling `O(h^{2.5})` derived in the F1 proof-sketch step 4.

**F2 layer C — rule-verdict, saturation floor.** Either path triggers
rule's `_SATURATION_FLOOR = 1e-11` clamp (`ph_num_002.py:115-121`):

| Fixture | Backend / BC | N pair | Rule `refinement_rate` | Rule status |
|---------|--------------|--------|------------------------|-------------|
| `u = 0` constant | spectral, periodic | 32→64 | `inf` | PASS |
| `x² − y²` harmonic polynomial | fd, non-periodic | 32→64 | `inf` | PASS |

**Case C acceptance:** `refinement_rate == float('inf')` and `status ==
'PASS'`. No algebraic rate asserted. Both residuals fall below
`_SATURATION_FLOOR` — the spectral+periodic path because Liouville forces
periodic harmonics to constants, and the polynomial path because
2nd-order FD is exact on polynomials of degree ≤ 2.

**SKIP-path contracts.** V1 scope boundary — Laplace-only. `pde="poisson"`
SKIPs with the Poisson source-subtraction justification; `pde="heat"`
SKIPs with the time-axis Laplacian justification (`ph_num_002.py:92-97`).
Anchor tests cover both paths as structural-boundary validation.

### Borrowed-credibility (external published reproduction layers)

**F3 absent with justification.** Per the 2026-04-24 user-revised contract
(plan-diff 17): "If no live external MMS benchmark is executable in V1,
mark absent-with-justification; published MMS / verification literature
can remain mathematical or supplementary context."

No live external MMS benchmark harness is executable for PH-NUM-002 in V1:

- **Oberkampf-Roy 2010 Chs 5–6** publishes the methodology (`p_obs`
  algorithm, Richardson extrapolation, asymptotic-regime diagnostics)
  but does **not** ship a reproducible numerical dataset with published
  `p_obs` values for a specific PDE+backend+BC triple that PH-NUM-002
  could consume. The textbook's role here is methodology + theoretical
  framing, not reproduction target.
- **Roy 2005 (JCP)** similarly specifies the methodology; its published
  numerical examples span multiple Roy-internal codes (not physics-lint)
  with custom MMS setups. No drop-in reproduction anchor.
- **PDEBench / Hansen ProbConserv / NSForCFD** do not report `p_obs` as a
  headline metric; they report aggregate RMSE or nRMSE, which is a
  different summary statistic.

No F3-INFRA-GAP risk (F3-absent is structural — no loader-infrastructure
gap would help; the published materials are methodology, not datasets).
Both Oberkampf-Roy 2010 and Roy 2005 retained above in Mathematical-
legitimacy as theoretical backbone + methodology reference. The Tier-2
theoretical-plus-multi-paper reproduction (Lax equivalence + Roy 2005
`p_obs` formula + harness-level p=2 anchor + rule-verdict p=2.5 boundary-
dominated anchor + saturation-floor anchor) carries the credibility here.

### Supplementary calibration context

- **Roy 2005 published numerical examples** — Roy, C.J. (2005) *J. Comput.
  Phys.* 205, 131–156, DOI
  [10.1016/j.jcp.2004.10.036](https://doi.org/10.1016/j.jcp.2004.10.036).
  Section 4 reports observed-order verification on linear advection,
  Laplace, and Navier-Stokes with Roy-internal codes. **Flagged:
  methodology-level reference**; not a direct reproduction target
  (different codes, different stencils). Physics-lint's measured
  asymptotes (Case A → 2, Case B → 2.5) are qualitatively consistent with
  the paper's observation that 2D FD codes with non-periodic BCs settle
  at the boundary-stencil order.
- **Richardson extrapolation stability** — plan §20 enumerate-the-splits
  item (b) "three-level vs four-level Richardson extrapolation." Not
  exercised in V1 (rule's shipped path uses simple two-level `log₂` ratio,
  `ph_num_002.py:127`). Plan-diff 16 logs the scoping.
- **p-refinement** — plan §20 enumerate-the-splits item (c) "h vs p
  refinement." Not applicable (physics-lint is FD-only; no polynomial-
  order refinement). Deferred to v1.2 per plan §3 backlog.

## Citation summary

- **Primary (mathematical-legitimacy, Tier 2)**: Strikwerda 2004 Ch 10
  Lax equivalence (section-level ⚠); Roy 2005 JCP `p_obs` formula (DOI
  10.1016/j.jcp.2004.10.036); Ciarlet 2002 §3.2 Céa's lemma (section-level
  ⚠); Oberkampf-Roy 2010 Chs 5–6 (section-level ⚠).
- **F2 harness-level**: `external_validation/_harness/mms.py`
  `mms_observed_order_fd2`. Tested at `N ∈ {16, 32, 64, 128, 256}` on
  `{exp(x)cos(y), sin(πx)sinh(πy)}`.
- **F2 rule-verdict boundary-dominated**: `PH-NUM-002.check()` on
  `fd`+non-periodic at `N ∈ {16, 32, 64, 128, 256}` on the same two
  fixtures.
- **F2 rule-verdict saturation**: `PH-NUM-002.check()` on spectral
  +periodic `u=0` and `fd`+non-periodic `x²−y²`, both returning
  `rate=inf`.
- **Pinned values**:
  - Case A `p_obs` asymptote 2.00, measured 2.01–2.04 at N ≥ 32→64;
  - Case B `refinement_rate` asymptote 2.50, measured 2.48–2.63 at
    N ≥ 16→32;
  - Case C `refinement_rate` = `inf` exactly.
- **F3**: absent-with-justification (no live MMS benchmark dataset;
  methodology-only references retained in F1 + Supplementary).
- **Verification date**: 2026-04-24.
- **Verification protocol**: four-layer (F1 Lax-equivalence + Roy 2005
  `p_obs` proof-sketch with L²-scaling derivation of Case A p=2 and
  Case B p=2.5 + F2 harness-level observed-order anchor + rule-verdict
  boundary-dominated + rule-verdict saturation floor + SKIP-path
  contracts).

## Pre-execution audit

PH-NUM-002 is a continuous-math rule (refinement-convergence detection).
Per complete-v1.0 plan §6.2 Tier B enumerate-the-splits allocation
(0.15 d), the splits audited are:

- **Laplace vs Poisson vs heat.** Rule V1 scope is Laplace-only per
  `ph_num_002.py:92` (explicit SKIP for Poisson/heat with documented
  reasons: Poisson needs source subtraction; heat would differentiate
  the time axis). Plan §20 "SymPy MMS for Laplace/Poisson/heat" narrowed
  to Laplace-only F2 with SKIP-path contract tests. **Plan-diff 14**.
- **Three-level vs four-level Richardson extrapolation.** Rule's shipped
  path is simple two-level `log₂` ratio (`ph_num_002.py:127`). Richardson
  extrapolation does not enter the rule's emitted quantity. Plan §20's
  three-vs-four-level split is not part of V1 scope; logged as
  Supplementary methodology reference. **Plan-diff 16**.
- **h-refinement vs p-refinement.** Physics-lint is FD-only (no FE
  polynomial-order refinement); p-refinement is structurally out-of-
  scope. Plan §3 backlog item for v1.2 per plan §20 Risks section.
- **Backend + BC combos (the actual load-bearing split).** Rule's shipped
  docstring enumerates three regimes (`ph_num_002.py:9-22`):
  fd+periodic interior-dominated ~4; fd+non-periodic boundary-dominated
  ~2–2.5; spectral+periodic saturation. Liouville forces periodic
  harmonics to constants; the fd+periodic regime is structurally
  unreachable with non-constant fixtures. V1 F2 scope: Case A harness
  methodology anchor (`p_obs ≈ 2`) + Case B rule fd+non-periodic
  (boundary-dominated `p_obs ≈ 2.5`) + Case C saturation floor. Rule-
  docstring structural-unreachability observation logged in the
  Liouville scope-truth note. **Plan-diff 18**.
- **Single-tolerance vs per-case tolerance.** Plan §20's
  "`p_obs` matches expected within 0.1" is a single global band. User's
  2026-04-24 revised Task 12 contract replaces it with per-case
  tolerance per (case, backend, BC): Case A `±0.1`, Case B `±0.25`,
  Case C exact-`inf`. **Plan-diff 15**.

Audit outcome: V1 F2 scope = three-case layout with per-case tolerance
bands; no reconfiguration needed beyond the plan-diffs listed. Audit
cost 0.15 d absorbed into Task 12 budget.

## Test design

- **Case A fixtures** (`test_anchor.py`):
  `harmonic_expcos_interior`, `harmonic_sin_sinh_interior` — return
  `(u, hx, hy)` tuples on `[0,1]²` at N∈{16,32,64,128,256,512}.
- **Case B fixtures**: `harmonic_expcos_rule`, `harmonic_sin_sinh_rule` —
  return `GridField` with `backend="fd"`, `periodic=False`.
- **Case C fixtures**: `const_zero_spectral` (spectral+periodic),
  `harmonic_polynomial_fd` (FD4 non-periodic on `x²−y²`).
- **SKIP-path fixtures**: `poisson_periodic`, `heat_periodic` —
  `DomainSpec` with `pde="poisson"` / `pde="heat"`.
- **DomainSpec** (Case B): `pde="laplace"`, `grid_shape=[Nx, Nx]`,
  `domain={"x": [0, 1], "y": [0, 1]}`, `periodic=False`,
  `boundary_condition={"kind": "dirichlet"}`, `field.backend="fd"`.
- **Wall-time budget**: < 20 s across full sweep (largest pair N=256→512
  on harness, 128→256 on rule).
- **Tests**: 21 total
  - 6 Case A parametrized `(fixture × N-pair)`
  - 2 Case A log-log slope convergence (one per fixture)
  - 8 Case B parametrized `(fixture × N-pair)`
  - 1 Case B asymptote check (measured rate → 2.50)
  - 2 Case C saturation paths
  - 2 SKIP contracts (Poisson, heat).

## Scope note

PH-NUM-002 V1 covers:

- **Case A (harness methodology anchor)**: 2D non-periodic smooth
  harmonic fixtures with interior-only 2nd-order FD Laplacian.
- **Case B (rule-verdict, fd+non-periodic)**: 2D non-periodic smooth
  harmonic fixtures on the rule's FD4 path; boundary-dominated `p_obs`
  ≈ 2.5.
- **Case C (rule-verdict, saturation floor)**: spectral+periodic on
  constants; fd+non-periodic on harmonic polynomials. Rule returns
  `rate=inf` PASS; no algebraic rate asserted.

Out of V1 scope:

- Non-constant periodic harmonic fixtures (Liouville: they do not exist
  on T²).
- Poisson, heat, wave PDEs (rule SKIPs structurally).
- Three- or four-level Richardson extrapolation (not part of rule's
  emitted quantity).
- p-refinement (no FE polynomial-order in physics-lint; v1.2 backlog).
- Curvilinear / adaptive / non-uniform grids (FD rule is uniform-grid).

Live external MMS benchmark reproduction is structurally unavailable in
V1 (no published dataset ships a reproducible `p_obs` value for a
specific PDE+backend+BC triple that matches physics-lint's rule path).
F3 absent-with-justification per user's 2026-04-24 revised contract.
