# 2026-04-24 Forward-look plan smoke precheck — Tasks 4–13

**Purpose.** Path C output per the 2026-04-24 §7.3 n≥2 escalation (plan
`docs/plans/2026-04-22-physics-lint-external-validation-complete.md`).
Tier-B execution surfaced five plan-execution mismatches across Tasks 2–3
(Task 2 diffs 1–2; Task 3 diffs 3–5, all documented in the commits
`30baf3e` and `0cedc7b`). The pattern is "plan-specified per-task
numerical/API details were not smoke-checked against rule source,
harness, or machine-precision regime at plan-writing time." Path C
commits ~0.2 ED to a forward-look precheck before Task 4 begins so
Tasks 4–13 can either execute cleanly or surface the known gaps *before*
a code commit requires a retrofit.

**Scope.** Execution-level numerical commitments only: N ranges,
tolerance floors, expected fixture values, p-values / intorder menus,
optional dependency behavior, cross-reference claims. Not re-audited:
plan structure, F1/F2/F3 theorem-level citations, literature-pin DOIs
(those are backbone discipline scoped to Task 0).

**Method.** For each remaining task (§12–§21 = Tasks 4–13), read the
rule source at `src/physics_lint/rules/<rule_id>.py`, compare each
plan-specified execution detail against the rule's emitted quantity
and the rule's stated V1 scope, and classify as:

- **CRITICAL** — plan requires the rule to emit a quantity the V1 rule
  does not emit (stub, different semantic, or out-of-scope branch).
  Harness-level computation or plan rewrite required before execution.
- **FLAG** — plan-specified tolerance or range likely to need widening
  or restructuring based on method-level analysis. Smoke-check at task
  execution time with 5-minute empirical runs.
- **CLEAR** — plan and rule align on the checked items. No action
  needed before task execution. Task-specific audit (§6.2 enumerate-
  the-splits) still applies.

**Disposition.** For CRITICAL findings, the recommended resolution is
harness-level Function 2 computation (the pattern used by Task 3's
Layer 1 + 2 residuals, which are computed independently of the rule's
emitted quantity). This preserves the plan's §1.3 three-function-
labeled structure — Function 2 validates the published theoretical
property on a closed-form fixture; the rule's emitted quantity is a
different semantic under V1 scope, and a fourth "rule-verdict sanity"
layer exercises the rule path separately.

---

## CRITICAL: three tasks whose rule is a V1 stub

### Task 5 (PH-BC-002) — rule is Laplace-imbalance, not arbitrary-F Gauss-Green

**Rule scope** (`src/physics_lint/rules/ph_bc_002.py:1-11`): "Week 1
scope: Laplace only (expected imbalance is zero). The Poisson arm raises
`NotImplementedError` to surface the unfinished wiring loudly instead of
silently computing a wrong answer." Emitted quantity is `∫(-Δu) dV +
∫∂u/∂n dS` for a Laplace field (both integrals ≈ 0 for a harmonic
function).

**Plan §13 Execution step 3:** "Implement `F=(x,y)` on unit square
triangulation + quadrilateralization; both yield LHS = RHS = 2 within
tolerance." Plan's fixture tests `∫div(F) dV = ∫F·n dS` for `F=(x,y)`
— this is the general Gauss-Green theorem. The rule does not take `F`;
it takes a Laplace field `u` and checks `∫Δu + ∮∂u/∂n = 0`.

**Disposition.** Function 2 is harness-level, not rule-level. Two
options:
- **Option A (harness-level F=(x,y) Gauss-Green).** Implement the
  `F=(x,y)` closed-form fixture in `external_validation/_harness/
  divergence.py` (already stubbed per §3) as an independent Gauss-Green
  check, with a fourth rule-verdict layer that exercises the rule's
  actual Laplace-imbalance scope on e.g., `u = x² − y²` (harmonic,
  analytical `∂u/∂n` on unit square).
- **Option B (rewrite plan §13 fixture to a harmonic Laplace fixture
  that matches the rule).** Use `u = x² − y²` or `u = sin(πx)sinh(πy)`
  where the rule emits ≈ 0 for the correct field.

Recommendation: Option A. Preserves the plan's Gauss-Green theoretical
anchor (F1 = Evans App C.2) while adding a rule-verdict layer on the
rule's actual Laplace scope. Matches Task 3's three-layer pattern.

**Tolerance note.** Plan's "1e-12" applies to Option A (closed-form
integration of a degree-1 polynomial). For Option B on a Laplace-
harmonic fixture with FD4 boundary normal-derivative, the achievable
tolerance is order `O(h²)` = `O(1e-4)` at N≈64 (boundary stencil is
2nd-order; cannot tighten below the boundary-FD rate without spectral
backend + periodic BC, which is incompatible with the trace-theorem
fixture).

### Task 7 (PH-SYM-004) — rule is V1 structural stub emitting SKIPPED

**Rule scope** (`src/physics_lint/rules/ph_sym_004.py:1-21`): "V1
scope: this rule is a structural stub that always emits `SKIPPED` once
past its declared/periodic gates. True translation equivariance is a
*model* property — comparing `f(roll(x))` against `roll(f(x))` on a
live callable — and requires adapter-mode plumbing that lands in V1.1.
The prior implementation measured the offline quantity
`||roll(u) - u|| / ||roll(u)||`, but `np.roll` preserves norm, so the
triangle inequality caps this quantity at 2.0. A PASS-if-<2.0 threshold
rubber-stamped random noise, smooth ramps, and most structured fields…
The false-pass was removed rather than shipping a metric that cannot
fail on realistic inputs."

**Plan §15 Execution step 3:** "Implement random FNO-layer + random
input + random grid-shift fixture; assert commutation error < 1e-5 in
1D and 2D." Plan's `f(T_s x) = T_s f(x)` check is a model-level
property on a live FNO layer — exactly what the rule *cannot* do in V1.

**Disposition.** Function 2 is harness-level, not rule-level.
Implement the random-FNO-layer shift-commutation test as a free-
standing harness function (sensible home: `external_validation/_harness/
symmetry.py`, already extended in Task 1 for PH-SYM-001/002). Add a
fourth "rule-verdict contract" layer that exercises the rule's V1 path
(verifies the rule SKIPs with the documented reason — the V1 stub
contract). CITATION.md must be explicit that F2 tests the theoretical
property, not the rule's V1 emitted quantity; the rule verdict layer
protects against the V1 stub silently drifting.

**Tolerance note.** Plan's "< 1e-5" applies to 1D/2D grid-aligned
shifts under float64. FNO layer default dtype may be float32 depending
on library version (neuraloperator, pytorch_fno) — verify dtype at
task execution; 1e-5 in float32 is tight, 1e-8 in float64 is
comfortable.

### Task 11 (PH-NUM-001) — rule is V1 structural stub emitting PASS-with-reason

**Rule scope** (`src/physics_lint/rules/ph_num_001.py:1-16`): "V1
structural stub. `MeshField.integrate` does not expose a `qorder` kwarg
in V1, so this rule cannot compare quadrature at orders `q` and `2q`.
It ships as a structural stub: the rule module exists, the rule ID is
in the registry, and the CLI surface is stable. V1.1 can plug in the
real q-vs-2q check without breaking any public API. In V1 the rule
emits `PASS` with a `reason` string that says `'qorder convergence
check is a stub until V1.1'`."

**Plan §19 Execution step 4:** "vary `intorder ∈ {1, 2, 3, 4}`; measure
convergence rate against element order `p ∈ {1, 2, 3}`; assert rate
matches Ciarlet's theoretical prediction within 10%." The rule does not
vary `intorder` and does not compute a convergence rate.

**Disposition.** Function 2 is harness-level, not rule-level.
Implement the `(p, intorder, MMS)` convergence sweep using scikit-fem's
own `intorder` parameter directly in `external_validation/_harness/
quadrature.py` (already stubbed per §3). Add a fourth "rule-verdict
contract" layer that exercises the rule's V1 stub path (asserts the
rule returns PASS with the `'stub until V1.1'` reason on a MeshField
input — protects against silent drift of the stub contract).

**Budget note.** Task 11's 2.6 ED already includes 0.5 d for scikit-
fem correctness-fixture scaffolding (per §2.1) and 0.1 d for F3-hunt-
secondary migration prose. The harness-level quadrature sweep is
scope-consistent with the scikit-fem scaffolding allocation; no budget
revision required.

---

## FLAGS: four tolerance / range specs needing 5-min smoke checks at task execution

### Task 4 (PH-BC-001) — Neumann fixture semantic mismatch

**Rule scope** (`src/physics_lint/rules/ph_bc_001.py:33-48`): takes
`boundary_target: np.ndarray`, computes `||field.values_on_boundary() -
boundary_target||`. Mode-branches on `||boundary_target||` (absolute
below 1e-8, relative otherwise). The rule measures Dirichlet-type trace
violation (value on boundary).

**Plan §12 Execution step 4:** "Implement three BC fixtures on unit
square with analytical BC values for Dirichlet, Neumann, periodic;
assert rule emits zero (numerical tolerance) when BC is satisfied,
nonzero when violated by known amount."

**Semantic mismatch.** Rule checks `u|∂Ω - g`, a Dirichlet-type check.
Neumann would check `∂u/∂n|∂Ω - g` (flux target), which the rule does
not emit — the rule calls `values_on_boundary()`, not a normal-
derivative method. For periodic, the "boundary" is vacuous (no boundary
on a torus); values_on_boundary-like behavior on a periodic field is
either empty or wraps-around, neither of which maps to a BC-violation
sense.

**Disposition.** Task 4's F2 should scope to Dirichlet-type trace
checks (the rule's actual semantic), with the Neumann-type check
deferred to a separate V1.1 rule or implemented as a harness-level
check on `∂u/∂n` that is labeled as "external validation of the
Neumann trace-theorem under the F1 anchor, not rule-verdict." Periodic
case trivially passes (no boundary).

**Severity.** Not a CRITICAL stub (the rule works for Dirichlet), but a
plan-prose FLAG: §12 Step 4's "Dirichlet, Neumann, periodic" phrasing
needs scoping to Dirichlet-trace-only for the rule layer, with
Neumann-type semantics surfaced as harness-level if desired. PDEBench
pin (bRMSE, Diffusion-sorption + 2D diffusion-reaction,
`docs/audits/2026-04-22-pdebench-hansen-pins.md:99`) is Dirichlet-type
boundary error, so the F3 pin remains compatible.

### Task 8 (PH-CON-001) — "1e-14 cosine-IC mass conservation"

**Rule scope** (`src/physics_lint/rules/ph_con_001.py:1-21`): branches
on `BCSpec.conserves_mass` (periodic or Neumann-homogeneous). Periodic
arm computes `∫u dx` at each timestep via `integrate_over_domain`
(rectangle rule, endpoint-exclusive on periodic grids). Characteristic
mass = `max(|M(0)|, ||u(0)||_{L¹})` to handle zero-mean periodic ICs.

**Plan §16 acceptance criterion:** "Cosine-IC conservation to 1e-14."
Fixture: `u₀(x) = cos(2πx)` on periodic, `u(x,t) = cos(2πx) exp(-4π²t)`.

**Smoke-check needed.** Analytical mass `∫₀¹ cos(2πx) dx = 0`
exactly; rectangle-rule on `cos(2πx)` sampled at endpoint-exclusive N
points gives zero to roundoff (mode lands exactly on integer wavenumber
k=1, so DFT-of-sampling = Kronecker δ_{k,1}; integration of k=1 over
one period is zero). 1e-14 is plausible for float64 — but the mass
*drift* metric depends on the FD time-stepper's conservativity. If the
solver is `du/dt = κ Δu` with conservative FD spatial operator and
exact time integration on the mode, mass is preserved to roundoff. If
RK4 or similar with truncation error in time, drift accumulates as
`O(Δt⁴)` over the trajectory. At task execution, smoke-check: (a) how
the fixture's numerical solution is generated; (b) whether the "mass"
is computed at the analytical snapshot or at the numerical solution
snapshots; (c) the achievable absolute mass drift on the periodic DC
mode across the trajectory.

**Disposition.** Tolerance likely correct if fixture is "analytical
snapshot" (exact `cos(2πx) exp(-4π²t)` evaluated at discrete t_k), but
may need widening to `1e-10` or `1e-12` if fixture is "numerically
evolved from IC via FD time-stepper." Task 8 should clarify fixture
generation mode in its CITATION.md before committing to 1e-14.

### Task 9 (PH-CON-002) — "1e-8 energy over 1000 leapfrog steps"

**Rule scope** (`src/physics_lint/rules/ph_con_002.py:39-85`): computes
E(t) via integration-by-parts (IBP) for the potential term,
`0.5 c² ∫|∇u|² = −0.5 c² ∫u·Δu`, valid on conserves_energy BCs
(hD / hN / PER). Laplacian computed through the field's own backend
(spectral on periodic, FD4 otherwise) — not via `np.gradient`.

**Plan §17 acceptance criterion:** "E(t) bounded within 1e-8 over 1000
leapfrog steps." Fixture: `u = sin(kx) cos(ckt)` periodic.

**Smoke-check needed.** Leapfrog has bounded energy oscillation of
magnitude `O(Δt²)`. For 1D wave on `[0, 2π]` with `c=1`, `k=1`, and
CFL `0.5` (stable leapfrog at `Δt = 0.5 · h / c`), at N=64 grid
points, `h = 2π/64 ≈ 0.098`, `Δt ≈ 0.049`, `Δt² ≈ 2.4e-3`. Bounded
oscillation magnitude scales as `Δt² · E_max`, so for unit-amplitude
wave with `E_max ~ 1`, the bounded oscillation is `~2.4e-3`, not
`1e-8`. The 1e-8 threshold is likely tight by 4–5 orders of magnitude
for typical CFL and grid settings.

**Disposition.** Two interpretation paths:
- **Very small Δt interpretation** (reproduce Hairer-Lubich-Wanner
  bounded-oscillation at higher-than-typical temporal resolution):
  use `Δt = 2π/(1e5)` to achieve `Δt² ≈ 4e-10` bounded oscillation,
  making 1e-8 achievable. 1000 steps then cover `T ≈ 0.063`, a small
  fraction of one wave period — may not demonstrate long-time
  behavior.
- **Widened tolerance** (keep typical CFL, report actual leapfrog
  bounded-oscillation): widen to `5e-3` or `1e-2` consistent with
  Hairer-Lubich-Wanner Ch IX theoretical prediction. Matches physical
  expectation.

Recommendation for Task 9: widen to the realistic range + document the
`Δt²·E_max` scaling in CITATION.md. Task 9's F3 is already absent
(Task 0 pin `docs/audits/2026-04-22-pdebench-hansen-pins.md:218`), so
the tolerance choice doesn't interact with F3.

### Task 12 (PH-NUM-002) — "p_obs matches expected within 0.1" per-case expected

**Rule scope** (`src/physics_lint/rules/ph_num_002.py:1-20`): the
expected rate varies by backend + BC:
- Spectral + periodic: saturates at machine precision, so measured
  rate is effectively infinite.
- FD4 + periodic (interior-dominated): ~4 per doubling.
- FD4 + non-periodic (boundary-dominated): ~2–2.5 per doubling.
- Rule threshold: 1.8 per doubling.

**Plan §20 acceptance criterion:** "p_obs matches expected within
0.1."

**Smoke-check needed.** The "expected" rate has to be specified per
(PDE, backend, BC) triple, not a single number. For the three PDEs
(Laplace/Poisson/heat) × three backend+BC combos, there are up to 9
expected rates to list. Plan's single "p_obs matches expected within
0.1" is ambiguous.

**Disposition.** Task 12's CITATION.md should include a per-case
expected-rate table before the acceptance test runs. Tolerance of
`0.1` is tight — for FD4 periodic interior at typical N, empirically
measured slope is often 3.8–4.0 (see Tier-B Task 3 Layer 2 measured
3.89 on FD4 periodic, inside [3.6, 4.4]). 0.1 is consistent with Task
2's `4 ± 0.4` tolerance and Task 3's `4 ± 0.4` tolerance; but for a
per-case rate like 2.5, 0.1 may need widening to 0.3–0.4.
Recommendation: document per-case expected rate + per-case tolerance in
Task 12's CITATION.md; treat 0.1 as a starting point, widen at smoke-
check if measured slope consistency across three-level vs four-level
Richardson demands it.

---

## CLEARS: four tasks whose execution-level details align with rule reality

### Task 6 (PH-SYM-003) — LEE anchor, adapter-only, 2D origin-centered

- Rule is adapter-only + torch (`ph_sym_003.py:1-13`): "V1 scope:
  adapter mode only. Dump mode emits SKIPPED… Also requires a 2D grid
  centered at the origin."
- Plan §14 execution steps call live-model forward-AD via
  `torch.autograd.functional.jvp`, matching rule path.
- RotMNIST published range [0.705%, 2.28%]: Cohen-Welling P4CNN
  ICML 2016 reports 2.28% verifiable; Weiler-Cesa E(2)-CNN NeurIPS 2019
  reports 0.705 ± 0.025 (plan §14 Step 4(b) acceptance test). Verify
  Weiler-Cesa number at task execution via arXiv:1911.08251 Table 3 /
  paper text; no pre-flight concern.
- F3 two-layer policy (PR-CI cached checkpoint + on-demand Modal
  validation) is scope-consistent with rule's adapter mode + torch
  dependency. Modal A100 availability + RotMNIST dataset pin are
  task-execution-time dependencies, not plan-vs-rule mismatches.

### Task 10 (PH-CON-004) — interior-only hotspot on 2D triangulation

- Rule is `MeshField` + scikit-fem + interior-only (boundary-touching
  elements excluded structurally, `ph_con_004.py:27-50`). Hotspot
  indicator = `max_elem / mean_elem`. Threshold 10 → PASS / WARN.
- Plan §18 execution step 5 "hotspots within 2 element-layers of the
  L-corner" is consistent with the interior-only operator.
- F3 absent pre-recorded in Task 0 Step 5 F3-hunt
  (`docs/audits/2026-04-22-f3-hunt-results.md:23`).
- scikit-fem pin required (audit Step `_harness/TEXTBOOK_AVAILABILITY.md`
  version pin at task execution).

### Task 13 (PH-VAR-002) — diagnostic-only INFO-emit closeout

- Rule is diagnostic-only, PASS on wave with INFO severity, SKIPPED
  otherwise (`ph_var_002.py:1-18`). Plan §21 is v1.0 closeout —
  README + v1.2 changelog + PR #3 merge mechanics.
- No F2/F3 fixture layer required beyond the diagnostic-contract
  verification (rule emits PASS+INFO on wave, SKIPPED on Laplace/
  Poisson/heat).
- Demkowicz-Gopalakrishnan 2025 Acta Numerica publication status was
  verified in Task 0 (`docs/audits/2026-04-22-pdebench-hansen-pins.md:
  257`, Appendix). Cross-reference clean.

### Task 3 (retroactive) — already verified in commit `0cedc7b`

Logged here so the precheck's summary table stays complete.

---

## Dependencies summary

Optional dependencies required by Tasks 4–13, with expected pin
discipline at task execution:

| Task | Dep | Version pin discipline | Failure mode if missing |
|------|-----|------------------------|-------------------------|
| 6 | torch | already in `requirements.txt` | rule emits SKIPPED |
| 6 | Modal A100 | task-time env config | pre-release validation layer ungated |
| 6 | ImageNet-1k | user-supplied, env-var-gated | opt-in Gruver reproduction skipped |
| 6 | RotMNIST | Modal volume + pinned SHA | PR-CI regression gate ungated |
| 7 | neuraloperator or pytorch_fno | new pin at task 7 | harness-level FNO shift-commutation fixture ungated |
| 10 | scikit-fem | pin Ex 22 API compatibility | rule SKIPs + harness ungated |
| 11 | scikit-fem | pin `intorder` API | harness-level quadrature sweep ungated |

---

## Recommended actions for Tasks 4–13

**Before Task 4 begins:** no blocking action. Tasks 4, 6, 8 (with §16
FLAG), 9 (with §17 FLAG), 10, 12 (with §20 FLAG), 13 can execute per
plan with the flagged tolerance smoke-checks at each task's pre-
execution audit. Tasks 5, 7, 11 require harness-level Function 2
(disposition path noted per task above); this is scope-consistent with
the tasks' existing budgets and harness scaffolding allocations.

**For each task at execution time:** add a "Plan-diff guard" step to
the task's pre-execution enumerate-the-splits audit (§6.2): before
drafting acceptance-criteria details, re-read the rule source + the
audit row cited; confirm the plan's numerical/API detail matches the
rule's actual V1 emitted quantity. If mismatch, resolve via (a)
harness-level Function 2 (as Tasks 5, 7, 11 above) or (b) tolerance
widening with justification in CITATION.md (as Tasks 9, 12 above); log
both dispositions as plan-diffs in the commit Provenance per §7.4.

**After Tasks 5, 7, 11 land:** verify the "rule-verdict contract" layer
(the fourth layer added to each task's test file, asserting the V1
stub's documented behavior) is preserved in each CITATION.md's Function
2 subsection so the V1.1 rule upgrade path is visibly called out.

**Forward pattern.** The five Tier-B plan-diffs to date plus the three
CRITICAL + four FLAG items identified here make a total of
five-plus-seven-plus-ambiguous plan-execution surface areas. Path C
closes the discoverable surface before execution; the residual is
tolerance-tuning at task-specific smoke-check time, not
semantic-compatibility resolution.

---

## Cross-reference sanity

Each Tier-B task's cross-references to Task 0 audits resolved correctly
in this precheck:

| Task | Claim | Audit row | Verified |
|------|-------|-----------|----------|
| 4 | PDEBench pin exists | pdebench-hansen-pins.md:99 (Diffusion-sorption + 2D diffusion-reaction) | ✓ |
| 6 | Finzi-EMLP + Hall + Varadarajan verified status | TEXTBOOK_AVAILABILITY.md ⚠ | ✓ |
| 8 | Hansen Table 1 ANP row | pdebench-hansen-pins.md:166 | ✓ |
| 9 | F3-absent justification | pdebench-hansen-pins.md:218 | ✓ |
| 10 | F3-absent justification | f3-hunt-results.md:23 | ✓ |
| 11 | F3-absent justification | f3-hunt-results.md:78 | ✓ |
| 13 | Demkowicz 2025 Acta Numerica publication status | pdebench-hansen-pins.md:257 Appendix | ✓ |
| 3 | F3-absent justification (retroactive) | f3-hunt-results.md:123 | ✓ (verified in commit 0cedc7b) |

No broken cross-references across Tasks 4–13 remaining after Tasks 2–3
corrections landed. Task 2's diff #2 ("F3-absent not pre-recorded for
PH-RES-002") is the only cross-reference gap in the plan; that gap
remained in-task because PH-RES-002 is outside the Task 0 F3-hunt scope
(10/11/3) and PDEBench-Hansen scope (4/8/9).

---

## Summary table

| Task | Rule state | Precheck disposition | Harness-level F2? |
|------|-----------|----------------------|--------------------|
| 4 (PH-BC-001) | production | FLAG (Neumann semantic mismatch; scope to Dirichlet) | no |
| 5 (PH-BC-002) | Laplace-only | **CRITICAL** (arbitrary-F fixture doesn't match rule) | **yes** |
| 6 (PH-SYM-003) | adapter-only | CLEAR | no |
| 7 (PH-SYM-004) | V1 stub (SKIPPED) | **CRITICAL** (rule doesn't emit F2 quantity) | **yes** |
| 8 (PH-CON-001) | production | FLAG (1e-14 fixture-mode-dependent) | no |
| 9 (PH-CON-002) | production | FLAG (1e-8 tight by ~5 OoM vs Δt²) | no |
| 10 (PH-CON-004) | production | CLEAR | no |
| 11 (PH-NUM-001) | V1 stub (PASS+reason) | **CRITICAL** (rule doesn't emit F2 quantity) | **yes** |
| 12 (PH-NUM-002) | production | FLAG (per-case expected rate, not single) | no |
| 13 (PH-VAR-002) | diagnostic-only | CLEAR | no |

**Totals:** 3 CRITICAL (harness-level F2 required) + 4 FLAG (tolerance/
scope smoke-check at task execution) + 4 CLEAR. Estimated Tier-B
execution overhead to absorb: +0.3 d across Tasks 5/7/11 for harness-
level F2 (within their respective existing budgets — scope-consistent
with existing harness scaffolding allocations); +0 d for FLAG items
(resolved inside each task's existing enumerate-the-splits audit
allocation per §6.2); 0 d for CLEAR items.
