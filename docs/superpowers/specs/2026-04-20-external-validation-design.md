# External-validation implementation plan — design spec

**Status.** Brainstorming complete; design ready for implementation planning (Tier A scope).
**Date.** 2026-04-20.
**Predecessors.**
- Week 4 plan at `docs/plans/2026-05-05-physics-lint-v1-week-4.md` (in flight; Task 4 hero pivot landed at `7fd46d5`; Tasks 5–7 and Task 8 release gate still ahead).
- Cleaned-draft external-validation spec (2026-04-18) handed into brainstorming as input; six review passes (Rev 1.0 → Rev 1.6) consolidated into the body below.
- Week 2½ dogfood design at `docs/superpowers/specs/2026-04-17-week-2.5-dogfood-a1-design.md` (precedent for the superpowers brainstorming → writing-plans flow).
**Successor.** Tier-A-only execution plan via `superpowers:writing-plans` at `docs/plans/2026-04-20-physics-lint-external-validation-tier-a.md` (standalone dated doc; **not** inserted into the Week 4 plan — see preamble item 2). The Rev 1.2+ read-back gate is inherited as a pre-ship checklist for the writing-plans output.

---

## Brainstorming-outcomes preamble (2026-04-20)

Four decisions made during the 2026-04-20 brainstorming session that qualify or supersede the Rev 1.6 body below. All other Rev 1.6 content is preserved verbatim to keep the six-pass revision arc immutable.

1. **Writing-plans scope: Tier A only.** Per Rev 1.6 §9 item 4 ("defer Tier-B plan commitment until Tier A is green and visa-deadline picture is clearer … re-estimate from empirical Tier-A velocity, not from this plan's a priori estimates"). The Tier-B roadmap in §4 remains documented here as design context and will receive backlog entries in `docs/backlog/v1.1.md` per §7. Tier-B execution-level decomposition is deferred to post-Tier-A planning.

2. **Tier-A plan location: standalone doc, not Week-4-Task-9 insertion.** Rev 1.6 §9 item 2 prescribes "Tier A is inserted as Week 4 'Task 9' — between Tasks 7 and 8" in `docs/plans/2026-05-05-physics-lint-v1-week-4.md`. This is **superseded**: the Tier-A execution plan will live at its own dated doc (`docs/plans/2026-04-20-physics-lint-external-validation-tier-a.md` or the dated variant the writing-plans agent-of-record selects), cross-referencing Week 4 Task 8's release gate as a precondition rather than nesting inside it. Rationale: the Week 4 plan is ~2400 lines and structurally closed with Tasks 0–8, an End-of-Week verification block, and a §10 provenance section; appending a 4.5-day Task 9 would bloat it and muddy the "Week 4 = CLI + SARIF + release" framing. Cross-referencing keeps the Week-4-Task-8 ↔ Tier-A-green dependency explicit without nesting.

3. **Read-back gate inheritance.** The Rev 1.2+ quality checklist (arithmetic-sum audit, inline-numeric verification, stale-string scans, task-number cross-reference, methodology-primitive precondition checks, internal-metric self-consistency) that produced this Rev 1.6 carries forward to the Tier-A writing-plans output as an explicit pre-ship pass. The writing-plans prompt will call this out verbatim so the constraint survives the context boundary.

4. **No in-body edits to Rev 1.6.** The Week-4-Task-9 override in item 2 above is captured here in the preamble only; the Rev 1.6 body below is unchanged. This preserves the audit trail of the six-pass revision arc — any future reader encounters Rev 1.6 exactly as it was reviewed, with the delta decisions separated into this preamble rather than silently edited into the body.

**Pre-kickoff readiness status (2026-04-20 snapshot).**

- `external_validation/` directory does not yet exist in the repo (clean slate for Task 0).
- `scikit-fem 12.0.1` installed in `.venv` (satisfies Rev 1.6 §9 item 3(b); versions ≥ 10 required, installed version exceeds).
- CI workflows present: `.github/workflows/ci.yml`, `.github/workflows/physics-lint.yml`. New file `external-validation.yml` will be added in Task 0 per Rev 1.6 §1.4.
- Week 4 in flight: commit `7fd46d5` ("pivot v1.0 hero to PH-POS-002") is the latest on `feature/v1-week-4`. Week 4 Tasks 5–7 (gallery, docs, README) are ahead of Tier-A start; Week 4 Task 8 (release) gates on Tier-A completion per preamble item 2.
- Local textbook copies (Evans, Ciarlet, Trefethen) status: not verified in this brainstorming session. Rev 1.6 §9 item 3(a) offers "section-level citation" as the fallback if local copies are unavailable. Deferred to Tier-A Task 0 close-out — the Tier-A writing-plans output will include an explicit textbook-verification-vs-section-level-fallback decision step.

---

**Revision 1.6** — three-point housekeeping pass on Rev. 1.5 addressing final external review (ChatGPT, 2026-04-18, pass 6). Changes: (1) Task 1 acceptance criteria rewritten — the task used an undefined `ratio` metric ("ratio < 10" for pass, "ratio > 100" for fail) that was never defined in the task's test design. Replaced with the clean verdict-based criterion the test design already used: `max_interior(u_pred) ≤ max_boundary(u_pred) + 10⁻³ · range(u)` (and symmetrically for the minimum). This matches the test-design language and removes a dangling internal metric. (2) Task 14 fixture-mismatch policy specified — the serialized baseline now has an explicit policy: `mesh_hash` mismatch fails hard (mesh determines the numerical result; silent pass-through would be meaningless), `scikit_fem_version` minor-patch mismatch warns and continues (minor patches rarely change numerical output), major-version mismatch fails with recalibration prompt, missing fixture fails with calibration-script link. The principle is "fixture is authoritative; mismatches are events that need explicit acknowledgment, never silent pass-through." (3) Provenance-date housekeeping — earlier same-day revision timestamps used awkward modifiers like "evening" / "night" / "later" and a Rev. 1.4/1.5 date of "2026-04-19" that implied calendar travel within a single conversation session. All revisions happened on 2026-04-18. Collapsed same-day timestamps to `2026-04-18 (pass N)` scheme throughout §10 and revision-header blocks so ordering is unambiguous without misleading dates. Read-back pass run against all arithmetic, inline math, task-number cross-references, dependency-graph consistency, methodology-primitive pre-conditions, and now also **internal-metric self-consistency** (every metric named in a task's acceptance criteria must be defined in that task's test design) before ship. No scope or rule-coverage changes.

**Revision 1.5** — two-point correction pass on Rev. 1.4 addressing final external review (ChatGPT, 2026-04-18, pass 5). Changes: (1) Task 0 `Citation` dataclass validation broadened from `{arxiv_id, doi}` to `{arxiv_id, doi, isbn, url}` — "at least one must be present." The previous rule would have rejected every book anchor the plan uses (Evans, Protter–Weinberger, Gilbarg–Trudinger, Ciarlet, Trefethen, Bangerth–Rannacher, Verfürth); ISBN is the operational identifier for these, and `url` covers cases like the Salari–Knupp OSTI page or scikit-fem documentation where neither DOI nor ISBN is practical. (2) Task 4 Layer 2 quadrature changed from "Simpson composite on the 64×64 grid" (which is a parity landmine — composite Simpson requires even subinterval count per axis, and 64 grid points gives 63 odd subintervals) to "composite trapezoidal on the 64×64 grid." Trapezoidal has no parity constraint, is correct, and costs nothing in accuracy for this Layer-2 ratio-bounded sanity check. Read-back pass run against all arithmetic, inline math, task-number cross-references, dependency-graph consistency, and methodology-primitive pre-conditions (domain compatibility, grid parity, boundary handling) before ship. No scope or rule-coverage changes.

**Revision 1.4** — three-point correction pass on Rev. 1.3 addressing final external review (ChatGPT, 2026-04-18, pass 4). Changes: (1) Task 4 Layer 2 methodology corrected — the `‖u_pert − u‖_{H¹}` ground-truth computation was described as "via spectral derivatives on the known analytical form," which is methodologically wrong on a non-periodic homogeneous-Dirichlet unit square (spectral differentiation requires periodicity). The perturbation fields are explicit closed-form functions; their gradients can be computed exactly by hand or via `sympy`, and the H¹ norm is a standard quadrature on the square. No spectral tooling needed; (2) Task 14 baseline calibration serialized — `hotspot_fraction_baseline` now written to a machine-readable fixture (`external_validation/PH-CON-004/fixtures/baseline_calibration.json`), not only documented narratively in CITATION.md. Regression gate loads the fixture deterministically on every run; (3) Task 3 synthetic equivariant-operator construction tightened — the radial FFT-based Laplace inverse now explicitly fixes the zero Fourier mode to zero by convention, making the "provably C₄-equivariant" claim operationally complete (the Laplacian's kernel on a torus is the constant mode, so `(-Δ)⁻¹` is undefined there without a stated convention). Read-back pass run against all arithmetic, inline math, task-number cross-references, dependency-graph consistency, and methodology-primitive usage (spectral vs. analytical vs. quadrature) before ship. No scope or rule-coverage changes.

**Revision 1.3** — three-point correction pass on Rev. 1.2 addressing external review (ChatGPT, 2026-04-18, pass 3). Changes: (1) Task 3 harness-reuse citation corrected from "Tier-B tasks 9, 10" to "Tier-B tasks 10, 11" — Task 9 is `PH-BC-002` (divergence theorem), the actual downstream symmetry tasks that inherit the harness are Task 10 (`PH-SYM-003`) and Task 11 (`PH-SYM-004`); (2) Task 0 `Citation` dataclass validation rule softened from `arxiv_id XOR doi present` to `at least one of arxiv_id or doi must be present` — the plan intentionally cites several anchors with both (e.g., Hansen has Physica D DOI + arXiv; Bachmayr–Dahmen–Oster has IMA JNA DOI + arXiv), and XOR would reject the spec's own recommended citations; (3) Task 14 positive-control framing rewritten from absolute "hotspot fraction ≈ 0" (false for a smooth non-uniform source — the residual indicator naturally has nontrivial concentration above 0.5·max for broadly-peaked smooth fields) to comparative "separation = hotspot_fraction(perturbed) − hotspot_fraction(baseline) > 0.05"; the baseline establishes a reference concentration profile, the test asserts the perturbation produces significantly larger concentration. A v1.1 refinement note added about Path B (kurtosis-based localization metric) in the risks section. Read-back pass run against all arithmetic, inline math, task-number cross-references, and dependency-graph consistency before ship. No scope or rule-coverage changes.

**Revision 1.2** — five-point correction pass on Rev. 1.1 addressing external review (ChatGPT, 2026-04-18, pass 2). Changes: (1) Task 9 divergence formula corrected from the nonsensical `∇·F = 2xy + x² + y²` to the correct `∇·F = 4xy` for `F = (x²y, xy²)`; the integral value is unchanged; (2) Task 2 undefined `ε_num` symbol removed — the monotonicity-plus-`ε_quad` bound already in step 4 is sufficient; (3) Task 4 Layer-2 framing made precise — it is a "norm-equivalence sanity check informed by Bachmayr–Dahmen–Oster," not an external validation of variational correctness on arbitrary surrogate outputs (true external anchor deferred to v1.1 via Ernst–Rekatsinas–Urban); (4) Task 14 positive-control wording fixed — `u = x(1-x)y(1-y)` has `f = -Δu = 2[y(1-y) + x(1-x)]`, which is *not* uniform; the positive control demonstrates "no hotspots above discretization-and-projection noise floor," not "uniform residual"; (5) Task 12 Hansen citation updated to Physica D 2024 as primary (peer-reviewed journal), ICML 2023 as conference precedent, arXiv:2302.11002 for discoverability. Plus one minor touch-up: §6 Tier-A wall-clock corrected from "~4 d" to "~4.5 d" since Task 4 at 2 d cannot be split across parallel lanes. Read-back pass run against all arithmetic and inline math before ship.

**Revision 1.1** — budget reconciliation and six-point revision pass on Rev 1.0. Changes: (1) budget numbers reconciled bottom-up (Tier A ~6 d incl. Task 4 H⁻¹ extension, Tier B ~17 d, combined ~23 d); (2) Task 4 extended to cover norm-equivalence sanity check, not just discretization convergence; (3) Task 3 renamed from "paper reproduction" to "synthetic control + literature calibration"; (4) Task 10 LEE target reframed as three-layer anchor (paper metric + reference code + local regression target); (5) CI trigger/branch policy unified to "Tier A on PRs to main, Tier B on push/release only"; (6) Task 5 fixture simplified from sine-series to polynomial Poisson. No scope changes; this revision tightens the plan against internal inconsistencies surfaced in external review.

**Scope boundary.** This plan covers the 18 benchmark-anchorable rules only. The 3 meta-rules (`PH-VAR-001`, `PH-NUM-003`, `PH-NUM-004`) are validated by unit tests and fixture checks, documented in the cleaned-draft spec, and are not part of this plan.

**Release gating.** v1.0 ships the Tier-A subset (Tasks 1–5, ~6 engineer-days total including shared infrastructure, 6 rules). Tier-B (Tasks 6–17, ~17 engineer-days, 12 rules) is v1.1 roadmap, executed post-visa-deadline on a relaxed timeline.

---

## 0. Executive summary

**Goal.** Produce an `external_validation/` directory in the physics-lint repo containing one validated anchor per benchmark-anchorable rule, wired to a CI workflow, cited from the README's "External Validation" section, and documented to the standard of the existing rule documentation pages.

**Structure.** Each rule gets a task with (a) the anchor type (theory reproduction / paper reproduction / self-built reference), (b) the reference citation pinned to the exact section/figure/table used, (c) the test harness location, (d) the pinned expected value with tolerance, (e) acceptance criteria, (f) effort estimate, (g) risks.

**Schedule split.**
- **Tier A (v1.0, ~6 engineer-days).** Six rules across all five rule families. Ships with v1.0 release. Includes Task 0 shared infrastructure (~1 day, front-loaded) and Task 4's Bachmayr–Dahmen–Oster H⁻¹ sanity-check extension (~0.5 day).
- **Tier B (v1.1, ~17 engineer-days).** Remaining twelve benchmark-anchorable rules. Deferred to v1.1 per the visa-deadline-driven Option-(c) tradeoff.

**Total budget.** Tier A + Tier B = **~23 engineer-days combined**, reconciled bottom-up from §3 and §5 per-task estimates. This is ~5 days higher than the cleaned-draft spec's 14–18-day top-down estimate; the delta is attributable to per-task inclusions of negative controls, citation artifacts, harness reuse accounting, and the Task 4 H⁻¹ extension that were not separately budgeted in the spec's high-level estimates. The bottom-up number is authoritative for scheduling; the spec's range is retained as context for the scope decision, not for execution planning.

**Deliverables per rule.**
1. Test harness under `external_validation/<rule_id>/` (script + fixtures).
2. CI job in `.github/workflows/external-validation.yml` (skip-on-missing-dependency for opt-in anchors, run-by-default for the rest).
3. README entry in the "External Validation" section linking the harness, citation, and pinned expected value.
4. v1.1 backlog entry for any deferred expansion (e.g., tighter reproduction, broader paper anchor).

---

## 1. Shared infrastructure (Task 0)

Complete before any per-rule task starts. Estimated effort: **~1 day** (front-loaded cost, amortized across all subsequent tasks).

### 1.1 Directory layout

```
external_validation/
├── README.md                   # Section-by-section anchor index
├── _harness/
│   ├── __init__.py
│   ├── assertions.py           # assert_within, assert_slope_in_range, assert_ranking_matches
│   ├── citations.py            # Citation dataclass: paper, section, table/figure, pinned value
│   └── fixtures.py             # Shared analytical solutions: harmonic polys, MMS sin*sin, etc.
├── <rule_id>/
│   ├── test_anchor.py          # pytest-collectable
│   ├── fixtures/               # rule-specific data, checkpoints, reference outputs
│   ├── CITATION.md             # single-page citation with pinned value and verification protocol
│   └── README.md               # per-rule anchor documentation
```

### 1.2 Shared assertion primitives

The per-rule test harnesses all reduce to one of four assertion patterns:

| Pattern | Use case | Primitive |
|---|---|---|
| Value-within-tolerance | Reproduce a single pinned scalar | `assert_within(measured, expected, rel_tol, abs_tol)` |
| Slope-in-range | Reproduce a convergence rate | `assert_slope_in_range(hs, errs, expected_slope, tolerance)` |
| Ranking-matches | Reproduce ordinal agreement | `assert_ranking_matches(measured_ranking, expected_ranking)` |
| Pass-analytical | Rule passes on known-correct input | `assert_rule_passes(rule_fn, analytical_input)` |

Implemented once in `_harness/assertions.py`; reused across all 18 tasks.

### 1.3 Citation dataclass

```python
@dataclass
class Citation:
    paper_title: str
    authors: str
    venue: str               # e.g., "ICML 2023", "Math. Comp. 51(184)", "AMS, 2nd ed. 2010"
    arxiv_id: str | None     # e.g., "2210.02984"
    doi: str | None          # e.g., "10.1016/j.physd.2023.133952"
    isbn: str | None         # for books, e.g., "978-0-8218-4974-3" (Evans 2nd ed.)
    url: str | None          # for sources best identified by URL, e.g., OSTI report, docs page
    section: str             # e.g., "§2.2.3 Theorem 4"
    artifact: str            # e.g., "Table 3", "Figure 1", "Eq. (4)"
    pinned_value: str        # e.g., "LEE ≈ 0.10 ± 0.02"
    verification_date: str   # ISO date when value was pinned from the source
    verification_protocol: str  # How to re-verify (figure read, code execution, analytical derivation)
```

### 1.4 CI workflow structure

`.github/workflows/external-validation.yml` runs under a split policy: **Tier-A jobs run on every PR to `main`** (the repo's default branch), **Tier-B jobs run on push to `main` and on release-tag events only**. Matrix strategy: one job per rule. Per-job `continue-on-error` defaults to `false`. Opt-in anchors (requiring user-side datasets or third-party checkpoints) use an env-var-gate pattern: job checks for the required env var and skips cleanly if absent. The split policy is rationalized in §8 on CI runtime risk.

### 1.5 Acceptance criteria for Task 0

- `external_validation/_harness/` implements the four assertion primitives with 100% test coverage.
- `Citation` dataclass validates on construction (required fields non-empty; **at least one stable identifier** among `{arxiv_id, doi, isbn, url}` must be present — books use `isbn` (Evans, Ciarlet, Trefethen, Protter–Weinberger, Gilbarg–Trudinger, Bangerth–Rannacher, Verfürth); journal articles typically have `doi` plus optional `arxiv_id` preprint; preprint-only works have `arxiv_id` alone; sources best identified by URL (e.g., Salari–Knupp OSTI page at osti.gov/biblio/759450, scikit-fem documentation) use `url`; having multiple identifiers is valid and expected, e.g., Hansen has Physica D DOI + arXiv, Bachmayr–Dahmen–Oster has IMA JNA DOI + arXiv; `verification_date` is valid ISO 8601).
- CI workflow skeleton present; one no-op job per rule slot wired up; no anchor-specific logic yet.
- `external_validation/README.md` contains the table-of-18 scaffold with placeholder links.

---

## 2. Tier A — v1.0 ship scope (~6 engineer-days)

Six rules covering all five rule families (residual, BC-adjacent via positivity, conservation, positivity, symmetry). Selected for highest credibility-per-day. All Tier-A anchors are CI-runnable — no ImageNet, no external-checkpoint downloads at test time, no GPU requirements.

### Task 1 — `PH-POS-002`: weak maximum principle on harmonic polynomials

**Anchor type.** Classical theory reproduction.

**Reference.** Evans *Partial Differential Equations* (AMS, 2nd ed. 2010), §2.2.3 Theorem 4 (weak maximum principle for Laplace). Protter–Weinberger *Maximum Principles in Differential Equations* (Dover 1999) Chapter 2 §1 Theorem 1 as cross-reference. *Section numbers to be verified against local copy before this task is closed.*

**Test design.**
1. Fixture set: three harmonic polynomials on the unit square `[0,1]²`:
   - `u(x,y) = x² - y²`
   - `u(x,y) = xy`
   - `u(x,y) = x³ - 3xy²` (real part of `(x+iy)³`)
2. For each fixture, compute boundary values, interior values on a 64×64 grid.
3. Run `PH-POS-002` rule against each: expected verdict PASS. Verdict logic uses the max-principle inequalities `max_interior(u_pred) ≤ max_boundary(u_pred) + 10⁻³ · range(u)` and `min_interior(u_pred) ≥ min_boundary(u_pred) − 10⁻³ · range(u)`, where `max_interior` / `min_interior` are computed over grid nodes with `1 ≤ i,j ≤ N−2` (excluding the boundary ring), `max_boundary` / `min_boundary` over the boundary ring (`i ∈ {0, N−1}` or `j ∈ {0, N−1}`), and `range(u) = max(u) − min(u)` over the full grid.
4. Negative control: pass a known-non-harmonic polynomial (e.g., `u = x² + y²`) through the same harness; expected verdict FAIL.

**Pinned expected behavior.** PASS on all three harmonic polynomials; FAIL on the non-harmonic control.

**Acceptance criteria.**
- All three harmonic polynomials produce verdict PASS from `PH-POS-002` at 64×64 — i.e., for each fixture, `max_interior(u_pred) ≤ max_boundary(u_pred) + 10⁻³ · range(u)` and symmetrically for the minimum. Tolerance `10⁻³ · range(u)` is the max-principle tolerance named in the task's test design, well above float32 noise on these smooth polynomials.
- Non-harmonic control (`u = x² + y²`) produces verdict FAIL — interior maximum strictly exceeds boundary maximum beyond the tolerance.
- Test runs in < 5 s on CPU.
- `CITATION.md` documents Evans section reference and links to the test.

**Effort estimate.** ~0.5 day.

**Risks.**
- Floating-point noise at the machine-epsilon boundary may cause spurious near-miss failures on tight tolerances. Mitigation: tolerance is `10⁻³ · range(u)`, well above float32 noise on these smooth polynomials.
- Evans theorem numbering: verify against local copy before closing the task.

---

### Task 2 — `PH-CON-003`: heat energy-dissipation sign

**Anchor type.** Classical theory reproduction.

**Reference.** Evans §7.1.2 Theorem 2 (energy estimates for parabolic equations). *Section numbers to be verified.*

**Test design.**
1. Fixture: analytical heat-equation solution on `[0,1]²`, homogeneous Dirichlet BC, eigenfunction IC
   `u(x,y,0) = sin(πx)sin(πy)`, `u(x,y,t) = exp(-2π²t) sin(πx)sin(πy)`.
2. Timesteps `t ∈ {0, 0.05, 0.1, 0.15, 0.2}`. Grid 64×64.
3. Compute `E(t_k) = ½ ∫_Ω u²(x,y,t_k) dx` via trapezoidal quadrature.
4. Run `PH-CON-003` rule: assert `E(t_{k+1}) ≤ E(t_k) · (1 + ε_quad)` for `ε_quad = 10⁻⁴`.
5. Negative control: feed a constructed non-dissipative sequence (e.g., `u_fake(t) = u(0) · exp(+0.1 · t)`); assert FAIL.

**Pinned expected behavior.** PASS on the analytical heat solution; FAIL on the non-dissipative control.

**Acceptance criteria.**
- Analytical heat solution: per-step ratio `E(t_{k+1})/E(t_k)` equals the exact analytical value `exp(-4π²·0.05) ≈ 0.1389` to within `ε_quad = 10⁻⁴` (quadrature noise budget); monotonic decrease required at every step.
- Non-dissipative control fails at the first timestep.
- `CITATION.md` documents Evans §7.1.2 reference.
- Test runs in < 10 s on CPU.

**Effort estimate.** ~0.5 day.

**Risks.**
- Trapezoidal quadrature error on a coarse grid could violate the strict `ΔE ≤ 0` inequality by `O(h²)`; `ε_quad = 10⁻⁴` absorbs this on a 64×64 grid.
- Time-discretization noise on the analytical solution is zero (analytical is exact); no concern.

---

### Task 3 — `PH-SYM-001` + `PH-SYM-002`: discrete rotation and reflection equivariance

**Anchor type.** Synthetic control + literature calibration. *Not* a direct reproduction of Helwig's tables — Helwig evaluates on Navier–Stokes and NS-SYM, which are outside v1.0's Laplace/Poisson/heat/wave PDE scope. Helwig serves as the external calibration for the metric family (order-of-magnitude separation between equivariant and non-equivariant operators under 90° rotation); physics-lint's tests verify that the rule correctly distinguishes provably-equivariant from provably-non-equivariant synthetic operators on the v1.0 PDE scope.

**Reference.** Helwig, Zhang, Fu, Kurtin, Wojtowytsch, Ji, "Group Equivariant Fourier Neural Operators for Partial Differential Equations," *ICML 2023*, PMLR 202:12907–12930, arXiv:2306.05697. Specifically:
- Table 3 (2D NS rotation test): plain FNO relative MSE **8.41 ± 0.41** (unrotated) → **129.21 ± 3.90** (90°-rotated). G-FNO-p4 test MSE ≈ 4.78 ± 0.39 (equivariant-by-construction).
- Table 1: G-FNO-p4m relative MSE **2.37 ± 0.19** on the symmetric test set.

**Test design.**

*Shared symmetry harness* (built in Task 3; reused by Tier-B tasks 10 and 11 — `PH-SYM-003` and `PH-SYM-004`; Task 9 is `PH-BC-002` and does not use this harness):
- `harness.rotate_test(model, x, k)`: applies `torch.rot90(x, k, dims=(-2,-1))`, forward-passes, applies `torch.rot90(·, -k, ...)`, returns relative L² error.
- `harness.reflect_test(model, x, axis)`: same pattern with `torch.flip`.

*Rule-specific tests:*
1. **PH-SYM-001 positive control.** Construct a provably C₄-equivariant synthetic operator: the radial FFT-based Laplace inverse on a square power-of-2 grid, with **the zero Fourier mode fixed to zero by convention** (the Laplacian's kernel on a torus is the constant mode, so `(-Δ)⁻¹` is undefined there; setting the zero mode to zero makes the operator fully defined and the equivariance claim airtight). Run rotate_test for `k ∈ {1, 2, 3}`. Expected relative error ≤ 10⁻⁵ (floating-point noise only).
2. **PH-SYM-001 negative control.** Construct a provably non-equivariant operator (CNN with learned positional embeddings, untrained so weights are random). Expected relative error > 0.1 (random-output magnitude).
3. **PH-SYM-002 positive control.** Same equivariant Laplace inverse; run reflect_test for `axis ∈ {-2, -1}`. Expected ≤ 10⁻⁵.
4. **PH-SYM-002 negative control.** Same non-equivariant CNN; expected > 0.1.
5. *Citation anchor (documentation only).* README section references the Helwig Table 3 numbers as the external calibration — "plain FNO degrades by a factor of 15× under 90° rotation; physics-lint's PH-SYM-001 catches this degradation on equivalent models."

**Pinned expected behavior.**
- Equivariant positive controls: `e_C4 ≤ 10⁻⁵`, `e_reflect ≤ 10⁻⁵`.
- Non-equivariant negative controls: `e_C4 > 0.1`, `e_reflect > 0.1`.
- Helwig citation is the *external-calibration anchor*, not something physics-lint reproduces directly (no FNO checkpoint run); documented as "Helwig et al. report X on FNO; physics-lint verifies the rule correctly distinguishes equivariant from non-equivariant ops with this magnitude separation."

**Acceptance criteria.**
- Equivariant and non-equivariant controls produce the expected relative-error ranges on a 64×64 grid.
- Symmetry harness is reusable (Tier-B tasks for PH-SYM-003, PH-SYM-004 inherit it).
- `CITATION.md` pins the Helwig Table 3 and Table 1 values with a 2026-04-18 verification date.
- Test runs in < 15 s on CPU (no FNO inference — synthetic operators only).

**Effort estimate.** ~1 day combined, including harness construction. (Harness cost is absorbed here; Tier-B tasks 10 and 11 inherit it at near-zero marginal cost.)

**Risks.**
- Helwig's numbers are on 2D Navier–Stokes, not Laplace/Poisson/heat/wave. Rule calibration transfers (equivariance is architecturally identical across PDEs), but absolute magnitudes do not — document this in CITATION.md.
- Synthetic equivariant operator must be *provably* equivariant, not "equivariant-by-construction as far as we tested." Use radial FFT with `torch.fft.fftn` / `torch.fft.ifftn`: the `k_x² + k_y²` eigenvalue structure is rotation-invariant on a square grid under 90° rotations. **Zero-mode convention:** the Laplacian's kernel is the constant mode, so set `û(k=0) = 0` on the inverse; this makes the operator fully defined and the equivariance claim operationally complete.

---

### Task 4 — `PH-RES-001`: MMS convergence + H⁻¹ norm-equivalence sanity check on sin(πx)sin(πy)

**Anchor type.** Two-layer structure. (i) *Discretization-convergence layer (external theory anchor):* reproduces the Fornberg 1988 O(h⁴) rate on a smooth MMS solution — this is the genuine external theory reproduction. (ii) *Norm-equivalence sanity layer (internal sanity check informed by Bachmayr–Dahmen–Oster):* verifies that the ratio `‖r‖_{H⁻¹} / ‖u − u_exact‖_{H¹}` stays bounded across a smooth-perturbation family of a known analytical solution, consistent with the stable-variational-formulation framework. Layer 2 is *not* an external validation of variational correctness on arbitrary trained-surrogate outputs — its `[c_min, C_max]` bounds are calibrated from a scikit-fem FE reference on the same grid, making it a principled internal consistency check rather than a cross-source reproduction. The true external paper-reproduction anchor for variational correctness on arbitrary surrogates is Ernst–Rekatsinas–Urban 2025 (arXiv:2502.20336v3) and is deferred to v1.1 (see Task 4's Layer-2 scope-boundary risk and the v1.1 roadmap).

**Reference.**
- **Convergence-rate anchor:** Fornberg 1988, "Generation of finite difference formulas on arbitrarily spaced grids," *Math. Comp.* 51(184):699–706, DOI 10.1090/S0025-5718-1988-0935077-0. 4th-order central FD stencil on smooth solutions: O(h⁴) residual.
- **Variational-correctness anchor:** Bachmayr, Dahmen, Oster, "Variationally correct neural residual regression for parametric PDEs: On the viability of controlled accuracy," *IMA Journal of Numerical Analysis* 2025, DOI 10.1093/imanum/draf073 (arXiv:2405.20065). Framework establishing that a residual loss is variationally correct iff its value is uniformly proportional to the squared solution error in the norm of a stable variational formulation (Theorem 2.10, Babuška–Nečas).
- **MMS methodology anchor:** Salari & Knupp, "Code Verification by the Method of Manufactured Solutions," SAND2000-1444 (Sandia 2000), DOI 10.2172/759450.

**Test design.**

*Layer 1 — discretization convergence (sub-anchor).*
1. Fixture: MMS Poisson on unit square, `u(x,y) = sin(πx)sin(πy)`, `f(x,y) = 2π² sin(πx)sin(πy)`, homogeneous Dirichlet.
2. Grid resolutions `N ∈ {16, 32, 64, 128}` corresponding to `h ∈ {1/15, 1/31, 1/63, 1/127}`.
3. Evaluate `u` on each grid exactly; compute `PH-RES-001` raw residual at each resolution.
4. Fit log-log regression of residual vs `h` across the four points; assert slope `4.0 ± 0.2`.
5. Report regression `R² ≥ 0.99` (guards against non-asymptotic-regime noise).

*Layer 2 — norm-equivalence sanity (informed by the variational-correctness framework; internal consistency check).*
6. Perturbation family: construct three perturbed predictions of the MMS solution on the N=64 grid:
   - Low-frequency perturbation: `u_pert_1 = u + 0.01 · sin(πx)sin(πy)` (smooth, small H¹ error).
   - Mid-frequency perturbation: `u_pert_2 = u + 0.01 · sin(4πx)sin(4πy)` (oscillatory, larger H¹ error at same L² magnitude).
   - Boundary-respecting noise: `u_pert_3 = u + 0.01 · sin(πx)sin(πy) · η(x,y)` where η is smooth and vanishes on ∂Ω.
7. For each perturbed prediction, compute (a) `‖r‖_{H⁻¹}` via `PH-RES-001`, (b) true `‖u_pert − u‖_{H¹}` via **exact analytical derivatives** of the known perturbation formulas (computed by hand or via `sympy`; all three perturbations are explicit closed-form functions), followed by **composite trapezoidal quadrature** on the 64×64 grid. Trapezoidal is chosen over Simpson deliberately: composite Simpson requires an even number of subintervals per axis, and a 64-point grid gives 63 subintervals (odd), which would require either padding the grid or using a mixed Simpson/3-8 rule. Trapezoidal has no parity constraint, is O(h²) on a smooth integrand, and accuracy is ample for this Layer-2 ratio-bounded sanity check. No spectral differentiation — the domain is not periodic, and the perturbations are closed-form, so analytical differentiation plus trapezoidal quadrature is both exact in the derivative and simple in the quadrature.
8. Compute the ratio `ρ_k = ‖r_k‖_{H⁻¹} / ‖u_pert_k − u‖_{H¹}` for each k.
9. Assert `ρ_k ∈ [c_min, C_max]` for all three perturbations, where `c_min, C_max` are the norm-equivalence constants expected from the variational formulation (for Poisson on a unit square with homogeneous Dirichlet, both constants are O(1); the precise bounds are discretization-dependent and should be calibrated from a known-good FE reference on the same grid before the test is declared passing).

**Pinned expected behavior.**
- *Layer 1:* Log-log slope = 4.0 ± 0.2 over four refinement levels. Residual at `N=128` below 10⁻⁶.
- *Layer 2:* Ratio `ρ_k` bounded by `[c_min, C_max]` across the three perturbations, with both bounds O(1) and `C_max / c_min < 10` (indicating the rule's residual is a proxy for solution error within an order of magnitude across the perturbation family — not an exact equality, but the "uniformly proportional" property that variational correctness requires).

**Acceptance criteria.**
- *Layer 1:* Measured slope within [3.8, 4.2]. R² ≥ 0.99 across four points. Residual monotonically decreasing. At `N=128`, raw residual ≤ 10⁻⁶.
- *Layer 2:* All three `ρ_k` values fall inside the calibrated `[c_min, C_max]` range. Ratio `C_max / c_min < 10`.
- Test runs in < 45 s on CPU (Layer 1 + Layer 2 combined).
- `CITATION.md` documents both Fornberg and Bachmayr–Dahmen–Oster with verification dates and pins the calibrated `[c_min, C_max]` range.

**Effort estimate.** ~2 days.
- 0.25 d: MMS fixture implementation (shared with Layer 2).
- 0.5 d: refinement loop + log-log regression + assertion (Layer 1).
- 0.5 d: perturbation family construction, analytical H¹ error computation, ratio calibration on a known-good FE reference (Layer 2).
- 0.25 d: negative control for Layer 1 (a rule implementation with a known bug — e.g., 2nd-order stencil mislabeled as 4th-order — should produce slope ≈ 2 and fail the assertion).
- 0.25 d: citation documentation covering both layers.
- 0.25 d: integration with CI.

**Risks.**
- At `N=128`, residual may hit floating-point noise floor, breaking the O(h⁴) slope on the last point. Mitigation: inspect residuals and log a flag if slope is computed from only three points due to noise-floor saturation.
- Boundary-stencil choice affects observed slope on homogeneous Dirichlet: physics-lint's 4th-order Fornberg uses interior 5-point, boundary one-sided. Verify against the stencil order Task 4 actually ships.
- *Layer 2 calibration risk.* The `[c_min, C_max]` range depends on the grid, the BC, and the numerical Riesz map used to compute `H⁻¹` norm. The test is only as strong as the reference used to calibrate these bounds. Mitigation: calibrate against a scikit-fem FE reference on the same 64×64 grid; document the calibration procedure in CITATION.md so a reviewer can re-derive the bounds. If the scipy/scikit-fem Riesz computation itself is under audit, Layer 2 should warn rather than fail until the Riesz path is externally verified.
- *Layer 2 scope boundary.* The three-perturbation family tests norm-equivalence on *smooth* perturbations of a *known* analytical solution. It does *not* test norm-equivalence on arbitrary trained-surrogate outputs, which may be non-smooth. That broader claim is deferred to v1.1 (paper-reproduction anchor against Ernst–Rekatsinas–Urban 2025 a posteriori bounds, arXiv:2502.20336v3).

---

### Task 5 — `PH-POS-001`: positivity on canonical Poisson / heat cases

**Anchor type.** Classical theory reproduction.

**Reference.** Evans §2.2.4 Theorem 13 (positivity for Poisson with `f ≥ 0`, homogeneous Dirichlet); Evans §2.3.3 Theorem 8 (weak maximum principle for heat, positivity preservation). Protter–Weinberger Chapters 2 and 3. *Section numbers to be verified.*

**Test design.**
1. **Poisson fixture.** Analytical `u(x,y) = x(1-x)y(1-y)` on unit square. Directly verifiable: `u ≥ 0` in the interior (all four factors non-negative on `[0,1]`), `u = 0` on ∂Ω (each factor vanishes on two sides). Laplacian: `-Δu = 2[y(1-y) + x(1-x)] ≥ 0` on the unit square, satisfying the theorem's `f ≥ 0` hypothesis. No truncation bookkeeping, no series sum — analytical expression evaluated directly on the grid.
2. **Heat fixture.** `u_t = Δu` on unit square, periodic BC, IC `u(x,y,0) = 1 + 0.5 sin(2πx)sin(2πy)` (positive bump). Analytical: `u(x,y,t) = 1 + 0.5 exp(-8π²t) sin(2πx)sin(2πy)`.
   Evaluate at `t ∈ {0, 0.02, 0.05, 0.1}`. Expected: `u ≥ 0` at all timesteps.
3. Run `PH-POS-001` on each fixture; expected PASS.
4. Negative control: construct a prediction by adding a negative bump (`u_test = u - 0.8` on a 5×5 patch near grid center); expected FAIL.

**Pinned expected behavior.**
- Poisson analytical (polynomial): PASS, `min(u) ≥ 0` on every grid point (strict positivity in interior, exact zero on boundary).
- Heat analytical: PASS at every timestep.
- Constructed negative: FAIL with `min(u) < -tol`.

**Acceptance criteria.**
- Both analytical fixtures pass.
- Negative control fails.
- Heat positivity preserved across all four timesteps.
- Test runs in < 10 s on CPU.

**Effort estimate.** ~1 day.

**Risks.**
- Rule's sensitivity to boundary values (`u = 0` on ∂Ω by construction) interacts with PH-BC-001; ensure tests isolate positivity from BC error — run rule with BC checking disabled or scope the assertion to interior nodes only.
- *Fixture simplification note:* earlier draft used a sine-series Poisson analytical `u = (16/π⁴) Σ_{m,n odd} sin(mπx)sin(nπy) / (mn(m²+n²))` which requires truncation bookkeeping and introduces `O(M⁻²)` near-zero artifacts. The polynomial `u = x(1-x)y(1-y)` tests the same theorem with no harness risk; adopted in Rev. 1.1.

---

## 3. Tier A summary

| Task | Rule | Anchor | Effort (d) | Runs in CI |
|---|---|---|---|---|
| 0 | (shared infrastructure) | — | 1.0 | — |
| 1 | PH-POS-002 | Evans §2.2.3 Thm 4 | 0.5 | Yes |
| 2 | PH-CON-003 | Evans §7.1.2 Thm 2 | 0.5 | Yes |
| 3 | PH-SYM-001 + PH-SYM-002 | Helwig Tables 1, 3 as calibration + synthetic controls | 1.0 | Yes |
| 4 | PH-RES-001 | Fornberg 1988 + Bachmayr–Dahmen–Oster + Salari–Knupp (two-layer: discretization + norm-equivalence) | 2.0 | Yes |
| 5 | PH-POS-001 | Evans §2.2.4 Thm 13, §2.3.3 Thm 8 | 1.0 | Yes |

**Total Tier A: ~6.0 engineer-days** (1.0 infra + 0.5 + 0.5 + 1.0 + 2.0 + 1.0 per-rule). Task 0 is front-loaded infrastructure that amortizes across Tier A + Tier B; the subset-visible cost is ~5 days after Task 0 is done. The +0.5 d increase from Rev. 1.0 reflects Task 4's Bachmayr–Dahmen–Oster norm-equivalence sanity-check extension, added in Rev. 1.1 and scope-clarified in Rev. 1.2: the extension tests norm-equivalence on smooth perturbations of a known analytical solution (internal consistency informed by the variational-correctness framework), not external validation of variational correctness on arbitrary surrogate outputs — that broader anchor is deferred to v1.1 via Ernst–Rekatsinas–Urban.

**Deliverables.**
- 6 rules with passing external-validation tests in CI.
- README "External Validation" section with 6 entries, each citing the anchor and linking the harness.
- `external_validation/_harness/` reusable by Tier B.
- v1.0 ships with honest "6 of 18 rules externally anchored; remaining 12 on v1.1 roadmap" framing.

---

## 4. Tier B — v1.1 roadmap (~17 engineer-days)

Remaining twelve benchmark-anchorable rules. Ordered by dependency + effort. Each task below is written at the spec level; full day-by-day breakdowns defer to when v1.1 planning begins (post-visa-deadline).

### Task 6 — `PH-RES-002`: FD-vs-AD residual cross-check

**Anchor.** Chiu et al. "CAN-PINN," *CMAME* 395:114909 (2022), arXiv:2110.15832. AD vs 2nd-order ND discrepancy framework.

**Test design.** MMS smooth solution; compute residual via `torch.autograd.functional.jacobian` and via 4th-order Fornberg FD; assert discrepancy is O(h⁴) on a sequence of refinements.

**Effort.** ~1 day.

**Risks.** Non-smooth activations (ReLU) produce distributional AD derivatives that differ sharply from FD even on smooth inputs; gate the rule on smooth-activation models per PH-NUM-003.

---

### Task 7 — `PH-RES-003`: Spectral-vs-FD residual on periodic grids

**Anchor.** Trefethen *Spectral Methods in MATLAB* (SIAM 2000), Chapter 3 (spectral accuracy). *Chapter and theorem number to be verified against local copy.*

**Test design.** Periodic fixture `u = cos(2πx)cos(2πy)` on torus; compute residual via `torch.fft` spectral derivative and via 4th-order Fornberg FD; discrepancy should be O(h⁴) — FD converges algebraically, spectral converges exponentially, so their difference follows the slower rate.

**Effort.** ~0.5 day.

**Risks.** On under-resolved data (energy above Nyquist), both estimators are wrong; rule should warn if input spectrum has >1% energy above `f_s / 4`.

---

### Task 8 — `PH-BC-001`: Boundary condition violation

**Anchor.** PDEBench (Takamoto et al., *NeurIPS 2022*, arXiv:2210.07182) bRMSE metric definition (Appendix B of the supplement). Sukumar & Srivastava, *CMAME* 389:114333 (2022), arXiv:2104.08426 for tolerance context (qualitative only — specific exponent bands not pinned in the cleaned-draft spec).

**Test design.** Dirichlet, Neumann, periodic fixtures on unit square with known BC traces; run `PH-BC-001` in both relative and absolute modes; assert correct verdict on each.

**Effort.** ~1 day.

**Risks.** Relative mode on homogeneous Dirichlet is the denominator-blow-up issue flagged in v1.0 limitations and backlog (2026-04-17 entry); tests must cover this failure mode explicitly.

---

### Task 9 — `PH-BC-002`: Boundary flux imbalance (divergence theorem)

**Anchor.** Classical Gauss–Green theorem. MMS reference with analytic `F = (x²y, xy²)`, `∇·F = ∂_x(x²y) + ∂_y(xy²) = 2xy + 2xy = 4xy`, `∫_[0,1]² 4xy dx dy = 4 · (1/2) · (1/2) = 1`.

**Test design.** Compute LHS and RHS of divergence theorem on analytic F at four grid resolutions; assert imbalance → 0 at observed O(h²) or O(h⁴) depending on stencil.

**Effort.** ~1 day.

**Risks.** Surrogate outputs scalar u, not F; rule must construct F from u. For FNO on periodic torus, `∫_∂Ω F·n = 0` trivially — test requires non-periodic domain.

---

### Task 10 — `PH-SYM-003`: SO(2) Lie derivative equivariance (Gruver)

**Anchor.** Three-layer structure, named separately so each layer's credibility is clear on its own terms:

- **Layer 1 — metric definition (paper anchor).** Gruver, Finzi, Goldblum, Wilson, "The Lie Derivative for Measuring Learned Equivariance," *ICLR 2023*, arXiv:2210.02984. Equation (4) defines the Local Equivariance Error (LEE); Figure 3 sketches the PyTorch implementation via `torch.autograd.functional.jvp` on a grid-sample-rotated model. This is the external mathematical anchor.
- **Layer 2 — implementation faithfulness (reference-code anchor).** github.com/ngruver/lie-deriv (MIT license). The v1.0 physics-lint implementation runs the repo's `exps_e2e.py` pipeline on a user-supplied ResNet-50 + ImageNet validation set and compares its output to the pipeline the repo itself produces. This verifies physics-lint's metric is *computationally identical* to the reference, not just definitionally consistent.
- **Layer 3 — stability-regression target (local).** A pinned LEE value obtained from the author's own first measurement on a specific ResNet-50 + ImageNet-100-image configuration. This is *not* a paper-certified number and *not* a reproduction claim — it is a local regression gate that catches future physics-lint implementation regressions. If a future commit changes the measured LEE on the same fixture by more than ±20%, something broke.

**Test design.** *Opt-in user-side regression test, not CI-runnable by default.* Per the cleaned-draft spec's Path-1 commitment:
1. Ship the LEE metric implementation in `physics_lint/rules/ph_sym_003.py`, matching Gruver Figure 3 (Layer 1 + Layer 2).
2. Ship `external_validation/PH-SYM-003/reproduce_gruver.sh` that clones `ngruver/lie-deriv`, requires `$IMAGENET_VAL_DIR` env var, runs `python exps_e2e.py --modelname=resnet50 --num_datapoints=100` and compares the produced translation-LEE CSV to the pinned Layer-3 regression target.
3. CI job checks for `$IMAGENET_VAL_DIR`; if absent, skips with a message pointing at the README reproduction instructions. No bundled ImageNet access — the ImageNet terms of access prohibit redistribution.
4. README section documents the three-layer anchor explicitly: "Layer 1 cites the paper for the metric; Layer 2 verifies computational identity against the paper's reference code; Layer 3 is a local stability regression, not a paper reproduction."

**Acceptance criteria.**
- LEE metric implementation passes unit tests against Gruver Figure 3's reference sketch (Layer 1).
- Reproduction script produces a translation-LEE value within ±5% of the value produced by the `ngruver/lie-deriv` repo's own `exps_e2e.py` on the same inputs (Layer 2 — tight, because this is a same-code comparison, not a cross-paper one).
- Pinned Layer-3 regression target is documented in `CITATION.md` with the exact fixture description: ImageNet validation sample count, image preprocessing, ResNet-50 weight source (timm version + commit SHA), hardware class (CPU vs GPU), and the measured LEE value with its date of measurement. Future runs on the same fixture must reproduce within ±20%.
- CI skip-on-missing-env-var produces a clear, non-confusing message linking the reproduction instructions.

**Effort.** ~2 days.
- 0.5 d: LEE metric implementation matching Gruver Figure 3 (jvp + grid_sample); unit tests against the paper's sketch (Layer 1).
- 0.5 d: reproduction script + env-var gate + CI skip logic + same-code comparison test (Layer 2).
- 0.5 d: pin the Layer-3 regression target from an author's first run (requires local ImageNet access); document the fixture configuration.
- 0.5 d: documentation, README section, three-layer citation.

**Risks.**
- *Layer 1 risk — domain mismatch.* Gruver evaluates ImageNet classifiers; physics-lint target is PDE surrogates. The metric definition transfers, but absolute LEE thresholds do not. Layer-1 citation establishes the metric family, not a numerical benchmark for the PDE domain. Document in CITATION.md.
- *Layer 2 risk — reference-code drift.* If `ngruver/lie-deriv` updates its implementation after physics-lint pins its reference, the same-code comparison can produce divergent values. Mitigation: pin the `ngruver/lie-deriv` commit SHA in the reproduction script; upgrade deliberately as a separate PR.
- *Layer 3 risk — single-measurement pinned value.* The initial Layer-3 target is one measurement. The ±20% tolerance is conservative to absorb this single-point uncertainty; tighten after three or more independent measurements establish the true variance. Document in v1.1 backlog as a tightening item.
- Reflection padding in Gruver vs circular padding for periodic PDE surrogates differ at boundaries. The reproduction script uses Gruver's padding (to match the paper's ImageNet pipeline); the PDE-surrogate rule in `ph_sym_003.py` uses circular padding. This is by design — the two padding schemes test different things and should not be unified.

---

### Task 11 — `PH-SYM-004`: Translation equivariance (periodic grids)

**Anchor.** Cohen–Welling 2016 (G-CNN framework) + Helwig et al. 2023 §2.2 (FNO translation-equivariant by design on periodic grids).

**Test design.** Inherits the symmetry harness from Task 3. Random integer shifts via `torch.roll` on periodic fixtures; FFT-based Laplace inverse as positive control (exactly translation-equivariant; **reuses Task 3's zero-mode convention** — `û(k=0) = 0` on the inverse), non-equivariant CNN with positional embeddings as negative control.

**Effort.** ~0.25 day (harness-reuse).

**Risks.** Integer shifts hide subpixel aliasing; PH-SYM-003 (LEE) catches that. Document the complementarity.

---

### Task 12 — `PH-CON-001`: Mass conservation (heat)

**Anchor.** Hansen, Maddix, Alizadeh, Gupta, Mahoney, "Learning Physical Models that Can Respect Conservation Laws," *Physica D: Nonlinear Phenomena* 457 (2024), 133952, DOI 10.1016/j.physd.2023.133952 (peer-reviewed journal, primary citation). Earlier conference version: *ICML 2023*, PMLR 202:12469–12510. Preprint: arXiv:2302.11002. Table 1 diffusion CE numbers at `t = 0.5`. Code at github.com/amazon-science/probconserv.

**Test design.**
1. Port ProbConserv's 1D CE definition to 2D heat: `CE(t) = |∫u(·,t) - ∫u(·,0)| / |∫u(·,0)|`.
2. Analytical fixture: periodic heat on `[0,1]²`, IC `u(x,y,0) = 1 + 0.1 sin(2πx)sin(2πy)`, mass-conserving (periodic BC).
3. Assert `CE(t) < 10⁻⁶` at `t ∈ {0.01, 0.05, 0.1}` on analytical solution.
4. Citation anchor: README cites Hansen Table 1 as "diffusion-family conservation precedent; our threshold is physics-lint's calibration, not a 1D→2D port of Hansen's value."

**Effort.** ~2 days.
- 0.5 d: CE metric implementation.
- 0.5 d: analytical fixture.
- 0.5 d: negative control (inject small mass drift into a prediction; assert FAIL).
- 0.5 d: citation documentation and scope-note.

**Risks.** Hansen is 1D; physics-lint target is 2D/3D. Cited as precedent, not as threshold source — this must be explicit in the README.

---

### Task 13 — `PH-CON-002`: Wave energy conservation

**Anchor.** Evans §2.4.3 (wave energy identity). PDEBench cRMSE methodology as secondary citation.

**Test design.**
1. Self-built 2nd-order FD leap-frog reference on 2D periodic wave, CFL = 0.5.
2. Analytical fixture: `u(x,y,t) = cos(2πt/√2) sin(2πx)sin(2πy)`, exact energy-conserving solution on periodic domain.
3. Run rule on analytical fixture over `t ∈ [0, T]` with T = 10 wave-crossing times; assert `|E(t) - E(0)| / E(0) < 1%`.
4. Negative control: predictions with injected energy drift; assert FAIL.

**Effort.** ~3 days.
- 1 d: FD leap-frog reference implementation (worth getting right — reused for future rules).
- 0.5 d: energy functional implementation on a grid (trapezoidal quadrature for spatial integrals, finite differences for `u_t` and `∇u`).
- 0.5 d: analytical fixture + assertion.
- 1 d: negative control + documentation + citation.

**Risks.**
- No ML-surrogate paper reports wave-energy-drift numbers; anchor is classical theory + self-built reference.
- Discrete E(t) depends on discretization choices; test must use consistent discretization between reference and rule.
- Leap-frog preserves a *modified* energy to O(Δt²); be careful the reference doesn't look like it's drifting.

---

### Task 14 — `PH-CON-004`: Per-element conservation hotspot (mesh-based)

**Anchor.** Bangerth–Rannacher *Adaptive FEM for Differential Equations* (Birkhäuser 2003) Ch. 3; Verfürth *A Posteriori Error Estimation* (Oxford 2013) Ch. 1–2. scikit-fem Example 22 ("Adaptive Poisson equation"; residual indicators `η_K² = h_K²‖f‖²_{0,K}` and `η_E² = h_E‖[[∇u_h · n]]‖²_{0,E}`) as implementation template.

**Test design.**
1. Scope: 2D triangulated meshes only (v1.0); 3D tets deferred to v1.2.
2. Interpolate structured-grid prediction onto `MeshTri` P1 basis.
3. Compute per-element residual via scikit-fem `Functional` decorator in weak form (avoid second derivatives of noisy surrogate output).
4. Concentration metric: `hotspot_fraction = #{K : |r_K| / max|r_K| > 0.5} / N_elem`. This measures how concentrated the residual is near its maximum, *not* how localized a single anomaly is. A smooth non-uniform source will have nontrivial `hotspot_fraction` purely from the shape of its source; this is expected behavior, not a false positive.
5. **Baseline run (reference profile).** Analytical `u = x(1-x)y(1-y)` on a uniformly-refined mesh. Source `f = -Δu = 2[y(1-y) + x(1-x)]` is smooth and non-uniform (maximum at center, zero at corners). Measure `hotspot_fraction_baseline` — this is the reference concentration for a smooth non-localized field, a property of the source-shape-plus-discretization, not a bug. **Serialize the baseline** to `external_validation/PH-CON-004/fixtures/baseline_calibration.json` with schema `{"hotspot_fraction_baseline": float, "n_elem": int, "mesh_hash": str, "calibration_date": str (ISO 8601), "scikit_fem_version": str}`. The comparative acceptance test (step 7) loads this fixture deterministically on every run; a baseline re-calibration is a tracked edit to the fixture file with CITATION.md updated in the same commit. CITATION.md documents the calibration procedure and links to the fixture — the fixture is authoritative, CITATION.md is human-readable context.
6. **Perturbed run (negative control).** Same analytical `u` plus a localized additive bump of magnitude 0.01 in a disc of radius `2h` centered near the domain midpoint. Measure `hotspot_fraction_perturbed`.
7. **Comparative acceptance criterion.** `separation = hotspot_fraction_perturbed − hotspot_fraction_baseline > 0.05`. The rule must report *significantly larger* concentration on the perturbed field than on the baseline; the absolute concentration value is not the test. Additionally, the elements with `|r_K| / max|r_K| > 0.5` on the perturbed run must be spatially clustered on or near the perturbation patch (a clustering check at the element-index level, not an absolute-count check).

**Pinned expected behavior.**
- Baseline `hotspot_fraction_baseline` is a discretization- and mesh-dependent *reference value*, not a pinned number — it is serialized to `fixtures/baseline_calibration.json` during first calibration (mesh hash and scikit-fem version recorded alongside) and loaded on later runs. Baseline re-calibration is an explicit, tracked edit to the fixture file; CITATION.md documents the calibration procedure and provenance.
- **Fixture-mismatch policy.** (a) `mesh_hash` mismatch → **fail hard** with a "mesh differs from calibration; re-calibrate explicitly" message; the mesh determines the numerical result, and silently running against the wrong mesh would produce meaningless separation values. (b) `scikit_fem_version` minor-patch mismatch → **warn and continue**, logging the discrepancy for auditability; minor patches rarely change numerical output meaningfully. Major-version mismatch (leading semver component differs) → **fail** with a recalibration prompt. (c) Fixture completely missing → **fail** with a prompt linking to the calibration script. The principle: the fixture is authoritative; mismatches are events that need explicit acknowledgment, never silent pass-through.
- `separation > 0.05` on the perturbed control.
- Spatial localization: flagged elements of the perturbed run cluster on the perturbation patch.

**Effort.** ~3 days.
- 1 d: structured-grid → mesh projection and weak-form residual.
- 1 d: concentration metric + baseline calibration + comparative separation test + spatial-localization clustering check.
- 1 d: documentation; v1.1-to-v1.2 roadmap entry for 3D tets.

**Risks.**
- Projection error masquerading as conservation violation. Mitigate: on the baseline run, projection error must be `<< ` the `0.05` separation threshold.
- No direct ML-surrogate precedent found in checked sources (per cleaned-draft wording); anchor is classical FE a posteriori theory.
- *Metric choice — v1.1 refinement opportunity.* The `hotspot_fraction` metric measures concentration near the maximum, which only indirectly captures *localization*. A more principled localization metric is the ratio of `max|r_K|` to the *median* of `|r_K|`: a localized anomaly drives this ratio high (spike vs background), while a smooth broad peak keeps it bounded (max/median of order 2–5). Adopting a kurtosis- or max/median-based localization metric is deferred to v1.1 as Path B. The v1.0 comparative-separation approach is methodologically sound as a v1.0 floor but benefits from this tightening in v1.1.

---

### Task 15 — `PH-NUM-001`: Quadrature convergence warning (mesh)

**Anchor.** Ciarlet *FEM for Elliptic Problems* (SIAM Classics 2002) Ch. 4 §4.1 Theorems 4.1.2–4.1.6. scikit-fem Example 1 ("Poisson equation with unit load") as implementation template. *Ciarlet theorem numbers to be verified against local copy.*

**Test design.**
1. Smooth integrand `f(x,y) = sin(πx)sin(πy)` on unit square; analytical integral `4/π²`.
2. Sweep `intorder ∈ {1, 2, 3, 4}` × refinement levels 0–5 using `scikit-fem.Basis(..., intorder=k)`.
3. Log-log regression per `intorder`; assert slope `≥ 2·intorder + 2 - 0.5`.
4. Flag saturation at machine precision.

**Effort.** ~1 day.

**Risks.** Default intorder varies by element; user overrides can trigger spurious warnings. Round-off dominates on very fine meshes.

---

### Task 16 — `PH-NUM-002`: Refinement convergence rate below expected

**Anchor.** Salari–Knupp SAND2000-1444 (Sandia 2000), DOI 10.2172/759450. MASA library (Malaya et al., *Engineering with Computers* 29:487–496, 2013) as code precedent.

**Test design.**
1. SymPy MMS generator for Laplace/Poisson/heat.
2. Refinement loop: `h/2^k` for `k = 0..4`.
3. L²/H¹ errors; log-log regression of last 3–4 points; assert slope within expected band (4 ± 0.5 for 4th-order Fornberg Laplace, etc.).
4. Asymptotic-range check: flag if `E_h / E_{h/2}` outside `[2^p · (1 ± 0.25)]`.

**Effort.** ~2 days.

**Risks.** Neural surrogates show resolution-invariance rather than classical h^p convergence; rule defaults to WARN (not FAIL) on trained-surrogate inputs per v1.0 design. Rule is primarily for verification, not convergence guarantee.

---

### Task 17 — `PH-VAR-002`: Hyperbolic norm-equivalence conjectural (info flag, not a benchmark)

**Anchor.** Documentation-only. Citations: Gopalakrishnan–Sepúlveda 2019; Ernesti–Wieners, *Comput. Methods Appl. Math.* 19 (2019), 465–481; Henning–Palitta–Simoncini–Urban, *ESAIM: M2AN* 56 (2022), 1173–1198, arXiv:2107.12119.

**Test design.**
1. Unit test confirming rule emits INFO-severity warning on any wave-equation invocation.
2. Documentation links the above references.

**Effort.** ~0.25 day.

**Risks.** None — this is a citation-and-warning rule, not a numerical check.

---

## 5. Tier B summary

| Task | Rule | Effort (d) | Runs in CI |
|---|---|---|---|
| 6 | PH-RES-002 | 1.0 | Yes |
| 7 | PH-RES-003 | 0.5 | Yes |
| 8 | PH-BC-001 | 1.0 | Yes |
| 9 | PH-BC-002 | 1.0 | Yes |
| 10 | PH-SYM-003 | 2.0 | Opt-in (ImageNet) |
| 11 | PH-SYM-004 | 0.25 | Yes |
| 12 | PH-CON-001 | 2.0 | Yes |
| 13 | PH-CON-002 | 3.0 | Yes |
| 14 | PH-CON-004 | 3.0 | Yes |
| 15 | PH-NUM-001 | 1.0 | Yes |
| 16 | PH-NUM-002 | 2.0 | Yes |
| 17 | PH-VAR-002 | 0.25 | Yes (unit test only) |

**Total Tier B: ~17 engineer-days.** (Higher than the 9–13 estimate in the cleaned-draft spec because per-task breakdowns include negative controls and documentation time, which the spec's high-level estimates did not. Flag this to the visa-timeline decision: Tier B is not a 2-week job, it's closer to 3–4 weeks in practice.)

**Reconciliation.** The cleaned-draft spec's 14–18-day estimate is Tier A + high-level Tier B. Plan-level Tier B at 17 days matches the upper bound of that range plus margin.

---

## 6. Dependency graph

```
Task 0 (shared infra)
  └── Task 1 (PH-POS-002)            [Tier A]
  └── Task 2 (PH-CON-003)             [Tier A]
  └── Task 3 (PH-SYM-001+002)         [Tier A; builds symmetry harness]
  │     └── Task 10 (PH-SYM-003)      [Tier B; inherits harness]
  │     └── Task 11 (PH-SYM-004)      [Tier B; inherits harness]
  └── Task 4 (PH-RES-001)             [Tier A; builds MMS + refinement harness]
  │     └── Task 7 (PH-RES-003)       [Tier B]
  │     └── Task 16 (PH-NUM-002)      [Tier B; inherits refinement loop]
  └── Task 5 (PH-POS-001)             [Tier A]
  └── Task 6 (PH-RES-002)             [Tier B]
  └── Task 8 (PH-BC-001)              [Tier B]
  └── Task 9 (PH-BC-002)              [Tier B; inherits MMS from Task 4]
  └── Task 12 (PH-CON-001)            [Tier B]
  │     └── Task 13 (PH-CON-002)      [Tier B; builds FD leap-frog ref]
  └── Task 14 (PH-CON-004)            [Tier B; scikit-fem dependency]
  └── Task 15 (PH-NUM-001)            [Tier B; scikit-fem dependency]
  └── Task 17 (PH-VAR-002)            [Tier B; unit test only]
```

**Critical path for Tier A (v1.0 ship).** Task 0 → Tasks 1+2+3+4+5 in parallel (each independent once infra exists). Maximum single-task duration: Task 4 at 2.0 d (two-layer discretization + norm-equivalence), which cannot be split across lanes. With two parallel lanes after Task 0: Lane A runs Task 4 (2.0 d); Lane B runs Tasks 1+2+3+5 (0.5 + 0.5 + 1.0 + 1.0 = 3.0 d). Lane B dominates. Total wall-clock: ~1.0 d (Task 0) + ~3.0 d (Lane B) = **~4.0 d ideal, ~4.5 d with context-switch overhead, for ~6 engineer-days of work**.

**Critical path for Tier B (v1.1 ship).** Task 13 (wave energy conservation, 3 d) and Task 14 (per-element conservation, 3 d) are the two longest single tasks. Total wall-clock at 2-tasks-in-parallel: ~9 days.

---

## 7. Release criteria

### Tier A (v1.0)

- All 6 Tier-A tasks green in CI.
- `external_validation/README.md` documents each anchor with citation, pinned expected value, and harness link.
- Main README "External Validation" section links the `external_validation/README.md` and states the 6-of-18 scope honestly ("v1.0 ships with external validation for 6 rules across all 5 rule families; remaining 12 are on the v1.1 roadmap").
- `docs/backlog/v1.1.md` has 12 Tier-B entries with acceptance criteria copied from this plan.
- No regressions on existing 314-test suite.

### Tier B (v1.1)

- All 12 Tier-B tasks green in CI (with PH-SYM-003 opt-in skip on missing ImageNet).
- `external_validation/README.md` updated to 18-of-18 coverage.
- v1.0 README limitations section ("6 of 18 anchored") updated to v1.1 state ("18 of 18 anchored").
- v1.2 roadmap entries for the three v1.1 scope-carveouts: (a) PH-CON-004 3D tet meshes, (b) PH-SYM-003 pinned LEE value tightened after multiple author measurements, (c) Jekel reopening path if compressible Euler is added to v1.2 PDE coverage.

---

## 8. Risks and mitigations

**Schedule risk.** Tier A at 6 days (5 days after Task 0 infrastructure amortizes) is tight if Task 0 surfaces unexpected problems. Mitigation: Task 0 is a hard gate; if it's not green by end of day 1, pause and diagnose before starting Tier-A rule tasks. Task 4's 2-day budget is the other tight spot — the norm-equivalence layer requires a scikit-fem FE reference for ratio calibration, and if that calibration is noisier than expected the Layer-2 assertion may need WARN-instead-of-FAIL softening before Tier A is declared done.

**Textbook theorem-number risk.** Evans, Ciarlet, and Trefethen section/theorem numbers appear throughout. All are flagged "to be verified against local copy" in this plan. If any turn out to be wrong, soften to chapter/section level (per ChatGPT's fallback advice). This is not a schedule risk — it's a pre-close-out check per task.

**Anchor-drift risk.** Helwig Table 3, Hansen Table 1, and other paper-specific numeric citations pin values as of 2026-04-18. If any paper retracts, corrects, or supersedes the cited value, the anchor becomes stale. Mitigation: CITATION.md includes `verification_date`; v1.1 release gate includes a re-verification pass on all paper anchors.

**Tier-B budget overrun vs cleaned-draft spec.** Tier-B at 17 days sits at the top of the cleaned-draft spec's 14–18 day upper estimate for the full external-validation program, and once Tier A's 6 days are added the combined ~23 days exceeds that upper bound by ~5 days. The delta is attributable to per-task negative controls, citation artifact authoring, and harness-construction accounting that the spec's top-down estimate did not separately budget. Mitigation: do not commit to a v1.1 ship date until Tier A is complete and Tier B has a confirmed budget. If v1.1 ship is visa-constrained, trim Tier B to the cheapest ~6 tasks (total ~6 days — tasks 7, 9, 11, 15, 16, 17 in dependency-graph order) rather than all 12.

**CI runtime risk.** Tier-A tasks 1–5 each run in < 30 s on CPU; combined Tier-A CI job ~2 min. Tier-B tasks 12–14 involve FD reference solvers and mesh operations; combined Tier-B CI job could reach 10+ min. Mitigation: Tier-B workflow runs on push to `main` and on release-tag events only, not every PR. Tier-A is the gate for PR CI; Tier-B is the gate for release tags. Policy is stated canonically in §1.4.

---

## 9. Next actions

1. **Hand this plan to Codex alongside the cleaned-draft external-validation spec.** The spec is the framing document; this plan is the executable breakdown.
2. **Gate on Week 4 completion.** Per the Week 4 sequencing discussion: close Tasks 4 (hero pivot push), 5–7 (gallery, docs, README), then run Task 8 (release gate) *after* Tier-A external validation (Tasks 1–5 of this plan). Tier A is inserted as Week 4 "Task 9" — between Tasks 7 and 8.
3. **Tier A kickoff readiness check.** Before starting Task 0, confirm: (a) local copies of Evans, Ciarlet, Trefethen available for theorem-number verification, or accept section-level citations; (b) scikit-fem ≥ 10.0 installed (pinned in `pyproject.toml`); (c) no outstanding concerns from the cleaned-draft spec review.
4. **Tier B scheduling.** Defer Tier-B plan commitment until Tier A is green and visa-deadline picture is clearer (post-~May 5 when Week 4 + Tier A complete). Tier B budget: re-estimate from empirical Tier-A velocity, not from this plan's a priori estimates.

---

## 10. Document provenance

- **Revision 1.6 (2026-04-18, pass 6):** three-point housekeeping pass addressing final external review (ChatGPT, 2026-04-18, pass 6). Changes: (1) Task 1 acceptance criteria rewritten — the task used an undefined `ratio` metric ("ratio < 10" for pass, "ratio > 100" for fail). "Ratio" was presumably imported from the plan's Invariant 2 (`violation_ratio`), but Task 1's test design is verdict-based max-principle logic (interior extrema vs. boundary extrema within `10⁻³ · range(u)` tolerance), and the ratio concept was never tied to that logic. Deleted the ratio thresholds; acceptance criteria now cite the verdict criterion the test design already named. (2) Task 14 fixture-mismatch policy specified — the Rev. 1.4 serialization stored `mesh_hash` and `scikit_fem_version` alongside the baseline value but didn't state what happens on mismatch. New explicit policy: `mesh_hash` mismatch fails hard, `scikit_fem_version` minor-patch mismatch warns and continues, major-version mismatch fails, missing fixture fails. All branches require explicit acknowledgment — no silent pass-through. (3) Provenance dates normalized to `2026-04-18 (pass N)` scheme — the earlier "evening" / "night" / "later" modifiers and "2026-04-19" forward-dating were cosmetic artifacts of writing provenance blocks during an extended same-day session; they implied calendar travel that didn't happen. All six revisions occurred on 2026-04-18; pass numbers provide unambiguous ordering. Read-back pass now also checks **internal-metric self-consistency**: every metric named in a task's acceptance criteria must be defined in that task's test design. No scope or rule-coverage changes.
- **Revision 1.5 (2026-04-18, pass 5):** two-point correction pass addressing final external review (ChatGPT, 2026-04-18, pass 5). Changes: (1) Task 0 `Citation` dataclass validation broadened — Rev. 1.3's "at least one of `arxiv_id` or `doi`" was still too narrow. Seven book anchors the plan uses (Evans, Protter–Weinberger, Gilbarg–Trudinger, Ciarlet, Trefethen, Bangerth–Rannacher, Verfürth) have neither arXiv IDs nor operationally-useful DOIs. ISBN is the right identifier for these; some sources (Salari–Knupp OSTI, scikit-fem docs) are best identified by URL. New rule: at least one of `{arxiv_id, doi, isbn, url}` must be present. Dataclass schema extended with `isbn` and `url` fields. (2) Task 4 Layer 2 quadrature — Rev. 1.4's "Simpson composite on the 64×64 grid" would have been a parity landmine: composite Simpson requires even subinterval count per axis, and 64 grid points give 63 odd subintervals. `scipy.integrate.simpson` either applies a mixed 1/3+3/8 rule silently or errors depending on version. Replaced with "composite trapezoidal on the 64×64 grid," which has no parity constraint and is accuracy-adequate for this Layer-2 ratio-bounded sanity check. The Rev. 1.4 fix (analytical derivatives instead of spectral) is preserved; only the quadrature step is changed. Read-back pass now includes **methodology-primitive pre-condition checks** (domain compatibility, grid parity, boundary handling) as a standing pre-ship habit, in response to the Rev. 1.3→1.4→1.5 pattern of back-to-back numerical-methods preconditions errors (spectral-on-non-periodic in Rev. 1.3, Simpson-on-odd-subintervals in Rev. 1.4). No scope or rule-coverage changes.
- **Revision 1.4 (2026-04-18, pass 4):** three-point correction pass addressing final external review (ChatGPT, 2026-04-18, pass 4). Changes: (1) Task 4 Layer 2 methodology corrected — the `‖u_pert − u‖_{H¹}` ground-truth computation was described as "via spectral derivatives on the known analytical form," which is wrong on a non-periodic homogeneous-Dirichlet unit square. Spectral differentiation requires periodicity (or Chebyshev with specific boundary handling); the fixture is neither. Since all three perturbations are explicit closed-form functions, the fix is straightforward: compute gradients analytically (by hand or via `sympy`), apply standard quadrature (Simpson composite) on the 64×64 grid. Both exact and simpler than any spectral alternative. (2) Task 14 baseline calibration serialized to `external_validation/PH-CON-004/fixtures/baseline_calibration.json` with schema `{hotspot_fraction_baseline, n_elem, mesh_hash, calibration_date, scikit_fem_version}`. Regression gate now loads the fixture deterministically on every run; CITATION.md documents procedure and provenance, fixture is authoritative. Closes the gap where baseline was only narratively documented. (3) Task 3 synthetic equivariant-operator construction tightened — the radial FFT-based Laplace inverse explicitly fixes the zero Fourier mode to zero by convention. The Laplacian's kernel on a torus is the constant mode, so `(-Δ)⁻¹` is undefined there without a stated convention; `û(k=0) = 0` makes the operator fully defined and the equivariance claim operationally complete. Task 11's inheritance of the same operator now cross-references the Task 3 convention. Read-back pass run against arithmetic, inline math, task-number cross-references, dependency-graph consistency, and methodology-primitive usage before ship. No scope or rule-coverage changes.
- **Revision 1.3 (2026-04-18, pass 3):** three-point correction pass addressing external review (ChatGPT, 2026-04-18, pass 3). Changes: (1) Task 3 harness-reuse citation — the text said "Tier-B tasks 9, 10" but the actual downstream symmetry tasks are 10 (`PH-SYM-003`) and 11 (`PH-SYM-004`); Task 9 is `PH-BC-002` (divergence theorem, not symmetry). Fixed in both the shared-harness description and the effort-estimate note. (2) Task 0 `Citation` dataclass validation — XOR rule softened to "at least one of `arxiv_id` or `doi` must be present" since the plan intentionally cites journal-published works with both (Hansen has Physica D DOI + arXiv; Bachmayr–Dahmen–Oster has IMA JNA DOI + arXiv). The old XOR rule would have rejected the spec's own recommended citations. (3) Task 14 positive-control framing rewritten from absolute "hotspot fraction ≈ 0" (which is false for a smooth non-uniform source, since the residual indicator naturally reflects source-shape concentration) to comparative "separation = hotspot_fraction(perturbed) − hotspot_fraction(baseline) > 0.05" with added spatial-localization clustering check. The baseline run establishes a reference concentration profile; the test asserts the perturbation produces significantly larger concentration than the baseline. Path B (kurtosis or max/median localization metric) noted in risks as a v1.1 refinement opportunity. Read-back pass run against arithmetic, inline math, task-number cross-references (Task 3 references to Tasks 10/11, Task 9/10/11 distinction), and dependency-graph consistency before ship. No scope or rule-coverage changes.
- **Revision 1.2 (2026-04-18, pass 2):** five-point correction pass addressing external review (ChatGPT, 2026-04-18, pass 2). Changes: (1) Task 9 divergence formula corrected (`∇·F = 4xy` for `F = (x²y, xy²)`, not `2xy + x² + y²` — the integral value is still 1 but the displayed intermediate was wrong); (2) Task 2 undefined `ε_num` symbol removed and replaced with a cleaner "equals analytical value `exp(-4π²·0.05)` to within `ε_quad = 10⁻⁴`" assertion; (3) Task 4 Layer-2 framing made precise — Layer 2's `[c_min, C_max]` bounds are calibrated from an internal scikit-fem FE reference, so Layer 2 is an "internal consistency check informed by Bachmayr–Dahmen–Oster," not an external validation of variational correctness on arbitrary surrogate outputs; the true external anchor for variational correctness is Ernst–Rekatsinas–Urban, deferred to v1.1; (4) Task 14 positive-control wording fixed — "uniform residual" was literally false for `u = x(1-x)y(1-y)` since `f = 2[y(1-y) + x(1-x)]` is smooth and non-uniform; replaced with "no hotspots above discretization-and-projection noise floor" (further refined in Rev. 1.3 to a comparative separation metric after Rev. 1.2's "≈ 0" framing was flagged as still overreaching); (5) Task 12 Hansen citation updated to Physica D: Nonlinear Phenomena 457:133952 (2024) as primary peer-reviewed journal, ICML 2023 as earlier conference version, arXiv:2302.11002 for discoverability. Plus §6 Tier-A wall-clock arithmetic made explicit (4.0 d ideal under perfect parallelism, 4.5 d with realistic overhead). Read-back pass run against all arithmetic and inline math before ship. No scope or rule-coverage changes.
- **Revision 1.1 (2026-04-18):** six-point revision pass addressing external review (ChatGPT, 2026-04-18). Changes: (1) budget reconciled bottom-up (Tier A ~6 d, Tier B ~17 d, combined ~23 d; Rev. 1.0's "14–18 engineer-days combined" claim was an unreconciled echo of the cleaned-draft spec that contradicted the per-task tables); (2) Task 4 extended from discretization-only to two-layer anchor (discretization convergence + norm-equivalence sanity); (3) Task 3 renamed from "peer-reviewed paper reproduction" to "synthetic control + literature calibration" since Helwig's tables are on NS/NS-SYM, outside the v1.0 PDE scope; (4) Task 10 reframed as three-layer anchor (paper metric + reference-code faithfulness + local stability regression) to avoid overclaiming a single-measurement pinned value as an external anchor; (5) CI trigger/branch policy unified — Tier A runs on PRs to `main`, Tier B runs on push to `main` and release-tag events only; (6) Task 5 Poisson fixture simplified from sine-series (with truncation bookkeeping) to polynomial `u = x(1-x)y(1-y)` (analytically positive, exact on boundary, no series bookkeeping). No scope or rule-coverage changes.
- **Revision 1.0 (2026-04-18):** initial plan, derived from the cleaned-draft external-validation spec of the same date and from the Week 4 implementation plan (Rev. 4.3). Produced after the Week 2½ scope-control precedent and the 2026-04-17 interpretation-pass recovery. Tier-A / Tier-B split follows the Option-(c) visa-deadline-driven tradeoff.
- This document is executable — every task has acceptance criteria, effort estimate, risk notes, and citation anchors. It is *not* self-verifying: paper-specific numeric citations should be re-verified against their sources on first implementation, and textbook theorem numbers should be verified against local copies before closure.
- Hand-off recipient: Codex, as the v1.0 + v1.1 external-validation implementation spec.
