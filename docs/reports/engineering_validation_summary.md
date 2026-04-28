# Engineering Validation Summary for physics-lint v1.0

This document is the engineering-evidence companion to the separately drafted
`physics_lint_v1_mathematical_foundations.md` (Function-1 source of truth, to
be moved into `docs/reports/`). It records what was built, measured, and
mechanically verified for the v1.0 release. It is not a mathematical proof,
not formal verification, and not a peer-review certification. PASS in any
table below means "the rule's implementation matches its documented
contract on the harness fixtures the anchor exercises," not "the rule is
universally physically correct."

## 1. Final Repository State

- Branch: `external-validation-tier-a`
- Final commit: `6907699` ("feat(external-validation): Task 13 v1.0 closeout — 18-of-18 external anchors shipped")
- PR: [#3 OPEN](https://github.com/tyy0811/physics-lint/pull/3) — "Complete v1.0 external validation (18 of 18 rules)"
- CI status: all 28 checks SUCCESS (18 per-rule external-validation matrix jobs + closeout-scripts job + 5-shard `physics-lint` core CI matrix + 3 dogfood-lint shards + `physics-lint` aggregate gate); PR is `MERGEABLE`
- Release target: v1.0.0 (held until adversarial review + PR review triaged)
- Date: 2026-04-25

The branch is 64 commits ahead of `origin/master`, 131 files changed
(+18,714 / -3,260 lines), bundling complete-v1.0 external-validation
work plus Week-4 CLI / SARIF / report-serializer surface and the
dogfood CI shards.

## 2. Validation Coverage Summary

- Benchmark-anchorable rules: 18 (PH-RES-001..003, PH-BC-001..002, PH-CON-001..004, PH-POS-001..002, PH-SYM-001..004, PH-NUM-001..002, PH-VAR-002)
- Rules externally anchored: 18 of 18 (100%)
- External-validation tests: 268 collected (267 passed, 1 skipped) — 132 per-rule tests across 18 `external_validation/PH-*/test_anchor.py` files plus 44 shared-harness unit tests in `external_validation/_harness/tests/`
- Unit tests: 283 collected, 283 passed in `tests/` (excluding `tests/dogfood` and `tests/test_make_ci_dumps.py`, which are pre-existing environmental imports unrelated to the v1.0 surface)
- Skipped tests: 1 in the external-validation suite (a configuration-conditional fixture in `PH-RES-002`); 0 in the unit-test suite
- Closeout scripts: 2, both pass (see Section 7); each emits 18 `OK:` lines, exit 0
- Lint/format status: `ruff check external_validation/ src/ scripts/ tests/` reports "All checks passed!" (exit 0); `ruff format --check` reports "145 files already formatted" (exit 0)

## 3. Rule-by-Rule Validation Matrix

Production-scope phrasing is the rule's own self-description; F1 / F2 / F3
columns synthesize from each rule's `CITATION.md` and `README.md`.
"F3 status" uses the language `external_validation/README.md` adopts:
present / absent-with-justification / absent-by-structure / supplementary.

| Rule | Family | Production scope | F1 math basis | F2 correctness fixture | F3 status | Tests | Main limitation |
|---|---|---|---|---|---|---:|---|
| PH-RES-001 | Residual | Variationally-correct residual norm with config-dependent norm-equivalence (periodic+spectral vs non-periodic+FD) | Fornberg 1988 interior O(h⁴); Bachmayr-Dahmen-Oster H⁻¹ norm-equivalence on periodic spectral grids | Two-path: log-log slope on `sin(πx)sin(πy)` MMS at `N ∈ {16,32,64,128}`; BDO norm-equivalence two-layer | present (Fornberg interior reproduction + BDO two-layer) | 12 | Norm-equivalence holds only on periodic+spectral; non-periodic+FD path falls back to L² (characterized, not fixed) |
| PH-RES-002 | Residual | Max interior relative discrepancy ratio between AD-computed and FD-computed Laplacian on a smooth callable | Griewank-Walther Ch 3 reverse-mode AD accuracy + LeVeque FD consistency-order + Chiu et al. 2022 CAN-PINN structural framing | AD-vs-FD agreement on smooth MMS, max-discrepancy-shrinks-at-O(h⁴) on log-log refinement | absent-with-justification (no directly-comparable AD-vs-FD per-point baseline; CAN-PINN cross-check is framework-level) | 5 | Compares two numerical primitives, not against a closed-form truth |
| PH-RES-003 | Residual | Spectral-vs-FD residual on periodic grids — exponential spectral collapse + polynomial FD decay | Trefethen 2000 Chs 3-4 spectral accuracy; Fornberg O(h⁴) corroboration; LeVeque consistency-order | Closed-form `exp(sin x)` periodic fixture at `N ∈ {16,32,64}` matching exponential-decay fit `R² > 0.99` | absent-with-justification (Trefethen's canonical demo is a plot, not a tabulated reproduction target; Boyd 2001 in Supplementary) | 7 | Trefethen and Boyd carry curve-shape framing only |
| PH-BC-001 | Boundary | Discrete-L² Dirichlet boundary trace `\|\|γ(u) - boundary_target\|\|`; Neumann/flux semantics out of v1.0 scope | Evans §5.5 Theorem 1 (trace operator `γ : H¹(Ω) → H^{1/2}(∂Ω)`, section-level) + PDEBench bRMSE semantic-equivalence framing | Three analytic Dirichlet fixtures: zero-on-exact-trace, perturbation-scaling, absolute-vs-relative mode branch | absent-with-justification (PDEBench HDF5 loader not shipped in v1.0; pinned rows in Supplementary, F3-INFRA-GAP) | 5 | Dirichlet-only production scope; Neumann would require a separate normal-derivative path |
| PH-BC-002 | Boundary | Laplace-imbalance `∫Δu dV + ∮∂u/∂n` on harmonic fields; Poisson arm raises `NotImplementedError` until v1.1 | Evans App C.2 Theorem 1 / Gilbarg-Trudinger §2.4 Gauss-Green / divergence theorem | Harness-level: `F = (x, y)` Gauss-Green to roundoff on triangulation + quadrilateralization. Rule-verdict: harmonic Laplace fixtures emit ≈ 0; Poisson SKIP path | absent-by-structure (Gauss-Green reproduction on MMS fixtures is tautological under stated preconditions; LeVeque FVM in Supplementary) | 7 | V1-stub CRITICAL three-layer; production rule narrower than F1 theorem |
| PH-CON-001 | Conservation | Integral conservation drift `\|∫u(t) - ∫u(0)\|` on heat-type PDEs | Evans §2.3 mass-conservation balance law + Dafermos Ch I + Hansen ProbConserv CE framing | Analytical-snapshot fixture `u = cos(2πx)cos(2πy)·exp(-8π²κt)` (zero-mean periodic eigenmode); drift floor ~1e-18, tolerance 1e-15 with ~1000× safety factor | absent-with-justification (Hansen ProbConserv loader not shipped in v1.0; Hansen Table 1 ANP row pinned in Supplementary, F3-INFRA-GAP) | 6 | Analytical-snapshot only; numerically-evolved time-stepper validation is out of v1.0 scope |
| PH-CON-002 | Conservation | Wave-energy drift `\|E(t) - E(0)\| / E(0)` on hyperbolic snapshots | Evans §2.4.3 wave-energy identity + Strauss §2.2 + Hairer-Lubich-Wanner Ch IX symplectic conservation | Two-layer: harness-authoritative analytical `E(t)` from `(u_t, ∇u)` snapshots (roundoff ~5e-16); rule-verdict log-log slope 1.94 from rule's internal 2nd-order-central FD `u_t` | absent-with-justification (no peer-reviewed per-point wave-energy table; PDEBench + Hansen in Supplementary as scope-incompatible) | 9 | Analytical-snapshot only; not a leapfrog time-stepper |
| PH-CON-003 | Conservation | Heat energy-dissipation sign `dE/dt ≤ 0` via per-step ratio | Evans §2.3.3 Theorem 4 / §7.1.2 Theorem 2 parabolic energy estimate (✅ primary-source verified, tight theorem-number framing) | Analytical eigenmode `sin(πx)sin(πy)·exp(-2π²t)` with closed-form per-step ratio `exp(-0.2π²) ≈ 0.13888` at `Δt = 0.05` | absent-with-justification (rule's emitted `dE/dt` differs definitionally from available published columns) | 3 | Forward-difference `dE/dt` primitive; Rev 1.6 central-diff bug fixed in `e691dd3` |
| PH-CON-004 | Conservation | `max_K / mean_K` of `∫_K (Δ_{L²-proj zero-trace} u)² dx` over interior elements of a 2D triangulation — narrower than Verfürth full residual estimator | Verfürth 2013 Thm 1.12 + Bangerth-Rannacher 2003 + Ainsworth-Oden 2000 (chapter-level ⚠) | L-shape singularity hotspot fixture; ratio refinement-invariant at ~1.70 element-layers across uniform refinements | absent-with-justification (effectivity-index values depend on estimator+marker+solver triple; Becker-Rannacher 2001 DWR in Supplementary) | 9 | 2D triangulated meshes only (3D tetrahedral deferred to v1.2); narrower-estimator-than-theorem scoping |
| PH-POS-001 | Positivity | Discrete-predicate verdict for Poisson + heat positivity | Evans §2.2.3 Theorem 4 + p. 27 Positivity corollary (✅ primary-source verified); Evans §2.3.3 Theorem 4 (✅ primary-source verified) | Closed-form positive harmonic / parabolic fixtures + Evans-corner negative control | absent-with-justification (discrete-predicate rule; theorems reproduced as structural identity, no numerical baseline to reproduce) | 4 | Discrete predicate; raw_value is a binary verdict |
| PH-POS-002 | Positivity | Discrete-predicate weak-maximum-principle check `max_Ω u ≤ max_∂Ω u` | Evans §2.2.3 Theorem 4 (✅ primary-source verified, tight framing) | Three harmonic polynomials (`x²-y²`, `xy`, `x³-3xy²`) + interior-overshoot negative control | absent-with-justification (analogous to PH-POS-001) | 4 | Discrete predicate; v1.0 known issue: relative-mode floor underflow on homogeneous-Dirichlet samples (deferred to v1.2) |
| PH-SYM-001 | Symmetry | C₄ discrete-rotation equivariance error on a 2D periodic square grid | Hall 2015 §2.5 + §3.7 (⚠) + Varadarajan 1984 §2.9-2.10 (⚠) + Cohen-Welling 2016 G-equivariant CNN structural bridge | C₄ structural-equivalence on closed-form rotation-equivariant operators; non-equivariant negative control | absent-with-justification (no published baseline directly comparable to rule's `rotate_test` emitted quantity; Helwig 2023 in Supplementary) | 4 | Section-level Hall/Varadarajan framing (no tight theorem-number) |
| PH-SYM-002 | Symmetry | Z₂ discrete-reflection equivariance error on a 2D periodic square grid | Hall 2015 §2.5 + §3.7 (⚠) + Varadarajan 1984 §2.9-2.10 (⚠) | Mirror-image structural-equivalence on closed-form reflection-equivariant operators; non-equivariant negative control | absent-with-justification (analogous to PH-SYM-001) | 4 | Section-level framing |
| PH-SYM-003 | Symmetry | SO(2) Lie-derivative diagnostic — adapter-mode-only; per-point L² norm of `(L_A f)(x)` against a floor; **scalar-invariant only, not global finite equivariance** | Hall 2015 + Varadarajan 1984 (section-level ⚠) + Kondor-Trivedi 2018 compact-group equivariance | F2 harness-authoritative on FFT-Laplace inverse (exactly C₄- + reflection-equivariant) and non-equivariant CNN; rule-verdict via `torch.autograd.functional.jvp` with six gating preconditions | absent-with-justification (RotMNIST + Modal A100 + ImageNet-opt-in Gruver pre-demoted per F3-INFRA-GAP; Cohen-Welling, Weiler-Cesa, Gruver in Supplementary) | 18 | V1 narrower than F1: infinitesimal scalar-invariant only; finite-multi-output equivariance deferred |
| PH-SYM-004 | Symmetry | Translation-equivariance V1 stub: `SKIPPED`-always past declared-symmetry + periodicity gates | Kondor-Trivedi 2018 + Li et al. 2021 FNO §2 convolution theorem | Four controlled operators (identity / circular convolution 1D-2D / Fourier multiplier 1D-2D) + coordinate-dependent non-equivariant negative control via `shift_commutation_error` | absent-with-justification (Helwig / FNO / equivariant-NN landscape pinning deferred; F3 contract revised 2026-04-24) | 14 | V1-stub CRITICAL three-layer; live-callable adapter mode deferred to v1.2 |
| PH-NUM-001 | Numerical | FEM quadrature exactness V1 stub: `PASS` with pass-through `field.integrate()` baseline and reason `"qorder convergence check is a stub until V1.1"` on `MeshField` input | Ciarlet 2002 §4.1 Thms 4.1.2-4.1.6 (⚠) + Strang 1972 Variational Crimes (⚠) + Brenner-Scott 2008 §10.3 (⚠) | Three scoped cases via `_harness/quadrature.py`: (A) `degree ≤ intorder` exact to roundoff; (B) under-integrated bounded away from 0; (C) error drops by factor ~7.8e12 across `intorder ∈ {2,4,6,8,10}` | absent-with-justification (no peer-reviewed paper tabulates `p_obs` for the specific `(p, intorder, MMS)` triples; Ern-Guermond §8.3 + MOOSE in Supplementary) | 11 | V1-stub CRITICAL three-layer; full MMS h-refinement + observed `p_obs` deferred to v1.2 |
| PH-NUM-002 | Numerical | Refinement convergence rate `p_obs = log₂(r_h / r_{h/2})` on declared MMS cases | Ciarlet 2002 §3.2 Céa's lemma (⚠) + Strikwerda 2004 Ch 10 Lax equivalence (⚠) | Three scoped cases: (A) harness-authoritative `mms_observed_order_fd2` `p_obs → 2`; (B) FD non-periodic `p_obs ≈ 2.50` from boundary-band scaling; (C) saturation floor below `1e-11` returns `rate = inf` | absent-with-justification (Roy 2005 + Oberkampf-Roy 2010 supply algorithm framing only; per 2026-04-24 user-revised contract) | 8 | Case-specific expected rate; periodic harmonics on T² are constants by Liouville (structural unreachability documented as scope-truth) |
| PH-VAR-002 | Diagnostic | Info-flag wave-equation diagnostic: emits `info`-severity `PASS` with literature-pointer reason on wave-equation specs; `SKIPPED` otherwise | Gopalakrishnan-Sepúlveda 2019 + Ernesti-Wieners 2019 + Henning-Palitta-Simoncini-Urban 2022 + Demkowicz-Gopalakrishnan 2010/2011 DPG stack | Diagnostic contract verification only (not a numerical fixture): rule emits info-PASS on wave specs, SKIPPED on every other PDE kind | absent-by-structure (info-flag rules emit no numerical output to reproduce; Demkowicz-Gopalakrishnan 2025 Acta Numerica DOI 10.1017/S0962492924000102 in Supplementary) | 5 | Info-severity diagnostic; literature-maturity-gated promotion to numerical-comparison rule deferred to v1.2 |

Judgment-call cells flagged for spot-check:

- **PH-RES-001 F3 status.** The README labels PH-RES-001 as the only
  F3-PRESENT rule, and the CITATION.md describes a Fornberg interior
  reproduction layer plus a BDO norm-equivalence two-layer. We follow the
  README's "F3 present" categorization; the underlying anchor mixes a
  textbook-rate reproduction (Fornberg) with a structural identity layer
  (BDO), which is more nuanced than a single "F3 = published-row
  reproduction" target.
- **PH-CON-002 F3 status.** The CITATION.md text reads "absent with
  justification (pre-recorded by Task 0 Step 4 pin audit)"; the F3-hunt
  audit treats the disposition as a structural absence (no second-order-
  in-time energy functional in PDEBench / Hansen). We carry through the
  CITATION.md's exact phrasing ("absent with justification") rather than
  recoding it as "absent-by-structure."
- **PH-BC-002 F3 status.** Phrased as "absent by structure" in the
  CITATION.md. Recorded here as such.
- **Test counts.** The matrix uses per-file `def test_` counts from
  each `test_anchor.py`, which excludes parameterized expansions and
  shared `_harness/tests/` cases that fire under every rule's matrix
  job. The total of per-rule counts (132) plus shared-harness tests
  (44) plus parametrized expansions equals the 268 collected by pytest.

## 4. F3 Borrowed-Credibility Scorecard

| Status | Count | Rules | Meaning |
|---|---:|---|---|
| F3 present / executable | 1 | PH-RES-001 | Live reproduction of a published numerical baseline (Fornberg 1988 interior O(h⁴) rate) plus a BDO norm-equivalence two-layer |
| F3 absent with justification | 14 | PH-RES-002, PH-RES-003, PH-BC-001, PH-CON-001, PH-CON-002, PH-CON-003, PH-CON-004, PH-NUM-001, PH-NUM-002, PH-POS-001, PH-POS-002, PH-SYM-001, PH-SYM-002, PH-SYM-003, PH-SYM-004 | The rule's emitted quantity has no directly-comparable published numerical baseline executable in v1.0 (either no such row exists in the literature, or the row exists but the loader is deferred) |
| F3 absent by structure | 2 | PH-BC-002, PH-VAR-002 | Reproduction is structurally not applicable (Gauss-Green is tautological under its preconditions; info-flag rules emit no numerical output) |
| Supplementary calibration context only | applies to most absent rules | (per-rule) | Calibration-only references appear under "Supplementary calibration context" subsections; flagged "calibration, not reproduction" per plan §1.2 |
| Opt-in / deferred (F3-INFRA-GAP) | 3 | PH-BC-001, PH-CON-001, PH-SYM-003 | Pinned reproduction targets exist (PDEBench rows / Hansen Table 1 ANP / RotMNIST + escnn + Modal A100), loaders deferred to v1.2 (see `docs/backlog/v1.2.md`) |

(Counts add to 17 in the absent rows; PH-SYM-004 and PH-NUM-001 also
pre-demoted F3 candidates per V1-stub CRITICAL framing — see Section 5.1.)

## 5. Validated Engineering Patterns

These are patterns the v1.0 program adopted and applied across multiple
rules — i.e., they are not one-off rationalizations.

### 5.1 Three-layer contract for narrow or stubbed rules
- Used by: PH-BC-002, PH-NUM-001, PH-SYM-004 (V1-stub CRITICAL); PH-SYM-003 (narrower-than-theorem CRITICAL); applied secondarily to PH-CON-004 (narrower-estimator scope) and PH-RES-001 (narrower-than-Demkowicz scope).
- Meaning: F1 mathematical-legitimacy carries the full theorem family. F2 splits into a harness-level fixture (authoritative on controlled operators) and a separate rule-verdict contract that exercises the production rule's actual emitted-quantity scope (which may be a stub, a narrower indicator, or a discrete predicate). The two are kept distinct in prose so a reader does not infer broader production coverage from the F1 citation.
- Why it matters: Prevents implicit overclaim. A `SKIPPED`-always rule still has a verifiable structural anchor; a narrower estimator still has a defensible mathematical lineage; the reader sees both layers without conflating them.

### 5.2 Analytical-snapshot versus numerical-time-stepper separation
- Used by: PH-CON-001 (heat mass), PH-CON-002 (wave energy), PH-CON-003 (heat dissipation); applies in spirit to PH-CON-004 (mesh-snapshot indicator).
- Meaning: F2 fixtures are analytic closed-form snapshots of the conserved quantity, not output from a numerical time-stepper. Tolerance is at float64 roundoff; mixing analytical-snapshot tolerance with numerically-evolved tolerance is explicitly disallowed. A leapfrog or RK4 time-stepper validation would be a separate anchor with method-dependent tolerance — out of v1.0 scope.
- Why it matters: Conservation rules in production typically run against trained-surrogate output. Validating the rule's drift metric against a closed-form snapshot at machine roundoff isolates "is the rule's measurement primitive correct" from "does the model conserve under integration," which are different questions with different evidence requirements.

### 5.3 Harness-level F2 versus production-rule scope separation
- Used by: PH-BC-001 (Dirichlet vs Neumann), PH-BC-002 (full Gauss-Green vs Laplace-imbalance V1), PH-SYM-003 (controlled-operator C₄/reflection vs SO(2) Lie-derivative scalar invariant), PH-SYM-004 (FFT operators vs SKIP-always V1), PH-NUM-001 (quadrature exactness vs PASS-with-stub-reason V1), PH-CON-004 (L-shape localization vs `max_K / mean_K` ratio).
- Meaning: When the F1 mathematical theorem covers strictly more ground than the v1.0 production rule, F2 is split into (i) a harness-level fixture in `external_validation/_harness/` that validates the F1 statement on closed-form operators, and (ii) a rule-verdict contract that validates the rule's narrower production behavior against its declared scope.
- Why it matters: Lets the F1 citation stand on its full-theorem strength without the rule appearing to claim that strength. Anyone reading the harness fixture sees the theorem's full property tested; anyone reading the rule-verdict contract sees the production scope's actual behavior.

### 5.4 F3 infrastructure-gap handling (F3-INFRA-GAP)
- Used by: PH-BC-001 (PDEBench loader), PH-CON-001 (Hansen ProbConserv loader), PH-SYM-003 (RotMNIST + escnn + Modal A100 + Gruver ImageNet).
- Meaning: When a directly-comparable published numerical baseline exists but its loader / dataset / opt-in dependency stack is not shipped in v1.0, the F3 layer is pre-demoted to "absent with justification — F3-INFRA-GAP" in the rule's CITATION.md, with the pinned row preserved in Supplementary calibration context. The infrastructure cost is logged as a v1.2 backlog item (see `docs/backlog/v1.2.md`) with explicit acceptance criteria that promote F3 from absent to present once the loader ships.
- Why it matters: Avoids two failure modes — (a) shipping an F3 layer that fails or is brittle in CI because the loader cache misses, and (b) silently dropping a known reproduction target. The pinned row + Supplementary placement keeps the future reproduction reachable; the v1.2 backlog entry creates a concrete promotion path.

### 5.5 Rule-internal numerical-derivative split
- Used by: PH-CON-002 (wave-energy `u_t` via 2nd-order central FD inside `ph_con_002.py:65`); PH-CON-003 (heat `dE/dt` via forward-difference primitive after the Rev 1.6 central-difference bug fix in `e691dd3`).
- Meaning: The rule's measurement primitive that takes a numerical derivative is documented separately from the analytic fixture, with its truncation order named explicitly. The rule-verdict contract layer measures the resulting `O(Δt²)` or `O(Δt)` drift on the analytical snapshot; the harness-level layer computes the same conserved quantity directly from analytical derivatives at roundoff.
- Why it matters: Two readings of the same drift metric are kept separate — the truth-up-to-roundoff reading from the harness, and the production-cost reading from the rule. The drift gap between them quantifies the cost of the in-rule numerical primitive, which is the only honest way to read it.

## 6. CI and Reproducibility

```bash
# full unit tests
source .venv/bin/activate
pytest tests/ --import-mode=importlib \
    --ignore=tests/dogfood --ignore=tests/test_make_ci_dumps.py
# 283 passed in ~22 s

# external validation tests (all 18 anchors + shared harness)
pytest external_validation/ --import-mode=importlib
# 267 passed, 1 skipped in ~8 s

# closeout scripts
python scripts/check_citation_md_three_function_structure.py \
    external_validation/PH-*/CITATION.md
python scripts/check_theorem_number_framing.py \
    external_validation/PH-*/CITATION.md
# both: 18 OK lines, exit 0

# lint / format
ruff check external_validation/ src/ scripts/ tests/      # All checks passed!
ruff format --check external_validation/ src/ scripts/ tests/  # 145 files already formatted
```

What passes: all 283 unit tests, all 267 of 268 external-validation
tests (1 conditional skip in `PH-RES-002`), both closeout scripts, and
all CI checks on PR #3. What is intentionally excluded: `tests/dogfood`
and `tests/test_make_ci_dumps.py` rely on local-environment imports
unrelated to the v1.0 surface and are a pre-existing condition on this
machine; they fail on `master` as well and are excluded from the unit-
test count for that reason. The full external-validation suite runs in
under 30 s on CPU with no GPU / Modal / ImageNet / escnn / e3nn /
RotMNIST dependency. CI runs all 18 rules in a `fail-fast: false`
matrix on every PR to `master` plus the closeout-scripts job
(`.github/workflows/external-validation.yml`); the core `physics-lint`
matrix (`.github/workflows/physics-lint.yml`) covers Python 3.10 / 3.11
/ 3.12 with multiple NumPy + PyTorch combinations including the
NumPy 1.x ABI ceiling.

## 7. Closeout Scripts

- `scripts/check_citation_md_three_function_structure.py`: enforces
  every per-rule `CITATION.md` contains a `## Function-labeled citation
  stack` section with the three required subsections
  (`### Mathematical-legitimacy`, `### Correctness-fixture`,
  `### Borrowed-credibility`), and that the Borrowed-credibility
  subsection is either populated with at least one identifier-bearing
  citation (arxiv/doi/isbn/url) or carries an explicit `F3 absent` /
  `absent with justification` marker. Verified result: 18 of 18
  `OK:` lines, exit 0.
- `scripts/check_theorem_number_framing.py`: enforces that any textbook
  flagged ⚠ in `external_validation/_harness/TEXTBOOK_AVAILABILITY.md`
  is cited at section level, not at tight theorem number. Verified
  result: 18 of 18 `OK:` lines, exit 0.

Both scripts are wired into the `external-validation.yml` workflow's
`closeout-scripts` job and gate every PR.

## 8. Deferred v1.2 Work

Items explicitly deferred from v1.0 per `docs/backlog/v1.2.md`:

- 3D PH-CON-004 tetrahedral meshes (Fichera-corner fixture; `MeshTet` path)
- Full PH-NUM-001 MMS h-refinement with observed `p_obs` measurement
- PDEBench HDF5 loader and Hansen ProbConserv format adapter (F3 promotion path for PH-BC-001 and PH-CON-001)
- RotMNIST + escnn + e3nn + Modal A100 + ImageNet-opt-in Gruver reproduction (F3 promotion path for PH-SYM-003)
- PH-SYM-004 adapter mode (live-callable `f(roll(x)) == roll(f(x))` path; jvp-like pattern from PH-SYM-003)
- PH-VAR-002 hyperbolic norm-equivalence theory tightening (literature-maturity-gated promotion from info-flag to numerical-comparison rule)
- True pointwise `MeshField.laplacian()` via Superconvergent Patch Recovery (currently `NotImplementedError`; users redirected to the L²-projected zero-trace path)
- Six-surrogate dogfood expansion (ensemble, OT-CFM / flow matching, improved DDPM, DPS) for the laplace-uq-bench cross-comparison
- A1 metrics-compatibility shim (`upstream_compatible_raw_value` mode for PH-RES-001 / PH-POS-002 byte-identical sanity-axis comparison)
- "MSE misses what physics catches" marketing scatter plot
- CLI `check` auto-extraction of per-rule kwargs (`boundary_target`, `boundary_values`, `refined_field`)
- CLI `[tool.physics-lint.rules]` per-rule override surface
- Pre-commit hook multi-target support and `self-test --rule <id>`
- PH-BC-001 / PH-RES-001 relative-mode denominator regularization on homogeneous-Dirichlet samples

## 9. Known Limitations

- **Not formally verified.** No rule's implementation has been mechanically proved correct. Fixtures verify behavior on closed-form / analytical / controlled-operator inputs; they do not certify the rule on arbitrary user fields.
- **Not peer reviewed.** The v1.0 release has not been submitted to peer review. Adversarial review (Codex) and PR review are scheduled before tagging v1.0.0; the version tag is held pending those reviews.
- **PASS does not prove physical correctness.** A rule's PASS verdict means its measurement primitive matched the harness's analytical or controlled-operator expectation within the documented tolerance. PASS on a trained-surrogate dump means the rule's emitted quantity sits below the rule's threshold on that dump; it does not certify the surrogate is physically faithful, nor does it certify the rule covers all relevant failure modes.
- **F3 is not executable for every rule.** Only PH-RES-001 ships a live F3 reproduction. Fourteen rules carry F3-absent-with-justification (no comparable published row, or row exists but the loader is deferred); two rules (PH-BC-002, PH-VAR-002) carry F3-absent-by-structure. The `external_validation/README.md` "approximately one rule with live F3, the rest absent-with-justification" framing is the load-bearing claim; the per-rule CITATION.md Borrowed-credibility subsections record the exact reasoning.
- **Some validations are harness-level, not full production generality.** Where F1's theorem covers more than the v1.0 production rule (PH-BC-002, PH-NUM-001, PH-SYM-003, PH-SYM-004, and to a lesser extent PH-CON-004 and PH-RES-001), the F2 harness-level layer is the authoritative validation of the theorem statement on controlled operators; the rule-verdict contract validates only the production rule's narrower scope. The two layers are deliberately separate to avoid implicit overclaim.
- **Some rules are diagnostic or v1-stub-scoped.** PH-VAR-002 is an info-severity diagnostic with no numerical output. PH-NUM-001, PH-SYM-004, and the Poisson arm of PH-BC-002 ship as v1-stubs (PASS-with-reason or SKIP-always) until v1.1 / v1.2 wires through full production behavior. The CRITICAL three-layer pattern documents this rather than concealing it.
- **Section-level theorem framing.** Several textbook citations (Hall 2015, Varadarajan 1984, Ciarlet 2002, Trefethen 2000, Verfürth 2013, and others) carry ⚠ status in `external_validation/_harness/TEXTBOOK_AVAILABILITY.md` because primary-source verification was not available; these are cited at section / chapter level, not at tight theorem number. The closeout script `check_theorem_number_framing.py` enforces this mechanically.
- **Two rules' production scope is narrower than the cited theorem.** PH-CON-004 implements only the interior volumetric `||Δu_{L²-proj}||²` term, not the full Verfürth residual estimator with source and facet-jump terms. PH-SYM-003 implements the infinitesimal scalar-invariant Lie derivative, not global finite multi-output equivariance. Both are framed accordingly in F1; no broader guarantee is inherited.

## 10. Provenance Summary

- Number of commits on the branch (relative to `origin/master`): 64
- Number of plan diffs logged: 34 cumulative (recorded in commit-message Provenance trailers under the "Plan-diffs (plan-vs-committed-state drift, §7.4)" header; Tasks 2-12 contributed 1-29; Task 6 added 30; Tasks 7-12 closeout normalization confirmed 34; Task 13 closeout adds 0 new diffs)
- Where plan diffs are recorded: in commit-message Provenance trailers throughout the branch's commit history (visible via `git log origin/master..HEAD --pretty=full`); the discipline is documented in the complete-v1.0 plan §7.4 plan-vs-committed-state drift handling
- PR link: https://github.com/tyy0811/physics-lint/pull/3
- Final CI link: see "Checks" tab on PR #3 — all 28 jobs passed (18 per-rule external-validation matrix + closeout-scripts + 5 core `physics-lint` matrix shards + 3 dogfood-lint shards + aggregate `physics-lint` gate)
