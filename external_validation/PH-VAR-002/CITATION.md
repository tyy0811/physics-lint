# PH-VAR-002 — Hyperbolic norm-equivalence conjectural caveat (info-flag diagnostic, F3-absent-by-structure)

## Scope-separation discipline (read first)

PH-VAR-002 validates **a diagnostic info-flag contract for the
hyperbolic (wave-equation) path of physics-lint**. The rule does not
compute a numerical residual against the field; it emits an
`info`-severity `PASS` with a literature-pointer reason whenever it
is invoked on a wave-equation `DomainSpec`, and `SKIPPED` on every
other PDE kind. The external-validation anchor therefore has no
numerical F2 layer — just a contract-verification unit test — and
**F3 is absent by structure** (an info-flag rule has no numerical
output to reproduce).

This anchor is deliberately conservative:

- We do **not** claim physics-lint validates hyperbolic norm-
  equivalence. The rule's existence is precisely to flag that the
  Bachmayr-Ernst parabolic variational-correctness framework does
  not cover wave-equation problems, and that any wave-equation
  `PASS` returned by other rules should be read as "within
  conjectural tolerance," not "certified by theory."
- We do **not** promote Demkowicz-Gopalakrishnan 2025 (Acta
  Numerica) to F3 credibility. The paper reviews DPG hyperbolic-
  problem methodology without publishing a directly-comparable
  numerical baseline on physics-lint's emitted quantity. It lives
  in Supplementary calibration context as framing.

**Wording discipline.** Required:

> PH-VAR-002 emits an info-severity diagnostic on wave-equation
> problems pointing users to DPG hyperbolic norm-equivalence
> literature; it does not certify hyperbolic norm-equivalence and
> does not compute against the field.

Avoid:

> PH-VAR-002 validates hyperbolic norm-equivalence.
> PH-VAR-002 certifies wave-equation residuals.
> PH-VAR-002 reproduces Demkowicz-Gopalakrishnan 2025.

## Function-labeled citation stack

Per complete-v1.0 plan §1.3 three-function labeling. Authored during
Task 13 of the complete-v1.0 plan on 2026-04-24.

### Mathematical-legitimacy (Tier 2 multi-paper)

The rule's existence is motivated by a multi-paper backbone
identifying that the parabolic variational-correctness framework
does not automatically extend to hyperbolic problems, and that
specialized DPG / Bochner-space machinery is required:

- **Bachmayr, M., Dahmen, W., Oster, M.** (2024). *Variationally
  correct residual-based norms.* The residual-based variational-
  correctness framework that grounds physics-lint's parabolic
  residual-norm rules. Physics-lint's extension to wave is the
  conjecture this rule flags. (Already cited on PH-RES-001; the
  hyperbolic gap is what PH-VAR-002 points out.)
- **Ernst, O.G., Sprungk, B., Tamellini, L.** (2025). *BDO-type
  variational-correctness sharper bounds.* Sharpens the BDO norm-
  equivalence framework in the parabolic regime. Does not cover
  hyperbolic problems.
- **Gopalakrishnan, J., Sepúlveda, P.** (2019). *A DPG method for
  the wave equation in Banach space-time settings.* DPG
  construction of well-posed discretization for the wave equation;
  constitutes one of the plausible routes to a future tight
  hyperbolic norm-equivalence statement.
- **Ernesti, M., Wieners, C.** (2019). *Space-time discontinuous
  Petrov–Galerkin methods for linear wave equations in
  heterogeneous media.* Companion DPG hyperbolic discretization with
  a posteriori error control.
- **Henning, P., Palitta, D., Simoncini, V., Urban, K.** (2022).
  *A reduced basis DPG method for the wave equation.* Reduced-
  basis + DPG for time-harmonic wave problems; another plausible
  anchor for a tightened V1.x PH-VAR-002.
- **Demkowicz, L., Gopalakrishnan, J.** (2010 / 2011). *A class of
  discontinuous Petrov–Galerkin methods.* Parts I–II. Original DPG
  framework; foundational backbone for all the above.

**Structural argument.** Under the parabolic variational-
correctness framework (Bachmayr-Dahmen-Oster 2024, Ernst et al.
2025), residual-norm equivalence holds between the trial-norm
`||u||_V` and the dual-residual norm `||R(u)||_{V'}` because the
parabolic operator admits a coercive a-symmetric bilinear form in
an appropriate Bochner space. The wave equation does **not** share
this property: the hyperbolic operator has conserved energy rather
than a dissipative coercive bound, and the corresponding norm-
equivalence result requires the DPG / optimal-test-function
machinery of Gopalakrishnan-Sepúlveda 2019 + Ernesti-Wieners 2019
+ Henning et al. 2022, or the newer Demkowicz-Gopalakrishnan 2025
Acta Numerica review. **Physics-lint's shipped V1 machinery for
wave-equation residual norms is therefore a conjectural extension
of the parabolic framework, not a proven hyperbolic norm-
equivalence result.** PH-VAR-002's info-flag exists to keep this
distinction visible in every wave-equation report.

All six Mathematical-legitimacy references above are arXiv
preprints or peer-reviewed journal papers with DOI-identifiable
mathematics; no textbook ⚠ status applies. Tight theorem-pointing
is permitted; physics-lint's V1 cites the papers at section-level
because the multi-paper aggregation is the claim, not any one
theorem.

### Correctness-fixture (CI-runnable, non-credibility-claiming)

`test_anchor.py` exercises the rule's diagnostic contract directly:

- **Wave PDE → PASS with info severity + DPG-literature reason.**
  Assert `status == "PASS"`, `severity == "info"`, reason string
  contains "Bachmayr-Ernst variational framework" +
  "conjectural" + "diagnostic, not certification."
- **Laplace / Poisson / heat PDE → SKIPPED with per-PDE reason.**
  Assert `status == "SKIPPED"`, reason string names the current PDE
  and "applies to wave only."
- **PhysicsLintReport aggregation invariant.** Because the rule is
  info-severity, its PASS must not move `PhysicsLintReport.exit_code`
  (checked via `PhysicsLintReport._STATUS_RANK` discipline in
  `src/physics_lint/report.py:53-68`). Test asserts a
  `PhysicsLintReport([wave-pass-result])` has `exit_code == 0`.

The F2 layer is a **contract-verification** fixture, not a
numerical-reproduction fixture. No floor, no tolerance, no
comparison against an analytical answer — the rule emits no number
to compare.

### Borrowed-credibility (external published reproduction layers)

**F3 absent by structure.** Info-flag rules emit no numerical
quantity against the field, so there is nothing to reproduce
against a published baseline. Per complete-v1.0 plan §1.2, F3-
absent is structural for this anchor class. The Borrowed-
credibility subsection is populated by this justification; no
citation is required here.

### Supplementary calibration context

**Theoretical framing only — NOT reproduction.** The following
reference names the current state of hyperbolic DPG methodology
and sits outside V1's conjectural parabolic-framework extension.
V1.1+ tightening of PH-VAR-002 (if/when the hyperbolic norm-
equivalence literature matures to a directly-computable target)
would promote content from here into F1 or F3.

- **Demkowicz, L., Gopalakrishnan, J.** (2025). *The discontinuous
  Petrov–Galerkin method.* Acta Numerica 34:293–384.
  **DOI:** [10.1017/S0962492924000102](https://doi.org/10.1017/S0962492924000102).
  Published online 2025-07-01 by Cambridge University Press.
  Publication status verified per
  `../../docs/audits/2026-04-22-pdebench-hansen-pins.md` Appendix
  (Task 0 Step 6, 2026-04-23). Reviews the state of DPG
  hyperbolic-problem methodology; framing reference for why
  PH-VAR-002 is conjectural-until-tightened in V1.x.

## Citation summary

- **Primary (mathematical-legitimacy)**: multi-paper DPG +
  variational-correctness stack — Bachmayr-Dahmen-Oster 2024,
  Ernst-Sprungk-Tamellini 2025, Gopalakrishnan-Sepúlveda 2019,
  Ernesti-Wieners 2019, Henning-Palitta-Simoncini-Urban 2022,
  Demkowicz-Gopalakrishnan 2010/2011.
- **Correctness-fixture (F2)**: `test_anchor.py` diagnostic-
  contract tests — PASS + info severity on wave; SKIPPED on
  Laplace / Poisson / heat; PhysicsLintReport aggregation
  invariant.
- **Borrowed-credibility (F3)**: **absent by structure** (info-
  flag rules emit no numerical quantity).
- **Supplementary calibration context**: Demkowicz-Gopalakrishnan
  2025 Acta Numerica DOI 10.1017/S0962492924000102.
- **Pinned value**: verdict-based — PASS + info severity on wave;
  SKIPPED on non-wave. No numerical value.
- **Verification date**: 2026-04-24 (Task 13 of complete-v1.0
  plan).
- **Verification protocol**: diagnostic-contract verification only.

## Scope note

PH-VAR-002 covers the wave-equation info-flag pathway in V1. The
following are **out of V1 scope** and remain open for V1.x work:

- **Numerical hyperbolic norm-equivalence tightening.** V1.x
  may promote PH-VAR-002 from info-flag to a numerical-comparison
  rule if/when the hyperbolic norm-equivalence literature matures
  to a directly-computable target (e.g., a DPG optimal-test-
  function residual norm with explicit constants).
- **Other conjectural rule paths.** PH-VAR-002's discipline (flag
  the conjectural extension, don't silently assume parabolic
  machinery covers hyperbolic) is specific to the wave-equation
  residual-norm path. Analogous info-flag rules for other
  conjectural extensions (e.g., variable-coefficient regimes where
  the variational framework hasn't been extended) would be new
  rules, not extensions of PH-VAR-002.
