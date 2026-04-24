# PH-VAR-002 external-validation anchor

**Scope separation (read first):** PH-VAR-002 validates **a
diagnostic info-flag contract for the hyperbolic (wave-equation)
path of physics-lint**. The production rule (`ph_var_002.py`) does
not compute against the field; it emits an `info`-severity `PASS`
with a literature-pointer reason on wave-equation problems, and
`SKIPPED` on every other PDE kind. The anchor is therefore a
**contract-verification** test, not a numerical-reproduction test.

Per complete-v1.0 plan §21 Task 13 + user's 2026-04-24 revised Task
13 contract (F3 absent-with-justification is acceptable for info-
flag rules):

- **F1 Mathematical-legitimacy:** multi-paper DPG + variational-
  correctness stack — Bachmayr-Dahmen-Oster 2024 + Ernst et al.
  2025 (parabolic variational-correctness framework physics-lint's
  shipped rules rely on); Gopalakrishnan-Sepúlveda 2019 + Ernesti-
  Wieners 2019 + Henning-Palitta-Simoncini-Urban 2022 + Demkowicz-
  Gopalakrishnan 2010/2011 (plausible DPG routes to a future
  tightened hyperbolic norm-equivalence). The structural argument
  identifies that the parabolic framework does **not** extend to
  hyperbolic problems without specialized machinery; V1's wave-
  equation residual norms are therefore a **conjectural** extension
  of the parabolic framework, not a proven hyperbolic result.
- **F2 Correctness-fixture:** `test_anchor.py` exercises the
  rule's diagnostic contract: PASS + info severity + literature-
  pointer reason on wave PDE; SKIPPED with per-PDE reason on
  Laplace / Poisson / heat; PhysicsLintReport aggregation
  invariant (info-severity PASS does not move the exit code).
- **F3 Borrowed-credibility:** **absent by structure.** Info-flag
  rules emit no numerical quantity against the field, so there is
  nothing to reproduce against a published baseline.
- **Supplementary calibration context:** Demkowicz-Gopalakrishnan
  2025 Acta Numerica DOI 10.1017/S0962492924000102 — theoretical
  framing for why PH-VAR-002 is conjectural-until-tightened in
  V1.x.

**Wording discipline.** Do not write "PH-VAR-002 validates
hyperbolic norm-equivalence" or "PH-VAR-002 certifies wave-equation
residuals." The rule does neither. Write: "PH-VAR-002 emits an
info-severity diagnostic on wave-equation problems pointing users
to DPG hyperbolic norm-equivalence literature; it does not certify
hyperbolic norm-equivalence and does not compute against the
field."

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-VAR-002/ -v
```

Expected: 6 passed in < 1 s (1 PASS+INFO on wave + 3 SKIPPED on
laplace/poisson/heat + 1 PhysicsLintReport aggregation invariant +
1 wording-discipline guard).

## Citation stack (three-function-labeled)

See `CITATION.md` for the full function-labeled citation stack.
Summary:

- **F1 Mathematical-legitimacy** (Tier 2 multi-paper): Bachmayr-
  Dahmen-Oster 2024 + Ernst-Sprungk-Tamellini 2025 + Gopalakrishnan-
  Sepúlveda 2019 + Ernesti-Wieners 2019 + Henning-Palitta-
  Simoncini-Urban 2022 + Demkowicz-Gopalakrishnan 2010/2011.
- **F2 Correctness-fixture**: diagnostic-contract verification
  (PASS+info on wave; SKIPPED on non-wave; aggregation invariant).
- **F3 Borrowed-credibility**: **absent by structure** (info-flag
  rules emit no numerical quantity).
- **Supplementary calibration context**: Demkowicz-Gopalakrishnan
  2025 Acta Numerica.

## V1 scope-truth observation

Info-flag rules like PH-VAR-002 are **deliberately non-certifying**.
The rule exists to make the parabolic-framework-extends-conjecturally
-to-hyperbolic gap visible in user reports, not to close the gap.
A user who runs physics-lint on a wave-equation model and sees
"PH-VAR-002 PASS (info) — Hyperbolic norm-equivalence is not
established within the parabolic Bachmayr-Ernst variational
framework; treat the wave residual as diagnostic, not certification"
is receiving the rule's correct behavior. Closing the gap is a
V1.x tightening project that depends on the hyperbolic norm-
equivalence literature maturing.
