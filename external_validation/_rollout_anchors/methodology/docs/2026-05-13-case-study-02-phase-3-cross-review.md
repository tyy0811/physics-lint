# Case Study 02 — Phase 3 boundary Codex prose cross-review (round-codex-Phase3)

**Date:** 2026-05-14 (review fire); filename retains Phase-3 date (`2026-05-13`) per the Phase 2 / Phase 3 dated-artifact convention naming by phase, not by sub-fire.
**Reviewer voice:** Application-reviewer / Stuttgart-Munich-first-time-reader (parallel to round-prose-2 + round-code-1 prose dimension).
**Diff scope reviewed:** `0b97c14..4347abb` on `feature/case-study-02-physicsnemo-mgn` — Phase 3 Tasks 4-8 prose commits (`ba7942a`, `68fe470`, `486bb86`, `5828980`, `4347abb`); Tasks 1-3 (`2f6a360`, `4616809`, `79face7`) are infrastructure (renderer + new dated cross-stack artifact body, not under prose review).
**Files under review:**
- `external_validation/_rollout_anchors/02-physicsnemo-mgn/README.md`
- `external_validation/_rollout_anchors/methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md`
- `external_validation/_rollout_anchors/01-lagrangebench/README.md`
**Pre-registration:** Phase 3 design §3.3 review-methods + §4.3 acceptance criterion 8; A10G contingency budget pre-registered (≤1 fire ~$2) but not exercised by these findings.

---

## Five finding axes (per Phase 3 Task 9 pre-registration)

1. Overclaiming
2. Hidden assumptions
3. Narrative drift
4. Scope-qualifier completeness
5. Cross-stack-claim clarity

---

## Findings

### Finding 1 — README restates a retired max-abs Gate D as if it still gates rollout

- **Severity:** HIGH
- **Axis:** 1 = OVERCLAIM
- **Cell classification:** cell-1 (re-discovery; the retirement of the 10⁻³ max-abs criterion is already named in [D0-23](../DECISIONS.md))
- **Evidence:** `02-physicsnemo-mgn/README.md` "Inference-gate test" subsection states *"Inference must pass `test_inference_matches_ngc_sample` (max-abs-error ≤ 10⁻³ on velocity components vs NGC's shipped sample) before rollouts proceed; this is the gate-determining test for Gate D (spec §6). Gate D PASSED at Phase 1 with RMSE-1 = 3.19e-3 ≤ 5e-3"* — the prose conflates the original `10⁻³` max-abs gate (retired at D0-23 as a category error: no shipped NGC reference tensor to bind max-abs against) with the recalibrated RMSE-1 ≤ 5e-3 criterion that actually held the Phase 1 Gate D verdict.
- **Suggested absorption:** Prose edit. Replace the "Inference-gate test" subsection with a single paragraph clarifying that the original 10⁻³ max-abs criterion was retired at D0-23 and the Phase 1 gate-determining criterion is the RMSE-1 ≤ 5e-3 band (≈ 1.36× Pfaff et al. CylinderFlow RMSE-1 baseline) that Phase 1 satisfied with RMSE-1 = 3.19e-3.

### Finding 2 — Cross-stack table headline says "rule schema unmodified" where README correctly says PH-CON-001 is harness reapplication

- **Severity:** HIGH
- **Axis:** 5 = CROSS-STACK CLARITY
- **Cell classification:** cell-2 (novel-in-scope; in-rung prose contradiction not previously surfaced)
- **Evidence:** `2026-05-13-case-study-02-cross-stack-conservation-table.md` Headline section: *"physics-lint's harness ran the same conservation rule schema, unmodified, across THREE upstream rollouts of TWO substrate classes"*. CS02 README "PH-CON-001 routing" section correctly says: *"This is **structural-identity reapplication**, not 'rule ran without modification.'"* — the two prose claims are in tension. The cross-stack artifact's "unmodified" claim binds on result-row schema reuse (`mass_conservation_defect` / `energy_drift` / `dissipation_sign_violation` as the three pinned rule-ids in the SARIF output schema), NOT on public v1.0 rule-code reuse. A first-time reader reaches the cross-stack headline first and may carry that framing forward.
- **Suggested absorption:** Prose edit on cross-stack artifact's Headline. Replace *"the same conservation rule schema, unmodified"* with *"the same three-row conservation result-schema (rule-ids: `mass_conservation_defect`, `energy_drift`, `dissipation_sign_violation`)"* and add the clarifying sentence: *"The CS02 PH-CON-001 cell is a mesh-harness structural-identity reapplication; the schema-uniformity claim binds on result-row + run-level field reuse across stacks, NOT on public v1 rule-code reuse — see CS02 README "PH-CON-001 routing" for the routing detail."*

### Finding 3 — Floor-bounds correction is clear, but it lands after a PASS table that primes the wrong skim-read

- **Severity:** MEDIUM
- **Axis:** 1 = OVERCLAIM
- **Cell classification:** cell-2 (novel-in-scope; ordering issue not previously identified)
- **Evidence:** CS02 README results table at "## Rule × checkpoint results" reports `PH-CON-001 ... v1 PASS ... v2 PASS` for the 5.881e-02 cell, but the disambiguating *"What this demonstrates: 'MGN reproduces GT at the floor.' What this does NOT demonstrate: 'MGN is physically incompressible to 5%'"* fence lives ≈50 lines later in the "What physics-lint did NOT catch" §1. A cold reader skimming the results table reaches "MGN passes mass-conservation at 5%" before reaching the fence.
- **Suggested absorption:** Prose edit. Add a one-sentence inline caveat immediately under the results table (between the table and the scope-qualifier sub-section): *"PASS verdicts for PH-CON-001 mean MGN is within the GT/harness-FE-on-P1 floor envelope on this trajectory; they are NOT a claim of physical incompressibility to 5 % — see "What physics-lint did NOT catch" §1 for the floor-bounds-resolution distinction."*

### Finding 4 — "representative" overstates what the single-trajectory selector proves

- **Severity:** MEDIUM
- **Axis:** 4 = SCOPE-QUALIFIER COMPLETENESS
- **Cell classification:** cell-2 (novel-in-scope; word-choice overclaim absorbable in-rung)
- **Evidence:** CS02 README "Scope qualifier" section: *"selected via the Phase 2 pre-fire Strouhal audit (Task 1, 23 / 100 in-band) to be representative of the in-band subset under the pre-registered centerline-convention selection rule (median-`strouhal_U_max` among the 23 in-band trajectories)"*. The word *"representative"* is structurally ambiguous: a centerline-Strouhal-rank-80/100 trajectory is central in *one* coordinate but not necessarily representative of defect-magnitude distribution or other behavior. The same paragraph later says *"does NOT claim CI-gate calibration or full-distribution representativeness"* — the local representative claim and the global non-representativeness disclaimer rub against each other.
- **Suggested absorption:** Prose edit. Replace *"to be representative of the in-band subset"* with *"as the centerline-Strouhal-central member of the in-band subset"*.

### Finding 5 — First-time readers hit D-entry shorthand before the local meaning is introduced

- **Severity:** MEDIUM
- **Axis:** 2 = HIDDEN ASSUMPTIONS
- **Cell classification:** cell-2 (novel-in-scope; one-sentence glossary additions absorbable in-rung)
- **Evidence:** CS02 README's first results table cells say *"SKIP (open-driven-dissipative, D0-22 + D0-23 v9)"* and *"v3 PASS (substrate-class dispatch fires on both arms)"* before the substrate-class taxonomy or D0-22/D0-23 dispatch is explained anywhere on the page. The bridge sentence then references *"rung-4 series's methodology trail"* and the *"A + B + C triad"* without preface. A cold reader needs prior reading of v2.1 + DECISIONS to follow.
- **Suggested absorption:** Prose edit. Add a one-sentence glossary line above the results table: *"Terminology: `open-driven-dissipative` is the D0-22 substrate class whose PH-CON-002/003 assumptions fail by design; D0-23 v9 is the CS02 mesh-side substrate-class dispatch extension; `rung-4a` denotes the LB-only cross-stack-table predecessor."*

### Finding 6 — Cross-stack "NOT" fence is correct but appears after the model-quality comparison has been rhetorically staged

- **Severity:** LOW
- **Axis:** 5 = CROSS-STACK CLARITY
- **Cell classification:** cell-2 (novel-in-scope; lightly absorbable ordering fix)
- **Evidence:** Cross-stack artifact Headline foregrounds the asymmetry: *"`mass_conservation_defect` raw-value renders LB-side at 0.000e+00 (TGV2D's exact mass conservation) and CS02 MGN at 5.881e-02"* — the numeric comparison primes a model-quality read. The "What this table is NOT" §1 fence later says *"The MGN column's `5.881e-02` mass-conservation-defect value is NOT directly comparable to the LB columns' `0.000e+00` values"* — correct but downstream of the prime.
- **Suggested absorption:** Prose edit. Add a single sentence to the Headline immediately after the numeric comparison: *"These two numbers are NOT directly comparable as model quality — they bind on different substrate classes (dissipative-isotropic vs open-driven-dissipative). The table's value is the schema-uniformity claim, not a model-quality comparison; see "What this table is NOT" §1 for the framing."*

---

## Triage summary

| # | Finding | Severity | Cell | Trigger classification | Absorption shape |
|---|---|---|---|---|---|
| 1 | Inference-gate test conflates retired max-abs with current RMSE-1 | HIGH | cell-1 | Prose-level absorbable | Edit "Inference-gate test" subsection |
| 2 | Cross-stack headline "rule schema unmodified" vs README "structural-identity reapplication" | HIGH | cell-2 | Prose-level absorbable | Edit cross-stack Headline + add clarifying sentence |
| 3 | Floor-bounds correction lands after PASS table primes wrong skim-read | MEDIUM | cell-2 | Prose-level absorbable | Add inline caveat under results table |
| 4 | "representative" overstates the single-trajectory selector | MEDIUM | cell-2 | Prose-level absorbable | Word-replacement edit |
| 5 | D-entry shorthand precedes local meaning introduction | MEDIUM | cell-2 | Prose-level absorbable | Add one-sentence glossary |
| 6 | Cross-stack NOT fence ordered after model-quality framing | LOW | cell-2 | Prose-level absorbable | Add forward-cite sentence to Headline |

**Totals:** 1 cell-1 (re-discovery; absorption sharpens phrasing), 5 cell-2 (novel-in-scope; absorption in-rung), 0 cell-3 (out-of-Phase-3), 0 cell-4 (genuinely-new-framing).

**A10G budget:** UNUSED. All 6 findings are prose-level absorbable; no SARIF data-level finding and no fresh-inference-need finding surfaced. The pre-registered ≤1 A10G contingency stays untouched. Per [[feedback_cap_rationale_not_literal]]: cap binds by rationale (no GPU-budget activation justified by findings), not by literal text.

**Meta-summary** (reviewer's voice): *"The dominant failure mode is not missing caveats; the writeup has most of the right caveats. The problem is ordering and overloaded shorthand: first-screen tables and headlines make compact PASS/schema claims, while the disambiguating prose often arrives later or in another file. A cold application reviewer can therefore skim into 'MGN passes incompressibility,' 'same rule ran unmodified,' or '0 vs 5.881 compares model quality' before the document corrects them."*

The pattern (ordering / forward-cite hygiene; reader-first cross-reference placement) is a fifth empirical instance of the prose-cross-review-vs-artifact-cross-review distinction's prose-side findings character — first-impression weaknesses rather than fail-open paths. This is the kind of finding [[project_cs02_phase3_plan_ready_handoff]]'s Q4b Variant i linear sequence anticipated and budgets for in-rung absorption.

---

## Phase 3 absorption plan (Task 10 input)

All 6 findings absorb prose-only in Task 10 commits. Recommended commit grouping:

- **Commit A** — CS02 README absorptions (findings 1, 3, 4, 5)
- **Commit B** — Cross-stack-table artifact absorptions (findings 2, 6)
- **Commit C** — DECISIONS.md D0-24 Phase 3 absorption table update (parallel to Phase 2's 5-finding triage table at lines 2501-2511)

No SARIF re-emission; no Modal fire; no `gt.sarif` / `mgn.sarif` regen; no golden-test churn (the table cells are unchanged, only the prose around them).
