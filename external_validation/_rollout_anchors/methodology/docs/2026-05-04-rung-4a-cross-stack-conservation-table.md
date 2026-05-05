# Rung 4a — Cross-stack conservation table (writeup)

**Date:** 2026-05-04
**Predecessor:** rung 3.5 PASS on both stacks (D0-18 amendment 1 implementation at `d03df3e`); npzs frozen on Modal Volume.
**Successor:** rung 4b — equivariance brainstorm session (separate, no code).
**Design doc:** [`./2026-05-04-rung-4a-cross-stack-conservation-design.md`](./2026-05-04-rung-4a-cross-stack-conservation-design.md)
**SARIF artifacts:** [`../../01-lagrangebench/outputs/sarif/`](../../01-lagrangebench/outputs/sarif/)
**Methodology pre-registrations:** [D0-19](../DECISIONS.md#d0-19--2026-05-04--harness-sarif-result-schema-rung-4a-pre-registration), [D0-20](../DECISIONS.md#d0-20--2026-05-04--generator-vs-consumer-separation-architecture-rung-4a-pre-registration)

---

## Headline

physics-lint's harness ran the same conservation rule schema, unmodified, across SEGNN-TGV2D and GNS-TGV2D rollouts of the same dissipative system. Every result row is structurally identical between the two SARIF artifacts (D0-19-enforced); D0-18's dissipative-system skip-with-reason fires identically with the same `skip_reason` string on both — per-stack KE endpoints are recorded in dedicated `properties.ke_initial` / `properties.ke_final` fields, not interpolated into the reason — and points to `dissipation_sign_violation` as the load-bearing alternative. The methodology-evolution machinery — D0-18's skip-with-reason path — is exercised end-to-end against real upstream output.

The "20 identical fires" claim above is schema-enforced, not coincidental: D0-19 §3.4 specifies that for a fixed (rule, stack), all 20 result rows MUST have identical `ruleId`, `level`, `message.text`, plus either identical `properties.raw_value` or identical `properties.skip_reason`.

---

## Cross-stack conservation table

| Rule | gns-tgv2d | segnn-tgv2d |
|---|---|---|
| `mass_conservation_defect` | 0.000e+00 (x20 identical) | 0.000e+00 (x20 identical) |
| `energy_drift` | SKIP (x20, D0-18) | SKIP (x20, D0-18) |
| `dissipation_sign_violation` | 0.000e+00 (x20 identical) | 0.000e+00 (x20 identical) |

**Provenance (D0-19 three-sha):**

- **gns-tgv2d**: pkl_inference=f48dd3f376, npz_conversion=f48dd3f376, sarif_emission=5ed9fa3009
- **segnn-tgv2d**: pkl_inference=8c3d080397, npz_conversion=5857144, sarif_emission=5ed9fa3009

---

## What rung 4a is NOT

1. **Not a SEGNN-vs-GNS model comparison.** Both stacks emit `mass_conservation_defect = 0.0`, both fire D0-18 SKIP on `energy_drift`, both emit `dissipation_sign_violation = 0.0`. Model differentiation lives in equivariance → rung 4b (separate brainstorm, separate session).

2. **Not a GitHub Security-tab integration demo.** Harness-style SARIF emits `level: "note"` rows for PASS-equivalent values; 4a has no findings to populate the Security tab meaningfully. The Security-tab demo is deferred to 4b, where equivariance is expected to produce real warning-level findings (GNS APPROXIMATE band) that exercise the rendering path. An empty Security tab is not a demo of integration.

3. **Not the integrating top-level README.** Composed when 4b's writeup lands; until then `methodology/docs/` carries dated deliverables and is the source of truth.

4. **Not a physics-lint v1.x core change.** The skip-with-reason mechanism, dissipative-system detection, and audit-trail provenance fields all live in the harness layer (`external_validation/_rollout_anchors/_harness/`), not in physics-lint v1.0's public rule path. v1.0's `master`-branch docs are amended as part of 4a to document the dissipative-system limit explicitly alongside the existing PH-BC-001 / PH-RES-001 honest limits, with wording that includes an explicit cross-branch qualifier (the harness layer currently lives on `feature/rollout-anchors` pending merge to `master`). v1.0's behavior on dissipative systems is preserved as-shipped, with the harness-layer skip-with-reason machinery flagged as the v1.x graduation prototype. The graduation itself is a future D-entry, not implied by 4a.

5. **Not a bilateral test of D0-18's mechanism.** TGV2D is dissipative, so 4a exercises the skip-fires path. The opposite path (conservative system, skip does not fire, `energy_drift` evaluates raw_value normally) is not exercised — both 4a stacks are on the same dissipative dataset. Bilateral validation requires a conservative-system anchor (case study 02 if PhysicsNeMo includes a conservative target, or a dedicated future case study). 4a also does not exercise the borderline case — a system that *should* be conservative but is *numerically* dissipating due to a model bug, where D0-18's heuristic (dataset-name primary, KE-monotone-decreasing secondary) could mis-classify as dissipative and silently skip the very PH-CON-002 firing that would catch the bug. Diagnostic gap flagged for future case studies.

---

## Rederivability

Rendered at physics-lint `feature/rollout-anchors` sha `5ed9fa3009` via:

```bash
python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py \
    --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/
```

Re-run the command at the same sha with the committed SARIFs at that sha → identical output. The renderer's output is deterministic; any divergence reflects a SARIF artifact change, a renderer change, or both — all three cases are caught by the golden test in `methodology/tests/test_render_cross_stack_table.py`.

---

## Integrating-README trigger

This dated writeup at `methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md` is one of a planned series of dated deliverables under `methodology/docs/`. The integrating top-level README — composing 4a's writeup with rung 4b's equivariance writeup and any subsequent rungs — is composed when rung 4b's writeup lands. Until then, this `docs/` directory is the source of truth in dated-deliverable form.
