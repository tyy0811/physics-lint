# Case Study 02 — Unified cross-stack conservation table (writeup; extends rung-4a)

**Date:** 2026-05-13
**Predecessor:** [`./2026-05-04-rung-4a-cross-stack-conservation-table.md`](./2026-05-04-rung-4a-cross-stack-conservation-table.md) (LB-only two-column; preserved frozen at its commit date).
**Design doc:** [`./2026-05-11-case-study-02-physicsnemo-mgn-design.md`](./2026-05-11-case-study-02-physicsnemo-mgn-design.md) §3.3 activity 3.
**SARIF artifacts:** [`../../01-lagrangebench/outputs/sarif/`](../../01-lagrangebench/outputs/sarif/) (LB-side, rung-4a state) + [`../../02-physicsnemo-mgn/outputs/sarif/`](../../02-physicsnemo-mgn/outputs/sarif/) (CS02 Phase 2).
**Methodology pre-registrations:** [D0-19](../DECISIONS.md#d0-19--2026-05-04--harness-sarif-result-schema-rung-4a-pre-registration), [D0-20](../DECISIONS.md#d0-20--2026-05-04--generator-vs-consumer-separation-architecture-rung-4a-pre-registration), [D0-22](../DECISIONS.md), [D0-23](../DECISIONS.md), [D0-24](../DECISIONS.md).

---

## Headline

physics-lint's harness ran the same conservation rule schema, unmodified, across THREE upstream rollouts of TWO substrate classes: GNS-TGV2D + SEGNN-TGV2D (rung-4a, dissipative-isotropic) and PhysicsNeMo-MGN on cylinder_flow vortex shedding (CS02 Phase 2, open-driven-dissipative). Every result row's structural contract holds across all three columns under D0-19 + D0-20 enforcement; D0-22/D0-23 substrate-class dispatch fires SKIP-with-reason identically on `energy_drift` + `dissipation_sign_violation` on the CS02-side (D0-23 v9 open-driven-dissipative path; cited via D0-22 amendment 1 / D0-22 in the emitter skip_reason), with the LB-side (D0-18 dissipative path) carrying `SKIP (x20, D0-18)` on `energy_drift` and raw-value 0.0 on the other two rules; `mass_conservation_defect` raw-value renders LB-side at 0.000e+00 (TGV2D's exact mass conservation) and CS02 MGN at 5.881e-02 (the harness-FE-on-P1 floor on the canonical cylinder_flow trajectory; see `02-physicsnemo-mgn/README.md` "What physics-lint did NOT catch" §1 for the floor-bounds-resolution distinction). **Renderer-deterministic provenance is preserved across the extension**: any divergence between the rung-4a-era LB columns in this extended table vs the original rung-4a artifact reflects a SARIF change or a renderer change, caught by the golden test at `methodology/tests/test_render_cross_stack_table.py`.

---

## Unified cross-stack conservation table

| Rule | gns-tgv2d | segnn-tgv2d | modulus_ns_meshgraphnet-vortex_shedding_2d |
|---|---|---|---|
| `mass_conservation_defect` | 0.000e+00 (x20 identical) | 0.000e+00 (x20 identical) | 5.881e-02 (x1 identical) |
| `energy_drift` | SKIP (x20, D0-18) | SKIP (x20, D0-18) | SKIP (x1, D0-22 (amendment 1)) |
| `dissipation_sign_violation` | 0.000e+00 (x20 identical) | 0.000e+00 (x20 identical) | SKIP (x1, D0-22) |

**Provenance (D0-19 three-sha):**

- **gns-tgv2d**: pkl_inference=f48dd3f376, npz_conversion=f48dd3f376, sarif_emission=8e49339469
- **segnn-tgv2d**: pkl_inference=8c3d080397, npz_conversion=5857144, sarif_emission=8e49339469
- **modulus_ns_meshgraphnet-vortex_shedding_2d**: pkl_inference=n/a_cs02_no_pkl_stage, npz_conversion=n/a_cs02_no_conversion_stage, sarif_emission=a6fbd14

**Inference run status (rung-4c §9 review-gate fold-in):**

- **gns-tgv2d**: n/a (pre-salvage-tag-schema)
- **segnn-tgv2d**: n/a (pre-salvage-tag-schema)
- **modulus_ns_meshgraphnet-vortex_shedding_2d**: from_completed_inference

---

## What extending the table preserves

1. **Frozen-original discipline.** The rung-4a artifact's two-column table is preserved at its commit date as the LB-only state. This artifact extends with a third column WITHOUT replacing the rung-4a artifact (which retains regression-check value: if the extended renderer accidentally changes the LB columns, the original rung-4a artifact is the comparison baseline).

2. **Renderer-deterministic provenance.** The extended renderer (Phase 3 Task 2) ingests from both `01-lagrangebench/outputs/sarif/` and `02-physicsnemo-mgn/outputs/sarif/`; the `arm == 'gt-control'` filter excludes CS02's GT control-arm SARIF from cross-stack columns. The golden test pins the unified table's shape. Any drift between SARIFs and rendered table is caught.

3. **Cross-substrate schema reuse.** D0-19's run-level + result-level schema (v1.0) was sufficient for the CS02-onwards convention without parameterization. CS02 SARIFs carry the 10 LB-required fields with sentinel values (`lagrangebench_sha = 'n/a_cs02_physicsnemo'`) where LB-specific provenance does not apply; CS02-specific fields (`arm`, `case_study`, `physicsnemo_sha`, `rollout_contract`, `trajectory_index`, `inference_run_status`, etc.) extend the run-level metadata additively per D0-19's optional-field convention. No schema bump was required.

---

## What this table is NOT

1. **Not a cross-stack model comparison on the same substrate.** The LB columns are on TGV2D (dissipative-isotropic); the CS02 column is on cylinder_flow (open-driven-dissipative). The MGN column's `5.881e-02` mass-conservation-defect value is NOT directly comparable to the LB columns' `0.000e+00` values — they bind on different physics regimes. The table's value is the *schema-uniformity* claim, not a model-quality comparison.

2. **Not a CS02-internal writeup.** Case-study-internal results (PH-CON-001 routing per D0-03; scope-qualifier for the single-trajectory regime; floor-bounds-resolution distinction; substrate-variability finding) live at `02-physicsnemo-mgn/README.md`. This dated artifact's scope is cross-stack consistency.

3. **Not the integrating top-level README.** That composes when amendment 1 lands (Ahmed Body + the cross-stack table's fourth column), per the rung-4a precedent.

4. **Not a comparison on `dissipation_sign_violation`.** The LB-side reports raw-value `0.000e+00` (TGV2D's monotone-non-increasing KE produces zero dissipation-sign-violation defect at every step), while the CS02-side fires SKIP via the open-driven-dissipative dispatch (D0-22). The cells are not parallel in meaning — one is "rule evaluated, zero defect," the other is "rule did not evaluate, substrate class excluded by design." Both are valid outcomes under the unified harness.

---

## Rederivability

Rendered at physics-lint `feature/case-study-02-physicsnemo-mgn` (see Phase-3-close commit on this branch) via:

```bash
python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py \
    --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/ \
    --include-glob '*_tgv2d_8e49339469.sarif' \
    --sarif-dir external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/
```

Re-run at the same sha → identical output. Determinism golden-tested at `methodology/tests/test_render_cross_stack_table.py` (`test_renderer_unified_cs02_table_golden_output_matches_expected`).

---

## Predecessor → successor pointer

Predecessor `2026-05-04-rung-4a-cross-stack-conservation-table.md` is preserved frozen at its commit date (the LB-only artifact). This dated artifact is the canonical cross-stack table going forward; amendment 1 (Ahmed Body) extends with a fourth column when that case-study row lands.
