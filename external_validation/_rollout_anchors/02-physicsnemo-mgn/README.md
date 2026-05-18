# Case Study 02 — NVIDIA PhysicsNeMo MeshGraphNet

*Day 2 deliverable; substituted by FNO-on-Darcy under spec §6 Gate D
fallback (folder renamed to `02-fno-darcy/` if Gate D triggers).*

## Targets

| Priority | Checkpoint | Domain | Headline rule |
|---|---|---|---|
| P0 | `modulus_ns_meshgraphnet` (vortex shedding 2D) | Incompressible NS, cylinder wake | `PH-CON-001` (mass / divergence-free) + `PH-CON-002`/`003` |
| P1 | `modulus_ahmed_body_meshgraphnet` | Steady RANS, car-like geometry | `PH-BC-001` (no-slip) |
| P2 stretch | `modulus_ns_meshgraphnet` | Same as P0 | `PH-RES-001` (BDO momentum residual) — only if Day 2 hour 4 leaves ≥3h buffer |

`PH-NUM-002` resolution sweep is deferred to v1.1 backlog (spec §1.2).

## Rule × checkpoint results

P0 results for `modulus_ns_meshgraphnet` on cylinder_flow vortex shedding (Phase 2 fires on canonical trajectory M = 44 — see scope qualifier below). SARIFs at [`outputs/sarif/`](outputs/sarif/) (`gt.sarif` = ground-truth control arm; `mgn.sarif` = MGN model under test). D0-24 verdict bands pre-registered before fires; all 7 verdicts pinned PASS at [DECISIONS.md D0-24](../methodology/DECISIONS.md).

*Terminology used in the table below:* `open-driven-dissipative` is the [D0-22](../methodology/DECISIONS.md) substrate class whose `PH-CON-002` / `PH-CON-003` assumptions fail by design (kinetic-energy budget is not strictly dissipative and not strictly conservative; cylinder-wake flow imports KE from inflow and dissipates in the wake); `D0-23 v9` is the CS02 mesh-side substrate-class dispatch extension that fires the SKIP-with-reason path on this class; `rung-4a` denotes the LB-only cross-stack-table predecessor (`gns-tgv2d` + `segnn-tgv2d`).

**Read this first — N = 1.** The table below reports PH-CON-001 on a *single* cylinder_flow trajectory (canonical `M = 44`), not a distribution. The `0.41 %` MGN/GT gap is a **coverage** result — physics-lint fired its rule on a real published checkpoint — *not* a statistical or CI-gate-calibration claim. It also sits inside the ~5.8 % harness-FE-on-P1 discretization floor: it shows MGN reproduces GT *at the floor*, not that MGN is physically incompressible to 5 %. Full scope treatment in the "Scope qualifier" and "What physics-lint did NOT catch §1" sections below; this fence is hoisted above the table deliberately, so the number is never read without it.

| Rule | GT (control arm) | MGN | D0-24 verdict |
|---|---|---|---|
| `PH-CON-001` mass / divergence-free | 5.857e-02 (5.857 %) | 5.881e-02 (5.881 %) | v1 PASS (GT ≤ 6 %); v2 PASS (gap = 0.41 % ≤ 20 %) |
| `PH-CON-002` energy drift | SKIP (open-driven-dissipative, D0-22 + D0-23 v9) | SKIP (same dispatch) | v3 PASS (substrate-class dispatch fires on both arms) |
| `PH-CON-003` dissipation sign violation | SKIP (same dispatch) | SKIP (same dispatch) | v4 PASS (same dispatch) |

*PASS verdicts for `PH-CON-001` mean MGN is within the GT / harness-FE-on-P1 floor envelope on this trajectory; they are NOT a claim of physical incompressibility to 5 % — see "What physics-lint did NOT catch" §1 for the floor-bounds-resolution distinction.*

See [`methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md`](../methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md) for the unified three-column cross-stack consistency table including the rung-4a LB-side columns (GNS-TGV2D + SEGNN-TGV2D).

### Scope qualifier — single trajectory, in-band subset

Phase 2 results report PH-CON-001 defect on a single cylinder_flow test trajectory (trajectory `M = 44`, Strouhal `St_U_max = 0.192` ∈ design band `[0.16, 0.21]`, cylinder diameter `D = 0.135`, inflow `U_max = 1.502`; Reynolds number not directly captured at Phase 2 — derivable as `Re = U · D / ν` from the cylinder_flow benchmark's `ν` if needed in Phase 3), selected via the Phase 2 pre-fire Strouhal audit (Task 1, 23 / 100 in-band) as the centerline-Strouhal-central member of the in-band subset under the pre-registered selection rule (median-`strouhal_U_max` among the 23 in-band trajectories). Trajectory `44` ranks 80th of 100 by `strouhal_U_max` on the full test set (the in-band subset occupies ranks 69-91; out-of-band trajectories sit at lower `strouhal_U_max` values, where the literature-anchored design band would not hold anyway). Coverage-not-statistics framing: physics-lint's value here is rule-firing on a real-world checkpoint, not a distribution over initial conditions. CI-gate threshold derivation from defect-magnitude distributions would require `N > 1` and is deferred (Phase 2 does NOT claim CI-gate calibration or full-distribution representativeness).

### Bridge to the cross-stack story

The numbers above land in the unified cross-stack table at [`methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md`](../methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md) as the third column (alongside `gns-tgv2d` and `segnn-tgv2d`). The schema-uniformity claim — *the same three-row conservation result-schema (rule-ids: `mass_conservation_defect`, `energy_drift`, `dissipation_sign_violation`) and run-level field set reuse across three upstream rollouts of two substrate classes, NOT public v1 rule-code reuse* (the CS02 PH-CON-001 cell is mesh-harness structural-identity reapplication; see "PH-CON-001 routing — harness, not public rule" below) — is what Case Study 02 supplies to the rung-4 series's methodology trail per [`physics-lint-validation-plan-v2.1.md`](../methodology/docs/physics-lint-validation-plan-v2.1.md) §1.5 (Case Study 02 as a falsification surface). Whether the A + B + C triad generalizes is tested by the Phase 1 + Phase 2 + Phase 3 cross-review findings and triage; see [DECISIONS.md D0-23 + D0-24](../methodology/DECISIONS.md) for the per-pattern verdicts.

## Reproducibility

### Modal entrypoints (Phase 1 + Phase 2 fires)

| Entrypoint | Purpose | Compute |
|---|---|---|
| `02-physicsnemo-mgn/modal_app.py::audit_ngc_sample_reproduction` | Phase 1 Gate D — NGC sample reproduction RMSE vs Pfaff et al. CylinderFlow RMSE-1 baseline (verdict 4 + 5) | A10G, ~10 min |
| `02-physicsnemo-mgn/modal_app.py::audit_gate_a_pyg_to_meshfield` | Phase 1 Gate A — PyG-to-MeshField materialization smoke (verdict 2) | CPU, <1 min |
| `02-physicsnemo-mgn/modal_app.py::smoke_substrate_class_vortex_shedding` | Phase 1 substrate-class smoke (verdicts 6 + 7 — ∫∇·v dV / KE budget / Strouhal) | A10G, ~5 min |
| `02-physicsnemo-mgn/modal_app.py::audit_strouhal_test_trajectories` | Phase 2 Task 1 — Strouhal pre-check across cylinder_flow test trajectories (refinement 1) | CPU, ~10 min |
| `02-physicsnemo-mgn/modal_app.py::lint_gt_trajectory` | Phase 2 Task 5 — GT-trajectory CPU lint → `gt.sarif` (control arm) | CPU, ~3 min |
| `02-physicsnemo-mgn/modal_app.py::mgn_rollout_p0_vortex_shedding` | Phase 2 Task 6 — MGN inference on canonical trajectory 44 (599 rollout steps) | A10G, ~10 min |
| `02-physicsnemo-mgn/modal_app.py::lint_mgn_rollout` | Phase 2 Task 7 — MGN-rollout CPU lint → `mgn.sarif` | CPU, ~3 min |

### NGC checkpoint provenance

| Field | Value |
|---|---|
| `checkpoint_id` | `0153803c8b2c0947` |
| `ngc_version` | `latest` |
| `ckpt_sha256` | `0153803c8b2c0947948757a2298eec6ef21e8ea28131834963db08f34b4a0726` |
| `physicsnemo_sha` | `1ca85d65ac2ce28ea9762910c09a954c08a37140` |

Adapter: `_legacy_checkpoint_name_remap.py` (Phase 1 Gate D fix) — encoder/decoder `.mlp.` → `.model.` rename + parallel-`{edge,node}_blocks` → interleaved-`processor_layers` restructure + edge-MLP first-Linear input-column reorder `[src, dst, edge]` → `[edge, src, dst]` (the bug found at Gate D Band-C re-audit; see [DECISIONS.md D0-23](../methodology/DECISIONS.md)).

### Phase-3-close git_sha

`d5b0983` (pre-Task-14 final-prose sha; the Task 14 close commit itself supersedes by one).

### Image spec / dependency pinning

`02-physicsnemo-mgn/modal_app.py` builds the Modal image with: `physicsnemo @ 1ca85d65ac2ce28ea9762910c09a954c08a37140`, `torch == 2.10.x`, `torch-scatter`, `scikit-fem == 12.0.1`, `dgl` removed (see [DECISIONS.md D0-23](../methodology/DECISIONS.md) for the image-build history that resolved the Gate D image-side failures).

### Cross-references — methodology

- **Validation plan v2.1**: [`methodology/docs/physics-lint-validation-plan-v2.1.md`](../methodology/docs/physics-lint-validation-plan-v2.1.md) — §1.5 Case Study 02 as a falsification surface (open prediction; this README closes Phase 3 of that prediction).
- **Validation plan v2.1.1**: [`methodology/docs/physics-lint-validation-plan-v2.1.1.md`](../methodology/docs/physics-lint-validation-plan-v2.1.1.md) — Phase 3 amendment landing the four §5.5 items (pattern-C 4th instance + per-section convergence + D0-22 falsification + prose-vs-artifact-cross-review-modes).
- **Methodology README**: [`methodology/README.md`](../methodology/README.md) — rung-4 series integrating README (composing 4a + 4b + 4c + case-study artifacts).
- **DECISIONS catalogue**: [`methodology/DECISIONS.md`](../methodology/DECISIONS.md) — D0-22 (substrate-class taxonomy) · D0-23 (Phase 1 audit verdicts) · D0-24 (Phase 2 + Phase 3 audit verdicts; this case study's authoritative numbers).
- **SCHEMA**: [`_harness/SCHEMA.md`](../_harness/SCHEMA.md) — SARIF run-level + result-level contract; D0-19 v1.0 already accommodates CS02-side optional fields (`arm`, `case_study`, `physicsnemo_sha`, `rollout_contract`, `trajectory_index`, `inference_run_status`, `ckpt_sha256`, `n_rollout_steps`, `ngc_version`, `rollout_npz_path`).

### Inference-gate test

The original spec §6 Gate D criterion was a `max-abs-error ≤ 10⁻³` reproduction check vs an NGC-shipped reference tensor. D0-23 **retired** that criterion as a category error: the NGC checkpoint does not ship a paired reference tensor against which a max-abs reproduction error is well-defined. The Phase 1 Gate D verdict was held by the recalibrated `RMSE-1 ≤ 5e-3` band (≈ 1.36× Pfaff et al. CylinderFlow RMSE-1 baseline), which Phase 1 satisfied with `RMSE-1 = 3.19e-3`. The original `test_inference_matches_ngc_sample` test name in the test suite predates the recalibration and refers to the same RMSE-1 check post-D0-23. See [D0-23 verdict 4 recalibrated band](../methodology/DECISIONS.md) for the full audit trail.

## PH-CON-001 routing — harness, not public rule

Per `physics-lint-validation/DECISIONS.md` D0-03 (2026-05-04 audit),
`PH-CON-001` as shipped in physics-lint v0.0.0.dev0 returns SKIPPED on
`pde != "heat"`. NS data is `pde = "navier_stokes"` (or analogue), so
the public-API rule cannot be invoked directly on NS rollouts. The
mesh case study therefore routes `PH-CON-001` through the mesh harness
in the same way the LagrangeBench-side particle adapter reapplies the
PH-SYM-001/002 structural identities — *the structural mass-conservation
identity (∫ρ over the domain, ∇·v on incompressible NS) is reapplied by
the harness, validated against the analytical mass-conservation fixture*
(see `_rollout_anchors/_harness/tests/fixtures/mass_conservation_fixture.py`).

This is **structural-identity reapplication**, not "rule ran without
modification." See the matching bullet in `_rollout_anchors/README.md`
"What physics-lint did NOT catch" and the v3 plan §6 risk-register
class-level entry on V1 rules with input-domain restrictions.

## What physics-lint did NOT catch

This section names the limits of what Phase 2's PH-CON-001/002/003 fires demonstrate, so the writeup's narrow claims are not over-read into broader ones they do not support.

### 1. Floor-bounds-resolution — PH-CON-001 at this discretization

Phase 2 fires PH-CON-001 with `raw_value = 5.881e-02` (5.881 %) for MGN and `raw_value = 5.857e-02` (5.857 %) for the GT control arm. The harness-FE-on-P1 floor is `≈ 5.8 %` on this trajectory (`≈ 5 %` on Phase 1 substrate-smoke's auto-selected trajectory) and bounds PH-CON-001's discriminating resolution at this discretization (test-trajectory mesh; `N_nodes = 1787`, `N_cells = 3340`; P1 basis). The MGN/GT gap of 0.41 % sits well inside the harness-floor envelope.

What this demonstrates: **"MGN reproduces GT at the floor."**

What this does NOT demonstrate: **"MGN is physically incompressible to 5 %."**

The two claims are distinct. PH-CON-001 at this discretization bounds MGN's deviation from GT-equivalence rather than from physical incompressibility. A tighter discretization would be needed to distinguish whether the `5 %` reflects MGN model error or floor error. Phase 3 holds this distinction explicitly to avoid the over-read; tighter-discretization re-fire is out of Phase 3 scope (would be empirical work, properly scoped to v1.1 backlog or a future case study).

### 2. Substrate-variability — single-trajectory + in-band subset

Phase 2's pre-fire Strouhal audit (Task 1) found 23 / 100 test trajectories land in the literature-anchored design band `[0.16, 0.21]` on the either-or convention check; 77 / 100 land out-of-band on the same check (mostly with `St_U_mean > 0.21` while `St_U_max ∈ [0.16, 0.21]` — the U-convention ambiguity). The canonical trajectory selection (median-`strouhal_U_max` among the in-band 23) is deterministic and reproducible. This is a substrate-variability characterization, NOT a methodology failing — physics-lint's value is rule-firing on a real-world checkpoint, not a distribution over initial conditions. CI-gate threshold derivation from defect-magnitude distributions would require `N > 1` and is deferred.

### 3. PH-CON-001 routing — harness, not public rule (class-level: V1-rules-with-input-domain-restrictions)

PH-CON-001 as shipped in physics-lint v0.0.0.dev0 returns SKIPPED on `pde != "heat"` (per [DECISIONS.md D0-03](../methodology/DECISIONS.md)). The mesh case study routes PH-CON-001 through the mesh harness as **structural-identity reapplication** — the structural mass-conservation identity (∫ρ over the domain; ∇·v on incompressible NS) is reapplied by the harness, validated against the analytical mass-conservation fixture at [`_harness/tests/fixtures/mass_conservation_fixture.py`](../_harness/tests/fixtures/mass_conservation_fixture.py). This is NOT "rule ran without modification." The class-level pattern (V1 rules with input-domain restrictions; the harness reapplies the structural identity rather than the v1.0 rule code itself) is the load-bearing methodology claim; see `physics-lint-validation-plan-v3.md` §6 risk-register class-level entry on V1 rules with input-domain restrictions for the cross-rung pattern catalogue.

### 4. PH-NUM-002 multi-resolution → v1.1 backlog

Cylinder_flow comes at a single mesh resolution per trajectory; PH-NUM-002 (which depends on a resolution sweep) is not exercised in Phase 2. Deferred to v1.1 backlog per spec §1.2.

### 5. PH-SYM-* not on mesh side

PH-SYM-001/002/003/004 (rotational / reflection / translation / Galilean equivariance) are scoped particle-side only per spec §1.2; the mesh side does not exercise them. Not deferred, out of scope.

### 6. Ahmed Body + PH-RES-001 deferred to amendments

Ahmed Body + PH-BC-001 (no-slip on car-like geometry) → deferred to amendment 1 per [design §5.1](../methodology/docs/2026-05-11-case-study-02-physicsnemo-mgn-design.md) (different physics regime; prime pattern-B candidate; BLOCKING-2 raw-data availability question to resolve). PH-RES-001 (BDO momentum residual) → deferred to amendment 2 OR case study 03 per design §5.2.

### 7. v2.1.2 §1.4 forward-flag — prose-scope-qualifier walls

The floor-bounds-resolution distinction above is the FOURTH empirical instance of physics-lint's prose-scope-qualifier discipline (after round-code-1's three walls — level rendering, file-path location, prose framing). The cumulative pattern is strong enough to formalize as a v2.1.2 §1.4 methodology entry. Tracked separately at `physics-lint-validation-plan-v2.1.2.md` (forthcoming); does NOT block Phase 3 close.
