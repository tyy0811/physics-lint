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

| Rule | GT (control arm) | MGN | D0-24 verdict |
|---|---|---|---|
| `PH-CON-001` mass / divergence-free | 5.857e-02 (5.857 %) | 5.881e-02 (5.881 %) | v1 PASS (GT ≤ 6 %); v2 PASS (gap = 0.41 % ≤ 20 %) |
| `PH-CON-002` energy drift | SKIP (open-driven-dissipative, D0-22 + D0-23 v9) | SKIP (same dispatch) | v3 PASS (substrate-class dispatch fires on both arms) |
| `PH-CON-003` dissipation sign violation | SKIP (same dispatch) | SKIP (same dispatch) | v4 PASS (same dispatch) |

See [`methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md`](../methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md) for the unified three-column cross-stack consistency table including the rung-4a LB-side columns (GNS-TGV2D + SEGNN-TGV2D).

### Scope qualifier — single trajectory, in-band subset

Phase 2 results report PH-CON-001 defect on a single cylinder_flow test trajectory (trajectory `M = 44`, Strouhal `St_U_max = 0.192` ∈ design band `[0.16, 0.21]`, cylinder diameter `D = 0.135`, inflow `U_max = 1.502`; Reynolds number not directly captured at Phase 2 — derivable as `Re = U · D / ν` from the cylinder_flow benchmark's `ν` if needed in Phase 3), selected via the Phase 2 pre-fire Strouhal audit (Task 1, 23 / 100 in-band) to be representative of the in-band subset under the pre-registered centerline-convention selection rule (median-`strouhal_U_max` among the 23 in-band trajectories). Trajectory `44` ranks 80th of 100 by `strouhal_U_max` on the full test set (the in-band subset occupies ranks 69-91; out-of-band trajectories sit at lower `strouhal_U_max` values, where the literature-anchored design band would not hold anyway). Coverage-not-statistics framing: physics-lint's value here is rule-firing on a real-world checkpoint, not a distribution over initial conditions. CI-gate threshold derivation from defect-magnitude distributions would require `N > 1` and is deferred (Phase 2 does NOT claim CI-gate calibration or full-distribution representativeness).

### Bridge to the cross-stack story

The numbers above land in the unified cross-stack table at [`methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md`](../methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md) as the third column (alongside `gns-tgv2d` and `segnn-tgv2d`). The schema-uniformity claim — *the same conservation rule schema runs unmodified across three upstream rollouts of two substrate classes* — is what Case Study 02 supplies to the rung-4 series's methodology trail per [`physics-lint-validation-plan-v2.1.md`](../methodology/docs/physics-lint-validation-plan-v2.1.md) §1.5 (Case Study 02 as a falsification surface). Whether the A + B + C triad generalizes is tested by the Phase 1 + Phase 2 + Phase 3 cross-review findings and triage; see [DECISIONS.md D0-23 + D0-24](../methodology/DECISIONS.md) for the per-pattern verdicts.

## Reproducibility

Modal entrypoint: `modal_app.py`. Inference script: `run_inference.py`.
Lint driver: `lint_rollouts.py` — invokes
`_rollout_anchors/_harness/mesh_rollout_adapter.py` (per-timestep
materialization) plus the public `physics-lint check` CLI per timestep.

Inference must pass `test_inference_matches_ngc_sample` (max-abs-error
≤ 10⁻³ on velocity components vs NGC's shipped sample) before rollouts
proceed; this is the gate-determining test for Gate D (spec §6).

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
