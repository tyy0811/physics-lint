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
