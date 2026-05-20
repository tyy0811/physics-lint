# Case Study 02 — NVIDIA PhysicsNeMo MeshGraphNet

*Day 2 deliverable; substituted by FNO-on-Darcy under spec §6 Gate D
fallback (folder renamed to `02-fno-darcy/` if Gate D triggers).*

## Case study reference

CS02 validates physics-lint against NVIDIA's PhysicsNeMo MeshGraphNet (MGN)
checkpoint for 2D cylinder vortex shedding. The substrate is incompressible
Navier-Stokes on a mesh; the rule set fired is PH-CON-001 (mass conservation),
with PH-CON-002 and PH-CON-003 correctly emitting `SKIP` via the
`open-driven-dissipative` substrate-class dispatch (D0-22 + D0-23 v9 — KE is
neither strictly dissipative nor strictly conservative on a cylinder-wake
flow that imports KE from inflow and dissipates in the wake).

**Headline result.** PH-CON-001 fires across 5 in-band trajectories sampled
from the 23 in-band members (Strouhal in [0.16, 0.21]) of the Phase-2
pre-fire audit. The median MGN/GT mass-conservation gap is `-0.36%` of GT,
range `[-1.07%, +0.41%]`; every per-trajectory gap sits inside the ~5.8%
harness-FE-on-P1 discretization floor. The canonical trajectory-44 pair
lands at 5.857% (GT) / 5.881% (MGN). All 7 D0-24 verdicts PASS.

**What the result demonstrates and does NOT demonstrate.** PASS verdicts
for PH-CON-001 mean MGN is within the GT / harness-FE-on-P1 floor envelope
on that trajectory; they are **NOT** a claim of physical incompressibility
to 5%. PH-CON-001 at this discretization bounds MGN's deviation from
GT-equivalence rather than from physical incompressibility — distinguishing
the two would require a tighter discretization (deferred to v1.x; see
"What physics-lint did NOT catch §1" below for the floor-bounds-resolution
distinction). The result is a small-N statistical characterization across
the in-band subset, not a full-distribution claim or a CI-gate-threshold
derivation.

**Scope qualifier — PH-BC-001 no-slip is structurally inapplicable.**
[D0-27][D0-27-cs02-link]: the CS02 inference protocol masks
boundary nodes during rollout (`v_diff_masked = torch.where(mask2,
pred_i_velo, zeros)`), freezing wall-node velocities at their step-0
ground-truth value of zero. A no-slip check on this rollout computes
`||v_wall|| ~ 0` and PASSes by construction, detecting nothing about the
surrogate. P2.2 retired the planned mesh wall-node BC capability-build
on this finding. PH-SYM-001/002/003/004 are scoped particle-side only per
spec §1.2 and are not exercised on the mesh side.

**PH-CON-001 routing — harness, not public rule.** PH-CON-001 as shipped
in physics-lint v1.0 returns `SKIPPED` on `pde != "heat"`. CS02 routes
PH-CON-001 through the mesh harness as **structural-identity reapplication**:
the mass-conservation identity (∫ρ over the domain, ∇·v on incompressible
NS) is reapplied by the harness, validated against the analytical mass-
conservation fixture. This is NOT "rule ran without modification" — it is
the class-level pattern for V1 rules with input-domain restrictions, and
the load-bearing methodology claim of CS02.

The detailed validation harness, the per-trajectory table, the cross-stack
table integrating with rung-4a/4b, and the full "what physics-lint did NOT
catch" enumeration are below.

[D0-27-cs02-link]: https://github.com/tyy0811/physics-lint/blob/master/external_validation/_rollout_anchors/methodology/DECISIONS.md

## Targets

| Priority | Checkpoint | Domain | Headline rule |
|---|---|---|---|
| P0 | `modulus_ns_meshgraphnet` (vortex shedding 2D) | Incompressible NS, cylinder wake | `PH-CON-001` (mass / divergence-free) + `PH-CON-002`/`003` |
| P1 | `modulus_ahmed_body_meshgraphnet` | Steady RANS, car-like geometry | `PH-BC-001` (no-slip) — *retired from P2.2: Outcome N, no body-surface velocity output; see §8 below + [DECISIONS.md D0-27](../methodology/DECISIONS.md)* |
| P2 stretch | `modulus_ns_meshgraphnet` | Same as P0 | `PH-RES-001` (BDO momentum residual) — only if Day 2 hour 4 leaves ≥3h buffer |

`PH-NUM-002` resolution sweep is deferred to v1.1 backlog (spec §1.2).

## Rule × checkpoint results

P0 results for `modulus_ns_meshgraphnet` on cylinder_flow vortex shedding (P2.1 fires across 5 in-band trajectories; trajectory 44 remains the canonical cross-stack member — see scope qualifier below). Per-trajectory SARIFs for the N=5 table are at [`outputs/p2_multi_trajectory/`](outputs/p2_multi_trajectory/) (`traj<T>_gt.sarif` = ground-truth control arm; `traj<T>_mgn.sarif` = MGN model under test); the Phase-2 canonical traj-44 pair is also kept at [`outputs/sarif/`](outputs/sarif/). D0-24 verdict bands pre-registered before fires; all 7 verdicts pinned PASS at [DECISIONS.md D0-24](../methodology/DECISIONS.md).

*Terminology used in the table below:* `open-driven-dissipative` is the [D0-22](../methodology/DECISIONS.md) substrate class whose `PH-CON-002` / `PH-CON-003` assumptions fail by design (kinetic-energy budget is not strictly dissipative and not strictly conservative; cylinder-wake flow imports KE from inflow and dissipates in the wake); `D0-23 v9` is the CS02 mesh-side substrate-class dispatch extension that fires the SKIP-with-reason path on this class; `rung-4a` denotes the LB-only cross-stack-table predecessor (`gns-tgv2d` + `segnn-tgv2d`).

**Read this first — N = 5, small-N statistics.** The table below reports PH-CON-001 across **5 cylinder_flow trajectories** spanning the in-band Strouhal range (P2.1 expansion; trajectory selection rule in [DECISIONS.md D0-26](../methodology/DECISIONS.md)). The median MGN/GT mass-conservation gap is `-0.36 %` of GT, range `[-1.07 %, +0.41 %]`. This is a **small-N statistical** result, not an N=1 coverage point — but it is still *not* a CI-gate-calibration claim (threshold derivation from a defect-magnitude distribution would need a far larger N). Every per-trajectory gap sits inside the ~5.8 % harness-FE-on-P1 discretization floor: the result shows MGN reproduces GT *at the floor* across the in-band subset, not that MGN is physically incompressible to 5 %. Full scope treatment in the "Scope qualifier" and "What physics-lint did NOT catch §1" sections below; this fence is hoisted above the table deliberately, so the numbers are never read without it.

**PH-CON-001 mass / divergence-free — per trajectory:**

| `traj_idx` | GT (FE-on-P1 floor) | MGN | gap (% of GT) | D0-24 v2 band |
|---|---|---|---|---|
| 88 | 5.275e-02 | 5.219e-02 | -1.07 % | PASS |
| 48 | 5.338e-02 | 5.319e-02 | -0.36 % | PASS |
| 44 | 5.857e-02 | 5.881e-02 | +0.41 % | PASS |
| 38 | 6.440e-02 | 6.442e-02 | +0.03 % | PASS |
| 60 | 5.589e-02 | 5.569e-02 | -0.36 % | PASS |

Median gap = -0.36 % of GT; range [-1.07 %, +0.41 %] across N = 5 trajectories.

**PH-CON-002 / PH-CON-003** SKIP on every trajectory via the `open-driven-dissipative` substrate-class dispatch (D0-22 + D0-23 v9) — this is trajectory-independent, unchanged from the N=1 result; D0-24 v3 / v4 PASS.

*PASS verdicts for `PH-CON-001` mean MGN is within the GT / harness-FE-on-P1 floor envelope on that trajectory; they are NOT a claim of physical incompressibility to 5 % — see "What physics-lint did NOT catch" §1 for the floor-bounds-resolution distinction.*

See [`methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md`](../methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md) for the unified three-column cross-stack consistency table including the rung-4a LB-side columns (GNS-TGV2D + SEGNN-TGV2D).

### Scope qualifier — 5-trajectory in-band subset

P2.1 reports PH-CON-001 across 5 cylinder_flow test trajectories selected from the 23 in-band members (Strouhal in `[0.16, 0.21]`) of the Phase-2 pre-fire audit, by an even spread over `strouhal_U_max` with the Phase-2 canonical trajectory 44 as the median anchor (selection rule pre-registered in [DECISIONS.md D0-26](../methodology/DECISIONS.md)). The median MGN/GT gap is `-0.36 %` of GT (median GT floor `5.59 %`, median MGN `5.57 %`). This is a small-N characterization across the in-band subset, not a full-distribution claim: the 5 trajectories are an even-spread sample of the in-band 23, the out-of-band 77 are not exercised, and CI-gate threshold derivation from defect-magnitude distributions would need `N` far larger than 5. Coverage-and-small-N-statistics framing: physics-lint's value here is rule-firing on a real-world checkpoint and showing the result holds across the in-band subset, not a calibrated distribution over initial conditions.

### Bridge to the cross-stack story

The canonical trajectory-44 value above lands in the unified cross-stack table at [`methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md`](../methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md) as the third column (alongside `gns-tgv2d` and `segnn-tgv2d`) — the table reports the canonical-trajectory cell as a schema-uniformity artifact, not the P2.1 N=5 distribution. The schema-uniformity claim — *the same three-row conservation result-schema (rule-ids: `mass_conservation_defect`, `energy_drift`, `dissipation_sign_violation`) and run-level field set reuse across three upstream rollouts of two substrate classes, NOT public v1 rule-code reuse* (the CS02 PH-CON-001 cell is mesh-harness structural-identity reapplication; see "PH-CON-001 routing — harness, not public rule" below) — is what Case Study 02 supplies to the rung-4 series's methodology trail per [`physics-lint-validation-plan-v2.1.md`](../methodology/docs/physics-lint-validation-plan-v2.1.md) §1.5 (Case Study 02 as a falsification surface). Whether the A + B + C triad generalizes is tested by the Phase 1 + Phase 2 + Phase 3 cross-review findings and triage; see [DECISIONS.md D0-23 + D0-24](../methodology/DECISIONS.md) for the per-pattern verdicts.

## Reproducibility

### Modal entrypoints (Phase 1 + Phase 2 fires)

| Entrypoint | Purpose | Compute |
|---|---|---|
| `02-physicsnemo-mgn/modal_app.py::audit_ngc_sample_reproduction` | Phase 1 Gate D — NGC sample reproduction RMSE vs Pfaff et al. CylinderFlow RMSE-1 baseline (verdict 4 + 5) | A10G, ~10 min |
| `02-physicsnemo-mgn/modal_app.py::audit_gate_a_pyg_to_meshfield` | Phase 1 Gate A — PyG-to-MeshField materialization smoke (verdict 2) | CPU, <1 min |
| `02-physicsnemo-mgn/modal_app.py::smoke_substrate_class_vortex_shedding` | Phase 1 substrate-class smoke (verdicts 6 + 7 — ∫∇·v dV / KE budget / Strouhal) | A10G, ~5 min |
| `02-physicsnemo-mgn/modal_app.py::audit_strouhal_test_trajectories` | Phase 2 Task 1 — Strouhal pre-check across cylinder_flow test trajectories (refinement 1) | CPU, ~10 min |
| `02-physicsnemo-mgn/modal_app.py::lint_gt_trajectory` | Phase 2 Task 5 — GT-trajectory CPU lint → `gt.sarif` (control arm) | CPU, ~3 min |
| `02-physicsnemo-mgn/modal_app.py::mgn_rollout_p0_vortex_shedding` | Phase 2 Task 6 / P2.1 — MGN inference on a cylinder_flow trajectory (599 rollout steps); P2.1 fires it across trajectories {88, 48, 44, 38, 60} | A10G, ~10 min |
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

### 2. Substrate-variability — in-band subset only

Phase 2's pre-fire Strouhal audit found 23 / 100 test trajectories in the literature-anchored design band `[0.16, 0.21]`; 77 / 100 land out-of-band (mostly `St_U_mean > 0.21` while `St_U_max` is in band — the U-convention ambiguity). P2.1 exercises 5 of the in-band 23. What is NOT covered: the out-of-band 77, and a sample dense enough to derive a CI-gate threshold from the defect-magnitude distribution. The in-band subset gives a median and a range (all 5 trajectories stay inside the harness-FE-on-P1 floor); it does not give a calibrated full-distribution claim. This is a substrate-variability characterization, NOT a methodology failing — physics-lint's value is rule-firing on a real-world checkpoint and confirming the result holds across the in-band subset.

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

### 8. PH-BC-001 no-slip is structurally unreachable on a masked-wall MGN rollout

PH-BC-001 checks a boundary-condition violation as a boundary-trace error
`||u_boundary - g||`; for a no-slip wall the prescribed value `g` is zero
velocity, so the rule enters its absolute mode and PASSes iff
`||v_wall|| < abs_tol_fail`. On the CS02 cylinder MGN rollout the wall nodes
are never predicted: the inference protocol in `mgn_rollout_p0_vortex_shedding`
(`modal_app.py`) masks boundary nodes -- `v_diff_masked = torch.where(mask2,
pred_i_velo, zeros)` then `v_next = v_diff_masked + invar[:, 0:2]` (where
`invar[:, 0:2]` is the current rollout velocity state) -- so a masked wall
node stays frozen at its step-0 ground-truth value, which is `0` (no-slip). A
no-slip check on this rollout computes `||v_wall|| ~ 0` on every trajectory
and PASSes trivially: it is structurally prevented from detecting a surrogate
violation, because the surrogate never assigns wall-node velocities.

This is a documented limit of the same class as the resolution / quadrature
rules whose validity regime is degenerate-only -- the check is well-defined,
but the data path makes its outcome predetermined. It is why P2.2 retired the
planned mesh wall-node BC capability-build: a build whose only near-term
target produces a predetermined PASS adds no detection. P2.2 closes as this
finding rather than as a number in a cross-stack table cell; see
[DECISIONS.md D0-27](../methodology/DECISIONS.md).

The Ahmed Body MeshGraphNet checkpoint does not output a body-surface velocity
field at all -- it predicts surface pressure and wall shear stress, from which
the experiment computes drag downstream (verified in
`preflight/ahmed_body_protocol_audit.md`). A PH-BC-001-style no-slip *velocity*
BC check is inapplicable there, degenerate for a different structural reason
than the cylinder's masking. The velocity-BC capability-build has no home
among the current targets and is retired from P2.2; any future velocity-BC
work requires a new target and a fresh decision entry. A pressure/force-based
surface check would be a different rule, outside P2.2 and P4.1 scope.
