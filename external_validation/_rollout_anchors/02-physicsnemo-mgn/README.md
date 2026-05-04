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

*[Populated by Day 2 rollouts. SARIF outputs in `outputs/lint.sarif`.]*

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
