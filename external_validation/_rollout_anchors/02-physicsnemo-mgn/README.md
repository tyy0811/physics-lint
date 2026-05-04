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
