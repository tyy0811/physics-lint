# Case Study 01 — LagrangeBench (TUM, NeurIPS 2023)

*Day 1 deliverable; populated after Modal A100 rollout generation.*

## Targets

Three datasets, two architectures = up to six rollout sets. Aim for the
top 3 first; do the rest only if time remains.

| Priority | Dataset | Architecture | Headline rule |
|---|---|---|---|
| P0 | TGV 2D (Taylor-Green vortex) | SEGNN | `PH-CON-003` (monotone energy decay) + `PH-SYM-001`/`002` equivariance |
| P0 | TGV 2D | GNS | Same rules → expect equivariance flag |
| P1 | Dam break 2D | GNS | `PH-CON-001` (mass), `PH-BC-001` (wall) |
| P2 | Reverse Poiseuille 2D | SEGNN | `PH-BC-001` (no-slip) |
| P3 | Dam break 2D | SEGNN | Cross-validate P1 result |
| P3 | TGV 3D | GNS | Stretch only |

## Rule × model results

*[Populated by Day 1 rollouts. SARIF outputs in `outputs/lint.sarif`.]*

## Reproducibility

Modal entrypoint: `modal_app.py`. Inference script: `run_inference.py`.
Lint driver: `lint_rollouts.py` — invokes
`_rollout_anchors/_harness/particle_rollout_adapter.py`.

Checkpoints, hashes, and git SHAs commit to the SARIF metadata per the
schema in [`../_harness/SCHEMA.md`](../_harness/SCHEMA.md).

## Cross-references

- **Cross-stack writeup (rung 4a):** [`../methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md`](../methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md) — methodology writeup over the SARIF artifacts in `outputs/sarif/`.
- **Methodology decisions (D0-19, D0-20):** [`../methodology/DECISIONS.md`](../methodology/DECISIONS.md).
