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

### Cross-stack tables

- **Original LB-only (rung-4a, frozen):** [`../methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md`](../methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md) — the original two-column cross-stack table over GNS-TGV2D + SEGNN-TGV2D. Preserved frozen at its commit date for regression-check value (LB-side comparison baseline if the extended renderer ever drifts).
- **Extended (CS02, canonical going forward):** [`../methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md`](../methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md) — unified three-column table extending rung-4a with the CS02 MGN-on-vortex-shedding column. The schema-uniformity claim now spans two substrate classes (dissipative-isotropic + open-driven-dissipative). Amendment 1 (Ahmed Body) extends with a fourth column when that row lands.

Both artifacts exist deliberately: the rung-4a artifact preserves the LB-only state at its commit date; the 2026-05-13 artifact extends that state with cross-substrate evidence per the frozen-original-plus-pointer methodology discipline.

### Methodology decisions

- **D0-19, D0-20** (rung-4a pre-registrations): [`../methodology/DECISIONS.md`](../methodology/DECISIONS.md).
- **D0-23, D0-24** (Case Study 02 Phase 1 + Phase 2 audit verdicts): same file.
