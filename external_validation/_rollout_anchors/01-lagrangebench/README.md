# Case Study 01 — LagrangeBench (TUM, NeurIPS 2023)

## Case study reference

LagrangeBench (Toshev et al., NeurIPS 2023; `tumaer/lagrangebench`) ships pretrained
checkpoints for two architectures — GNS (non-equivariant) and SEGNN
(E(2)-equivariant by design) — across seven particle-based fluid datasets in JAX.
This case study validates physics-lint against the published GNS-TGV2D and
SEGNN-TGV2D checkpoints; the rule set fired is PH-SYM-001 (rotation
equivariance) and PH-SYM-002 (reflection equivariance) under the harness's
schema-uniform application across both stacks.

**Headline result — equivariance gap detected on published checkpoints.**
Across 20 trajectories per (rule, stack) at single-step inference
(`n_rollout_steps = 1`), SEGNN-TGV2D's per-trajectory equivariance error sits
monomodally at the float32 noise floor (~2.3×10⁻⁷ to 3.4×10⁻⁷ across the
80 active-symmetry rows); GNS-TGV2D's same-rule signature is bimodal,
splitting roughly 50/50 between an APPROXIMATE band (~3.6×10⁻⁴ to
4.2×10⁻⁴) and a FAIL band quantized at ~0.02. The ~3.2 OOM gap between
SEGNN's monomodal floor and GNS's APPROXIMATE-band lower mode is the
load-bearing cross-stack signature — SEGNN's E(2)-equivariance is
exact-by-construction; GNS's is approximate-by-training, consistent with
Helwig et al.'s data-augmentation characterization. The full table and
SARIF artifacts are linked under "Validation harness" below.

**P2.1 multi-trajectory expansion.** Per [DECISIONS.md D0-26][D0-26-link],
the rung-4b table reports 20 trajectories per (rule, stack) on TGV2D for
both GNS and SEGNN. The trajectory set is deterministic and reproducible
from the committed audit JSON; selection was pre-registered before any
Modal fire.

**P2.3 scope qualifier (structural-empirical link).** Per [DECISIONS.md
D0-28][D0-28-link], the equivariance gap above is reported on the
GNS-as-shipped checkpoint — a single realization of the "non-equivariant
architecture under typical training" class. The project deliberately does
**not** extend rung-4b with a self-trained non-equivariant GNN as a
second architectural data point; the rollout-anchor portfolio rests on
published checkpoints (F3 borrowed-credibility framing, see
`methodology/docs/2026-05-01-rollout-anchor-extension-design.md` §1.1),
and a self-trained baseline would be in a structurally weaker evidence
class. The structural-empirical-link argument is defeasible — a
non-equivariant architecture trained with full SO(2) augmentation could
in principle approximate equivariance to near-noise-floor accuracy — but
the rung-4b reading on GNS-as-shipped is consistent with the
architectural reason.

[D0-26-link]: https://github.com/tyy0811/physics-lint/blob/master/external_validation/_rollout_anchors/methodology/DECISIONS.md
[D0-28-link]: https://github.com/tyy0811/physics-lint/blob/master/external_validation/_rollout_anchors/methodology/DECISIONS.md

**What this case study does NOT cover.** PH-BC-001 no-slip on a body-surface
velocity field is structurally inapplicable to LagrangeBench's particle
representation (no mesh; no surface trace operator). PH-CON-001 mass and
PH-CON-002/003 energy/dissipation are exercised in the rung-4a sibling
artifact (`gns-tgv2d` + `segnn-tgv2d` columns) rather than re-derived here.
The CS01 deliverables are scoped to particle-side equivariance evidence
on the two published checkpoints; broader architectural sweeps are out
of v1.0 scope.

## Priority-tier dataset table

Three datasets, two architectures = up to six rollout sets. The table
below records the v1.0-realized rows (rung-4b TGV2D) and the rows that
stayed deferred per roadmap Amendment 4.

| Priority | Dataset | Architecture | Headline rule | Status |
|---|---|---|---|---|
| P0 | TGV 2D (Taylor-Green vortex) | SEGNN | `PH-SYM-001` / `PH-SYM-002` equivariance | **Shipped (rung-4b)** — monomodal float32 floor |
| P0 | TGV 2D | GNS | Same rules → expected equivariance flag | **Shipped (rung-4b)** — APPROXIMATE / FAIL bimodal |
| P1 | Dam break 2D | GNS | `PH-CON-001` (mass), `PH-BC-001` (wall) | Deferred (Amendment 4: out-of-scope for v1.0 — see [DECISIONS.md D0-27](../methodology/DECISIONS.md) on PH-BC-001 wall-checks) |
| P2 | Reverse Poiseuille 2D | SEGNN | `PH-BC-001` (no-slip) | Deferred (same wall-check limit as P1) |
| P3 | Dam break 2D | SEGNN | Cross-validate P1 result | Deferred (gated by P1) |
| P3 | TGV 3D | GNS | Stretch only | Deferred (3D not in v1.0 substrate scope) |

The rung-4b TGV2D pair is the v1.0 shipped CS01 result; the other rows
remain in the v1.x backlog.

## Validation harness

**Cross-stack equivariance table.** The rung-4b writeup is the canonical
authored artifact for the 20-trajectories-per-(rule, stack) cross-stack
signature; it pins SHA-bound paths to the SARIFs, the rendered table,
and the rollout-depth figure at its commit date:

- **Writeup:** [`../methodology/docs/2026-05-07-rung-4b-equivariance-table.md`](../methodology/docs/2026-05-07-rung-4b-equivariance-table.md).

Current per-stack SARIFs and figures live under
[`outputs/sarif/`](outputs/sarif/) and [`outputs/figures/`](outputs/figures/);
filenames carry the generating-commit SHA prefix and may have been
re-emitted since the rung-4b writeup was frozen. Follow the writeup's
links for the authored snapshot; consult the directory listings for the
current state.

**Modal app.** [`modal_app.py`](modal_app.py) ships the local
entrypoints used for inference, lint emission, and Strouhal smoke
checks. Run `modal run modal_app.py::<entrypoint>` for a specific arm
(see the file for the entrypoint catalogue and image / GPU profiles).

**Checkpoints, hashes, git SHAs.** Committed to the SARIF metadata per
the schema in [`../_harness/SCHEMA.md`](../_harness/SCHEMA.md).

## Cross-references

### Cross-stack tables

- **Original LB-only (rung-4a, frozen):** [`../methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md`](../methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md) — the original two-column cross-stack table over GNS-TGV2D + SEGNN-TGV2D. Preserved frozen at its commit date for regression-check value (LB-side comparison baseline if the extended renderer ever drifts).
- **Extended (CS02, canonical going forward):** [`../methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md`](../methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md) — unified three-column table extending rung-4a with the CS02 MGN-on-vortex-shedding column. The schema-uniformity claim now spans two substrate classes (dissipative-isotropic + open-driven-dissipative).

Both artifacts exist deliberately: the rung-4a artifact preserves the
LB-only state at its commit date; the 2026-05-13 artifact extends that
state with cross-substrate evidence per the frozen-original-plus-pointer
methodology discipline.

### Methodology decisions

- **D0-19, D0-20** (rung-4a pre-registrations): [`../methodology/DECISIONS.md`](../methodology/DECISIONS.md).
- **D0-21** (rung-4b cross-stack equivariance pre-registration): same file.
- **D0-23, D0-24** (Case Study 02 Phase 1 + Phase 2 audit verdicts): same file.
- **D0-26** (P2.1 CS02 multi-trajectory expansion): same file.
- **D0-27** (P2.2 closes as structural-degeneracy finding, PH-BC-001 no-slip on masked-wall MGN): same file.
- **D0-28** (P2.3 closes as documented methodology choice, equivariance gap second-architecture question): same file.
