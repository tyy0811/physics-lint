# physics-lint-validation: Rollout-Anchor Extension Plan v3.0

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan section-by-section. v3 is structured as wall-clock-day-bounded sections rather than per-task TDD checkboxes; each day's deliverables and acceptance criteria are explicit and gate the next day's work.

**Status:** Draft, pre-execution.
**Owner:** Jane Yeung.
**Predecessor:** [`physics-lint-validation-plan-v2.md`](physics-lint-validation-plan-v2.md), superseded by this document via the spec's Appendix B diff list. Sections of v2 not enumerated in Appendix B are carried forward verbatim; see this plan's Appendix B for the v2-to-v3 mapping audit.
**Spec:** [`2026-05-01-rollout-anchor-extension-design.md`](2026-05-01-rollout-anchor-extension-design.md) — authoritative design source. See spec §1.5 for the location-by-artifact-type table that governs every path in this plan; see spec §1.1 for the explicit out-of-scope statement (this branch does not modify `physics_lint.field.*` public API).
**Branch:** `feature/rollout-anchors`. Parallel branches `chore/v1.0-stub-resolution` and conditional `feature/field-protocol-v1.1` are out of scope per spec §1.1.
**Budget:** 3 days wall-clock + 1 day buffer = 4 days max. Modal A100 ceiling $30 (unchanged from v2).
**Goal:** Extend physics-lint's `external_validation/` framework with a rollout-domain F3 anchor — populating a new "rollout-domain F3 status" column in the 18-rule anchor matrix for the six v1.0 rules where the anchor is meaningful, plus PH-RES-001 as a P2 stretch on the mesh side only.

---

## 1. Strategic framing (read first, do not skip)

The case studies serve **two distinct audiences simultaneously**, which dictates every downstream choice:

- **Audience A (academic, Munich/Stuttgart SciML):** Gerdts, Günnemann, Fehr/Kneifl, TUM/UniBw committees. Recognises LagrangeBench, MeshGraphNets, FNO, PDEBench. Values methodological rigor, falsifiable claims, equivariance theory.
- **Audience B (industrial, BMW Passive Safety):** Jehle, BMW R&D, NVIDIA-collaboration-style engineers. Recognises NVIDIA PhysicsNeMo, Modulus, Nabian arXiv:2510.15201. Values production-stack relevance, deployment gates, CI integration.

The two-case design is **deliberately bipartisan** — one case per audience. Do not collapse this into a single case study to save time; the dual coverage is the differentiator.

**Framing claim (option B; replaces v2's sibling-repo framing).** This work *extends physics-lint's existing `external_validation/` framework* with a `_rollout_anchors/` subdirectory that adds the rollout-domain F3 anchor previously absent from the 18-of-18 anchor matrix. The merged implementation lands inside `physics-lint/external_validation/_rollout_anchors/` via the `feature/rollout-anchors` PR; planning artifacts (this plan, the spec, in-flight DECISIONS.md, cover-letter drafts, cross-review notes) live in `physics-lint-validation/` per spec §1.5 and are snapshotted into physics-lint at merge time per spec §1.5 option (b). The framing claim — "see the new rollout-domain F3 column on the anchor matrix" — beats "see also sibling repo" by a wide margin for Audience A reviewers who actually read `external_validation/README.md`.

**Explicit out-of-scope statement** (verbatim from spec §1.1; reproduce in `_rollout_anchors/README.md` §1):

> *"This branch does not modify `physics_lint.field.*` public API. Any extension to the Field protocol — temporal axis, particle support, MeshField widening to non-scikit-fem mesh sources — is out of scope. Such extensions, if needed, would land on a separate `feature/field-protocol-v1.1` branch sequenced independently after v1.0 ships. Stub resolution for PH-SYM-004 / PH-BC-002 / PH-NUM-001 is out of scope and lives on the parallel `chore/v1.0-stub-resolution` branch."*

The negative statement is load-bearing: it stops scope creep on Day 2 when the temptation to "just widen MeshField a bit" appears.

**The headline claim we are aiming for** (replaces v2's SEGNN-vs-GNS-as-headline-result framing; verbatim from spec §0):

> *On the mesh side (NVIDIA PhysicsNeMo MeshGraphNet), the existing public Field/rule API consumed per-timestep materializations of trained third-party output without rule modification. On the particle side (LagrangeBench SEGNN/GNS), the rule API does not natively accept particle clouds; the rule structural identities — finite-group equivariance for PH-SYM-001/002, conservation balance for PH-CON-001/002/003 — were reapplied via a thin private harness validated against analytical fixtures, both paths emitting SARIF in the same schema.*

This sentence is the writeup target. Every step below is in service of being able to write it honestly. (See spec §3.3 for the precise framing of "what 'without modification' actually means" and spec §1.4 for the explicit "what we are NOT claiming" list.)

---

## 2. Pre-flight (Day 0, ≤4 hours, CPU only, no Modal GPU spend)

### 2.1 Day-0 audit (replaces v2 §2.1's PyPI-installable check, which is moot per spec §1.1; verbatim from spec §5.1)

Three audit questions plus a controlled-fixture deliverable. All run on CPU; no Modal GPU spend on Day 0.

**Audit Q1 — MeshField-from-DGL feasibility.** On one PhysicsNeMo NGC sample timestep, attempt `MeshField(basis=reconstructed_basis, dofs=node_values_at_t)` where the scikit-fem `Basis` is reconstructed from `(node_positions, edge_index)`. Verdict feeds Gate A (§7).

- **PASS** if MeshField construction works on the sample → mesh public-API path is live in full form.
- **PARTIAL** if MeshField fails but `GridField(values=resampled, h=spacing, periodic=False)` works after a regular-grid resampling pass → mesh public-API path lives with a documented resampling step, cited in Appendix A.2 cover-letter variant of the spec.
- **FAIL** if neither works → cross-stack-via-public-API claim is dropped per §7 Gate A action; both case studies route through `_harness/`-private adapters.

**Audit Q2 — PH-CON-001 emitted-quantity sanity check on the same timestep.** Read the rule source first; if its emitted quantity makes assumptions about the field's domain that NGC output violates (e.g., expects a periodic domain, expects normalized density), document the mismatch in DECISIONS.md and either adapt the input or scope the rule out for this case.

**Audit Q3 — particle-harness model-loading split confirmed.** PH-CON-001/002/003 must run from cached `.npz` (read-only path; no JAX dependency). PH-SYM-001/002 require the JAX/Haiku model to run rotated inference, so the model-loading half of the particle harness ships behind the `[validation-rollout]` extra. Confirm both halves work end-to-end on a stub configuration.

**Controlled-fixture harness validation layer (3–4h; the Day-0 deliverable that gates Day 1).** Build the fixtures listed in spec §4.2 under `_rollout_anchors/_harness/tests/fixtures/`:

1. `c4_invariant_4particle.py` — four particles at the vertices of a square centered on the origin, C₄-invariant velocity assignment. **Expected:** both paths emit ε_C4 ≤ 10⁻⁶ (machine precision; configuration is exactly C₄-invariant by construction).
2. `c4_perturbed_4particle.py` — same configuration with one particle displaced by a known δ. **Expected:** both paths emit ε_C4 = O(δ); the two paths' values agree to within 10⁻⁴.
3. `c4_grid_equivalent.py` — shared discretisation utility used by fixtures 1 and 2 to materialize the gridded equivalent.
4. `mass_conservation_fixture.py` — fluid configuration with known mass at t₀ and t₁; both paths must report the same defect within 10⁻⁴.
5. `test_harness_vs_public_api.py` — pytest assertion harness that computes ε_harness_vs_public for each fixture × rule and asserts ≤ 10⁻⁴.

Run **Gate B** (§7) on the fixture layer. Pre-registered tolerance: ε ≤ 10⁻⁴ → PASS; 10⁻⁴ < ε ≤ 10⁻² → APPROXIMATE (proceed, document, disclaim); ε > 10⁻² → FAIL stop-the-bus, with the 5-minute fixture-sanity-check qualifier in §7 Gate B.

### 2.2 Modal account checks

- Verify A100 quota and current spend.
- Set hard budget cap: $30. If we hit $25, freeze new training runs.
- Create the Modal app skeleton with two entrypoints under the implementation tree: `_rollout_anchors/01-lagrangebench/modal_app.py` and `_rollout_anchors/02-physicsnemo-mgn/modal_app.py` (paths governed by spec §1.5 location table).

### 2.3 NGC API key

- Sign up for NVIDIA NGC (free, takes 5 minutes), generate API key, store in Modal secret.
- Test `ngc registry model download-version "nvidia/modulus/modulus_ns_meshgraphnet:v0.1"` on a sandbox to confirm the key works. **Do this first** — auth issues mid-Day-2 are a budget hit.

### 2.4 Implementation tree (replaces v2's `physics-lint-validation/` repo skeleton; verbatim from spec §2.1)

This is the **implementation tree** that lands in physics-lint via the `feature/rollout-anchors` PR. Planning artifacts live in `physics-lint-validation/` per spec §1.5; the `docs/` subdirectory below holds **frozen snapshots** of the spec, plan, and DECISIONS.md copied in at merge time per spec §1.5 option (b).

```
physics-lint/
└── external_validation/
    ├── README.md                          # gains "rollout-domain F3 status" column
    └── _rollout_anchors/                  # NEW — this branch's primary deliverable
        ├── README.md                      # framing, headline, what-NOT-claimed; cross-references docs/
        ├── docs/                          # frozen snapshots at merge time per spec §1.5 option (b)
        │   ├── 2026-05-01-rollout-anchor-extension-design.md   # snapshot of physics-lint-validation/docs/...
        │   ├── 2026-05-01-rollout-anchor-extension-plan.md     # snapshot
        │   └── DECISIONS.md               # snapshot of physics-lint-validation/DECISIONS.md
        ├── 01-lagrangebench/
        │   ├── README.md                  # case-study writeup with rule × model table
        │   ├── modal_app.py               # Modal entrypoint for rollout generation
        │   ├── run_inference.py           # JAX/Haiku model load + rollout export
        │   ├── lint_rollouts.py           # invokes particle_rollout_adapter
        │   ├── outputs/
        │   │   ├── rollouts/              # *.npz (gitignored or git-lfs if >50MB)
        │   │   ├── lint.sarif             # canonical lint output, committed
        │   │   └── figures/
        │   └── tests/                     # ≥3 unit tests on adapter contract
        ├── 02-physicsnemo-mgn/            # (or 02-fno-darcy/ under Gate D fallback)
        │   ├── README.md
        │   ├── modal_app.py
        │   ├── run_inference.py           # NGC checkpoint inference
        │   ├── lint_rollouts.py           # invokes mesh_rollout_adapter + public API
        │   ├── outputs/
        │   └── tests/
        └── _harness/                      # PRIVATE shared infrastructure
            ├── SCHEMA.md                  # documents both .npz schemas + tolerances
            ├── particle_rollout_adapter.py  # LagrangeBench-side
            ├── mesh_rollout_adapter.py    # PhysicsNeMo-side, materializes Field
            ├── sarif_emitter.py           # shared SARIF output
            └── tests/
                └── fixtures/              # Day-0 controlled-fixture validation layer
                    ├── c4_invariant_4particle.py
                    ├── c4_perturbed_4particle.py
                    ├── c4_grid_equivalent.py
                    ├── mass_conservation_fixture.py
                    └── test_harness_vs_public_api.py
```

This tree mirrors the existing `external_validation/` per-rule structure (`PH-*/` with `test_anchor.py` + `CITATION.md`). No reinvention.

---

## 3. Case Study 01: LagrangeBench (Day 1, 6–8 hours)

### 3.1 Targets (in priority order)

Three datasets, two architectures = up to six rollout sets. **Aim for the top 3 first; do the rest only if time remains.**

| Priority | Dataset | Architecture | Headline rule |
|---|---|---|---|
| P0 | TGV 2D (Taylor-Green vortex) | SEGNN | PH-CON-003 (monotone energy decay) + PH-SYM-001/002 equivariance baseline |
| P0 | TGV 2D | GNS | Same rules → expect equivariance flag |
| P1 | Dam break 2D | GNS | PH-CON-001 (mass), PH-BC-001 (wall) |
| P2 | Reverse Poiseuille 2D | SEGNN | PH-BC-001 (no-slip) |
| P3 | Dam break 2D | SEGNN | Cross-validate P1 result |
| P3 | TGV 3D | GNS | Stretch only |

P0 + P1 = the headline. P2/P3 only if Day 1 finishes early.

### 3.2 Step-by-step

**Step 1 — Install (30 min).** On Modal A100 image:

```bash
pip install --upgrade pip
pip install -U "jax[cuda12]" jaxlib  # match Modal CUDA version
git clone https://github.com/tumaer/lagrangebench && cd lagrangebench
pip install -e ".[dev]"
bash download_data.sh tgv2d datasets/
bash download_data.sh dam2d datasets/
```

Smoke test: `python main.py mode=infer dataset=tgv2d` on the included toy config to confirm JAX sees the GPU. **Hour-2 micro-gate** (§7) — `jax.devices()` must return at least one A100; if not, pivot per the gate's FAIL action immediately, do NOT let environmental failure consume hours 2–6 silently.

**Step 2 — Download checkpoints (15 min).** From the LagrangeBench README's gdown links:

```bash
gdown <gns_tgv2d_zip_url>
gdown <segnn_tgv2d_zip_url>
gdown <gns_dam2d_zip_url>
unzip -d checkpoints/ ./*.zip
```

**Step 3 — Generate rollouts (1–2 hours).** For each (dataset, model) pair:

```bash
python main.py mode=infer \
    load_ckp=checkpoints/segnn_tgv2d/best/ \
    config=configs/segnn_tgv2d.yaml \
    eval.n_rollout_steps=100 \
    eval.n_trajs=20
```

Export trajectories to `_rollout_anchors/01-lagrangebench/outputs/rollouts/segnn_tgv2d.npz` per the schema in spec §3.2 `particle_rollout.npz`:

```python
{
    "positions":     ndarray,  # (T, N_particles, D)  fp32
    "velocities":    ndarray,  # (T, N_particles, D)  fp32
    "particle_type": ndarray,  # (N_particles,)       int32  fluid/wall/...
    "dt":            float64,
    "domain_box":    ndarray,  # (2, D)
    "metadata": {
        "ckpt_hash":         str,   # SHA-256 of checkpoint file
        "ckpt_path":         str,
        "git_sha":           str,   # physics-lint commit at time of generation
        "lagrangebench_sha": str,   # external repo commit
        "dataset":           str,   # "tgv2d" | "dam2d" | ...
        "model":             str,   # "segnn" | "gns"
        "seed":              int,
        "framework":         str,   # "jax+haiku"
        "framework_version": str,
    },
}
```

**The schema is documented authoritatively in `_rollout_anchors/_harness/SCHEMA.md`.** Do not deviate — both the particle harness adapter and the controlled-fixture validation depend on it.

**Step 4 — Run physics-lint via the particle harness (1 hour).** *Replaces v2 §3.2 step 4's `lint_rollout(.npz)` API, which does not exist in physics-lint.*

The particle harness lives at `_rollout_anchors/_harness/particle_rollout_adapter.py` and ships with two halves:

- **Read-only path** (PH-CON-001/002/003): reads cached `particle_rollout.npz`, computes mass / KE / dE/dt directly from particle positions and velocities, emits SARIF. Does not require the JAX model object — works with `pip install physics-lint` (no extras needed).
- **Model-loading path** (PH-SYM-001/002): requires `jax`/`haiku` from the `[validation-rollout]` extra, loads SEGNN/GNS checkpoint, runs identity rollout from x₀, runs rotated rollout from R x₀, applies R⁻¹ to derotated rollout, computes ε_C4 / ε_refl over the pre-registered angle set, emits SARIF.

Each emitted SARIF result has `properties.source = "rollout-anchor-harness"` so consumers can distinguish harness-emitted vs. public-rule-emitted SARIF. The harness was validated against analytical fixtures on Day 0 per Gate B (§7); the structural-identities-held claim is therefore falsifiable, not asserted.

The six rules tested on the particle side: PH-SYM-001 (C₄), PH-SYM-002 (reflection), PH-CON-001 (mass), PH-CON-002 (energy), PH-CON-003 (dissipation sign), PH-BC-001 (wall non-penetration; dam break only). PH-SYM-003 is **out of scope** (its emitted quantity is infinitesimal scalar Lie-derivative, not the global-finite equivariance the rotation sweep measures); PH-SYM-004 is **out of scope** (SKIP-always V1 stub).

```python
# Example invocation, particle harness model-loading path:
from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    run_equivariance_test,
)
import math

result = run_equivariance_test(
    npz_path="01-lagrangebench/outputs/rollouts/segnn_tgv2d.npz",
    checkpoint_path="checkpoints/segnn_tgv2d/best/",
    rules=["PH-SYM-001", "PH-SYM-002"],
    angles=[0, math.pi / 4, math.pi / 2, math.pi, 3 * math.pi / 2],
    output_sarif_path="01-lagrangebench/outputs/lint_segnn_tgv2d.sarif",
)
```

**Step 5 — Equivariance test wiring (replaces v2 §3.2 step 5; rotation sweep moves into the particle harness).**

The PH-SYM-001 / PH-SYM-002 rules at the rollout level are implemented inside `particle_rollout_adapter.py`'s model-loading path, **not** as a physics-lint rule extension. The mechanic is unchanged from v2:

1. Load the checkpoint into Haiku.
2. Generate one rollout from initial conditions $x_0$.
3. Apply rotation $R \in SO(2)$ to $x_0$, generate a second rollout.
4. Apply $R^{-1}$ to the second rollout's positions/velocities.
5. Compute $\epsilon_{\text{rot}} = \|R^{-1} f(R x_0) - f(x_0)\|$ as a function of rotation angle $\theta \in \{0, \pi/4, \pi/2, \pi, 3\pi/2\}$ (C₄ test). Repeat with $R$ replaced by reflection for PH-SYM-002.

The pre-registered tolerance bands carry forward unchanged from v2 (and are reproduced in spec §4.4):

- **ε_rot ≤ 10⁻⁵** → PASS — machine-precision equivariance. SEGNN expected.
- **10⁻⁵ < ε ≤ 10⁻²** → APPROXIMATE — flagged, in approximate-equivariance band. GNS expected.
- **ε > 10⁻²** → FAIL — equivariance broken.

The harness emits SARIF with `properties.source = "rollout-anchor-harness"` to distinguish from rule-emitted SARIF, and references the fixture-validated harness-vs-public-API tolerance ε_harness_vs_public ≤ 10⁻⁴ (Gate B PASS) as a guarantee that the harness's emitted ε_rot is comparable to what the public rule would emit on the gridded equivalent.

**Hold the threshold honest before running** — write the threshold into the test code, then run, then report. Do not retroactively tune the threshold to match the observed value. (This is the measurement-before-amendment discipline applied to the validation itself.)

**Note on the §4.2-vs-§4.4 threshold distinction (spec §4.4).** ε_C4 ≤ 10⁻⁶ in fixture #1 is the *fixture-construction* tolerance (configuration is exactly C₄-invariant by construction; observed ε is float32 round-off). ε_rot ≤ 10⁻⁵ for SEGNN is the *trained-model* PASS band (theoretically equivariant, empirically bounded by accumulated round-off). The two are not interchangeable.

**Step 6 — Capture + write up (1–2 hours).**

For each (dataset, model) pair, write a short subsection of `_rollout_anchors/01-lagrangebench/README.md`:

```markdown
### TGV 2D — SEGNN
- Checkpoint: segnn_tgv2d, best/, SHA-256 <hash>
- Rollout: 20 trajectories × 100 steps
- PH-CON-001: PASS (Δparticle_count = 0, Δ∫ρ < 1e-9)
- PH-CON-002: KE MSE = X (LagrangeBench reports Y; ratio Z)
- PH-CON-003: PASS (KE monotone decreasing across all 20 trajectories)
- PH-SYM-001 (C4): PASS (ε_rot = 4.3e-7, < 1e-5 threshold)
- PH-SYM-002 (reflection): PASS (ε_refl = 3.8e-7)

### TGV 2D — GNS
- ...
- PH-SYM-001 (C4): FLAG (ε_rot = 1.2e-2, in approximate-equivariance band)
- ...
```

Note: PH-SYM-003 and PH-SYM-004 lines are omitted relative to v2, per the in-scope rule list (spec §1.2).

### 3.3 Day 1 acceptance criteria

Before moving to Day 2:

- [ ] At least one (dataset, model) pair has full SARIF output committed.
- [ ] SEGNN-vs-GNS equivariance comparison shows quantitatively distinct results (the headline).
- [ ] `outputs/rollouts/*.npz` is reproducible (re-running the Modal entrypoint gives bit-identical results given fixed seed).
- [ ] DECISIONS.md cumulative entries ≥ 6 (Day 0 audit + Day 1 results; the wider 10-entry target lands by end of Day 3).

**If by hour 6 of Day 1 the headline equivariance result is not visible** → defrag: drop GNS dam break and reverse Poiseuille, focus all remaining time on getting one clean SEGNN-vs-GNS TGV comparison. The headline matters more than coverage.

---

## 4. Case Study 02: NVIDIA PhysicsNeMo MeshGraphNet (Day 2, 4–6 hours)

### 4.1 Targets

Two checkpoints, both already on NGC:

| Priority | Checkpoint | Domain | Headline rule |
|---|---|---|---|
| P0 | `modulus_ns_meshgraphnet` (vortex shedding 2D) | Incompressible NS, cylinder wake | PH-CON-001 (mass / divergence-free) + PH-CON-002/003 |
| P1 | `modulus_ahmed_body_meshgraphnet` | Steady RANS, car-like geometry | PH-BC-001 (no-slip) |
| **P2 stretch** | `modulus_ns_meshgraphnet` | Same as P0 | PH-RES-001 (BDO momentum residual) — only if Day 2 hour 4 leaves ≥3h buffer |

P0 first — vortex shedding has cleaner physics for our rules. P1 is the BMW-recognition signal (Ahmed Body = automotive). Note: PH-NUM-002 resolution sweep is **deferred to v1.1 backlog** (spec §1.2) — its multi-resolution harness is itself a separate deliverable.

### 4.2 Step-by-step

**Step 1 — Container + install (30 min).** On Modal A100:

```bash
# Use NVIDIA's container directly; it has DGL + PyTorch pre-installed
# nvcr.io/nvidia/modulus/modulus:25.08
pip install nvidia-physicsnemo
ngc config set  # use stored API key
```

**Step 2 — Download checkpoints (15 min).**

```bash
ngc registry model download-version "nvidia/modulus/modulus_ns_meshgraphnet:v0.1"
ngc registry model download-version "nvidia/modulus/modulus_ahmed_body_meshgraphnet:v0.1"
```

Verify hashes; commit hashes to `DECISIONS.md`.

**Sanity-check inference matches NGC's shipped sample input** — each NGC checkpoint ships with a sample input plus expected output; `test_inference_matches_ngc_sample` runs inference and compares against the shipped expected output to within a documented tolerance (default: max-abs-error ≤ 10⁻³ on velocity components, looser on pressure if NGC's tolerance is documented as such). This is the gate-determining test for Gate D (§7); a divergence here is the "values wrong" failure mode that distinguishes Gate D's two FAIL paths.

**Step 3 — Run NGC inference scripts (1 hour).** Each NGC checkpoint ships with a sample input and an inference script. Run as documented:

```bash
cd modulus_ns_meshgraphnet_v0.1/
python inference.py --input sample_input/ --output outputs/
```

**Export rollouts to the `mesh_rollout.npz` schema** (spec §3.2):

```python
{
    "node_positions": ndarray,           # (N_nodes, D)  static (mesh fixed across t)
    "edge_index":     ndarray,           # (2, N_edges)  int64  (omitted for FNO grid case)
    "node_type":      ndarray,           # (N_nodes,)    INTERIOR/INFLOW/WALL/...
    "node_values":    dict[str, ndarray],  # per-field, each (T, N_nodes [, D_field])
    "dt":             float64,
    "metadata": {
        "ckpt_hash":         str,
        "ngc_version":       str,        # "v0.1" etc.
        "git_sha":           str,
        "dataset":           str,        # "vortex-shedding-2d" | "ahmed-body" | "darcy"
        "model":             str,
        "framework":         str,        # "pytorch+dgl" | "pytorch+neuraloperator"
        "framework_version": str,
        "resampling_applied": bool,      # True under Gate A PARTIAL fallback
    },
}
```

**Step 4 — Run physics-lint via the public API per timestep (1 hour).** *Replaces v2 §3.2 step 4's `lint_rollout(.npz)` API — and unlike Case Study 01, the mesh side uses physics-lint's existing public CLI directly.*

The mesh adapter at `_rollout_anchors/_harness/mesh_rollout_adapter.py` materializes one timestep of the rollout into a Field-API-compatible object: `MeshField(basis=reconstructed_basis, dofs=node_values_at_t)` if Gate A returned PASS, or `GridField(values=resampled, h=spacing, periodic=False)` if Gate A returned PARTIAL. Rule kernels then consume per-timestep via the existing `physics-lint check` CLI without modification.

Rules to test:

- **PH-CON-001** (mass) on vortex shedding: divergence-free check on velocity field.
- **PH-CON-002** (energy) on vortex shedding: KE budget.
- **PH-CON-003** (dissipation sign) on vortex shedding: sign of dE/dt.
- **PH-BC-001** on both: no-slip wall enforcement, $|v|_{\text{wall}} < 10^{-4}$.
- **PH-RES-001** *(P2 stretch, only if Day 2 hour 4 leaves ≥3h)*: BDO norm-equivalence on momentum residual.

PH-NUM-002, PH-SYM-001/002, PH-SYM-003, and PH-SYM-004 are explicitly **not** tested on the mesh side per spec §1.2's in-scope/out-of-scope split.

**Step 5 — Cross-stack consistency subsection (1 hour; replaces v2 §4.2 step 5 with trimmed framing per spec §3.3).**

The interesting story across both case studies, in trimmed form:

> *The rule structural identities held across both stacks. On the mesh side (PhysicsNeMo MGN), the existing public Field/rule API consumed per-timestep MeshField/GridField materializations of the model output without rule modification. On the particle side (LagrangeBench), the rule API does not natively accept particle clouds; the structural identities — finite-group equivariance for PH-SYM-001/002, conservation balance for PH-CON-* — were reapplied via a thin harness that mirrors what the public rule emits at the grid/mesh level, validated against analytical fixtures (Gate B PASS, ε_harness_vs_public ≤ 10⁻⁴). Both harness and public-API paths emit SARIF in the same schema.*

This is the bridge sentence that makes the writeup more than a sum of its parts. **The trimmed version replaces v2's "ran without rule modification across two completely different stacks" framing**, which was empirically false in a way detectable on a single careful read of `external_validation/README.md`. See spec §1.4 for the explicit "what we are NOT claiming" list.

### 4.3 Day 2 acceptance criteria

- [ ] PhysicsNeMo vortex shedding (P0) runs inference end-to-end on Modal — or, under Gate D fallback, neuraloperator FNO on Darcy substitutes.
- [ ] SARIF outputs committed for vortex shedding (or Darcy fallback) via the public `physics-lint check` CLI.
- [ ] Cross-stack consistency check passes: harness output and public-API output emit the same SARIF schema and (where comparable on Gate B fixtures) agree to within 10⁻⁴.

**Drop-decision if behind schedule:** drop Ahmed Body (P1) and PH-RES-001 stretch, keep vortex shedding (P0). The story still holds — the BMW-recognition signal is weaker but the technical claim is unaffected. **Do NOT drop Case Study 02 entirely** — see Gate D in §7 for the FNO-on-Darcy fallback. The dual-audience structure is the strategic spine; one-stack writeup is not an option.

---

## 5. Day 3: writeup, optional retrain, application integration

### 5.1 Writeup (3–4 hours, mandatory)

Build `_rollout_anchors/README.md` (root of the merged subdirectory). Structure:

```markdown
# Rollout-domain F3 anchor

Two third-party case studies extending physics-lint's external_validation/
framework with a rollout-domain anchor. See [docs/2026-05-01-rollout-anchor-extension-design.md](docs/2026-05-01-rollout-anchor-extension-design.md)
for the design spec and [docs/2026-05-01-rollout-anchor-extension-plan.md](docs/2026-05-01-rollout-anchor-extension-plan.md)
for the implementation plan (both snapshotted at merge per spec §1.5 option (b)).

## Headline claim
[1 paragraph, the trimmed bridge sentence from §4.2 step 5]

## Case studies
### 01 — LagrangeBench (TUM, NeurIPS 2023)
[1 paragraph + table + link]
### 02 — NVIDIA PhysicsNeMo MeshGraphNet (NGC pretrained) — or FNO on Darcy under Gate D fallback
[1 paragraph + table + link]

## Cross-stack consistency
[Table: rule × case study × path (public-API or harness) × result]

## What physics-lint did NOT catch
[Verbatim from spec §1.4; methodological-honesty signal that matters for Audience A]

## Reproducibility
[Modal entrypoints, checkpoint hashes, git SHA, conda lockfile]

## Citations
- Toshev et al. (LagrangeBench, NeurIPS 2023)
- Pfaff et al. (MeshGraphNets, ICLR 2021)
- Nabian et al. (arXiv:2510.15201, BMW/GM-style crash with PhysicsNeMo)
- Lahoz Navarro & Jehle et al. (Applied Sciences 2024) — frame physics-lint as
  the deterministic-violation gate complementing their BNN UQ approach
```

**The "what physics-lint did NOT catch" section is non-negotiable.** It is the same methodological-honesty move as sim-to-data Decision 19 and inverseops V3. Writing it well is what separates this work from a marketing demo. Examples to cover (v2 list, **plus two new bullets** from the trimmed framing):

- **(NEW)** physics-lint's public Field API does not natively accept particle clouds; the LagrangeBench path uses a private harness that *reapplies* the rule's structural identity, not the public rule itself. Validated against analytical fixtures to ε_harness_vs_public ≤ 10⁻⁴, but not equivalent to a public-API run.
- **(NEW)** The rotation-sweep ε_rot computation does not match PH-SYM-003's emitted quantity. PH-SYM-003 is infinitesimal scalar Lie-derivative; the harness computes global-finite multi-output equivariance. The harness emits a different quantity than PH-SYM-003 and labels it accordingly.
- Plasticity/irreversibility rules are not yet implemented (PH-CSH-* roadmap, separate issue).
- Contact-non-penetration on deforming meshes is not tested (no public checkpoint exists).
- Equivariance tests are statistical (over a finite set of rotation angles); they cannot prove equivariance, only fail to disprove it.
- LagrangeBench is fluid SPH, not solid impact — domain transfer is implied, not demonstrated.

### 5.2 Optional retrain (Day 3, only if buffer remains and budget allows)

If Days 1–2 finished in 10 hours total and Modal spend is below $15:

- Retrain MeshGraphNets on DeepMind `deforming_plate` for 5 epochs via NVIDIA's PhysicsNeMo PyTorch port.
- Estimated cost: 10–15 A100-hours, $15–25.
- The lint result here is **deliberately weak** — a partially-trained model violating physics. The headline becomes: *"physics-lint flagged PH-CON-002 violations on a 5-epoch checkpoint; flag count drops monotonically with training epochs (5/15/25), demonstrating the tool functions as a CI training-completeness gate."*

This is a strong optional addition but **not** worth blowing the budget for. Decision rule: only proceed if both (a) wall-clock buffer ≥6 hours remains, AND (b) Modal spend < $15.

### 5.3 Application integration (1 hour, mandatory; cover-letter paragraph rewritten per spec Appendix A)

Update three places:

1. **physics-lint repo README** — add a "Validated on" badge row linking to `_rollout_anchors/` after the merge.
2. **CV — physics-lint entry** — append: *"Validated against pretrained third-party SOTA surrogates via rollout-domain anchor extension (LagrangeBench GNS/SEGNN, NVIDIA PhysicsNeMo MeshGraphNet); see external_validation/_rollout_anchors/."*
3. **BMW cover letter (if not yet sent) — replace the Tier-2 smoke-project paragraph with the spec Appendix A.1 variant** (default — Gate A PASS):

> *"To anchor the linter against trained third-party surrogates, I extended physics-lint's external-validation framework with rollout-domain anchors. On the mesh side, the existing public Field/rule API consumed per-timestep materializations of NVIDIA PhysicsNeMo MeshGraphNet checkpoints (vortex shedding, Ahmed Body) without rule modification. On the particle side (LagrangeBench SEGNN/GNS, NeurIPS 2023), the rule API does not natively accept particle clouds; I reapplied the rule structural identities — finite-group equivariance for PH-SYM-001/002, conservation balance for PH-CON-001/002/003 — via a thin harness validated against analytical fixtures, both paths emitting SARIF in the same schema. PhysicsNeMo MGN is the closest open analogue to the production stack used in Nabian et al. arXiv:2510.15201. Extending physics-lint with nonlinear-structural-mechanics primitives (contact, plastic-strain irreversibility, frame indifference) is the natural first six months of the proposed PhD."*

**Gated alternatives** (use the variant matching actual gate outcomes at writeup time; full text in spec Appendix A):

- **Gate A PARTIAL** (mesh side via GridField resampling) → use spec Appendix **A.2**.
- **Gate D fallback** (PhysicsNeMo unusable, FNO on Darcy substituted) → use spec Appendix **A.3**. *Drops the Nabian sentence; the recognition hook is replaced by a stronger technical claim ("without rule modification or resampling").*
- **Gate A FAIL + Gate D PASS** (both-harnesses, no public-API mesh path) → use spec Appendix **A.4**. *Adds Field-protocol widening to the "first six months" forward vision, which is honest under this gate outcome — the gap was surfaced by the validation work itself.*

If the cover letter is already sent: hold this paragraph for the (likely) follow-up email to Jehle one week post-submission.

---

## 6. Risk register (replaces v2 §6 wholesale; verbatim from spec §8)

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| LagrangeBench JAX checkpoint loading non-trivial | Medium | Hour-2 JAX micro-gate (§7); Gate C fallback to read-only PH-CON-* path. |
| NGC API key auth fails mid-Day-2 | Low | Tested in Day 0 Q3 pre-flight. |
| MeshField cannot wrap DGL graph (Gate A) | Medium | GridField resampling fallback (PARTIAL); both-harnesses fallback (FAIL). Protocol PR explicitly **not** triggered in this branch. |
| Particle harness diverges from public API on fixtures (Gate B) | Medium | Stop-the-bus discipline + 5-minute fixture-sanity-check qualifier. Pre-registered tolerance 10⁻⁴ in `SCHEMA.md`. |
| Equivariance threshold band mis-calibrated for SEGNN | Medium | Pre-compute on a 5-trajectory pilot before full run; if SEGNN gives 10⁻⁴ instead of 10⁻⁶, document honestly and amend the band, but log the decision in `DECISIONS.md` transparently — do not silently retune. |
| Modal A100 quota / availability | Low | $30 hard cap; H100 fallback if A100 unavailable. |
| PhysicsNeMo NGC checkpoint format changed | Low–Medium | Day 0 NGC catalog read; Gate D FNO-on-Darcy fallback covers it cleanly. |
| Story risk: SEGNN doesn't pass equivariance to machine precision | Low (theory says it should) | If it doesn't, the headline becomes "physics-lint detects implementation deviations from theoretical equivariance in SEGNN" — still publishable, more interesting. |
| Sunk-cost pressure overrides Gate B FAIL stop-the-bus | Medium (always present under deadline) | Discipline encoded in spec verbatim; "refuse to generate SARIF you'd be unwilling to defend" line in spec §4.1 is the load-bearing reminder. |
| Retraining `deforming_plate` blows budget | Medium | Day 3 retrain is OPTIONAL; gated on $15 spend remaining and ≥6h buffer. |
| Rule false-pass on regular grids (the original Transolver concern) | Mitigated by design | LagrangeBench is irregular particles; PhysicsNeMo MGN is irregular meshes. FNO-on-Darcy fallback uses regular grids but the rule semantics are the same per-timestep public-API path. |
| Scope creep into protocol changes mid-branch | Medium | Explicit out-of-scope statement (spec §1.1, §2.2, §7) reproduced verbatim in this plan and `_rollout_anchors/README.md`. |
| **V1 rules with documented input-domain restrictions on out-of-domain data** *(class-level, added 2026-05-04 per DECISIONS.md D0-03)* | Medium | V1 rules with input-domain restrictions (e.g., PH-CON-001 returns SKIPPED on `pde != "heat"`) require harness-routed reapplication on out-of-domain rollouts. Documented in case-study READMEs as "structural-identity reapplication" rather than "rule ran without modification"; flagged in `_rollout_anchors/README.md` "What we are NOT claiming". PH-CON-001 on PhysicsNeMo NS is the surfaced instance; Day 2 watches for additional ones. |

---

## 7. Decision gates (replaces v2 §7 wholesale; verbatim from spec §6)

The v2 gates assumed a `lint_rollout(.npz)` API and v1.0-on-PyPI status that no longer apply.

### Gate A — Day 0 Q1: MeshField-from-DGL feasibility

| Verdict | Trigger | Action |
|---------|---------|--------|
| **PASS** | `MeshField(basis=reconstructed_basis, dofs=...)` works on one PhysicsNeMo NGC sample timestep | Mesh public-API path is live. Cross-stack claim holds in full form. |
| **PARTIAL** | MeshField fails; GridField after regular-grid resampling works | Mesh public-API path is live with a documented resampling step (cited in cover-letter paragraph variant A.2). Cross-stack claim holds, slightly weaker. |
| **FAIL** | Neither works | Cross-stack-via-public-API claim is **dropped**. Both case studies run through `_harness/`-private adapters. Cover-letter paragraph drops the "without rule modification" sentence and substitutes "via two domain-specific harnesses, both validated against analytical fixtures." Protocol-widening is **not** triggered in this branch — that's `feature/field-protocol-v1.1`, post-v1.0. |

### Gate B — Day 0 controlled-fixture validation

| Verdict | Trigger | Action |
|---------|---------|--------|
| **PASS** | ε_harness_vs_public ≤ 10⁻⁴ on all fixtures | Proceed to Day 1. |
| **APPROXIMATE** | 10⁻⁴ < ε ≤ 10⁻² on at least one fixture, ε ≤ 10⁻² on all | Document divergence in `SCHEMA.md`, proceed, explicit disclaimer in `_rollout_anchors/README.md` §"What we are NOT claiming". Honest-finding branch. |
| **FAIL** | ε > 10⁻² on at least one fixture | **Stop the bus.** Apply the 5-minute fixture-sanity-check qualifier (next paragraph). If the fixture is sound and the harness diverges, do not proceed to Day 1 LagrangeBench rollouts until the harness is fixed or the writeup is pivoted to mesh-only. |

**Fixture-sanity-check qualifier** (before declaring stop-the-bus): the controlled-fixture layer is itself new code written under time pressure on Day 0. Code under time pressure has bugs at a higher rate than mature code. Before declaring Gate B FAIL, run a 5-minute audit:

1. Is the fixture's known-by-construction quantity computed correctly? Re-verify by hand on the simplest fixture (e.g., 4-particle exact C₄ invariance).
2. Does the public-API rule emit what its docstring says it emits on the gridded equivalent? Cross-check against `tests/test_*` for the rule.
3. If the fixture is the bug source, fix the fixture and re-run the harness-vs-public-API check. This is a legitimate measurement-before-amendment loop, not a sunk-cost rationalisation.
4. After the fixture passes once, subsequent failures shift the prior toward "harness bug" and stop-the-bus tightens.

### Gate C — Day 1 hour 4: JAX checkpoint loading + headline visibility

| Verdict | Trigger | Action |
|---------|---------|--------|
| **PASS** | SEGNN and GNS both load; ε_C4 visibly distinct between models (machine-precision band vs ~10⁻² band) | Continue with PH-CON-* + dam break P1 if buffer. |
| **PARTIAL** | Exactly one model loads | Defrag — single-model writeup; story trimmed to "physics-lint structural-identity harness measures equivariance defect on [model]". Weaker but defensible. |
| **FAIL** | Neither model loads | Particle case study collapses to PH-CON-* on cached `.npz` only (read-only path; no `jax`/`haiku` model-loading required). No SEGNN-vs-GNS comparison. Story is weakest here but still defensible: "rollout-domain conservation-balance anchor on LagrangeBench TGV." Defer SEGNN-vs-GNS comparison to v1.1 backlog. |

### Hour-2 micro-gate (Day 1, environmental, separate from Gate C)

JAX-on-CUDA failure is the most common environmental-not-methodological failure mode and silently eats half a day. At hour 2 of Day 1, run `jax.devices()` inside the Modal container.

| Verdict | Trigger | Action |
|---------|---------|--------|
| **PASS** | Returns at least one A100 device | Proceed to Gate C trajectory. |
| **FAIL** | Returns CPU only or errors out | Pivot at hour 2 to either: (a) JAX-CPU read-only mode for PH-CON-* path (skip PH-SYM-* on Day 1, defer to Day 1.5 if buffer permits); or (b) Modal-image debugging side-quest (capped at 2h before falling back to (a)). **Do not let this consume hours 2–6 silently.** |

### Gate D — Day 2 hour 4: PhysicsNeMo NGC checkpoint usability

Three failure modes, two outcomes.

| Verdict | Trigger | Action |
|---------|---------|--------|
| **PASS** | Checkpoint downloads, inference runs, output matches NGC's shipped sample | Continue. P2 stretch PH-RES-001 attempted only if hour 4 leaves ≥3h buffer. |
| **FAIL — checkpoint unavailable** | NGC download / format incompatibility | Switch Case Study 02 to neuraloperator FNO on Darcy. FNO output is on a regular grid; GridField materialization is direct; **the public-API claim strengthens** (no resampling disclaimer needed). Cover-letter paragraph uses spec Appendix A.3. |
| **FAIL — values wrong** | Checkpoint downloads, inference runs, but `test_inference_matches_ngc_sample` diverges | Same fallback as the previous row (FNO on Darcy), **but** the failure log explicitly notes "NGC checkpoint reproduction failure" rather than presenting it as a "checkpoint unavailable" event. Honest record matters here — a reproduction-failure on the NGC checkpoint is itself a publishable methodological finding (separately from this work). |

**Not an option:** dropping Case Study 02 entirely and shipping a one-case-study writeup. The dual-audience structure is the strategic spine; one-stack writeup loses cross-stack consistency claim and weakens both audiences uniformly. FNO fallback is strictly better than collapse.

### Always-on rule

If a result surprises us (SEGNN's ε_C4 lands in the FLAG band, GNS's ε_C4 lands in the PASS band, harness reproduces public-API exactly to 10⁻¹⁵, etc.), **stop and audit the measurement before amending the narrative.** Measurement-before-amendment discipline applied to this validation work itself.

---

## 8. Out of scope (do not get tempted)

- Implementing PH-CSH-* (plasticity, contact, frame indifference) rules. That belongs in the **physics-lint roadmap issue** opened separately. This validation runs the existing v1.0 rules.
- Comparing against Transolver. No public checkpoint, retraining cost out of budget.
- ReGUNet reproduction. No public code.
- Crash-specific results. The honest framing is "validated on adjacent benchmarks (Lagrangian SPH, mesh-based NS)" with explicit acknowledgement that crash-specific contact + plasticity is out of public-checkpoint reach today — and that is precisely the gap the proposed PhD project would close.
- Custom dataset generation. Use shipped data only.
- Performance optimisation. Inference latency is not the story.
- **(NEW per spec §7) Modifying `physics_lint.field.*` public API.** Reproduces the spec §1.1 out-of-scope statement.
- **(NEW per spec §7) Stub resolution for PH-SYM-004 / PH-BC-002 / PH-NUM-001.** Lives on the parallel `chore/v1.0-stub-resolution` branch.
- **(NEW per spec §7) Field protocol widening for graph-native or particle data.** Lives on `feature/field-protocol-v1.1`, post-v1.0.
- **(NEW per spec §7) PyPI v1.0(-rc) publish.** Gated on this branch + stub resolution both landing.

---

## 9. Cross-review checklist (before publishing)

User's standard 5-round Claude/ChatGPT cross-review applies here. Specifically check:

- [ ] Are the equivariance thresholds defensible? (Pre-registered, not retroactively tuned.)
- [ ] Is the "what physics-lint did NOT catch" section honest enough that a Stuttgart/Munich reviewer would believe it?
- [ ] Are all checkpoint hashes, git SHAs, and Modal lockfiles committed? Reproducibility audit.
- [ ] Does the headline claim survive the one-sentence test: can you state the result honestly in 30 words?
- [ ] Cover-letter paragraph (Appendix A.1) — does it overclaim? Specifically, the phrase "the closest open analogue to BMW's MGN-on-LS-DYNA crash workflow" — is it defensible? (Yes, but only if Nabian et al. arXiv:2510.15201 is the BMW-adjacent reference, which we have established.)
- [ ] DECISIONS.md ≥10 entries on this branch (the wider 20-entry target lives at the `feature/rollout-anchors` + `chore/v1.0-stub-resolution` combined level).

---

## 10. Definition of done

This validation work is **done** when all of the following are true:

1. **(Replaced from v2 #1 per Appendix B)** `physics-lint/external_validation/_rollout_anchors/` merged via `feature/rollout-anchors` PR, with a new "rollout-domain F3 status" column in the 18-rule anchor matrix at `physics-lint/external_validation/README.md`.
2. `_rollout_anchors/README.md` headline claim is one falsifiable sentence supported by SARIF artifacts committed in the repo.
3. Two case studies, each with reproducible Modal entrypoint, SARIF output, README, and ≥3 unit tests on adapter contract. (Or one case study under Gate C/D collapse paths, with the framing-trim documented.)
4. **(Updated)** DECISIONS.md ≥10 entries on this branch (the wider 20-entry target lives at the `feature/rollout-anchors` + `chore/v1.0-stub-resolution` combined level).
5. CI workflow runs lint on cached `.npz` rollouts in <5 minutes (proof that physics-lint is genuinely a CI gate; `_rollout_anchors/.github/workflows/rollout-anchors.yml`).
6. **(Augmented)** Cross-review by ChatGPT (≥3 rounds) on the methodology, not just the prose. Notes live at `physics-lint-validation/reviews/` per spec §1.5 (planning-only; not snapshotted into physics-lint at merge).
7. The cover-letter paragraph (§5.3) survives a "would Jehle find this overclaiming?" test AND a "would the academic co-PI notice the harness-vs-public-API distinction?" test.
8. **(NEW from spec §9 item 4)** `_rollout_anchors/_harness/tests/fixtures/test_harness_vs_public_api.py` passes in CI with verdict (PASS / APPROXIMATE / FAIL) captured in DECISIONS.md.
9. **(NEW from spec §9 item 10)** `pyproject.toml` `[project.optional-dependencies] validation-rollout` extra defined; `pip install physics-lint` (without extras) remains slim and base test suite unaffected.
10. **(NEW)** `_rollout_anchors/README.md` cross-references the in-tree `docs/` snapshot per spec §1.5 option (b). The merged artifact is self-contained — no cross-repo dependency on `physics-lint-validation/`.

Time budget on this definition: 3 days wall-clock + 1 day buffer = 4 days max. If we hit Day 5, something is wrong with the plan, not the execution.

---

## Appendix A — Cross-review escalation list (verbatim from spec §10)

These are the methodology questions that go to ChatGPT cross-review (≥3 rounds, per §10 item 6). Notes at `physics-lint-validation/reviews/` per spec §1.5.

1. **v0.9 vs v1.0-rc version label.** Three rules ship as v1.0 stubs; this branch does not resolve them (out of scope per §8); `chore/v1.0-stub-resolution` does. Question: at the moment of merging this branch, is the right version label `0.9.0` or `1.0.0rc1`? User's instinct: `1.0.0rc1` is fine if there's a real rc cadence (4–6 weeks to 1.0 final). Open-ended → `0.9.0` is more honest. Cross-review should pressure-test which is overclaim-vs-underclaim.
2. **Cross-stack claim trim — too aggressive?** Spec §1.4 enumerates what we are NOT claiming. ChatGPT should adversarially read `_rollout_anchors/README.md` against `external_validation/README.md` and find the residual overclaim, if any.
3. **Cover-letter paragraph length.** Spec Appendix A's paragraph is longer than v2 §5.3's. ChatGPT pass should price the length against the close-reading defensibility it buys. If length must be cut, the Nabian sentence is more cuttable than the structural-identities sentence.
4. **PH-RES-001 P2 stretch — defensible?** Rule coverage is asymmetric across stacks: particle side runs 6 rules (PH-SYM-001/002 + PH-CON-001/002/003 + PH-BC-001), mesh side runs 4 rules in the base case (PH-CON-001/002/003 + PH-BC-001) plus PH-RES-001 stretch for 5. PH-SYM-001/002 are particle-only because LagrangeBench's discrete-rotation and reflection structures are well-defined on particles, while the mesh case studies (vortex shedding, Ahmed Body, FNO Darcy) don't have natural finite-group inputs to test against. Question for cross-review: is the asymmetry an honest reflection of what each stack supports, or does it weaken the cross-stack consistency claim by making the per-stack rule sets non-comparable? If the latter, drop PH-SYM-001/002 from the cross-stack claim and frame them as a particle-side-only contribution within the same writeup.

---

## Appendix B — v2-to-v3 mapping (audit trail)

For audit purposes — a cross-reviewer can run `diff -u physics-lint-validation-plan-v2.md 2026-05-01-rollout-anchor-extension-plan.md` and confirm only Appendix-B-driven changes appear.

| v2 § | v3 § | Treatment | Source |
|------|------|-----------|--------|
| §1 Strategic framing | §1 | Audience-A/B prose **verbatim**; sibling-repo language dropped; option-B framing inserted; spec §1.1 out-of-scope statement reproduced; v2's headline-result sentence replaced by spec §0 headline claim. | spec Appendix B item 1 |
| §2.1 Environment audit | §2.1 | Replaced — Q1/Q2/Q3 + controlled-fixture deliverable. | spec §5.1 |
| §2.2 Modal account checks | §2.2 | **Verbatim**, with paths qualified to the implementation tree. | carry-forward |
| §2.3 NGC API key | §2.3 | **Verbatim**. | carry-forward |
| §2.4 Repo skeleton | §2.4 | Replaced with implementation tree. | spec §2.1 verbatim |
| §3.1 Targets | §3.1 | **Verbatim**. | carry-forward |
| §3.2 step 1–3 | §3.2 step 1–3 | **Verbatim**, with hour-2 JAX micro-gate inserted into step 1 and `.npz` schema reproduced from spec §3.2. | carry-forward + spec §3.2 |
| §3.2 step 4 lint_rollout API | §3.2 step 4 | Replaced with particle-harness invocation (read-only + model-loading halves). | spec §3 |
| §3.2 step 5 rotation sweep | §3.2 step 5 | Mechanic verbatim from v2; relocation to particle harness; SARIF `properties.source = "rollout-anchor-harness"`; threshold-distinction note added. | spec §4.4 |
| §3.2 step 6 | §3.2 step 6 | **Verbatim** except example markdown drops PH-SYM-003 / PH-SYM-004 lines per in-scope rule list. | carry-forward + spec §1.2 |
| §3.3 Day 1 acceptance | §3.3 | **Verbatim** except DECISIONS.md count adjusted for the cumulative target. | carry-forward |
| §4.1 Targets | §4.1 | **Verbatim** with PH-NUM-002 deferred to v1.1 backlog and PH-RES-001 added as P2 stretch. | carry-forward + spec §1.2 |
| §4.2 step 1–3 | §4.2 step 1–3 | **Verbatim** with `test_inference_matches_ngc_sample` definition inlined. | carry-forward |
| §4.2 step 4 | §4.2 step 4 | Per-timestep public-API framing; rule list narrowed to in-scope rules. | spec §3 + §1.2 |
| §4.2 step 5 cross-stack | §4.2 step 5 | Replaced with trimmed structural-identities-held framing. | spec §3.3 |
| §4.3 Day 2 acceptance | §4.3 | **Verbatim** except PH-NUM-002 line removed. | carry-forward + spec §1.2 |
| §5.1 Writeup | §5.1 | **Verbatim** with 2 added "what physics-lint did NOT catch" bullets covering the trimmed framing. | carry-forward + spec §1.4 |
| §5.2 Optional retrain | §5.2 | **Verbatim**. | carry-forward |
| §5.3 Application integration | §5.3 | Cover-letter paragraph replaced; gated variants A.2 / A.3 / A.4 enumerated. | spec Appendix A |
| §6 Risk register | §6 | Replaced wholesale. | spec §8 verbatim |
| §7 Decision gates | §7 | Replaced with Gates A/B/C/D + hour-2 micro-gate. | spec §6 verbatim |
| §8 Out of scope | §8 | v2 verbatim list + 4 new items. | carry-forward + spec §7 |
| §9 Cross-review checklist | §9 | **Verbatim** with one item retitled "headline claim" and DECISIONS.md count updated. | carry-forward |
| §10 Definition of done | §10 | Item 1 replaced; items 4 and 6 lightly edited; 3 new items (8, 9, 10). | spec Appendix B + spec §9 |
| (none in v2) | Appendix A | Cross-review escalation list. | spec §10 verbatim |
| (none in v2) | Appendix B | This mapping table. | new audit aid |

---
