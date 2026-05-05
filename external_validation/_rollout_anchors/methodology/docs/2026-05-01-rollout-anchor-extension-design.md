# Rollout-anchor extension to physics-lint external_validation/ — design spec

**Status.** Brainstorming complete (2026-05-01, three-section design with confirmed user responses); design ready for implementation planning via `superpowers:writing-plans`.
**Date.** 2026-05-01.
**Predecessors.**
- v2 implementation plan handed into brainstorming (`physics-lint-validation: Implementation Plan v1.0`, owner Jane Yeung, dated 2026-05-01 input). Treated as input, not as ground truth — five pre-design findings invalidated portions of v2 §2.1, §3.2, §5.3 and forced the option-A/B/C reframing below.
- `physics-lint/external_validation/README.md` 18-of-18 anchor matrix (post-`b308b30`) — this work adds a new "rollout-domain F3 status" column to that matrix.
- `physics-lint/docs/superpowers/specs/2026-04-20-external-validation-design.md` — precedent for the brainstorming-to-writing-plans flow inside physics-lint.
- `physics-lint/src/physics_lint/field/_base.py`, `…/mesh.py`, `…/grid.py` — load-bearing for the "Field protocol stays untouched" constraint in §3.

**Successor.** Implementation plan v3 via `superpowers:writing-plans`, written to `physics-lint-validation/docs/2026-05-01-rollout-anchor-extension-plan.md`, scoped to the `feature/rollout-anchors` branch only (see §2.4 branch structure and §1.5 location table).

**Repo split.** This spec lives in the planning workspace `physics-lint-validation/`, not in physics-lint. Implementation lands in physics-lint via the `feature/rollout-anchors` PR; planning artifacts (this spec, the v3 plan, DECISIONS.md, cover-letter drafts, cross-review notes) live in physics-lint-validation across the lifetime of the work and are snapshotted into physics-lint at merge time per §1.5 option (b). See §1.5 for the full location-by-artifact-type table.

---

## 0. Executive summary

**Goal.** Extend `physics-lint/external_validation/` with a new F3 anchor type — rollouts from pretrained third-party SOTA neural PDE surrogates — populating the "rollout-domain F3 status" column for the six v1.0 rules where the anchor is meaningful (plus PH-RES-001 as a P2 stretch on the mesh side only; see §1.2), while leaving the public `physics_lint.field.*` API and the existing F1/F2/F3-by-fixture anchors untouched.

**Headline claim.** *On the mesh side (NVIDIA PhysicsNeMo MeshGraphNet), the existing public Field/rule API consumed per-timestep materializations of trained third-party output without rule modification. On the particle side (LagrangeBench SEGNN/GNS), the rule API does not natively accept particle clouds; the rule structural identities — finite-group equivariance for PH-SYM-001/002, conservation balance for PH-CON-001/002/003 — were reapplied via a thin private harness validated against analytical fixtures, both paths emitting SARIF in the same schema.* This claim is deliberately trimmed from the v2 plan's "ran without rule modification across two completely different stacks" framing, which was empirically false in a way detectable on a single careful read of `external_validation/README.md`.

**Scope of this spec.** The `feature/rollout-anchors` branch and its deliverables only. Stub resolution (PH-SYM-004 / PH-BC-002 / PH-NUM-001) and any Field-protocol widening are independent branches sequenced separately (§2.4).

**Time budget.** 3 days wall-clock + 1 day buffer = 4 days max. Modal A100 ceiling $30 (unchanged from v2). PyPI 1.0 publish is **out of scope** for this branch — gated on this work + stub resolution both landing.

**Deliverables.**
1. `external_validation/_rollout_anchors/` subdirectory with two case-study folders, two private adapters, shared SARIF emitter, controlled-fixture harness validation layer, and root README.
2. New "rollout-domain F3 status" column in `external_validation/README.md` 18-rule anchor matrix.
3. Updated cover-letter paragraph (Appendix A) with mesh-side / particle-side / Gate-D-PARTIAL variants pre-written.
4. Decision-gate log: lives at `physics-lint-validation/DECISIONS.md` during planning per §1.5; snapshotted into `physics-lint/external_validation/_rollout_anchors/docs/DECISIONS.md` at merge. ≥10 entries on this branch (the wider 20-entry target lives at the `feature/rollout-anchors` + `chore/v1.0-stub-resolution` combined level).

---

## 1. Framing & scope

### 1.1 What this work is, and what it isn't

**Is.** A rollout-domain F3 anchor for the rules whose emitted quantity is meaningful on a trained-model rollout. Specifically: a controlled-fixture-validated harness for particle data, plus per-timestep public-API consumption for mesh data. The contribution is the *rollout anchor* — not new rules, not protocol changes, not a separate validation framework.

**Isn't.** A "validation of physics-lint v1.0" in the sense that v2 of the plan implied. physics-lint's existing `external_validation/` already covers all 18 v1.0 rules with F1 (mathematical-legitimacy), F2 (correctness-fixture), and where present F3 (borrowed-credibility / published-baseline reproduction) anchors. Calling this work "validating physics-lint v1.0" overstates the delta. The honest framing: this work *fills a specific gap* — the rollout-domain F3 anchor that the existing analytical-fixture-and-published-baseline anchors do not cover.

**Explicit out-of-scope statement** (to be reproduced verbatim in the v3 plan §1 and in `_rollout_anchors/README.md` §1):

> *"This branch does not modify `physics_lint.field.*` public API. Any extension to the Field protocol — temporal axis, particle support, MeshField widening to non-scikit-fem mesh sources — is out of scope. Such extensions, if needed, would land on a separate `feature/field-protocol-v1.1` branch sequenced independently after v1.0 ships. Stub resolution for PH-SYM-004 / PH-BC-002 / PH-NUM-001 is out of scope and lives on the parallel `chore/v1.0-stub-resolution` branch."*

The negative statement is load-bearing: it stops scope creep on Day 2 when the temptation to "just widen MeshField a bit" appears.

### 1.2 Rules in scope

Six core rules + one P2 stretch. All are non-stub v1.0 rules whose emitted quantity is meaningful on a rollout.

| Rule | Domain (this anchor) | Emitted quantity | Path |
|------|----------------------|------------------|------|
| PH-SYM-001 | Particle (LagrangeBench) | C₄ finite-group equivariance defect ε_C4 | Particle harness, validated against fixture |
| PH-SYM-002 | Particle (LagrangeBench) | Z₂ reflection equivariance defect ε_refl | Particle harness, validated against fixture |
| PH-CON-001 | Both | Mass-balance defect (particle count + Σm; or divergence-free L² for mesh) | Particle harness; mesh public API |
| PH-CON-002 | Both | Kinetic-energy budget | Particle harness; mesh public API |
| PH-CON-003 | Both | Sign of dE/dt (monotone decay where physics demands) | Particle harness; mesh public API |
| PH-BC-001 | Both | Wall-trace magnitude on no-slip / non-penetration boundaries | Particle harness (dam break); mesh public API |
| **PH-RES-001 (P2 stretch)** | Mesh only (PhysicsNeMo) | Bachmayr–Dahmen–Oster norm-equivalence on momentum residual | Mesh public API; only attempted if Day 2 hour 4 leaves ≥3h buffer |

**Explicitly out of scope for this anchor:**

- **PH-SYM-003** — emits an *infinitesimal scalar-invariant SO(2) Lie-derivative*, not a global finite multi-output equivariance. The rotation-sweep ε_rot computation in v2 §3.2 step 5 is a different quantity. Including PH-SYM-003 in this anchor would either misrepresent what physics-lint emits or require adding a new global-finite-equivariance rule (out of scope for v1.0).
- **PH-SYM-004** — SKIP-always V1 stub. Running it on rollouts produces vacuous SARIF.
- **PH-BC-002** — PASS-with-stub-reason V1 stub. Same.
- **PH-NUM-001** — PASS-with-stub-reason V1 stub. Same.
- **PH-NUM-002** — resolution sweep. Requires controlled multi-resolution inference harness work that is itself a separate deliverable; defer to v1.1 backlog unless trivial. (Trivial means: the PhysicsNeMo checkpoint is parameterized over mesh resolution out of the box; if not, defer.)
- **PH-RES-002, PH-RES-003, PH-POS-001, PH-POS-002, PH-CON-004, PH-VAR-002** — no in-scope rollout that targets them, or rule semantics not aligned with rollout-domain emission.

### 1.3 Models in scope

| Case study | Stack | Models | Datasets | Day |
|------------|-------|--------|----------|-----|
| 01 LagrangeBench | JAX/Haiku, particles | SEGNN, GNS | TGV 2D (P0); dam break (P1 if buffer) | 1 |
| 02 PhysicsNeMo MGN | PyTorch/DGL, mesh | NGC `modulus_ns_meshgraphnet` | Vortex shedding 2D (P0); Ahmed Body (P1 if buffer) | 2 |

**Gate D fallback:** if PhysicsNeMo NGC checkpoints are unusable (download fails, format mismatch, or `test_inference_matches_ngc_sample` divergence), Case Study 02 switches to **neuraloperator FNO on Darcy** as a public-API-trivial fallback. The dual-audience structure is preserved — one-case-study collapse is **not** an option (§6 Gate D).

### 1.4 What we are NOT claiming

A non-trivial section of `_rollout_anchors/README.md` is dedicated to enumerating these. The list is the methodological-honesty signal that matters for Audience A (Munich/Stuttgart SciML reviewers) — analogous to `external_validation/README.md`'s "Caveats on the 18-of-18 framing" block.

- We are not claiming physics-lint's rules ran without modification on particle data. They didn't — particle data goes through a private harness that *reapplies* the rule's structural identity, validated against analytical fixtures.
- We are not claiming the rotation-sweep ε_rot computation matches PH-SYM-003's emitted quantity. It doesn't — PH-SYM-003 is infinitesimal scalar Lie-derivative; the rotation sweep is global-finite multi-output equivariance. The harness emits a *different* quantity than PH-SYM-003 and labels it accordingly.
- We are not claiming equivariance proofs. Finite angle sweeps cannot prove equivariance; they can only fail to disprove it within the sampled set.
- We are not claiming domain transfer to crash mechanics. LagrangeBench is fluid SPH; PhysicsNeMo is incompressible NS / RANS. Solid-impact / contact / plasticity is out of public-checkpoint reach today, and that gap is precisely what the proposed PhD project would close.
- We are not claiming PyPI v1.0 status. physics-lint at the time of this branch's merge remains 0.0.0.dev0 → v1.0-rc (or v0.9; see §10 open questions). PyPI publication is gated on this branch + stub resolution both landing.

### 1.5 Location and split between planning and implementation

This work is split across two repositories along the natural permission gradient — planning artifacts iterate privately in the working repo `physics-lint-validation/`, while the merged implementation lands publicly in `physics-lint/`. The split tracks that gradient deliberately: planning is private and high-iteration; implementation is public and stable.

| Artifact | Location | Reason |
|---|---|---|
| Design spec (this document) | `physics-lint-validation/docs/2026-05-01-rollout-anchor-extension-design.md` | Planning, iterates pre-merge, doesn't need to be in physics-lint history. |
| Implementation plan v3 | `physics-lint-validation/docs/2026-05-01-rollout-anchor-extension-plan.md` | Same. |
| `DECISIONS.md` (planning entries during execution) | `physics-lint-validation/DECISIONS.md` | Same — the in-flight scratchpad. |
| Cover-letter draft and revisions | `physics-lint-validation/cover-letters/` | Same. |
| ChatGPT cross-review notes | `physics-lint-validation/reviews/` | Same. |
| `_rollout_anchors/` subdirectory tree | `physics-lint/external_validation/_rollout_anchors/` | Implementation, lands via `feature/rollout-anchors` PR. |
| Per-case-study folders (`01-lagrangebench`, `02-physicsnemo-mgn` or `02-fno-darcy`) | `physics-lint/external_validation/_rollout_anchors/` | Same. |
| `_harness/` adapters and `SCHEMA.md` | `physics-lint/external_validation/_rollout_anchors/_harness/` | Same — the implementation's load-bearing pieces. |
| Controlled-fixture test code and fixtures | `physics-lint/external_validation/_rollout_anchors/_harness/tests/` | Tests live with what they test; fixtures committed alongside. |
| Cached `.npz` rollout files (≤50 MB) | `physics-lint/external_validation/_rollout_anchors/<case>/outputs/rollouts/` | CI replay needs them; commit via Git LFS if size pushes the budget. |
| Cached `.npz` rollout files (>50 MB) | HF Hub mirror, referenced by URL in adapter config | Don't bloat the physics-lint repo. |
| SARIF outputs (canonical) | `physics-lint/external_validation/_rollout_anchors/<case>/outputs/lint.sarif` | Committed evidence, small files. |
| Final case-study READMEs | `physics-lint/external_validation/_rollout_anchors/<case>/README.md` | Public, lands with the PR. |
| `_rollout_anchors/README.md` (root of the subdirectory) | `physics-lint/external_validation/_rollout_anchors/` | Public, includes one-line cross-reference to the in-tree snapshot at `_rollout_anchors/docs/`. |
| New "rollout F3 status" column on the anchor matrix | `physics-lint/external_validation/README.md` | Public, lands with the PR. |
| `RolloutField` adapter (or any new public Field surface) | **Does not exist.** | §1.1 / §2.2 / §3 confirmed: no public API surface added; the harness lives below the public API. Including this row deliberately to prevent Day-2 scope creep ("let me just add a small public RolloutField helper"). |

**Cross-reference at merge time — option (b), default.** When `feature/rollout-anchors` lands, the as-merged spec and the as-merged `DECISIONS.md` are copied into `physics-lint/external_validation/_rollout_anchors/docs/` as **frozen snapshots**. `_rollout_anchors/README.md` references the in-tree snapshots, not a path in `physics-lint-validation/`. This makes the merged artifact self-contained: physics-lint takes no cross-repo dependency on physics-lint-validation's stability or layout. The pre-merge planning copies in `physics-lint-validation/` continue to exist as the working scratchpad; the in-tree snapshots are authoritative for anyone reading the merged subdirectory after the PR merges.

**Why option (b) and not option (a) (cross-repo SHA reference).** Option (a) would have `physics-lint`'s `_rollout_anchors/README.md` reference a specific git SHA in `physics-lint-validation/`. That introduces a stability dependency: if `physics-lint-validation/` is ever reorganised, deleted, or made private, the public physics-lint reference becomes a broken pointer. Option (b)'s in-tree snapshot has no such dependency — the merged artifact is self-contained and physics-lint-validation can be reorganised freely without breaking anything in physics-lint. The cost (a few KB of duplicated text per merge) is trivial.

---

## 2. Architecture

### 2.1 Where the work lives

This is the **implementation tree** that lands in physics-lint via the `feature/rollout-anchors` PR. Planning artifacts (this spec, the v3 plan, in-flight DECISIONS.md, cover-letter drafts, cross-review notes) live in `physics-lint-validation/` per §1.5; the `docs/` subdirectory below holds **frozen snapshots** of the spec, plan, and DECISIONS.md copied in at merge time per §1.5 option (b).

```
physics-lint/
└── external_validation/
    ├── README.md                          # gains "rollout-domain F3 status" column
    └── _rollout_anchors/                  # NEW — this branch's primary deliverable
        ├── README.md                      # framing, headline, what-NOT-claimed; cross-references docs/
        ├── docs/                          # frozen snapshots at merge time per §1.5 option (b)
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

### 2.2 The Field protocol stays untouched

This is reasserted from §1.1 because it constrains §2.3 and §3 directly. The Field ABC at `src/physics_lint/field/_base.py` — `values()`, `at(x)`, `grad()`, `laplacian()`, `integrate(weight)`, `values_on_boundary()` — is single-timestep, single-field. We do not add a temporal axis. We do not add a particle subclass. We do not change `MeshField`'s scikit-fem `Basis` requirement. Adapters live in `_rollout_anchors/_harness/` (private), not in `src/physics_lint/field/*` (public).

If Gate A (§6) returns FAIL — i.e., MeshField cannot wrap PhysicsNeMo DGL output and GridField resampling is also infeasible — the cross-stack-via-public-API claim is **dropped**, not escalated to a protocol PR in this branch. Both case studies then run through `_harness/`-private adapters, and the framing falls back to "two domain-specific harnesses, both validated against analytical fixtures."

### 2.3 New optional-extras layout in pyproject.toml

```toml
[project.optional-dependencies]
validation-rollout = [
    "jax[cuda12]>=0.4.30",
    "jaxlib>=0.4.30",
    "dm-haiku>=0.0.12",
    "nvidia-physicsnemo>=0.5",     # or pinned as lockfile dictates
    "dgl>=2.0",
    "modal>=0.64",
    "gdown>=5.0",
]
```

Base `pip install physics-lint` remains slim (numpy, scipy, torch, pydantic, typer, rich). Rollout-anchor work activates via `pip install 'physics-lint[validation-rollout]'`. Modal images for the case-study `modal_app.py` entrypoints install this extra plus the per-case-study deps.

### 2.4 Branch structure

Three independent branches; this spec scopes only the first.

| Branch | Scope | Sequencing |
|--------|-------|------------|
| `feature/rollout-anchors` | This spec — _rollout_anchors/ subdirectory + README column | This work, 3-day wall clock |
| `chore/v1.0-stub-resolution` | PH-SYM-004 / PH-BC-002 / PH-NUM-001 implement-or-defer-or-remove decisions | Parallel, 1–2 day decision-and-cleanup, can land before or after this branch |
| `feature/field-protocol-v1.1` | Conditional, post-v1.0; widens Field protocol if Gate A FAIL forces it | NOT triggered in this branch under any gate outcome |

Both this branch and `chore/v1.0-stub-resolution` must land before the PyPI v1.0(-rc) tag. Order between them is fungible.

---

## 3. Adapter shapes & .npz schemas

### 3.1 Two adapters, both private

```
_rollout_anchors/_harness/
├── particle_rollout_adapter.py
├── mesh_rollout_adapter.py
└── sarif_emitter.py
```

**`mesh_rollout_adapter.py` — PhysicsNeMo path.** Materializes one timestep of a mesh rollout into a Field-API-compatible object. Preferred: `MeshField(basis=reconstructed_basis, dofs=node_values_at_t)` if the DGL graph can be coerced to a scikit-fem `Basis` (Gate A PASS). Fallback: `GridField(values=resampled, h=spacing, periodic=False)` if the output can be sampled onto a regular grid (Gate A PARTIAL). Rule kernels then consume per-timestep via the existing public `physics-lint check` API. **No new public surface.**

**`particle_rollout_adapter.py` — LagrangeBench path.** Two halves:

- **Read-only path** for PH-CON-001/002/003: reads cached `particle_rollout.npz`, computes mass / KE / dE/dt directly from particle positions and velocities, emits SARIF. Does not require the JAX model object.
- **Model-loading path** for PH-SYM-001/002: requires `jax`/`haiku` extras, loads SEGNN/GNS checkpoint, runs identity rollout from x₀, runs rotated rollout from R x₀, applies R⁻¹ to derotated rollout, computes ε_C4 / ε_refl over a finite angle set {0, π/4, π/2, π, 3π/2} for C₄ (and {identity, reflection} for Z₂), emits SARIF. Each emitted SARIF result has `properties.source = "rollout-anchor-harness"` so consumers can distinguish harness-emitted vs. public-rule-emitted SARIF.

**`sarif_emitter.py`** — shared schema producer used by both adapters. Same JSON keys as physics-lint's public SARIF (`runs[].results[].ruleId`, `message`, `locations`, `properties`); the `properties` object carries `source`, `harness_validation_passed`, `harness_vs_public_epsilon` (filled from Gate B), and `case_study` for downstream consumers.

### 3.2 Two .npz schemas

Per-domain, not unified. The two adapters are the only consumers.

**`particle_rollout.npz` — LagrangeBench:**

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

**`mesh_rollout.npz` — PhysicsNeMo (or FNO under Gate D fallback):**

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

Schemas are documented in `_rollout_anchors/_harness/SCHEMA.md` with the harness-vs-public-API tolerance (10⁻⁴; see §4) and the SARIF property definitions.

### 3.3 What "without modification" actually means

The cross-stack claim is precise:

- **Mesh path (PhysicsNeMo or Darcy fallback):** the existing public `physics-lint check` CLI runs per-timestep on materialized Field instances. Zero rule-source modification; rule kernels untouched. **This is the "ran without modification" half of the claim.**
- **Particle path (LagrangeBench):** the public rule API does not accept particle clouds. The harness *reapplies* the rule's structural identity at the particle level. The harness's output is *validated against the public-API output on the equivalent grid-discretised fixture* (Gate B, §4) so the structural-identities-held claim is falsifiable rather than asserted.

Both paths emit SARIF in the same schema. Both are anchored to the same rule IDs. Neither pretends to be the other.

---

## 4. Controlled-fixture harness validation layer

### 4.1 Why this is load-bearing

The trimmed cross-stack claim ("structural identities held across both stacks") rests entirely on the particle harness reapplying the public rule's structural identity to within a documented tolerance. Without a controlled-fixture validation step, the claim is structural-only — same JSON keys, no cross-validation that the values are comparable. With it, "harness reproduces public-API rule emission to within 10⁻⁴ on these fixtures" is a falsifiable, pre-registered, CI-runnable statement.

Time cost: **3–4 hours, on Day 0**, before any particle-side rollout work runs. Not deferred to writeup.

### 4.2 Fixture set

Under `_rollout_anchors/_harness/tests/fixtures/`:

1. **`c4_invariant_4particle.py`** — four particles at the vertices of a square centered on the origin, with C₄-invariant velocity assignment. Public-API path: discretise onto a 64×64 grid, run PH-SYM-001's rotation test on the gridded scalar field (e.g., a smooth bump centered on each particle). Particle harness path: apply C₄ to particle positions and velocities, compute ε_C4 directly. **Expected:** both paths emit ε_C4 ≤ 10⁻⁶ (machine precision; the configuration is exactly C₄-invariant by construction).
2. **`c4_perturbed_4particle.py`** — same configuration with one particle displaced by a known δ. **Expected:** both paths emit ε_C4 = O(δ); the two paths' values agree to within 10⁻⁴.
3. **`c4_grid_equivalent.py`** — shared discretisation utility (used by fixtures 1 and 2 to materialize the gridded equivalent on which the public-API path is run).
4. **`mass_conservation_fixture.py`** — fluid configuration with known mass at t₀ and t₁, where both paths must report the same defect within 10⁻⁴.
5. **`test_harness_vs_public_api.py`** — pytest-runnable assertion harness. Computes ε_harness_vs_public for each fixture × rule and asserts ≤ 10⁻⁴.

### 4.3 Tolerance bands and Gate B verdict

| ε_harness_vs_public | Verdict | Action |
|---------------------|---------|--------|
| ≤ 10⁻⁴ | **PASS** | Proceed to Day 1 LagrangeBench rollouts. |
| 10⁻⁴ < ε ≤ 10⁻² | **APPROXIMATE** | Document divergence in `SCHEMA.md`, proceed, disclaim explicitly in `_rollout_anchors/README.md` "What we are NOT claiming". This is the honest-finding branch. |
| > 10⁻² | **FAIL** | **Stop the bus.** See §6 Gate B for the fixture-sanity-check qualifier and the recovery procedure. |

The tolerance is documented in `SCHEMA.md`. It is **pre-registered** before fixtures are run; it is not retroactively tuned to match observed values. (Methodological-honesty discipline applied to the validation-of-validation step.)

### 4.4 Equivariance threshold pre-registration (separate from Gate B)

The Day-1 SEGNN-vs-GNS comparison uses a *separate* threshold band for ε_C4 / ε_refl interpretation:

| ε_rot or ε_refl | Verdict |
|-----------------|---------|
| ≤ 10⁻⁵ | PASS — machine-precision equivariance. SEGNN expected. |
| 10⁻⁵ < ε ≤ 10⁻² | APPROXIMATE — flagged, in approximate-equivariance band. GNS expected. |
| > 10⁻² | FAIL — equivariance broken. |

These thresholds are pre-registered before SEGNN/GNS runs. If the pilot shows SEGNN at, e.g., 10⁻⁴ instead of 10⁻⁶, the thresholds are **not** silently retuned; the divergence is logged in `DECISIONS.md` with a transparent explanation, and the band may be amended only with the discrepancy explicitly cited in the writeup.

**Note on the §4.2-vs-§4.4 threshold distinction.** §4.2 fixture #1 sets ε_C4 ≤ 10⁻⁶ as the *fixture-construction* tolerance: the 4-particle configuration is exactly C₄-invariant by construction, so observed ε is dominated by floating-point round-off and should land near machine epsilon. §4.4 sets ε_rot ≤ 10⁻⁵ as the *trained-model* PASS band: theoretically-equivariant SEGNN is exactly equivariant in theory, but in fp32 inference, ε is bounded by accumulated round-off ~10⁻⁶ to 10⁻⁷ over a rollout, comfortably under 10⁻⁵. The two thresholds measure different things — the fixture is a tighter sanity check on harness arithmetic; the trained-model band is a looser empirical envelope around theoretical equivariance — and are not interchangeable.

---

## 5. Sequencing

### 5.1 Day 0 (≤4h, CPU only, no Modal GPU spend)

- Audit Q1 — MeshField-from-DGL feasibility on one PhysicsNeMo NGC sample. Verdict feeds Gate A.
- Audit Q2 — PH-CON-001 emitted-quantity sanity check on that timestep.
- Audit Q3 — particle-harness model-loading split confirmed (cached `.npz` for PH-CON; `jax`/`haiku` extras for PH-SYM).
- Build controlled-fixture harness layer per §4 (3–4h).
- Run Gate B verdict on the fixture layer.
- Modal/NGC pre-flight: NGC API key tested; Modal A100 quota verified; secrets stored.
- DECISIONS.md entries 1–6 written.

### 5.2 Day 1 — LagrangeBench (6–8h, ~1h Modal A100)

- **Hour 0–2:** install LagrangeBench on Modal A100 image; download SEGNN + GNS TGV-2D checkpoints; verify hashes; **hour-2 JAX micro-gate** — `jax.devices()` must return A100. If not, see §6.
- **Hour 2–4:** generate identity rollouts → `particle_rollout.npz`; run particle harness read-only path (PH-CON-001/002/003); commit SARIF for the conservation half; **Gate C verdict**.
- **Hour 4–6:** run particle harness model-loading path for SEGNN — ε_C4, ε_refl across the pre-registered angle set. Repeat for GNS. Commit comparison SARIF + figures.
- **Hour 6–8:** if buffer, P1 dam break GNS for PH-BC-001. Otherwise write `case-studies/01-lagrangebench/README.md` and ship.

### 5.3 Day 2 — PhysicsNeMo MGN, or FNO under Gate D (4–6h, ~1h Modal A100)

- **Hour 0–2:** NVIDIA Modal container + `nvidia-physicsnemo` install; NGC download for `modulus_ns_meshgraphnet:v0.1`; verify hashes. **Sanity-check inference matches NGC's shipped sample input** — each NGC checkpoint ships with a sample input plus expected output; `test_inference_matches_ngc_sample` runs inference and compares against the shipped expected output to within a documented tolerance (default: max-abs-error ≤ 10⁻³ on velocity components, looser on pressure if NGC's tolerance is documented as such).
- **Hour 2–4:** generate vortex-shedding rollout → `mesh_rollout.npz`; run mesh adapter; per-timestep public-API physics-lint check on PH-CON-001/002/003 + PH-BC-001; commit SARIF; **Gate D verdict**.
- **Hour 4–6:** P1 Ahmed Body if buffer; **P2 stretch PH-RES-001 momentum residual** only if hour 4 leaves ≥3h. Write `case-studies/02-*/README.md`.

### 5.4 Day 3 — writeup, optional retrain, application integration (4–6h)

- Write `_rollout_anchors/README.md` (root): headline, two case-study summaries, cross-stack table, "what physics-lint did NOT catch" section, reproducibility section.
- Update `external_validation/README.md` 18-rule anchor matrix with new "rollout-domain F3 status" column.
- Application integration: cover-letter paragraph (Appendix A, mesh-side / particle-side / Gate-D-PARTIAL variants), CV bullet, physics-lint root README "validated on" link.
- Optional retrain: gated on (wall-clock buffer ≥ 6h) AND (Modal spend < $15) per v2 §5.2 logic, unchanged.
- Verify DECISIONS.md ≥ 10 entries for this branch (the wider 20-entry target lives at the `feature/rollout-anchors` + `chore/v1.0-stub-resolution` combined level).

### 5.5 Modal budget table

| Day | Modal A100 hours | Cumulative spend cap |
|-----|------------------|----------------------|
| Day 0 | 0 (CPU) | $1 (auth tests) |
| Day 1 | ~1h LagrangeBench infer | $4–5 |
| Day 2 | ~1h PhysicsNeMo / FNO infer | $9–10 |
| Day 3 (no retrain) | 0 | $9–10 |
| Day 3 (with retrain) | +10–15h | $25–30 |

Hard cap $30; freeze new GPU runs at $25.

---

## 6. Decision gates

Replaces v2 §7 in its entirety. The v2 gates assumed a `lint_rollout(.npz)` API and v1.0-on-PyPI status that no longer apply.

### Gate A — Day 0 Q1: MeshField-from-DGL feasibility

| Verdict | Trigger | Action |
|---------|---------|--------|
| **PASS** | `MeshField(basis=reconstructed_basis, dofs=...)` works on one PhysicsNeMo NGC sample timestep | Mesh public-API path is live. Cross-stack claim holds in full form. |
| **PARTIAL** | MeshField fails; GridField after regular-grid resampling works | Mesh public-API path is live with a documented resampling step (cited in cover-letter paragraph variant). Cross-stack claim holds, slightly weaker. |
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
| **FAIL — checkpoint unavailable** | NGC download / format incompatibility | Switch Case Study 02 to neuraloperator FNO on Darcy (§1.3). FNO output is on a regular grid; GridField materialization is direct; **the public-API claim strengthens** (no resampling disclaimer needed). Cover-letter paragraph uses Appendix A.3. |
| **FAIL — values wrong** | Checkpoint downloads, inference runs, but `test_inference_matches_ngc_sample` diverges | Same fallback as the previous row (FNO on Darcy), **but** the failure log explicitly notes "NGC checkpoint reproduction failure" rather than presenting it as a "checkpoint unavailable" event. Honest record matters here — a reproduction-failure on the NGC checkpoint is itself a publishable methodological finding (separately from this work). |

**Not an option:** dropping Case Study 02 entirely and shipping a one-case-study writeup. The dual-audience structure is the strategic spine; one-stack writeup loses cross-stack consistency claim and weakens both audiences uniformly. FNO fallback is strictly better than collapse.

### Always-on rule

If a result surprises us (SEGNN's ε_C4 lands in the FLAG band, GNS's ε_C4 lands in the PASS band, harness reproduces public-API exactly to 10⁻¹⁵, etc.), **stop and audit the measurement before amending the narrative.** Measurement-before-amendment discipline applied to this validation work itself.

---

## 7. Out of scope (explicit)

Reproduces and tightens v2 §8.

- Implementing PH-CSH-* (plasticity, contact, frame indifference) rules. Roadmap.
- Comparing against Transolver. No public checkpoint, retraining cost out of budget.
- ReGUNet reproduction. No public code.
- Crash-specific results. Honest framing: "validated on adjacent benchmarks (Lagrangian SPH, mesh-based NS)" — see §1.4.
- Custom dataset generation. Use shipped data only.
- Performance optimisation. Inference latency is not the story.
- **Modifying `physics_lint.field.*` public API.** §1.1 / §2.2.
- **Stub resolution for PH-SYM-004 / PH-BC-002 / PH-NUM-001.** Lives on `chore/v1.0-stub-resolution`, parallel branch.
- **Field protocol widening for graph-native or particle data.** Lives on `feature/field-protocol-v1.1`, post-v1.0.
- **PyPI v1.0(-rc) publish.** Gated on this branch + stub resolution both landing.

---

## 8. Risks & mitigations

Replaces v2 §6.

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| LagrangeBench JAX checkpoint loading non-trivial | Medium | Hour-2 JAX micro-gate (§6); Gate C fallback to read-only PH-CON-* path. |
| NGC API key auth fails mid-Day-2 | Low | Tested in Day 0 Q3 pre-flight. |
| MeshField cannot wrap DGL graph (Gate A) | Medium | GridField resampling fallback (PARTIAL); both-harnesses fallback (FAIL). Protocol PR explicitly **not** triggered in this branch. |
| Particle harness diverges from public API on fixtures (Gate B) | Medium | Stop-the-bus discipline + 5-minute fixture-sanity-check qualifier. Pre-registered tolerance 10⁻⁴ in `SCHEMA.md`. |
| Equivariance threshold band mis-calibrated for SEGNN | Medium | Pre-compute on a 5-trajectory pilot before full run; if SEGNN gives 10⁻⁴ instead of 10⁻⁶, document honestly and amend the band, but log the decision in `DECISIONS.md` transparently — do not silently retune. |
| Modal A100 quota / availability | Low | $30 hard cap; H100 fallback if A100 unavailable. |
| PhysicsNeMo NGC checkpoint format changed | Low–Medium | Day 0 NGC catalog read; Gate D FNO-on-Darcy fallback covers it cleanly. |
| Story risk: SEGNN doesn't pass equivariance to machine precision | Low (theory says it should) | If it doesn't, the headline becomes "physics-lint detects implementation deviations from theoretical equivariance in SEGNN" — still publishable, more interesting. |
| Sunk-cost pressure overrides Gate B FAIL stop-the-bus | Medium (always present under deadline) | Discipline encoded in this spec verbatim; "refuse to generate SARIF you'd be unwilling to defend" line in §4.1 is the load-bearing reminder. |
| Retraining `deforming_plate` blows budget | Medium | Day 3 retrain is OPTIONAL; gated on $15 spend remaining and ≥6h buffer. |
| Rule false-pass on regular grids (the original Transolver concern) | Mitigated by design | LagrangeBench is irregular particles; PhysicsNeMo MGN is irregular meshes. FNO-on-Darcy fallback uses regular grids but the rule semantics are the same per-timestep public-API path. |
| Scope creep into protocol changes mid-branch | Medium | Explicit out-of-scope statement (§1.1, §2.2, §7) reproduced verbatim in v3 plan and `_rollout_anchors/README.md`. |

---

## 9. Definition of done

This branch is **done** when all of the following are true:

1. `physics-lint/external_validation/_rollout_anchors/` merged via `feature/rollout-anchors` PR.
2. `_rollout_anchors/README.md` headline result is one falsifiable sentence supported by SARIF artifacts committed in the repo.
3. Two case studies, each with reproducible Modal entrypoint, SARIF output, README, and ≥3 unit tests on adapter contract. (Or one case study under Gate C/D collapse paths, with the framing-trim documented.)
4. `_rollout_anchors/_harness/tests/fixtures/test_harness_vs_public_api.py` passes in CI with verdict captured in `DECISIONS.md`.
5. `external_validation/README.md` 18-rule anchor matrix has a new "rollout-domain F3 status" column populated for the six rules in §1.2.
6. CI workflow lints cached `.npz` rollouts in <5 min (no GPU; `_rollout_anchors/.github/workflows/rollout-anchors.yml`).
7. `DECISIONS.md` ≥ 10 entries on this branch.
8. Cross-review by ChatGPT (≥3 rounds) on methodology, not just prose. Notes live at `physics-lint-validation/reviews/` per §1.5 (planning-only; not snapshotted into physics-lint at merge).
9. Cover-letter paragraph (Appendix A) survives a "would Jehle find this overclaiming?" test AND a "would the academic co-PI notice the harness-vs-public-API distinction?" test.
10. `pyproject.toml` `[project.optional-dependencies] validation-rollout` extra defined; `pip install physics-lint` (without extras) remains slim and base test suite unaffected.

Time budget: 3 days wall-clock + 1 day buffer = 4 days max. If we hit Day 5 with this branch incomplete, something is wrong with the plan, not the execution.

---

## 10. Open questions to escalate to ChatGPT cross-review

1. **v0.9 vs v1.0-rc version label.** Three rules ship as v1.0 stubs; this branch does not resolve them (out of scope per §7); `chore/v1.0-stub-resolution` does. Question: at the moment of merging this branch, is the right version label `0.9.0` or `1.0.0rc1`? User's instinct: `1.0.0rc1` is fine if there's a real rc cadence (4–6 weeks to 1.0 final). Open-ended → `0.9.0` is more honest. Cross-review should pressure-test which is overclaim-vs-underclaim.
2. **Cross-stack claim trim — too aggressive?** §1.4 enumerates what we are NOT claiming. ChatGPT should adversarially read `_rollout_anchors/README.md` against `external_validation/README.md` and find the residual overclaim, if any.
3. **Cover-letter paragraph length.** Appendix A's paragraph is longer than v2 §5.3's. ChatGPT pass should price the length against the close-reading defensibility it buys. If length must be cut, the Nabian sentence is more cuttable than the structural-identities sentence.
4. **PH-RES-001 P2 stretch — defensible?** Rule coverage is asymmetric across stacks: particle side runs 6 rules (PH-SYM-001/002 + PH-CON-001/002/003 + PH-BC-001), mesh side runs 4 rules in the base case (PH-CON-001/002/003 + PH-BC-001) plus PH-RES-001 stretch for 5. PH-SYM-001/002 are particle-only because LagrangeBench's discrete-rotation and reflection structures are well-defined on particles, while the mesh case studies (vortex shedding, Ahmed Body, FNO Darcy) don't have natural finite-group inputs to test against. Question for cross-review: is the asymmetry an honest reflection of what each stack supports, or does it weaken the cross-stack consistency claim by making the per-stack rule sets non-comparable? If the latter, drop PH-SYM-001/002 from the cross-stack claim and frame them as a particle-side-only contribution within the same writeup.

---

## Appendix A — Cover-letter paragraph variants

Three variants pre-written for the three Gate A outcomes. The §5.3 paragraph in v3 of the plan reproduces the variant matching the actual outcome at writeup time.

### A.1 Gate A PASS — full mesh public-API path

> *"To anchor the linter against trained third-party surrogates, I extended physics-lint's external-validation framework with rollout-domain anchors. On the mesh side, the existing public Field/rule API consumed per-timestep materializations of NVIDIA PhysicsNeMo MeshGraphNet checkpoints (vortex shedding, Ahmed Body) without rule modification. On the particle side (LagrangeBench SEGNN/GNS, NeurIPS 2023), the rule API does not natively accept particle clouds; I reapplied the rule structural identities — finite-group equivariance for PH-SYM-001/002, conservation balance for PH-CON-001/002/003 — via a thin harness validated against analytical fixtures, both paths emitting SARIF in the same schema. PhysicsNeMo MGN is the closest open analogue to the production stack used in Nabian et al. arXiv:2510.15201. Extending physics-lint with nonlinear-structural-mechanics primitives (contact, plastic-strain irreversibility, frame indifference) is the natural first six months of the proposed PhD."*

### A.2 Gate A PARTIAL — GridField resampling on mesh side

> *"…On the mesh side, the existing public Field/rule API consumed NVIDIA PhysicsNeMo MeshGraphNet checkpoints (vortex shedding, Ahmed Body) per-timestep via grid resampling, without rule modification. On the particle side …"* (rest unchanged from A.1).

### A.3 Gate D fallback — FNO on Darcy

> *"…On the mesh side, the existing public Field/rule API consumed neuraloperator FNO checkpoints on Darcy flow without rule modification or resampling. On the particle side (LagrangeBench SEGNN/GNS, NeurIPS 2023), the rule API does not natively accept particle clouds; I reapplied the rule structural identities — finite-group equivariance for PH-SYM-001/002, conservation balance for PH-CON-001/002/003 — via a thin harness validated against analytical fixtures, both paths emitting SARIF in the same schema. Extending physics-lint with nonlinear-structural-mechanics primitives (contact, plastic-strain irreversibility, frame indifference) is the natural first six months of the proposed PhD."*

(Nabian sentence dropped; the recognition hook is replaced by a stronger technical claim.)

### A.4 Gate A FAIL + Gate D PASS — both-harnesses, no public-API mesh path

> *"…I extended physics-lint's external-validation framework with rollout-domain anchors implemented via two domain-specific harnesses — one for graph-native PhysicsNeMo MeshGraphNet output, one for LagrangeBench particle clouds — both validated against analytical fixtures, both emitting SARIF in the same schema. PhysicsNeMo MGN is the closest open analogue to the production stack used in Nabian et al. arXiv:2510.15201. Widening physics-lint's public Field protocol to accept graph-native and particle data — and extending it with nonlinear-structural-mechanics primitives (contact, plastic-strain irreversibility, frame indifference) — is the natural first six months of the proposed PhD."*

(Adds Field-protocol widening to the "first six months" forward vision, which is honest under this gate outcome — the gap was surfaced by the validation work itself.)

---

## Appendix B — Diff list against v2 plan

For the writing-plans agent. v3 (`physics-lint-validation/docs/2026-05-01-rollout-anchor-extension-plan.md`) must execute these rewrites; the spec content above is authoritative on outcome but not on prose. Per §1.5, v3 lives in physics-lint-validation/ during planning and is snapshotted into physics-lint at merge.

- **§1 of v2** → drop sibling-repo language; insert option-B framing; insert explicit out-of-scope statement (this spec §1.1 verbatim).
- **§2.1 of v2** → replace Day-0 audit with Q1/Q2/Q3 + controlled-fixture deliverable (§5.1 here).
- **§2.4 of v2** → replace `physics-lint-validation/` repo skeleton with `external_validation/_rollout_anchors/` skeleton (§2.1 here).
- **§3.2 step 4 of v2** → replace `lint_rollout(.npz)` API with two-path: per-timestep public-API for mesh; private particle harness for LagrangeBench (§3 here).
- **§3.2 step 5 of v2** → rotation-sweep test moves from "physics-lint rule" to "particle harness inside `_rollout_anchors/_harness/`", emits SARIF with `properties.source = "rollout-anchor-harness"`, validated against fixture by Gate B (§4 here).
- **§4.2 step 5 of v2** → cross-stack claim trimmed to "structural identities held" framing (§3.3 here, §1.4 caveats).
- **§5.3 of v2** → cover-letter paragraph rewritten per Appendix A variants (this spec).
- **§6 of v2** → risk register replaced (§8 here).
- **§7 of v2** → decision gates replaced with Gates A/B/C/D + hour-2 micro-gate (§6 here).
- **§10 of v2** → "physics-lint-validation public repo" definition-of-done item replaced with "_rollout_anchors/ subdirectory merged into physics-lint via feature/rollout-anchors branch + new column in 18-rule anchor matrix README" (§9 here).

---
