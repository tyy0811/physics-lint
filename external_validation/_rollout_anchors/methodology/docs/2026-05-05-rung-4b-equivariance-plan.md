# Rung 4b — Cross-stack equivariance SARIF + writeup table (implementation plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compute first-step equivariance error ε on the existing 40 frozen rung-4a npzs (SEGNN-TGV2D + GNS-TGV2D) under rotation, reflection, and translation; emit per-(rule, transform_param, traj_index) SARIF rows at schema_version v1.1; render a tripartite-grouped cross-stack equivariance table; write the dated methodology writeup.

**Architecture:** Same generator/consumer split as rung 4a (D0-20). Modal-side generator computes ε(t) per (rule, transform_param, traj_index) and persists ε(t) npzs to Modal Volume (uniform single artifact tier: T_steps=1 for non-figure-subset, T_steps=100 for figure-subset). Consumer (`lint_eps_dir`) reads ε(t) npzs and emits HarnessResult rows. Sibling renderer (`render_eps_table.py`) reads v1.1 SARIFs, asserts schema_version, and renders tripartite-grouped output. PH-SYM-003 ships as substrate-incompatibility SKIP-with-reason via the same D0-19 §3.4 emission machinery as rung 4a's PH-CON-002 dissipative SKIP — trigger logic in the rule adapter, emission shape shared.

**Tech Stack:** Python 3.11+, NumPy, JAX/Haiku (LagrangeBench dependency, only loaded inside Modal entrypoints), SARIF v2.1.0, pytest, ruff (lint), codespell (docs), Modal CLI for one-shot volume download. Pre-commit hooks via `.venv/bin/pre-commit` (already configured at repo root).

**Predecessor:** Design doc `methodology/docs/2026-05-05-rung-4b-equivariance-design.md` (committed at `d9a8baa` on `feature/rung-4b-equivariance`). **Read it first** — it carries the full rationale for every choice in this plan, including the load-bearing claim (§1.2), tripartite evidence framing (§3.2), float32 floor threshold rationale (§3.3), uniform single-artifact-tier shape (§3.4), correctness primitives for transforms (§3.5), and the trigger-vs-emission separation discipline for PH-SYM-003 (§3.6).

**Branch:** All work on `feature/rung-4b-equivariance`. No commits on `master` for this rung — D0-21 is the rung's only D-entry, lands in `methodology/DECISIONS.md` on the feature branch, merges with the rest at PR.

**Working dir convention:** All paths absolute under `/Users/zenith/Desktop/physics-lint/`. Tests run via `cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest <args>`. Commits run with the same PATH override so pre-commit hooks resolve.

**Module import convention:** harness modules use `from external_validation._rollout_anchors._harness.<module> import <name>`. Tests use `--import-mode=importlib` per existing convention.

**Reuse from rung 4a (do not modify):** `_harness/sarif_emitter.py` (HarnessResult dataclass, run-level/result-level property contract, three-stage sha provenance) — extended in T6 for v1.1, *not rewritten*. `_harness/particle_rollout_adapter.py` (`load_rollout_npz`) — re-imported as the reference x_0 source. `_harness/lint_npz_dir.py` — left untouched; rung 4b ships a sibling `lint_eps_dir.py`. `methodology/tools/render_cross_stack_table.py` — left untouched; rung 4b ships a sibling `render_eps_table.py` per design §5.1.

---

## Task 0: Pre-registrations (D0-21 + .gitignore)

**Why first:** Per the project's pre-registration discipline, D0-21 lands before any code change. The .gitignore entry for `outputs/trajectories/*.npz` lands at the same time so future commits don't accidentally bring in Modal-Volume-mirrored trajectory data.

**Files:**
- Modify: `external_validation/_rollout_anchors/methodology/DECISIONS.md`
- Modify: `external_validation/_rollout_anchors/01-lagrangebench/.gitignore`
- Create: `external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories/.gitkeep`

### T0.1: Append D0-21 to methodology/DECISIONS.md

- [ ] **Step 1: Find insertion point**

```bash
tail -10 /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/DECISIONS.md
```

Expected: last lines are D0-20's "**Realized — ...**" closing paragraph or `---` separator. If file does not end with `---`, prepend `\n---\n\n` to the new entry below.

- [ ] **Step 2: Append D0-21 entry**

Append this content to `external_validation/_rollout_anchors/methodology/DECISIONS.md` (preceded by `\n---\n\n` if needed):

```markdown
## D0-21 — 2026-05-05 — Rung 4b cross-stack equivariance (pre-registration)

**Question.** Rung 4b extends rung 4a's cross-stack conservation result to
equivariance: compute first-step ε under rotation/reflection/translation
on the existing 40 frozen rollouts; emit SARIF at schema_version v1.1;
render a tripartite-grouped cross-stack equivariance table. Multiple
sub-questions (threshold rationale, rule set, persistence, GPU class,
renderer choice, schema bump) need pre-registration before code lands so
the implementation can't drift from the design's load-bearing claims.

**Decision (pre-registered before any code change).**

Composite entry covering rung 4b's design surface. Full rationale lives
in `methodology/docs/2026-05-05-rung-4b-equivariance-design.md`
(committed at `d9a8baa`, branch `feature/rung-4b-equivariance`); this
entry names the load-bearing decisions for searchability in
DECISIONS.md.

1. **Load-bearing claim.** "physics-lint's PH-SYM rule schema, with
   equivariance thresholds set at the float32 numerical-precision floor
   (ε ≤ 10⁻⁵ PASS, 10⁻⁵ < ε ≤ 10⁻² APPROXIMATE, ε > 10⁻² FAIL), runs
   unmodified across SEGNN-TGV2D and GNS-TGV2D rollouts; per-stack ε
   values are emitted in the same SARIF schema as 4a (schema_version
   v1.1) and reported as observed." Frozen at design time. Robust to
   both probe outcomes (SEGNN at floor expected; GNS at floor or in
   APPROXIMATE band — secondary finding wording adapts post-hoc).
2. **Threshold rationale.** Float32 numerical-precision floor (~10⁻⁷/op
   accumulating with depth and conditioning). Architecture-agnostic, no
   cross-paper citation needed at threshold layer. Helwig et al. ICML
   2023 demoted to writeup body context for interpreting observed GNS
   values.
3. **Rule set (γ).** PH-SYM-001 active at θ ∈ {π/2, π, 3π/2} +
   construction-trivial smoke at θ=0; PH-SYM-002 active at y-axis
   reflection through box center; PH-SYM-003 SKIP-with-reason
   ("PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with
   original"); PH-SYM-004 active at t=(L/3, L/7), construction-trivial
   PASS at machine-zero.
4. **Tripartite evidence framing in writeup body and renderer output:**
   architectural-evidence rows / construction-trivial rows /
   substrate-incompatible-SKIP. Prevents the reader from interpreting
   four PASSes as the same kind of evidence.
5. **PH-SYM-003 / PBC-square IS analogous to PH-CON-002 / dissipative**
   under D0-18's skip-with-reason mechanism. Meta-correction: this was
   missed at the brainstorm's non-claim-5 framing, surfaced at design
   pass not mid-execution. Recording the correction makes the
   writeup-framing-baked-into-design discipline self-validating.
6. **Trigger-vs-emission separation for SKIP paths:** rule-specific
   trigger logic ("substrate has periodic boundaries AND rotation angle
   is non-trivial-symmetry of substrate cell") in
   `symmetry_rollout_adapter.py`; shared D0-19 §3.4 emission machinery
   (skip_reason as guaranteed-identical-across-rows) populates the
   SARIF row. Same emission shape as PH-CON-002, different trigger
   source.
7. **Reportable ε.** First-step scalar ε per (rule, transform_param,
   traj_index) in SARIF (load-bearing); ε(t) over T=100 in writeup
   figure (supplementary, not threshold-classified). Compared quantity:
   positions only (matches LagrangeBench's published position-MSE
   metric). Per-particle aggregation: RMS-across-particles
   (ε = sqrt(mean_i ‖R⁻¹ r'_i,1 - r_i,1‖²)).
8. **Pipeline shape (II) with uniform single-tier artifact:** every
   (rule, transform_param, traj_index) tuple persists one ε(t) npz;
   T_steps=1 for non-figure-subset (~hundreds of bytes), T_steps=100
   for figure-subset (~few KB, 6 traces total: 1 angle × 3 trajs ×
   2 stacks). Single schema, single consumer code path
   (`lint_eps_dir` reads `eps_t[0]` always; figure renderer filters
   for `len(eps_t) > 1`). Preserves rung 4a's single-artifact-tier
   structure.
9. **Persistence.** Modal-Volume-only with local mirror (gitignored),
   matching rung 4a's pattern. SARIF + verdicts committed; ε(t) npzs
   not committed. Rotated state intentionally non-persisted (lives in
   Modal-job memory only); recovery via sub-minute re-run.
10. **GPU class A10G** (matched to rung 4a). D0-17 amendment 1
    principle generalized from within-rollout sha consistency to
    GPU-class consistency — at float32-floor measurement scale,
    GPU-class FP behavior differences (TF32, FMA scheduling,
    kernel-fusion) sit at the same order as the threshold (~10⁻⁶–10⁻⁷).
    Matched A10G keeps measurement noise below the floor-vs-architectural
    -error distinction.
11. **SARIF schema_version v1.0 → v1.1 bump.** New rule ids: PH-SYM-001,
    PH-SYM-002, PH-SYM-003, PH-SYM-004. New per-row extra_properties:
    `eps_pos_rms` (None for SKIP), `transform_kind`, `transform_param`.
    v1.0 fields unchanged. Schema_version field on every SARIF row
    enables fail-loud renderer assertion. v1.0 SARIFs unreadable by
    v1.1 renderer; v1.1 SARIFs unreadable by v1.0 renderer.
12. **Sibling renderer choice.** Rung 4b ships
    `methodology/tools/render_eps_table.py`, separate from rung 4a's
    `render_cross_stack_table.py`. Each renderer focused on one schema
    version. Rationale: rung 4b's tripartite-grouped layout differs
    structurally from rung 4a's flat per-rule list; modifying rung 4a's
    renderer risks regressions on v1.0 behavior. Code-duplication is
    bounded; DRY-ification of formatting primitives via
    `methodology/tools/render_lib.py` is a forward-flag for n=3 schema
    versions, not preemptive at n=2.
13. **4-stage provenance for ε(t) npzs.** Generalizes D0-19's 3-stage
    shape: `physics_lint_sha_pkl_inference` and
    `physics_lint_sha_npz_conversion` carried unchanged from rung 4a's
    reference rollouts; `physics_lint_sha_eps_computation` (new)
    recorded at ε(t)-computation time; `physics_lint_sha_sarif_emission`
    recorded at SARIF-emission time (consumer-side). The four shas may
    be identical or distinct.

**Forward flags (recorded for future rungs / case studies):**

- *Float32→float64 threshold-rationale revisit:* the float32 floor
  argument is precision-dependent. Case study 02 (PhysicsNeMo MGN), if
  it runs at float64, needs threshold revisiting (float64 floor
  ~10⁻¹⁶ per op).
- *(rule, substrate) compatibility matrix generalization:* D0-18
  detects PH-CON-002 SKIP via dataset-name + system_class heuristics;
  PH-SYM-003 / PBC-square SKIP triggers on a different (rule,
  substrate) compatibility axis. Future rung could formalize a per-rule
  compatibility matrix in the rule schema rather than per-rule
  heuristic triggers.
- *Sibling-vs-extend renderer choice:* at n=3 schema versions, extract
  shared formatting primitives into `methodology/tools/render_lib.py`.
- *(I)-rejection framing precision:* the full-rotated-state pipeline
  (Option I in the design) was rejected for over-persistence-for-
  unclear-ROI under Modal-Volume cost, *not* commit-bulk. Future
  readers should not interpret it as a commit-bulk rejection.
- *Translation vector "non-grid-commensurate" wording:* L/3 and L/7
  are rational fractions of L; the structural argument doesn't depend
  on Diophantine incommensurability. Captured here for terminology
  precision.

**Realized.** [Filled in post-execution at the rung 4b table writeup
step, naming the merge sha that closes this rung's loop.]
```

- [ ] **Step 3: Verify entry parses cleanly**

```bash
cd /Users/zenith/Desktop/physics-lint && grep -c "^## D0-" external_validation/_rollout_anchors/methodology/DECISIONS.md
```

Expected: previous count + 1 (one new top-level D-entry).

- [ ] **Step 4: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/methodology/DECISIONS.md && git commit -m "methodology/DECISIONS.md D0-21: rung 4b equivariance pre-registration"
```

Expected: commit succeeds; codespell hook passes (acronyms like SO(2), PBC, FP, FMA, ICML are project-domain terms).

### T0.2: Add .gitignore entry for outputs/trajectories/

- [ ] **Step 1: Read current .gitignore**

```bash
cat /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/.gitignore
```

Expected output: existing patterns for `01-lagrangebench/outputs/rollouts/*.npz`, `02-physicsnemo-mgn/outputs/rollouts/*.npz`, `01-lagrangebench/outputs/figures/*`, `02-physicsnemo-mgn/outputs/figures/*` plus `.gitkeep` negations.

- [ ] **Step 2: Append trajectories pattern**

Append these lines to `external_validation/_rollout_anchors/.gitignore`:

```
01-lagrangebench/outputs/trajectories/*/*.npz
01-lagrangebench/outputs/trajectories/_local_mirror/**
!01-lagrangebench/outputs/trajectories/.gitkeep
```

The first pattern matches Modal-Volume-mirrored ε(t) npzs in any subdirectory (per design §3.6, npzs land in `outputs/trajectories/{model}_{dataset}_<eps_computation_sha>/eps_*.npz`). The second handles a `_local_mirror/` convention if used. The negation preserves the `.gitkeep` placeholder so the empty directory is committed.

- [ ] **Step 3: Create .gitkeep placeholder**

```bash
mkdir -p /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories && touch /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories/.gitkeep
```

- [ ] **Step 4: Verify gitignore matches as expected**

```bash
cd /Users/zenith/Desktop/physics-lint && touch external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories/test.npz && git status --short external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories/ && rm external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories/test.npz
```

Expected: only `?? external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories/.gitkeep` (test.npz hidden by gitignore). If `test.npz` is shown, the pattern didn't match — fix before committing.

- [ ] **Step 5: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/.gitignore external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories/.gitkeep && git commit -m "01-lagrangebench/outputs/trajectories: gitignore Modal-Volume-mirrored eps_t npzs (rung 4b)"
```

---

## Task 1: SCHEMA.md updates (ε(t) npz schema + SARIF schema_version v1.1)

**Files:**
- Modify: `external_validation/_rollout_anchors/_harness/SCHEMA.md`

### T1.1: Add §3.x ε(t) npz schema

- [ ] **Step 1: Find insertion point in SCHEMA.md**

```bash
grep -n "^## " /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/_harness/SCHEMA.md
```

Expected: top-level sections are `## 1. particle_rollout.npz`, `## 2. mesh_rollout.npz`, `## 3. SARIF property surface`, `## 4. Tolerances`. The new ε(t) npz schema lands as `## 1.5. eps_t.npz — rung 4b equivariance trajectory artifact`.

- [ ] **Step 2: Append §1.5 between particle/mesh rollout schemas**

Insert this section after the `## 1. particle_rollout.npz` block ends (find the end via `## 2. mesh_rollout.npz`). Insert just before `## 2.`:

````markdown
## 1.5. `eps_t.npz` — rung 4b equivariance trajectory artifact

Per design §3.4 and §4.2. Modal-Volume-only persistence; gitignored
locally; one npz per (rule, model, transform_param, traj_index).
Uniform schema for both T_steps=1 (non-figure-subset SARIF rows) and
T_steps=100 (figure-subset trajectory data) — single consumer code
path reads `eps_t[0]` always.

```python
{
    "eps_t":              ndarray,  # (T_steps,) fp32   ε at each forward step (T_steps ∈ {1, 100})
    "rule_id":            str,      # "PH-SYM-001" | "PH-SYM-002" | "PH-SYM-003" | "PH-SYM-004"
    "transform_kind":     str,      # "rotation" | "reflection" | "translation" | "identity" | "skip"
    "transform_param":    object,   # rotation: float (radians); reflection: str ("y_axis") ;
                                    # translation: tuple (tx, ty); identity / skip: None
    "traj_index":         int,      # parsed from filename, validated contiguous (D0-19 §)
    "model_name":         str,      # "segnn" | "gns"
    "dataset_name":       str,      # "tgv2d"
    "ckpt_hash":          str,      # SHA-256 of LB checkpoint dir (carried from rung 4a's reference rollout)
    "physics_lint_sha_pkl_inference":   str,  # carried from rung 4a's reference rollout npz
    "physics_lint_sha_npz_conversion":  str,  # carried from rung 4a's reference rollout npz
    "physics_lint_sha_eps_computation": str,  # NEW: physics-lint commit at ε(t)-computation time
    "skip_reason":        str | None,         # populated only for transform_kind="skip" (PH-SYM-003)
}
```

**Filename convention:**

```
outputs/trajectories/{model}_{dataset}_{eps_computation_sha[:10]}/eps_{rule_id}_{transform_kind}_{transform_param_str}_traj{NN}.npz
```

Where `{transform_param_str}` is:
- rotation: angle as `pi_2`, `pi`, `3pi_2`, `0` (slash-free for path safety)
- reflection: `y_axis`
- translation: `L_3_L_7`
- skip (SO(2)): `so2_continuous`

Example:
```
outputs/trajectories/segnn_tgv2d_8d9a8baa12/eps_PH-SYM-001_rotation_pi_2_traj07.npz
```

**Provenance discipline (4-stage shas).** Per design §4.2, the four
`physics_lint_sha_*` fields may be identical or distinct. The
ε_computation sha is recorded by the Modal entrypoint at job execution
time (read from the `git_sha` argument passed to the entrypoint, parallel
to rung 4a's `lagrangebench_rollout_*` entrypoints). Equality is allowed
but never required.

**T_steps semantics.** `T_steps=1` rows carry `eps_t.shape == (1,)`
with the first-step ε scalar. `T_steps=100` rows carry the full
ε(t) array for the writeup figure subset (1 angle × 3 trajs × 2 stacks).
The figure renderer filters for `len(eps_t) > 1`; SARIF emission
reads `eps_t[0]` regardless of T_steps.

**SKIP rows (PH-SYM-003 only).** `eps_t.shape == (0,)` (empty array)
or `eps_t.shape == (1,)` with `eps_t[0] = numpy.nan` — chosen so the
npz format remains uniform but the SKIP-vs-active distinction is
unambiguous on the consumer side. Implementation pins this in
`symmetry_rollout_adapter.write_eps_t_npz` (see T4.3); the consumer
checks `transform_kind == "skip"` first, then reads `skip_reason`.
````

- [ ] **Step 3: Verify the section parses cleanly as Markdown**

```bash
cd /Users/zenith/Desktop/physics-lint && grep -n "^## " external_validation/_rollout_anchors/_harness/SCHEMA.md
```

Expected: `## 1.5. eps_t.npz — rung 4b equivariance trajectory artifact` appears between `## 1.` and `## 2.`.

### T1.2: Add §3.5 — SARIF schema_version v1.1

- [ ] **Step 1: Find SARIF section**

```bash
grep -n "schema_version\|harness_sarif_schema_version\|^### 3" /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/_harness/SCHEMA.md
```

Expected: existing §3 covers v1.0 (rung 4a). Find the last subsection of §3 to anchor the new §3.5.

- [ ] **Step 2: Append §3.5 inside the SARIF section**

Append this content as the last subsection inside `## 3. SARIF property surface`:

````markdown
### 3.5. SARIF schema_version v1.1 — rung 4b PH-SYM extensions

v1.0 (rung 4a) emits PH-CON rule rows with extra_properties:
`traj_index`, `npz_filename`, `skip_reason` (D0-19 §3.4 dissipative
SKIP), `ke_initial`, `ke_final`.

v1.1 (rung 4b) adds:

| Field                 | Type            | Present on              | Notes |
|-----------------------|-----------------|-------------------------|-------|
| `eps_pos_rms`         | float \| None   | all PH-SYM rows         | None for SKIP rows; positive scalar for active rows |
| `transform_kind`      | str             | all PH-SYM rows         | "rotation" \| "reflection" \| "translation" \| "identity" \| "skip" |
| `transform_param`     | str             | all PH-SYM rows         | rendered as canonical string ("pi_2", "y_axis", "L_3_L_7", "so2_continuous", "0") |
| `eps_t_npz_filename`  | str             | all PH-SYM rows         | basename of the ε(t) npz on Modal Volume; same role as v1.0's `npz_filename` |

v1.0 fields (rule_id, level, message, raw_value, source, case_study,
dataset, model, ckpt_hash, traj_index, npz_filename, skip_reason where
applicable) remain unchanged. v1.0 SARIFs do not contain the four new
fields; v1.1 SARIFs do not contain v1.0's `ke_initial`/`ke_final` (those
are PH-CON-002-specific; PH-SYM SKIP uses `skip_reason` only).

**Schema-version field.** Every SARIF v1.1 emission records
`runs[0].properties.harness_sarif_schema_version = "1.1"`. The renderer
asserts on equality at read time. v1.0 SARIFs (with
`harness_sarif_schema_version = "1.0"`) cannot be rendered by the v1.1
renderer; v1.1 SARIFs cannot be rendered by the v1.0 renderer. Both
renderers fail-loud on mismatch.

**Backward compatibility.** v1.1 emitters do not write to v1.0 rule ids
(PH-CON-001/002/003); v1.0 emitters do not write v1.1 rule ids. Each
rung's case-study driver invokes one schema version; cross-rung
artifacts live in distinct SARIF files.
````

- [ ] **Step 3: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/SCHEMA.md && git commit -m "_harness/SCHEMA.md: rung 4b additions (§1.5 eps_t.npz, §3.5 SARIF v1.1)"
```

---

## Task 2: Harness primitives — `symmetry_rollout_adapter.py`

**Files:**
- Create: `external_validation/_rollout_anchors/_harness/symmetry_rollout_adapter.py`
- Create: `external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py`

The adapter holds rotation/reflection/translation primitives, the per-particle RMS aggregation, the so2 substrate-skip trigger, and the ε(t) computation. Each primitive gets a paired test before implementation. All primitives are pure-NumPy (no JAX dependency at the consumer side; JAX is only invoked from the Modal-side `compute_eps_t` step which calls the network forward).

### T2.1: Failing test for `rotate_about_box_center`

- [ ] **Step 1: Create test file with first failing test**

Create `external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py`:

```python
"""Unit tests for symmetry_rollout_adapter primitives.

Pure NumPy fixtures; no JAX, no Modal. Each primitive's contract is
asserted via hand-crafted synthetic-but-realistic fixtures per the
"test fixtures hand-crafted, not copied from production" discipline.
"""

from __future__ import annotations

import numpy as np
import pytest


def _box_center_4particle_fixture(L: float = 1.0):
    """Four particles at corners of a unit square inscribed in [0, L]², centered.

    Particles arranged at (L/2 ± 0.25, L/2 ± 0.25). C₄ rotation about
    box center maps corner_0 → corner_1 → corner_2 → corner_3 → corner_0.
    Velocities chosen so that the rotated velocity field is a known
    rotation of the original (C₄-symmetric vector field).
    """
    half = L / 2
    d = 0.25
    positions = np.array(
        [
            [half - d, half - d],  # corner 0 (bottom-left)
            [half + d, half - d],  # corner 1 (bottom-right)
            [half + d, half + d],  # corner 2 (top-right)
            [half - d, half + d],  # corner 3 (top-left)
        ],
        dtype=np.float32,
    )
    # Velocity field aligned with corner ordering: each particle has
    # velocity pointing 90° counterclockwise of its position-vector
    # from box center. Under C₄ rotation about box center, both
    # positions and velocities permute by the same C₄ action.
    velocities = np.array(
        [
            [+1.0, -1.0],  # corner 0: velocity pointing toward corner 1
            [+1.0, +1.0],  # corner 1: velocity pointing toward corner 2
            [-1.0, +1.0],  # corner 2: velocity pointing toward corner 3
            [-1.0, -1.0],  # corner 3: velocity pointing toward corner 0
        ],
        dtype=np.float32,
    )
    return positions, velocities, L


def test_rotate_about_box_center_pi_2_maps_corner0_to_corner1():
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        rotate_about_box_center,
    )

    positions, velocities, L = _box_center_4particle_fixture()
    rotated_pos, rotated_vel = rotate_about_box_center(
        positions=positions, velocities=velocities, theta=np.pi / 2, L=L
    )

    # C₄ rotation by π/2 about box center: corner_0 (bottom-left) → corner_1 (bottom-right).
    # The original positions array indexed [corner_0, corner_1, corner_2, corner_3];
    # after rotation, position 0 should land where corner_3 originally was.
    # (Standard math convention: rotation by +π/2 takes (1,0) → (0,1), so
    # bottom-left of box → bottom-right when measured from box center.)
    # Specifically: (-d, -d) about origin rotates by π/2 to (+d, -d), which is corner_1.
    np.testing.assert_allclose(
        rotated_pos[0],
        positions[1],  # original corner_1 position
        atol=1e-6,
        err_msg="C₄ rotation by π/2 should map corner_0 position to corner_1 position",
    )
    np.testing.assert_allclose(
        rotated_vel[0],
        velocities[1],  # original corner_1 velocity (vector field is C₄-equivariant)
        atol=1e-6,
        err_msg="C₄ rotation by π/2 should map corner_0 velocity to corner_1 velocity",
    )
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py::test_rotate_about_box_center_pi_2_maps_corner0_to_corner1 -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'external_validation._rollout_anchors._harness.symmetry_rollout_adapter'`.

### T2.2: Implement `rotate_about_box_center`

- [ ] **Step 1: Create symmetry_rollout_adapter.py with rotate_about_box_center**

Create `external_validation/_rollout_anchors/_harness/symmetry_rollout_adapter.py`:

```python
"""Rung 4b equivariance harness primitives.

Pure-NumPy transforms (rotation, reflection, translation) on
particle-rollout state, the substrate-skip trigger for PH-SYM-003
(SO(2) on PBC-square), the per-particle RMS aggregation primitive for
ε computation, and the ε(t) computation orchestrator.

JAX is *not* imported here — the network forward pass lives in the
Modal-side entrypoint (`01-lagrangebench/modal_app.py`), which feeds
this module the pre-computed pair (f¹(x_0), f¹(R x_0)). This separation
keeps the consumer-side dependency surface clean: rendering, linting,
and test infrastructure only need NumPy.

Per design §3.5: rotation/reflection pivot at box center (L/2, L/2);
velocities transform with the same R as positions; PBC `mod L` wrap
after every transform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def rotate_about_box_center(
    *,
    positions: NDArray[np.float32],
    velocities: NDArray[np.float32],
    theta: float,
    L: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Rotate positions and velocities by angle θ about the box center (L/2, L/2).

    Velocities transform with the same R as positions (canonical
    equivariance-test correctness — see design §3.5 item 2). PBC `mod L`
    wrap after rotation to handle floating-point excursions outside
    [0, L]² at cell boundaries.

    Parameters
    ----------
    positions : (N, 2) fp32
    velocities : (N, 2) fp32
    theta : float
        Rotation angle in radians.
    L : float
        Box side length. Periodic-square cell is [0, L]².

    Returns
    -------
    rotated_positions : (N, 2) fp32, in [0, L]² after PBC wrap
    rotated_velocities : (N, 2) fp32, no PBC wrap (velocity is a tangent vector)
    """
    cos_t = np.cos(theta).astype(np.float32)
    sin_t = np.sin(theta).astype(np.float32)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)

    box_center = np.array([L / 2, L / 2], dtype=np.float32)

    # Translate to box-center frame, rotate, translate back.
    rel_positions = positions - box_center
    rotated_rel = rel_positions @ R.T  # (N, 2) @ (2, 2)
    rotated_positions = (rotated_rel + box_center).astype(np.float32)
    # PBC wrap.
    rotated_positions = np.mod(rotated_positions, np.float32(L))

    # Velocities: rotate with the same R; no translation, no PBC wrap (tangent vector).
    rotated_velocities = (velocities @ R.T).astype(np.float32)

    return rotated_positions, rotated_velocities
```

- [ ] **Step 2: Run test to confirm it passes**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py::test_rotate_about_box_center_pi_2_maps_corner0_to_corner1 -v
```

Expected: PASS.

### T2.3: Add round-trip and identity tests for rotation

- [ ] **Step 1: Append two more tests to test_symmetry_rollout_adapter.py**

Append to the existing test file:

```python


def test_rotate_about_box_center_2pi_is_identity_within_floor():
    """Rotation by 2π must return positions to within float32 round-off."""
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        rotate_about_box_center,
    )

    positions, velocities, L = _box_center_4particle_fixture()
    rot_pos, rot_vel = rotate_about_box_center(
        positions=positions, velocities=velocities, theta=2 * np.pi, L=L
    )
    # The threshold here is the float32 floor — a few times 1e-7 per op,
    # and rotation about box center does ~10 fp ops per element. We allow
    # 1e-6 to leave headroom while still asserting cleanly above noise.
    np.testing.assert_allclose(rot_pos, positions, atol=1e-6)
    np.testing.assert_allclose(rot_vel, velocities, atol=1e-6)


def test_rotate_about_box_center_zero_is_exactly_identity():
    """Rotation by 0 must return positions exactly bit-equal — float32 cos(0)=1, sin(0)=0."""
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        rotate_about_box_center,
    )

    positions, velocities, L = _box_center_4particle_fixture()
    rot_pos, rot_vel = rotate_about_box_center(
        positions=positions, velocities=velocities, theta=0.0, L=L
    )
    # cos(0)=1, sin(0)=0 exactly in float32. Composition is exact.
    np.testing.assert_array_equal(rot_pos, positions)
    np.testing.assert_array_equal(rot_vel, velocities)
```

- [ ] **Step 2: Run all rotation tests**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py -k "rotate" -v
```

Expected: 3 PASSED. If `test_rotate_about_box_center_zero_is_exactly_identity` fails because float32 cos/sin behavior differs from expectation, drop the strict bit-equality assertion in favor of `atol=0` is too strict — fall back to `atol=1e-7` and document why.

- [ ] **Step 3: Commit T2.1–T2.3 together**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/symmetry_rollout_adapter.py external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py && git commit -m "_harness/symmetry_rollout_adapter: rotate_about_box_center (positions+velocities, PBC wrap)"
```

### T2.4: Failing test for `reflect_y_axis`

- [ ] **Step 1: Append failing test**

```python


def _reflection_symmetric_fixture(L: float = 1.0):
    """Four particles arranged y-axis-reflection-symmetric about x = L/2.

    Particles at (L/2 ± d, L/2 ± d) with the pair on the right being
    the mirror image of the pair on the left.
    """
    half = L / 2
    d = 0.25
    positions = np.array(
        [
            [half - d, half - d],  # left-bottom
            [half + d, half - d],  # right-bottom (mirror of left-bottom)
            [half - d, half + d],  # left-top
            [half + d, half + d],  # right-top (mirror of left-top)
        ],
        dtype=np.float32,
    )
    # Velocities mirror in x-component.
    velocities = np.array(
        [
            [+1.0, +0.5],
            [-1.0, +0.5],
            [+1.0, -0.5],
            [-1.0, -0.5],
        ],
        dtype=np.float32,
    )
    return positions, velocities, L


def test_reflect_y_axis_swaps_left_right_pairs():
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        reflect_y_axis,
    )

    positions, velocities, L = _reflection_symmetric_fixture()
    reflected_pos, reflected_vel = reflect_y_axis(
        positions=positions, velocities=velocities, L=L
    )
    # Reflection across x = L/2: pos_x' = L - pos_x.
    # Particle 0 (left-bottom) and particle 1 (right-bottom) swap.
    np.testing.assert_allclose(reflected_pos[0], positions[1], atol=1e-6)
    np.testing.assert_allclose(reflected_pos[1], positions[0], atol=1e-6)
    np.testing.assert_allclose(reflected_pos[2], positions[3], atol=1e-6)
    np.testing.assert_allclose(reflected_pos[3], positions[2], atol=1e-6)
    # Velocity reflection: v_x' = -v_x, v_y' = v_y.
    np.testing.assert_allclose(reflected_vel[0], np.array([-1.0, 0.5]), atol=1e-6)
    np.testing.assert_allclose(reflected_vel[1], np.array([+1.0, 0.5]), atol=1e-6)
```

- [ ] **Step 2: Run to confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py::test_reflect_y_axis_swaps_left_right_pairs -v
```

Expected: FAIL with `ImportError: cannot import name 'reflect_y_axis'`.

### T2.5: Implement `reflect_y_axis`

- [ ] **Step 1: Append `reflect_y_axis` to symmetry_rollout_adapter.py**

Append to `symmetry_rollout_adapter.py`:

```python


def reflect_y_axis(
    *,
    positions: NDArray[np.float32],
    velocities: NDArray[np.float32],
    L: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Reflect positions and velocities across the line x = L/2.

    Operation: pos'_x = L - pos_x; pos'_y = pos_y; vel'_x = -vel_x;
    vel'_y = vel_y. PBC `mod L` wrap on positions after operation
    (positions stay in [0, L]² for any input in [0, L]² but
    floating-point edge cases can push slightly outside).

    The "y-axis reflection" name follows the standard physics convention:
    reflection across an axis parallel to the y-direction (here, the
    line x = L/2 through box center). The component perpendicular to
    the axis (x-component) flips; the component parallel to the axis
    (y-component) is preserved.
    """
    reflected_positions = positions.copy()
    reflected_positions[:, 0] = np.float32(L) - reflected_positions[:, 0]
    reflected_positions = np.mod(reflected_positions, np.float32(L))

    reflected_velocities = velocities.copy()
    reflected_velocities[:, 0] = -reflected_velocities[:, 0]

    return reflected_positions, reflected_velocities
```

- [ ] **Step 2: Run test, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py::test_reflect_y_axis_swaps_left_right_pairs -v
```

Expected: PASS.

- [ ] **Step 3: Commit T2.4 + T2.5**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/symmetry_rollout_adapter.py external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py && git commit -m "_harness/symmetry_rollout_adapter: reflect_y_axis (x = L/2 reflection)"
```

### T2.6: Failing test for `translate_pbc`

- [ ] **Step 1: Append test**

```python


def test_translate_pbc_wraps_at_box_boundary():
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        translate_pbc,
    )

    L = 1.0
    positions = np.array(
        [
            [0.1, 0.2],
            [0.9, 0.5],  # near right boundary
            [0.5, 0.95],  # near top boundary
        ],
        dtype=np.float32,
    )
    velocities = np.zeros_like(positions)  # translation is identity on velocities
    t = (np.float32(1.0 / 3), np.float32(1.0 / 7))  # the design's (L/3, L/7) for L=1

    translated_pos, translated_vel = translate_pbc(
        positions=positions, velocities=velocities, t=t, L=L
    )

    # Particle 0: (0.1 + 1/3, 0.2 + 1/7) = (0.4333..., 0.3428...) — no wrap.
    np.testing.assert_allclose(translated_pos[0], np.array([0.1 + 1/3, 0.2 + 1/7]), atol=1e-6)
    # Particle 1: (0.9 + 1/3, 0.5 + 1/7) = (1.2333..., 0.6428...) — wraps in x to 0.2333.
    np.testing.assert_allclose(translated_pos[1], np.array([(0.9 + 1/3) - 1.0, 0.5 + 1/7]), atol=1e-6)
    # Particle 2: (0.5 + 1/3, 0.95 + 1/7) = (0.8333..., 1.0928...) — wraps in y to 0.0928.
    np.testing.assert_allclose(translated_pos[2], np.array([0.5 + 1/3, (0.95 + 1/7) - 1.0]), atol=1e-6)
    # Velocities unchanged.
    np.testing.assert_array_equal(translated_vel, velocities)
```

- [ ] **Step 2: Run, confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py::test_translate_pbc_wraps_at_box_boundary -v
```

Expected: FAIL with `ImportError: cannot import name 'translate_pbc'`.

### T2.7: Implement `translate_pbc`

- [ ] **Step 1: Append to symmetry_rollout_adapter.py**

```python


def translate_pbc(
    *,
    positions: NDArray[np.float32],
    velocities: NDArray[np.float32],
    t: tuple[float, float],
    L: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Translate positions by t = (tx, ty), then PBC `mod L` wrap.

    Translation is identity on velocities — the translation matrix is
    the identity on tangent vectors (per design §3.5 item 2).

    For the rung 4b construction-trivial smoke test, t = (L/3, L/7).
    Per design §3.5 translation specifics, the choice of t doesn't
    affect *whether* PH-SYM-004 passes (translation + PBC commute
    exactly is a substrate property). (L/3, L/7) is non-grid-
    commensurate to avoid accidental commensurability with structure
    in x_0.
    """
    t_arr = np.asarray(t, dtype=np.float32)
    translated_positions = np.mod(positions + t_arr, np.float32(L)).astype(np.float32)
    return translated_positions, velocities.copy()
```

- [ ] **Step 2: Run test, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py::test_translate_pbc_wraps_at_box_boundary -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/symmetry_rollout_adapter.py external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py && git commit -m "_harness/symmetry_rollout_adapter: translate_pbc (with mod L wrap, identity on velocities)"
```

### T2.8: Failing test for `eps_pos_rms`

- [ ] **Step 1: Append test**

```python


def test_eps_pos_rms_matches_hand_computed_4particle():
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        eps_pos_rms,
    )

    # Two states, 4 particles, 2D. Per-particle squared error:
    # particle 0: ‖(0.01, 0.02)‖² = 0.0001 + 0.0004 = 0.0005
    # particle 1: ‖(0.0, 0.0)‖²    = 0.0
    # particle 2: ‖(0.05, 0.0)‖²   = 0.0025
    # particle 3: ‖(0.0, 0.04)‖²   = 0.0016
    # mean of 4: (0.0005 + 0 + 0.0025 + 0.0016) / 4 = 0.00115
    # sqrt: ≈ 0.033912
    a = np.array([
        [0.10, 0.20],
        [0.30, 0.40],
        [0.50, 0.60],
        [0.70, 0.80],
    ], dtype=np.float32)
    b = a + np.array([
        [0.01, 0.02],
        [0.00, 0.00],
        [0.05, 0.00],
        [0.00, 0.04],
    ], dtype=np.float32)
    # ε = sqrt(mean_i ‖a_i - b_i‖²) — sign convention per design §3.4.
    expected = np.sqrt((0.0005 + 0.0 + 0.0025 + 0.0016) / 4)
    actual = eps_pos_rms(a=a, b=b)
    assert isinstance(actual, float)
    assert abs(actual - expected) < 1e-7, f"got {actual}, expected {expected}"


def test_eps_pos_rms_zero_when_arrays_equal():
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        eps_pos_rms,
    )

    a = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    assert eps_pos_rms(a=a, b=a.copy()) == 0.0
```

- [ ] **Step 2: Run, confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py -k "eps_pos_rms" -v
```

Expected: 2 FAILED with `ImportError: cannot import name 'eps_pos_rms'`.

### T2.9: Implement `eps_pos_rms`

- [ ] **Step 1: Append to symmetry_rollout_adapter.py**

```python


def eps_pos_rms(*, a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Per-particle RMS aggregation: ε = sqrt(mean_i ‖a_i - b_i‖²).

    Per design §3.4 and SCHEMA.md §3.x. `a` and `b` are (N, D) arrays
    of per-particle positions; the output is a scalar ε. Order:
    per-particle squared L2 norm → mean over N → sqrt.

    This is the "RMS-across-particles" aggregation; alternatives
    (mean-across-particles, max-across-particles) would change what
    the threshold means and are excluded by the design.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: a {a.shape} vs b {b.shape}")
    diff = a - b  # (N, D)
    per_particle_sq_norm = np.sum(diff * diff, axis=-1)  # (N,)
    return float(np.sqrt(np.mean(per_particle_sq_norm)))
```

- [ ] **Step 2: Run, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py -k "eps_pos_rms" -v
```

Expected: 2 PASSED.

- [ ] **Step 3: Commit T2.8 + T2.9**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/symmetry_rollout_adapter.py external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py && git commit -m "_harness/symmetry_rollout_adapter: eps_pos_rms (per-particle RMS aggregation per SCHEMA.md §3.x)"
```

### T2.10: Failing test for `so2_substrate_skip_trigger`

- [ ] **Step 1: Append test**

```python


def test_so2_substrate_skip_trigger_fires_on_periodic_square():
    """The SO(2) trigger should fire for any non-{0, π/2, π, 3π/2} angle."""
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        so2_substrate_skip_trigger,
    )

    # Continuous-rotation angle (e.g., π/4) on a periodic-square box
    # should always fire the SKIP trigger.
    assert so2_substrate_skip_trigger(theta=np.pi / 4, has_periodic_boundaries=True) is True
    assert so2_substrate_skip_trigger(theta=0.5, has_periodic_boundaries=True) is True


def test_so2_substrate_skip_trigger_does_not_fire_on_c4_angles():
    """C₄-angles {0, π/2, π, 3π/2} preserve the periodic-square cell."""
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        so2_substrate_skip_trigger,
    )

    for theta in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
        assert so2_substrate_skip_trigger(theta=theta, has_periodic_boundaries=True) is False, (
            f"trigger should not fire on C₄ angle {theta}"
        )


def test_so2_substrate_skip_trigger_does_not_fire_on_non_periodic_substrate():
    """If the substrate has no periodic boundaries, SO(2) is structurally measurable."""
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        so2_substrate_skip_trigger,
    )

    assert so2_substrate_skip_trigger(theta=np.pi / 4, has_periodic_boundaries=False) is False
```

- [ ] **Step 2: Run, confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py -k "so2_substrate_skip" -v
```

Expected: 3 FAILED with `ImportError`.

### T2.11: Implement `so2_substrate_skip_trigger`

- [ ] **Step 1: Append to symmetry_rollout_adapter.py**

```python


_C4_ANGLES = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi)


def so2_substrate_skip_trigger(
    *,
    theta: float,
    has_periodic_boundaries: bool,
    angle_tolerance: float = 1e-9,
) -> bool:
    """Return True if (rule, substrate) compatibility makes ε measurement
    structurally invalid for the given rotation angle.

    Per design §3.6 trigger-vs-emission separation: this function
    contains the rule-specific trigger logic. Emission (skip_reason
    population, raw_value=None, level="note") happens downstream in
    the consumer (lint_eps_dir.py).

    Trigger condition: substrate has periodic boundaries AND theta is
    not a non-trivial-symmetry angle of the substrate cell.

    For a periodic-square substrate, the cell-preserving rotations are
    C₄ = {0, π/2, π, 3π/2} (and 2π = 0). Any other angle rotates the
    cell to one that doesn't tile with the original — the rotated state
    is not a valid input to f, and ε computed from it is substrate-
    confounded rather than architectural.
    """
    if not has_periodic_boundaries:
        return False
    # theta normalized to [0, 2π) for comparison.
    theta_norm = float(theta) % (2 * np.pi)
    for c4_angle in _C4_ANGLES:
        if abs(theta_norm - c4_angle) < angle_tolerance:
            return False
    return True
```

- [ ] **Step 2: Run, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py -k "so2_substrate_skip" -v
```

Expected: 3 PASSED.

- [ ] **Step 3: Commit T2.10 + T2.11**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/symmetry_rollout_adapter.py external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py && git commit -m "_harness/symmetry_rollout_adapter: so2_substrate_skip_trigger (rule-specific trigger; emission elsewhere)"
```

---

## Task 3: ε(t) computation orchestrator + npz I/O

**Files:**
- Modify: `external_validation/_rollout_anchors/_harness/symmetry_rollout_adapter.py` (T3.1, T3.2)
- Modify: `external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py`

### T3.1: Failing test for `compute_eps_t_from_pair`

- [ ] **Step 1: Append test**

The Modal-side entrypoint is responsible for actually invoking the network forward; this primitive consumes pre-computed pairs of state arrays. We test it with synthetic pairs.

```python


def test_compute_eps_t_from_pair_t_equal_1():
    """T_steps=1: input is two (1, N, 2) arrays, output is (1,) eps_t array."""
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        compute_eps_t_from_pair,
    )

    # Reference and rotated-then-untransformed states; deliberately
    # offset by a known amount.
    reference = np.array([[[0.10, 0.20], [0.30, 0.40], [0.50, 0.60], [0.70, 0.80]]], dtype=np.float32)
    candidate = reference + np.array([[[0.01, 0.02], [0.0, 0.0], [0.05, 0.0], [0.0, 0.04]]], dtype=np.float32)

    eps_t = compute_eps_t_from_pair(reference=reference, candidate=candidate)
    assert eps_t.shape == (1,), f"expected shape (1,), got {eps_t.shape}"
    expected = np.sqrt((0.0005 + 0.0 + 0.0025 + 0.0016) / 4)
    np.testing.assert_allclose(eps_t[0], expected, atol=1e-7)


def test_compute_eps_t_from_pair_t_equal_3():
    """T_steps=3: input is two (3, N, 2) arrays, output is (3,) eps_t array."""
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        compute_eps_t_from_pair,
    )

    reference = np.zeros((3, 4, 2), dtype=np.float32)
    # Step 0: zero offset; step 1: one corner offset; step 2: all corners
    candidate = reference.copy()
    candidate[1, 0] = np.array([0.1, 0.0])  # only particle 0 differs at step 1
    candidate[2] = np.array([
        [0.1, 0.0],
        [0.0, 0.1],
        [0.0, 0.0],
        [0.1, 0.1],
    ], dtype=np.float32)

    eps_t = compute_eps_t_from_pair(reference=reference, candidate=candidate)
    assert eps_t.shape == (3,)
    np.testing.assert_allclose(eps_t[0], 0.0, atol=1e-7)
    # Step 1: one particle has ‖(0.1, 0.0)‖² = 0.01; mean over 4 = 0.0025; sqrt = 0.05
    np.testing.assert_allclose(eps_t[1], 0.05, atol=1e-6)
    # Step 2: per-particle sq-norms (0.01, 0.01, 0.0, 0.02); mean = 0.01; sqrt = 0.1
    np.testing.assert_allclose(eps_t[2], 0.1, atol=1e-6)


def test_compute_eps_t_from_pair_shape_mismatch_raises():
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        compute_eps_t_from_pair,
    )

    reference = np.zeros((1, 4, 2), dtype=np.float32)
    candidate = np.zeros((1, 5, 2), dtype=np.float32)  # wrong N
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_eps_t_from_pair(reference=reference, candidate=candidate)
```

- [ ] **Step 2: Run, confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py -k "compute_eps_t" -v
```

Expected: 3 FAILED with `ImportError: cannot import name 'compute_eps_t_from_pair'`.

### T3.2: Implement `compute_eps_t_from_pair`

- [ ] **Step 1: Append to symmetry_rollout_adapter.py**

```python


def compute_eps_t_from_pair(
    *,
    reference: NDArray[np.float32],
    candidate: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute ε(t) = sqrt(mean_i ‖reference_t,i - candidate_t,i‖²) for each t.

    Inputs are (T_steps, N_particles, D) arrays. Output is (T_steps,)
    fp32. The Modal-side entrypoint is responsible for producing the
    `reference` (= forward(x_0)) and `candidate` (= R⁻¹ forward(R x_0))
    arrays; this primitive handles the per-step RMS aggregation only.

    See SCHEMA.md §1.5 and design §3.4 for the artifact-tier shape.
    """
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: reference {reference.shape} vs candidate {candidate.shape}")
    diff = reference - candidate  # (T, N, D)
    per_particle_sq_norm = np.sum(diff * diff, axis=-1)  # (T, N)
    eps_t = np.sqrt(np.mean(per_particle_sq_norm, axis=-1))  # (T,)
    return eps_t.astype(np.float32)
```

- [ ] **Step 2: Run, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py -k "compute_eps_t" -v
```

Expected: 3 PASSED.

- [ ] **Step 3: Run all symmetry_rollout_adapter tests as a regression check**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py -v
```

Expected: all PASSED (rotate, reflect, translate, eps_pos_rms, so2 trigger, compute_eps_t — total ~12 tests).

- [ ] **Step 4: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/symmetry_rollout_adapter.py external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py && git commit -m "_harness/symmetry_rollout_adapter: compute_eps_t_from_pair (per-step RMS aggregation)"
```

### T3.3: ε(t) npz writer / reader

- [ ] **Step 1: Append npz I/O test**

```python


def test_write_eps_t_npz_then_read_round_trips(tmp_path):
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        read_eps_t_npz,
        write_eps_t_npz,
    )

    eps_t = np.array([4.3e-7, 4.4e-7, 4.5e-7], dtype=np.float32)
    written = write_eps_t_npz(
        out_dir=tmp_path,
        eps_t=eps_t,
        rule_id="PH-SYM-001",
        transform_kind="rotation",
        transform_param="pi_2",
        traj_index=7,
        model_name="segnn",
        dataset_name="tgv2d",
        ckpt_hash="abc123",
        physics_lint_sha_pkl_inference="aaaaaaaaaa",
        physics_lint_sha_npz_conversion="bbbbbbbbbb",
        physics_lint_sha_eps_computation="cccccccccc",
        skip_reason=None,
    )
    assert written.name == "eps_PH-SYM-001_rotation_pi_2_traj07.npz"
    assert written.exists()

    record = read_eps_t_npz(written)
    np.testing.assert_array_equal(record["eps_t"], eps_t)
    assert record["rule_id"] == "PH-SYM-001"
    assert record["transform_kind"] == "rotation"
    assert record["transform_param"] == "pi_2"
    assert record["traj_index"] == 7
    assert record["physics_lint_sha_eps_computation"] == "cccccccccc"
    assert record["skip_reason"] is None


def test_write_eps_t_npz_skip_path_records_skip_reason(tmp_path):
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        read_eps_t_npz,
        write_eps_t_npz,
    )

    written = write_eps_t_npz(
        out_dir=tmp_path,
        eps_t=np.array([np.nan], dtype=np.float32),
        rule_id="PH-SYM-003",
        transform_kind="skip",
        transform_param="so2_continuous",
        traj_index=0,
        model_name="segnn",
        dataset_name="tgv2d",
        ckpt_hash="abc123",
        physics_lint_sha_pkl_inference="aaaaaaaaaa",
        physics_lint_sha_npz_conversion="bbbbbbbbbb",
        physics_lint_sha_eps_computation="cccccccccc",
        skip_reason="PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with original",
    )
    record = read_eps_t_npz(written)
    assert record["transform_kind"] == "skip"
    assert record["skip_reason"].startswith("PBC-square breaks SO(2)")
    assert np.isnan(record["eps_t"][0])
```

- [ ] **Step 2: Run, confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py -k "write_eps_t_npz" -v
```

Expected: 2 FAILED with `ImportError`.

### T3.4: Implement npz I/O

- [ ] **Step 1: Append to symmetry_rollout_adapter.py**

```python


from pathlib import Path


def write_eps_t_npz(
    *,
    out_dir: Path,
    eps_t: NDArray[np.float32],
    rule_id: str,
    transform_kind: Literal["rotation", "reflection", "translation", "identity", "skip"],
    transform_param: str,
    traj_index: int,
    model_name: str,
    dataset_name: str,
    ckpt_hash: str,
    physics_lint_sha_pkl_inference: str,
    physics_lint_sha_npz_conversion: str,
    physics_lint_sha_eps_computation: str,
    skip_reason: str | None,
) -> Path:
    """Persist one ε(t) npz per SCHEMA.md §1.5.

    Filename: eps_{rule_id}_{transform_kind}_{transform_param}_traj{NN}.npz
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"eps_{rule_id}_{transform_kind}_{transform_param}_traj{traj_index:02d}.npz"
    out_path = out_dir / filename

    # numpy savez_compressed emits both arrays and python objects via
    # allow_pickle. We avoid pickle for the strings/scalars by using
    # 0-d numpy arrays of dtype=object only for the optional skip_reason.
    np.savez(
        out_path,
        eps_t=eps_t,
        rule_id=np.array(rule_id),
        transform_kind=np.array(transform_kind),
        transform_param=np.array(transform_param),
        traj_index=np.array(traj_index, dtype=np.int32),
        model_name=np.array(model_name),
        dataset_name=np.array(dataset_name),
        ckpt_hash=np.array(ckpt_hash),
        physics_lint_sha_pkl_inference=np.array(physics_lint_sha_pkl_inference),
        physics_lint_sha_npz_conversion=np.array(physics_lint_sha_npz_conversion),
        physics_lint_sha_eps_computation=np.array(physics_lint_sha_eps_computation),
        skip_reason=np.array(skip_reason if skip_reason is not None else "", dtype=str),
        skip_reason_present=np.array(skip_reason is not None, dtype=bool),
    )
    return out_path


def read_eps_t_npz(path: Path) -> dict:
    """Read an ε(t) npz back into a dict matching SCHEMA.md §1.5."""
    with np.load(path, allow_pickle=False) as data:
        skip_reason_present = bool(data["skip_reason_present"])
        record = {
            "eps_t": data["eps_t"].astype(np.float32),
            "rule_id": str(data["rule_id"]),
            "transform_kind": str(data["transform_kind"]),
            "transform_param": str(data["transform_param"]),
            "traj_index": int(data["traj_index"]),
            "model_name": str(data["model_name"]),
            "dataset_name": str(data["dataset_name"]),
            "ckpt_hash": str(data["ckpt_hash"]),
            "physics_lint_sha_pkl_inference": str(data["physics_lint_sha_pkl_inference"]),
            "physics_lint_sha_npz_conversion": str(data["physics_lint_sha_npz_conversion"]),
            "physics_lint_sha_eps_computation": str(data["physics_lint_sha_eps_computation"]),
            "skip_reason": str(data["skip_reason"]) if skip_reason_present else None,
        }
    return record
```

- [ ] **Step 2: Run, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py -k "write_eps_t_npz" -v
```

Expected: 2 PASSED.

- [ ] **Step 3: Full regression test on the adapter module**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py -v
```

Expected: all PASSED (~14 tests now).

- [ ] **Step 4: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/symmetry_rollout_adapter.py external_validation/_rollout_anchors/_harness/tests/test_symmetry_rollout_adapter.py && git commit -m "_harness/symmetry_rollout_adapter: write/read_eps_t_npz (uniform schema, T_steps=1 or 100)"
```

---

## Task 4: `lint_eps_dir` consumer

**Files:**
- Create: `external_validation/_rollout_anchors/_harness/lint_eps_dir.py`
- Create: `external_validation/_rollout_anchors/_harness/tests/test_lint_eps_dir.py`

Sibling to rung 4a's `lint_npz_dir.py`. Reads ε(t) npzs from a directory, validates per-(traj_index, rule, transform) shape, and emits HarnessResult rows. Reuses 4a's `EmptyNpzDirectoryError`-style fail-loud pattern and `D0-19 §3.4` skip_reason discipline.

### T4.1: Failing test for `lint_eps_dir` on a fixture directory

- [ ] **Step 1: Create test file with fixture-building helper + first test**

Create `external_validation/_rollout_anchors/_harness/tests/test_lint_eps_dir.py`:

```python
"""Tests for lint_eps_dir: ε(t) npz dir → HarnessResult rows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _make_eps_dir_with_one_active_row(tmp_path: Path) -> Path:
    """Create a fixture dir with one active PH-SYM-001 ε(t) npz."""
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        write_eps_t_npz,
    )

    eps_dir = tmp_path / "fixture_eps_dir"
    eps_dir.mkdir()
    write_eps_t_npz(
        out_dir=eps_dir,
        eps_t=np.array([4.3e-7], dtype=np.float32),
        rule_id="PH-SYM-001",
        transform_kind="rotation",
        transform_param="pi_2",
        traj_index=0,
        model_name="segnn",
        dataset_name="tgv2d",
        ckpt_hash="abc123",
        physics_lint_sha_pkl_inference="aaaaaaaaaa",
        physics_lint_sha_npz_conversion="bbbbbbbbbb",
        physics_lint_sha_eps_computation="cccccccccc",
        skip_reason=None,
    )
    return eps_dir


def test_lint_eps_dir_active_row_yields_one_harness_result(tmp_path):
    from external_validation._rollout_anchors._harness.lint_eps_dir import lint_eps_dir

    eps_dir = _make_eps_dir_with_one_active_row(tmp_path)
    results = lint_eps_dir(
        eps_dir=eps_dir,
        case_study="01-lagrangebench",
        dataset="tgv2d",
        model="segnn",
        ckpt_hash="abc123",
    )
    assert len(results) == 1
    r = results[0]
    assert r.rule_id == "PH-SYM-001"
    assert r.raw_value == pytest.approx(4.3e-7, abs=1e-12)
    assert r.level == "note"
    assert r.message.startswith("eps_pos_rms=")
    assert r.extra_properties["transform_kind"] == "rotation"
    assert r.extra_properties["transform_param"] == "pi_2"
    assert r.extra_properties["traj_index"] == 0
    assert r.extra_properties["eps_pos_rms"] == pytest.approx(4.3e-7, abs=1e-12)
    assert r.extra_properties["eps_t_npz_filename"] == "eps_PH-SYM-001_rotation_pi_2_traj00.npz"
    assert "skip_reason" not in r.extra_properties


def test_lint_eps_dir_empty_dir_raises():
    from external_validation._rollout_anchors._harness.lint_eps_dir import (
        EmptyEpsDirectoryError,
        lint_eps_dir,
    )

    with pytest.raises(EmptyEpsDirectoryError, match="No eps_.*\\.npz files"):
        lint_eps_dir(
            eps_dir=Path("/tmp/nonexistent_eps_dir_for_test"),
            case_study="01-lagrangebench",
            dataset="tgv2d",
            model="segnn",
            ckpt_hash="abc123",
        )
```

- [ ] **Step 2: Run, confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_lint_eps_dir.py -v
```

Expected: 2 FAILED with `ModuleNotFoundError`.

### T4.2: Implement `lint_eps_dir` (active-row path)

- [ ] **Step 1: Create lint_eps_dir.py**

Create `external_validation/_rollout_anchors/_harness/lint_eps_dir.py`:

```python
"""ε(t) npz dir → HarnessResult rows (rung 4b consumer).

Sibling to lint_npz_dir.py; reads ε(t) npzs (SCHEMA.md §1.5) and emits
SARIF v1.1 result rows. Single artifact tier (uniform schema for both
T_steps=1 and T_steps=100 npzs) per design §3.4.

Per design §3.6 trigger-vs-emission separation, the SO(2) substrate
trigger has already fired upstream in symmetry_rollout_adapter.py;
this module reads `transform_kind == "skip"` from the npz and emits
the skip_reason via the shared D0-19 §3.4 emission machinery (same
shape as PH-CON-002 dissipative SKIP).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from external_validation._rollout_anchors._harness.sarif_emitter import HarnessResult
from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
    read_eps_t_npz,
)


class EmptyEpsDirectoryError(Exception):
    """Raised when lint_eps_dir is invoked on a directory with no eps_*.npz files.

    Silent empty SARIF is a methodology hazard. Same fail-loud pattern as
    rung 4a's EmptyNpzDirectoryError in lint_npz_dir.py.
    """


def lint_eps_dir(
    *,
    eps_dir: Path | str,
    case_study: str,
    dataset: str,
    model: str,
    ckpt_hash: str,
) -> list[HarnessResult]:
    """Read all eps_*.npz files from `eps_dir`, emit one HarnessResult per file.

    Active rows: raw_value = eps_t[0] (first-step ε); message includes the
    eps_pos_rms scalar and transform_param.

    SKIP rows (PH-SYM-003): raw_value = None; message = "SKIP: <skip_reason>";
    extra_properties.skip_reason populated per D0-19 §3.4.
    """
    eps_dir = Path(eps_dir)
    npz_paths = sorted(eps_dir.glob("eps_*.npz"))
    if not npz_paths:
        raise EmptyEpsDirectoryError(
            f"No eps_*.npz files found in {eps_dir}. "
            f"Expected at least one ε(t) npz; run the Modal entrypoint to populate."
        )

    results: list[HarnessResult] = []
    for npz_path in npz_paths:
        record = read_eps_t_npz(npz_path)
        rule_id = record["rule_id"]
        transform_kind = record["transform_kind"]

        extra: dict[str, Any] = {
            "transform_kind": transform_kind,
            "transform_param": record["transform_param"],
            "traj_index": record["traj_index"],
            "eps_t_npz_filename": npz_path.name,
        }

        if transform_kind == "skip":
            # PH-SYM-003 substrate-incompatibility SKIP.
            extra["skip_reason"] = record["skip_reason"] or "(no reason)"
            extra["eps_pos_rms"] = None
            level: str = "note"
            message = f"SKIP: {record['skip_reason'] or '(no reason)'}"
            raw_value: float | None = None
        else:
            eps_first = float(record["eps_t"][0])
            extra["eps_pos_rms"] = eps_first
            level = "note"
            message = f"eps_pos_rms={eps_first:.3e} (transform={transform_kind} {record['transform_param']})"
            raw_value = eps_first

        results.append(
            HarnessResult(
                rule_id=rule_id,
                level=level,  # type: ignore[arg-type]
                message=message,
                raw_value=raw_value,
                case_study=case_study,
                dataset=dataset,
                model=model,
                ckpt_hash=ckpt_hash,
                extra_properties=extra,
            )
        )
    return results
```

- [ ] **Step 2: Run tests, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_lint_eps_dir.py -v
```

Expected: 2 PASSED.

### T4.3: SKIP-path test for `lint_eps_dir`

- [ ] **Step 1: Append SKIP test**

Append to `test_lint_eps_dir.py`:

```python


def test_lint_eps_dir_skip_row_yields_skip_reason_in_extra_properties(tmp_path):
    from external_validation._rollout_anchors._harness.lint_eps_dir import lint_eps_dir
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        write_eps_t_npz,
    )

    eps_dir = tmp_path / "skip_fixture"
    eps_dir.mkdir()
    write_eps_t_npz(
        out_dir=eps_dir,
        eps_t=np.array([np.nan], dtype=np.float32),
        rule_id="PH-SYM-003",
        transform_kind="skip",
        transform_param="so2_continuous",
        traj_index=0,
        model_name="segnn",
        dataset_name="tgv2d",
        ckpt_hash="abc123",
        physics_lint_sha_pkl_inference="aaaaaaaaaa",
        physics_lint_sha_npz_conversion="bbbbbbbbbb",
        physics_lint_sha_eps_computation="cccccccccc",
        skip_reason="PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with original",
    )

    results = lint_eps_dir(
        eps_dir=eps_dir,
        case_study="01-lagrangebench",
        dataset="tgv2d",
        model="segnn",
        ckpt_hash="abc123",
    )
    assert len(results) == 1
    r = results[0]
    assert r.rule_id == "PH-SYM-003"
    assert r.raw_value is None
    assert r.message.startswith("SKIP: ")
    assert "PBC-square breaks SO(2)" in r.message
    assert r.extra_properties["skip_reason"].startswith("PBC-square breaks SO(2)")
    assert r.extra_properties["eps_pos_rms"] is None
    assert r.extra_properties["transform_kind"] == "skip"


def test_lint_eps_dir_skip_reason_identical_across_rows_same_rule(tmp_path):
    """D0-19 §3.4 contract: skip_reason is guaranteed-identical across rows
    within (rule, stack). The lint_eps_dir consumer must preserve this —
    each PH-SYM-003 row carries the same skip_reason string."""
    from external_validation._rollout_anchors._harness.lint_eps_dir import lint_eps_dir
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        write_eps_t_npz,
    )

    eps_dir = tmp_path / "skip_multi"
    eps_dir.mkdir()
    skip_reason = "PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with original"
    for traj in range(3):
        write_eps_t_npz(
            out_dir=eps_dir,
            eps_t=np.array([np.nan], dtype=np.float32),
            rule_id="PH-SYM-003",
            transform_kind="skip",
            transform_param="so2_continuous",
            traj_index=traj,
            model_name="segnn",
            dataset_name="tgv2d",
            ckpt_hash="abc123",
            physics_lint_sha_pkl_inference="aaaaaaaaaa",
            physics_lint_sha_npz_conversion="bbbbbbbbbb",
            physics_lint_sha_eps_computation="cccccccccc",
            skip_reason=skip_reason,
        )

    results = lint_eps_dir(
        eps_dir=eps_dir,
        case_study="01-lagrangebench",
        dataset="tgv2d",
        model="segnn",
        ckpt_hash="abc123",
    )
    assert len(results) == 3
    skip_reasons = {r.extra_properties["skip_reason"] for r in results}
    assert len(skip_reasons) == 1, "skip_reason must be guaranteed-identical across rows (D0-19 §3.4)"
```

- [ ] **Step 2: Run, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_lint_eps_dir.py -v
```

Expected: 4 PASSED.

- [ ] **Step 3: Commit T4.1–T4.3**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/lint_eps_dir.py external_validation/_rollout_anchors/_harness/tests/test_lint_eps_dir.py && git commit -m "_harness/lint_eps_dir: ε(t) npz dir → HarnessResult rows (active + SKIP path, D0-19 §3.4)"
```

---

## Task 5: `sarif_emitter` v1.1 schema_version bump

**Files:**
- Modify: `external_validation/_rollout_anchors/_harness/sarif_emitter.py`
- Create: `external_validation/_rollout_anchors/_harness/tests/test_sarif_emitter_v1_1.py`

The existing `sarif_emitter.py` writes `harness_sarif_schema_version` as a run-level property. T5 adds a parameter to control the version (default unchanged at "1.0" so rung 4a's emit_sarif.py keeps working) and verifies that the v1.1 path emits the new run-level field correctly.

### T5.1: Failing test for emitting at v1.1

- [ ] **Step 1: Create test file**

Create `external_validation/_rollout_anchors/_harness/tests/test_sarif_emitter_v1_1.py`:

```python
"""Tests for sarif_emitter at schema_version v1.1 (rung 4b)."""

from __future__ import annotations

import json

from external_validation._rollout_anchors._harness.sarif_emitter import (
    HarnessResult,
    emit_sarif,
)


def _v1_1_run_level_properties() -> dict:
    return {
        "source": "rollout-anchor-harness",
        "harness_sarif_schema_version": "1.1",
        "physics_lint_sha_pkl_inference": "aaaaaaaaaa",
        "physics_lint_sha_npz_conversion": "bbbbbbbbbb",
        "physics_lint_sha_sarif_emission": "cccccccccc",
        "physics_lint_sha_eps_computation": "dddddddddd",
        "lagrangebench_sha": "eeeeeeeeee",
        "checkpoint_id": "segnn_tgv2d/best",
        "model_name": "segnn",
        "dataset_name": "tgv2d",
        "rollout_subdir": "rollouts/segnn_tgv2d_post_d03df3e",
    }


def test_emit_sarif_v1_1_writes_schema_version_string(tmp_path):
    out_path = tmp_path / "v1_1.sarif"
    results = [
        HarnessResult(
            rule_id="PH-SYM-001",
            level="note",
            message="eps_pos_rms=4.3e-07 (transform=rotation pi_2)",
            raw_value=4.3e-7,
            case_study="01-lagrangebench",
            dataset="tgv2d",
            model="segnn",
            ckpt_hash="abc123",
            extra_properties={
                "transform_kind": "rotation",
                "transform_param": "pi_2",
                "traj_index": 0,
                "eps_pos_rms": 4.3e-7,
                "eps_t_npz_filename": "eps_PH-SYM-001_rotation_pi_2_traj00.npz",
            },
        )
    ]
    emit_sarif(out_path=out_path, results=results, run_level_properties=_v1_1_run_level_properties())

    sarif = json.loads(out_path.read_text())
    run_props = sarif["runs"][0]["properties"]
    assert run_props["harness_sarif_schema_version"] == "1.1"
    assert run_props["physics_lint_sha_eps_computation"] == "dddddddddd"

    result_props = sarif["runs"][0]["results"][0]["properties"]
    assert result_props["transform_kind"] == "rotation"
    assert result_props["transform_param"] == "pi_2"
    assert result_props["eps_pos_rms"] == 4.3e-7
    assert result_props["eps_t_npz_filename"] == "eps_PH-SYM-001_rotation_pi_2_traj00.npz"


def test_emit_sarif_v1_0_path_still_works_unchanged(tmp_path):
    """Backward-compat: v1.0 emission with rung-4a-shaped properties unchanged."""
    out_path = tmp_path / "v1_0.sarif"
    results = [
        HarnessResult(
            rule_id="harness:mass_conservation_defect",
            level="note",
            message="raw_value=0.000e+00",
            raw_value=0.0,
            case_study="01-lagrangebench",
            dataset="tgv2d",
            model="segnn",
            ckpt_hash="abc123",
            extra_properties={"traj_index": 0, "npz_filename": "particle_rollout_traj00.npz"},
        )
    ]
    v1_0_run_props = {
        "source": "rollout-anchor-harness",
        "harness_sarif_schema_version": "1.0",
        "physics_lint_sha_pkl_inference": "aaaa",
        "physics_lint_sha_npz_conversion": "bbbb",
        "physics_lint_sha_sarif_emission": "cccc",
        "lagrangebench_sha": "dddd",
        "checkpoint_id": "segnn_tgv2d/best",
        "model_name": "segnn",
        "dataset_name": "tgv2d",
        "rollout_subdir": "rollouts/segnn_tgv2d_post_d03df3e",
    }
    emit_sarif(out_path=out_path, results=results, run_level_properties=v1_0_run_props)

    sarif = json.loads(out_path.read_text())
    run_props = sarif["runs"][0]["properties"]
    assert run_props["harness_sarif_schema_version"] == "1.0"
    # v1.0 must not contain the new v1.1 sha field.
    assert "physics_lint_sha_eps_computation" not in run_props
```

- [ ] **Step 2: Run, confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_sarif_emitter_v1_1.py -v
```

Expected: depending on the existing emitter's flexibility, either both PASS (if the emitter already accepts arbitrary run_level_properties dicts and copies extra_properties unchanged) or one or both FAIL (if the emitter doesn't accept arbitrary keys). Investigate failure mode before fixing.

### T5.2: Adjust `sarif_emitter.py` if needed for v1.1 properties

- [ ] **Step 1: Inspect current emitter contract**

```bash
grep -n "run_level_properties\|extra_properties\|schema_version" /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/_harness/sarif_emitter.py
```

Expected: the emitter already accepts `run_level_properties: dict[str, Any]` and copies `extra_properties` to `result.properties` (rung 4a's pattern). If a fixed allow-list rejects v1.1 keys, find that gate.

- [ ] **Step 2: Make minimal modification (only if T5.1 fails)**

If both T5.1 tests pass, skip this step. Otherwise: relax any allow-list restrictions on run_level_properties keys to accept any string-keyed dict (the SARIF spec permits arbitrary properties); preserve the schema_version assertion. Specifically: do not enumerate keys in the emitter's run-level write path — use `runs[0]["properties"] = dict(run_level_properties)` as the pass-through.

If a code change is needed, it should be a few-line diff. Verify by re-running the test.

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_sarif_emitter_v1_1.py -v
```

Expected: 2 PASSED.

- [ ] **Step 3: Confirm rung 4a's existing tests still pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/ -v
```

Expected: all PASSED, including rung 4a's existing tests (`test_lint_npz_dir.py`, `test_d0_18_dissipative_skip.py`, etc.).

- [ ] **Step 4: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/sarif_emitter.py external_validation/_rollout_anchors/_harness/tests/test_sarif_emitter_v1_1.py && git commit -m "_harness/sarif_emitter: schema_version v1.1 support (PH-SYM extra_properties pass-through)"
```

---

## Task 6: Case-study driver — `emit_sarif_eps.py`

**Files:**
- Create: `external_validation/_rollout_anchors/01-lagrangebench/emit_sarif_eps.py`
- Create: `external_validation/_rollout_anchors/01-lagrangebench/tests/test_emit_sarif_eps.py`

Sibling to rung 4a's `emit_sarif.py`. Reads ε(t) npzs from a rung-4b ε(t) directory on local mirror, assembles run-level properties (per design §4.2 4-stage provenance), and invokes `lint_eps_dir` + `emit_sarif`.

### T6.1: Failing test

- [ ] **Step 1: Create test**

Create `external_validation/_rollout_anchors/01-lagrangebench/tests/test_emit_sarif_eps.py`:

```python
"""Tests for emit_sarif_eps.py — rung 4b case-study driver."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _populate_eps_dir(eps_dir: Path) -> None:
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        write_eps_t_npz,
    )

    eps_dir.mkdir(parents=True, exist_ok=True)
    common_kwargs = dict(
        out_dir=eps_dir,
        model_name="segnn",
        dataset_name="tgv2d",
        ckpt_hash="abc123",
        physics_lint_sha_pkl_inference="8c3d080000",
        physics_lint_sha_npz_conversion="5857144000",
        physics_lint_sha_eps_computation="d9a8baa000",
    )
    # PH-SYM-001 active: 4 angles × 1 traj
    for ang_str in ("pi_2", "pi", "3pi_2", "0"):
        write_eps_t_npz(
            **common_kwargs,
            eps_t=np.array([4.3e-7], dtype=np.float32),
            rule_id="PH-SYM-001",
            transform_kind="rotation" if ang_str != "0" else "identity",
            transform_param=ang_str,
            traj_index=0,
            skip_reason=None,
        )
    # PH-SYM-002 active
    write_eps_t_npz(
        **common_kwargs,
        eps_t=np.array([4.4e-7], dtype=np.float32),
        rule_id="PH-SYM-002",
        transform_kind="reflection",
        transform_param="y_axis",
        traj_index=0,
        skip_reason=None,
    )
    # PH-SYM-004 active
    write_eps_t_npz(
        **common_kwargs,
        eps_t=np.array([1.5e-15], dtype=np.float32),
        rule_id="PH-SYM-004",
        transform_kind="translation",
        transform_param="L_3_L_7",
        traj_index=0,
        skip_reason=None,
    )
    # PH-SYM-003 SKIP
    write_eps_t_npz(
        **common_kwargs,
        eps_t=np.array([np.nan], dtype=np.float32),
        rule_id="PH-SYM-003",
        transform_kind="skip",
        transform_param="so2_continuous",
        traj_index=0,
        skip_reason="PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with original",
    )


def test_emit_sarif_eps_produces_v1_1_sarif(tmp_path):
    from external_validation._rollout_anchors._01_lagrangebench.emit_sarif_eps import (
        emit_sarif_eps,
    )

    eps_dir = tmp_path / "segnn_tgv2d_d9a8baa000"
    _populate_eps_dir(eps_dir)
    out_sarif = tmp_path / "segnn_tgv2d_eps.sarif"

    emit_sarif_eps(
        eps_dir=eps_dir,
        out_sarif_path=out_sarif,
        case_study="01-lagrangebench",
        dataset="tgv2d",
        model="segnn",
        ckpt_hash="abc123",
        ckpt_id="segnn_tgv2d/best",
        physics_lint_sha_pkl_inference="8c3d080000",
        physics_lint_sha_npz_conversion="5857144000",
        physics_lint_sha_eps_computation="d9a8baa000",
        physics_lint_sha_sarif_emission="d9a8baa000",
        lagrangebench_sha="ee0001eeee",
        rollout_subdir="rollouts/segnn_tgv2d_post_d03df3e",
    )

    sarif = json.loads(out_sarif.read_text())
    run_props = sarif["runs"][0]["properties"]
    assert run_props["harness_sarif_schema_version"] == "1.1"
    assert run_props["physics_lint_sha_eps_computation"] == "d9a8baa000"

    rule_ids = {r["ruleId"] for r in sarif["runs"][0]["results"]}
    assert rule_ids == {"PH-SYM-001", "PH-SYM-002", "PH-SYM-003", "PH-SYM-004"}

    # SKIP row check
    skip_rows = [r for r in sarif["runs"][0]["results"] if r["ruleId"] == "PH-SYM-003"]
    assert len(skip_rows) == 1
    assert skip_rows[0]["properties"]["skip_reason"].startswith("PBC-square breaks SO(2)")
```

Note: the dotted module path `_01_lagrangebench` requires the directory `01-lagrangebench` to be importable. The hyphen makes it not a valid Python identifier; the test uses an `__init__.py` shim. If the existing `01-lagrangebench/` directory uses a different import path (e.g., a `sys.path` insertion in `emit_sarif.py`), match that pattern. Check first:

```bash
head -30 /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py | grep -E "^(import|from|sys\\.)"
```

If `emit_sarif.py` uses `sys.path` bootstrapping (per rung 4a's plan-deviation note in the post-merge log), follow that pattern in the test.

- [ ] **Step 2: Adjust the test's import to match the existing repo convention if needed**

If `emit_sarif.py` does `sys.path.insert(0, str(Path(__file__).parent))` and imports `from emit_sarif import ...`, do the same for `emit_sarif_eps`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01-lagrangebench"))
from emit_sarif_eps import emit_sarif_eps  # type: ignore[import-not-found]
```

(Run the inspection step above to determine the actual existing pattern.)

- [ ] **Step 3: Run, confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/01-lagrangebench/tests/test_emit_sarif_eps.py -v
```

Expected: FAIL with `ModuleNotFoundError: emit_sarif_eps`.

### T6.2: Implement `emit_sarif_eps.py`

- [ ] **Step 1: Create the driver**

Create `external_validation/_rollout_anchors/01-lagrangebench/emit_sarif_eps.py`:

```python
"""Rung 4b case-study driver: ε(t) npz dir → committed SARIF artifact.

Sibling to emit_sarif.py (rung 4a). Assembles run-level v1.1 SARIF
properties from arguments (4-stage sha provenance per SCHEMA.md §1.5
and §3.5), invokes lint_eps_dir to read the ε(t) npzs, and writes the
SARIF via the shared emit_sarif primitive.

The ε(t) npzs themselves are produced by the Modal entrypoints in
modal_app.py (lagrangebench_eps_p0_segnn_tgv2d / _p1_gns_tgv2d); this
driver runs after a `modal volume get` brings them to the local mirror.
"""

from __future__ import annotations

import sys
from pathlib import Path

# sys.path bootstrap to make `from emit_sarif_eps import ...` work from tests.
# Pattern adopted from emit_sarif.py (rung 4a). Adjust if rung 4a uses a
# different convention.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from external_validation._rollout_anchors._harness.lint_eps_dir import lint_eps_dir
from external_validation._rollout_anchors._harness.sarif_emitter import emit_sarif


def emit_sarif_eps(
    *,
    eps_dir: Path,
    out_sarif_path: Path,
    case_study: str,
    dataset: str,
    model: str,
    ckpt_hash: str,
    ckpt_id: str,
    physics_lint_sha_pkl_inference: str,
    physics_lint_sha_npz_conversion: str,
    physics_lint_sha_eps_computation: str,
    physics_lint_sha_sarif_emission: str,
    lagrangebench_sha: str,
    rollout_subdir: str,
) -> None:
    """Read ε(t) npzs from eps_dir, emit SARIF v1.1 to out_sarif_path."""
    results = lint_eps_dir(
        eps_dir=eps_dir,
        case_study=case_study,
        dataset=dataset,
        model=model,
        ckpt_hash=ckpt_hash,
    )

    run_level_properties = {
        "source": "rollout-anchor-harness",
        "harness_sarif_schema_version": "1.1",
        "physics_lint_sha_pkl_inference": physics_lint_sha_pkl_inference,
        "physics_lint_sha_npz_conversion": physics_lint_sha_npz_conversion,
        "physics_lint_sha_eps_computation": physics_lint_sha_eps_computation,
        "physics_lint_sha_sarif_emission": physics_lint_sha_sarif_emission,
        "lagrangebench_sha": lagrangebench_sha,
        "checkpoint_id": ckpt_id,
        "model_name": model,
        "dataset_name": dataset,
        "rollout_subdir": rollout_subdir,
    }

    emit_sarif(
        out_path=out_sarif_path,
        results=results,
        run_level_properties=run_level_properties,
    )
```

- [ ] **Step 2: Run test, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/01-lagrangebench/tests/test_emit_sarif_eps.py -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/01-lagrangebench/emit_sarif_eps.py external_validation/_rollout_anchors/01-lagrangebench/tests/test_emit_sarif_eps.py && git commit -m "01-lagrangebench/emit_sarif_eps: rung 4b case-study driver (4-stage provenance, v1.1 emission)"
```

---

## Task 7: Modal entrypoints

**Files:**
- Modify: `external_validation/_rollout_anchors/01-lagrangebench/modal_app.py`

The Modal entrypoints load the LB checkpoint, iterate over (rule, transform_param, traj_index), call the network forward 1 step (or T_steps=100 for the figure subset), compute ε(t) via `compute_eps_t_from_pair`, and persist ε(t) npzs to Modal Volume. They use the same volume layout convention as rung 4a (per design §3.6: `outputs/trajectories/{model}_{dataset}_<eps_computation_sha>/`).

Per project convention, full Modal-app code lands inline; the engineer needs to read it, not infer it.

### T7.1: Add `lagrangebench_eps_p0_segnn_tgv2d` entrypoint

- [ ] **Step 1: Find insertion point in modal_app.py**

```bash
grep -n "^@app\\.function\\|^def lagrangebench_rollout_p" /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/modal_app.py
```

Expected: existing `lagrangebench_rollout_p0_segnn_tgv2d` and `_p1_gns_tgv2d` rung-4a entrypoints. Insert the new `lagrangebench_eps_p0_segnn_tgv2d` right after `_p1_gns_tgv2d`.

- [ ] **Step 2: Append the entrypoint**

Append this content after the rung-4a entrypoints, before any `@app.local_entrypoint()` definitions:

```python
@app.function(
    image=lagrangebench_image,
    gpu="A10G",  # rung 4b: matched to rung 4a per D0-21 §10
    timeout=60 * 30,
    volumes={"/rollouts": rollouts_volume},
)
def lagrangebench_eps_p0_segnn_tgv2d(
    git_sha: str,
    full_git_sha: str,
    rung_4a_rollout_subdir: str,
) -> dict:
    """Compute ε(t) for SEGNN-TGV2D under PH-SYM rules; persist ε(t) npzs.

    Inputs: rung-4a's frozen rollout subdir on Modal Volume (under
    /rollouts/{rung_4a_rollout_subdir}/), which contains the 20
    `particle_rollout_traj{NN}.npz` reference rollouts.

    For each (rule, transform_param, traj_index):
      1. Load the reference rollout's step-0 state (positions, velocities)
         from the rung-4a npz.
      2. Apply the transform (rotation / reflection / translation / SKIP).
      3. Forward 1 step (or T_steps=100 for figure subset) on the
         pre-loaded SEGNN model checkpoint.
      4. Compute ε(t) via the per-step RMS aggregation primitive.
      5. Persist ε(t) npz to /rollouts/trajectories/segnn_tgv2d_{git_sha[:10]}/.

    Returns: {"npz_count": int, "elapsed_s": float, "git_sha_eps": str}.
    """
    import json
    import time
    from pathlib import Path

    import numpy as np
    from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
        load_rollout_npz,
    )
    from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import (
        compute_eps_t_from_pair,
        reflect_y_axis,
        rotate_about_box_center,
        so2_substrate_skip_trigger,
        translate_pbc,
        write_eps_t_npz,
    )

    started = time.time()
    rung_4a_dir = Path("/rollouts") / rung_4a_rollout_subdir
    out_dir = Path("/rollouts/trajectories") / f"segnn_tgv2d_{full_git_sha[:10]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load LB SEGNN model + checkpoint (same setup as rung 4a's _rollout_p0_segnn_tgv2d).
    #    The exact API is in lagrangebench / haiku; replicate the pattern
    #    used by rung 4a's lagrangebench_rollout_p0_segnn_tgv2d. The
    #    relevant function is `lb_run.infer_one_step(model, x0)` returning
    #    a step-1 state.
    from lagrangebench.evaluate import infer  # local API; check exact import in rung-4a entrypoint
    # ... (model + checkpoint loading; see rung-4a _rollout_p0_segnn_tgv2d for the pattern)
    # Important: the model must be loaded once and reused across all
    # transforms / trajectories — the per-checkpoint init cost (~30-60s on
    # A10G) is the dominant compute time, not the inference itself.
    model, params, dataset = _load_segnn_tgv2d_for_eps(rung_4a_dir)
    L_box = float(dataset.metadata["box_size"])  # TGV2D periodic-square side length

    npz_count = 0
    # Iteration plan per design §3.7:
    #   PH-SYM-001 (C₄): θ ∈ {pi_2, pi, 3pi_2} active + 0 smoke = 4 angles × 20 trajs = 80 npzs
    #   PH-SYM-002 (refl): 1 axis × 20 trajs = 20 npzs
    #   PH-SYM-003 (SO(2)): SKIP, 1 conceptual angle (no real rotation done) × 20 trajs = 20 npzs
    #   PH-SYM-004 (transl): 1 vector × 20 trajs = 20 npzs
    #   Figure subset T=100: PH-SYM-001 at pi_2 × trajs (0, 7, 14) — replaces 3 of the 80 active rows
    common_provenance = dict(
        model_name="segnn",
        dataset_name="tgv2d",
        ckpt_hash=dataset.metadata["ckpt_hash"],
        physics_lint_sha_pkl_inference=dataset.metadata["pkl_inference_sha"],
        physics_lint_sha_npz_conversion=dataset.metadata["npz_conversion_sha"],
        physics_lint_sha_eps_computation=full_git_sha,
    )

    figure_subset_trajs = (0, 7, 14)
    figure_subset_T_steps = 100

    for traj_index in range(20):
        ref_npz_path = rung_4a_dir / f"particle_rollout_traj{traj_index:02d}.npz"
        rollout = load_rollout_npz(ref_npz_path)
        x0_pos = rollout["positions"][0]  # (N, 2)
        x0_vel = rollout["velocities"][0]  # (N, 2)

        # PH-SYM-001 angles, including smoke at θ=0.
        for theta, transform_param_str in (
            (0.0, "0"),
            (np.pi / 2, "pi_2"),
            (np.pi, "pi"),
            (3 * np.pi / 2, "3pi_2"),
        ):
            T_steps = (
                figure_subset_T_steps
                if (traj_index in figure_subset_trajs and transform_param_str == "pi_2")
                else 1
            )
            r_pos, r_vel = rotate_about_box_center(
                positions=x0_pos, velocities=x0_vel, theta=theta, L=L_box
            )
            ref_traj = infer(model, params, x0_pos, x0_vel, n_steps=T_steps)  # (T, N, 2)
            rot_traj = infer(model, params, r_pos, r_vel, n_steps=T_steps)
            # Untransform the rotated trajectory: R⁻¹ rot_traj(t) for each t.
            untransformed = _apply_inverse_rotation_per_step(rot_traj, theta=theta, L=L_box)
            eps_t = compute_eps_t_from_pair(reference=ref_traj, candidate=untransformed)
            transform_kind = "rotation" if theta != 0.0 else "identity"
            write_eps_t_npz(
                out_dir=out_dir,
                eps_t=eps_t,
                rule_id="PH-SYM-001",
                transform_kind=transform_kind,
                transform_param=transform_param_str,
                traj_index=traj_index,
                skip_reason=None,
                **common_provenance,
            )
            npz_count += 1

        # PH-SYM-002: y-axis reflection
        r_pos, r_vel = reflect_y_axis(positions=x0_pos, velocities=x0_vel, L=L_box)
        ref_traj = infer(model, params, x0_pos, x0_vel, n_steps=1)
        refl_traj = infer(model, params, r_pos, r_vel, n_steps=1)
        # Reflection is its own inverse: applying again recovers original-frame.
        unreflected = _apply_inverse_reflection_per_step(refl_traj, L=L_box)
        eps_t = compute_eps_t_from_pair(reference=ref_traj, candidate=unreflected)
        write_eps_t_npz(
            out_dir=out_dir,
            eps_t=eps_t,
            rule_id="PH-SYM-002",
            transform_kind="reflection",
            transform_param="y_axis",
            traj_index=traj_index,
            skip_reason=None,
            **common_provenance,
        )
        npz_count += 1

        # PH-SYM-003: SO(2) SKIP (substrate-incompatibility).
        # The trigger fires because TGV2D substrate has periodic boundaries
        # and any non-{0, π/2, π, 3π/2} rotation breaks the cell. Per design
        # §3.6 trigger-vs-emission separation, the trigger lives here in the
        # adapter/entrypoint; the SARIF emission shape comes from
        # lint_eps_dir's shared D0-19 §3.4 path.
        assert so2_substrate_skip_trigger(
            theta=np.pi / 4, has_periodic_boundaries=True
        ), "trigger should fire on PBC + non-C₄ angle"
        write_eps_t_npz(
            out_dir=out_dir,
            eps_t=np.array([np.nan], dtype=np.float32),
            rule_id="PH-SYM-003",
            transform_kind="skip",
            transform_param="so2_continuous",
            traj_index=traj_index,
            skip_reason="PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with original",
            **common_provenance,
        )
        npz_count += 1

        # PH-SYM-004: translation by (L/3, L/7).
        t_vec = (L_box / 3.0, L_box / 7.0)
        r_pos, r_vel = translate_pbc(positions=x0_pos, velocities=x0_vel, t=t_vec, L=L_box)
        ref_traj = infer(model, params, x0_pos, x0_vel, n_steps=1)
        trans_traj = infer(model, params, r_pos, r_vel, n_steps=1)
        # Untranslate by -t.
        untranslated = _apply_inverse_translation_per_step(trans_traj, t=t_vec, L=L_box)
        eps_t = compute_eps_t_from_pair(reference=ref_traj, candidate=untranslated)
        write_eps_t_npz(
            out_dir=out_dir,
            eps_t=eps_t,
            rule_id="PH-SYM-004",
            transform_kind="translation",
            transform_param="L_3_L_7",
            traj_index=traj_index,
            skip_reason=None,
            **common_provenance,
        )
        npz_count += 1

    elapsed = time.time() - started
    rollouts_volume.commit()  # persist out_dir to Volume
    return {
        "npz_count": npz_count,
        "elapsed_s": elapsed,
        "git_sha_eps": full_git_sha,
        "out_dir": str(out_dir),
    }
```

The helpers `_load_segnn_tgv2d_for_eps`, `_apply_inverse_rotation_per_step`, `_apply_inverse_reflection_per_step`, `_apply_inverse_translation_per_step` are local to `modal_app.py` and follow the rung-4a pattern (model loading) plus apply the inverse of each transform per timestep. Their full code lands at the top of the entrypoint section per rung-4a's pattern; replicate the locality from `lagrangebench_rollout_p0_segnn_tgv2d`'s helpers.

Implementation note: the inverse-transform helpers operate per-step on the trajectory `(T_steps, N, 2)`. For rotation, the inverse is `rotate_about_box_center(theta=-theta)` applied to each step. For reflection, the inverse is the operation itself (involution). For translation, the inverse is `translate_pbc(t=(-t_x, -t_y))`. Velocities are not used by `compute_eps_t_from_pair` (the consumer aggregates positions only per design §3.4 (P)), so the helpers only need to inverse-transform positions.

- [ ] **Step 3: Local sanity-check (no Modal run)**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" python -c "import ast; ast.parse(open('external_validation/_rollout_anchors/01-lagrangebench/modal_app.py').read()); print('OK')"
```

Expected: `OK` (Python parses cleanly).

- [ ] **Step 4: Add the GNS-TGV2D entrypoint as a near-mirror**

Repeat the structure as `lagrangebench_eps_p1_gns_tgv2d` with `model="gns"` and a separate `_load_gns_tgv2d_for_eps` helper (or parameterize the SEGNN helper if rung 4a already does that). Output dir: `/rollouts/trajectories/gns_tgv2d_{full_git_sha[:10]}/`.

- [ ] **Step 5: Add a `local_entrypoint` for invocation**

Per rung 4a's pattern (e.g., `rollout_p0_segnn_tgv2d` local entrypoint), add:

```python
@app.local_entrypoint()
def eps_p0_segnn_tgv2d() -> None:
    import subprocess
    full_git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git_sha = full_git_sha[:10]
    rung_4a_subdir = "rollouts/segnn_tgv2d_post_d03df3e"  # match rung 4a's published subdir
    res = lagrangebench_eps_p0_segnn_tgv2d.remote(
        git_sha=git_sha,
        full_git_sha=full_git_sha,
        rung_4a_rollout_subdir=rung_4a_subdir,
    )
    print("eps_p0_segnn_tgv2d:", res)


@app.local_entrypoint()
def eps_p1_gns_tgv2d() -> None:
    import subprocess
    full_git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    git_sha = full_git_sha[:10]
    rung_4a_subdir = "rollouts/gns_tgv2d_post_d03df3e"
    res = lagrangebench_eps_p1_gns_tgv2d.remote(
        git_sha=git_sha,
        full_git_sha=full_git_sha,
        rung_4a_rollout_subdir=rung_4a_subdir,
    )
    print("eps_p1_gns_tgv2d:", res)
```

The `rung_4a_rollout_subdir` value names rung 4a's existing Modal Volume location for the reference rollouts; verify the actual subdir on Volume before invoking (the rung-4a `eps_computation_sha`-suffixed subdir is the right anchor). Look up via `modal volume ls` if uncertain.

- [ ] **Step 6: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/01-lagrangebench/modal_app.py && git commit -m "01-lagrangebench/modal_app: rung 4b ε computation entrypoints (SEGNN + GNS, A10G)"
```

---

## Task 8: Renderer — `render_eps_table.py`

**Files:**
- Create: `external_validation/_rollout_anchors/methodology/tools/render_eps_table.py`
- Create: `external_validation/_rollout_anchors/methodology/tools/tests/test_render_eps_table.py`

Sibling renderer per design §5.1, focused on schema_version v1.1 only. Renders tripartite-grouped output (architectural / construction-trivial / substrate-incompatible-SKIP). Asserts `harness_sarif_schema_version == "1.1"` on read; fails loud on mismatch.

### T8.1: Failing test for v1.1 reader + schema-version assertion

- [ ] **Step 1: Create test fixture file + test**

Create `external_validation/_rollout_anchors/methodology/tools/tests/test_render_eps_table.py`:

```python
"""Tests for render_eps_table — rung 4b sibling renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _v1_1_sarif_fixture(model: str = "segnn", with_skip: bool = True) -> dict:
    """Build a v1.1 SARIF dict in-memory matching the rung-4b emission shape."""
    results = []
    # PH-SYM-001 active angles + smoke
    for ang_str, eps_val in (("pi_2", 4.3e-7), ("pi", 4.1e-7), ("3pi_2", 4.5e-7)):
        results.append({
            "ruleId": "PH-SYM-001",
            "level": "note",
            "message": {"text": f"eps_pos_rms={eps_val:.3e} (transform=rotation {ang_str})"},
            "properties": {
                "transform_kind": "rotation",
                "transform_param": ang_str,
                "traj_index": 0,
                "eps_pos_rms": eps_val,
                "eps_t_npz_filename": f"eps_PH-SYM-001_rotation_{ang_str}_traj00.npz",
            },
        })
    # PH-SYM-001 angle 0 (smoke)
    results.append({
        "ruleId": "PH-SYM-001",
        "level": "note",
        "message": {"text": "eps_pos_rms=0.000e+00 (transform=identity 0)"},
        "properties": {
            "transform_kind": "identity",
            "transform_param": "0",
            "traj_index": 0,
            "eps_pos_rms": 0.0,
            "eps_t_npz_filename": "eps_PH-SYM-001_identity_0_traj00.npz",
        },
    })
    # PH-SYM-002 active
    results.append({
        "ruleId": "PH-SYM-002",
        "level": "note",
        "message": {"text": "eps_pos_rms=4.400e-07 (transform=reflection y_axis)"},
        "properties": {
            "transform_kind": "reflection",
            "transform_param": "y_axis",
            "traj_index": 0,
            "eps_pos_rms": 4.4e-7,
            "eps_t_npz_filename": "eps_PH-SYM-002_reflection_y_axis_traj00.npz",
        },
    })
    # PH-SYM-004 active
    results.append({
        "ruleId": "PH-SYM-004",
        "level": "note",
        "message": {"text": "eps_pos_rms=1.500e-15 (transform=translation L_3_L_7)"},
        "properties": {
            "transform_kind": "translation",
            "transform_param": "L_3_L_7",
            "traj_index": 0,
            "eps_pos_rms": 1.5e-15,
            "eps_t_npz_filename": "eps_PH-SYM-004_translation_L_3_L_7_traj00.npz",
        },
    })
    if with_skip:
        results.append({
            "ruleId": "PH-SYM-003",
            "level": "note",
            "message": {"text": "SKIP: PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with original"},
            "properties": {
                "transform_kind": "skip",
                "transform_param": "so2_continuous",
                "traj_index": 0,
                "eps_pos_rms": None,
                "skip_reason": "PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with original",
                "eps_t_npz_filename": "eps_PH-SYM-003_skip_so2_continuous_traj00.npz",
            },
        })
    return {
        "$schema": "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {"driver": {"name": "physics-lint-rollout-anchor-harness", "version": "0.1.0"}},
            "results": results,
            "properties": {
                "source": "rollout-anchor-harness",
                "harness_sarif_schema_version": "1.1",
                "physics_lint_sha_pkl_inference": "8c3d080000",
                "physics_lint_sha_npz_conversion": "5857144000",
                "physics_lint_sha_eps_computation": "d9a8baa000",
                "physics_lint_sha_sarif_emission": "d9a8baa000",
                "lagrangebench_sha": "ee0001eeee",
                "checkpoint_id": f"{model}_tgv2d/best",
                "model_name": model,
                "dataset_name": "tgv2d",
                "rollout_subdir": f"rollouts/{model}_tgv2d_post_d03df3e",
            },
        }],
    }


def test_render_eps_table_reads_v1_1_sarif_pair(tmp_path):
    from external_validation._rollout_anchors.methodology.tools.render_eps_table import (
        render_eps_table,
    )

    segnn_path = tmp_path / "segnn_tgv2d_eps.sarif"
    gns_path = tmp_path / "gns_tgv2d_eps.sarif"
    segnn_path.write_text(json.dumps(_v1_1_sarif_fixture(model="segnn")))
    gns_path.write_text(json.dumps(_v1_1_sarif_fixture(model="gns")))

    output = render_eps_table(segnn_sarif_path=segnn_path, gns_sarif_path=gns_path)

    # Tripartite groupings appear as section headers.
    assert "Architectural-evidence rows" in output
    assert "Construction-trivial rows" in output
    assert "Substrate-incompatible SKIP" in output
    # Headline table rows present.
    assert "PH-SYM-001" in output
    assert "PH-SYM-002" in output
    assert "PH-SYM-003" in output
    assert "PH-SYM-004" in output


def test_render_eps_table_fails_loud_on_v1_0_input(tmp_path):
    from external_validation._rollout_anchors.methodology.tools.render_eps_table import (
        SchemaVersionMismatchError,
        render_eps_table,
    )

    v1_0_sarif = _v1_1_sarif_fixture(model="segnn")
    v1_0_sarif["runs"][0]["properties"]["harness_sarif_schema_version"] = "1.0"

    sarif_path = tmp_path / "v1_0.sarif"
    sarif_path.write_text(json.dumps(v1_0_sarif))

    with pytest.raises(SchemaVersionMismatchError, match="expected 1.1, got 1.0"):
        render_eps_table(segnn_sarif_path=sarif_path, gns_sarif_path=sarif_path)
```

- [ ] **Step 2: Run, confirm fail**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/methodology/tools/tests/test_render_eps_table.py -v
```

Expected: 2 FAILED with `ModuleNotFoundError`.

### T8.2: Implement `render_eps_table`

- [ ] **Step 1: Create render_eps_table.py**

Create `external_validation/_rollout_anchors/methodology/tools/render_eps_table.py`:

```python
"""Rung 4b sibling renderer — schema_version v1.1 only.

Per design §5.1: focused on one schema version. v1.0 SARIFs cannot be
rendered by this tool; the renderer raises SchemaVersionMismatchError
with a clear message. Rung 4a's render_cross_stack_table.py handles
v1.0 unchanged.

Output: tripartite-grouped markdown table (architectural-evidence /
construction-trivial / substrate-incompatible-SKIP) per design §5.2.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_EXPECTED_SCHEMA_VERSION = "1.1"


class SchemaVersionMismatchError(Exception):
    """Raised when a SARIF's harness_sarif_schema_version differs from this renderer's."""


def _classify_evidence(rule_id: str, transform_kind: str) -> str:
    """Map (rule_id, transform_kind) to one of the three evidence classes per design §3.2."""
    if transform_kind == "skip":
        return "substrate-incompatible-skip"
    if transform_kind == "identity":
        # PH-SYM-001 angle-0 smoke test — construction-trivial.
        return "construction-trivial"
    if transform_kind == "translation":
        # PH-SYM-004 — construction-trivial (translation + PBC commute exactly).
        return "construction-trivial"
    # PH-SYM-001 active angles + PH-SYM-002 reflection are architectural-evidence.
    return "architectural"


def _verdict_label(eps: float | None, evidence_class: str) -> str:
    """Per design §3.3 threshold band; SKIP gets its own label."""
    if evidence_class == "substrate-incompatible-skip":
        return "SKIP"
    assert eps is not None
    if eps <= 1e-5:
        return "PASS"
    if eps <= 1e-2:
        return "APPROXIMATE"
    return "FAIL"


def _format_eps(eps: float | None) -> str:
    if eps is None:
        return "—"
    if eps == 0.0:
        return "0.0e+00"
    return f"{eps:.3e}"


def _read_sarif(path: Path) -> dict[str, Any]:
    sarif = json.loads(Path(path).read_text())
    run_props = sarif["runs"][0]["properties"]
    schema_version = run_props.get("harness_sarif_schema_version", "<unset>")
    if schema_version != _EXPECTED_SCHEMA_VERSION:
        raise SchemaVersionMismatchError(
            f"render_eps_table: harness_sarif_schema_version mismatch in {path}: "
            f"expected {_EXPECTED_SCHEMA_VERSION}, got {schema_version}"
        )
    return sarif


def render_eps_table(*, segnn_sarif_path: Path, gns_sarif_path: Path) -> str:
    """Read both stacks' v1.1 SARIFs, render tripartite-grouped markdown table.

    Returns the markdown content as a string. The caller decides where to
    write it (typically into the rung 4b table writeup at
    `methodology/docs/2026-05-05-rung-4b-equivariance-table.md`).
    """
    segnn_sarif = _read_sarif(Path(segnn_sarif_path))
    gns_sarif = _read_sarif(Path(gns_sarif_path))

    # Build flat list of (model, result) tuples; partition by evidence class.
    rows: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "architectural": [],
        "construction-trivial": [],
        "substrate-incompatible-skip": [],
    }
    for model, sarif in (("segnn", segnn_sarif), ("gns", gns_sarif)):
        for result in sarif["runs"][0]["results"]:
            rule_id = result["ruleId"]
            props = result["properties"]
            evidence_class = _classify_evidence(rule_id, props["transform_kind"])
            rows[evidence_class].append((model, result))

    # Render each evidence class as its own subsection.
    lines: list[str] = []

    def _emit_class_table(title: str, class_key: str) -> None:
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Rule | Stack | transform_param | traj_index | ε | Verdict |")
        lines.append("|---|---|---|---|---|---|")
        for model, result in sorted(
            rows[class_key],
            key=lambda mr: (mr[1]["ruleId"], mr[1]["properties"]["transform_param"], mr[0]),
        ):
            props = result["properties"]
            eps = props.get("eps_pos_rms")
            verdict = _verdict_label(eps, class_key)
            extra = ""
            if class_key == "substrate-incompatible-skip":
                extra = f" — {props['skip_reason']}"
            lines.append(
                f"| {result['ruleId']} | {model.upper()} | "
                f"{props['transform_param']} | {props['traj_index']} | "
                f"{_format_eps(eps)} | {verdict}{extra} |"
            )
        lines.append("")

    _emit_class_table("Architectural-evidence rows", "architectural")
    _emit_class_table("Construction-trivial rows", "construction-trivial")
    _emit_class_table("Substrate-incompatible SKIP", "substrate-incompatible-skip")

    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint: render to stdout."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--segnn-sarif", type=Path, required=True)
    parser.add_argument("--gns-sarif", type=Path, required=True)
    args = parser.parse_args()
    print(render_eps_table(
        segnn_sarif_path=args.segnn_sarif,
        gns_sarif_path=args.gns_sarif,
    ))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add tests/__init__.py if missing**

```bash
cd /Users/zenith/Desktop/physics-lint && touch external_validation/_rollout_anchors/methodology/tools/__init__.py external_validation/_rollout_anchors/methodology/tools/tests/__init__.py
```

- [ ] **Step 3: Run tests, confirm pass**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/methodology/tools/tests/test_render_eps_table.py -v
```

Expected: 2 PASSED.

- [ ] **Step 4: Commit T8.1 + T8.2**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/methodology/tools/render_eps_table.py external_validation/_rollout_anchors/methodology/tools/tests/ external_validation/_rollout_anchors/methodology/tools/__init__.py && git commit -m "methodology/tools/render_eps_table: rung 4b sibling renderer (v1.1, tripartite-grouped)"
```

### T8.3: Golden-fixture renderer test

- [ ] **Step 1: Add a golden-fixture comparison test**

Per the "test fixtures hand-crafted, not copied from production" discipline, the renderer's output is pinned to a checked-in golden file.

Append to `test_render_eps_table.py`:

```python


def test_render_eps_table_golden_output_matches(tmp_path):
    """Pin the renderer output to a checked-in golden file so future schema
    changes that alter row formatting fail loud rather than silently
    drifting."""
    from external_validation._rollout_anchors.methodology.tools.render_eps_table import (
        render_eps_table,
    )

    fixtures_dir = Path(__file__).parent / "fixtures"
    segnn_sarif = fixtures_dir / "segnn_tgv2d_eps.sarif"
    gns_sarif = fixtures_dir / "gns_tgv2d_eps.sarif"
    expected_md = fixtures_dir / "expected_table.md"

    output = render_eps_table(segnn_sarif_path=segnn_sarif, gns_sarif_path=gns_sarif)

    if expected_md.exists():
        assert output == expected_md.read_text(), (
            "golden mismatch — if intentional, regenerate expected_table.md "
            "by running:\n"
            f"  python {Path(__file__).parents[2] / 'render_eps_table.py'} "
            f"--segnn-sarif {segnn_sarif} --gns-sarif {gns_sarif} > {expected_md}"
        )
    else:
        # First-run convenience: write the golden so the next run has a baseline.
        expected_md.write_text(output)
        pytest.skip(f"wrote initial golden to {expected_md}; rerun to validate equality")
```

- [ ] **Step 2: Generate golden fixtures**

```bash
cd /Users/zenith/Desktop/physics-lint && mkdir -p external_validation/_rollout_anchors/methodology/tools/tests/fixtures && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" python -c "
import json, sys
sys.path.insert(0, 'external_validation/_rollout_anchors/methodology/tools/tests')
from test_render_eps_table import _v1_1_sarif_fixture
import pathlib
out = pathlib.Path('external_validation/_rollout_anchors/methodology/tools/tests/fixtures')
(out / 'segnn_tgv2d_eps.sarif').write_text(json.dumps(_v1_1_sarif_fixture(model='segnn'), indent=2))
(out / 'gns_tgv2d_eps.sarif').write_text(json.dumps(_v1_1_sarif_fixture(model='gns'), indent=2))
"
```

Expected: two SARIF fixtures created.

- [ ] **Step 3: Run test (it skips on first run, writing the expected_table.md)**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/methodology/tools/tests/test_render_eps_table.py::test_render_eps_table_golden_output_matches -v
```

Expected: SKIPPED (first run, golden written).

- [ ] **Step 4: Re-run for the strict equality check**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest --import-mode=importlib external_validation/_rollout_anchors/methodology/tools/tests/test_render_eps_table.py::test_render_eps_table_golden_output_matches -v
```

Expected: PASSED.

- [ ] **Step 5: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/methodology/tools/tests/ && git commit -m "methodology/tools/render_eps_table: golden-fixture pinning of v1.1 output"
```

---

## Task 9: Execute on Modal + commit SARIFs

**Files:**
- Create: `external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/segnn_tgv2d_eps_<sha>.sarif`
- Create: `external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/gns_tgv2d_eps_<sha>.sarif`
- Local mirror: `external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories/{segnn,gns}_tgv2d_<sha>/eps_*.npz` (gitignored)

### T9.1: Run Modal entrypoints (SEGNN + GNS)

- [ ] **Step 1: Authenticate to Modal (if not already)**

```bash
cd /Users/zenith/Desktop/physics-lint && modal config set-token <token>  # if not configured
```

Expected: token already configured from rung 4a.

- [ ] **Step 2: Run SEGNN ε entrypoint**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" modal run external_validation/_rollout_anchors/01-lagrangebench/modal_app.py::eps_p0_segnn_tgv2d
```

Expected: completes in ~1–2 minutes; prints `eps_p0_segnn_tgv2d: {"npz_count": 80, "elapsed_s": ~60–120, "git_sha_eps": "<sha>", "out_dir": "/rollouts/trajectories/segnn_tgv2d_<sha[:10]>"}`.

Sanity-check: `npz_count == 80` (4 PH-SYM-001 angles + 1 PH-SYM-002 + 1 PH-SYM-003 SKIP + 1 PH-SYM-004) × 20 trajs = 80 npzs.

- [ ] **Step 3: Run GNS ε entrypoint**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" modal run external_validation/_rollout_anchors/01-lagrangebench/modal_app.py::eps_p1_gns_tgv2d
```

Expected: completes in ~1–2 minutes; `npz_count == 80`.

### T9.2: Pull npzs from Modal Volume

- [ ] **Step 1: Find the actual subdirs on Volume**

```bash
modal volume ls rollouts trajectories
```

Expected: two subdirs `segnn_tgv2d_<sha[:10]>` and `gns_tgv2d_<sha[:10]>`. Note the exact sha for the next step.

- [ ] **Step 2: Pull SEGNN trajectories to local mirror**

```bash
cd /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories && modal volume get rollouts trajectories/segnn_tgv2d_<sha[:10]>/ ./
```

Adjust the trailing-slash convention to match what worked for rung 4a's `modal volume get` — the rung-4a session noted that trailing slashes affected nesting depth. Verify after the get:

```bash
ls /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories/segnn_tgv2d_<sha[:10]>/ | head -10
```

Expected: 80 files matching `eps_PH-SYM-{001,002,003,004}_*_traj{NN}.npz`.

- [ ] **Step 3: Pull GNS trajectories**

```bash
cd /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories && modal volume get rollouts trajectories/gns_tgv2d_<sha[:10]>/ ./
```

- [ ] **Step 4: Verify gitignore is hiding the npzs**

```bash
cd /Users/zenith/Desktop/physics-lint && git status external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories/
```

Expected: empty (no `??` lines) — the .gitignore from T0.2 hides the npzs.

### T9.3: Run `emit_sarif_eps` for both stacks

- [ ] **Step 1: Run SEGNN driver**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" python -c "
import subprocess
from pathlib import Path
from external_validation._rollout_anchors._01_lagrangebench.emit_sarif_eps import emit_sarif_eps  # adjust import per actual module path

full_git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
sha10 = full_git_sha[:10]

out_dir = Path('external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif')
out_dir.mkdir(exist_ok=True)
out_path = out_dir / f'segnn_tgv2d_eps_{sha10}.sarif'

emit_sarif_eps(
    eps_dir=Path(f'external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories/segnn_tgv2d_{sha10}'),
    out_sarif_path=out_path,
    case_study='01-lagrangebench',
    dataset='tgv2d',
    model='segnn',
    ckpt_hash='<carry from rung 4a>',
    ckpt_id='segnn_tgv2d/best',
    physics_lint_sha_pkl_inference='<carry from rung 4a npz>',
    physics_lint_sha_npz_conversion='<carry from rung 4a npz>',
    physics_lint_sha_eps_computation=full_git_sha,
    physics_lint_sha_sarif_emission=full_git_sha,
    lagrangebench_sha='<carry from rung 4a npz>',
    rollout_subdir=f'rollouts/segnn_tgv2d_post_d03df3e',
)
print('wrote', out_path)
"
```

Substitute `<carry from rung 4a npz>` with the actual values read from rung 4a's reference rollout npz metadata. The SEGNN reference npzs are at `outputs/_local_mirror/segnn_tgv2d_*/particle_rollout_traj00.npz`; load one with `np.load()` and read `metadata`.

- [ ] **Step 2: Run GNS driver**

(Mirror for GNS, with `model='gns'` and `rollout_subdir=f'rollouts/gns_tgv2d_post_d03df3e'`.)

- [ ] **Step 3: Verify both SARIFs land**

```bash
ls -la external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/*_eps_*.sarif
```

Expected: two new SARIF files; sizes ~few hundred KB each (80 result rows × ~1KB per row).

- [ ] **Step 4: Sanity-check schema_version + row count**

```bash
PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" python -c "
import json
for path in ['external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/segnn_tgv2d_eps_*.sarif']:
    import glob
    for f in glob.glob(path):
        s = json.load(open(f))
        print(f, 'schema:', s['runs'][0]['properties']['harness_sarif_schema_version'], 'rows:', len(s['runs'][0]['results']))
"
```

Expected: `schema: 1.1 rows: 80` for both files.

- [ ] **Step 5: Commit SARIFs**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/*_eps_*.sarif && git commit -m "01-lagrangebench/outputs/sarif: rung 4b SEGNN + GNS eps SARIFs (v1.1, 80 rows each)"
```

---

## Task 10: Render writeup table content

- [ ] **Step 1: Run the renderer**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" python external_validation/_rollout_anchors/methodology/tools/render_eps_table.py --segnn-sarif external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/segnn_tgv2d_eps_<sha>.sarif --gns-sarif external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/gns_tgv2d_eps_<sha>.sarif > /tmp/rung_4b_table.md
```

Expected: `/tmp/rung_4b_table.md` contains tripartite-grouped output. Sanity-check `wc -l /tmp/rung_4b_table.md` is roughly 100+ lines.

- [ ] **Step 2: Inspect output**

```bash
cat /tmp/rung_4b_table.md
```

Expected:
- 3 markdown headers ("Architectural-evidence rows", "Construction-trivial rows", "Substrate-incompatible SKIP")
- ~120 architectural rows (3 angles × 20 trajs × 2 stacks for PH-SYM-001 + 1 axis × 20 trajs × 2 stacks for PH-SYM-002)
- ~80 construction-trivial rows (1 angle × 20 trajs × 2 stacks for PH-SYM-001 angle-0 + 1 vector × 20 trajs × 2 stacks for PH-SYM-004)
- ~40 SKIP rows (1 conceptual × 20 trajs × 2 stacks for PH-SYM-003)
- All ε values in PASS / APPROXIMATE bands per the float32-floor threshold

---

## Task 11: Write the dated table writeup

**Files:**
- Create: `external_validation/_rollout_anchors/methodology/docs/2026-05-05-rung-4b-equivariance-table.md`

### T11.1: Write the writeup

- [ ] **Step 1: Create the writeup**

Create the file with the structure below. Substitute observed values from the SARIFs into the body — particularly the SEGNN/GNS ε ranges and the secondary-finding sentence on Helwig consistency wording.

```markdown
# Rung 4b — Cross-stack equivariance SARIF + writeup table (executed)

**Date:** 2026-05-05
**Repo:** physics-lint
**Branch:** `feature/rung-4b-equivariance` (off `master` at `ba13b45`)
**Predecessor design:** `methodology/docs/2026-05-05-rung-4b-equivariance-design.md` (`d9a8baa`)
**D-entry:** D0-21 in `methodology/DECISIONS.md`
**Status:** Executed. SARIFs committed at v1.1 schema_version (80 rows × 2 stacks).

---

## Frozen headline (per design §1.2)

> *"physics-lint's PH-SYM rule schema, with equivariance thresholds set
> at the float32 numerical-precision floor (ε ≤ 10⁻⁵ PASS, 10⁻⁵ < ε ≤
> 10⁻² APPROXIMATE, ε > 10⁻² FAIL), runs unmodified across SEGNN-TGV2D
> and GNS-TGV2D rollouts; per-stack ε values are emitted in the same
> SARIF schema as 4a (schema_version v1.1) and reported as observed."*

The headline is robust to both probe outcomes — SEGNN at the float32
floor (expected, uninteresting confirmation of the floor calibration)
and GNS at either the floor or in the APPROXIMATE band (secondary
finding). Neither outcome rewords the headline.

---

## What each PH-SYM rule measures on this substrate

Three distinct kinds of evidence emit to the v1.1 SARIF; the table
groups by evidence type rather than by rule id, so the reader doesn't
read four PASSes as the same kind of evidence.

- **Architectural-evidence rows** (PH-SYM-001 at active angles {π/2,
  π, 3π/2}, PH-SYM-002 reflection): ε reflects the model's per-step
  equivariance under discrete rotations/reflections. Load-bearing for
  the headline's portability claim.
- **Construction-trivial rows** (PH-SYM-001 at θ = 0, PH-SYM-004
  translation): ε at machine-zero by construction. Identity rotation
  is trivially preserved; translation + PBC commute exactly.
  Construction-trivial rows are smoke tests — failure indicates a real
  bug in the rotation/translation mechanic — but do not differentiate
  models.
- **Substrate-incompatible SKIP** (PH-SYM-003 SO(2)): the rule schema
  emits SKIP-with-reason rather than a confounded numerical value. On
  a periodic-square substrate, non-{0, π/2, π, 3π/2} rotations don't
  preserve the cell — the rotated state isn't a valid input to f, and
  any ε computed from it would tangle architecture with substrate-
  incompatibility.

---

## Cross-stack equivariance table

[Paste output of Task 10 here.]

---

## Threshold rationale (float32 numerical-precision floor)

Per design §3.3. An exactly E(3)-equivariant network operating in
float32 has equivariance error bounded by accumulated round-off
(~10⁻⁷ per op, growing with depth and condition number); SEGNN's
expected ε lands at the floor (~10⁻⁷–10⁻⁸) with 1–2 orders of magnitude
margin below the 10⁻⁵ PASS threshold. GNS's ε is the load-bearing
empirical variability; the secondary-finding wording (below) adapts to
the observed value.

The threshold is principled (numerical-floor-based), architecture-
agnostic, and requires no cross-paper citation. **Forward flag:** if
case study 02 (PhysicsNeMo MGN) ever runs at float64, the threshold
band would mean different things (float64 floor ~10⁻¹⁶ per op); revisit
required.

---

## Helwig et al. ICML 2023 — architecture-level context

Helwig et al. ICML 2023 characterizes GNS as approximately equivariant
at the architecture level (primarily on dam-break, not specifically on
TGV2D). The 4b probe's calibration anchor is *not* Helwig — the float32
numerical-precision floor argument provides the threshold rationale
independently. Helwig is cited here as architecture-level *context for
interpreting* the observed GNS values, not as a calibration anchor.

[Secondary finding sentence to be filled post-execution:]

- If observed GNS ε ∈ (10⁻⁵, 10⁻²]: *"The observed GNS-TGV2D ε ≈ <X>
  is consistent with Helwig et al.'s architecture-level
  characterization of GNS as approximately equivariant."*
- If observed GNS ε ≤ 10⁻⁵: *"GNS-TGV2D ε ≈ <X> meets the
  equivariance bar at the float32 floor, contrary to Helwig et al.'s
  architecture-level characterization which is anchored primarily on
  dam-break; this suggests TGV2D's smoother low-Reynolds dynamics
  exposes less of GNS's architectural approximation than the prior
  characterization predicts."*

The headline is robust to either outcome; only this paragraph adapts.

---

## Honest limits (verbatim from design §1.3)

1. **Not a SEGNN-vs-GNS differentiation claim.** Model differentiation
   is a secondary finding, not the load-bearing claim.
2. **Not a cross-framework portability claim.** TGV2D / LagrangeBench
   only. PhysicsNeMo MGN equivariance is a separate rung.
3. **Not a multi-dataset claim.** TGV2D only.
4. **Not an equivariance-coverage-completeness claim.** Limited
   rotation/reflection/translation set.
5. **PH-SYM-003 / PBC-square IS the analogous SKIP mechanism** to
   PH-CON-002 / dissipative under D0-18. Meta-correction recorded in
   D0-21: this was missed at brainstorm's non-claim-5 framing,
   surfaced at design pass not mid-execution.

---

## Provenance

- **Reference rollout shas (carried from rung 4a):**
  `physics_lint_sha_pkl_inference=<value>`,
  `physics_lint_sha_npz_conversion=<value>`.
- **ε computation sha:** `physics_lint_sha_eps_computation=<value>`
  (commit at Modal-side ε(t) generation time).
- **SARIF emission sha:** `physics_lint_sha_sarif_emission=<value>`
  (commit at consumer-side SARIF write time).
- **GPU class:** A10G (matched to rung 4a per D0-17 amendment 1
  generalization to GPU-class consistency).
- **Compute total:** ~880 single-step inferences ≈ sub-minute on A10G.

---

## D-entries closed

- **D0-21** — rung 4b pre-registration. **Realized.** [Filled with
  merge sha when this rung lands on master.]
```

- [ ] **Step 2: Substitute the observed values into the writeup**

Read the SARIFs and fill the placeholder fields:
- `<carry from rung 4a npz>` shas
- The actual table content from `/tmp/rung_4b_table.md`
- Observed SEGNN ε range
- Observed GNS ε range and the corresponding secondary-finding sentence

- [ ] **Step 3: Verify Markdown parses cleanly**

```bash
cd /Users/zenith/Desktop/physics-lint && head -50 external_validation/_rollout_anchors/methodology/docs/2026-05-05-rung-4b-equivariance-table.md
```

Expected: clean structure, no broken Markdown.

- [ ] **Step 4: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/methodology/docs/2026-05-05-rung-4b-equivariance-table.md && git commit -m "methodology/docs: rung 4b equivariance table writeup (post-execution)"
```

### T11.2: Close the D0-21 loop

- [ ] **Step 1: Append "Realized" footer to D0-21**

Edit `external_validation/_rollout_anchors/methodology/DECISIONS.md`. Find the D0-21 entry's "**Realized.**" placeholder line and replace with:

```
**Realized.** Rung 4b implementation landed at `feature/rung-4b-equivariance`;
ε(t) generation on Modal A10G; SARIFs committed at `<sha>`; renderer +
table writeup at `methodology/docs/2026-05-05-rung-4b-equivariance-table.md`.
[On merge to master: append merge sha here.]
```

- [ ] **Step 2: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/methodology/DECISIONS.md && git commit -m "methodology/DECISIONS.md D0-21: realized footer (rung 4b implementation landed)"
```

---

## Self-review (against design spec)

**Spec coverage check (each design section → at least one task):**

| Spec section | Task |
|---|---|
| §1.1 Scope (reuse rung-4a npzs as x_0 source) | T7 (modal_app entrypoints load via load_rollout_npz) |
| §1.2 Frozen headline | T11 (writeup §1) |
| §1.3 Five non-claims | T11 (writeup honest-limits) |
| §1.4 D0-21 entry | T0.1 |
| §2 Architecture (file tree) | T2–T8 (creates each new module) |
| §3.1 Probe-as-calibration framing | T11 (Helwig context paragraph) |
| §3.2 Rule set (γ) + tripartite | T7 (rule iteration) + T8 (renderer grouping) + T11 (writeup subsection) |
| §3.3 Float32-floor threshold | T8 (verdict labels) + T11 (rationale paragraph) |
| §3.4 Reportable ε shape (P) + uniform single-tier | T2 (compute_eps_t_from_pair) + T3 (npz I/O) |
| §3.5 Mechanic correctness primitives | T2 (rotate, reflect, translate) |
| §3.6 Pipeline (Modal-Volume-only, trigger-vs-emission) | T0.2 (gitignore) + T7 (Modal entrypoints) + T2 (so2 trigger) + T4 (lint_eps_dir SKIP path) |
| §3.7 Compute budget + A10G | T7 (entrypoint `gpu="A10G"`) + T9 (run + verify ~sub-minute) |
| §4.1 Schema bump v1.0 → v1.1 | T1.2 (SCHEMA.md) + T5 (sarif_emitter v1.1) |
| §4.2 4-stage provenance | T3 (write/read_eps_t_npz with eps_computation_sha) + T6 (emit_sarif_eps run-level properties) |
| §5.1 Sibling renderer | T8 |
| §5.2 Tripartite-grouped output | T8 (renderer grouping) |
| §6 Test fixtures | T2.1 (c4_symmetric_4particle), T2.4 (reflection_symmetric), T4.3 (skip-mechanism fixture), T8.3 (golden) |
| §7 Forward flags + honest limits | T0.1 (D0-21 forward flags section) |
| §8 Acceptance criteria | T9 (commit SARIFs) + T11 (writeup) |
| §9 Predecessor → successor | T11 (writeup §predecessor / successor) |

No coverage gaps.

**Placeholder scan:** Two intentional placeholders remain in T9.3 (`<carry from rung 4a npz>` for sha values that must be read from the actual rollout npz metadata at execution time) and T11.1 (the secondary-finding sentence's `<X>` slot for the observed GNS value). Both are filled at execution time by the engineer reading actual data; they cannot be predicted in the plan. All other steps contain literal content.

**Type-consistency check:**
- `HarnessResult` (rung 4a's dataclass) used unchanged in T4.2 → T6.2 → T8.
- `eps_t` shape is consistently `(T_steps,)` fp32 across T2/T3/T4/T7/T8.
- `transform_kind` enum values consistent: `"rotation" | "reflection" | "translation" | "identity" | "skip"` (defined in T1.1, used in T2/T3/T4/T7/T8).
- `transform_param` is a string in all read paths (T3 read_eps_t_npz, T4 lint_eps_dir, T8 renderer); T7 entrypoint writes it via `write_eps_t_npz` which accepts a string parameter — consistent.
- `harness_sarif_schema_version` value `"1.1"` consistent across T1.2, T5.1, T6.2, T8.

No inconsistencies found.

---

## Execution Handoff

**Plan complete and saved to `external_validation/_rollout_anchors/methodology/docs/2026-05-05-rung-4b-equivariance-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task; review between tasks; fast iteration. Use `superpowers:subagent-driven-development`.

**2. Inline Execution** — execute tasks in this session using `superpowers:executing-plans`; batch execution with checkpoints for review.

**Which approach?**
