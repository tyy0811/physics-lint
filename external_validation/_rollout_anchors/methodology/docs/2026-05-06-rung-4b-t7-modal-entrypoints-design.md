# Rung 4b — T7 Modal entrypoints (LB-integration design pass)

**Date:** 2026-05-06
**Repo:** physics-lint
**Branch:** `feature/rung-4b-equivariance` (off `master` at `ba13b45`; PR #7 with consumer-side plumbing in flight)
**Status:** Design — pre-implementation. Re-design pass triggered by an under-specification surfaced during execution of the original plan's T7 (see `2026-05-05-rung-4b-equivariance-design.md` §7a addendum at commit `788c847`).
**Predecessor:** Rung 4b consumer-side plumbing (T0–T6 + T8) committed at `c5488d3`. Original plan ([`2026-05-05-rung-4b-equivariance-plan.md`](./2026-05-05-rung-4b-equivariance-plan.md)) under-specified LB's API at the brainstorm depth — implicit assumption was a Python primitive `infer(model, params, x_0, n_steps)`; LB exposes only a CLI subprocess driven via `main.py mode=infer dataset.src=<dir>`.
**Successor:** rung 4b T7 implementation plan, then execution, then T9–T11 follow-up.

---

## 1. Scope and framing

This document is the design pass for T7 only. The original plan's T0–T6 + T8 consumer-side plumbing is frozen (PR #7); ε(t) npz schema, SARIF v1.1, sibling renderer, and the case-study driver all consume whatever the Modal entrypoint produces, so getting T7 right is the only remaining design surface.

**Three options were enumerated in the §7a addendum:**

- **(a) Patch LB upstream** to accept a transformed-IC override via `dataset.src`. Cleanest architecturally; depends on LB maintainer responsiveness and introduces a cross-repo dependency on a not-yet-merged patch. Not chosen.
- **(b) Assemble the forward call manually** by importing LB's internal model + dataset modules and calling forward 1 step in Python. Fragile to LB version bumps (private internals can shift across releases and silently change measurement semantics). Not chosen.
- **(c) Subprocess pattern mirroring rung 4a** — apply the symmetry transform as numpy preprocessing, materialize a synthetic LB-format dataset, run `mode=infer eval.n_rollout_steps=1` via subprocess, post-process. **Chosen** under the matched-stack-consistency principle (D0-17 amendment 1's reasoning generalized to pipeline shape): same subprocess flow as rung 4a → measurement-noise floor stays where rung 4a calibrated it, and the only new code is the IC-materialization step.

The remainder of this document specifies the (c) shape — referred to internally as **(c.1) synthetic-dataset materialization** — at code-ready precision. The H5 schema, manifest schema, metadata reuse policy, off-by-one frame indexing, sanity-probe gate, and silent-mismatch hazards are all locked here so the writing-plans skill produces a code-ready plan rather than re-discovering LB's API during implementation.

---

## 2. Pipeline shape (per Modal entrypoint, run twice — SEGNN, GNS)

1. **Read 6-frame input windows** for traj 0–19 directly from the published TGV2D `test.h5` on Modal Volume using `h5py` (LB's data loader is bypassed; read-only operations don't need the full data-loader machinery). Each window is `positions[0:6, :, :]` shape `(6, N, 2)` plus `particle_type` shape `(N,)` int32.

2. **Write PH-SYM-003 SKIP rows directly** without invoking LB. 20 npzs per stack with `transform_kind="skip"`, `eps_t=[NaN]`, `skip_reason="PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with original"`. Skip-shortcut: running inference to discard the result is wasted compute.

3. **Apply each symmetry transform** (rotation × 4 angles, reflection × 1, translation × 1) to all 6 frames per traj using the existing `symmetry_rollout_adapter` primitives. Velocities are *not* written to the synthetic H5 — LB derives them from finite differences over `pos_input[:, 1:]` and `pos_input[:, :-1]` at load time (`features.py:68`), so velocity equivariance follows automatically from per-frame position rotation. No separate code path.

4. **Materialize two synthetic datasets per stack** at `/rollouts/synthetic/{model}_tgv2d_{sweep_kind}_<sha[:10]>/`:
   - `sweep_kind="main"`: 120 trajectories (6 transforms × 20 trajs), each shape `(7, N, 2)` (6 transformed input frames + 1 placeholder = copy of frame 5).
   - `sweep_kind="figure"`: 3 trajectories (1 angle × 3 trajs), each shape `(106, N, 2)` (6 transformed input frames + 100 placeholders, each = copy of frame 5).

5. **Run the sanity probe** (see §6) before kicking off the full sweep. Probe failure aborts the entrypoint with a diagnostic.

6. **Run LB** twice per stack via subprocess, mirroring rung 4a's invocation shape:
   - Main sweep: `mode=infer dataset.src=<main-dir> dataset.name=tgv_2d eval.test=True load_ckp=<ckpt>/best eval.n_rollout_steps=1 eval.infer.n_trajs=120 eval.infer.out_type=pkl`
   - Figure sweep: same except `eval.n_rollout_steps=100 eval.infer.n_trajs=3`.

7. **Read each rollout pkl**; extract `predicted_rollout[6]` (main, scalar ε) or `predicted_rollout[6:106]` (figure, ε(t) trajectory) — see §5 for the off-by-one rationale. Apply R⁻¹ in numpy. Read rung-4a npz `positions[6]` (or `positions[6:106]`) for the same `original_traj_index` from the manifest. Compute ε via `compute_eps_t_from_pair`. Persist eps_t npz per the existing SCHEMA.md §1.5 contract to `/rollouts/trajectories/{model}_tgv2d_<sha[:10]>/`. Figure-sweep npzs are written *after* main-sweep npzs and overwrite the 3 (PH-SYM-001, π/2, traj∈{0,7,14}) entries with their T_steps=100 counterparts (uniform schema; only `eps_t.shape` changes per design §3.4 "single artifact tier").

**Per-stack npz emission count:** 140 eps_t npzs total — 120 active rows (PH-SYM-001 × 4 angles × 20 + PH-SYM-002 × 1 × 20 + PH-SYM-004 × 1 × 20) plus 20 PH-SYM-003 SKIP rows. 3 of the 120 active rows carry `eps_t.shape == (100,)` after the figure-sweep overwrite; the remaining 117 carry `eps_t.shape == (1,)`. SKIP rows carry `eps_t.shape == (1,)` with `eps_t[0] == NaN`.

---

## 3. Synthetic H5 dataset spec (code-ready)

Per (model, sweep_kind) ∈ {(segnn, main), (segnn, figure), (gns, main), (gns, figure)} = 4 dataset directories total.

### Directory layout

```
<dataset_dir>/
├── train.h5          # 1 dummy 7-frame trajectory (LB requires file exists)
├── valid.h5          # 1 dummy 7-frame trajectory
├── test.h5           # All transformed trajectories (the load-bearing file)
├── metadata.json     # See §3.3
└── manifest.json     # See §3.4
```

LB's data loader requires `train.h5`, `valid.h5`, `test.h5` to all exist (`data.py` instantiates `H5Dataset("train", ...)` and friends regardless of `mode=infer`). Train and valid get a single dummy 7-frame trajectory each; their content doesn't matter (no training, no validation in `mode=infer`).

### `test.h5` contents (HDF5 group layout)

For each synthetic trajectory (`synthetic_traj_index` ∈ `range(num_trajs_test)`), one HDF5 group named `"{synthetic_traj_index:05d}"` containing two datasets:

| Dataset | Shape | Dtype | Source |
|---|---|---|---|
| `position` | `(T_steps, N, 2)` | float32 | 6 transformed input frames + (T_steps - 6) placeholder frames |
| `particle_type` | `(N,)` | int32 | Copied verbatim from published TGV2D `test.h5` traj 0 (TGV2D has uniform N and all-FLUID particles) |

Where `T_steps`:
- main sweep → 7 (= input_seq_length + n_rollout_steps with n_rollout_steps=1)
- figure sweep → 106 (= 6 + 100)

**Placeholder frame contents:** copy of the transformed input window's frame 5. Distribution-safe per the LB schema research (`features.py:68` — FD velocity derivation uses only frames[0:6], so frames ≥ 6 never feed into model input features). LB does use placeholder frames for ground-truth metric computation (`rollout.py:120, 174-176`), but we discard LB's metrics and read predicted_rollout directly.

**Particle ordering and count:** LB's data loader reads `position` and `particle_type` from H5 in their stored order (no re-permutation). For TGV2D `num_particles_max == N == constant` across trajectories; no padding needed. The synthetic H5's `particle_type` array must match the published test.h5 traj 0 ordering so the loaded particles are interpreted correctly by the model's neighbor-list construction.

### `metadata.json` contents (reuse policy)

**Reuse verbatim from published TGV2D `metadata.json`** for the silent-mismatch-hazard fields:

| Field | Source | Why reuse |
|---|---|---|
| `dim` | published (= 2) | physical |
| `dt` | published | EGNN model reads it (runner.py:260) |
| `dx` | published | controls connectivity radius |
| `bounds` | published | neighbor list spatial hashing |
| `periodic_boundary_conditions` | published (= [True, True]) | velocity FD displacement function |
| `default_connectivity_radius` | published | neighbor list construction |
| `num_particles_max` | published | tensor allocation, padding |
| `vel_mean`, `vel_std` | published | **velocity normalization at load time — silent-mismatch hazard if recomputed on synthetic split** |
| `acc_mean`, `acc_std` | published | **acceleration un-normalization in `case.py:integrate_fn` — silent-mismatch hazard if recomputed** |
| `solver`, `case` | published | downstream introspection |

**Synthesize for our split sizes:**

| Field | Value (main) | Value (figure) |
|---|---|---|
| `sequence_length_train` | 7 | 7 |
| `sequence_length_test` | 7 | 106 |
| `num_trajs_train` | 1 | 1 |
| `num_trajs_test` | 120 | 3 |

**Why reuse vs. recompute matters.** `vel_mean / vel_std / acc_mean / acc_std` are computed from finite-difference velocities on the dataset's fluid particles at dataset-generation time (`gen_dataset.py:203-262`). If we recomputed them on our synthetic split, the smaller sample size + transformed positions would shift the stats from the values the trained model expects → all model-input normalization would be slightly off → ε measurements would be polluted by normalization-shift rather than reflecting equivariance error. Reuse keeps the model's input distribution exactly as trained.

### `manifest.json` schema

Sidecar file, not consumed by LB; consumed by the post-processing step in pipeline §2 step 7 to map `synthetic_traj_index` → reference rung-4a npz file + transform parameters.

```json
{
  "schema_version": "1.0",
  "stack": "segnn",
  "dataset": "tgv2d",
  "sweep_kind": "main",
  "physics_lint_sha_eps_computation": "<10-char-prefix>",
  "ckpt_hash": "sha256:<64-char-hex>",
  "trajectories": [
    {
      "synthetic_traj_index": 0,
      "rule_id": "PH-SYM-001",
      "transform_kind": "rotation",
      "transform_param": "pi_2",
      "original_traj_index": 0
    },
    ...
  ]
}
```

Schema-enforcement contract:
- `len(trajectories) == num_trajs_test` (synthesized in §3.3); materialization code asserts equality.
- `synthetic_traj_index` is contiguous over `range(num_trajs_test)`; materialization code asserts contiguity.
- `ckpt_hash` is namespaced `"sha256:<hex>"` to make the digest algorithm self-documenting (matches the sha-naming pattern elsewhere; consumer asserts the prefix).
- Post-processing iterates the trajectories array in synthetic_traj_index order and matches each pkl's filename suffix (`rollout_<i>.pkl` → `synthetic_traj_index = i`) to the manifest entry.

---

## 4. PH-SYM-003 SKIP shortcut (no LB)

Per design §3.6 trigger-vs-emission separation: `so2_substrate_skip_trigger(theta=π/4, has_periodic_boundaries=True)` fires for the (rule, substrate) compatibility check. The Modal entrypoint, after reading the input windows, writes 20 SKIP eps_t npzs per stack *before* materializing any synthetic dataset:

```python
for traj_index in range(20):
    write_eps_t_npz(
        out_dir=out_dir,
        eps_t=np.array([np.nan], dtype=np.float32),
        rule_id="PH-SYM-003",
        transform_kind="skip",
        transform_param="so2_continuous",
        traj_index=traj_index,
        skip_reason="PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with original",
        ...  # provenance fields
    )
```

These 20 npzs do not enter the synthetic-dataset materialization or any LB invocation. The consumer (`lint_eps_dir`) reads them later and emits the SKIP SARIF rows via the shared D0-19 §3.4 emission machinery — no consumer-side change needed.

---

## 5. Reference-comparison frame index (off-by-one resolution)

LB's pkl `predicted_rollout` is `concatenate([initial_positions[:t_window], predicted_frames])` shape `(t_window + n_rollout_steps, N, D)` (verified at `lagrangebench/evaluate/rollout.py:269-272`). With `t_window = input_seq_length = 6`:

- `predicted_rollout[0:6]` = input window (raw from test.h5, frames 0-5)
- `predicted_rollout[6]` = first model prediction = f¹(x_0...x_5)
- `predicted_rollout[7:6+n_rollout_steps]` = subsequent predictions

Rung 4a's pkl→npz converter (`lagrangebench_pkl_to_npz.py:527`) writes `positions = predicted_rollout` without slicing, so rung-4a npzs have shape `(6 + 100, N, D) = (106, N, D)`. Therefore:

- **Reference for ε** = rung-4a npz `positions[6]` (NOT `positions[0]` — off-by-one would silently double-count the input window's identity contribution)
- **Candidate for ε** (main sweep) = synthetic pkl `predicted_rollout[6]`
- **Candidate for ε(t)** (figure sweep) = synthetic pkl `predicted_rollout[6:106]`

The post-processing code slices `[6:6+T_steps]` before passing to `compute_eps_t_from_pair`. The slice index `6` is hard-coded with a `# input_seq_length` comment and an assertion that the published metadata's `input_seq_length == 6` (defensive against an LB version bump that changes the default).

---

## 6. Pre-execution sanity probe (load-bearing gate)

Before the full sweep, the Modal entrypoint runs a 1-traj sanity probe end-to-end:

1. Read rung-4a `particle_rollout_traj00.npz` for SEGNN.
2. Read 6-frame input window for traj 0 from published TGV2D `test.h5`.
3. Apply `rotate_about_box_center(theta=π/2)` to all 6 frames.
4. Materialize a 1-trajectory synthetic dataset.
5. Run LB with `n_rollout_steps=1 eval.infer.n_trajs=1`.
6. Apply R⁻¹ to `predicted_rollout[6]`; compute ε against rung-4a `positions[6]`.

**Gate (must hold):** ε ≤ 1e-5 — abort and surface diagnostic if violated.

**Diagnostic bands within the abort message:**
- ε ∈ (1e-5, 1e-3]: "concerning, possible borderline floating-point variation or partial-bug; investigate before proceeding"
- ε > 1e-3: "clear bug, likely one of [coordinate-space mismatch / off-by-one frame index / normalization stat divergence / manifest-mapping error]; do NOT proceed to full sweep"

Single-threshold gate; (1e-5, 1e-3] is informational, not a separate gate band. SEGNN at the float32 floor on θ=π/2 is expected to land at ~1e-7 — well below 1e-5 — so false-positive risk at the tight gate is low.

The probe runs in ~30s on A10G and catches verification items 1, 2, 3, 4 from the design pass (coordinate space, velocity-storage path, frame-6 placeholder safety, off-by-one indexing). Item 5 (manifest schema) is checked at materialization time via the contiguity assertion in §3.4.

---

## 7. Compute scope (honest wall-time framing)

| Sub-job | LB invocations | Wall time (per invocation) |
|---|---|---|
| segnn main sweep (120 trajs × 1 step) | 1 | ~30-90s |
| segnn figure sweep (3 trajs × 100 steps) | 1 | ~30-90s |
| gns main sweep (120 trajs × 1 step) | 1 | ~30-90s |
| gns figure sweep (3 trajs × 100 steps) | 1 | ~30-90s |
| **Per-stack total (sequential within entrypoint)** | **2** | **~60-180s LB walltime + ~30s materialization + ~10s post-processing** |
| **Both stacks (parallel Modal entrypoints)** | **4** | **~3-7 min total wall time** |

**Honest framing**: wall time is ~3-7 min per stack, dominated by LB invocation overhead (~30-90s startup per subprocess: import + checkpoint load + dataset load). The underlying inference compute remains sub-minute (consistent with the original plan's compute estimate), but **wall time is not sub-minute** because subprocess invocation overhead dominates at this batch size. Recording this as a known property of the (c.1) execution model rather than a discrepancy to be explained later: a future reader observing minutes-not-seconds is not seeing a bug; they are seeing the LB-subprocess overhead the design knowingly inherited from rung 4a's pattern.

GPU class: A10G, matched to rung 4a per D0-21 item 10.

---

## 8. Silent-mismatch hazards (named explicitly)

Each hazard is paired with a guard that catches it loudly.

| Hazard | Guard |
|---|---|
| Coordinate-space mismatch (predicted_rollout in normalized space, rung-4a positions in raw) | Verified at design time: predicted_rollout is raw (un-normalization in `case.py:integrate_fn:246-250`); sanity probe re-verifies at runtime |
| Off-by-one frame index (compare positions[0] when meaning positions[6]) | Hard-coded slice `[6:6+T_steps]` with assertion that `metadata["input_seq_length"] == 6` |
| Recomputed normalization stats on synthetic split shifts model input distribution | metadata.json copies vel_mean/vel_std/acc_mean/acc_std verbatim from published TGV2D metadata; materialization code asserts file-level byte-identity with the published file's normalization fields |
| H5 group ordering vs. rollout pkl filename ordering desync | `manifest.trajectories` is contiguous in `synthetic_traj_index`; consumer matches `rollout_<i>.pkl` filename to `synthetic_traj_index = i` |
| Stored velocity dataset overrides FD-derivation | Synthetic H5 omits the `velocity` dataset entirely (verified LB ignores it, but absence prevents accidental use) |
| Frame-6 placeholder pollutes model input distribution | Verified at design time: FD-velocity uses only frames[0:6] (`features.py:68`), so frame-6 never enters input features |
| Particle-type ordering desync between synthetic and published H5 | Synthetic H5's particle_type is copied verbatim from published test.h5 traj 0; materialization code asserts shape and dtype |
| Sanity probe passes but full sweep fails (rare-traj-specific bug) | Sanity probe is necessary, not sufficient; consumer-side `lint_eps_dir` runs schema validation on every npz before SARIF emission |

---

## 9. Out of scope

- Pulling published TGV2D `metadata.json` from anywhere other than the Modal Volume location rung-4a already populated (`/vol/datasets/lagrangebench/2D_TGV_2500_10kevery100/metadata.json`). Path lookup, not a new download.
- Patching LB upstream (option (a)).
- Importing LB internals (option (b)).
- Changing the consumer-side plumbing (T0–T6 + T8 are frozen on PR #7).
- Producing a 6-trace ε(t) figure visualization (handled in T10/T11; this design only delivers the figure-subset npz data).
- PhysicsNeMo MGN equivariance (case study 02; separate framework, separate dataset format).

---

## 10. Acceptance-criteria deltas vs. original plan §8

The original plan's §8 acceptance criteria are unchanged in their ultimate goal. Three deltas:

1. **New gate before §8 #4** (committed SARIFs land): sanity probe ε ≤ 1e-5 must hold for the SEGNN/π/2/traj00 case before the full sweep is kicked off. If the gate fails, the entrypoint aborts with the diagnostic band; no SARIFs land.

2. **`physics_lint_sha_eps_computation` provenance scope expanded** to cover the synthetic-dataset materialization step in addition to the LB inference step. Both happen inside the same Modal entrypoint at the same git sha; recorded as a single sha (no four-stage sub-split).

3. **Synthetic dataset persisted** to `/rollouts/synthetic/{model}_tgv2d_{sweep_kind}_<sha[:10]>/` on Modal Volume; gitignored locally (extends `outputs/trajectories/` gitignore from T0). Recovery via Modal re-run; same persistence model as eps_t npzs and reference rollouts.

---

## 11. Predecessor → successor

- **Predecessor:** rung 4b consumer-side plumbing — T0–T6 + T8 committed at `c5488d3` on `feature/rung-4b-equivariance` (PR #7); §7a addendum on the original 4b design doc at `788c847`.
- **This document:** rung 4b T7 design pass — pins the LB-integration shape at code-ready precision, resolves the brainstorm-depth gap that surfaced during execution.
- **Next:** rung 4b T7 implementation plan (`2026-05-06-rung-4b-t7-modal-entrypoints-plan.md`) — derived from §§2–8 above via the writing-plans skill.
- **Then:** T7 execution + T9–T11 follow-up (plus a §7b addendum on the original 4b design doc + D0-21 amendment naming the (c.1) resolution).

---

## 12. D-entry footprint

**No new D-entries.** This is a continuation of D0-21's pre-registration; the §7a addendum on the original 4b design doc records the mid-execution under-specification, and a §7b addendum + D0-21 amendment will record the (c.1) resolution and observed sanity-probe outcome after T7 lands. The amendment cadence (single composite D0-21 with footers vs. multiple new D-entries) follows the original plan's choice to keep the rung 4b pre-registration legible as one entry.
