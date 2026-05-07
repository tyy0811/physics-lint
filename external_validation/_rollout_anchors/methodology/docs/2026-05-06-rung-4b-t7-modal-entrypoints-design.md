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

---

## 13. Post-execution amendment 1 — implementation-time corrections (T0–T7 landing)

**Recorded:** after T0–T7 landed (commits `e4b36d6` … `49b2201` on `feature/rung-4b-equivariance`); before T9 (Modal sweep) actually runs. This amendment captures corrections applied during execution; a future §7b addendum on the original 4b design doc will record the observed sanity-probe outcome once T9 fires.

**Meta-note:** the T7 implementation plan (`2026-05-06-rung-4b-t7-modal-entrypoints-plan.md`) transcribed several rung-4a conventions incorrectly at write time. The errors were grep-catchable from `01-lagrangebench/modal_app.py` and `01-lagrangebench/emit_sarif.py`; they were corrected during T4 implementation rather than gating the implementation on a plan re-write. Items 1–5 below are recording-time errors; items 6–7 are design-level patterns the plan glossed over but that have rung-4a precedent — also resolved correction-and-continue. Future plans for adjacent rungs should grep rung-4a's modal_app and emit_sarif for the actual constants before transcribing into prose.

### 13.1 Recording-time corrections (items 1–5)

The plan and this design doc (§10 #3) used `/rollouts/...` paths and named the wrong image variable; rung-4a's actual conventions are different. Corrected during T4 implementation:

| Plan literal | Actual rung-4a convention | Source of truth |
|---|---|---|
| `image=lagrangebench_image` | `image=rollout_image` | `modal_app.py:626, 1001` |
| `volumes={"/rollouts": rollout_volume}` | `volumes={"/vol": rollout_volume}` | `modal_app.py:627, 1474` |
| All `/rollouts/...` paths | `/vol/checkpoints/...`, `/vol/rollouts/lagrangebench/...`, `/vol/datasets/...`, `/vol/synthetic/...`, `/vol/trajectories/...` | `modal_app.py:712, 757, 790, 1474, …` |
| `cwd="/lagrangebench"` for LB subprocess | `os.chdir("/opt/lagrangebench")` then `subprocess.run(...)` (no `cwd=` arg) | `modal_app.py:795–858, 1520, 1540` |
| `rung_4a_subdir = "segnn_tgv2d_post_d03df3e"` (and gns analog) | `segnn_tgv2d_8c3d080397`, `gns_tgv2d_f48dd3f376` | `emit_sarif.py:54–57` |

The plan anticipated the cwd discrepancy (T4.2 step 3) and the rung-4a subdir name discrepancy (T6.1 step 4) as runtime-verification steps. The remaining three (image variable, mount path, /vol path tree) were not anticipated.

### 13.2 4-stage provenance contract on eps_t.npz (item 6)

**Implication-bearing change**: `physics_lint_sha_pkl_inference` and `physics_lint_sha_npz_conversion` are sourced from entrypoint args (hardcoded in the local entrypoint per the `emit_sarif.py` constants pattern), **not** from the rung-4a npz metadata. Reason: rung-4a's `RolloutMetadata` records only a single `git_sha` field — the npz_conversion_sha — because rung-4a's pkl-inference and npz-conversion can happen at the same git sha (the SEGNN case actually has them at distinct shas, `8c3d080397` vs `5857144`, because the SEGNN npzs were re-converted post-D0-17-amendment-1 in a standalone Modal run). The plan's `ref_metadata.get("physics_lint_sha_pkl_inference", "")` would have silently produced empty strings on every eps_t.npz, breaking the SCHEMA.md §1.5 4-stage provenance contract that the consumer-side `lint_eps_dir` + `emit_sarif_eps` pipeline relies on. Resolution pattern matches `emit_sarif.py`'s rung-4a-side approach: hardcoded constants in the local entrypoint, threaded through to the Modal entrypoint as args, recorded into the eps_t.npz at write time, cross-checked against the rung-4a npz's `git_sha` at runtime as a silent-mismatch guard.

### 13.3 Harness-module shipping for cross-importing modules (item 7)

**Methodology-level pattern, not just implementation detail**: rung-4a's `lagrangebench_pkl_to_npz.py` is shipped to the Modal container as a single bare file via `Image.add_local_file`, sys.path-inserted under `/opt/physics_lint_harness/`, and imported via the bare module name (`from lagrangebench_pkl_to_npz import …`). That pattern composes cleanly when the shipped module is self-contained — but rung-4b adds three new harness modules (`synthetic_dataset_materializer`, `eps_pkl_consumer`, `eps_modal_orchestrator`) of which two cross-import from a fourth (`symmetry_rollout_adapter`). The fully-qualified import (`from external_validation._rollout_anchors._harness.symmetry_rollout_adapter import …`) works under local pytest but fails inside the Modal container, where only the bare files are present and `external_validation` is not on sys.path.

**Resolution applied:** all four harness modules are now shipped via `add_local_file` (extending rung-4a's shipping list); the two cross-importing modules use try/except fallback imports (fully-qualified first, bare-name second) so the same source file resolves under both contexts. This is **not just a packaging detail** — it's a scaling pattern from one rung to the next that future case studies will hit whenever a rung needs more than one cross-importing harness module. The alternatives (extracting a single mega-module; shipping the whole `external_validation` tree via `add_local_python_source`; importing the symmetry primitives by inlining) all have non-trivial costs; the try/except + multiple-add_local_file pattern is the lowest-overhead generalization of rung-4a's single-file pattern.

**Forward-flag for future rungs:** if a third cross-importing harness module lands, the per-module `add_local_file` boilerplate scales linearly. At ~5 modules the right move is probably `add_local_python_source` for the whole `_harness/` package; not preemptive at n=4.

### 13.4 Cross-references

- T0 (gitignore): commit `e4b36d6`.
- T1 (synthetic_dataset_materializer): commit `2dcad1e`.
- T2 (eps_pkl_consumer): commit `fd54d34`.
- T3 (eps_modal_orchestrator): commit `db74648`.
- T4 (SEGNN entrypoint, items 1–4 + 6 + 7): commit `c7ee1cd`.
- T5 (GNS entrypoint, mirror): commit `2ccf2a3`.
- T6 (local entrypoints, item 5): commit `1b20efc`.
- T7 (GPU drift-guard): commit `49b2201`.

T9 (run on Modal) and T10/T11 (table render + writeup) deferred to a separate execution session; the §7b addendum on the original 4b design doc will fire after T9 lands.

---

## 14. Post-execution amendment 2 — LB loader-contract failure class (T9 first-fire)

**Recorded:** during the first T9 launch attempt at master sha `22492b06` (PR #7 merged). The SEGNN P0 sweep aborted at Step 3 (sanity probe) — but **not** at the ε > 1e-5 abort gate the design anticipated. Instead, LB's `H5Dataset.__init__` rejected the synthetic test.h5 at config-load time, before any pkl was produced. This amendment records the failure mode, the fix, and a methodology lesson the original design did not contemplate.

### 14.1 The failure

LB's `lagrangebench/data/data.py:144` enforces:

```python
assert self.sequence_length >= self.subseq_length, (
    "# steps in dataset trajectory ({sequence_length}) must be >= "
    "subsequence length ({subseq_length}). Reduce either input_seq_length "
    "or extra_seq_length/max pushforward steps."
)
```

where `subseq_length = input_seq_length + extra_seq_length`. For SEGNN-TGV2D at LB sha `b880a6c84a93792d2499d2a9b8ba3a077ddf44e2`:

- `input_seq_length = 6`
- `extra_seq_length = 4` (from `pushforward.unrolls = [0, 1, 2, 3]` — max unroll 3 + 1 target = 4)
- `subseq_length = 10`

The materializer (T1, commit `2dcad1e`) wrote `t_steps = 7` for main-sweep test.h5 (= `INPUT_SEQ_LENGTH + n_rollout_steps = 6 + 1`), based on the docstring assumption that the constraint was `T >= input_seq_length + n_rollout_steps`. That assumption was **incorrect at the LB-contract level**: `extra_seq_length` is the training-time pushforward horizon, baked into the config at training time, and enforced regardless of `eval.n_rollout_steps`. The assertion fires at `setup_data` for ALL splits (train.h5 with hardcoded `t_steps=7` would have failed too, even on the figure sweep where test.h5 was 106 frames).

### 14.2 The 5th failure class — paired against the four ε bands

The original design anticipated four ε-band failure modes, all sharing a structural property: **pipeline executes end-to-end, ε computes, the bug lives in the measurement.** The diagnostic mechanism is "ε magnitude lands in band X → bug class Y":

| ε band | Bug class |
|---|---|
| ε ∈ [1e-5, 1e-3] | "concerning" — coordinate-space mismatch, frame-index off-by-one, etc. |
| ε > 1e-3 | "clear bug" — normalization wrong, manifest mapping inverted, etc. |

The T9 first-fire abort exposes a **5th failure class** with a different structural property: **pipeline aborts before ε computation; the bug lives in the contract between materialized artifact and consumer's loader.** The diagnostic mechanism here is whatever the loader's error message is (LB's `AssertionError` here is unusually informative; PhysicsNeMo's may not be), and the prevention surface is **pre-flight assertions in materialization that mirror loader-side contracts** — caught at materializer-test time, before any Modal compute.

The two classes are not interchangeable:

|  | ε-band failures (4 classes) | Loader-contract failure (5th class) |
|---|---|---|
| When does it fire? | After end-to-end pipeline run | At loader config-load, before pipeline runs |
| What computes? | ε scalar; magnitude is the diagnostic | Nothing; loader's error message is the diagnostic |
| Where is the bug? | In the measurement (ε formula, frame indices, normalization) | In the artifact↔loader contract |
| Prevention surface | Diagnostic-band methodology (post-hoc band → bug class) | Pre-flight assertions in materializer mirroring loader contracts |
| Cost of catching late | Bad ε numbers in the table; misleading writeup | Modal cold-start + image-pull wasted (~30 s on A10G); no bad data, just a stop |

The diagnostic-band methodology earns its keep when the pipeline runs and ε is wrong. Loader-contract assertions earn their keep when the pipeline doesn't run at all. Both belong in the methodology toolkit; they cover non-overlapping failure surfaces.

### 14.3 The fix

Five changes on `feature/rung-4b-t7-subseq-length-fix` off master:

1. **`synthetic_dataset_materializer.py` constants.** Added `EXTRA_SEQ_LENGTH = 4` and `LB_SUBSEQ_LENGTH = INPUT_SEQ_LENGTH + EXTRA_SEQ_LENGTH = 10` with provenance comment citing LB sha `b880a6c`, ckpt `segnn_tgv2d/best`, and the `pushforward.unrolls = [0,1,2,3]` config dump. Pinned by D0-15 (rung-3 P0 invocation) inherited by D0-21 (rung-4b pre-registration). The LB sha is the captured-at-image-build value (the rollout_image clones `--depth 1` of master, not a sha pin); a rebuild can shift this if upstream LB has moved. Re-derive when the checkpoint changes or LB pushforward semantics change.

2. **Materializer dummy + metadata bump.** train.h5/valid.h5 dummies were hardcoded at `t_steps=7`; bumped to `LB_SUBSEQ_LENGTH=10`. `metadata["sequence_length_train"]` matches.

3. **Materializer assert tightened.** `t_steps >= input_seq_length + 1` (the old "n_placeholders >= 1" check) → `t_steps >= LB_SUBSEQ_LENGTH`. Fail-fast at materializer rather than at LB config-load.

4. **`modal_app.py` call-site bumps.** SEGNN + GNS sanity + main sweep call sites (4 total) bumped from `t_steps=7` to `t_steps=LB_SUBSEQ_LENGTH`. Figure sweep (`t_steps=106`) unchanged. `LB_SUBSEQ_LENGTH` imported alongside `materialize_synthetic_dataset` and `read_published_input_windows`.

5. **`test_synthetic_dataset_materializer.py` regression test.** New section "LB loader-contract assertions" with three tests: `test_lb_subseq_length_matches_pre_registration` (drift-guard for the constants), `test_apply_transform_rejects_t_steps_below_lb_subseq_length` (boundary tests around 10), and `test_materialized_h5s_satisfy_lb_h5dataset_assertion` (end-to-end check that every materialized split satisfies LB's actual H5Dataset assertion). The pattern: **each loader-side assertion that gates pipeline execution gets a paired pre-flight test in the materializer's test suite.**

### 14.4 Forward-flag for future case studies

`subseq_length` is the **first** LB-loader-contract assertion that bit us. Others may exist that haven't surfaced yet — the H5Dataset `__init__` runs more validation than just this assert; the runner's `setup_data` runs more still; and PhysicsNeMo (case study 02) will have its own loader-side assertions, with different shapes.

The pattern that future case studies inherit:

1. **Identify the consumer's loader-side assertions.** For LB: `lagrangebench/data/data.py:144` is one; future contributors should grep `H5Dataset` and `setup_data` for additional ones. For PhysicsNeMo: TBD when case study 02 lands.
2. **Mirror each one in materializer pre-flight.** Same shape as `test_materialized_h5s_satisfy_lb_h5dataset_assertion`: assert that the materializer's output satisfies the assertion the consumer will check.
3. **Cite the source line in the test docstring.** So when the consumer's version moves, future-you knows where to look.
4. **The "LB loader-contract assertions" section is the home for these.** A future PhysicsNeMo case study gets a sibling section ("MGN loader-contract assertions") in its own materializer test file.

This is structurally analogous to the `(rule, substrate)` compatibility forward-flag in the original 4b design doc: when a class of failures has a name and a designated home, future instances are recognized rather than rediscovered cold.

### 14.5 Cross-references

- T9 first-fire failure: Modal app run abort, ~30 s A10G, captured in session conversation. SEGNN sanity-probe synthetic dir: `/vol/synthetic/segnn_tgv2d_sanity_22492b06a3/` (orphan; safe to remove).
- Fix branch: `feature/rung-4b-t7-subseq-length-fix` off master sha `22492b06`.
- Re-fire path: same `modal run …::eps_p0_segnn_tgv2d` invocation; sanity probe is now expected to compute ε (and either pass at ≤ 1e-5 or land in one of the four ε bands the original design enumerated).
- Stale Volume artifact note: `/vol/synthetic/segnn_tgv2d_sanity_22492b06a3/` — written by the failed first-fire; can be cleaned via `modal volume rm` post-fix-success. Likewise `/rollouts/lagrangebench/segnn_tgv2d_f75e22d8dd/` (pre-D0-17-amendment-1 rung-4a rollout, superseded by `_8c3d080397`; orthogonal to this fix).

### 14.6 Second-pass refinement after source review — methodology validation

Before re-firing the SEGNN P0 sweep at the §14.3 fix, source review of LB at sha `b880a6c` (`/lagrangebench/data/data.py:130-145` + `runner.py:163-188`) surfaced **two concrete instances** of the failure class §14.4's forward-flag predicted ("others may exist that haven't surfaced yet"):

**Instance 1 — math error in §14.3.** The original fix declared `EXTRA_SEQ_LENGTH = 4`, claiming "max unroll 3 + 1 target = 4." But LB's source has the `+1` explicit:

```python
# data.py:131 (split == "train")
self.subseq_length = input_seq_length + 1 + extra_seq_length
```

where `extra_seq_length = pushforward.unrolls[-1]` is just the pushforward horizon (= 3, not 4). The "+1" is the target frame — semantically distinct from the unroll count (predict-next-frame target is separate from the pushforward unrolls used for training-time noise injection). The value `LB_SUBSEQ_LENGTH = 10` was correct by coincidence (`6 + 4` happens to equal `6 + 1 + 3`); the derivation was muddled.

**Renamed:** `EXTRA_SEQ_LENGTH` → `LB_PUSHFORWARD_UNROLLS_LAST = 3` (LB-namespaced, parallels other LB-derived constants); `LB_SUBSEQ_LENGTH` → `LB_TRAIN_SUBSEQ_LENGTH = INPUT_SEQ_LENGTH + 1 + LB_PUSHFORWARD_UNROLLS_LAST = 10`. Math now matches LB source verbatim.

**Instance 2 — latent figure-sweep failure.** Per `runner.py:163-188`, LB constructs three H5Datasets at setup_data with **different** `extra_seq_length` kwargs:

| Split | `extra_seq_length` source | At main/sanity (n_rollout_steps=1) | At figure (n_rollout_steps=100) |
|-------|---------------------------|-----------------------------------:|-------------------------------:|
| train | `cfg.train.pushforward.unrolls[-1]` = 3 | subseq_length = **10** | subseq_length = **10** (constant) |
| valid | `cfg.eval.n_rollout_steps` | subseq_length = **7** | subseq_length = **106** |
| test  | `cfg.eval.n_rollout_steps` | subseq_length = **7** | subseq_length = **106** |

The §14.3 fix wrote `valid.h5` and `train.h5` dummies at hardcoded `LB_SUBSEQ_LENGTH = 10`. That clears the sanity probe and main sweep — but **figure sweep would have failed at Step 7**, because `valid.h5` needs ≥ 106 frames when LB constructs `data_valid` with `extra_seq_length = n_rollout_steps = 100`. Same failure class as the original first-fire abort, just on a different split + different sweep + cost of an additional A10G cold-start.

**Fixed:** materializer now writes both dummies (`train.h5`, `valid.h5`) at `t_steps` (= `test.h5`'s length), uniformly. The `t_steps >= LB_TRAIN_SUBSEQ_LENGTH` assert covers the train floor; matching `test.h5`'s length covers the valid/test dynamic floor (since `t_steps` is sized by the caller for the sweep's `n_rollout_steps`).

**Methodology validation, not just fix log.** §14.4's forward-flag wasn't aspirational — it correctly predicted that further loader-contract failures existed and would surface. Source review (no Modal compute, no live testing) surfaced both at **$0 Modal cost** before any feedback loop ran. The cost-benefit ratio that justifies the pattern: source-reading is essentially free, Modal feedback loops aren't. This becomes the **implicit precedent for future LB-integration changes, and for PhysicsNeMo MGN integration when case study 02 lands**: default to source-review pre-flight before any compute, not just when something has already failed.

**Regression test refined:** `test_materialized_h5s_satisfy_lb_h5dataset_assertion` is now `@pytest.mark.parametrize("n_rollout_steps,t_steps", [(1, 10), (100, 106)])` — single test definition, both sweep shapes covered, future contributors add cases by extending the parametrize list. Same scaling shape as the synthetic-fixture-vs-LB-assertion contract pattern: one test definition, multiple instances of the contract.

**Pattern surface, restated.** The §14.4 forward-flag's prescription becomes a four-part procedure:

1. **Identify the consumer's loader-side assertions** by reading source at the pinned sha (not by trial-and-error against live runs).
2. **Mirror each one in materializer pre-flight**, parametrized over the dynamic-axis kwargs the consumer passes (`n_rollout_steps`, etc.).
3. **Cite source line + sha in the test docstring** so future contributors know where to look when the consumer's version moves.
4. **Default to source review before compute.** If the consumer's loader is open-source, the cost of reading it is almost always less than the cost of the Modal cycles you'd otherwise burn on incremental feedback.

This is the pattern PhysicsNeMo MGN (case study 02) inherits when it lands: read MGN's data-loader source at its pinned sha, identify its loader-side assertions, write paired pre-flight tests in the MGN materializer's test file, *before* firing any GPU run.
