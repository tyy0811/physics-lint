# Case Study 02 Phase 1 Cross-Review

Review target: CS02 PhysicsNeMo MGN Phase 1 boundary, with emphasis on Tasks 10-12 (`82d7f68`, `504e104`, `468de33`) and the layered-fail-open lens from round-codex-4.

### Finding 1: Dataset metadata is load-bearing for substrate dispatch but is neither required nor canonicalized

**Severity:** High
**Surface:** `external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py:90-91`, `:606-607`, `:666-667`; `external_validation/_rollout_anchors/_harness/SCHEMA.md:189-197`; `external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py:175-206`; D0-23 verdicts 9-10 at `external_validation/_rollout_anchors/methodology/DECISIONS.md:2383-2387`.
**Evidence:** The mesh dispatch table is keyed by underscore spelling:

```python
MGN_DATASET_SYSTEM_CLASS = {
    "vortex_shedding_2d": "open-driven-dissipative",
}
```

Both dispatch sites then do:

```python
dataset_name = rollout.metadata.get("dataset", "") if rollout.metadata else ""
system_class = MGN_DATASET_SYSTEM_CLASS.get(dataset_name)
```

But the authoritative harness schema still documents the mesh dataset spelling as `"vortex-shedding-2d" | "ahmed-body" | "darcy"` at `_harness/SCHEMA.md:193`, while D0-23/code/tests use `"vortex_shedding_2d"`. A Phase 2 materializer following the schema would set `"vortex-shedding-2d"`, miss the dispatch table, and the rule would emit a raw energy/dissipation value on the known open-driven-dissipative substrate. The tests actively lock in the fail-open fallback for missing dataset metadata: `test_energy_drift_on_mesh_does_not_skip_when_no_substrate_class_match` constructs metadata with no `"dataset"` and asserts `result.value is not None` (`test_mesh_rollout_adapter.py:175-206`). Task 12's metadata check requires only `"framework"` and `"model"` (`mesh_rollout_adapter.py:368-376`), not `"dataset"`, even though D0-23 verdict 9 makes dataset metadata the dispatch key.

Particle-side comparison: the particle adapter has the same fallback form (`particle_rollout_adapter.py:522-523`, `:601-602`) and tests it intentionally (`test_d0_18_dissipative_skip.py:236-252`, `test_d0_22_open_driven_skip.py:192-206`). For CS02 mesh P0, that duplicated shape is riskier because the single known MGN P0 dataset is already classified open-driven-dissipative, and the mesh schema says dataset is a required metadata field.

**Recommended fix:** In the CS02 MGN path, make dataset metadata fail-closed before rule interpretation. At minimum:

- update `_harness/SCHEMA.md` and all Phase 2 materializer docs/tests to one canonical spelling, preferably the D0-23/code spelling `"vortex_shedding_2d"`;
- require `"dataset"` in `_assert_loader_contract_mgn`;
- assert that P0 MGN rollouts use the expected dataset/model/framework tuple, or normalize a small explicit alias map before `MGN_DATASET_SYSTEM_CLASS.get`;
- replace the current mesh test that missing dataset emits raw with a test that MGN-scoped missing/unknown dataset raises or returns a diagnostic SKIP. Keep raw fallback only for non-MGN synthetic fixtures if needed.

**Pattern-C cell:** 2 (novel-in-scope - absorb in-rung)

### Finding 2: `_assert_loader_contract_mgn` no-ops on the exact wrong-key case it is supposed to defend

**Severity:** High
**Surface:** `external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py:314-320`, `:424-455`; `external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py:297-305`; D0-23 verdict 10 at `external_validation/_rollout_anchors/methodology/DECISIONS.md:2386-2387`.
**Evidence:** Task 12 documents `_assert_loader_contract_mgn` as defensive validation before rule kernels consume incoming MGN rollouts, but its first data check is:

```python
velocity = rollout.node_values.get("velocity")
if velocity is None:
    return
```

The corresponding test requires this no-op when `node_values={"pressure": ...}`. If a buggy or adversarial materializer writes the actual velocity under `"u"` or `"flow_field"`, `_assert_loader_contract_mgn` passes. Then the rule kernels call `_expect_velocity`, which returns a SKIP because `"velocity"` is absent (`mesh_rollout_adapter.py:445-454`). That turns a P0 loader-contract violation into an apparently legitimate rule SKIP, so the defensive validator and the rule precondition fail open together.

This is not a hypothetical alternate NGC key for Phase 1: D0-23 verdict 8 pins the actual key to `"velocity"` (`DECISIONS.md:2379-2381`). That makes absence of `"velocity"` in an MGN P0 rollout a contract failure, not merely an optional rule skip.

**Recommended fix:** Make `_assert_loader_contract_mgn` require `"velocity"` for MGN-scoped rollouts and raise an AssertionError naming D0-23 verdict 8 and the present keys. If the generic mesh adapter still needs missing-velocity SKIPs for synthetic/non-MGN fixtures, keep that behavior in `_expect_velocity`, but do not let the MGN loader-contract helper return successfully on absent `"velocity"`.

**Pattern-C cell:** 2 (novel-in-scope - absorb in-rung)

### Finding 3: The MGN loader-contract helper is not wired into any production path

**Severity:** Medium
**Surface:** `external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py:295-376`, `:483-550`, `:576-699`; repository-wide `rg` for `_assert_loader_contract_mgn` at HEAD `868eb94`; design acceptance at `external_validation/_rollout_anchors/methodology/docs/2026-05-11-case-study-02-physicsnemo-mgn-design.md:277-279`.
**Evidence:** `_assert_loader_contract_mgn` claims to fire before rule kernels consume incoming MGN rollouts (`mesh_rollout_adapter.py:300-301`), but the rule kernels do not call it. `mass_conservation_defect_on_mesh`, `energy_drift_on_mesh`, and `dissipation_sign_violation_on_mesh` go directly to `_expect_velocity` / `is_regular_grid` / computation (`mesh_rollout_adapter.py:506-520`, `:588-621`, `:650-686`). A repository-wide search at HEAD found `_assert_loader_contract_mgn` only in its definition, tests, and methodology docs; there is no non-test caller in a materializer, NPZ loader, lint path, or rule wrapper.

That makes Task 12 an opt-in helper rather than an absorbed guard. Phase 2 can still use it correctly, but Phase 1's current code layer does not enforce the contract if Phase 2 forgets the call. This is the layered-fail-open shape from round-codex-4: a defensive validator exists but is outside the execution path it is meant to protect.

**Recommended fix:** Phase 2 should wire `_assert_loader_contract_mgn` into the first trusted MGN boundary, before writing SARIF or invoking per-timestep rule kernels, and add a test that the Phase 2 lint/materialization path rejects an MGN rollout with fp64 velocity, wrong key, missing dataset, or invalid node_type. If a shared wrapper is introduced, keep it MGN-scoped so generic synthetic mesh tests are not forced into the NGC contract.

**Pattern-C cell:** 2 (novel-in-scope - absorb in-rung)

### Finding 4: The canonical NPZ loader upcasts `node_values` to float64, conflicting with the fp32 contract

**Severity:** Medium
**Surface:** `external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py:273-278`, `:324-335`; `external_validation/_rollout_anchors/_harness/SCHEMA.md:182-188`; `external_validation/_rollout_anchors/_harness/tests/test_mesh_read_only_path.py:53-68`; D0-23 verdict 10 at `external_validation/_rollout_anchors/methodology/DECISIONS.md:2386-2387`.
**Evidence:** `_harness/SCHEMA.md:187` specifies `node_values` as fp32. Task 12 then asserts `velocity_arr.dtype == np.float32` (`mesh_rollout_adapter.py:330`). But `load_mesh_rollout_npz` currently does:

```python
node_values = {k: np.asarray(v, dtype=float) for k, v in node_values_raw.items()}
```

On NumPy this produces float64 arrays. So a well-formed `mesh_rollout.npz` written with fp32 values and reloaded through the canonical loader will fail `_assert_loader_contract_mgn` if Phase 2 calls the helper after load. The existing round-trip test checks values but not dtype (`test_mesh_read_only_path.py:53-68`), so this contract drift is untested. Separately, if Phase 2 does not call the helper after load, the lint path will compute on float64 values even though the schema says fp32; that does not affect checkpoint inference, but it does contradict the stored-rollout boundary contract that Task 12 is trying to make explicit.

The NumPy dtype equality itself is not the issue: `np.dtype("float32") == np.float32` is true, while float16/float64 compare false. The issue is the adapter's load-time coercion to Python `float`/float64.

**Recommended fix:** Preserve stored dtype in `load_mesh_rollout_npz` for `node_values` (`np.asarray(v)` or `dtype=np.float32`), and add a round-trip test that fp32 node values remain fp32. If the intended internal representation is float64 for read-only numerical rules, document that separately and do not use `_assert_loader_contract_mgn` after this loader.

**Pattern-C cell:** 2 (novel-in-scope - absorb in-rung)

### Finding 5: Rollout-dir isolation is declared as a Phase 2 obligation, not verified by Phase 1

**Severity:** Low
**Surface:** D0-23 verdict 7 at `external_validation/_rollout_anchors/methodology/DECISIONS.md:2375-2377`; `external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py:1372-1380`, `:1518-1526`, `:1633-1641`, `:1941-1950`; Phase 2 acceptance at `external_validation/_rollout_anchors/methodology/docs/2026-05-11-case-study-02-physicsnemo-mgn-design.md:286-287`.
**Evidence:** Phase 1 audit entrypoints correctly stage `edge_stats.json` / `node_stats.json` in a temporary directory, `chdir` there, and restore CWD in `finally` blocks. For example, `audit_ngc_sample_reproduction` creates `work_dir = tempfile.mkdtemp(...)`, copies stats, calls `os.chdir(work_dir)`, and restores at `modal_app.py:1372-1380` / `:1518-1519`; `smoke_substrate_class_vortex_shedding` does the same at `:1633-1641` / `:1941-1942`.

However, there is no Phase 2 MGN rollout inference entrypoint in Phase 1 to verify. D0-23 verdict 7 says the persistent-volume decision is Y and that "Phase 2's inference entrypoint will do the same" (`DECISIONS.md:2375-2377`). That is a correct forward obligation, but it is not an already-tested property of the future rollout writer. Design Section 4.2 already carries the requirement that round-codex-4 rollout-dir isolation is applied iff Phase 1 committed persistent volume (`design.md:286-287`).

**Recommended fix:** Defer to Phase 2, but keep it as an explicit gate: the first Phase 2 inference entrypoint should create a per-rollout working directory, stage stats there, write outputs under a rollout-specific Volume subdirectory, and include a test or smoke assertion that two same-sha retries cannot read each other's CWD-relative stats or partial outputs.

**Pattern-C cell:** 1 (re-discovery - defer; already predicted by design Section 2.6 / Phase 2 acceptance)

## Search-Lens Null Results

- `_expect_velocity` key pinning: no evidence contradicts D0-23 verdict 8. The raw loader audit and D0-23 both pin the key to `"velocity"`; no helper key-list is needed for Phase 1. The issue is the absent-key no-op in Finding 2, not the actual key choice.
- `np.float32` literal comparison: no alternate NumPy dtype class was found that compares equal to `np.float32` while having a different size class. `np.dtype("float32") == np.float32` is true and has itemsize 4; float16/float64 compare false.
- `regular_grid` fixture deviation: using `metadata["regular_grid"] = True` matches `MeshRollout.is_regular_grid` (`mesh_rollout_adapter.py:189-207`). The plan's constructor keyword would have been invalid; the executed test fixture is the correct route.
- Dispatch after `is_regular_grid`: this is not a current fail-open on real NGC cylinder_flow rollouts because the graph-mesh path returns a SKIP before any raw energy value is emitted. It is still a Phase 2 readiness risk: when graph-mesh materialization is lifted, a test must prove the substrate dispatch actually fires.
- Direct calls to `kinetic_energy_series_on_mesh`: repository search found only tests and same-module callers. The helper still bypasses substrate dispatch by design and returns NaNs on graph-mesh/missing velocity, so Phase 2 SARIF/lint code should call `energy_drift_on_mesh` / `dissipation_sign_violation_on_mesh`, not `kinetic_energy_series_on_mesh` directly.

## Executive Summary

Findings: 5 total: 2 High, 2 Medium, 1 Low. Pattern-C cells: 4 cell-2 findings, 1 cell-1 forward/defer finding, 0 cell-3, 0 cell-4. Phase 1 should not close cleanly until the cell-2 metadata/key/dtype/wiring findings are triaged; the rollout-dir isolation item can remain a Phase 2 gate because the design already carries it explicitly.
