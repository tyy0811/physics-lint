# Case Study 02 — PhysicsNeMo MGN Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run vortex-shedding MGN inference end-to-end on Modal A10G (N=1 trajectory, selected via pre-fire Strouhal audit); lint both the GT trajectory (CPU control) and the MGN rollout (Modal) through the mesh harness; emit PH-CON-001/002/003 SARIFs at `02-physicsnemo-mgn/outputs/sarif/`; verify rule outputs land within Phase-1-pre-registered tolerances.

**Architecture:**
- Phase 2A pre-fires a Strouhal-band audit across all cylinder_flow test trajectories before committing to N=1, so the canonical trajectory is selected on a documented criterion (median-Strouhal-in-band), not "trajectory 0 by accident." Phase 2B wires Phase 1's `_assert_loader_contract_mgn` into the materializer boundary (cross-review Finding 3 absorption), reproduces Phase 1's rollout-dir isolation pattern (cross-review Finding 5 absorption) in the Modal MGN inference entrypoint, and emits `mesh_rollout.npz` files per `_harness/SCHEMA.md` §2.
- GT control + MGN rollout are linted in parallel CPU entrypoints — same mesh harness (`mesh_rollout_adapter.py::{mass_conservation_defect_on_mesh, energy_drift_on_mesh, dissipation_sign_violation_on_mesh}`); two SARIF artifacts (`gt.sarif`, `mgn.sarif`); cross-stack consistency table populates with both rows.
- Phase 2D smokes rule outputs against Phase 1's pre-registered tolerances; Pattern-A drift absorbed via D-entry amendment if any. Phase 2E lands the D0-24 (Phase 2 audit verdicts) D-entry and the Phase 2 boundary cross-review.

**Tech Stack:**
- Modal (GPU A10G for inference; CPU for lint and Strouhal audit) — image pinned per Phase 1's `02-physicsnemo-mgn/modal_app.py` shared image definition (`physicsnemo @ 1ca85d65`, torch 2.10.x, scikit-fem 12.0.1).
- PhysicsNeMo v2.0.0 PyG MGN datapipe + `MeshGraphNet(6, 3, 3)` model class + Phase 1's name-remap + edge-MLP column-reorder adapter.
- DeepMind cylinder_flow public dataset (already staged at `/vol/datasets/cylinder_flow/` on the `case-study-02-physicsnemo-artifacts` Volume per Phase 1).
- Harness `_rollout_anchors/_harness/mesh_rollout_adapter.py` (loader-contract assertions, substrate-class dispatch, `MeshField` materialization).
- SARIF v2.1.0 schema per `_harness/SCHEMA.md` §4 + Phase 1's `inference_run_status` framing.

**Inherited forward-flags** (Phase 1 cross-review, D0-23 triage table):
- **Finding 3** — wire `_assert_loader_contract_mgn` into the MGN materializer / lint path's first trusted boundary. Addressed in Task 3.
- **Finding 5** — Modal MGN inference entrypoint reproduces the rollout-dir isolation pattern (`tempfile.mkdtemp` + stage stats + `chdir` + `finally: chdir back`) and carries a smoke assertion that two same-sha retries cannot read each other's CWD-relative stats. Addressed in Task 6 + Task 7.

**User refinements** (from Phase 2 plan-writing round):
- **Strouhal pre-check across cylinder_flow test trajectories** before committing to N=1. Documented trajectory-selection criterion replaces "first by accident." (Refinement 1 → Task 1.)
- **Single-trajectory-vs-ensemble scope qualifier** in Phase 2 writeup + §3.3 "what physics-lint did NOT catch" names the harness-FE-on-P1 floor explicitly as bounding the rule's resolution (NOT "MGN is physically incompressible to 5%"). (Refinement 2 → carried into D0-24 entry text and Phase 3 plan's writeup tasks; flagged in Task 9.)
- **GT + MGN side-by-side lint** (control arm) → two CPU entrypoints, two SARIFs; Task 5 (GT) + Task 7 (MGN).

---

## Phase 2A — Pre-fire audit

### Task 1: Strouhal pre-check across cylinder_flow test trajectories (CPU; Modal)

**Goal:** Before committing N=1, audit the Strouhal-number distribution across ALL cylinder_flow test trajectories. Avoids the false-FAIL class where trajectory 0 happens to have Re or cylinder geometry that produces St ∉ [0.16, 0.21] and the substrate-class smoke verdict's design band is wrongly attributed to MGN. Produces JSON findings + a selection rationale for the Phase 2 canonical trajectory. Per refinement 1.

**Files:**
- Create: `external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py` (new entrypoint `audit_strouhal_test_trajectories`)
- Output: `02-physicsnemo-mgn/preflight/strouhal_test_trajectories.json` (committed)

- [ ] **Step 1: Read meta.json + decode one test record to confirm test-set trajectory count.**

Add a small helper in `modal_app.py` (or inline in the new entrypoint) that opens `/vol/datasets/cylinder_flow/test.tfrecord` with the same `description = {k: "byte" for k in meta["field_names"]}` machinery as `VortexSheddingDataset._load_tfrecord_dataset` (`vortex_shedding_dataset.py:286-306` @ `1ca85d65`), iterates raw records, counts trajectories. This is cheap — counts only, no decode.

```bash
modal run external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py::audit_strouhal_test_trajectories --git-sha "$(git rev-parse --short HEAD)" --count-only
```

Expected output: a single integer printed to stdout (e.g., `N_test = 100` or whatever the public DeepMind cylinder_flow test split actually contains). Record this as `N_test` in the findings JSON.

- [ ] **Step 2: For each test trajectory, compute Strouhal from a downstream wake sampling point.**

Implementation sketch (CPU entrypoint, ~10 minutes total runtime):

```python
@app.function(image=image, volumes={"/vol": mgn_volume}, timeout=60 * 30)
def audit_strouhal_test_trajectories(git_sha: str, count_only: bool = False) -> dict:
    """Phase-2 pre-fire Strouhal audit per refinement 1: span all cylinder_flow
    test trajectories with one CPU pass, FFT a downstream wake-point velocity,
    extract Strouhal St = f_s * D / U_max, write findings JSON.

    Selection criterion documented inline: the canonical Phase 2 trajectory is
    the one whose Strouhal is closest to the median of the in-band subset.
    """
    import os, json, tempfile
    import numpy as np

    # ... (entrypoint setup: chdir to a tempdir under /vol/datasets/cylinder_flow/,
    #      stage edge_stats.json + node_stats.json per the Phase-1 isolation pattern)

    # Decode each test trajectory's velocity field + mesh_pos
    results = []
    for traj_idx, traj in enumerate(iter_test_trajectories()):
        velocity = traj["velocity"]   # (600, N_nodes, 2)
        mesh_pos = traj["mesh_pos"][0]  # (N_nodes, 2) static

        # Sampling-point selection: downstream wake (x ≈ cylinder_center_x + 4 * D,
        # y ≈ cylinder_center_y). Use the Phase 1 substrate-smoke cylinder
        # detection (PhysicsNeMo-style boundary-type heuristic) to find the cylinder
        # center + diameter on THIS trajectory's mesh.
        cyl = _find_cylinder(traj["node_type"][0], mesh_pos)
        sample_xy = (cyl.center[0] + 4 * cyl.diameter, cyl.center[1])
        sample_idx = int(np.argmin(np.linalg.norm(mesh_pos - sample_xy, axis=1)))

        # FFT v_y at sample point; peak frequency = vortex-shedding frequency
        v_y_series = velocity[:, sample_idx, 1]
        warmup = 40  # skip transient — matches Phase 1 substrate-smoke convention
        v_y_trim = v_y_series[warmup:]
        freqs = np.fft.rfftfreq(len(v_y_trim), d=traj["dt"])
        spectrum = np.abs(np.fft.rfft(v_y_trim - v_y_trim.mean()))
        # Ignore DC; find dominant non-zero frequency
        peak_idx = int(np.argmax(spectrum[1:]) + 1)
        f_s = float(freqs[peak_idx])
        U_max = float(_inflow_U_max(traj))  # same as Phase 1 substrate smoke
        strouhal = f_s * cyl.diameter / U_max

        results.append({
            "traj_idx": traj_idx,
            "n_inflow_nodes": int(cyl.n_inflow_nodes),
            "cyl_diameter": float(cyl.diameter),
            "cyl_center": list(cyl.center),
            "U_max": U_max,
            "f_s_Hz": f_s,
            "strouhal": strouhal,
            "in_design_band": 0.16 <= strouhal <= 0.21,
        })

    in_band = [r for r in results if r["in_design_band"]]
    out_band = [r for r in results if not r["in_design_band"]]

    # Phase 2 trajectory selection: median-Strouhal in-band
    if in_band:
        sorted_in_band = sorted(in_band, key=lambda r: r["strouhal"])
        canonical = sorted_in_band[len(sorted_in_band) // 2]
        selection_reason = (
            f"{len(in_band)}/{len(results)} test trajectories land in design band "
            f"[0.16, 0.21]; canonical = median-Strouhal in-band (traj_idx="
            f"{canonical['traj_idx']}, St={canonical['strouhal']:.3f})."
        )
        verdict = "OK"
    else:
        canonical = None
        selection_reason = (
            f"0/{len(results)} test trajectories land in design band [0.16, 0.21]; "
            f"Strouhal range observed: [{min(r['strouhal'] for r in results):.3f}, "
            f"{max(r['strouhal'] for r in results):.3f}]. The literature-anchored "
            f"band may be wrong for this specific cylinder_flow distribution; "
            f"investigate before Phase 2 fires (see plan Task 1 outcome decision tree)."
        )
        verdict = "INVESTIGATE"

    findings = {
        "verdict": verdict,
        "n_test_trajectories": len(results),
        "n_in_design_band": len(in_band),
        "n_out_of_design_band": len(out_band),
        "design_band": [0.16, 0.21],
        "per_trajectory": results,
        "canonical_trajectory": canonical,
        "selection_reason": selection_reason,
        "physicsnemo_sha": "1ca85d65ac2ce28ea9762910c09a954c08a37140",
        "git_sha": git_sha,
    }

    # Write to both /vol mirror and (via fan-out caller) the local preflight dir
    out_path = "/vol/datasets/cylinder_flow/strouhal_test_trajectories.json"
    with open(out_path, "w") as f:
        json.dump(findings, f, indent=2)
    mgn_volume.commit()
    return findings
```

- [ ] **Step 3: Fire the entrypoint (CPU; ~10 min on Modal default container).**

```bash
modal run external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py::audit_strouhal_test_trajectories --git-sha "$(git rev-parse --short HEAD)"
```

Capture stdout (the dict return) + pull the JSON to the local preflight mirror:

```bash
modal volume get case-study-02-physicsnemo-artifacts /datasets/cylinder_flow/strouhal_test_trajectories.json external_validation/_rollout_anchors/02-physicsnemo-mgn/preflight/strouhal_test_trajectories.json
```

- [ ] **Step 4: Branch on the verdict per the refinement-1 decision tree.**

| Verdict | Action |
|---|---|
| `OK` with all trajectories in band | Proceed to Task 2 with `canonical_trajectory.traj_idx`. |
| `OK` with subset in band | Proceed with the in-band canonical; record the out-of-band count + indices in D0-24 (Phase 2 audit verdicts skeleton) as a substrate-variability finding. |
| `INVESTIGATE` (0 in band) | **STOP. Do NOT proceed to Task 2.** Open a discussion: is the literature-anchored band [0.16, 0.21] wrong for this dataset's Reynolds range, or is the cylinder detection / sampling-point logic buggy? Investigate before any A10G fire. |

- [ ] **Step 5: Commit.**

```bash
git add external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py \
        external_validation/_rollout_anchors/02-physicsnemo-mgn/preflight/strouhal_test_trajectories.json
git commit -m "02-physicsnemo-mgn: Task 1 — Strouhal pre-check across cylinder_flow test trajectories (Phase 2 refinement 1)"
```

---

### Task 2: Canonical-trajectory selection pinned in DECISIONS.md (D0-24 skeleton)

**Goal:** Pre-register the D0-24 (Phase 2 audit verdicts) D-entry with Task 1's selection + the verdict-bands for the Phase 2 fires. Per the Phase 1 D0-23 pattern: pre-commit verdict bands BEFORE firing so the routing is not interpreted-during-execution.

**Files:**
- Modify: `external_validation/_rollout_anchors/methodology/DECISIONS.md` (add D0-24 skeleton, status `open`)

- [ ] **Step 1: Append D0-24 skeleton to DECISIONS.md.**

Template (replace placeholders with Task 1 actuals):

```markdown
## D0-24 — 2026-05-13 — Case Study 02 Phase 2 audit verdicts (open)

Phase 2 fires the MGN inference end-to-end on the canonical cylinder_flow
test trajectory selected in Task 1, then lints both GT and MGN rollouts
through the mesh harness. Verdicts pre-registered here BEFORE firing
(D0-23 pattern; cap-rationale per [[feedback_cap_rationale_not_literal]]).

**Canonical trajectory (Task 1 verdict):** traj_idx=<FROM_TASK_1>, St=<FROM_TASK_1>,
cyl_diameter=<FROM_TASK_1>, U_max=<FROM_TASK_1>. Selection criterion:
median-Strouhal-in-band. See preflight/strouhal_test_trajectories.json.

**Verdict bands (pre-fire commit):**

1. **PH-CON-001 (mass) on GT trajectory:** harness-FE-on-P1 floor was
   ~5% relative divergence per Phase 1 substrate-smoke
   (preflight/substrate_class_smoke.json `incompressibility` row). Phase 2
   bands:
   - ≤ 6% (10 % wider than floor): PASS-control; the FE+P1 floor is
     stable across trajectories.
   - (6%, 10%]: MARGINAL; trajectory-dependent floor; record per-trajectory
     variability in writeup but no methodology amendment.
   - > 10%: FAIL-control; the harness floor is unstable across trajectories;
     widen substrate-smoke design-band claim or audit the FE implementation.

2. **PH-CON-001 (mass) on MGN rollout:** Phase 1 substrate-smoke had
   ~5% on the 399-step MGN rollout. Phase 2 bands (n_rollout_steps=599
   per inference.py default):
   - within ±20% of GT-trajectory PH-CON-001 verdict: PASS — MGN reproduces
     GT-level mass conservation at the harness floor.
   - 20%-50% above GT: MARGINAL; bounded model error above the floor; named
     in writeup but no amendment.
   - > 50% above GT: FAIL; MGN's mass-conservation defect exceeds the
     harness-floor noise floor; PH-CON-001 has a meaningful signal.

3. **PH-CON-002 (energy drift):** D0-23 v9 routes vortex_shedding_2d via
   the open-driven-dissipative dispatch ⇒ SKIP-with-reason expected. Bands:
   - SKIP-with-reason cites D0-22 + D0-23: PASS (dispatch fires as designed).
   - Raw value emitted (dispatch missed): FAIL; loader-contract assertion
     should have caught the missing/wrong dataset metadata.

4. **PH-CON-003 (dissipation_sign_violation):** same dispatch ⇒ SKIP
   expected. Bands as PH-CON-002.

5. **Loader-contract enforcement:** Task 3 wires `_assert_loader_contract_mgn`
   into the materializer; Task 3's 4 rejection tests must fail-closed on
   each of: fp64 velocity, wrong velocity key, missing `dataset`, invalid
   `node_type`. Verdict = PASS iff all 4 reject with informative
   AssertionError.

6. **Rollout-dir isolation (Finding 5 absorption):** Task 6's Modal
   entrypoint reproduces the Phase 1 pattern + Task 7's smoke assertion
   verifies two same-sha retries cannot read each other's CWD-relative
   stats. Verdict = PASS iff the smoke assertion fires green on a fresh
   container.

7. **`inference_run_status` field:** Task 7's SARIF emission must include
   the field per design §2.5; uniform `from_completed_inference` predicted;
   salvage triggers fire forward-flag instead of code change.

Open until all 7 verdicts pinned (Tasks 5/6/7/8/9) + Phase 2 cross-review
(Task 11) findings triaged.
```

- [ ] **Step 2: Commit the D-entry skeleton.**

```bash
git add external_validation/_rollout_anchors/methodology/DECISIONS.md
git commit -m "DECISIONS.md: D0-24 Phase 2 audit verdicts skeleton — canonical trajectory pinned + 7 verdict bands pre-registered"
```

---

## Phase 2B — Materializer + harness MeshField wiring + GT control lint

### Task 3: Wire `_assert_loader_contract_mgn` into the materializer boundary (Finding 3 absorption)

**Goal:** Phase 1 cross-review Finding 3: the helper is opt-in; no production caller. Wire it at the first trusted MGN boundary so the contract is enforced before rule kernels consume rollouts. The natural boundary is `load_mesh_rollout_npz` for the lint path (Task 5 + Task 7 both consume saved NPZ files); the inference entrypoint (Task 6) writes a fresh MeshRollout that should also be validated before NPZ write.

**Files:**
- Modify: `external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py` (call `_assert_loader_contract_mgn` from `load_mesh_rollout_npz` when MGN-scoped; expose a public `materialize_mgn_rollout()` helper that calls the assertion)
- Modify: `external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py` (4 rejection tests on the wired path)

- [ ] **Step 1: Write the failing test — `load_mesh_rollout_npz` validates MGN-scoped rollouts.**

```python
def test_load_mesh_rollout_npz_calls_loader_contract_when_mgn_scoped(tmp_path):
    """Phase-1 cross-review Finding 3 absorption: when the NPZ's metadata
    identifies an MGN rollout (model startswith "modulus_"), the loader
    must call _assert_loader_contract_mgn before returning. Verifies the
    helper is wired into the first trusted boundary."""
    # Construct a malformed-but-loadable rollout: well-formed enough that
    # MeshRollout.__post_init__ accepts, but with fp64 velocity that
    # _assert_loader_contract_mgn must reject.
    bad_rollout = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"velocity": np.ones((5, 10, 2), dtype=np.float64)},  # fp64 wrong
        dt=0.01,
        metadata={
            "framework": "pytorch+dgl",
            "model": "modulus_ns_meshgraphnet",  # MGN-scoped → contract enforced
            "dataset": "vortex_shedding_2d",
        },
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    save_mesh_rollout_npz(bad_rollout, tmp_path / "bad.npz")
    with pytest.raises(AssertionError, match="float32"):
        load_mesh_rollout_npz(tmp_path / "bad.npz")
```

- [ ] **Step 2: Run test — expected FAIL (no wiring yet).**

```bash
source .venv/bin/activate
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py::test_load_mesh_rollout_npz_calls_loader_contract_when_mgn_scoped -v
```

Expected: FAIL — `load_mesh_rollout_npz` returns the rollout without raising.

- [ ] **Step 3: Wire the contract into `load_mesh_rollout_npz`.**

In `mesh_rollout_adapter.py`, append to `load_mesh_rollout_npz` just before the `return MeshRollout(...)`:

```python
    rollout = MeshRollout(
        node_positions=node_positions,
        node_type=node_type,
        node_values=node_values,
        dt=dt,
        metadata=metadata,
        edge_index=edge_index,
    )
    # Finding 3 absorption (Phase 1 cross-review): enforce the MGN loader
    # contract at the first trusted boundary. Scope detection: metadata["model"]
    # starts with "modulus_" identifies an MGN rollout (vs synthetic /
    # FNO-on-Darcy / future stacks). Generic mesh rollouts (synthetic, FNO)
    # bypass — they don't claim to satisfy the MGN contract.
    model_name = str(rollout.metadata.get("model", ""))
    if model_name.startswith("modulus_"):
        _assert_loader_contract_mgn(rollout)
    return rollout
```

- [ ] **Step 4: Run the test (expected PASS).**

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py::test_load_mesh_rollout_npz_calls_loader_contract_when_mgn_scoped -v
```

Expected: 1 passed.

- [ ] **Step 5: Add three more rejection tests (wrong velocity key, missing dataset, invalid node_type) on the same wired path.**

```python
def test_load_mesh_rollout_npz_rejects_mgn_rollout_with_wrong_velocity_key(tmp_path):
    """Finding 3: MGN rollout with node_values under wrong key fails-loud at load."""
    bad = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"u": np.ones((5, 10, 2), dtype=np.float32)},  # wrong key
        dt=0.01,
        metadata={
            "framework": "pytorch+dgl",
            "model": "modulus_ns_meshgraphnet",
            "dataset": "vortex_shedding_2d",
        },
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    save_mesh_rollout_npz(bad, tmp_path / "bad_key.npz")
    with pytest.raises(AssertionError, match="velocity"):
        load_mesh_rollout_npz(tmp_path / "bad_key.npz")


def test_load_mesh_rollout_npz_rejects_mgn_rollout_missing_dataset_metadata(tmp_path):
    """Finding 3: MGN rollout missing 'dataset' metadata fails-loud at load
    (else v9 substrate-class dispatch silently no-ops)."""
    bad = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.zeros(10, dtype=np.int64),
        node_values={"velocity": np.ones((5, 10, 2), dtype=np.float32)},
        dt=0.01,
        metadata={"framework": "pytorch+dgl", "model": "modulus_ns_meshgraphnet"},
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    save_mesh_rollout_npz(bad, tmp_path / "bad_meta.npz")
    with pytest.raises(AssertionError, match="dataset"):
        load_mesh_rollout_npz(tmp_path / "bad_meta.npz")


def test_load_mesh_rollout_npz_rejects_mgn_rollout_with_invalid_node_type(tmp_path):
    """Finding 3: MGN rollout with node_type out of {0,3,4,5,6} fails-loud."""
    bad = MeshRollout(
        node_positions=np.zeros((10, 2), dtype=np.float32),
        node_type=np.array([0, 3, 4, 5, 6, 7, 0, 0, 0, 0], dtype=np.int64),  # 7 invalid
        node_values={"velocity": np.ones((5, 10, 2), dtype=np.float32)},
        dt=0.01,
        metadata={
            "framework": "pytorch+dgl",
            "model": "modulus_ns_meshgraphnet",
            "dataset": "vortex_shedding_2d",
        },
        edge_index=np.zeros((2, 0), dtype=np.int64),
    )
    save_mesh_rollout_npz(bad, tmp_path / "bad_ntype.npz")
    with pytest.raises(AssertionError, match="node_type"):
        load_mesh_rollout_npz(tmp_path / "bad_ntype.npz")
```

- [ ] **Step 6: Run the full mesh_rollout_adapter test suite (expected: all pass).**

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py -v
```

- [ ] **Step 7: Verify the synthetic mesh tests still pass (regression — generic mesh rollouts must NOT be subjected to MGN contract).**

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_read_only_path.py -v
```

Expected: 18 passed (no new tests; just confirming `framework="synthetic"` rollouts bypass the assertion).

- [ ] **Step 8: Commit.**

```bash
git add external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py \
        external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py
git commit -m "02-physicsnemo-mgn: Task 3 — wire _assert_loader_contract_mgn into load_mesh_rollout_npz (Phase-1 cross-review F3 absorption)"
```

---

### Task 4: Wire MeshField graph-mesh path into harness rule mirrors (Gate A PASS branch lifts the graph-mesh SKIP)

**Goal:** Phase 1's Gate A verdict (D0-23 v3) was PASS — `MeshTri + Basis(ElementTriP1())` reconstructs cleanly on cylinder_flow's mesh. But Phase 1 only verified the reconstruction; it never wired the MeshField path into the harness rule mirrors. As a result, `mass_conservation_defect_on_mesh` / `energy_drift_on_mesh` / `dissipation_sign_violation_on_mesh` all currently SKIP on graph-mesh inputs (the explicit `if not rollout.is_regular_grid: return HarnessDefect(skip_reason=...)` branch). Design §3.2 activity 4 requires the rules to RUN on graph mesh, not SKIP. Phase 2 lifts the SKIP by adding the scikit-fem-based MeshField branch — porting the substrate-class smoke's `∫|∇·v|/∫‖∇v‖_F` computation (already validated in Phase 1) into the harness.

Activating the rule mirrors on graph-mesh also activates the D0-23 v9 substrate-class dispatch (which has been dead code on real NGC rollouts until this Task) — Phase 2 finally exercises the dispatch end-to-end.

**Files:**
- Modify: `external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py` (graph-mesh branches in the three `*_on_mesh` functions + a new `kinetic_energy_series_on_mesh_via_fe` helper)
- Modify: `external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py` (new test fixture: graph-mesh rollout that exercises each rule's MeshField path)

- [ ] **Step 1: Locate the substrate-class smoke's FE-divergence computation as the template.**

```bash
grep -n "incompressibility\|∫|∇·v|\|fe_divergence\|scikit-fem\|ElementTriP1\|skfem\|MeshTri" external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py | head -20
```

Read the matching function in `modal_app.py` (likely inside `smoke_substrate_class_vortex_shedding`). It is the scikit-fem implementation that Phase 2 will lift into the harness. Note the baked-in self-test (`v=(y,-x)` → divergence ≈ 1.4e-15; `v=(x,0)` → divergence = total_area) — that self-test moves into the harness too, as an init-time assertion or a dedicated harness test.

- [ ] **Step 2: Write a failing test — `mass_conservation_defect_on_mesh` runs on a graph-mesh rollout (no SKIP).**

```python
def test_mass_conservation_defect_on_mesh_runs_on_graph_mesh_via_meshfield():
    """Phase 2 Task 4: Gate A PASS branch wires MeshField (scikit-fem P1)
    into the harness rule mirror. A graph-mesh rollout no longer SKIPs;
    instead, the rule computes ∫|∇·v|/∫‖∇v‖_F via FE.

    Fixture: synthetic triangle mesh + analytically-divergence-free
    velocity field (v = (y, -x) on a unit-square mesh). Expected output:
    ~machine-epsilon defect (FE-divergence floor).
    """
    # Build a small triangle mesh + DGL-style edge_index
    n_nodes = 25  # 5x5 vertex grid → ~32 triangles
    xs, ys = np.meshgrid(np.linspace(0, 1, 5), np.linspace(0, 1, 5), indexing="ij")
    positions = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)

    # Divergence-free: v = (y, -x)
    v0 = np.stack([positions[:, 1], -positions[:, 0]], axis=1).astype(np.float32)
    velocity = np.tile(v0[None, ...], (3, 1, 1))  # (T=3, N=25, D=2)

    # Cells: 2 triangles per 4-vertex cell. For 5x5 grid: 4x4 cells × 2 = 32 triangles.
    cells = _build_unit_square_triangulation(nx=5, ny=5)  # (32, 3) int64
    edge_index = _cells_to_edge_index(cells)  # (2, n_edges) int64

    rollout = MeshRollout(
        node_positions=positions,
        node_type=np.zeros(n_nodes, dtype=np.int64),
        node_values={"velocity": velocity},
        dt=0.01,
        metadata={
            "framework": "pytorch+dgl",  # graph mesh
            "model": "synthetic-test",   # NOT modulus_ — bypasses MGN contract
            "dataset": "synthetic_unit_square",
            "cells_2d": cells,  # supplied so the harness doesn't have to infer triangulation
        },
        edge_index=edge_index,
    )

    result = mass_conservation_defect_on_mesh(rollout)
    # Graph-mesh path must NOT SKIP anymore — it runs the FE divergence
    assert result.value is not None, (
        f"graph-mesh path must lift the SKIP; got skip_reason={result.skip_reason}"
    )
    # Divergence-free velocity → defect at FE floor (matches Phase 1 substrate-smoke's
    # 1.4e-15 self-test result for v=(y,-x))
    assert result.value < 1e-12, (
        f"divergence-free v=(y,-x) should yield FE-floor defect; got {result.value:.3e}"
    )
```

(Test helper `_build_unit_square_triangulation` + `_cells_to_edge_index` may need to live in the same test file or a fixtures module — keep them local for simplicity.)

- [ ] **Step 3: Run the test (expected FAIL — graph-mesh path still SKIPs).**

```bash
source .venv/bin/activate
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py::test_mass_conservation_defect_on_mesh_runs_on_graph_mesh_via_meshfield -v
```

Expected: FAIL — `result.value is None` because the function returns `HarnessDefect(skip_reason="mesh is graph-topology ...")`.

- [ ] **Step 4: Implement the MeshField graph-mesh branch.**

In `mesh_rollout_adapter.py`, factor a helper `_fe_divergence_defect(rollout, velocity)` that uses scikit-fem (lifted from the substrate-class smoke entrypoint). Then modify `mass_conservation_defect_on_mesh`:

```python
def mass_conservation_defect_on_mesh(rollout: MeshRollout) -> HarnessDefect:
    """[existing docstring...]

    Phase-2 Task 4 absorption: the prior graph-mesh SKIP is lifted —
    on graph-mesh rollouts where the mesh can be coerced to a
    scikit-fem MeshTri (Gate A PASS branch per D0-23 v3), the rule
    computes ∫|∇·v| / ∫‖∇v‖_F via FE (scikit-fem ElementTriP1).
    Falls back to the prior SKIP only when the mesh cannot be
    triangulated (Gate A FAIL — not expected for cylinder_flow per
    Phase 1).
    """
    velocity = _expect_velocity(rollout)
    if isinstance(velocity, HarnessDefect):
        return velocity
    if rollout.is_regular_grid:
        # Existing regular-grid FD path (unchanged)
        return _mass_conservation_defect_on_regular_grid(rollout, velocity)
    # Graph-mesh path (NEW — Phase 2 Task 4)
    return _mass_conservation_defect_on_graph_mesh_via_fe(rollout, velocity)


def _mass_conservation_defect_on_graph_mesh_via_fe(
    rollout: MeshRollout, velocity: np.ndarray
) -> HarnessDefect:
    """Compute ∫|∇·v|/∫‖∇v‖_F per timestep via scikit-fem P1 FE on the
    triangulated mesh. Mirrors the substrate-class smoke's incompressibility
    computation. Returns max over t."""
    try:
        import skfem
        from skfem import Basis, ElementTriP1, MeshTri
        from skfem.helpers import grad
    except ImportError as e:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"scikit-fem not available ({e}); graph-mesh PH-CON-001 needs the FE path. "
                f"Install via pyproject.toml extras."
            ),
        )

    # Triangulation: prefer rollout.metadata["cells_2d"] if present, else
    # reconstruct from edge_index (more work; defer to v1.1 if not needed).
    cells = rollout.metadata.get("cells_2d")
    if cells is None:
        return HarnessDefect(
            value=None,
            skip_reason=(
                "graph-mesh PH-CON-001 needs metadata['cells_2d'] for scikit-fem "
                "MeshTri reconstruction; provide it at materialization time. See "
                "D0-23 verdict 3 (Gate A PASS) for the PyG→scikit-fem coercion shape."
            ),
        )
    mesh = MeshTri(p=rollout.node_positions.astype(np.float64).T, t=np.asarray(cells).astype(np.int64).T)
    basis = Basis(mesh, ElementTriP1())

    max_relative = 0.0
    for t_idx in range(velocity.shape[0]):
        v_t = velocity[t_idx].astype(np.float64)  # (N_nodes, 2)
        # ∫|∇·v| via FE
        # ... implementation lifted from substrate-class smoke ...
        # ∫‖∇v‖_F via FE
        # ...
        relative = int_abs_div / max(int_norm_grad, 1e-12)
        if relative > max_relative:
            max_relative = relative
    return HarnessDefect(value=max_relative)
```

(Exact `_fe_divergence_defect` body lifted verbatim from `02-physicsnemo-mgn/modal_app.py` `smoke_substrate_class_vortex_shedding`'s FE-divergence section.)

- [ ] **Step 5: Run the test (expected PASS).**

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py::test_mass_conservation_defect_on_mesh_runs_on_graph_mesh_via_meshfield -v
```

Expected: 1 passed (`result.value < 1e-12`).

- [ ] **Step 6: Repeat for `energy_drift_on_mesh` + `dissipation_sign_violation_on_mesh`.**

Both functions currently SKIP on `not is_regular_grid`. The KE-integration analog uses scikit-fem to compute `∫ 0.5 ρ |v|² dV` per timestep — same triangulation, same Basis. Factor a helper `_kinetic_energy_series_on_graph_mesh_via_fe(rollout, velocity)` mirroring `kinetic_energy_series_on_mesh`.

Tests follow the same shape as Step 2 but exercise:
- `energy_drift_on_mesh` on a constant-velocity graph-mesh rollout → drift = 0
- `dissipation_sign_violation_on_mesh` on a constant-velocity rollout → 0 dissipation
- BOTH should ALSO exercise the D0-23 v9 dispatch by setting `dataset="vortex_shedding_2d"` on the metadata — the dispatch must SKIP-with-reason on this substrate class, which is the v9 verdict's primary intent.

```python
def test_energy_drift_on_graph_mesh_fires_substrate_class_dispatch():
    """Phase 2 Task 4 + D0-23 v9 cross-check: with the graph-mesh SKIP
    lifted, the v9 substrate-class dispatch now fires on real-shape MGN
    inputs. metadata['dataset']='vortex_shedding_2d' → SKIP-with-reason
    citing D0-22 + D0-23."""
    # Build a graph-mesh rollout with vortex_shedding_2d metadata and
    # constant-velocity (clears KE_REST_THRESHOLD so D0-08 doesn't preempt).
    rollout = _build_synthetic_graph_mesh_rollout(
        dataset="vortex_shedding_2d",
        constant_velocity=10.0,
    )
    result = energy_drift_on_mesh(rollout)
    assert result.value is None
    assert "open-driven-dissipative" in (result.skip_reason or "")
    assert "D0-22" in (result.skip_reason or "") or "D0-23" in (result.skip_reason or "")
```

- [ ] **Step 7: Run all graph-mesh tests + the existing mesh test suite (regression check).**

```bash
pytest external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py external_validation/_rollout_anchors/_harness/tests/test_mesh_read_only_path.py -v
```

Expected: all pass. The existing regular-grid tests are unaffected (the regular-grid branch is unchanged); the graph-mesh-SKIP tests in `test_mesh_read_only_path.py` should still pass IF they used `framework="pytorch+dgl"` without supplying `metadata["cells_2d"]` — the graph-mesh branch will then SKIP with the "needs cells_2d" reason (which is more informative than the old "Day 2 hour 1 audit" SKIP). Update those test expectations if needed.

- [ ] **Step 8: Add scikit-fem to the package extras if not already present.**

```bash
grep -n "scikit-fem\|skfem" pyproject.toml
```

If absent: add `scikit-fem>=12,<13` under `[project.optional-dependencies]` `validation-rollout` extra. Don't add to default extras — the harness's regular-grid path doesn't need it.

- [ ] **Step 9: Commit.**

```bash
git add external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py \
        external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py \
        pyproject.toml
git commit -m "_harness: wire MeshField (scikit-fem P1) into *_on_mesh rule mirrors — graph-mesh SKIP lifted; D0-23 v9 dispatch now exercises on real shape (Phase 2 Task 4)"
```

---

### Task 5: GT-trajectory lint entrypoint + SARIF (CPU; control arm)

**Goal:** Lint the canonical GT cylinder_flow test trajectory (selected in Task 1) through the mesh harness; emit `gt.sarif`. CPU-only — no inference, just FE-divergence on the stored DeepMind trajectory. Establishes the control-arm baseline that Task 7's MGN-rollout SARIF compares against. Per user refinement: GT + MGN side-by-side.

**Files:**
- Create: function `lint_gt_trajectory` in `02-physicsnemo-mgn/modal_app.py` (CPU; reads `/vol/datasets/cylinder_flow/test.tfrecord`)
- Create: `02-physicsnemo-mgn/outputs/sarif/gt.sarif` (committed)
- Modify: `02-physicsnemo-mgn/modal_app.py`

- [ ] **Step 1: Read the LB-side SARIF-emitter helpers as a template.**

```bash
grep -n "def emit_\|^def " external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py | head -10
ls external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/
```

The MGN side will reuse the same SARIF schema + `inference_run_status` field; ideally factor a shared `_harness/emit_sarif_mesh.py` if the LB-side emit_sarif isn't directly reusable. (Decide during implementation; the plan permits either.)

- [ ] **Step 2: Implement the GT lint entrypoint.**

```python
@app.function(image=image, volumes={"/vol": mgn_volume}, timeout=60 * 30)
def lint_gt_trajectory(git_sha: str, traj_idx: int) -> dict:
    """Phase 2 control-arm lint: PH-CON-001 (mass) + PH-CON-002 (energy_drift)
    + PH-CON-003 (dissipation_sign) on the canonical GT cylinder_flow test
    trajectory (selected via Task 1). CPU-only.

    Materializes the trajectory as a MeshRollout (Pattern-B P0 single-instance),
    applies the harness *_on_mesh mirrors, emits one SARIF artifact at
    /vol/rollouts/physicsnemo/vortex_shedding_<git_sha>/gt.sarif (Volume mirror)
    + local commit at 02-physicsnemo-mgn/outputs/sarif/gt.sarif.

    Note: D0-23 v9 dispatch will SKIP PH-CON-002/003 on this substrate
    (open-driven-dissipative) — that's expected and documented in D0-24 v3/v4.
    """
    import json, os, tempfile
    import numpy as np

    from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
        MeshRollout,
        dissipation_sign_violation_on_mesh,
        energy_drift_on_mesh,
        mass_conservation_defect_on_mesh,
    )

    # Stage stats per Phase 1 isolation pattern (CWD-relative reads in
    # VortexSheddingDataset:103,141 @ 1ca85d65) — even though THIS entrypoint
    # doesn't instantiate the dataset, the helper that decodes records reuses
    # the loader machinery, which needs the stats nearby.
    with tempfile.TemporaryDirectory() as work_dir:
        os.symlink("/vol/datasets/cylinder_flow/edge_stats.json",
                   os.path.join(work_dir, "edge_stats.json"))
        os.symlink("/vol/datasets/cylinder_flow/node_stats.json",
                   os.path.join(work_dir, "node_stats.json"))
        old_cwd = os.getcwd()
        os.chdir(work_dir)
        try:
            traj = _decode_test_trajectory(traj_idx)  # (velocity, mesh_pos, node_type, ...)
        finally:
            os.chdir(old_cwd)

    # Build MeshRollout — fp32 throughout per loader-contract assertion.
    rollout = MeshRollout(
        node_positions=traj["mesh_pos"][0].astype(np.float32),
        node_type=traj["node_type"][0].astype(np.int64).squeeze(-1),
        node_values={
            "velocity": traj["velocity"].astype(np.float32),
            "pressure": traj["pressure"].astype(np.float32),
        },
        dt=0.01,  # cylinder_flow standard
        metadata={
            "framework": "deepmind-cylinder-flow-gt",  # NOT pytorch+dgl — this is GT
            "model": "deepmind-meshgraphnets-2020",
            "dataset": "vortex_shedding_2d",
            "regular_grid": False,  # graph-mesh GT
            "git_sha": git_sha,
            "ngc_version": "n/a (DeepMind public dataset)",
            "ckpt_hash": "n/a (GT)",
        },
        edge_index=_cells_to_edge_index(traj["cells"][0]),
    )

    # Run the three rule mirrors. PH-CON-002/003 will SKIP per v9 dispatch.
    rule_results = {
        "PH-CON-001": mass_conservation_defect_on_mesh(rollout),
        "PH-CON-002": energy_drift_on_mesh(rollout),
        "PH-CON-003": dissipation_sign_violation_on_mesh(rollout),
    }

    # Emit SARIF (use the LB-side schema; adapt if needed). The
    # inference_run_status field per design §2.5 is "n/a (GT control arm,
    # no inference fired)" — the field documents that this rollout came
    # from ground truth, not a trained checkpoint.
    sarif = _build_sarif(
        rule_results=rule_results,
        run_metadata={
            "arm": "gt-control",
            "trajectory_index": traj_idx,
            "inference_run_status": "n/a_gt_control_arm",
            "git_sha": git_sha,
            "physicsnemo_sha": "1ca85d65ac2ce28ea9762910c09a954c08a37140",
        },
    )

    out_path_vol = (
        f"/vol/rollouts/physicsnemo/vortex_shedding_{git_sha}/gt.sarif"
    )
    os.makedirs(os.path.dirname(out_path_vol), exist_ok=True)
    with open(out_path_vol, "w") as f:
        json.dump(sarif, f, indent=2)
    mgn_volume.commit()
    return {"sarif_path": out_path_vol, "rule_summary": {k: str(v) for k, v in rule_results.items()}}
```

- [ ] **Step 3: Fire the entrypoint (CPU; ~3 min on Modal).**

```bash
modal run external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py::lint_gt_trajectory \
    --git-sha "$(git rev-parse --short HEAD)" \
    --traj-idx <FROM_TASK_1>
```

- [ ] **Step 4: Pull the SARIF locally + verify it parses.**

```bash
modal volume get case-study-02-physicsnemo-artifacts \
    /rollouts/physicsnemo/vortex_shedding_<sha>/gt.sarif \
    external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/gt.sarif

python -c "import json; d = json.load(open('external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/gt.sarif')); print(json.dumps([r['ruleId'] for r in d['runs'][0]['results']], indent=2))"
```

Expected output: rule IDs `["PH-CON-001", "PH-CON-002", "PH-CON-003"]`. PH-CON-002/003 should carry the substrate-class SKIP message (cites D0-22 + D0-23).

- [ ] **Step 5: Check PH-CON-001 against D0-24 verdict-band 1.**

The PH-CON-001 raw value should be ≤ 6% per the D0-24 v1 PASS band. If it lands in (6%, 10%]: MARGINAL — record per-trajectory variability. If > 10%: FAIL — halt Phase 2, audit harness FE implementation.

- [ ] **Step 6: Commit.**

```bash
git add external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py \
        external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/gt.sarif
git commit -m "02-physicsnemo-mgn: Task 5 — GT-trajectory lint entrypoint + gt.sarif (Phase 2 control arm; refinement 1's GT+MGN side-by-side)"
```

---

## Phase 2C — MGN inference + lint

### Task 6: Modal MGN inference entrypoint (A10G; one rollout; rollout-dir isolation per F5)

**Goal:** Run vortex-shedding MGN inference end-to-end on Modal A10G for the canonical test trajectory (selected in Task 1); write the rollout as `mesh_rollout.npz` per `_harness/SCHEMA.md` §2; reproduce the Phase 1 rollout-dir isolation pattern (Finding 5 absorption) with a smoke assertion (lands in Task 7) that two same-sha retries cannot read each other's CWD-relative stats. Pre-flight assertions cover all Phase 1 known-unknowns.

**Files:**
- Create: function `mgn_rollout_p0_vortex_shedding` in `02-physicsnemo-mgn/modal_app.py` (mirrors LB-side `lagrangebench_rollout_p1_*` shape)
- Create: `02-physicsnemo-mgn/preflight/mgn_rollout_p0_findings.json` (committed)

- [ ] **Step 1: Pre-flight assertions — list them at module top (Pattern-A discipline).**

In `modal_app.py`, near the inference entrypoint, add a docstring + helper:

```python
def _preflight_mgn_inference_p0(work_dir: str, git_sha: str) -> dict:
    """Pre-flight assertions before MGN inference fires. All must pass; any
    failure halts the rollout. Each asserts something the Phase 1 audit
    surfaced as a known-unknown or contract requirement.

    Phase 2 acceptance (design §4.2) requires:
    - persistent-volume write path (D0-23 v7 = Y; per Phase 1)
    - NGC checkpoint hash verification (against D0-23 pin)
    - rollout output schema (D0-23 v2 + v8: PyG Data tuple; velocity = first 2 cols of graph.y)
    - CWD discipline (KU §5.4 — chdir into work_dir before VortexSheddingDataset construction)
    - fp32 default-dtype (KU §5.6 — torch.set_default_dtype(torch.float32) BEFORE dataset)
    - split="test" (KU §5.3 — noise_std=0.02 is split-conditional)
    """
    import hashlib, os
    import torch

    findings = {}

    # 1. NGC checkpoint hash matches D0-23 v1 pin
    ckpt_path = "/vol/checkpoints/physicsnemo/modulus_ns_meshgraphnet_v0.1.pt"
    with open(ckpt_path, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    expected_sha = "<FROM_D0_23_v1>"  # paste the pin here at Task-5 execution time
    assert actual_sha == expected_sha, (
        f"NGC checkpoint hash drift: {actual_sha} != pinned {expected_sha}. "
        f"Halt Phase 2; investigate whether the Volume was clobbered."
    )
    findings["ckpt_sha256"] = actual_sha

    # 2. fp32 default dtype BEFORE dataset construction (KU §5.6)
    assert torch.get_default_dtype() == torch.float32, (
        f"torch.set_default_dtype(torch.float32) must precede dataset "
        f"construction; got {torch.get_default_dtype()}"
    )
    findings["torch_default_dtype"] = "float32"

    # 3. CWD discipline (KU §5.4): we are inside work_dir and stats files
    #    are present
    assert os.getcwd() == work_dir, f"expected cwd={work_dir}, got {os.getcwd()}"
    assert os.path.isfile(os.path.join(work_dir, "edge_stats.json"))
    assert os.path.isfile(os.path.join(work_dir, "node_stats.json"))
    findings["work_dir"] = work_dir

    # 4. Volume mount present + writable for rollout output
    out_dir = f"/vol/rollouts/physicsnemo/vortex_shedding_{git_sha}"
    os.makedirs(out_dir, exist_ok=True)
    test_path = os.path.join(out_dir, ".preflight_writable_check")
    with open(test_path, "w") as f:
        f.write("ok")
    os.remove(test_path)
    findings["rollout_output_dir"] = out_dir

    return findings
```

- [ ] **Step 2: Implement `mgn_rollout_p0_vortex_shedding` (A10G).**

Skeleton (filled with exact code at execution time; the LB-side `lagrangebench_rollout_p1_segnn_tgv2d` is the structural model):

```python
@app.function(image=image, volumes={"/vol": mgn_volume}, gpu="A10G", timeout=60 * 60 * 2)
def mgn_rollout_p0_vortex_shedding(git_sha: str, full_git_sha: str, traj_idx: int) -> dict:
    """Phase 2 P0 MGN inference: one rollout on the canonical test trajectory
    selected in Task 1. Reproduces inference.py's denorm→re-norm→model→denorm
    →mask→integrate protocol verbatim. Writes mesh_rollout.npz per SCHEMA.md §2.

    F5 absorption (Phase-1 cross-review Finding 5): rollout-dir isolation
    pattern — tempfile.mkdtemp() under /vol → stage stats → os.chdir → run
    inference → finally: chdir back. Smoke assertion: two same-sha retries
    cannot read each other's CWD-relative stats (verified by Task 7's
    retry test).
    """
    import os, json, tempfile, shutil
    import numpy as np, torch

    # F5: rollout-dir isolation — start with a fresh temp dir under /vol so
    # parallel retries don't collide on CWD-relative stats reads.
    rollout_dir = tempfile.mkdtemp(prefix=f"mgn_rollout_p0_{git_sha}_", dir="/vol/scratch")
    shutil.copy("/vol/datasets/cylinder_flow/edge_stats.json", rollout_dir)
    shutil.copy("/vol/datasets/cylinder_flow/node_stats.json", rollout_dir)
    old_cwd = os.getcwd()
    os.chdir(rollout_dir)

    try:
        torch.set_default_dtype(torch.float32)
        preflight = _preflight_mgn_inference_p0(rollout_dir, git_sha)

        # ... load NGC checkpoint via Phase 1's name-remap + edge-MLP
        #     column-reorder adapter; construct VortexSheddingDataset(
        #     split="test", num_samples=N_test, num_steps=600); pick out
        #     trajectory traj_idx; run inference.py's protocol verbatim
        #     for n_rollout_steps = 599 (= 600 - 1; standard MGN rollout
        #     horizon for cylinder_flow).
        rollout_velocity, rollout_pressure, mesh_pos, node_type, cells = _run_mgn_inference(
            traj_idx=traj_idx,
            n_rollout_steps=599,
        )

        # Build the MeshRollout and save as NPZ per SCHEMA.md §2.
        # `_assert_loader_contract_mgn` is wired into save→load round-trip via
        # Task 3's load-time check; Task 6 also calls it explicitly on the
        # in-memory rollout before save so failures surface at write time.
        from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
            MeshRollout,
            _assert_loader_contract_mgn,
            save_mesh_rollout_npz,
        )

        rollout = MeshRollout(
            node_positions=mesh_pos.astype(np.float32),
            node_type=node_type.astype(np.int64),
            node_values={
                "velocity": rollout_velocity.astype(np.float32),
                "pressure": rollout_pressure.astype(np.float32),
            },
            dt=0.01,
            metadata={
                "framework": "pytorch+dgl",  # honored even though v2.0.0 is PyG (KU §5.x)
                "model": "modulus_ns_meshgraphnet",
                "dataset": "vortex_shedding_2d",
                "regular_grid": False,
                "git_sha": git_sha,
                "ngc_version": "v0.1",
                "ckpt_hash": preflight["ckpt_sha256"],
                "physicsnemo_sha": "1ca85d65ac2ce28ea9762910c09a954c08a37140",
            },
            edge_index=_cells_to_edge_index(cells),
        )
        _assert_loader_contract_mgn(rollout)  # belt-and-braces fail-loud

        out_npz = (
            f"/vol/rollouts/physicsnemo/vortex_shedding_{git_sha}/mgn_rollout.npz"
        )
        save_mesh_rollout_npz(rollout, out_npz)
        findings = {
            "rollout_dir": rollout_dir,
            "out_npz": out_npz,
            "traj_idx": traj_idx,
            "n_rollout_steps": 599,
            "preflight": preflight,
        }
        out_findings = (
            f"/vol/rollouts/physicsnemo/vortex_shedding_{git_sha}/mgn_rollout_p0_findings.json"
        )
        with open(out_findings, "w") as f:
            json.dump(findings, f, indent=2)
        mgn_volume.commit()
        return findings
    finally:
        os.chdir(old_cwd)
        # F5: rollout_dir is NOT removed here (next-retry isolation evidence
        # for the smoke assertion in Task 7); periodic cleanup handled out-of-band.
```

- [ ] **Step 3: Fire on Modal A10G.**

```bash
modal run external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py::mgn_rollout_p0_vortex_shedding \
    --git-sha "$(git rev-parse --short HEAD)" \
    --full-git-sha "$(git rev-parse HEAD)" \
    --traj-idx <FROM_TASK_1>
```

Expected duration: 8-15 minutes on A10G (per Phase 1 substrate-smoke timings; 399 → 599 steps is ~1.5× longer).

Cap discipline (per [[feedback_cap_rationale_not_literal]]): one A10G fire for Task 6. If it fails, diagnose CPU-only before any re-fire; no fix-iterate-on-GPU pattern.

- [ ] **Step 4: Pull artifacts locally.**

```bash
modal volume get case-study-02-physicsnemo-artifacts \
    /rollouts/physicsnemo/vortex_shedding_<sha>/mgn_rollout_p0_findings.json \
    external_validation/_rollout_anchors/02-physicsnemo-mgn/preflight/mgn_rollout_p0_findings.json
```

(NPZ stays on Volume; Task 7 reads it from there for lint.)

- [ ] **Step 5: Commit.**

```bash
git add external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py \
        external_validation/_rollout_anchors/02-physicsnemo-mgn/preflight/mgn_rollout_p0_findings.json
git commit -m "02-physicsnemo-mgn: Task 6 — MGN rollout p0 on canonical test traj (A10G; rollout-dir isolation per F5)"
```

---

### Task 7: MGN-rollout lint entrypoint + mgn.sarif (CPU)

**Goal:** Lint the saved MGN rollout NPZ through the same three rule mirrors as Task 5's GT lint. Emits `mgn.sarif`. F5 smoke assertion (two same-sha retries cannot read each other's CWD-relative stats) lands here as a separate test.

**Files:**
- Modify: `02-physicsnemo-mgn/modal_app.py` (new entrypoint `lint_mgn_rollout`)
- Create: `02-physicsnemo-mgn/outputs/sarif/mgn.sarif` (committed)
- Create: `02-physicsnemo-mgn/tests/test_rollout_dir_isolation.py` (CPU smoke test)

- [ ] **Step 1: Implement `lint_mgn_rollout` (CPU).**

```python
@app.function(image=image, volumes={"/vol": mgn_volume}, timeout=60 * 30)
def lint_mgn_rollout(git_sha: str) -> dict:
    """Phase 2 MGN-arm lint: PH-CON-001 + PH-CON-002 + PH-CON-003 on the
    saved mesh_rollout.npz from Task 6. Mirrors Task 5's GT lint structure;
    same harness, same SARIF schema.

    inference_run_status = "from_completed_inference" — the rollout came from
    a successful Task 6 fire (no salvage).
    """
    import json, os
    from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
        dissipation_sign_violation_on_mesh,
        energy_drift_on_mesh,
        load_mesh_rollout_npz,
        mass_conservation_defect_on_mesh,
    )

    npz_path = (
        f"/vol/rollouts/physicsnemo/vortex_shedding_{git_sha}/mgn_rollout.npz"
    )
    rollout = load_mesh_rollout_npz(npz_path)  # F3-wired loader-contract enforced

    rule_results = {
        "PH-CON-001": mass_conservation_defect_on_mesh(rollout),
        "PH-CON-002": energy_drift_on_mesh(rollout),
        "PH-CON-003": dissipation_sign_violation_on_mesh(rollout),
    }

    sarif = _build_sarif(
        rule_results=rule_results,
        run_metadata={
            "arm": "mgn-rollout",
            "trajectory_index": rollout.metadata.get("traj_idx"),
            "inference_run_status": "from_completed_inference",  # design §2.5
            "git_sha": git_sha,
            "ckpt_hash": rollout.metadata["ckpt_hash"],
            "physicsnemo_sha": rollout.metadata["physicsnemo_sha"],
        },
    )
    out_path = (
        f"/vol/rollouts/physicsnemo/vortex_shedding_{git_sha}/mgn.sarif"
    )
    with open(out_path, "w") as f:
        json.dump(sarif, f, indent=2)
    mgn_volume.commit()
    return {"sarif_path": out_path, "rule_summary": {k: str(v) for k, v in rule_results.items()}}
```

- [ ] **Step 2: Fire the entrypoint (CPU; ~3 min).**

```bash
modal run external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py::lint_mgn_rollout \
    --git-sha "$(git rev-parse --short HEAD)"
```

- [ ] **Step 3: Pull the SARIF locally + verify it parses.**

```bash
modal volume get case-study-02-physicsnemo-artifacts \
    /rollouts/physicsnemo/vortex_shedding_<sha>/mgn.sarif \
    external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/mgn.sarif

python -c "import json; d = json.load(open('external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/mgn.sarif')); print(json.dumps([(r['ruleId'], r.get('message', {}).get('text', '')[:80]) for r in d['runs'][0]['results']], indent=2))"
```

Expected: PH-CON-001 raw value within 20% of GT's (D0-24 v2 PASS band); PH-CON-002/003 SKIP messages citing D0-22 + D0-23.

- [ ] **Step 4: Write the F5 rollout-dir-isolation smoke assertion (CPU pytest).**

```python
# tests/test_rollout_dir_isolation.py
"""Phase-1 cross-review Finding 5 absorption: verify that two same-sha
retries of the MGN rollout entrypoint cannot read each other's CWD-relative
stats. This is a CPU smoke test against the Modal-image filesystem isolation
contract; it does not fire the GPU rollout.
"""

import os
import tempfile

import pytest


def test_two_same_sha_retries_cannot_share_cwd_relative_stats(tmp_path):
    """Two tempfile.mkdtemp() calls with same prefix yield distinct paths;
    a process chdir'd into one cannot see the other's stats files via
    relative path. Models the round-codex-4 retry-isolation invariant.
    """
    # Simulate two retries of the entrypoint with the same git_sha prefix.
    dir_a = tempfile.mkdtemp(prefix="mgn_rollout_p0_abc123_", dir=str(tmp_path))
    dir_b = tempfile.mkdtemp(prefix="mgn_rollout_p0_abc123_", dir=str(tmp_path))
    assert dir_a != dir_b, "tempfile.mkdtemp must produce distinct paths even with same prefix"

    # Stage stats in dir_a only
    with open(os.path.join(dir_a, "edge_stats.json"), "w") as f:
        f.write('{"edge_mean": [0, 0, 0]}')

    # chdir to dir_b — must NOT see edge_stats.json via relative path
    old = os.getcwd()
    os.chdir(dir_b)
    try:
        assert not os.path.isfile("edge_stats.json"), (
            "F5 violation: dir_b can read dir_a's CWD-relative stats; "
            "rollout-dir isolation broken."
        )
    finally:
        os.chdir(old)
```

- [ ] **Step 5: Run the smoke test.**

```bash
pytest external_validation/_rollout_anchors/02-physicsnemo-mgn/tests/test_rollout_dir_isolation.py -v
```

Expected: 1 passed.

- [ ] **Step 6: Commit.**

```bash
git add external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py \
        external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/mgn.sarif \
        external_validation/_rollout_anchors/02-physicsnemo-mgn/tests/test_rollout_dir_isolation.py
git commit -m "02-physicsnemo-mgn: Task 7 — MGN-rollout lint + mgn.sarif + F5 rollout-dir-isolation smoke test"
```

---

### Task 8: `inference_run_status` field landing (design §2.5)

**Goal:** Confirm both SARIFs (Task 5 gt.sarif + Task 7 mgn.sarif) carry the `inference_run_status` field per design §2.5. Uniformly `from_completed_inference` predicted for MGN; `n/a_gt_control_arm` for GT. If salvage triggers fire, forward-flag instead of patching the code.

**Files:**
- Modify: SARIF emission helper (location decided in Task 5 step 1; either LB-shared or MGN-local)
- Test: `02-physicsnemo-mgn/tests/test_sarif_inference_run_status.py`

- [ ] **Step 1: Read both SARIFs locally; confirm the field is present.**

```bash
python <<'EOF'
import json
for arm in ("gt", "mgn"):
    sarif = json.load(open(f"external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/{arm}.sarif"))
    props = sarif["runs"][0]["properties"]
    assert "inference_run_status" in props, f"{arm}.sarif missing inference_run_status"
    print(f"{arm}.sarif inference_run_status = {props['inference_run_status']!r}")
EOF
```

Expected:
```
gt.sarif inference_run_status = 'n/a_gt_control_arm'
mgn.sarif inference_run_status = 'from_completed_inference'
```

- [ ] **Step 2: Add a regression test for the field.**

```python
# tests/test_sarif_inference_run_status.py
"""Phase 2 design §2.5: SARIF emissions carry inference_run_status.
Uniformly from_completed_inference for inferred rollouts; n/a_gt_control_arm
for the GT control. Salvage triggers fire a forward-flag instead of a code
patch."""

import json
import pathlib

import pytest

OUTPUTS = pathlib.Path("external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif")


@pytest.mark.parametrize("arm,expected", [
    ("gt", "n/a_gt_control_arm"),
    ("mgn", "from_completed_inference"),
])
def test_sarif_inference_run_status_present_and_pinned(arm, expected):
    sarif = json.loads((OUTPUTS / f"{arm}.sarif").read_text())
    props = sarif["runs"][0]["properties"]
    assert props.get("inference_run_status") == expected
```

- [ ] **Step 3: Run the test.**

```bash
pytest external_validation/_rollout_anchors/02-physicsnemo-mgn/tests/test_sarif_inference_run_status.py -v
```

Expected: 2 passed.

- [ ] **Step 4: Commit.**

```bash
git add external_validation/_rollout_anchors/02-physicsnemo-mgn/tests/test_sarif_inference_run_status.py
git commit -m "02-physicsnemo-mgn: Task 8 — inference_run_status field regression test (design §2.5)"
```

---

## Phase 2D — Smoke + D-entry

### Task 9: Phase 2 smoke verdict + D0-24 verdicts 1-7 pinned

**Goal:** Compare rule outputs from Tasks 5 + 7 against D0-24's pre-registered bands; record each verdict; commit. Pattern-A drift handling: if any verdict lands in a MARGINAL or FAIL band, absorb via D-entry amendment (cell-2) or pause for user discussion (cell-4 / methodology pivot).

**Files:**
- Modify: `external_validation/_rollout_anchors/methodology/DECISIONS.md` (D0-24 verdicts 1-7 filled)

- [ ] **Step 1: Compute the verdict for each D0-24 band.**

```bash
python <<'EOF'
import json
gt = json.load(open("external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/gt.sarif"))
mgn = json.load(open("external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/mgn.sarif"))

def rule_value(sarif, rule_id):
    for r in sarif["runs"][0]["results"]:
        if r["ruleId"] == rule_id:
            return r["properties"].get("raw_value"), r.get("message", {}).get("text", "")
    return None, "<missing>"

for rule in ("PH-CON-001", "PH-CON-002", "PH-CON-003"):
    gv, gt_msg = rule_value(gt, rule)
    mv, mgn_msg = rule_value(mgn, rule)
    print(f"{rule}: GT={gv!r}, MGN={mv!r}")
    if rule == "PH-CON-001":
        # D0-24 v1: GT ≤ 6% PASS, (6, 10]% MARGINAL, >10% FAIL
        # D0-24 v2: MGN within ±20% of GT PASS, 20-50% above MARGINAL, >50% FAIL
        ...
EOF
```

- [ ] **Step 2: Fill D0-24 verdicts 1-7 with the computed values + verdict labels.**

For each of D0-24's 7 verdict bands (pre-registered in Task 2), record the numeric outcome + the verdict label (PASS / MARGINAL / FAIL / SKIP-as-designed).

- [ ] **Step 3: Pattern-A drift handling.**

If any verdict is MARGINAL or FAIL:

| Verdict shape | Action |
|---|---|
| MARGINAL on PH-CON-001 (GT or MGN) | Record per-trajectory variability in writeup framing; no methodology amendment. |
| FAIL on PH-CON-001 GT (> 10%) | Halt — harness-floor instability is a separate investigation; not Phase 2 scope. |
| FAIL on PH-CON-001 MGN (> 50% above GT) | Genuine MGN mass-conservation signal — IS the headline finding for case study 02; record without amendment. |
| FAIL on loader-contract enforcement (v5) | Bug in Task 3's wiring — fix, re-run Task 3-6 from the failure point. |
| FAIL on rollout-dir isolation (v6) | Bug in Task 6's tempfile pattern — fix + re-fire Task 6 (separate fix-iteration A10G fire, NOT a verdict-confirmation re-fire). |
| Cell-4 surprise (anything else) | Pause; surface to user. |

- [ ] **Step 4: Commit.**

```bash
git add external_validation/_rollout_anchors/methodology/DECISIONS.md
git commit -m "DECISIONS.md: D0-24 verdicts 1-7 pinned — Phase 2 smoke results vs pre-registered bands"
```

---

## Phase 2E — Cross-review boundary

### Task 10: Acceptance-criteria checkpoint + Phase 3 writeup-framing forward-flag

**Goal:** Verify all Phase 2 design §4.2 acceptance checkboxes hold; record the refinement-2 scope qualifier (single-trajectory, coverage-not-statistics) as a Phase-3 writeup requirement; flag the v2.1.2 §1.4 "Prose scope-qualifiers" methodology entry as a separate amendment (out of Phase 2 scope).

**Files:**
- Modify: `external_validation/_rollout_anchors/methodology/DECISIONS.md` (acceptance checklist + forward-flag block)

- [ ] **Step 1: Walk through design §4.2 box-by-box.**

For each checkbox in §4.2, paste the commit SHA + one-line evidence into a checklist under D0-24.

- [ ] **Step 2: Add the Phase-3 writeup-framing requirement block.**

Append to D0-24:

```markdown
**Phase 3 writeup-framing requirements (refinement 2 forward-flag):**

The Phase 3 plan must include:

1. **Scope qualifier paragraph** (verbatim, in case-study-02 README's results
   section): "Phase 2 results report PH-CON-001 defect on a single
   cylinder_flow test trajectory (trajectory M=<traj_idx>, Strouhal S=<St>,
   Reynolds R=<Re>), selected via the Phase 2 pre-fire Strouhal audit to be
   representative of the test-set distribution. Coverage-not-statistics
   framing: physics-lint's value here is rule-firing on a real-world
   checkpoint, not a distribution over initial conditions. CI-gate threshold
   derivation from defect-magnitude distributions would require N>1 and is
   deferred (Phase 2 does NOT claim CI-gate calibration)."

2. **Floor-bounds-resolution distinction** in §3.3 "what physics-lint did
   NOT catch": name the harness-FE-on-P1 floor (~5%) explicitly as bounding
   PH-CON-001's discriminating resolution. The rule's verdict on GT+MGN at
   the floor demonstrates "MGN reproduces GT at the floor," NOT "MGN is
   physically incompressible to 5%." At this discretization floor,
   PH-CON-001 bounds MGN's deviation from GT-equivalence rather than from
   physical incompressibility. A tighter discretization would distinguish
   whether MGN's 5% reflects model error or floor error.

3. **Methodology forward-flag (out of Phase 2 + 3 scope):** v2.1.2 §1.4
   "Prose scope-qualifiers — claim precision about what evidence demonstrates
   vs adjacent claims it could be over-read to support." Fourth empirical
   instance (after round-code-1's three walls); the cumulative pattern is
   strong enough to formalize as a methodology entry. Tracked separately,
   does NOT block Phase 2/3 completion.
```

- [ ] **Step 3: Commit.**

```bash
git add external_validation/_rollout_anchors/methodology/DECISIONS.md
git commit -m "DECISIONS.md: D0-24 Phase 2 acceptance checklist + refinement-2 forward-flag for Phase 3 writeup framing"
```

---

### Task 11: Phase 2 boundary cross-review (Codex pass)

**Goal:** Dispatch a Codex/GPT-5-CLI adversarial review of Phase 2's verdicts + code + SARIFs. Same shape as the Phase 1 cross-review (`c534307`). Findings triaged in Task 12.

**Files:**
- Create: `external_validation/_rollout_anchors/methodology/docs/2026-05-XX-case-study-02-phase-2-cross-review.md` (date-stamped at execution time)

- [ ] **Step 1: Compose the cross-review prompt.**

Use the Phase 1 cross-review prompt (commit c534307's `Agent` invocation) as a template. Scope:

- D0-24 verdicts 1-7 (Phase 2 audit + smoke).
- Code-absorption commits: Tasks 3 (`_assert_loader_contract_mgn` wiring), 4 (MeshField graph-mesh path lift), 5 (GT lint), 6 (MGN inference + F5 isolation), 7 (MGN lint + smoke test), 8 (inference_run_status).
- The two Phase 1 forward-flags it inherited (F3 + F5): did Phase 2 close them cleanly?
- Layered fail-open lens: did Phase 2 introduce any NEW fail-open shapes (e.g., GT-vs-MGN metadata divergence; SARIF schema drift; MGN-scope detection by `model.startswith("modulus_")` — what about future `nvidia-physicsnemo_` rename?)

- [ ] **Step 2: Dispatch the Codex review.**

Via `codex:codex-rescue` subagent OR `codex exec` directly. Capture findings to `docs/2026-05-XX-case-study-02-phase-2-cross-review.md`.

- [ ] **Step 3: Commit the cross-review findings doc.**

```bash
git add external_validation/_rollout_anchors/methodology/docs/2026-05-XX-case-study-02-phase-2-cross-review.md
git commit -m "methodology: case study 02 Phase 2 boundary cross-review findings (round-codex-Phase2)"
```

---

### Task 12: Triage Phase 2 cross-review findings

**Goal:** Pattern-C four-cell triage of each Phase 2 cross-review finding. Cell-2 absorptions land as TDD commits per the Phase 1 Task 15 pattern. Update D0-24 with the cross-review summary table.

**Files:**
- Modify: `external_validation/_rollout_anchors/methodology/DECISIONS.md` (D0-24 cross-review summary)
- Possibly modify: code files per cell-2 absorptions

- [ ] **Step 1: For each finding, triage into the four cells.**

Per Phase 1 plan §1968-1974 (same four-cell schema):
- Cell 1 (re-discovery under prior scope): defer to prior decision; cite the discipline-marker.
- Cell 2 (novel-in-scope): in-rung absorption. Land follow-up commits.
- Cell 3 (novel-out-of-scope): forward-flag to Phase 3 or amendment 1.
- Cell 4 (genuinely new framing): re-examine prior decision with new information. Earn the cell-4 bar.

- [ ] **Step 2: Land cell-2 absorption commits.**

Each cell-2 finding → TDD red-green:
1. Failing test capturing the finding's scenario.
2. Implement the fix.
3. Verify test passes.
4. Commit with reference to the cross-review finding.

- [ ] **Step 3: Record cell distribution in D0-24.**

Append to D0-24 (same template as D0-23's triage table at commit 41232d2):

```markdown
**Phase 2 boundary cross-review (Tasks 11-12) — findings triaged:**

| # | Finding (1-line) | Severity | Cell | Disposition |
|---|---|---|---|---|
| 1 | ... | ... | ... | Absorbed at <sha> / Forward to Phase 3 / Defer (re-discovery) |
| ... | ... | ... | ... | ... |

**Totals:** N cell-1, M cell-2, P cell-3, Q cell-4.
```

- [ ] **Step 4: Commit the triage summary.**

```bash
git add external_validation/_rollout_anchors/methodology/DECISIONS.md
git commit -m "DECISIONS.md: D0-24 Phase 2 cross-review triage summary; Phase 2 COMPLETE"
```

- [ ] **Step 5: Push the branch.**

```bash
git push
```

(Branch is already tracking origin from Phase 1's Task 15 step 5.)

- [ ] **Step 6: Update the design doc §7 successor block.**

In `methodology/docs/2026-05-11-case-study-02-physicsnemo-mgn-design.md` §7:
- Change "Successor (session 3): Phase 2 + Phase 3 execution per §3.2 / §3.3 + §4.2 / §4.3, off a fresh writing-plans round." → "Phase 2 COMPLETE at sha <sha>; see D0-24 for verdicts + cross-review triage. Phase 3 (writeup + cross-stack table) gets a fresh writing-plans round next."

Commit:

```bash
git add external_validation/_rollout_anchors/methodology/docs/2026-05-11-case-study-02-physicsnemo-mgn-design.md
git commit -m "design doc: mark Phase 2 complete; D0-24 verdicts + cross-review pinned"
git push
```

---

## Phase 2 acceptance criteria check (per design §4.2)

Verify each design §4.2 checkbox before declaring Phase 2 done:

- [ ] Modal MGN inference entrypoint committed; pre-flight assertions cover persistent-volume write path, NGC checkpoint hash verification, rollout output schema, CWD discipline, fp32 default-dtype, split="test" (Task 6).
- [ ] Round-codex-4 rollout-dir isolation applied (D0-23 v7 = Y; F5 absorption at Task 6 + Task 7 smoke test).
- [ ] P0 vortex-shedding rollout completed end-to-end on Modal (N=1 per refinement 1; rung-4c discipline — ship at empirically-feasible N).
- [ ] Per-timestep MeshField materialization works (Gate A PASS branch from D0-23 v3; Task 4 wires the path into the harness rule mirrors; Tasks 5 + 7 exercise it).
- [ ] PH-CON-001 SARIF committed at `02-physicsnemo-mgn/outputs/sarif/` (gt.sarif + mgn.sarif via Tasks 5 + 7); values within Phase-1-pre-registered mass-conservation drift bound (D0-24 v1 + v2 PASS / MARGINAL / FAIL routing).
- [ ] PH-CON-002 SARIF committed; behavior consistent with v9 dispatch (SKIP-with-reason or RAW per dispatch).
- [ ] PH-CON-003 SARIF committed; SKIP/RAW outcome per dispatch.
- [ ] `inference_run_status` field lands per §2.5 (Task 8).
- [ ] Smoke check: rule outputs vs Phase-1-pre-registered tolerances; pattern-A drift absorbed via D-entry amendment if any (Task 9).
- [ ] Phase 2 boundary cross-review complete; findings triaged (Tasks 11-12).

Plus the two Phase 1 forward-flags Phase 2 inherited:

- [ ] F3: `_assert_loader_contract_mgn` wired into load_mesh_rollout_npz (Task 3); 4 rejection tests pass on the wired path.
- [ ] F5: rollout-dir isolation pattern reproduced in Task 6's entrypoint; smoke assertion at `tests/test_rollout_dir_isolation.py` passes (Task 7 step 4).

If any unchecked → fix before opening Phase 3's writing-plans round.

---

## Successor: Phase 3 writing-plans

Phase 3 (writeup + cross-stack table + v2.1.1 amendment) opens a fresh writing-plans round AFTER Phase 2 completes, using:
- This plan's final state (D0-24 resolved).
- The design doc §3.3 + §4.3.
- Phase 2's cross-review findings.
- The refinement-2 forward-flag block under D0-24 (scope qualifier paragraph + floor-bounds-resolution framing + v2.1.2 §1.4 methodology entry).

Do NOT extend this plan to cover Phase 3 — the per-phase plan boundary is intentional per design §7 (audit verdicts feed forward; plans are per-phase so verdicts inform the next plan's shape).
