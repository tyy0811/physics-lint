# Case Study 02 Phase 2 Cross-Review

Review target: CS02 PhysicsNeMo MeshGraphNet Phase 2 commits `f22319d..a15b144`, with emphasis on the two Phase 1 forward-flags and the round-codex-4 layered fail-open lens.

### Finding 1: Phase 2 SARIF advertises schema v1.0 but fails the v1.0 run-level contract

**Severity:** High
**Surface:** `external_validation/_rollout_anchors/_harness/SCHEMA.md:257-273`; `external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py:38-50`, `:102-118`; `external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py:2389-2402`, `:2792-2809`; `external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/gt.sarif:5-17`; `external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/mgn.sarif:5-21`.
**Evidence:** The committed GT and MGN SARIFs both set `harness_sarif_schema_version = "1.0"` (`gt.sarif:10`, `mgn.sarif:11`). The schema says all 10 D0-19 run-level fields are required, including `physics_lint_sha_pkl_inference`, `physics_lint_sha_npz_conversion`, `lagrangebench_sha`, and `rollout_subdir` (`SCHEMA.md:257-273`), and the renderer enforces exactly that list (`render_cross_stack_table.py:38-50`, `:114-118`). Reproducer:

```text
$ python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py --sarif-dir external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif --include-glob '*.sarif'
MissingRunLevelFieldError: external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/gt.sarif: missing required D0-19 run-level fields: ['physics_lint_sha_pkl_inference', 'physics_lint_sha_npz_conversion', 'lagrangebench_sha'].
```

A direct key audit also showed `gt.sarif` missing 3 required v1.0 fields and `mgn.sarif` missing 4, including `rollout_subdir`. This is an observed contract break, not only a future risk: a v1.0 consumer fails loud on the Phase 2 artifacts. The divergence came from the two Phase 2 emitters constructing new CS02-specific `run_properties` dictionaries while still labeling them schema v1.0 (`modal_app.py:2389-2402`, `:2792-2809`).

```text
gt.sarif missing_schema_v1_0_fields= ['lagrangebench_sha', 'physics_lint_sha_npz_conversion', 'physics_lint_sha_pkl_inference']
mgn.sarif missing_schema_v1_0_fields= ['lagrangebench_sha', 'physics_lint_sha_npz_conversion', 'physics_lint_sha_pkl_inference', 'rollout_subdir']
```
**Recommended fix:** Either emit the full D0-19 v1.0 run-level field set with explicit CS02 sentinel values where a LagrangeBench stage is inapplicable, or bump the harness SARIF schema version and teach the renderer a CS02/mesh provenance schema before writing `harness_sarif_schema_version = "1.0"`. Add a regression test that runs the renderer or a schema validator over `gt.sarif` and `mgn.sarif`.
**Pattern-C cell:** 2 (novel-in-scope)

### Finding 2: `physics_lint_sha_sarif_emission` is overloaded as the rollout directory key

**Severity:** Medium
**Surface:** `external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py:2389-2402`, `:2769-2809`; `external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/gt.sarif:13-15`; `external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/mgn.sarif:16-19`; commits `4173b32` and `11c7df2`.
**Evidence:** Task 5 introduced `gt.sarif` at commit `4173b32`, but the artifact records `"physics_lint_sha_sarif_emission": "4debbbf"` and a `vortex_shedding_4debbbf` output directory (`gt.sarif:13-15`). Task 7 introduced `mgn.sarif` at commit `11c7df2`, but the artifact records both `"physics_lint_sha_inference": "4173b32"` and `"physics_lint_sha_sarif_emission": "4173b32"` (`mgn.sarif:16-17`). Reproducer:

```text
$ git log --oneline --follow -- external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/gt.sarif
4173b32 02-physicsnemo-mgn: Task 5 -- GT-trajectory lint entrypoint + gt.sarif

$ git log --oneline --follow -- external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/mgn.sarif
11c7df2 02-physicsnemo-mgn: Task 7 -- MGN-rollout lint + mgn.sarif + F5 rollout-dir-isolation smoke test
4173b32 02-physicsnemo-mgn: Task 5 -- GT-trajectory lint entrypoint + gt.sarif
```

The code explains the mismatch: both emitters accept a single `git_sha` parameter and use it as both artifact path key and `physics_lint_sha_sarif_emission` (`modal_app.py:2400-2401`, `:2769`, `:2805-2807`). For MGN lint, that `git_sha` must identify the Task 6 rollout directory (`vortex_shedding_4173b32`), so it cannot also prove the Task 7 SARIF-emission code revision. Observed defect: SARIF provenance cannot distinguish rollout genesis from SARIF-emission code. Hypothetical risk: a later re-emission can silently preserve the old rollout key while falsely claiming the old code emitted the new SARIF.
**Recommended fix:** Split the parameter surface: `rollout_key_sha` or `rollout_id` for `/vol/rollouts/...`, `physics_lint_sha_inference` for Task 6, and `physics_lint_sha_sarif_emission` from `git rev-parse --short HEAD` inside the emitting code or an explicit second argument. Re-emit both SARIFs after the schema decision in Finding 1.
**Pattern-C cell:** 2 (novel-in-scope)

### Finding 3: F5 closes CWD-relative stats isolation but leaves same-sha persistent outputs collidable

**Severity:** Medium
**Surface:** `external_validation/_rollout_anchors/02-physicsnemo-mgn/modal_app.py:2484-2489`, `:2564-2574`, `:2694-2712`, `:2769-2775`; `external_validation/_rollout_anchors/02-physicsnemo-mgn/tests/test_rollout_dir_isolation.py:28-91`; `external_validation/_rollout_anchors/methodology/DECISIONS.md:2490-2492`, `:2515`, `:2525-2526`, `:2540-2541`.
**Evidence:** Task 6 does use a unique temporary CWD (`tempfile.mkdtemp(prefix=f"mgn_rollout_p0_{git_sha}_")`) and stages `{edge,node}_stats.json` there before `chdir` (`modal_app.py:2564-2574`). That closes the CWD-relative stats-read part of F5. But the persistent output layer is still fixed only by `git_sha`: preflight writes under `/vol/rollouts/physicsnemo/vortex_shedding_{git_sha}` (`modal_app.py:2484-2489`), the rollout writes `mgn_rollout.npz` and `mgn_rollout_p0_findings.json` in that same fixed directory (`modal_app.py:2694-2712`), and Task 7 reads `/vol/rollouts/physicsnemo/vortex_shedding_{git_sha}/mgn_rollout.npz` (`modal_app.py:2769-2775`). Reproducer:

```text
$ python - <<'PY'
for git_sha in ['4173b32','4173b32']:
    out_dir=f'/vol/rollouts/physicsnemo/vortex_shedding_{git_sha}'
    print(out_dir + '/mgn_rollout.npz')
    print(out_dir + '/mgn_rollout_p0_findings.json')
PY
/vol/rollouts/physicsnemo/vortex_shedding_4173b32/mgn_rollout.npz
/vol/rollouts/physicsnemo/vortex_shedding_4173b32/mgn_rollout_p0_findings.json
/vol/rollouts/physicsnemo/vortex_shedding_4173b32/mgn_rollout.npz
/vol/rollouts/physicsnemo/vortex_shedding_4173b32/mgn_rollout_p0_findings.json
```

The smoke tests verify `tempfile.mkdtemp` uniqueness and bare-filename isolation (`test_rollout_dir_isolation.py:28-91`), not production-path persistent-volume collision. D0-24 nevertheless marks F5 closed (`DECISIONS.md:2515`, `:2540-2541`) and says pre-flight assertions cover the persistent-volume write path (`DECISIONS.md:2525`). Observed defect: same-sha retries have isolated stats CWDs but share output filenames; Task 7 has no run-id or manifest check to know which retry produced the NPZ. Hypothetical risk: concurrent or retried same-sha fires can overwrite or mix `mgn_rollout.npz`, findings, and later `mgn.sarif`.
**Recommended fix:** Keep the temporary CWD pattern, but also write each Task 6 fire under a unique persistent run id, e.g. `/vol/rollouts/physicsnemo/vortex_shedding_{git_sha}/{run_id}/mgn_rollout.npz`, and pass that exact run id into `lint_mgn_rollout`. Add a smoke test for the path constructor or a manifest check proving two same-sha runs produce distinct persistent output paths.
**Pattern-C cell:** 1 (re-discovery)

### Finding 4: MGN loader-contract scope is keyed to a brittle `modulus_` model-name prefix

**Severity:** Medium
**Surface:** `external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py:315-326`; `external_validation/_rollout_anchors/_harness/tests/test_mesh_rollout_adapter.py:335-339`, `:343-417`, `:432-456`; `external_validation/_rollout_anchors/methodology/DECISIONS.md:2338-2346`, `:2486-2488`, `:2513`, `:2540`.
**Evidence:** Task 3 wires `_assert_loader_contract_mgn` into `load_mesh_rollout_npz`, but the gate is `model_name.startswith("modulus_")` (`mesh_rollout_adapter.py:315-326`). The test comments make the scope explicit: `"modulus_"` means MGN contract, while "future stacks" bypass (`test_mesh_rollout_adapter.py:335-339`). The four fail-loud tests all use `model = "modulus_ns_meshgraphnet"` (`test_mesh_rollout_adapter.py:343-417`), while the bypass test intentionally accepts any non-`modulus_` model (`test_mesh_rollout_adapter.py:432-456`). Reproducer for the residual fail-open:

```text
$ python - <<'PY'
from pathlib import Path
import tempfile, numpy as np
from external_validation._rollout_anchors._harness.mesh_rollout_adapter import load_mesh_rollout_npz
with tempfile.TemporaryDirectory() as d:
    p=Path(d)/'future_renamed_mgn_bad.npz'
    np.savez(p, node_positions=np.zeros((2,2), dtype=np.float32), node_type=np.array([0,4], dtype=np.int32), node_values=np.array({'u': np.zeros((2,2,2), dtype=np.float64)}, dtype=object), dt=np.float64(0.01), metadata=np.array({'framework':'pytorch+dgl','model':'nvidia_physicsnemo_ns_meshgraphnet','dataset':'vortex_shedding_2d','regular_grid': False}, dtype=object))
    r=load_mesh_rollout_npz(p)
    print('loaded_without_assertion', sorted(r.node_values), r.node_values['u'].dtype, r.metadata['model'])
PY
loaded_without_assertion ['u'] float64 nvidia_physicsnemo_ns_meshgraphnet
```

Observed defect in the current code: an otherwise MGN-shaped cylinder-flow rollout with a PhysicsNeMo-style model name bypasses all F3 assertions. Current Phase 2 artifacts use `modulus_ns_meshgraphnet`, so the shipped P0 NPZ is covered; the residual risk is specifically the next vendor-prefix or artifact-name change. That risk is concrete in this case study because D0-23 already records Modulus-to-PhysicsNeMo mechanism drift and the old NGC stack deprecation (`DECISIONS.md:2338-2346`).
**Recommended fix:** Replace prefix inference with an explicit metadata contract such as `metadata["rollout_contract"] == "physicsnemo_mgn_vortex_shedding_p0"` or a small allowlisted tuple over `(case_study, dataset, framework, model_family)`. Keep the synthetic bypass, but make future MGN-family names opt in through a stable family field rather than the legacy artifact prefix.
**Pattern-C cell:** 2 (novel-in-scope)

### Finding 5: The Phase 3 scope qualifier overstates canonical-trajectory representativeness

**Severity:** Medium
**Surface:** `external_validation/_rollout_anchors/02-physicsnemo-mgn/preflight/strouhal_test_trajectories.json:983-1002`, `:2215-2239`; `external_validation/_rollout_anchors/methodology/DECISIONS.md:2464-2466`, `:2547-2553`.
**Evidence:** The data file and D0-24 agree that the canonical trajectory is `traj_idx=44` with `St_U_max=0.192` and that selection was "median-Strouhal in-band sorted by `strouhal_U_max`" among 23 in-band trajectories (`strouhal_test_trajectories.json:2215-2239`; `DECISIONS.md:2464-2466`). The Phase 3 required prose then says that this trajectory was selected "to be representative of the test-set distribution on the centerline convention" (`DECISIONS.md:2547`). Reproducer:

```text
$ python - <<'PY'
import json
from pathlib import Path
rows=json.loads(Path('external_validation/_rollout_anchors/02-physicsnemo-mgn/preflight/strouhal_test_trajectories.json').read_text())['per_trajectory']
vals=sorted((r['strouhal_U_max'], r['traj_idx']) for r in rows if 'strouhal_U_max' in r)
canon=[x for x in vals if x[1] == 44][0]
inb={r['traj_idx'] for r in rows if r.get('in_design_band')}
print('full median lower/upper', vals[49], vals[50])
print('canonical', canon)
print('rank of canonical among all', vals.index(canon)+1, 'of', len(vals))
print('in_band rank range', min(i+1 for i,x in enumerate(vals) if x[1] in inb), max(i+1 for i,x in enumerate(vals) if x[1] in inb))
PY
full median lower/upper (0.02777178319363324, 66) (0.029088230059905858, 18)
canonical (0.19204371821530614, 44)
rank of canonical among all 80 of 100
in_band rank range 69 91
```

Observed defect: `traj_idx=44` is representative of the in-band subset by the pre-registered rule, but it is not representative of the full 100-trajectory `St_U_max` distribution; it ranks 80th of 100, and the in-band subset itself occupies ranks 69-91. The deterministic-selection claim is supported: the JSON has 23 in-band rows, the canonical value has no tie in the reproducer, and the data records the exact selected trajectory (`strouhal_test_trajectories.json:983-1002`, `:2215-2237`). The unsupported part is the broader "representative of the test-set distribution" wording. The floor-bounds-resolution paragraph is otherwise well-supported by the GT/MGN values and already avoids the "physically incompressible to 5%" over-read (`DECISIONS.md:2507`, `:2549`).
**Recommended fix:** In the Phase 3 forward-flag prose, replace "representative of the test-set distribution on the centerline convention" with "representative of the in-band subset under the pre-registered centerline-convention selection rule." Keep the 23/100 substrate-variability paragraph adjacent so readers do not infer distributional coverage or CI-gate calibration from N=1.
**Pattern-C cell:** 4 (novel-framing)
