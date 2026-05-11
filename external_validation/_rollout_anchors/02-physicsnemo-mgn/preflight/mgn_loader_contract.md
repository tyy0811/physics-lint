# MGN Loader Contract — Pre-flight Source Review for Case Study 02

**Date:** 2026-05-11
**Author task:** Apply the four-step source-review-before-compute pattern (case study 01 rung-4b §5.1) to NVIDIA PhysicsNeMo MeshGraphNet before any GPU fire.
**Status:** Pre-compute. No GPU work performed; no code modified. Research document only.

---

## 1. Pinned PhysicsNeMo sha + rationale

**Pinned sha:** `1ca85d65ac2ce28ea9762910c09a954c08a37140`
**Tag:** `v2.0.0` (most recent release; published 2026-03-10T23:47:04Z)
**Resolution method:** `gh api repos/NVIDIA/physicsnemo/git/refs/tags/v2.0.0` → ref points to commit
`1ca85d65ac2ce28ea9762910c09a954c08a37140` (commit message: *"Improved docs for module.py + multiple
cleanups in docs (#1478)"*; author Charlelie Laurent; commit date 2026-03-09T23:42:38Z).

**Rationale.** v2.0.0 is the latest tagged release in the upstream repo and is therefore the most
stable, well-known-good cut available. The previous releases (`v1.3.0` 2025-11-18, `v1.2.0` 2025-08-25,
`v1.1.1` 2025-06-16, `v1.1.0` 2025-06-10) are all from the older naming era; v2.0.0 reflects the
post-rename `physicsnemo` package and the current `examples/cfd/vortex_shedding_mgn/` and
`examples/cfd/external_aerodynamics/aero_graph_net/` layout. Picking a tag rather than a `main`
HEAD is the conservative choice: tags don't move, so the materializer-side citations in §3 below
remain valid for the life of this case study. (Case study 01 used `b880a6c84a93792d2499d2a9b8ba3a077ddf44e2`
on LagrangeBench's `main` because LB has no recent release tag; physicsnemo does, so we use it.)

**Capture mechanism.** Like case study 01's Modal image `git clone --depth 1`, the case-study-02
materializer image must pin to this sha at image-build time (not at runtime) to prevent silent
drift. Recommended: pin in the Modal image build to `pip install nvidia-physicsnemo==2.0.0` or
`git clone --depth 1 --branch v2.0.0 https://github.com/NVIDIA/physicsnemo`, with the resolved sha
above recorded in the materializer-side image manifest.

---

## 2. Source location at the pinned sha

### P0 — Vortex shedding 2D (`modulus_ns_meshgraphnet`)

| Layer | Path at `1ca85d65...` |
|---|---|
| Example train entrypoint | `examples/cfd/vortex_shedding_mgn/train.py` |
| Example inference entrypoint | `examples/cfd/vortex_shedding_mgn/inference.py` |
| Example train config | `examples/cfd/vortex_shedding_mgn/conf/config.yaml` |
| Dataset class (loader) | `physicsnemo/datapipes/gnn/vortex_shedding_dataset.py` |
| Dataset class name | `VortexSheddingDataset(Dataset)` (line 34) |
| Imported in train.py as | `from physicsnemo.datapipes.gnn.vortex_shedding_dataset import VortexSheddingDataset` (train.py line 30) |
| Raw-data source | DeepMind MeshGraphNets cylinder_flow tfrecord (download script `examples/cfd/vortex_shedding_mgn/raw_dataset/download_dataset.sh`) |

The example trains-from-scratch. The NGC checkpoint `modulus_ns_meshgraphnet`
(<https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_ns_meshgraphnet>)
is a separately distributed pretrained snapshot for which **the inference script in
`examples/cfd/vortex_shedding_mgn/inference.py` is the canonical rollout driver** (see §5
known-unknowns about checkpoint-vs-source version compatibility).

### P1 — Ahmed body (`modulus_ahmed_body_meshgraphnet`)

| Layer | Path at `1ca85d65...` |
|---|---|
| Example train entrypoint | `examples/cfd/external_aerodynamics/aero_graph_net/train.py` |
| Example data config (Ahmed) | `examples/cfd/external_aerodynamics/aero_graph_net/conf/data/ahmed.yaml` |
| Dataset class (loader) | `physicsnemo/datapipes/gnn/ahmed_body_dataset.py` |
| Dataset class name | `AhmedBodyDataset(Dataset)` (line 74) |
| Resolved via Hydra | `_target_: physicsnemo.datapipes.gnn.ahmed_body_dataset.AhmedBodyDataset` (ahmed.yaml line 17) |
| Raw-data source | NVIDIA-internal (README: "*To request access to the full dataset, please reach out to the [NVIDIA PhysicsNeMo team](mailto:physicsnemo-team@nvidia.com).*") |

**Migration note.** The modulus → physicsnemo rename did not move these two files relative to the
package root. Both `physicsnemo/datapipes/gnn/vortex_shedding_dataset.py` and
`physicsnemo/datapipes/gnn/ahmed_body_dataset.py` are present at the v2.0.0 tag and are still the
loaders referenced by the example train/inference scripts. The Ahmed-body example was moved into
the umbrella `examples/cfd/external_aerodynamics/aero_graph_net/` directory which now also serves
DrivAerNet under the same train.py; Ahmed-body is selected via the Hydra config group `data=ahmed`.

**P1 caveat.** Ahmed-body raw data is **not** in the public repo. The case-study-02 materializer
will need either (a) the NGC `modulus_datasets-ahmed_body_test` resource
(<https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_datasets-ahmed_body_test>)
or (b) the NGC `modulus_ahmed_body_meshgraphnet` model package's bundled sample inputs. Both routes
require NGC API access. This is a Gate-D-relevant point for spec §6 fallback discussion: if
NGC-only distribution blocks materialization, the Ahmed-body P1 target may need to demote.

---

## 3. Loader-side assertions

Each row cites the file at sha `1ca85d65ac2ce28ea9762910c09a954c08a37140`. Line numbers are
verbatim from the downloaded source.

### 3.1 VortexSheddingDataset assertions (P0)

| # | Assertion (plain) | Source | Code quote | Parameterized by |
|---|---|---|---|---|
| V1 | `meta.json` exists in `data_dir` | `physicsnemo/datapipes/gnn/vortex_shedding_dataset.py:286` | `with open(os.path.join(path, "meta.json"), "r") as fp:` (implicit `FileNotFoundError`) | `data_dir` |
| V2 | `<split>.tfrecord` exists in `data_dir` (split ∈ {train, valid, test}) | same file:289 | `tfrecord_path = os.path.join(path, split + ".tfrecord")` (consumed by `TFRecordDataset` constructor:300-305) | `split` |
| V3 | `meta["field_names"]` is iterable of record keys; every key is `"byte"`-encoded | same file:297 | `description = {k: "byte" for k in meta["field_names"]}` | `meta.json` content |
| V4 | `meta["features"]` provides per-key `dtype` ∈ {float32, float64, int32, int64} (other dtypes attempted via `getattr(np, v["dtype"])`) | same file:425–431 | `dtype_map = {"float32": np.float32, "float64": np.float64, "int32": np.int32, "int64": np.int64}` then `dtype = dtype_map.get(v["dtype"], getattr(np, v["dtype"]))` | per-field meta dtype |
| V5 | Each feature has `shape` field; raw bytes reshape into it | same file:436 | `data = data.reshape(v["shape"])` (implicit `ValueError` on size mismatch) | per-field meta shape |
| V6 | `meta["trajectory_length"]` present for static-tiled features | same file:441 | `data = np.tile(data, (meta["trajectory_length"], 1, 1))` | `trajectory_length` |
| V7 | For `type=="dynamic_varlen"` fields, a sibling `"length_" + k` field exists | same file:443–444 | `row_len = np.frombuffer(rec_bytes["length_" + k], dtype=np.int32)` | per-field meta `type` |
| V8 | Records include keys: `cells`, `mesh_pos`, `node_type`, `velocity`, `pressure` | same file:86–124 | first-axis slice `arr[:num_steps]` and accesses `data_np["cells"]`, `data_np["mesh_pos"]`, `data_np["node_type"]`, `data_np["velocity"]`, `data_np["pressure"]` | `num_steps`, dataset record schema |
| V9 | `cells` has at least 1 frame; cell array is a `(num_cells, 3)`-like indexable (triangle mesh assumption) | same file:86, 311–313 | `src, dst = self.cell_to_adj(data_np["cells"][0])` and within `cell_to_adj`: `src = [cells[i][indx] for i in range(num_cells) for indx in [0, 1, 2]]` | dataset record schema |
| V10 | `mesh_pos[0]` and `cells[0]` are time-invariant (loader bakes graph from frame 0 only — README: "*A single adj matrix is used for each transient simulation. Do not use with adaptive mesh or remeshing*", line 39) | same file:86, 88 | `src, dst = self.cell_to_adj(data_np["cells"][0])` then `add_edge_features(graph, data_np["mesh_pos"][0])` | stationary-mesh invariant |
| V11 | `velocity` has trajectory length ≥ 2 (so `_drop_last` and `_push_forward` produce ≥ 1 frame) | same file:122–123, 372–377 | `features["velocity"] = self._drop_last(data_np["velocity"])` → `torch.tensor(invar[0:-1], …)`; `targets["velocity"] = self._push_forward_diff(…)` → `torch.tensor(invar[1:] - invar[0:-1], …)` | `num_steps` |
| V12 | After slicing, all four time-varying fields have first-axis length `num_steps` | same file:84–85, 120 | `data_np = {key: arr[:num_steps] for key, arr in data_np.items()}` | `num_steps` |
| V13 | `length = num_samples * (num_steps - 1)` is the dataset cardinality (consumer indexes 0…length-1) | same file:73, 162–163, 186 | `self.length = num_samples * (num_steps - 1)` then `gidx = idx // (self.num_steps - 1)` and `tidx = idx % (self.num_steps - 1)` | `num_samples`, `num_steps` |
| V14 | For non-train splits, `edge_stats.json` and `node_stats.json` exist in CWD (saved during train fit) | same file:103, 141 | `self.edge_stats = load_json("edge_stats.json")` and `self.node_stats = load_json("node_stats.json")` (implicit `FileNotFoundError`) | `split` |
| V15 | Stats are dim-matched to edge/node feature widths at normalize time | same file:340–341, 347–351 | `if (invar.size()[-1] != mu.size()[-1]) or (invar.size()[-1] != std.size()[-1]): raise AssertionError("input and stats must have the same size")` and `raise AssertionError("Graph edge data must be same size as stats.")` | stats schema vs feature width |
| V16 | `node_type` values map into `{0, 3, 4, 5, 6}` (the one-hot encoder shifts by `-3` for non-zero) | same file:362–368 | `node_type = torch.where(node_type == 0, torch.zeros_like(node_type), node_type - 3); node_type = F.one_hot(node_type.long(), num_classes=4)` (out-of-range value → `RuntimeError`) | record schema |
| V17 | Edge attribute width is 3 (relative-displacement xy + L2 norm) | same file:334, and conf/config.yaml `num_edge_features: 3` | `graph.edge_attr = torch.cat((disp, disp_norm), dim=1)` | mesh dimensionality (2D) |
| V18 | Node-feature width at `__getitem__` is `velocity(2) + one_hot(4) = 6`; target width is `velocity_diff(2) + pressure(1) = 3` | same file:165–174, conf/config.yaml `num_input_features: 6`, `num_output_features: 3` | `node_features = torch.cat((self.node_features[gidx]["velocity"][tidx], self.node_type[gidx]), dim=-1)` then `node_targets = torch.cat((velocity_diff, pressure), dim=-1)` | record schema |

### 3.2 AhmedBodyDataset assertions (P1)

| # | Assertion (plain) | Source | Code quote | Parameterized by |
|---|---|---|---|---|
| A1 | `data_dir/<split>/` directory exists | `physicsnemo/datapipes/gnn/ahmed_body_dataset.py:150–151` | `if not self.data_dir.is_dir(): raise IOError(f"Directory not found {self.data_dir}")` | `data_dir`, `split` |
| A2 | `data_dir/<split>_info/` directory exists | same file:152–154 | `if not self.info_dir.is_dir(): raise IOError(f"Directory not found {self.info_dir}")` | `data_dir`, `split` |
| A3 | At least one `case<N>.vtp` file exists in the split dir; case ids parse as integers | same file:165–166 | `for case_file in sorted(self.data_dir.glob("*.vtp")): case_id = int(str(case_file.stem).removeprefix("case"))` (implicit `ValueError` if non-integer suffix) | filename convention |
| A4 | Each `caseN.vtp` has a sibling `caseN_info.txt` in the info dir | same file:168–170 | `if not case_info_file.is_file(): raise IOError(f"File not found {case_info_file}")` | filename convention |
| A5 | `num_samples` must not exceed available cases | same file:178–183 | `if self.num_samples > self.length: raise ValueError(f"Number of available {self.split} dataset entries ({self.length}) is less than the number of samples ({self.num_samples})")` | `num_samples` |
| A6 | VTP file must be readable and have `.vtp` extension | `physicsnemo/datapipes/gnn/utils.py:43–48` (`read_vtp_file`) | `if not os.path.exists(file_path): raise FileNotFoundError(...)` and `if not file_path.endswith(".vtp"): raise ValueError(...)` | file path |
| A7 | VTP reader must return non-None polydata | `physicsnemo/datapipes/gnn/utils.py:60–61` | `if polydata is None: raise ValueError(f"Failed to read polydata from {file_path}")` | VTP file content |
| A8 | Polydata `GetPoints()` and `GetPolys()` must be non-None | ahmed_body_dataset.py:547–558 | `if points is None: raise ValueError("Failed to get points from the polydata.")` and `if polys is None: raise ValueError("Failed to get polygons from the polydata.")` | VTP file content |
| A9 | Polydata `GetPointData()` must be non-None | same file:582–584 | `if point_data is None: raise ValueError("Failed to get point data from the polydata.")` | VTP file content |
| A10 | Output keys (`p`, `wallShearStress` by default) must each be a named point-data array in the VTP | same file:586–597 | `if array_name in outvar_keys: …graph[array_name] = torch.tensor(array_data, dtype=torch.float32)` (silent omission if missing → downstream `KeyError` in `normalize_node`:343) | `outvar_keys` |
| A11 | `caseN_info.txt` must be YAML-parseable and include keys: `Velocity`, `Re (based on length)`, `Length`, `Width`, `Height`, `GroundClearance`, `SlantAngle`, `FilletRadius` | same file:504–515 | `info = yaml.safe_load(file); return FileInfo(info["Velocity"], info["Re (based on length)"], info["Length"], info["Width"], info["Height"], info["GroundClearance"], info["SlantAngle"], info["FilletRadius"])` (implicit `KeyError`) | info file schema |
| A12 | For non-train splits, `node_stats.json` and `edge_stats.json` must exist in CWD | same file:229–236 | `if not os.path.exists("node_stats.json"): raise FileNotFoundError("node_stats.json not found! …")` and analogous for `edge_stats.json` | `split` |
| A13 | Every normalize key must exist as a node attribute on every loaded graph | same file:330–345 | `self.graphs[i][key] = (self.graphs[i][key] - self.node_stats[key + "_mean"]) / self.node_stats[key + "_std"]` (implicit `KeyError` on missing attr) | `normalize_keys` |
| A14 | `compute_drag=True` requires `pyvista` available (lazy `OptionalImport`) and a `width`/`height` pair in info file | same file:264–276 | `mesh = pv.read(file_path); … frontal_area = info.width * info.height / 2 * (10 ** (-6))` | `compute_drag` |
| A15 | `pos` attribute on every graph (set from VTP vertices), float32 | same file:579 | `graph.pos = torch.tensor(vertices, dtype=torch.float32)` | VTP content |
| A16 | Input-feature concatenation requires every key in `invar_keys` to be present on each graph | same file:347–349 | `self.graphs[i].x = torch.cat([self.graphs[i][key] for key in self.input_keys], dim=-1)` (implicit `KeyError`) | `invar_keys` |
| A17 | Output-feature concatenation requires every key in `outvar_keys` to be present | same file:350–352 | `self.graphs[i].y = torch.cat([self.graphs[i][key] for key in self.output_keys], dim=-1)` | `outvar_keys` |
| A18 | Edge attr width matches `edge_stats["edge_mean"]`/`["edge_std"]` dim (implicit; no shape assert at normalize time, would silently broadcast) | same file:372–376 | `self.graphs[i].edge_attr = (self.graphs[i].edge_attr - self.edge_stats["edge_mean"]) / self.edge_stats["edge_std"]` | edge_stats schema |

---

## 4. Pre-flight test scaffold

For each assertion above, the materializer's pre-flight test should mirror the loader's contract on
a small fixture (e.g., the NGC-shipped sample input, or a single-trajectory tfrecord). Tests should
live alongside the harness adapter — analogous to LB's `_harness/tests/` — and each test's docstring
must cite `<filename>:<line>` at sha `1ca85d65ac2ce28ea9762910c09a954c08a37140`. None of the tests
below need GPU; all are pure data inspection.

### 4.1 P0 (VortexSheddingDataset) — pre-flight tests

| Test name (suggested) | Mirrors assertion(s) | What it should verify |
|---|---|---|
| `test_vortex_meta_json_present_at_data_dir` | V1, V3, V4, V6 | `meta.json` opens; required keys (`field_names`, `features`, `trajectory_length`) parse; every per-field `dtype` is in the supported set; raise if any unknown dtype is found. |
| `test_vortex_tfrecord_present_for_each_split` | V2 | For each split the materializer plans to use (`test`, optionally `valid`/`train`), assert `<split>.tfrecord` exists under `data_dir`. Independent file per split. |
| `test_vortex_decode_first_record_shape_dtype` | V5, V8, V9, V11, V12, V16 | Decode one record using the same `description = {k: "byte" for k in meta["field_names"]}` machinery; check that decoded `cells`, `mesh_pos`, `node_type`, `velocity`, `pressure` all reshape per `meta["features"][k]["shape"]`; check `velocity.shape[0] ≥ 2`; check `node_type` values are subset of `{0,3,4,5,6}`. Combine in one test — same record, multiple invariants. |
| `test_vortex_dynamic_varlen_has_length_sibling` | V7 | For any meta field with `type == "dynamic_varlen"`, assert the decoded record carries the sibling `"length_" + k` byte field. (`cells` is typically dynamic_varlen for cylinder_flow — verify on fixture rather than assume.) |
| `test_vortex_stationary_mesh_invariant` | V10 | Verify on one trajectory that `cells[t] == cells[0]` and `mesh_pos[t] == mesh_pos[0]` for all sliced `t < num_steps`. README explicitly states this assumption is unchecked at load time. |
| `test_vortex_length_arithmetic_matches_consumer_indexing` | V13 | Instantiate `VortexSheddingDataset(num_samples=N, num_steps=K)` with the smallest viable N (e.g., 1) and K (e.g., 3), then assert `len(dataset) == N * (K - 1)` and that `dataset[len-1]` returns without error while `dataset[len]` raises. |
| `test_vortex_stats_json_present_when_not_train` | V14 | For `split="test"` materialization, assert `edge_stats.json` and `node_stats.json` are findable in the directory the dataset is instantiated from (the loader uses CWD-relative paths — the materializer must `chdir` or symlink). |
| `test_vortex_stats_dim_matches_features` | V15 | Load `node_stats.json` and assert: `velocity_mean` and `velocity_std` are length-2; `pressure_mean`/`std` are length-1; `velocity_diff_mean`/`std` are length-2. Load `edge_stats.json` and assert `edge_mean`/`edge_std` are length-3. This mirrors the `AssertionError` raised inside `normalize_node` / `normalize_edge`. |
| `test_vortex_emit_widths_match_inference_config` | V17, V18 | Run `dataset[0]` once, assert `graph.x.shape[-1] == 6`, `graph.y.shape[-1] == 3`, `graph.edge_attr.shape[-1] == 3` — matches `conf/config.yaml` `num_input_features`, `num_output_features`, `num_edge_features`. Single test; three asserts on the same returned tuple. |

### 4.2 P1 (AhmedBodyDataset) — pre-flight tests

| Test name (suggested) | Mirrors | What it should verify |
|---|---|---|
| `test_ahmed_split_dirs_present` | A1, A2 | `data_dir/<split>/` and `data_dir/<split>_info/` both `is_dir()`. |
| `test_ahmed_case_file_pairing` | A3, A4 | For every `case<N>.vtp` in the split dir, a `case<N>_info.txt` exists in the info dir; every `N` parses as `int`. |
| `test_ahmed_num_samples_within_available` | A5 | Given the materializer's chosen `num_samples`, assert `num_samples ≤ len(sorted(data_dir.glob("*.vtp")))`. (Cheap: counts files, doesn't open VTPs.) |
| `test_ahmed_vtp_readable_and_keyed` | A6, A7, A8, A9, A10 | Open one VTP via `physicsnemo.datapipes.gnn.utils.read_vtp_file`; assert non-None points/polys/point_data; assert every key in `outvar_keys` (default `{"p", "wallShearStress"}`) is present as a named array. Combine: single VTP, multiple invariants. |
| `test_ahmed_info_file_schema` | A11 | Parse one `caseN_info.txt` with `yaml.safe_load`; assert all eight required keys (`Velocity`, `Re (based on length)`, `Length`, `Width`, `Height`, `GroundClearance`, `SlantAngle`, `FilletRadius`) are present. |
| `test_ahmed_stats_json_present_when_not_train` | A12 | For inference materialization (`split="test"`), `node_stats.json` and `edge_stats.json` must be in CWD. The loader's error message is informative — pre-flight should fail with the same message. |
| `test_ahmed_normalize_key_coverage` | A13, A16, A17 | After `create_graph` runs on a single case, assert that every entry in the union (`normalize_keys ∪ invar_keys ∪ outvar_keys`) is present on the graph object. (Single graph, multi-key assert.) |
| `test_ahmed_pos_dtype_float32` | A15 | `graph.pos.dtype == torch.float32` on the freshly-constructed graph. |
| `test_ahmed_compute_drag_optdeps_if_requested` | A14 | If the materializer plans to set `compute_drag=True`, assert `import pyvista` succeeds in the materializer image and that `info_file` has non-zero `Width`, `Height` (frontal_area would otherwise be 0 → division-by-zero in `coeff`). Skip if `compute_drag=False`. |
| `test_ahmed_edge_stats_dim_matches_edge_attr` | A18 | Mirror V15's pattern — the Ahmed loader does **not** check this at normalize time (would silently broadcast). Therefore the pre-flight check is *more* important here, not less. Assert `edge_stats["edge_mean"].shape[-1] == 4` (xyz disp + L2 norm = 4 in 3D, vs 3 for 2D vortex). Verify on fixture. |

### 4.3 Shared-test consolidation

- V5 + V8 + V12 + V16: one decoded-record test covers all four (single record, multiple asserts).
- A6 + A7 + A8 + A9 + A10: one VTP-open test covers all five.
- A11: single info-file parse.

This reduces P0 to ~8 distinct tests and P1 to ~9 distinct tests — comparable in count to the LB
materializer pre-flight suite.

---

## 5. Known-unknowns

Items that source inspection alone could not resolve. Each gets an action.

### 5.1 NGC-checkpoint ↔ v2.0.0 source compatibility (BLOCKING)

The NGC checkpoints `modulus_ns_meshgraphnet`
(<https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_ns_meshgraphnet>) and
`modulus_ahmed_body_meshgraphnet`
(<https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/models/modulus_ahmed_body_meshgraphnet>)
were trained against older modulus (pre-rename) source. The `MeshGraphNet` model class API may have
changed between training-era modulus and v2.0.0 physicsnemo. The repo READMEs do not advertise
which source version the NGC checkpoints are pinned against.

**Action (a) empirical smoke:** Before any rollout fire, run `load_checkpoint(ckpt_path,
models=self.model)` against v2.0.0's `MeshGraphNet` constructor with `num_input_features=6,
num_edge_features=3, num_output_features=3` (matching v2.0.0 `conf/config.yaml`) and confirm
state-dict keys match exactly (no missing/unexpected keys, no shape mismatches). This is
zero-GPU-cost — load on CPU. The mesh-side analog of LB's `test_inference_matches_ngc_sample`
gate (case study 02 README line 27: `max-abs-error ≤ 10⁻³`).

**Action (b) doc request:** If keys don't match, file an upstream question (NGC model card or
PhysicsNeMo issue) asking which physicsnemo commit/release the NGC checkpoint was trained against.
Do **not** silently rename keys to make the load succeed — that's the same pattern as case
study 01's hardcoded-`valid.h5` failure mode (silent contract drift).

### 5.2 Ahmed-body data distribution (BLOCKING for P1)

The Ahmed-body raw dataset is gated ("contact `physicsnemo-team@nvidia.com`"). The NGC resource
`modulus_datasets-ahmed_body_test`
(<https://catalog.ngc.nvidia.com/orgs/nvidia/teams/modulus/resources/modulus_datasets-ahmed_body_test>)
appears to be the public-facing test subset. Source inspection cannot confirm the test subset
includes the `*_info.txt` files (A2, A4) or matches the loader's `caseN.vtp` filename convention
(A3). The NGC `modulus_ahmed_body_meshgraphnet` model card mentions a bundled "set of sample
inputs" — those are the canonical pre-flight fixture.

**Action (a):** Download the NGC sample inputs and run §4.2 tests A1–A11 against them. If naming
diverges from `case<N>.vtp` / `case<N>_info.txt`, the materializer needs a rename adapter — not a
patched loader, since the loader is the contract.

**Action (b):** If NGC distribution is insufficient (no `*_info.txt` ships), Ahmed-body P1 demotes
per case-study-02 README's Gate-D fallback — note this in the rollouts manifest.

### 5.3 `noise_std=0.02` is applied only on `split="train"`

`VortexSheddingDataset` line 127: `if split == "train": features["velocity"], targets["velocity"] = self._add_noise(...)`. For test/inference rollouts (the case study 02 use case) noise is zero. This is implicit at load time, but the NGC checkpoint was *trained* with noise injection per
DeepMind's MGN methodology. No source-level assertion enforces this.

**Action:** Document expectation in materializer that `split="test"` is the only correct path for
rollout; raise an explicit assertion if anyone tries to instantiate with `split="train"` for the
purpose of running pretrained inference.

### 5.4 Stats-file CWD coupling

Both loaders read `node_stats.json` / `edge_stats.json` from the current working directory
(VortexShedding line 103, 141; AhmedBody line 229–238). Hydra's `chdir: True` in the example
`config.yaml` masks this — the train script changes directory into `./outputs/` before constructing
the dataset, and that's where the stats files end up. The materializer must reproduce this CWD
discipline.

**Action:** Materializer-side pre-flight should `os.chdir()` to a known directory containing the
stats JSONs before calling `VortexSheddingDataset(split="test", ...)`. Bake this into the adapter
and document it in the harness docstring, citing line 103/141 at this sha.

### 5.5 `meta.json` `trajectory_length` vs `num_steps`

Loader slices `arr[:num_steps]` (line 85, 120). If `num_steps > trajectory_length`, numpy returns
the full array silently (no error), but the static-tiled features were already tiled to
`trajectory_length` (line 441) — leading to a silent mismatch in time-axis lengths between
static-tiled and dynamic fields. Not asserted anywhere.

**Action:** Materializer pre-flight test should `assert num_steps <= meta["trajectory_length"]`
before instantiating the dataset. Mirrors LB's `subseq_length` known-unknown (rung-4b §5.1
analog) — assertion at downstream slicing, not at load time.

### 5.6 fp32 vs fp64 precision contract

`_decode_record` accepts both `float32` and `float64` (line 425–426); downstream
`torch.tensor(invar[0:-1], dtype=torch.float)` (line 373) promotes/demotes to whichever `float` is
the default torch dtype. If the materializer image's default torch dtype is fp64, this changes
numerical behavior vs the NGC checkpoint's training-time fp32. Not asserted.

**Action:** Materializer image must `torch.set_default_dtype(torch.float32)` before dataset
construction, and pre-flight should assert it on entry.

### 5.7 `node_type` value range (V16)

The one-hot encoder line 363–368 maps `0 → 0` and `non-zero → value - 3`, then `F.one_hot(...,
num_classes=4)`. This implicitly assumes `node_type ∈ {0, 3, 4, 5, 6}` (so `non-zero - 3 ∈
{0,1,2,3}`). Any other value triggers `F.one_hot`'s `RuntimeError: Class values must be smaller
than num_classes`. The DeepMind cylinder_flow schema does match this, but it's not documented in
the loader.

**Action:** Pre-flight `test_vortex_decode_first_record_shape_dtype` should assert the unique
values of `node_type[0]` are a subset of `{0, 3, 4, 5, 6}` before dataset construction. Source
citation: lines 363–368.

---

## 6. Substrate-class taxonomy mapping (rung-4c §5.2 "classify when you exercise")

Per the D0-22 empirical-classification discipline, **do not pre-classify**. The classification is
post-empirical-probe — it requires running the physics-lint probe rules against the rollout and
observing which class's invariants hold and which fail. What this document can do is list the
candidate classes for the P0 target and name the discriminating observable.

### P0 target: 2D incompressible Navier–Stokes, cylinder wake

**Why pre-classification is wrong here.** The vortex-shedding domain is incompressible NS with
boundary forcing (inflow velocity, no-slip cylinder, outflow); whether it presents as `conservative`,
`dissipative`, or `open-driven-dissipative` depends on what the rollout actually exhibits over
the windowed measurement — and that depends on the rollout horizon, the Reynolds regime, and
whether the cylinder wake reaches its limit-cycle vortex-shedding state within the window.

**Candidate substrate classes (subject to empirical probe):**

1. **`open-driven-dissipative`** — the physically motivated default. The flow has continuous mass
   influx at the inlet and outflux at the outlet (open boundaries); viscous dissipation
   internally; the limit-cycle vortex shedding is the dissipative-system's attractor when forcing
   balances dissipation. This is what the textbook cylinder-wake problem is.

2. **`dissipative`** (closed-form) — only correct if the rollout's effective system is treated as
   "the box minus the boundary fluxes", i.e., interior kinetic energy decays absent ongoing
   forcing. Probably wrong because the actual rollout includes the open boundaries explicitly.

3. **`conservative`** — wrong but worth probing as a sanity gate. Incompressible NS is **not**
   conservative (viscosity dissipates KE). If a `conservative`-class probe (e.g., total kinetic
   energy invariance up to numerical roundoff) shows the rollout is conservative-like, that is a
   physics bug (training-data error or numerical-scheme energy-injection), not a property of NS.

**Discriminating observables (what the probe should measure):**

- **Divergence of velocity field, ∫|∇·v| dV over the interior:** for incompressible NS this should
  be ~0 up to discretization/numerical noise. Non-zero ∇·v is the headline mass-conservation
  signal (`PH-CON-001` per case-study-02 README). This discriminates "respecting incompressibility"
  from "not."

- **Kinetic energy budget, dKE/dt vs (inflow KE flux) − (outflow KE flux) − (viscous dissipation):**
  if the budget closes (within numerical tolerance), the rollout is `open-driven-dissipative` in
  the proper sense. If KE grows unboundedly with zero net inflow, the rollout is
  energy-non-conserving in a way that suggests training drift.

- **Vortex-shedding Strouhal number (St = f_shedding · D / U_∞):** should land in St ≈ 0.16–0.21
  for Re ∈ [100, 300] (the cylinder-flow dataset's regime). This is a sanity gate for whether the
  rollout is producing the correct limit-cycle, not just any oscillatory state.

**Therefore:** Do not stamp a class on the manifest at materialization time. Run the rollout,
measure the three observables above, and let the class fall out of the measurement. This matches
rung-4c §5.2's discipline and avoids the "candidate class → confirmation bias" failure mode.

### P1 target: Ahmed-body steady RANS

Steady-state by definition — time-averaging removes the dissipative-dynamics signal. The
candidate-class question collapses to "does the steady-state pressure/wall-shear distribution
respect the RANS momentum balance at the boundary." `PH-BC-001` (no-slip) is the natural probe.
Substrate-class taxonomy is less informative here — the steady solver is a fixed-point system,
not a dynamical-system rollout — so the rung-4c "classify when you exercise" doctrine has weaker
purchase. Flag this in the rung-4c §5.2 discussion: **steady solvers are a known boundary case for
the empirical-classification framework, not a counterexample.**

---

## 7. Provenance

- **Document date:** 2026-05-11
- **physics-lint commit read against:**
  `e3c58474bb93ee91eeb9c90d245289505b0e46d0`
  (branch `feature/rung-4c-substrate-class-extension`, verified via
  `git -C /Users/zenith/Desktop/physics-lint rev-parse HEAD`).
- **PhysicsNeMo sha pinned:**
  `1ca85d65ac2ce28ea9762910c09a954c08a37140`
  (tag `v2.0.0`, published 2026-03-10T23:47:04Z; resolved via
  `gh api repos/NVIDIA/physicsnemo/git/refs/tags/v2.0.0`).
- **Source files inspected (all at the pinned sha, fetched via
  `raw.githubusercontent.com/NVIDIA/physicsnemo/1ca85d65ac2ce28ea9762910c09a954c08a37140/...`):**
    - `physicsnemo/datapipes/gnn/vortex_shedding_dataset.py` (449 lines)
    - `physicsnemo/datapipes/gnn/ahmed_body_dataset.py` (599 lines)
    - `physicsnemo/datapipes/gnn/utils.py` (read_vtp_file, save_json)
    - `examples/cfd/vortex_shedding_mgn/train.py` (verified imports)
    - `examples/cfd/vortex_shedding_mgn/inference.py` (verified rollout API)
    - `examples/cfd/vortex_shedding_mgn/conf/config.yaml` (verified feature widths)
    - `examples/cfd/vortex_shedding_mgn/raw_dataset/download_dataset.sh` (verified DeepMind data path)
    - `examples/cfd/vortex_shedding_mgn/README.md` (verified stationary-mesh assumption documented in repo)
    - `examples/cfd/external_aerodynamics/aero_graph_net/train.py` (verified Hydra-instantiate path)
    - `examples/cfd/external_aerodynamics/aero_graph_net/conf/data/ahmed.yaml` (verified `_target_` resolves to AhmedBodyDataset)
    - `examples/cfd/external_aerodynamics/aero_graph_net/README.md` (verified Ahmed-data NGC-only distribution)
- **NGC catalog entries verified (web search 2026-05-11):**
    - `modulus_ns_meshgraphnet`
    - `modulus_ahmed_body_meshgraphnet`
    - `modulus_datasets-ahmed_body_test`
- **Methodology reference:**
  `external_validation/_rollout_anchors/methodology/docs/2026-05-07-rung-4b-equivariance-table.md` §5.1
  (four-step source-review-before-compute pattern).
- **No GPU work performed. No code modified. No paths invented — every external citation above
  is fetch-confirmed.**
