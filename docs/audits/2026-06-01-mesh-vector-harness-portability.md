# Mesh-vector harness-portability audit (2026-06-01)

**Task 1 of the v1.2.0 plan** (`docs/plans/2026-06-01-physics-lint-v1-2-0-mesh-vector.md`).
No package code is written here. This audit is the guard against building a
public claim on research-harness behavior — the failure mode that produced the
P3.3 substrate blocker (a spec inheriting an unverified "the Action detected it"
label). Every verdict below is grounded in the harness source at master
`86ceb0a`, cited by `file:line`, not inherited from the plan's proposed table.

Sources read:
- `external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py` (the mesh path)
- `external_validation/_rollout_anchors/_harness/lint_mesh_rollout.py` (the CLI shim)
- `external_validation/_rollout_anchors/_harness/SCHEMA.md` §2 (npz schema) + §4.1 (Gate B)

Installed toolchain at audit time: `scikit-fem 12.0.1` (pyproject pins `[mesh] = ["scikit-fem>=10"]`).

---

## 1. Classification table

Verdicts: **transplantable** (pure NumPy/scikit-fem, no Modal/DGL/torch — lifts into the package as-is or re-expressed), **needs-redesign** (research-coupled shape; promote to a public surface), **research-only** (out of v1.2.0 scope).

| harness symbol | lines | verdict | rationale (verified) | maps to (package) |
|---|---|---|---|---|
| `_build_graph_mesh_basis` | 627–643 | **transplantable** | pure NumPy + skfem: `MeshTri(node_positions.astype(f64).T, cells.T)` (both `ascontiguousarray`) + `Basis(ElementTriP1())`. No Modal/DGL/torch. Reads `metadata["cells_2d"]` + `node_positions`. | `MeshVectorField._build_basis` (Task 2) |
| `_fe_divergence_defect_max` | 646–666 | **transplantable (verbatim)** | pure NumPy + skfem `basis.interpolate(v[:,i]).grad` / `.dx`. Frobenius-normalized (see §2). Signature `(basis, velocity)`. | `ph_con_005._fe_divergence_defect_max` (Task 5) |
| `_can_compute_graph_mesh_fe` | 568–624 | **transplantable (re-expressed, not 1:1)** | three precondition checks: (1) `import skfem`, (2) `metadata["cells_2d"] is not None`, (3) velocity `ndim==3 and shape[-1]==2`. The package expresses these differently: (1) the `[mesh]` import guard on `MeshVectorField`, (2)+(3) `MeshVectorField.__init__` shape validation. | field export guard + `MeshVectorField.__init__` (Task 2) |
| `mass_conservation_defect_on_mesh` (graph branch) | 684–715 | **transplantable** | for `not is_regular_grid`, returns exactly `HarnessDefect(value=_fe_divergence_defect_max(basis, velocity))` (714–715). The `‖∇·v‖_L²/‖v‖_L²` in its docstring (691) is the **regular-grid FD branch**, not the graph path. | ported via `_fe_divergence_defect_max` |
| `MeshRollout` | 151–282 | **needs-redesign** | research dataclass: `node_values` dict + `node_type` + `edge_index` + `dt` + freeform `metadata` (incl. the ad-hoc `cells_2d`). Promote the ∇·v-relevant slice to an explicit public schema. | `loader._load_mesh_vector_dump` + public npz schema (Task 4) |
| `load_mesh_rollout_npz` / `save_mesh_rollout_npz` | 283–352 / 464–484 | **needs-redesign** | require `{node_positions, node_type, node_values, dt, metadata}` — **`dt` is required** (291). Public schema promotes `cells`/`velocity` to top-level and makes `dt` **optional** (PH-CON-005 ignores it). | `loader._load_mesh_vector_dump` (Task 4) |
| `_assert_loader_contract_mgn` | 355–461 | **research-only** | MGN-checkpoint-specific loader contract; reads `metadata["dataset"]`/`["framework"]`/`["model"]` + `node_type` bounds. None of these are on the ∇·v path. Not ported. | — |
| `_fe_kinetic_energy_series` | 669–681 | **research-only (v1.2.x energy follow-on)** | `KE(t)=0.5 ∫|v|² dV`; the energy-drift follow-on rule's kernel, not v1.2.0 scope. | — (spec §9) |
| `materialize_grid_field` | 117–142 | **research-only** | regular-grid / FNO-on-Darcy path (`GridField` wrap). Out of v1.2.0 scope. | — |
| `_gridded_velocity_view` + regular-grid FD branch | 552–565, 717–730 | **research-only** | regular-grid path; uses the `‖∇·v‖_L²/‖v‖_L²` normalization (a *different* form — explicitly NOT what v1.2.0 ports). | — |

**`lint_mesh_rollout.py`** (the harness CLI shim, 3.6 KB) is research-only: it wires `load_mesh_rollout_npz` → the `*_on_mesh` defect mirrors → `sarif_emitter.py`. The public path replaces it with `loader.load_target` → `physics-lint check` dispatch (Tasks 4 + 6).

---

## 2. Frozen-formula confirmation

`_fe_divergence_defect_max` (646–666) computes, per timestep, on the P1 basis:

```
div  = gvx[0] + gvy[1]                                  # ∂vx/∂x + ∂vy/∂y
frob = sqrt(gvx[0]² + gvx[1]² + gvy[0]² + gvy[1]²)      # ‖∇v‖_F
rel  = ∫|div| dx / max(∫ frob dx, 1e-12)
defect = max over t of rel
```

i.e. **`max_t ( ∫|∇·v| dV / ∫‖∇v‖_F dV )`** — the **Frobenius-gradient
normalization**, NOT `‖∇·v‖_L²/‖v‖_L²`. PH-CON-005 ports **this** form
verbatim. (The `‖∇·v‖_L²/‖v‖_L²` form named in `mass_conservation_defect_on_mesh`'s
docstring at line 691 is the *regular-grid FD branch* — a different
normalization the package does not implement in v1.2.0.) For graph-mesh,
`mass_conservation_defect_on_mesh` and `_fe_divergence_defect_max` coincide
(714–715), so the two CS02-relevant references agree. The `max(int_frob, 1e-12)`
guard and the `.astype(np.float64)` per-frame upcast are part of the port.

This confirms the spec correction folded during plan-grounding: §3's formula is
the FE Frobenius form porting `_fe_divergence_defect_max`, not `‖∇·v‖_L²/‖v‖_L²`.

---

## 3. Metadata-bag check (spec §5 / pin 3)

**On the ∇·v path, the only inputs are `node_positions`, `cells`, `velocity`.**
Traced through the three functions the path touches:

- `_can_compute_graph_mesh_fe` (568–624): reads `metadata["cells_2d"]` (597) and `node_values["velocity"]` (607).
- `_build_graph_mesh_basis` (627–643): reads `metadata["cells_2d"]` (637) and `node_positions` (639).
- `_fe_divergence_defect_max` (646–666): reads only `basis` + `velocity` (655–656).

The public schema promotes `metadata["cells_2d"] → cells` and
`node_values["velocity"] → velocity` to top-level keys; `node_positions` stays.
**No rule-required field hides in the freeform `metadata` bag on the ∇·v path.**

Two corroborating observations:
1. `cells_2d` is **not** in the documented §2 `metadata` schema (SCHEMA.md
   189–198 lists only `ckpt_hash/ngc_version/git_sha/dataset/model/framework/framework_version/resampling_applied`).
   It is an ad-hoc materializer-staged key (`_can_compute_graph_mesh_fe`
   docstring: "Materializers must stage one frame's cells under this key").
   Promoting it to a first-class `cells` key is exactly the redesign the audit
   table flags for `MeshRollout`.
2. The `metadata["dataset"]`/`["framework"]`/`["model"]` reads are confined to
   `_assert_loader_contract_mgn` (392–461) — the MGN loader contract, which
   belongs to the energy/dissipation follow-ons (system-class dispatch), **not**
   PH-CON-005. The `metadata["grid_shape"]` reads (243–244) are on the
   regular-grid path. Neither is on the ∇·v path.

---

## 4. Reusable fixtures

`_harness/tests/` ships `synthetic_rollouts.py` and a `fixtures/` directory.
These are **informational** for the package: per the plan, the package tests use
**self-contained analytic fixtures** (a linear velocity field, where P1 is exact):
`v=(x, −y)` → `∇·v = 0` → defect ≈ 0; `v=(x, 0)` → `∇·v = 1`, `‖∇v‖_F = 1` →
defect = 1.0 exactly. No harness fixture is imported by the package unit tests.

**Exception — Task 7 Gate-B reference**: the harness-fidelity test
(`_harness/tests/test_ph_con_005_vs_harness.py`) *does* import the harness
directly (`MeshRollout`, `_build_graph_mesh_basis`, `_fe_divergence_defect_max`)
to assert the public rule reproduces it within ε ≤ 1e-4. The package is
importable from there (`_harness/__init__.py` and `_harness/tests/__init__.py`
both exist, so the dotted import path resolves).

---

## 5. Gate-B note (for Task 7)

SCHEMA.md §4.1 pre-registers the harness-vs-public tolerance: **ε ≤ 10⁻⁴ = PASS**
(10⁻⁴ < ε ≤ 10⁻² = APPROXIMATE; > 10⁻² = FAIL). The §4.1 "Reader's note" (written
for the *symmetry* Gate B) establishes the relevant epistemics: when both paths
run the **same computation on the same input**, ε = 0.000e+00 is the *expected,
by-design* outcome, not surprising cross-method agreement.

For PH-CON-005 this is the situation: the Task-7 reference builds the harness
basis via `_build_graph_mesh_basis` and the public basis via `MeshVectorField`,
but **both** call `skfem.MeshTri(positions.T, cells.T)` + `Basis(ElementTriP1())`
on identical arrays, and both run the identical `_fe_divergence_defect_max`
kernel. So ε ≈ 0 is expected; the Gate-B PASS asserts *port fidelity* (no
transcription/normalization drift), not an independent epistemic check. The
rule's credibility is this fidelity guarantee plus the analytic-fixture exactness
(§4), since no absolute PASS/FAIL band is sound (CS02 GT itself ≈ 5.8%).

---

## 6. CS02 doc freeze status (for Task 8)

**Verdict: both Task-8 target READMEs are LIVING scaffolds — direct edit in place
(via the branch PR) is correct. No D-entry/pointer routing is required.**

Targets: `external_validation/_rollout_anchors/02-physicsnemo-mgn/README.md` and
`external_validation/_rollout_anchors/README.md` (§5).

Evidence:
- **No freeze marker** in either file. (`grep` for frozen/snapshot/sha-bound/do-not-edit
  returns only false positives: "design spec … snapshotted into `docs/`" describes
  the *spec/plan* snapshots, and "node stays frozen at its step-0 … value" is about
  a boundary-node velocity.)
- **Active in-place edit history.** The CS02 README's most recent edit is
  `aa38285` — **PR #21**, the v1.1.0 "correct CS01/CS02 substrate framing"
  correction — preceded by P3.1 (`82e0ecd`), P2.2 (`560e319`, `c7c1c95`), P2.1
  (`325cf40`). A README that is corrected in place via PR #21 (the immediately
  prior doc-correction PR) is, by precedent, edited in place, not frozen.
- The `feedback_frozen_writeup_convention` targets `docs/YYYY-MM-DD-*.md` dated
  snapshot files, not case-study READMEs. The CS02 directory has **no
  `DECISIONS.md`** and there is **no `methodology/` directory**, so the
  frozen-route target does not even exist for these files.

**Precise edit sites for Task 8** (so the new claim is verified-to-source and its
siblings traced, per `feedback_correction_can_introduce_new_overclaim`):
- `02-physicsnemo-mgn/README.md` lines **44–54** — the "PH-CON-001 routing —
  harness, not public rule" paragraph, ending "*CLI/Action loader integration for
  mesh — and for particle — is planned for v1.2.0 (see `docs/backlog/v1.2.md`)*."
  v1.2.0 delivers the **∇·v** path through the public CLI via **PH-CON-005 on a
  `mesh_vector` target** — but PH-CON-001 itself still SKIPs on `pde != heat`, and
  scalar-`MeshField` + particle + energy/dissipation remain follow-ons. Do not
  let the correction over-claim a general "mesh runs through the public API".
- `_rollout_anchors/README.md` §5 lines **73–80** — the "PH-CON-001 … returns
  SKIPPED on `pde != heat` … harness reapplies … structural-identity reapplication"
  bullet. After v1.2.0 there is a public path for **∇·v only** (PH-CON-005); the
  bullet must say so without implying PH-CON-001 itself now runs on NS.

---

## 7. Audit verdict

The graph-mesh ∇·v path is **transplantable**: `_build_graph_mesh_basis` and
`_fe_divergence_defect_max` lift into the package with no research coupling, the
rule-facing inputs are exactly `node_positions`/`cells`/`velocity` (no hidden
metadata dependency), and the formula is the Frobenius-normalized FE defect. The
research schema (`MeshRollout` + npz I/O) needs the documented redesign into a
promoted public schema. Energy/dissipation kernels and the regular-grid +
scalar-mesh paths are correctly deferred. No blocker to Tasks 2–9.
