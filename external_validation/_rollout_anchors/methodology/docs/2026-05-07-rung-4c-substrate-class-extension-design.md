# Rung 4c — Substrate-class extension to dam-break-2D (design)

**Date:** 2026-05-07
**Repo:** physics-lint
**Branch (intended):** `feature/rung-4c-substrate-class-extension`, off `feature/rung-4b-t7-subseq-length-fix` tip (PR #8 in flight at sha `861c95c`); rebase to `master` post-PR-#8-merge.
**Status:** Design — pre-implementation. Pre-registers **D0-22** in `external_validation/_rollout_anchors/methodology/DECISIONS.md` before any code change.
**Predecessor:** rung 4b — cross-stack equivariance (`2026-05-07-rung-4b-equivariance-table.md`); 2 SARIFs + 6-trace ε(t) figure at sha `255af5de8d` on PR #8; D0-21 (with amendment 2 for the loader-contract failure class) landing on master via PR #8.
**Successor:** rung 4c implementation plan (`2026-05-07-rung-4c-substrate-class-extension-plan.md`); then execution; then `-table.md`.

---

## 1. Scope and framing

### 1.1 What rung 4c is

Rung 4c extends the harness's substrate-detection layer to a second LagrangeBench substrate class — `open-driven-dissipative` — by reclassifying `dam2d` empirically (post-rollout KE(t) shape inspection) and adding a new SKIP path on `dissipation_sign_violation` for that class. Reuses rung 4a's full SARIF emit pipeline (`_harness/sarif_emitter.py`, `lint_npz_dir.py`, `01-lagrangebench/emit_sarif.py`) and rung 4a's cross-stack renderer (`methodology/tools/render_cross_stack_table.py`) without modification, demonstrating that the schema-uniform machinery handles a second substrate class identically. Generates 20 SEGNN-dam2d + 20 GNS-dam2d trajectories on Modal A10G (40 npzs total, parallel to rung 4a's TGV2D pattern), commits 2 dam-break SARIFs, renders a dam-break cross-stack table, and writes a dated methodology table writeup at `methodology/docs/2026-05-07-rung-4c-substrate-class-extension-table.md`.

### 1.2 Load-bearing claim (frozen headline sentence)

> *"physics-lint's harness substrate-detection layer extends to a second LagrangeBench substrate class — `open-driven-dissipative` — via D0-22's new SKIP path on `dissipation_sign_violation`. Rung-4a's TGV2D conservation rule schema (`harness:mass_conservation_defect`, `harness:energy_drift`, `harness:dissipation_sign_violation`) runs unmodified on dam-break-2D rollouts; per-stack rows are emitted in the same v1.0 SARIF schema as rung-4a, with `dam2d → "open-driven-dissipative"` reclassified empirically (KE(t) measured to rise during gravity-loaded fall) following the *classify when you exercise* discipline that rung-4b's PH-SYM-003 PBC-square-SO(2) substrate-incompatibility SKIP precedented."*

This sentence is the writeup's lede and is frozen at design time to prevent narrative drift during implementation. The headline is robust under all expected probe outcomes:
- `mass_conservation_defect`: 0.0 trivially on both stacks (LB SPH preserves particle count) — same shape as rung-4a's TGV2D mass row, *as expected*; the schema uniformity is the load-bearing observation, not the numeric value.
- `energy_drift`: SKIP via existing D0-08 (KE-rest IC, since dam-break starts at rest before column release) — exercises an *existing* SKIP gate on a new substrate, not a new mechanism.
- `dissipation_sign_violation`: SKIP via *new* D0-22 (open-driven systems have non-monotone KE by physics, so the rule's strictly-dissipative-or-conservative assumption doesn't apply) — the substantive new methodology contribution.

**Pre-flight gating-condition (signpost):** the headline is contingent on the pre-flight 1-traj smoke (§3.3 step 4) confirming that LB's dam-break-2D rollouts exhibit a rise-then-fall KE(t) shape (gravity-loaded PE→KE→dissipation). If the smoke contradicts this — e.g., LB's dam2d implementation includes a pre-equilibration phase that masks the gravitational PE→KE conversion, or KE(t) is monotone-non-increasing for some other LB-pipeline reason — then D0-22's applicability to dam2d is invalidated, the rung does not proceed, and the design returns to brainstorm. The empirical-classification discipline (§3.2) is precisely what gates this; the rise-then-fall prediction is strong physics-intuition but not certain. The headline is *conditional* on the smoke confirming the prediction.

### 1.3 What rung 4c is NOT (explicit deferral list, signposted in the writeup body)

1. **Not a bilateral test of D0-18's mechanism.** D0-18 (dissipative-by-design SKIP on `energy_drift`) requires `system_class == "dissipative"` AND `KE(t)` monotone-non-increasing. Dam-break post-D0-22 has `system_class == "open-driven-dissipative"` (not equal), so D0-18 does not fire on dam-break — D0-08 (KE-rest IC) fires instead. The bilateral D0-18 forward-flag from rung-4a §1.3 (5) — requiring a *strictly conservative* substrate where `energy_drift` evaluates raw_value normally — stays intact and unfulfilled, pointing at case study 02 (PhysicsNeMo MGN incompressible NS as a candidate conservative anchor) or a future case study. Rung 4c does not collapse this signpost.

2. **Not a SEGNN-vs-GNS model comparison.** Both stacks emit identical structural rows: `mass_conservation_defect = 0.000e+00 (x20 identical)`, `energy_drift = SKIP D0-08 (x20 identical)`, `dissipation_sign_violation = SKIP D0-22 (x20 identical)`. Model differentiation lives in equivariance → rung 4b (already landed). The cross-stack uniformity here is the *load-bearing evidence* that the harness's rule schema runs unmodified across architectures on a second substrate class, not a probe of model differences.

3. **Not the integrating top-level README.** Composed downstream after rung 4c lands; rung 4c writeup is a dated deliverable under `methodology/docs/`, parallel to rung-4a/4b.

4. **Not a wall-non-penetration claim.** Plan v2 §3.1 P1's "PH-BC (wall)" entry assumed an SPH-particle-wall rule that does not exist in physics-lint v1.0 (PH-BC-001 in production is Dirichlet boundary trace on a unit square, a mesh-FEM rule). Plan v2 → v2.1 amendment removes "PH-BC (wall)" from §3.1; rung 4c emits no wall-non-penetration row. Wall-non-penetration as a future SPH-particle rule (PH-BC-NNN with `particle_type == WALL` flag) is post-visa work, not bundled here.

5. **Not a multi-rung renderer.** Rung 4c stands alone — `render_cross_stack_table.py` reads the two dam-break SARIFs and emits a dam-break-only cross-stack table, mirroring rung-4a's pattern exactly. Cross-rung composition (rung-4a TGV2D + rung-4c dam-break + rung-4b equivariance integrated into one artifact) is the integrating-README's job, not rung 4c's. The renderer per-rung pattern from rung-4a/4b continues; the integrating-README is the named-event trigger that composes.

6. **Not a catalogue-wide reclassification.** Only `dam2d` is reclassified empirically. `rpf2d`, `ldc2d`, `rpf3d`, `ldc3d`, `tgv3d` retain their pre-D0-22 `"dissipative"` labels even though physics intuition strongly suggests rpf/ldc are also forced/open flows (almost certainly misclassified) and tgv3d inherits 2D-TGV physics (likely correctly classified but unverified). The *classify when you exercise* discipline (§7.5) is precisely the principle that rules out preemptive reclassification without empirical probing.

7. **Not a multi-rule-trigger-axis abstraction.** D0-21 §forward-flag-2 named the "(rule, substrate) compatibility matrix" as a future generalization. Rung 4c demonstrates that pattern's empirical instance — dissipation_sign_violation × open-driven-dissipative becomes a new SKIP cell — without yet promoting the matrix to a first-class rule-schema field. Promotion is post-rung-4c work; D0-22 cites D0-21 §forward-flag-2 to keep the line of evidence visible.

### 1.4 D-entries 4c creates

- **D0-22** — *Rung 4c pre-registration: `open-driven-dissipative` substrate class introduced into `LAGRANGEBENCH_DATASET_SYSTEM_CLASS` (taxonomy bumped from implicit-binary to tri-state); `dam2d` reclassified empirically from `"dissipative"` to `"open-driven-dissipative"` based on post-rollout KE(t) shape; new SKIP path on `dissipation_sign_violation` gated on `system_class == "open-driven-dissipative"`; "classify when you exercise" empirical-classification principle named as a first-class methodology output of the rung-4 series; (rule, substrate) compatibility matrix forward-flag from D0-21 cited as the abstraction this exercises but does not yet promote; source-review-catches-issue-before-compute pattern (now bilateral across rung-4b math + figure-sweep + rung-4c catalogue-misclassification) recorded for elevation in integrating-README.*

D0-22 lands as a single composite entry, mirroring D0-21's shape. Subsidiary amendments are appended under D0-22 footers as implementation surfaces them.

---

## 2. Architecture

### 2.1 Cross-subtree split

```
physics-lint repo (branch: feature/rung-4c-substrate-class-extension)
└── external_validation/_rollout_anchors/
    ├── _harness/
    │   ├── particle_rollout_adapter.py    [EDIT: LAGRANGEBENCH_DATASET_SYSTEM_CLASS["dam2d"] flipped to "open-driven-dissipative"; new D0-22 SKIP path in dissipation_sign_violation()]
    │   ├── SCHEMA.md                       [EDIT: §3.x SKIP-reason template for D0-22]
    │   └── tests/
    │       ├── test_d0_22_open_driven_skip.py    [NEW: positive-path + negative-path + paired-regeneration synthetic fixture]
    │       └── test_d0_18_dissipative_skip.py    [EDIT: dam2d expectation flips from "dissipative" to "open-driven-dissipative"]
    ├── 01-lagrangebench/
    │   ├── modal_app.py                    [EDIT: LAGRANGEBENCH_DATASET_DIRS["dam_2d"]=<discovered_via_LB>; new lagrangebench_rollout_p1_segnn_dam2d, _p1_gns_dam2d functions; rollout_p1_{segnn,gns}_dam2d CLI entrypoints]
    │   ├── emit_sarif.py                   [EDIT: extend driver to emit segnn_dam2d_<sha>.sarif and gns_dam2d_<sha>.sarif]
    │   ├── outputs/sarif/
    │   │   ├── segnn_dam2d_<sarif_emission_sha>.sarif   [NEW: committed]
    │   │   └── gns_dam2d_<sarif_emission_sha>.sarif     [NEW: committed]
    │   └── tests/                          [EDIT: regression coverage for dam2d emission]
    └── methodology/
        ├── DECISIONS.md                    [APPEND: D0-22 + plan-v2.1 cross-ref]
        ├── docs/
        │   ├── 2026-05-07-rung-4c-substrate-class-extension-design.md   [THIS DOC]
        │   ├── 2026-05-07-rung-4c-substrate-class-extension-plan.md     [next deliverable]
        │   ├── 2026-05-07-rung-4c-substrate-class-extension-table.md    [post-execution]
        │   └── physics-lint-validation-plan-v2.1.md                     [v2.1 amendment, separate doc; v2 stays frozen]
        └── tools/
            └── render_cross_stack_table.py [UNCHANGED: dam-break SARIFs handled identically to TGV2D — load-bearing evidence of substrate-agnosticism]
```

**Property:** rung 4c does NOT modify the renderer or the SARIF schema. The methodology-layer surface area is bounded to (taxonomy entry + new SKIP path + 2 modal-app rollout functions + 1 emit_sarif extension). All downstream consumer code stays schema-compatible. The renderer-stays-untouched property is itself a load-bearing observation in the writeup body — the cross-substrate generalization works because the schema generalizes, not because of consumer-side accommodation.

### 2.2 Reuse from rung 4a / 4b (unchanged)

- `_harness/sarif_emitter.py` — emits dam-break SARIFs in the same v1.0 schema as TGV2D conservation; no schema bump.
- `_harness/lint_npz_dir.py` — generic npz-dir → HarnessResult pipeline; reads dam2d npzs the same way it reads tgv2d npzs.
- `01-lagrangebench/emit_sarif.py` — case-study driver pattern; minimal extension to add dam2d emission alongside existing tgv2d.
- `methodology/tools/render_cross_stack_table.py` (v1.0 EXPECTED_SCHEMA_VERSION; D0-19/D0-20 enforcement) — handles dam-break SARIFs without modification because (a) the three rule_ids are unchanged, (b) the SKIP-row-invariant assertion (presence of `properties.skip_reason`, identity within (rule, stack)) holds for D0-22's new SKIP rows the same way it holds for D0-18/D0-08 SKIPs, (c) the row aggregation (`value (xN identical)` vs `min=, max=, n=`) is value-content-agnostic.
- D0-19 three-stage sha provenance (`pkl_inference`, `npz_conversion`, `sarif_emission`) — extended trivially to dam2d artifacts; same provenance fields, same audit-trail shape.
- D0-20 generator-vs-consumer separation — extended to dam2d: Modal generates dam2d npzs; consumer (`lint_npz_dir`) reads them and emits SARIF; renderer reads the SARIF.
- Rung-4b "classify when you exercise" precedent — explicit empirical-classification discipline applied to the substrate-class question, mirroring how PH-SYM-003 PBC-square-SO(2) was classified only on the substrate measured.

### 2.3 New code surface

1. **`particle_rollout_adapter.py` edit (D0-22 implementation):**
   - `LAGRANGEBENCH_DATASET_SYSTEM_CLASS["dam2d"] = "open-driven-dissipative"` (was `"dissipative"`).
   - In `dissipation_sign_violation()`: after the existing D0-08 KE-rest gate, add a new gate before the raw computation:
     ```python
     dataset_name = rollout.metadata.get("dataset", "") if rollout.metadata else ""
     system_class = LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get(dataset_name)
     if system_class == "open-driven-dissipative":
         return HarnessDefect(
             value=None,
             skip_reason=(
                 f"system_class='open-driven-dissipative' (dataset={dataset_name!r}); "
                 "dE/dt > 0 over a stretch by physics (gravitational PE → KE conversion); "
                 "the strictly-dissipative-or-conservative assumption underpinning "
                 "dissipation_sign_violation does not apply. See DECISIONS.md D0-22."
             ),
         )
     ```
   - Existing D0-18 SKIP path on `energy_drift` is unchanged; the AND-gate `system_class == "dissipative" and is_monotone_decreasing` correctly fails on dam-break (system_class doesn't match), so D0-08 KE-rest fires instead due to start-at-rest IC.

2. **`SCHEMA.md` edit:** §3.x adds a new SKIP-reason template for `dissipation_sign_violation` with the open-driven-dissipative class; cross-references D0-22.

3. **New tests in `_harness/tests/test_d0_22_open_driven_skip.py`:**
   - **Positive path:** synthetic `ParticleRollout` fixture with metadata `{"dataset": "dam2d"}` and a non-monotone KE(t) profile (rise-then-fall) → assert `dissipation_sign_violation` returns SKIP with the D0-22 reason string verbatim.
   - **Negative path:** synthetic `ParticleRollout` fixture with metadata `{"dataset": "tgv2d"}` (or any non-open-driven) and a non-monotone KE(t) profile → assert `dissipation_sign_violation` returns a raw value (no SKIP) — proves the gate is `system_class`-conditioned, not KE-shape-conditioned.
   - **Paired-regeneration:** the synthetic fixture is hand-crafted (not copied from a Modal rollout), per `feedback_test_fixtures_hand_crafted_not_copied.md`; expected outputs (SKIP reason string + raw value on the negative path) are paired with the fixture and regenerated together when either changes.
   - **Negative-path-fail-loud assertion:** assert that a `system_class` value that is neither `"dissipative"` nor `"open-driven-dissipative"` falls through to raw computation rather than mis-classifying — guards against future taxonomy entries with typos or absent class strings.

4. **`test_d0_18_dissipative_skip.py` edit:** the existing test asserts `LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get("dam2d") == "dissipative"` (line 108 surveyed pre-design); update to assert `"open-driven-dissipative"` post-D0-22. This is a schema-truth-source-of-truth flip, not a logic change in D0-18 itself.

5. **`modal_app.py` edits:**
   - `LAGRANGEBENCH_DATASET_DIRS["dam_2d"] = "<discovered_via_LB_inspection>"` — actual upstream dataset directory name discovered via Modal-side `bash download_data.sh dam2d` smoke (parallel to how `tgv_2d` was discovered).
   - New function `lagrangebench_rollout_p1_segnn_dam2d` mirroring `lagrangebench_rollout_p1_gns_tgv2d` shape, with the differences enumerated as inline comments (`ckpt_root`, `zip_path`, `LAGRANGEBENCH_CHECKPOINT_GDOWN_IDS["segnn_dam2d"]`, `rollout_subdir`, `RolloutMetadata.dataset="dam2d"`, `RolloutMetadata.model="segnn"`).
   - New function `lagrangebench_rollout_p1_gns_dam2d` mirroring the same with model="gns".
   - Both functions reuse the same `rollout_image`, `ROLLOUT_GENERATION_GPU_CLASS = "A10G"`, `rollout_volume`, `lagrangebench_pkl_to_npz` conversion, `UPSTREAM_COMPAT_PATCHES`-tracked CLI args.
   - New CLI entrypoints `rollout_p1_segnn_dam2d` and `rollout_p1_gns_dam2d` parallel to `rollout_p0_segnn_tgv2d` / `rollout_p1_gns_tgv2d`.

6. **`emit_sarif.py` edit:** extend the driver's case-pair iteration to include `("segnn_dam2d", "/vol/rollouts/lagrangebench/segnn_dam2d_<sha>/")` and `("gns_dam2d", "/vol/rollouts/lagrangebench/gns_dam2d_<sha>/")` alongside the existing TGV2D pair. Output filename pattern `<model>_<dataset>_<sarif_emission_sha>.sarif` already supports the dataset axis; no driver-shape change needed.

7. **2 new committed SARIFs at `01-lagrangebench/outputs/sarif/`:**
   - `segnn_dam2d_<sarif_emission_sha>.sarif`
   - `gns_dam2d_<sarif_emission_sha>.sarif`
   - Both schema_version v1.0 (no bump); 3-sha provenance per D0-19; dataset metadata field carries `"dam2d"` at the npz level and propagates through.

8. **Plan amendment doc `methodology/docs/physics-lint-validation-plan-v2.1.md`:** separate doc, v2 stays frozen; v2.1 carries the §3.1 P1 row update (substrate-class-extension framing), §3.2 step-6 PH-BC drop, §5.3 cover-letter update, §6 risks register entry, changelog appendix with source-review-correction acknowledgment paralleling rung-4b amendment 2 §14.6.

---

## 3. Detailed design

### 3.1 Substrate-class taxonomy extension

The taxonomy is currently implicit-binary in `LAGRANGEBENCH_DATASET_SYSTEM_CLASS`: a string value `"dissipative"` (TGV-class) or absence (default fire-raw). D0-22 makes it tri-state at minimum:

| Class label | Meaning | Triggers |
|---|---|---|
| `"dissipative"` | Closed system, KE-only dissipation, monotone-non-increasing KE expected | D0-18 SKIP on `energy_drift` (when KE(t) also monotone) |
| `"open-driven-dissipative"` | External work source (gravity / forced flow); KE non-monotone by physics | D0-22 SKIP on `dissipation_sign_violation` |
| absent / other | Default fire-raw | Neither D0-18 nor D0-22 SKIPs fire (D0-08 KE-rest still applies orthogonally) |

D0-08 (KE-rest IC SKIP on `energy_drift`) is **orthogonal to the taxonomy** — it gates on `KE(0) < KE_REST_THRESHOLD`, independent of `system_class`. On dam-break-2D, D0-08 fires because the IC is a fluid column at rest before release; this is the SKIP path that fires on `energy_drift` for dam-break (not D0-18). Documented as a forward-flag in §7.6: D0-22 does not amend or absorb D0-08; the two SKIP gates are sibling members of a "substrate-detection family" rather than a single gate.

### 3.2 Empirical reclassification of `dam2d`

`dam2d` is currently mapped to `"dissipative"` (line 255 of `particle_rollout_adapter.py` pre-D0-22), set during the D0-18 design pass when dam-break had not yet been measured. The mapping is preemptive and turns out to be wrong: dam-break-2D's KE rises during the gravity-loaded fall (PE → KE conversion), violating the closed-dissipative class's monotone-non-increasing precondition. Rung 4c flips this empirically — *after* measuring KE(t) on the SEGNN-dam2d and GNS-dam2d 1-traj smoke rollouts (pre-flight step §3.3) and confirming the rise-then-fall shape — to `"open-driven-dissipative"`.

The flip is recorded in three artifacts:
1. **Code:** the mapping change in `particle_rollout_adapter.py`.
2. **Test:** `test_d0_18_dissipative_skip.py` updated to assert the new label.
3. **Methodology:** D0-22 entry in DECISIONS.md citing the empirical KE(t) observation as the justification, with the 1-traj smoke rollout shas as the evidence pointer.

The discipline this exercises is **classify when you exercise** (§7.5): substrate-class labels are claims-of-fact about the substrate, not pre-registrations of intent; the claim should land only after the substrate has been measured. Pre-D0-22 the `dam2d → "dissipative"` mapping was a reasonable physics-intuition default that turned out to be wrong; the correction lives at code-edit time, not at brainstorm time.

### 3.3 Pre-flight discipline (CLAUDE.md 5-step checklist, adapted)

Per the global pre-flight checklist that gates GPU runs >5 min, and reinforced by rung-4b's three-instance source-review-catches-issue-before-compute pattern:

1. **Data inspection (5 min, CPU-only).** Run `bash download_data.sh dam2d` on Modal in a 1-min CPU-only function (or in the existing rung-2 image's `lagrangebench_smoke` analogue); print dataset stats from upstream `metadata.json`: particle count, fluid/wall type counts, dt, domain box, IC velocity stats. Verify start-at-rest (mean velocity ≈ 0 at t=0). Discover the actual upstream dataset directory name and populate `LAGRANGEBENCH_DATASET_DIRS["dam_2d"]` accordingly.

2. **Conversion round-trip (2 min, CPU-only).** Verify `lagrangebench_pkl_to_npz` round-trips on a synthetic dam-break-shaped pkl fixture (hand-crafted, not copied from a Modal rollout, per the test-fixtures-hand-crafted-not-copied discipline) before any Modal fire. The conversion is the loader-contract surface that surfaced as the 5th failure class in rung-4b amendment 2; pre-flight assertions here have proven load-bearing.

3. **Rule sanity test (3 min, CPU-only).** Invoke `mass_conservation_defect`, `energy_drift`, `dissipation_sign_violation` on a synthetic dam-break-shaped `ParticleRollout` fixture (rise-then-fall KE profile, `metadata={"dataset": "dam2d"}`, `KE(0) ≈ 0`). Assert D0-22 predictions:
   - `mass_conservation_defect` → raw 0.0 (constant particle count + uniform mass)
   - `energy_drift` → SKIP D0-08 (KE-rest IC)
   - `dissipation_sign_violation` → SKIP D0-22 (open-driven class)
   Verifies the new SKIP path pre-Modal-fire.

4. **Reference reproduction (substituted by 1-traj Modal smoke).** LagrangeBench does not publish dam-break-2D headline numbers in the form physics-lint consumes (the LB paper reports trajectory-level position MSE / Sinkhorn divergence; physics-lint reads conservation defects). Substitute: 1-traj smoke per stack on Modal A10G (~1 min each), verify rollout completes and KE(t) shows the expected gravity-loaded rise-then-fall shape, *before* the empirical reclassification of `dam2d` is committed and *before* scaling to 20 trajs. The smoke is the empirical-justification evidence for D0-22.

5. **End-to-end pipeline smoke (1 min, CPU-only after Modal step 4).** 1-traj rollout → pkl → npz → SARIF row → renderer reads it; verify all stages don't raise before the 20-traj fire. Catches loader-contract regressions and emit-time failures at the cheapest possible point.

The 5-step checklist gates the 20-traj Modal fire. Failure at any step blocks the 20-traj fire and is recorded in a `preflight/2026-05-XX-rung-4c.txt` file in the repo per the CLAUDE.md procedural rule.

### 3.4 Pipeline + compute budget

Modal Volume layout (parallel to rung-4a):

```
/vol/rollouts/lagrangebench/segnn_dam2d_<sha>/   20 × particle_rollout_traj{NN}.npz
/vol/rollouts/lagrangebench/gns_dam2d_<sha>/     20 × particle_rollout_traj{NN}.npz
```

Compute budget — target **< $0.20 USD total** at A10G $0.86/hr ≈ 14 min budget:

| Stage | Runtime | Cumulative |
|---|---|---|
| Pre-flight smoke SEGNN-dam2d (1 traj × 100 steps) | ~1 min A10G | 1 min |
| Pre-flight smoke GNS-dam2d (1 traj × 100 steps) | ~1 min A10G | 2 min |
| SEGNN-dam2d 20-traj rollout | ~5 min A10G | 7 min |
| GNS-dam2d 20-traj rollout | ~3 min A10G | 10 min |
| SARIF emit + render | <1 min CPU | <11 min |

Margin under budget: ~3 min A10G for unexpected re-tries. Rung-4b's first-fire abort cost ~30 s; equivalent margin sufficient here.

### 3.5 Pipeline & artifact structure (data flow)

```
Modal Volume (immutable, frozen at conversion-time shas)
  /vol/rollouts/lagrangebench/segnn_dam2d_<sha>/   20 × particle_rollout_traj{NN}.npz
  /vol/rollouts/lagrangebench/gns_dam2d_<sha>/     20 × particle_rollout_traj{NN}.npz
        │
        │  modal volume get  (one-shot, ~30s per stack)
        ▼
01-lagrangebench/outputs/_local_mirror/             (gitignored cache)
        │
        │  python 01-lagrangebench/emit_sarif.py
        │    ├─ for each (model, dataset, dir) triple:
        │    │    _harness/lint_npz_dir.py:
        │    │      ├─ for each npz: load_rollout_npz (rollout.metadata["dataset"] = "dam2d"),
        │    │      │   invoke 3 defects (mass, energy_drift, dissipation_sign_violation)
        │    │      ├─ for harness:energy_drift rows: D0-08 SKIP fires (KE(0) < threshold);
        │    │      │   ke_initial / ke_final extra_properties attached
        │    │      ├─ for harness:dissipation_sign_violation rows: D0-22 SKIP fires
        │    │      │   (system_class="open-driven-dissipative"); skip_reason attached
        │    │      └─ return list[HarnessResult]
        │    ├─ assemble run-level properties (3 shas + LB sha + 4 IDs + schema_version=1.0)
        │    └─ _harness/sarif_emitter.py:emit_sarif(results, run_properties=..., output_path=...)
        ▼
01-lagrangebench/outputs/sarif/                     (committed, ~30 KB each)
  segnn_dam2d_<sarif_emission_sha>.sarif
  gns_dam2d_<sarif_emission_sha>.sarif
        │
        │  cross-subtree boundary (no Python imports; only artifact contract per D0-20)
        ▼
methodology/tools/render_cross_stack_table.py       (UNCHANGED v1.0 schema-locked)
  ├─ load both dam-break SARIFs
  ├─ assert source == "rollout-anchor-harness" on both
  ├─ assert harness_sarif_schema_version == "1.0" (D0-19 fail-loud)
  ├─ assert all required run-level fields present
  ├─ for each (rule, stack) group: assert SKIP-row invariants per D0-19 §3.4
  │    (D0-22 SKIP rows satisfy these the same way D0-18/D0-08 SKIPs do)
  └─ render markdown table
        ▼
methodology/docs/2026-05-07-rung-4c-substrate-class-extension-table.md
  (post-execution writeup; mirrors rung-4a writeup shape)
```

**Property:** the data flow is byte-identical to rung-4a's data flow with `dam2d` substituted for `tgv2d` at the dataset-name field; the only structural change is the SKIP rule_id distribution (D0-22 SKIP on dissipation_sign_violation rather than raw=0.0). This is the load-bearing observation that the schema generalizes.

---

## 4. Schema and provenance

### 4.1 No schema bump

Rung-4c stays at `harness_sarif_schema_version = "1.0"`, the rung-4a baseline. The new D0-22 SKIP rows on `dissipation_sign_violation` use the existing SKIP-row machinery (`HarnessDefect(value=None, skip_reason="...")` propagated through `lint_npz_dir.py:140` to `sarif_emitter.py`, emitted as `properties.skip_reason` per D0-19 §3.4). The renderer's `EXPECTED_SCHEMA_VERSION = "1.0"` assertion passes unchanged. Rung-4b's schema_version v1.1 bump was eps-SARIF-specific (PH-SYM extra_properties); rung-4c does not touch eps SARIFs.

### 4.2 3-stage sha provenance per D0-19

Each dam-break SARIF carries 3 shas at the run-level properties:
- `pkl_inference_sha`: physics-lint sha at which the LagrangeBench inference subprocess was invoked (Modal-side).
- `npz_conversion_sha`: physics-lint sha at which `lagrangebench_pkl_to_npz` was invoked (may equal pkl_inference_sha if conversion happens in the same Modal run; may differ if conversion is re-run on the volume via `lagrangebench_convert_pkls_in_volume`).
- `sarif_emission_sha`: physics-lint sha at which `emit_sarif.py` was run (CPU-only, locally or on Modal).

Filename embeds `sarif_emission_sha` for traceability (parallel to rung-4a's `<model>_<dataset>_<sarif_emission_sha>.sarif` pattern).

### 4.3 Metadata field propagation

`particle_rollout.metadata["dataset"] = "dam2d"` is set during `lagrangebench_pkl_to_npz` conversion (existing code path, no edit needed — the metadata schema already supports arbitrary dataset names per SCHEMA.md §1). The `dataset` field is what `dissipation_sign_violation` (and `energy_drift`) consult to dispatch on `system_class`; the lookup chain is `metadata["dataset"]` → `LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get(dataset_name)` → SKIP-or-fire decision.

---

## 5. Renderer

### 5.1 Existing renderer unchanged

`methodology/tools/render_cross_stack_table.py` is **unchanged** for rung 4c. The renderer's contract (D0-19 §3.4 SKIP-row invariant + D0-20 schema-version fail-loud assertion + 3-rule-id iteration over `harness:mass_conservation_defect`, `harness:energy_drift`, `harness:dissipation_sign_violation`) handles dam-break SARIFs identically to TGV2D SARIFs because:

- The 3 rule_ids are unchanged; the SKIP-row distribution shifts from `{energy_drift: SKIP D0-18, mass: raw, dissipation_sign: raw}` (TGV2D) to `{energy_drift: SKIP D0-08, mass: raw, dissipation_sign: SKIP D0-22}` (dam-break) but the row count and structure are identical.
- The SKIP-row invariant — presence of `properties.skip_reason`, identity of `skip_reason` and `message.text` within (rule, stack) — holds for D0-22 SKIPs the same way it holds for D0-18 SKIPs because both are emitted via the same `lint_npz_dir.py:140` path that attaches `skip_reason` as a result-level property.
- The aggregation logic (`value (xN identical)` for raw rows, `n=, min=, max=` for non-uniform rows) is value-content-agnostic.

The renderer-stays-untouched property is the **load-bearing evidence** in the writeup body that the harness's substrate-detection layer extends cleanly: no consumer-side accommodation is needed for the new substrate class. The methodology contribution is bounded to the generator side (`particle_rollout_adapter.py`).

### 5.2 Cross-rung composition deferred to integrating-README

Rung-4c's renderer output is a dam-break-only cross-stack table, mirroring rung-4a's TGV2D-only cross-stack table and rung-4b's eps-only tripartite table. The cross-rung composition (rung-4a TGV2D + rung-4c dam-break + rung-4b equivariance integrated into one cross-substrate methodology artifact) is the integrating-README's job, not rung-4c's. Per rung-4a §1.3 (3) and rung-4b §1.3 (3), the integrating-README is the named-event trigger that composes dated deliverables; rung 4c's landing is one of the gating events, not the integrating-composition itself.

### 5.3 Sibling-renderer forward-flag (n=2 case)

Rung-4b's design §5.1 observed that at n=2 (cross-stack-conservation renderer + eps renderer), separate-sibling-renderers is the right pattern — the schema differences (v1.0 conservation vs v1.1 eps) make extension fragile and the duplication is low. Rung-4c's choice to reuse the rung-4a renderer (rather than write a third sibling) is the dual case: when the schema is unchanged, the renderer is reused; when the schema bumps, a sibling is written. The pattern stays consistent.

D0-21 §forward-flag-7 (sibling-vs-extend at n=3+) carries forward unchanged: when a third schema version (v1.2 etc.) arrives, extract shared formatting primitives into `methodology/tools/render_lib.py` and have version-specific renderers compose them. Rung-4c does not trigger this forward-flag (n stays at 2 schema versions; rung-4c reuses v1.0).

---

## 6. Test fixtures

### 6.1 D0-22 SKIP-path tests (`test_d0_22_open_driven_skip.py`)

All fixtures hand-crafted, not copied from production rollouts (per `feedback_test_fixtures_hand_crafted_not_copied.md`). Synthetic-but-realistic shapes pin the test to the schema contract, not to a specific run's incidentals.

**Positive-path fixture: `synthetic_open_driven_rise_then_fall.py`**
- Shape: 4 particles, T=10 timesteps, D=2 dimensions
- KE(t) profile: rises monotonically from t=0 to t=4 (PE→KE conversion phase), peaks at t=4, decays to t=9 (dissipation phase) — physics-shaped, not arbitrary noise
- `metadata = {"dataset": "dam2d", ...}` triggers `system_class == "open-driven-dissipative"` lookup
- `KE(0) > KE_REST_THRESHOLD` to avoid D0-08 KE-rest gate firing first (so the test isolates D0-22 specifically)
- Expected: `dissipation_sign_violation(rollout)` returns `HarnessDefect(value=None, skip_reason="system_class='open-driven-dissipative' (dataset='dam2d'); ...")` with the verbatim D0-22 reason template

**Negative-path-A fixture: same KE shape, different dataset**
- Identical KE profile (rise-then-fall) but `metadata = {"dataset": "tgv2d"}` (or any non-open-driven label)
- Expected: `dissipation_sign_violation(rollout)` returns a raw value (no SKIP). Proves the gate is `system_class`-conditioned, not KE-shape-conditioned.

**Negative-path-B fixture: open-driven dataset with monotone-decreasing KE**
- `metadata = {"dataset": "dam2d"}` but KE monotone-decreasing
- Expected: SKIP D0-22 fires regardless of KE shape (the gate is purely `system_class`-conditioned). Verifies the implementation does not accidentally co-condition on KE shape.

**Fail-loud-fixture: typo'd substrate label**
- `metadata = {"dataset": "dam2d_typo"}` → `LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get("dam2d_typo") is None`
- Expected: falls through to raw computation, no SKIP. Guards against future taxonomy entries with typos creating silent fall-through behavior.

### 6.2 Paired-regeneration discipline

Each fixture is paired with its expected output (SKIP reason string + expected raw values for the negative paths). When the fixture changes, the expected output regenerates in the same edit; when the D0-22 reason template changes, the fixture regenerates. The pairing prevents fixture-vs-expected-output drift, the failure mode that test_fixtures_hand_crafted_not_copied was written to prevent.

### 6.3 D0-18 test edit (`test_d0_18_dissipative_skip.py`)

Existing test at line 108 asserts `LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get("dam2d") == "dissipative"`. Update to `== "open-driven-dissipative"` post-D0-22. The test's purpose is to pin the substrate-class taxonomy as a source-of-truth claim; the value flip is a schema-evolution edit, not a test-logic edit.

---

## 7. Forward flags + honest limits captured in D0-22

1. **Bilateral D0-18 still requires conservative-system anchor.** Rung-4a §1.3 (5) flagged this; rung-4c does not collapse it. D0-08 (KE-rest IC) fires on dam-break, not D0-18. The "energy_drift evaluates raw_value normally on a strictly conservative substrate" path is unexercised across rung-4a, rung-4b, rung-4c. Case study 02 (PhysicsNeMo MGN incompressible NS as a candidate) or a future LB substrate is the path. D0-22 cites the outstanding flag.

2. **Catalogue-wide reclassification deferred.** `rpf2d`, `ldc2d`, `rpf3d`, `ldc3d`, `tgv3d` retain pre-D0-22 `"dissipative"` labels. Two-tier split:
   - **Almost certainly misclassified, awaiting empirical probe:** `rpf2d` (reverse Poiseuille, forced flow), `ldc2d` (lid-driven cavity, forced flow), `rpf3d` and `ldc3d` (3D variants).
   - **Likely correctly classified but unverified:** `tgv3d` (3D-TGV inherits 2D-TGV physics; closed dissipative system, KE monotone-non-increasing as turbulent KE decays).
   The two-tier split is preserved for future-rung-actionability: a rung exercising rpf or ldc walks into a known-misclassification (and the empirical probe is its first move); a rung exercising tgv3d treats it as inherited-from-validated.

3. **D0-08 not absorbed into D0-22.** The two SKIP gates are sibling members of a "substrate-detection family," not a single gate. D0-08 (KE-rest IC) is orthogonal to the taxonomy (gates on `KE(0)`, not `system_class`); D0-22 (open-driven) is taxonomy-dispatched. Future generalization (the (rule, substrate) compatibility matrix from D0-21 §forward-flag-2) may unify the family at the rule-schema layer, but D0-22 does not preempt that work.

4. **PH-SYM substrate-symmetry-SKIP for rung 4d.** Gravity-direction-pinning on dam-break breaks SO(2) symmetry — rotating the IC by π/2 produces a physically different setup (gravity now points sideways relative to the fluid column). PH-SYM-001/002/003 rules on dam-break should SKIP under a substrate-symmetry-incompatibility mechanism analogous to rung-4b's PBC-square-SO(2) SKIP. Forward-flagged for rung 4d; not in scope for rung 4c. The bilateral-substrate-symmetry-SKIP validation (TGV-2D PBC-square + dam-break-2D gravity-direction) is the rung-4d analogue of rung-4c's bilateral-substrate-class-extension scope.

5. **"Classify when you exercise" empirical-classification principle.** Now bilateral across the rung-4 series:
   - Rung 4b: PH-SYM-003 PBC-square-SO(2) SKIP classified only for the substrate measured (TGV-2D under PBC-square), not retrospectively across all (rule, substrate) combinations.
   - Rung 4c: `dam2d` reclassified empirically after KE(t) inspection; `rpf2d`/`ldc2d`/`rpf3d`/`ldc3d`/`tgv3d` forward-flagged but not reclassified.
   Pattern reads as: **substrate properties get verdicts only after empirical probing, never on theoretical intuition alone, even when the theoretical guess is almost certainly correct.** Generalizes to case study 02 (different framework, different substrates, same discipline). To be elevated as a first-class methodology output of the rung-4 series in integrating-README composition.

6. **Source-review-catches-issue-before-compute pattern (now trilateral).** Three instances in the rung-4 series:
   - Rung 4b first-pass math correction (TRAIN_PUSHFORWARD_UNROLLS_LAST=3 conflated +1 target with pushforward count) — caught at source review of LB at sha b880a6c between first-pass fix and re-fire.
   - Rung 4b first-pass latent figure-sweep failure (valid.h5 hardcoded subseq_length=10 vs. dynamic upstream value) — caught at the same source review pass.
   - Rung 4c catalogue-misclassification (`dam2d → "dissipative"` was preemptive and wrong; `rpf2d`/`ldc2d` likely the same) — caught at source review of `particle_rollout_adapter.py` during this design pass.
   Three instances at $0 Modal cost. Worth elevating to a first-class methodology contribution of the rung-4 series in integrating-README composition rather than three scattered amendments. The pattern reads as: **a source-review pre-flight pass between design and execution catches issues that brainstorm-only and execution-only review miss; the cost is hours, the saving is multiple GPU runs and methodology errors that would otherwise land in writeups.**

7. **Plan v2.1 amendment as a separate doc.** v2.1 lives at `methodology/docs/physics-lint-validation-plan-v2.1.md`, parallel to v2; v2 stays frozen as the original. Mirrors rung-4b amendment 2's appended-section pattern (preserve the original framing, layer the correction on top) rather than in-place rewriting. Diff is auditable as a complete document; reviewers reading v2.1 see both the correction and what was corrected.

---

## 8. Acceptance criteria

Rung 4c is considered passed when all of the following hold:

- [ ] D0-22 committed in `methodology/DECISIONS.md` before any code change (pre-registration discipline; mirrors rung-4a/4b D-entry sequencing).
- [ ] `LAGRANGEBENCH_DATASET_SYSTEM_CLASS["dam2d"]` flipped to `"open-driven-dissipative"` in `_harness/particle_rollout_adapter.py`; new D0-22 SKIP gate added in `dissipation_sign_violation()`.
- [ ] `_harness/SCHEMA.md` §3.x documents the new SKIP-reason template for D0-22.
- [ ] `_harness/tests/test_d0_22_open_driven_skip.py` exists with positive-path + negative-path-A + negative-path-B + fail-loud fixtures, all passing.
- [ ] `_harness/tests/test_d0_18_dissipative_skip.py` updated for the dam2d expectation flip; passing.
- [ ] `01-lagrangebench/modal_app.py` extended: `LAGRANGEBENCH_DATASET_DIRS["dam_2d"]` populated from upstream inspection; `lagrangebench_rollout_p1_segnn_dam2d` and `_p1_gns_dam2d` functions present; CLI entrypoints `rollout_p1_segnn_dam2d` and `rollout_p1_gns_dam2d` defined.
- [ ] `01-lagrangebench/emit_sarif.py` extended to emit `segnn_dam2d_<sha>.sarif` and `gns_dam2d_<sha>.sarif` alongside existing TGV2D pair.
- [ ] `01-lagrangebench/outputs/sarif/segnn_dam2d_<sha>.sarif` and `gns_dam2d_<sha>.sarif` committed; both schema_version v1.0; 3-sha provenance per D0-19; full row set (3 rules × 20 trajs each).
- [ ] Both dam-break SARIFs structurally identical row-by-row in (rule_id, traj_index): same set of rows present, identical `skip_reason` strings within (rule, stack), per D0-19 §3.4 invariant.
- [ ] D0-22 SKIP rows on `dissipation_sign_violation` carry the verbatim D0-22 reason template; D0-08 SKIP rows on `energy_drift` carry the verbatim D0-08 reason template; both pass renderer's fail-loud assertions.
- [ ] `methodology/tools/render_cross_stack_table.py` produces a clean dam-break cross-stack table without modification (load-bearing evidence of substrate-agnosticism).
- [ ] `methodology/docs/2026-05-07-rung-4c-substrate-class-extension-table.md` writeup committed with frozen headline (§1.2), three-rule cross-stack table, deferral list (§1.3 verbatim), bilateral-D0-18 forward-flag preserved, "classify when you exercise" + source-review-catches-issue methodology elevation in honest-limits.
- [ ] `methodology/docs/physics-lint-validation-plan-v2.1.md` committed with §3.1 P1 row update + §3.2 step-6 PH-BC drop + §5.3 cover-letter update + §6 risks register entry + changelog appendix.
- [ ] Total Modal compute < $0.50 (target < $0.20).
- [ ] All existing tests still PASS (pytest --import-mode=importlib external_validation/ -o "addopts=").

The headline (§1.2) is true under the expected probe outcomes (mass=0.0 trivial, energy_drift SKIP D0-08, dissipation_sign_violation SKIP D0-22 — all on both stacks); no writeup wording is determined by empirical result beyond the empirical-justification paragraph for the `dam2d` reclassification.

---

## 9. Predecessor → successor → next deliverable

- **Predecessor:** rung 4b (`2026-05-07-rung-4b-equivariance-table.md`) — cross-stack equivariance result; D0-21 + amendment 2 on PR #8 in flight at sha `255af5de8d`.
- **This document:** rung 4c design — pre-registers D0-22.
- **Next:** `2026-05-07-rung-4c-substrate-class-extension-plan.md` — implementation plan with TDD-discipline task breakdown, derived from §§3, 6 above. Generated via the writing-plans skill.
- **Then:** rung 4c execution — pre-flight 5-step checklist (§3.3) → empirical reclassification + D0-22 land → 20-traj rollouts → emit SARIFs → render table → write writeup → plan v2.1 amendment.
- **Then:** `2026-05-07-rung-4c-substrate-class-extension-table.md` — post-execution writeup.
- **Then (separate, named-event-triggered):** integrating-README composition at `methodology/README.md`. Composes rung-4a + rung-4b + rung-4c writeups into one cross-rung methodology artifact; foregrounds (a) "classify when you exercise" empirical-classification principle, (b) source-review-catches-issue-before-compute pattern (now trilateral), (c) rung-4b §3.2 architecture-claim coupling, (d) rung-4b obs (4) GNS-translation FP-noise-bounded by LB feature pipeline. Gating event: rung 4c writeup commit.

---

## 10. Plan v2 → v2.1 amendment (separate deliverable, same PR)

Lands as `methodology/docs/physics-lint-validation-plan-v2.1.md`. v2 stays frozen at its current path; v2.1 is the corrected document with a changelog appendix capturing the diff.

Changes:

- **§3.1 P1 row** updated:
  ```
  Before: | P1 | Dam break 2D | GNS | PH-CON-001 (mass), PH-BC (wall) |
  After:  | P1 | Dam break 2D | GNS + SEGNN | Substrate-class extension to open-driven-dissipative
                                              (D0-22): PH-CON-001 mass ACTIVE +
                                              dissipation_sign_violation SKIP (new) +
                                              energy_drift SKIP (D0-08 KE-rest, existing) |
  ```
  Architecture column updated `GNS → GNS + SEGNN` to reflect the dual-stack scope; headline-rule column updated to substrate-class-extension framing.

- **§3.1 P3 row absorption** noted in changelog: plan v2 §3.1 P3 row was `| P3 | Dam break 2D | SEGNN | Cross-validate P1 result |`. P1's expansion to dual-stack scope absorbs P3's SEGNN-dam2d cross-validation goal (the cross-stack uniformity *is* the cross-validation in the substrate-class-extension framing). v2.1 strikes the P3 row and notes the absorption explicitly.

- **§3.2 step 6 (subsection writeup template)** drops the `PH-BC-001` row from the dam-break section. PH-BC-particle-wall as a future SPH-particle rule is post-visa work, deferred separately.

- **§5.3 cover-letter paragraph** drops "PH-BC" reference; picks up substrate-class-extension framing as the dam-break headline contribution.

- **§6 risks register** adds:
  > *Plan-vs-actual-rule mismatch surfaced during implementation.* Plan v2 §3.1 P1 specified "PH-BC (wall)" for dam-break-2D, but no SPH-particle wall rule existed in physics-lint v1.0 (PH-BC-001 in production is Dirichlet boundary trace on a unit square, a mesh-FEM rule). The mismatch was caught at rung-4c design pass via source review of the production rule set, before any implementation. Plan v2.1 corrects the row honestly. Pattern is the third instance of source-review-catches-issue-before-compute in the rung-4 series.

- **New §11 changelog appendix** with the v2 → v2.1 diff entries above + a source-review-correction acknowledgment paragraph paralleling rung-4b amendment 2 §14.6.

The plan v2.1 amendment is a separate small deliverable in the same PR as the rung-4c design doc + implementation, for atomicity. Reviewers reading the PR see the correction, the design that prompted it, and the implementation that satisfies it as one bundle.
