# Rung 4a — Cross-stack conservation SARIF + writeup table (design)

**Date:** 2026-05-04
**Branch (primary):** `physics-lint/feature/rollout-anchors`
**Branch (secondary, one commit only):** `physics-lint/master`
**Status:** Design — pre-implementation. Pre-registers D0-19 + D0-20 in `external_validation/_rollout_anchors/methodology/DECISIONS.md` (post-migration location) before any code change.
**Predecessor:** rung 3.5 PASS on both stacks (D0-18 amendment 1 implementation at `d03df3e`); npzs frozen on Modal Volume.
**Successor:** rung 4b — equivariance brainstorm session (separate, no code).

---

## 1. Scope and framing

### 1.1 What rung 4a is

Package the existing 40 `particle_rollout_traj{NN}.npz` artifacts on Modal Volume (20 SEGNN-TGV2D + 20 GNS-TGV2D) into committed harness-style SARIF artifacts via `_harness/sarif_emitter.py` (extended for run-level properties), render a cross-stack conservation table over those artifacts, write a dated methodology writeup at `methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md`, and amend physics-lint v1.0's `master`-branch README to document PH-CON-002's dissipative-system behavior as an honest limit alongside the existing PH-BC-001 / PH-RES-001 entries.

### 1.2 Load-bearing claim (frozen headline sentence)

> *"physics-lint's harness ran the same conservation rule schema, unmodified, across SEGNN-TGV2D and GNS-TGV2D rollouts of the same dissipative system. Every result row is structurally identical between the two SARIF artifacts (D0-19-enforced); D0-18's dissipative-system skip-with-reason fires identically with the same `skip_reason` string on both — per-stack KE endpoints are recorded in dedicated `properties.ke_initial` / `properties.ke_final` fields, not interpolated into the reason — and points to `dissipation_sign_violation` as the load-bearing alternative. The methodology-evolution machinery — D0-18's skip-with-reason path — is exercised end-to-end against real upstream output."*

This sentence is the writeup's lede. It is frozen in the design to prevent narrative drift during implementation.

### 1.3 What rung 4a is NOT (explicit deferral list, signposted in the writeup body)

1. **Not a SEGNN-vs-GNS model comparison.** Both stacks emit `mass_conservation_defect = 0.0`, both fire D0-18 SKIP on `energy_drift`, both emit `dissipation_sign_violation = 0.0`. Model differentiation lives in equivariance → rung 4b (separate brainstorm, separate session).
2. **Not a GitHub Security-tab integration demo.** Harness-style SARIF emits `level: "note"` rows for PASS-equivalent values; 4a has no findings to populate the Security tab meaningfully. The Security-tab demo is deferred to 4b, where equivariance is expected to produce real warning-level findings (GNS APPROXIMATE band) that exercise the rendering path. An empty Security tab is not a demo of integration.
3. **Not the integrating top-level README.** Composed when 4b's writeup lands; until then `methodology/docs/` carries dated deliverables and is the source of truth.
4. **Not a physics-lint v1.x core change.** The skip-with-reason mechanism, dissipative-system detection, and audit-trail provenance fields all live in the harness layer (`external_validation/_rollout_anchors/_harness/`), not in physics-lint v1.0's public rule path. v1.0's `master`-branch docs are amended as part of 4a to document the dissipative-system limit explicitly alongside the existing PH-BC-001 / PH-RES-001 honest limits, with wording that includes an explicit cross-branch qualifier (the harness layer currently lives on `feature/rollout-anchors` pending merge to `master`). v1.0's behavior on dissipative systems is preserved as-shipped, with the harness-layer skip-with-reason machinery flagged as the v1.x graduation prototype. The graduation itself is a future D-entry, not implied by 4a.
5. **Not a bilateral test of D0-18's mechanism.** TGV2D is dissipative, so 4a exercises the skip-fires path. The opposite path (conservative system, skip does not fire, `energy_drift` evaluates raw_value normally) is not exercised — both 4a stacks are on the same dissipative dataset. Bilateral validation requires a conservative-system anchor (case study 02 if PhysicsNeMo includes a conservative target, or a dedicated future case study). 4a also does not exercise the borderline case — a system that *should* be conservative but is *numerically* dissipating due to a model bug, where D0-18's heuristic (dataset-name primary, KE-monotone-decreasing secondary) could mis-classify as dissipative and silently skip the very PH-CON-002 firing that would catch the bug. Diagnostic gap flagged for future case studies.

### 1.4 D-entries 4a creates

- **D0-19** — *what is in the SARIF artifact*. Pre-registers the harness SARIF schema: 3-sha provenance (pkl_inference, npz_conversion, sarif_emission), run-level vs result-level fields, guaranteed-identical / may-vary contract on result rows, schema_version field, energy_drift skip_reason template change forced by the contract, multi-stage sha-equality note (the three shas may be identical or distinct; equality is allowed but never required).
- **D0-20** — *how the SARIF artifact is consumed*. Pre-registers the generator-vs-consumer separation architecture: renderer in `methodology/tools/`, no Python imports cross-subtree, `harness_sarif_schema_version` as wire protocol, fail-loud-on-mismatch primitive, golden-fixture test surface. (Reframed from "cross-repo" since the post-migration methodology subtree lives in physics-lint alongside the harness; the discipline is intra-repo subtree separation.)
- **No separate verdict-capture entry.** Rung 4a's deliverable is a writeup; the writeup carries the rung-passed claim. Differs from D0-16's verdict-log pattern, which captured a Modal-fired empirical run.

The v1.0 docs amendment on physics-lint `master` does not get its own D-entry. Referenced from D0-18 amendment 1's footer as the v1.0-side counterpart to the harness-layer handling, with a post-merge-cleanup-TODO line naming the trigger for editing the cross-branch qualifier when `feature/rollout-anchors` merges to `master`.

---

## 2. Architecture

### 2.1 Cross-subtree split (post-migration)

```
physics-lint repo (branch: feature/rollout-anchors)
└── external_validation/_rollout_anchors/
    ├── _harness/
    │   ├── particle_rollout_adapter.py    [EDIT: energy_drift skip_reason template]
    │   ├── sarif_emitter.py               [EDIT: accept run-level properties dict]
    │   ├── lint_npz_dir.py                [NEW: generic npz-dir → HarnessResults]
    │   ├── SCHEMA.md                      [EDIT: §3.x harness SARIF result schema]
    │   └── tests/
    │       ├── test_d0_18_dissipative_skip.py    [EDIT: skip_reason template change]
    │       └── test_lint_npz_dir.py              [NEW]
    ├── 01-lagrangebench/
    │   ├── emit_sarif.py                  [NEW: case-study driver]
    │   ├── outputs/sarif/
    │   │   ├── segnn_tgv2d_<sarif_emission_sha>.sarif   [NEW: committed]
    │   │   └── gns_tgv2d_<sarif_emission_sha>.sarif     [NEW: committed]
    │   ├── outputs/_local_mirror/         [NEW: gitignored Volume cache]
    │   └── README.md                      [EDIT: sibling-relative pointer to writeup]
    └── methodology/
        ├── DECISIONS.md                   [APPEND: D0-19, D0-20, post-merge-TODO]
        ├── docs/
        │   ├── 2026-05-04-rung-4a-cross-stack-conservation-design.md   [THIS DOC]
        │   └── 2026-05-04-rung-4a-cross-stack-conservation-table.md    [NEW: writeup]
        ├── tools/
        │   └── render_cross_stack_table.py    [NEW]
        └── tests/
            ├── __init__.py                    [NEW]
            ├── fixtures/
            │   ├── segnn_tgv2d_fixture.sarif  [NEW: asymmetric-shas variant]
            │   ├── gns_tgv2d_fixture.sarif    [NEW: collapsed-shas variant]
            │   └── expected_table.md          [NEW: golden output]
            └── test_render_cross_stack_table.py    [NEW]

physics-lint repo (branch: master)
└── README.md   [EDIT: append to ## v1.0 known limitations]
```

**Property:** Generator (the harness + driver under `_rollout_anchors/`) and consumer (the renderer under `methodology/tools/`) communicate only through the SARIF artifact contract. No Python imports cross between the two subtrees. Schema version is the wire protocol; D0-19 + SCHEMA.md §3.x are the spec.

### 2.2 Data flow

```
Modal Volume (immutable, frozen at conversion-time shas)
  /vol/rollouts/lagrangebench/segnn_tgv2d_8c3d080397/   20 × particle_rollout_traj{NN}.npz
                                                          (npzs converted at 5857144;
                                                           pkls inferred at 8c3d080)
  /vol/rollouts/lagrangebench/gns_tgv2d_f48dd3f376/     20 × particle_rollout_traj{NN}.npz
                                                          (pkls + npzs at f48dd3f)
        │
        │  modal volume get  (one-shot, ~30s per stack at LAN; metadata-only billing)
        ▼
01-lagrangebench/outputs/_local_mirror/                 (gitignored cache)
        │
        │  python 01-lagrangebench/emit_sarif.py
        │    ├─ for each (model, dir) pair:
        │    │    _harness/lint_npz_dir.py:
        │    │      ├─ for each npz: load_rollout_npz, invoke 3 defects, build HarnessResult rows
        │    │      ├─ for harness:energy_drift rows: recompute ke_initial / ke_final
        │    │      │   from rollout, attach as extra_properties (per D0-19)
        │    │      └─ return list[HarnessResult]
        │    ├─ assemble run-level properties (3 shas + LB sha + 4 IDs + schema_version)
        │    └─ _harness/sarif_emitter.py:emit_sarif(results, run_properties=..., output_path=...)
        ▼
01-lagrangebench/outputs/sarif/                         (committed, ~30 KB each)
  segnn_tgv2d_<sarif_emission_sha>.sarif
  gns_tgv2d_<sarif_emission_sha>.sarif
        │
        │  cross-subtree boundary (no Python imports; only artifact contract)
        ▼
methodology/tools/render_cross_stack_table.py
  ├─ load both SARIFs (path provided as CLI arg)
  ├─ assert source == "rollout-anchor-harness" on both
  ├─ assert harness_sarif_schema_version == EXPECTED_SCHEMA_VERSION
  │   (raise SchemaVersionMismatch on mismatch — fail loud, not degraded)
  ├─ assert all required run-level fields present
  ├─ aggregate per-traj rows per (rule, stack):
  │     "all 20 identical" detection → single cell with single value
  │     non-uniform → summary stats (min/max/mean) per cell
  └─ emit markdown table to stdout
        │
        ▼
methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md
  (writeup includes the rendered table inline, the rederivability footer
   with exact render command + sha, and sibling-relative paths to SARIF
   artifacts. Commit-pinned GitHub URLs as optional secondary for
   GitHub-rendered-markdown readers.)
```

### 2.3 No GPU compute on critical path

Step 10's `modal volume get` is a download (metadata-only billing, ~$0). All lint computation is pure-CPU pure-Python on local mirror dir. Rung 4a does not fire any Modal function.

---

## 3. SARIF schema (D0-19 content)

### 3.1 Run-level properties (`runs[0].properties`, constants per SARIF)

| Field | Type | Example | Notes |
|---|---|---|---|
| `source` | string | `"rollout-anchor-harness"` | Discriminator vs public-API SARIF; renderer asserts. |
| `harness_sarif_schema_version` | string | `"1.0"` | Renderer's `EXPECTED_SCHEMA_VERSION` binds on this. Co-evolves with `physics_lint_sha_sarif_emission` by construction (any schema change is a sha change), but denormalized for assertion-locality. |
| `physics_lint_sha_pkl_inference` | string | `"8c3d080397..."` (SEGNN) / `"f48dd3f376..."` (GNS) | Sha at which LB CLI ran on Modal to produce pkls. |
| `physics_lint_sha_npz_conversion` | string | `"5857144..."` (SEGNN) / `"f48dd3f376..."` (GNS) | Sha at which pkl→npz conversion ran. May equal inference sha (single-shot) or differ (multi-session, as SEGNN). |
| `physics_lint_sha_sarif_emission` | string | post-d03df3e (TBD at step 10) | Sha at which the lint code emitted this SARIF. |
| `lagrangebench_sha` | string | `"b880a6c84a..."` | LB upstream sha (the inference engine producing the pkls). |
| `checkpoint_id` | string | `"segnn_tgv2d"` / `"gns_tgv2d"` | Trained-checkpoint identity (LB's gdown identifier or symbolic name). |
| `model_name` | string | `"segnn"` / `"gns"` | LB CLI key. |
| `dataset_name` | string | `"tgv2d"` | LB dataset identifier. |
| `rollout_subdir` | string | `"/vol/rollouts/lagrangebench/segnn_tgv2d_8c3d080397/"` | Volume artifact location at npz-genesis time. |

### 3.2 Multi-stage sha equality

The three `physics_lint_sha_*` fields **may be identical** (single-shot run where inference + conversion + emission collapse to one sha) or **distinct** (multi-session, as production SEGNN demonstrates with three different shas). Equality is allowed but never required. Renderer assertion logic does NOT impose equality.

### 3.3 Result-level fields (`runs[0].results[*]`)

Standard SARIF fields (used as-is):

| Field | Notes |
|---|---|
| `ruleId` | `"harness:mass_conservation_defect"`, `"harness:energy_drift"`, `"harness:dissipation_sign_violation"` (the `harness:` prefix distinguishes harness rules from public-API rule IDs) |
| `level` | `"note"` for both PASS-equivalent values and SKIPs. Existing harness convention; D0-19 documents it. |
| `message.text` | Human-readable summary. |

Result-level `properties`:

| Field | Guaranteed-identical across trajs (within stack)? | Notes |
|---|---|---|
| `traj_index` | NO | 0..19 |
| `npz_filename` | NO | `"particle_rollout_traj{NN}.npz"` |
| `raw_value` | YES (when defect emits a value AND the value happens to be load-bearing-identical, as it does for the four 0.0 cells in the 4a data) | Float, present iff row is not a SKIP |
| `skip_reason` | YES (template constant after the energy_drift change in §3.4) | String, present iff row is a SKIP |
| `ke_initial` | NO (per-trajectory varying) | Float, present only on `harness:energy_drift` SKIP rows; recomputed by `lint_npz_dir.py` from the rollout |
| `ke_final` | NO (per-trajectory varying) | Float, present only on `harness:energy_drift` SKIP rows; recomputed by `lint_npz_dir.py` from the rollout |

### 3.4 Schema-enforced invariants on result rows

For a fixed (rule, stack), all 20 result rows MUST have:
- Identical `ruleId`, `level`, `message.text`.
- Either identical `properties.raw_value` OR identical `properties.skip_reason` (rule emits exactly one of the two on every row; existing `HarnessDefect` invariant).

For `harness:energy_drift` SKIP rows specifically, the `properties.skip_reason` string is a constant template — no per-row value interpolation. Per-row varying values (`ke_initial`, `ke_final`) live in dedicated `properties.*` fields adjacent to the reason.

Consumers MAY assert these invariants at render time. The schema makes them checkable; checking is not mandatory.

### 3.5 The energy_drift skip_reason template change (forced by §3.4)

**Current emission** (`_harness/particle_rollout_adapter.py:energy_drift`, will be edited):

```python
skip_reason=(
    f"system_class='dissipative' (dataset={dataset_name!r}) and "
    f"KE(t) monotone-non-increasing across the rollout (KE(0)={e0:.3e}, "
    f"KE(end)={float(e_series[-1]):.3e}); ..."
)
```

**D0-19-mandated emission** (template-constant; values move to `properties.ke_initial` / `ke_final` on the SARIF row, attached by `lint_npz_dir.py`):

```python
skip_reason=(
    "system_class='dissipative' (dataset='tgv2d'); "
    "KE(t) monotone-non-increasing across the rollout; "
    "see properties.ke_initial / ke_final for values; "
    "consult dissipation_sign_violation as load-bearing alternative."
)
```

`HarnessDefect` itself stays unchanged (only `value` and `skip_reason` fields). Other rules don't get `ke_initial` / `ke_final`.

---

## 4. Sequencing

| # | Branch | Path | Commit |
|---|---|---|---|
| 1 | feature/rollout-anchors | `methodology/DECISIONS.md` | **D0-19** entry (SARIF schema pre-registration) |
| 2 | feature/rollout-anchors | `methodology/DECISIONS.md` | **D0-20** entry (consumption architecture pre-registration) |
| 3 | **master** | `README.md` | Append PH-CON-002 dissipative-system entry to `## v1.0 known limitations`, wording (a) + cross-branch qualifier. **Independent commit, parallel with all other steps.** |
| 4 | feature/rollout-anchors | `_harness/SCHEMA.md` | §3.x harness SARIF result schema spec |
| 5 | feature/rollout-anchors | `_harness/particle_rollout_adapter.py` | `energy_drift` — strip KE values from skip_reason template |
| 6 | feature/rollout-anchors | `_harness/tests/test_d0_18_dissipative_skip.py` | Update assertions for new constant template; audit for hardcoded reason-string equality, convert to substring/template-equality |
| 7 | feature/rollout-anchors | `_harness/sarif_emitter.py` | Extend `emit_sarif` to accept `run_properties: dict` |
| 8 | feature/rollout-anchors | `_harness/lint_npz_dir.py` (NEW) + `_harness/tests/test_lint_npz_dir.py` (NEW) | Generic npz-dir → HarnessResults; recompute ke_initial / ke_final for energy_drift rows; wrapper-preserves-D0-18-signal test |
| 9 | feature/rollout-anchors | `01-lagrangebench/emit_sarif.py` (NEW) | Case-study driver; assemble run-level properties (10 fields per §3.1); call lint_npz_dir twice; call emit_sarif twice |
| 10 | feature/rollout-anchors | `01-lagrangebench/outputs/sarif/{segnn,gns}_tgv2d_<sarif_emission_sha>.sarif` | `modal volume get` → run emit_sarif.py → commit SARIFs (filename pinned to feature/rollout-anchors HEAD at emission time) |
| 11 | feature/rollout-anchors | `01-lagrangebench/README.md` | One-line sibling-relative pointer to `../methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md` |
| 12 | feature/rollout-anchors | `methodology/tools/render_cross_stack_table.py` (NEW) + `methodology/tests/{__init__.py, fixtures/*, test_render_cross_stack_table.py}` (NEW) | Renderer + golden + version-mismatch + asymmetric-shas tests |
| 13 | feature/rollout-anchors | `methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md` (NEW) | Writeup with pasted table + rederivability footer + sibling-relative paths to SARIF artifacts |
| 14 | feature/rollout-anchors | `methodology/DECISIONS.md` | Post-merge-cleanup-TODO line under D0-18 amendment 1's footer naming the trigger for editing the master README qualifier when feature/rollout-anchors merges |

### 4.1 Parallelism

- Step 3 (master README) is fully independent — runs in parallel with everything.
- Steps 1, 2, 4 are pre-registrations; should land first per project pre-registration discipline. Independent of each other.
- Steps 5–6 (energy_drift template) and steps 7–8 (sarif_emitter + lint_npz_dir) are independent of each other after step 4. Can be developed in parallel.
- Step 12 depends on step 4's schema spec but not on the SARIF generation (renderer tested against synthetic fixtures). Can land in parallel with steps 5–10.

### 4.2 Critical path

`1 → 4 → (5+6 || 7+8) → 9 → 10 → 13`. ~7 deep, all on feature/rollout-anchors.

### 4.3 Frozen npzs property

Steps 5–10 do not touch Modal Volume. The npzs at `/vol/rollouts/lagrangebench/segnn_tgv2d_8c3d080397/` and `/vol/rollouts/lagrangebench/gns_tgv2d_f48dd3f376/` are read-only inputs. Step 10's `modal volume get` is a download; SARIF generation is local-CPU. After step 10, the local mirror dir is gitignored; only the SARIF artifacts are committed.

---

## 5. Testing + error handling

### 5.1 Test surface

| Test file | Layer | Coverage |
|---|---|---|
| `_harness/tests/test_d0_18_dissipative_skip.py` (UPDATE) | Harness emission | Update assertions for new constant template (§3.5); audit hardcoded-reason-string assertions and convert to substring or template-equality. Truth-table coverage (system_class × monotonicity, precedence vs D0-08) is structurally unaffected by the template change. |
| `_harness/tests/test_lint_npz_dir.py` (NEW) | Generic npz-dir → HarnessResults | (a) Reads synthetic npz, invokes 3 defects, builds correctly-shaped HarnessResult rows. (b) `ke_initial` / `ke_final` extra_properties present on energy_drift SKIP rows, absent on other rows. (c) HarnessResult ordering deterministic across invocations. (d) Empty-dir handling: `EmptyNpzDirectoryError` raises. (e) Non-npz files in dir ignored. (f) **Wrapper-preserves-D0-18-signal**: synthetic dissipative-system npz → resulting `harness:energy_drift` HarnessResult row has `skip_reason` set (not `raw_value`). |
| `methodology/tests/test_render_cross_stack_table.py` (NEW) | Renderer | (a) Golden test: synthetic fixtures → rendered table matches `expected_table.md` byte-for-byte. (b) `SchemaVersionMismatch` raises on `bumped_schema_fixture` (programmatically derived). (c) `SourceTagMismatch` raises on wrong-source fixture (programmatically derived). (d) `MissingRunLevelField` raises (programmatically derived: load segnn fixture → delete one required field → assert raise). (e) Aggregation logic: "all 20 identical → single cell" detection fires on uniform values; falls back to summary stats on non-uniform. (f) **Asymmetric-shas case**: feeds renderer fixtures with deliberately-distinct pkl_inference_sha / npz_conversion_sha / sarif_emission_sha values, asserts renderer handles them correctly (distinct columns/cells, no equality assumption, no crash). |

### 5.2 Test fixtures (hand-crafted synthetic-but-realistic; never copied from production)

| Fixture | Purpose |
|---|---|
| `methodology/tests/fixtures/segnn_tgv2d_fixture.sarif` | 60 rows (3 rules × 20 trajs); deterministic synthetic values; **asymmetric shas by default** (`synthetic_inference_sha`, `synthetic_conversion_sha`, `synthetic_emission_sha` — three distinct values) reflecting production SEGNN's multi-stage genesis. Synthetic dataset_name (`synthetic_dissipative_d`) so the fixture exercises the schema, not LB-specific names. |
| `methodology/tests/fixtures/gns_tgv2d_fixture.sarif` | Same shape; **collapsed-shas variant** (single `synthetic_combined_sha` for inference+conversion, separate `synthetic_emission_sha`) reflecting production GNS's single-shot run. |
| `methodology/tests/fixtures/expected_table.md` | Hand-written to match the synthetic fixtures' rendered output. Paired regeneration with the fixtures (same commit when either changes). |

Negative-path fixtures (`bumped_schema`, `wrong_source`, `missing_field`) derived programmatically at test time from `segnn_tgv2d_fixture.sarif` rather than committed as separate files. Reduces fixture maintenance and binds the negative-path tests to the canonical fixture's schema-conformance.

### 5.3 Error-handling — fail-loud assertions, no degraded paths

| Assertion site | Check | Failure mode |
|---|---|---|
| Renderer entry | `runs[0].properties.source == "rollout-anchor-harness"` | Raise `SourceTagMismatch` with both expected and actual values |
| Renderer entry | `runs[0].properties.harness_sarif_schema_version == EXPECTED_SCHEMA_VERSION` | Raise `SchemaVersionMismatch` with both expected and actual + pointer to SCHEMA.md §3.x |
| Renderer entry | All required run-level fields present (the 10 from §3.1) | Raise `MissingRunLevelField` naming the missing field |
| `lint_npz_dir.py` | `lint_npz_dir(empty_dir)` | Raise `EmptyNpzDirectoryError` (no silent empty SARIF) |
| `emit_sarif.py` driver | Both stack directories present in local mirror | Raise `MissingLocalMirrorError` with the `modal volume get` command in the message |
| `sarif_emitter.py` extension | `run_properties` dict has the required keys (per §3.1) | Raise `IncompleteRunLevelProperties` naming missing keys |

No warnings, no logs-and-continues, no best-effort outputs. The fail-loud discipline is what makes the generator-vs-consumer separation enforceable per D0-20.

### 5.4 No production-SARIF integration test

The renderer is fully covered by synthetic fixtures (§5.2). The rederivability footer in the writeup (run command + paired output) is the load-bearing integration check — manual but auditable. Production-SARIF fixtures would couple tests to specific Modal Volume state and force fixture updates on every regeneration; the hand-crafted-fixtures discipline takes precedence.

---

## 6. Writeup conventions (for step 13)

### 6.1 Structure

The writeup at `methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md` contains (in order):

1. **Header**: date, scope, predecessor (rung 3.5 PASS at d03df3e), successor (rung 4b — separate brainstorm).
2. **Headline** (§1.2's frozen sentence, copied verbatim).
3. **Rendered cross-stack conservation table** (pasted from renderer stdout).
4. **What rung 4a is NOT** (the 5-item list from §1.3, reproduced).
5. **Citation to D0-19** at the load-bearing "20 identical fires" claim, half-sentence inline.
6. **Rederivability footer** — exact render command + sha at which the table was rendered:

   ```
   Rendered at physics-lint feature/rollout-anchors sha <sarif_emission_sha> via:
   $ python methodology/tools/render_cross_stack_table.py \
       --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/
   ```

7. **Cross-references**: sibling-relative paths to the two SARIF artifacts (commit-pinned GitHub URLs as optional secondary for GitHub-rendered-markdown readers).
8. **Integrating-trigger footer**: one-line statement that the top-level integrating README is composed when 4b's writeup lands; until then `methodology/docs/` is the source of truth in dated-deliverable form.

### 6.2 Bidirectional findability

`01-lagrangebench/README.md` adds a one-line sibling-relative pointer: *"Cross-stack writeup: see [`../methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md`]."* Goes opposite direction from the writeup's paths to the SARIF artifacts.

---

## 7. Forward-looking

### 7.1 Rung 4b dependencies on 4a

- D0-19's SARIF schema is reused. Rung 4b adds equivariance result rows but does not modify the run-level shape.
- The renderer at `methodology/tools/render_cross_stack_table.py` is extended (not replaced) to handle equivariance rule rows. Schema version bumps; the renderer's `EXPECTED_SCHEMA_VERSION` is updated; old fixtures remain valid for regression tests (4a results don't change, just have new sibling rows for equivariance).
- Three open questions for 4b's brainstorm (already flagged): trajectory-aligned vs first-step-only ε (deepest, affects threshold semantics); ε-probe of GNS-on-TGV2D specifically before committing to APPROXIMATE-band framing (the cited GNS-approximate prior result may be dataset-specific to dam break, not TGV2D); rotation-of-velocities mechanic (R applied to v at IC construction, not just to positions).
- Rung 4b is the natural Security-tab integration rung because it produces real findings. 4a's harness-style SARIF convention is preserved; the Security-tab demo emits a parallel public-v1.0-style SARIF for that path.

### 7.2 Post-merge cleanup TODO

When `feature/rollout-anchors` merges to `master`, the cross-branch qualifier in master's `## v1.0 known limitations` PH-CON-002 entry — "currently on the `feature/rollout-anchors` branch pending merge to master" — becomes stale and must be edited out. Tracked at step 14 in `methodology/DECISIONS.md` under D0-18 amendment 1's footer.

### 7.3 v1.x graduation question (open)

If a future D-entry proposes graduating D0-18's dissipative-system handling into physics-lint v1.x core, the SARIF convention divergence (harness-style emits `level: "note"` for PASS-equivalent rows; public-v1.0 suppresses PASS rows) must be revisited explicitly: either the note-level rows get dropped (matching v1.0 PASS-suppression) or they get retained as a deliberate two-tier convention (harness-derived rules emit informational findings; pure-v1.0 rules emit only on findings) with documented rationale. D0-19 records this as a forward-flag; the graduation decision belongs to a future D-entry.

---

## 8. Out of scope (do not pull in during implementation)

- Equivariance test wiring (PH-SYM-001 / 002 / 003 / 004 trained-model band per SCHEMA.md §4.2). Belongs to rung 4b.
- Modifications to physics-lint v1.0 core (`src/physics_lint/rules/ph_con_002.py`, `src/physics_lint/sarif.py`, etc.). The harness-layer placement is preserved per D0-18.
- New rule additions (PH-CSH-* roadmap, etc.).
- Performance optimization on `lint_npz_dir.py` or the renderer. Both run in seconds on 4a's data; not a story.
- Generalizing `lint_npz_dir.py` for case study 02 (PhysicsNeMo). Generic-enough-for-now is sufficient; case study 02 may want different defects and is its own rung.
- Direct-Modal-Volume in-Modal lint emission. The local-mirror pattern is locked for 4a.
