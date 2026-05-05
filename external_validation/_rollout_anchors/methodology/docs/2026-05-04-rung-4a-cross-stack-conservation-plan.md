# Rung 4a — Cross-stack conservation SARIF + writeup table (implementation plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package the existing 40 frozen npzs (SEGNN-TGV2D + GNS-TGV2D on Modal Volume) into committed harness-style SARIF artifacts, render a cross-stack conservation table over them, write the dated methodology writeup, and amend physics-lint v1.0's master-branch README to document PH-CON-002's dissipative-system behavior as an honest limit.

**Architecture:** Generator (harness + driver under `_rollout_anchors/`) and consumer (renderer under `methodology/tools/`) communicate only through the SARIF artifact contract — schema_version is the wire protocol; no Python imports cross the subtree boundary. Fail-loud assertions on the consumer side (SchemaVersionMismatch, etc.) enforce the separation. The energy_drift skip_reason gets a template-constant rewrite so D0-19's "guaranteed-identical across trajs" contract is schema-enforced, not coincidental — per-traj-varying KE endpoints move into dedicated SARIF properties.

**Tech Stack:** Python 3.11+, NumPy, SARIF v2.1.0, pytest, ruff (lint), codespell (docs). Modal CLI for one-shot volume download. Pre-commit hooks via `.venv/bin/pre-commit` (already configured at repo root).

**Predecessor:** Design doc `methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-design.md` (committed at `efa7b05`). Read it first — it carries the full rationale for every choice in this plan.

**Branch:** All work on `feature/rollout-anchors` except T0.3 (one commit on `master`).

**Working dir convention:** All paths in this plan are absolute under `/Users/zenith/Desktop/physics-lint/`. Tests run via `cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest <args>`. Commits run with the same PATH override so pre-commit hooks resolve.

**Module import convention:** harness modules use `from external_validation._rollout_anchors._harness.<module> import <name>`.

---

## Task 0: Pre-registrations (D0-19, D0-20, master README amendment)

**Why first:** Per the project's pre-registration discipline, decision entries land before the implementation they describe. D0-19 specifies the SARIF schema; D0-20 specifies the consumption architecture; master README amendment closes the v1.0-honest-limits docs loop deferral #4 cites.

**Files:**
- Modify: `external_validation/_rollout_anchors/methodology/DECISIONS.md` (T0.1, T0.2)
- Modify: `README.md` on master branch (T0.3)

### T0.1: Append D0-19 entry to methodology/DECISIONS.md

- [ ] **Step 1: Find insertion point**

```bash
tail -10 /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/DECISIONS.md
```

Expected: last lines are D0-18 amendment 1's "**Realized — amendment 1.**" closing paragraph + a horizontal rule `---` or end-of-file. If file ends without `---`, prepare to append `\n---\n\n` before the new entry.

- [ ] **Step 2: Append D0-19 entry**

Append this content to `external_validation/_rollout_anchors/methodology/DECISIONS.md` (preceded by `\n---\n\n` if the file does not already end with `---`):

```markdown
## D0-19 — 2026-05-04 — Harness SARIF result schema (rung 4a pre-registration)

**Question.** Rung 4a will package the 40 npzs from rung 3.5 PASS into
committed SARIF artifacts via the existing `_harness/sarif_emitter.py`.
The artifact's contract — what fields are at the run level vs the result
level, what fields are guaranteed-identical across rows vs allowed to
vary, what the schema_version is, what the renderer can assume — is
load-bearing for the rung 4a writeup's "20 identical fires across both
stacks" claim. That claim is defensible only if the schema enforces the
identity, not if a grep happens to find it on this run's data.

**Decision (pre-registered before any code change).**

The harness SARIF schema gets formal field-level guarantees. Run-level
properties (`runs[0].properties`) carry constants per artifact. Result-
level properties (`runs[0].results[*].properties`) carry per-row data,
explicitly classified as guaranteed-identical-across-rows-within-stack
or may-vary.

Run-level fields (10 total):

- `source` (literal `"rollout-anchor-harness"` — discriminator vs public-API SARIF)
- `harness_sarif_schema_version` (string, e.g., `"1.0"`; renderer asserts on equality)
- `physics_lint_sha_pkl_inference` (sha at which LB CLI ran on Modal to produce pkls)
- `physics_lint_sha_npz_conversion` (sha at which pkl→npz conversion ran)
- `physics_lint_sha_sarif_emission` (sha at which the lint code emitted this SARIF)
- `lagrangebench_sha` (LB upstream sha — the inference engine producing the pkls)
- `checkpoint_id` (LB gdown identifier or symbolic name)
- `model_name` (LB CLI key, e.g., `"segnn"` / `"gns"`)
- `dataset_name` (LB dataset identifier, e.g., `"tgv2d"`)
- `rollout_subdir` (Volume artifact location at npz-genesis time)

The three `physics_lint_sha_*` fields **may be identical** (single-shot
run where inference + conversion + emission collapse to one sha) or
**distinct** (multi-session, as production SEGNN demonstrates: pkl
inference at `8c3d080`, npz conversion at `5857144`, SARIF emission at
post-`d03df3e`). Equality is allowed but never required. Renderer
assertion logic does NOT impose equality across stages.

`harness_sarif_schema_version` co-evolves with `physics_lint_sha_sarif_emission`
by construction (any schema change is a sha change), but is denormalized
into the SARIF for renderer assertion-locality.

Result-level fields:
- `traj_index` (int 0..19; may-vary)
- `npz_filename` (e.g., `"particle_rollout_traj00.npz"`; may-vary)
- `raw_value` (float, when defect emits a value; guaranteed-identical
  iff value is load-bearing-identical, as it is for the four 0.0 cells
  in 4a's data)
- `skip_reason` (string, when defect SKIPs; guaranteed-identical —
  template constant after the energy_drift change below)
- `ke_initial` (float; present only on `harness:energy_drift` SKIP rows; may-vary)
- `ke_final` (float; present only on `harness:energy_drift` SKIP rows; may-vary)

For a fixed (rule, stack), all 20 result rows MUST have identical
`ruleId`, `level`, `message.text`, plus either identical `raw_value` or
identical `skip_reason` (the existing `HarnessDefect` invariant: rule
emits exactly one of the two on every row).

Consumers MAY assert these invariants at render time. The schema makes
them checkable, not mandatory-to-check.

**Energy_drift skip_reason template change (forced by the contract).**

Current emission interpolates per-row varying KE values into the
skip_reason string:

    f"...KE(0)={e0:.3e}, KE(end)={float(e_series[-1]):.3e}..."

This makes skip_reason per-row varying, violating the "guaranteed-identical"
classification. D0-19 mandates a template-constant skip_reason; the
varying values move to dedicated `properties.ke_initial` / `ke_final`
fields on the SARIF row, attached by the new `lint_npz_dir.py` module.

New skip_reason template (interpolates `dataset_name` only, which is
constant per stack):

    f"system_class='dissipative' (dataset={dataset_name!r}); "
    "KE(t) monotone-non-increasing across the rollout; "
    "see properties.ke_initial / ke_final for values; "
    "consult dissipation_sign_violation as load-bearing alternative."

`HarnessDefect` itself stays unchanged (only `value` and `skip_reason`
fields). Other rules don't get `ke_initial` / `ke_final`.

**Schema version:** v1.0 (this entry pins it).

**Forward-flag (v1.x graduation question).** If a future D-entry
proposes graduating D0-18's dissipative-system handling into physics-
lint v1.x core, the SARIF convention divergence (harness-style emits
`level: "note"` for PASS-equivalent rows; public-v1.0 suppresses PASS
rows) must be revisited explicitly: either drop the note-level rows
(matching v1.0 PASS-suppression) or retain them as a deliberate two-
tier convention (harness-derived rules emit informational findings;
pure-v1.0 rules emit only on findings). Recorded here so the question
doesn't get rediscovered cold.

**Realized.** This entry now. SCHEMA.md §3.x extension lands at the
same sequencing position; D0-20 follows immediately as the consumer-
side counterpart. Implementation lands per the 14-step sequence in
`methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-design.md`
§4.

---
```

- [ ] **Step 3: Verify entry renders by reading it back**

```bash
grep -A 2 "^## D0-19 " /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/DECISIONS.md
```

Expected: heading line `## D0-19 — 2026-05-04 — Harness SARIF result schema (rung 4a pre-registration)` followed by blank line and `**Question.**`.

- [ ] **Step 4: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/methodology/DECISIONS.md && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git commit -m "$(cat <<'EOF'
methodology/DECISIONS.md D0-19: harness SARIF result schema

Pre-registers the rung 4a SARIF artifact contract: 10 run-level fields
(source, schema_version, three stage-specific physics_lint shas,
lagrangebench_sha, 4 IDs), result-level field classification
(guaranteed-identical / may-vary), energy_drift skip_reason template
change forced by the contract (per-row KE values move to dedicated
ke_initial / ke_final properties). schema_version v1.0. Forward-flag
on v1.x graduation level-mapping question.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected output: pre-commit hooks run (ruff/format/codespell pass on docs-only commit; field-property-tests skip), then `[feature/rollout-anchors <sha>] methodology/DECISIONS.md D0-19: ...`.

### T0.2: Append D0-20 entry to methodology/DECISIONS.md

- [ ] **Step 1: Append D0-20 entry**

Append this content to `external_validation/_rollout_anchors/methodology/DECISIONS.md` (preceded by `\n---\n\n` to separate from D0-19):

```markdown
## D0-20 — 2026-05-04 — Generator-vs-consumer separation architecture (rung 4a pre-registration)

**Question.** D0-19 specifies *what's in* the harness SARIF artifact.
D0-20 specifies *how it's consumed* by the cross-stack writeup table
renderer. Orthogonal pre-registrations: D0-19 is the artifact contract;
D0-20 is the consumption architecture.

**Decision (pre-registered before renderer code).**

1. **Renderer lives in `methodology/tools/`** alongside other methodology
   tooling, not in `_harness/`. Generator (harness + driver under
   `_rollout_anchors/`) and consumer (renderer under `methodology/tools/`)
   communicate only through the SARIF artifact contract. **No Python
   imports cross the subtree boundary.** This makes the renderer
   testable in isolation against synthetic fixtures, and prevents the
   methodology subtree from accruing the harness as an import-time
   dependency.

2. **Schema version is the wire protocol.** Renderer hard-codes an
   `EXPECTED_SCHEMA_VERSION` constant. First action on reading any
   SARIF: assert `runs[0].properties.harness_sarif_schema_version ==
   EXPECTED_SCHEMA_VERSION`. On mismatch, raise `SchemaVersionMismatch`
   — **fail loud, not degraded**. No warnings, no log-and-continue, no
   best-effort coercion. The assertion is the load-bearing primitive
   that lets the cross-subtree separation hold without introducing
   silent version drift.

3. **Source-tag assertion** parallels schema-version: assert
   `runs[0].properties.source == "rollout-anchor-harness"`; raise
   `SourceTagMismatch` on failure. Distinguishes harness SARIF from
   public-API SARIF reaching the renderer by accident.

4. **Run-level field-presence assertions.** All 10 D0-19 run-level
   fields are required on read; missing → raise `MissingRunLevelField`
   naming the missing field. No defaulting.

5. **Test surface — golden + version-mismatch + asymmetric-shas.**
   - Golden: synthetic fixtures → rendered table matches an
     expected_table.md byte-for-byte.
   - Version-mismatch: bumped-version fixture (programmatically derived)
     → SchemaVersionMismatch raises.
   - Source-tag mismatch: wrong-source fixture (programmatically derived)
     → SourceTagMismatch raises.
   - Missing-field: deleted-field fixture (programmatically derived) →
     MissingRunLevelField raises.
   - Asymmetric-shas: feeds renderer fixtures with deliberately-distinct
     three-stage shas; asserts renderer handles them correctly without
     equality assumption (per D0-19's may-be-identical-or-distinct
     contract).
   - Aggregation: "all 20 identical → single cell" detection fires on
     uniform values; falls back to summary stats on non-uniform.

6. **Test fixtures hand-crafted synthetic-but-realistic, NEVER copied
   from production.** Production SARIFs have incidental properties
   (specific shas, paths, run-time-only fields) that have nothing to do
   with the schema contract; copying them couples the test to those
   incidentals. Fixtures use placeholder shas (`synthetic_inference_sha`,
   etc.) and synthetic dataset names. SEGNN fixture defaults to
   asymmetric three-stage shas (matching production); GNS fixture
   defaults to collapsed shas (also matching production). Negative-path
   fixtures (bumped_schema, wrong_source, missing_field) derived
   programmatically at test time from the canonical fixture rather than
   committed as separate files.

7. **Renderer output convention.** Renderer is pure stdin-out: emits
   markdown table to stdout, never writes to the writeup file directly.
   Writeup includes the table via copy-paste, plus a rederivability
   footer with the exact render command + sha at which the table was
   rendered. The footer is what converts copy-paste from "two artifacts
   that happen to agree right now" into "an artifact reproducible from
   the commit-pinned source."

**Reframing note.** Pre-migration, this would have been "cross-repo"
separation (renderer in physics-lint-validation, generator in physics-
lint). Post-migration (`971b8fc`), the methodology subtree lives in
physics-lint alongside the harness; the discipline is intra-repo
subtree separation with the same wire-protocol shape. The cross-repo
URL discipline reduces to sibling-relative paths for in-repo navigation
(commit-pinned GitHub URLs as optional secondary for GitHub-rendered-
markdown readers).

**Realized.** This entry now. Renderer implementation lands per the
14-step sequence in
`methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-design.md`
§4 step 12.

---
```

- [ ] **Step 2: Verify entry renders**

```bash
grep -c "^## D0-20" /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/DECISIONS.md
```

Expected: `1`.

- [ ] **Step 3: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/methodology/DECISIONS.md && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git commit -m "$(cat <<'EOF'
methodology/DECISIONS.md D0-20: generator-vs-consumer separation architecture

Pre-registers the rung 4a renderer architecture orthogonal to D0-19's
artifact contract. Renderer in methodology/tools/, no Python imports
cross-subtree, harness_sarif_schema_version + source-tag as wire
protocol with fail-loud-on-mismatch (SchemaVersionMismatch /
SourceTagMismatch / MissingRunLevelField). Test surface: golden +
version-mismatch + source-mismatch + missing-field + asymmetric-shas
+ aggregation. Fixtures hand-crafted synthetic-but-realistic, never
copied from production. Reframed from "cross-repo" to "intra-repo
subtree separation" post-migration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: hooks pass; commit succeeds.

### T0.3: Master README amendment for v1.0 honest limits

This is the only commit on `master` branch. Sequence: stash check → switch → edit → commit → switch back.

- [ ] **Step 1: Confirm clean working tree on feature/rollout-anchors**

```bash
cd /Users/zenith/Desktop/physics-lint && git status -sb
```

Expected: `## feature/rollout-anchors` and no further uncommitted-files lines (or only known-untracked files like `01-lagrangebench/outputs/_local_mirror/` if T6 has run partially). Stop and resolve any uncommitted changes before proceeding — `git switch` will refuse if there are conflicting uncommitted edits.

- [ ] **Step 2: Switch to master**

```bash
cd /Users/zenith/Desktop/physics-lint && git switch master
```

Expected: `Switched to branch 'master'`.

- [ ] **Step 3: Read the existing v1.0 known limitations section**

```bash
grep -n "## v1.0 known limitations" /Users/zenith/Desktop/physics-lint/README.md
```

Expected: line number (around 167 per Section 2 verification).

```bash
sed -n '165,200p' /Users/zenith/Desktop/physics-lint/README.md
```

Expected output: the heading, the existing PH-BC-001 / PH-RES-001 honest-limits paragraph, and surrounding context.

- [ ] **Step 4: Append the PH-CON-002 entry to the section**

The amendment lands as a new bullet/paragraph immediately after the existing PH-BC-001 / PH-RES-001 entry within the `## v1.0 known limitations` section, before any subsequent heading. Edit `/Users/zenith/Desktop/physics-lint/README.md` to append this content (find the line of the closing of the PH-BC-001 / PH-RES-001 paragraph, ensure a blank line follows, then add):

```markdown
**`PH-CON-002` evaluates `raw_value` on dissipative systems, producing FAIL on physically-correct dissipative-by-design behavior.** TGV2D, RPF2D, LDC2D, DAM2D and analogous viscous-SPH systems dissipate energy as a property of the physics; PH-CON-002's relative-drift form (`max|E(t) - E(0)| / |E(0)|`) trips the FAIL threshold on rollouts where ~99% of initial KE has correctly dissipated to viscosity. This is the primary use case for ML PDE surrogates (most ML targets are dissipative); a writeup footnote saying "ignore those FAILs" is harder to defend than the right rule semantics.

The harness layer (currently on the `feature/rollout-anchors` branch pending merge to master, at `external_validation/_rollout_anchors/_harness/`) demonstrates a skip-with-reason mechanism that addresses this — a two-half positive-evidence gate (system_class hint AND KE-monotone-non-increasing) avoids masking buggy supposed-conservative surrogates while restoring correct semantics on dissipative-by-design rollouts. The harness layer is the prototype for v1.x graduation; the v1.0 public PH-CON-002 rule is preserved as-shipped here pending that future D-entry. See physics-lint-validation `DECISIONS.md` D0-18 + the rung 4a writeup at `external_validation/_rollout_anchors/methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md` (post-merge) for the full discussion.
```

- [ ] **Step 5: Commit on master**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add README.md && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git commit -m "$(cat <<'EOF'
README: PH-CON-002 dissipative-system honest limit

Append PH-CON-002 entry to ## v1.0 known limitations alongside the
existing PH-BC-001 / PH-RES-001 entry. v1.0 evaluates raw_value on
dissipative systems, which produces FAIL on physically-correct
dissipative-by-design behavior — primary use case for ML PDE
surrogates, not a fringe edge case. Honest-limit framing per
physics-lint-validation DECISIONS.md D0-18: harness layer
(currently on feature/rollout-anchors pending merge) prototypes
the v1.x graduation skip-with-reason mechanism; v1.0 public rule
preserved as-shipped pending future D-entry.

Cross-branch qualifier in the wording is provisional — gets edited
out at feature/rollout-anchors merge per the post-merge-cleanup-TODO
tracked in methodology/DECISIONS.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: hooks pass (codespell on the new bullet); commit succeeds. Note master sha: `git -C /Users/zenith/Desktop/physics-lint rev-parse master` — record it for the post-merge-cleanup entry in T11.

- [ ] **Step 6: Switch back to feature/rollout-anchors**

```bash
cd /Users/zenith/Desktop/physics-lint && git switch feature/rollout-anchors
```

Expected: `Switched to branch 'feature/rollout-anchors'`. T0.3 is the only master-side work in rung 4a.

---

## Task 1: SCHEMA.md §3.x harness SARIF result schema

**Files:**
- Modify: `external_validation/_rollout_anchors/_harness/SCHEMA.md`

**Why:** D0-19's contract has a normative home in the harness's existing SCHEMA.md, alongside §1 (npz schema) and §3 (source-tag rationale). Renderer cites SCHEMA.md §3.x at the assertion site.

- [ ] **Step 1: Find insertion point in SCHEMA.md**

```bash
grep -n "^## " /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/_harness/SCHEMA.md
```

Expected: numbered section headings; identify the position after the existing §3 (source-tag rationale) but before §4 (Tolerances) — the new §3.x extends §3.

- [ ] **Step 2: Append the schema spec**

Insert after the existing §3 content (before `## 4. Tolerances`). Add a new subsection `### 3.x Harness SARIF result schema (D0-19)`:

```markdown
### 3.x Harness SARIF result schema (D0-19)

Pre-registered in `physics-lint/external_validation/_rollout_anchors/methodology/DECISIONS.md` D0-19. Schema version: v1.0.

#### Run-level properties (`runs[0].properties`)

All 10 fields REQUIRED. Renderer raises `MissingRunLevelField` on any absent field.

| Field | Type | Description |
|---|---|---|
| `source` | string | Literal `"rollout-anchor-harness"`. Discriminator vs public-API SARIF. |
| `harness_sarif_schema_version` | string | `"1.0"` for this version. Renderer's `EXPECTED_SCHEMA_VERSION` binds on equality. Bumps on any contract change in this section; co-evolves with `physics_lint_sha_sarif_emission` by construction. |
| `physics_lint_sha_pkl_inference` | string | Sha at which the LB CLI ran on Modal to produce pkls. |
| `physics_lint_sha_npz_conversion` | string | Sha at which pkl→npz conversion ran. May equal inference sha (single-shot run) or differ (multi-session). |
| `physics_lint_sha_sarif_emission` | string | Sha at which the lint code emitted this SARIF. May equal the other two or differ. |
| `lagrangebench_sha` | string | LB upstream sha (the inference engine producing the pkls). |
| `checkpoint_id` | string | LB gdown identifier or symbolic name. |
| `model_name` | string | LB CLI key, e.g., `"segnn"` or `"gns"`. |
| `dataset_name` | string | LB dataset identifier, e.g., `"tgv2d"`. |
| `rollout_subdir` | string | Volume artifact location at npz-genesis time, e.g., `"/vol/rollouts/lagrangebench/segnn_tgv2d_<sha>/"`. |

The three `physics_lint_sha_*` fields MAY be identical (single-shot) or distinct (multi-session). The renderer MUST NOT assume equality across stages.

#### Result-level fields (`runs[0].results[*]`)

Standard SARIF fields:

- `ruleId` — one of `"harness:mass_conservation_defect"`, `"harness:energy_drift"`, `"harness:dissipation_sign_violation"` (the `harness:` prefix distinguishes harness rules from public-API rule IDs).
- `level` — `"note"` for both PASS-equivalent values and SKIPs.
- `message.text` — human-readable summary.

Result-level `properties`:

| Field | Guaranteed-identical across trajs (within stack)? | Description |
|---|---|---|
| `traj_index` | NO (may-vary) | Integer 0..(N-1) where N is the number of trajectories in the rollout. |
| `npz_filename` | NO (may-vary) | E.g., `"particle_rollout_traj00.npz"`. |
| `raw_value` | YES iff present AND value happens to be identical across rows | Float. Present iff row is not a SKIP. |
| `skip_reason` | YES (template constant) | String. Present iff row is a SKIP. Template-constant — no per-row value interpolation. |
| `ke_initial` | NO (may-vary) | Float. Present only on `harness:energy_drift` SKIP rows. |
| `ke_final` | NO (may-vary) | Float. Present only on `harness:energy_drift` SKIP rows. |

#### Schema-enforced invariants

For a fixed (rule, stack), all result rows MUST have:

- Identical `ruleId`, `level`, `message.text`.
- Identical `properties.raw_value` if present, OR identical `properties.skip_reason` if present (HarnessDefect emits exactly one of the two on every row).

Consumers MAY assert these invariants at render time. The schema makes them checkable; checking is not mandatory.

#### Energy_drift skip_reason template

Per D0-19, the `harness:energy_drift` SKIP path uses a template-constant skip_reason (no per-row value interpolation). Per-row varying KE values move to `properties.ke_initial` / `properties.ke_final`. See `_harness/particle_rollout_adapter.py:energy_drift` for the canonical template; renderer-side documentation cites that path.
```

- [ ] **Step 3: Verify section appears between §3 and §4**

```bash
grep -n "^### 3\.x\|^## 4" /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/_harness/SCHEMA.md
```

Expected: `### 3.x Harness SARIF result schema (D0-19)` appears before `## 4. Tolerances`.

- [ ] **Step 4: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/SCHEMA.md && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git commit -m "$(cat <<'EOF'
_harness/SCHEMA.md §3.x: harness SARIF result schema (D0-19)

Adds normative spec for harness SARIF run-level vs result-level
fields, the guaranteed-identical / may-vary classification, schema
v1.0 pin, and the energy_drift skip_reason template note. Citation
target for renderer-side D0-19 assertions.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: energy_drift skip_reason template change (TDD)

**Files:**
- Modify: `external_validation/_rollout_anchors/_harness/particle_rollout_adapter.py:461-521` (the `energy_drift` function, specifically lines ~507-519 that build the dissipative-skip skip_reason)
- Modify: `external_validation/_rollout_anchors/_harness/tests/test_d0_18_dissipative_skip.py:262-271` (the `test_skip_reason_includes_ke_endpoints` test that asserts on KE values in the string)

**Why:** D0-19's contract requires skip_reason be template-constant. Current code interpolates `KE(0)` and `KE(end)`; this makes skip_reason per-row varying. Strip the value interpolation; values move to SARIF properties via T4. The two-half-gate truth-table tests (lines 125-173) are unaffected.

### T2.1: Replace test_skip_reason_includes_ke_endpoints with a template-content test

- [ ] **Step 1: Edit `test_d0_18_dissipative_skip.py`, replacing the test at lines 262-271**

Open `/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/_harness/tests/test_d0_18_dissipative_skip.py`. Replace the existing test (lines 262-271):

```python
def test_skip_reason_includes_ke_endpoints() -> None:
    """The skip_reason must include KE(0) and KE(end) so a future reader
    can quickly assess whether the dissipation magnitude is physical
    (e.g., partial decay vs ~total decay).
    """
    rollout = _build_dissipative_rollout(dataset_name="tgv2d")
    result = energy_drift(rollout)
    assert result.skip_reason is not None
    assert "KE(0)" in result.skip_reason
    assert "KE(end)" in result.skip_reason
```

with:

```python
def test_skip_reason_signposts_ke_property_fields() -> None:
    """Per D0-19's guaranteed-identical contract on skip_reason, KE
    endpoint values must NOT be interpolated into the reason string
    (that would make skip_reason per-row varying). The reason instead
    signposts the dedicated SARIF properties (`ke_initial`, `ke_final`)
    where the per-row values live.
    """
    rollout = _build_dissipative_rollout(dataset_name="tgv2d")
    result = energy_drift(rollout)
    assert result.skip_reason is not None
    assert "KE(0)" not in result.skip_reason
    assert "KE(end)" not in result.skip_reason
    assert "ke_initial" in result.skip_reason
    assert "ke_final" in result.skip_reason
    assert "properties." in result.skip_reason


def test_skip_reason_template_constant_across_invocations() -> None:
    """Per D0-19, the skip_reason template is constant — no per-row
    value interpolation. Two invocations with the same dataset_name but
    different KE endpoints must produce IDENTICAL skip_reason strings.
    """
    rollout_a = _build_dissipative_rollout(dataset_name="tgv2d", decay_rate=0.5)
    rollout_b = _build_dissipative_rollout(dataset_name="tgv2d", decay_rate=2.0)
    reason_a = energy_drift(rollout_a).skip_reason
    reason_b = energy_drift(rollout_b).skip_reason
    assert reason_a is not None and reason_b is not None
    assert reason_a == reason_b, (
        "D0-19 contract violation: skip_reason varies across invocations "
        "with different KE endpoints. Per-row varying values must move to "
        "SARIF properties.ke_initial / ke_final, not interpolate into the "
        "reason string."
    )
```

- [ ] **Step 2: Run the updated test, verify FAIL on current code**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/_harness/tests/test_d0_18_dissipative_skip.py::test_skip_reason_signposts_ke_property_fields external_validation/_rollout_anchors/_harness/tests/test_d0_18_dissipative_skip.py::test_skip_reason_template_constant_across_invocations -v
```

Expected: both FAIL. The first fails on `assert "ke_initial" in result.skip_reason` (current template doesn't mention `ke_initial`). The second fails on `assert reason_a == reason_b` (current template interpolates `e0` / `e_series[-1]` differently across the two rollouts).

- [ ] **Step 3: Verify the rest of test_d0_18_dissipative_skip.py still passes**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/_harness/tests/test_d0_18_dissipative_skip.py -v --deselect external_validation/_rollout_anchors/_harness/tests/test_d0_18_dissipative_skip.py::test_skip_reason_signposts_ke_property_fields --deselect external_validation/_rollout_anchors/_harness/tests/test_d0_18_dissipative_skip.py::test_skip_reason_template_constant_across_invocations
```

Expected: all other tests pass. The pre-existing `test_skip_reason_names_dataset_for_audit_trail` (line 251-259) still passes because the new template will retain `dataset_name` interpolation.

### T2.2: Update energy_drift to use the template-constant skip_reason

- [ ] **Step 1: Edit `particle_rollout_adapter.py:energy_drift`**

Open `/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/_harness/particle_rollout_adapter.py`. Find the dissipative-system skip-with-reason block (around lines 507-519, the `if system_class == "dissipative" and is_monotone_decreasing:` branch). Replace its skip_reason argument:

```python
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"system_class='dissipative' (dataset={dataset_name!r}) and "
                f"KE(t) monotone-non-increasing across the rollout (KE(0)={e0:.3e}, "
                f"KE(end)={float(e_series[-1]):.3e}); relative drift is a "
                f"misfire for dissipative-by-design systems where the "
                f"dissipation magnitude IS the physics. See "
                f"DECISIONS.md D0-18; consult dissipation_sign_violation "
                f"for the load-bearing test on this system class."
            ),
        )
```

with:

```python
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"system_class='dissipative' (dataset={dataset_name!r}); "
                "KE(t) monotone-non-increasing across the rollout; "
                "see properties.ke_initial / ke_final for values; "
                "relative drift is a misfire for dissipative-by-design "
                "systems where the dissipation magnitude IS the physics. "
                "See DECISIONS.md D0-18; consult dissipation_sign_violation "
                "for the load-bearing test on this system class."
            ),
        )
```

The change strips two f-string interpolations (`KE(0)={e0:.3e}` and `KE(end)={float(e_series[-1]):.3e}`), splits the long sentence on the same comma, and inserts the property-pointer phrase. `dataset_name` interpolation is preserved (constant per stack, satisfies the guaranteed-identical contract within a run).

- [ ] **Step 2: Run the two new tests, verify PASS**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/_harness/tests/test_d0_18_dissipative_skip.py::test_skip_reason_signposts_ke_property_fields external_validation/_rollout_anchors/_harness/tests/test_d0_18_dissipative_skip.py::test_skip_reason_template_constant_across_invocations -v
```

Expected: both PASS.

- [ ] **Step 3: Run the full test_d0_18_dissipative_skip.py suite, verify ALL PASS**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/_harness/tests/test_d0_18_dissipative_skip.py -v
```

Expected: 12 tests pass (the original 11 minus the replaced one plus the two new ones = 12).

- [ ] **Step 4: Run the full _harness test suite as a regression guard**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/_harness/tests/ -v
```

Expected: all tests pass (95+ green, including the read-only-path tests, mesh tests, lagrangebench_pkl_to_npz tests, etc.).

- [ ] **Step 5: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/particle_rollout_adapter.py external_validation/_rollout_anchors/_harness/tests/test_d0_18_dissipative_skip.py && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git commit -m "$(cat <<'EOF'
_harness/energy_drift: template-constant skip_reason per D0-19

Strip per-row varying KE values from the dissipative-system skip_reason
string; signpost the dedicated SARIF properties (ke_initial, ke_final)
where the values now live (attached at SARIF emission time by the new
lint_npz_dir module landing in T4). Preserves dataset_name interpolation
(constant per stack — satisfies the guaranteed-identical contract).

test_d0_18_dissipative_skip.py: replace test_skip_reason_includes_ke_endpoints
with test_skip_reason_signposts_ke_property_fields +
test_skip_reason_template_constant_across_invocations. Two-half-gate
truth-table tests unaffected.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: sarif_emitter.py extension for run-level properties (TDD)

**Files:**
- Modify: `external_validation/_rollout_anchors/_harness/sarif_emitter.py:74-105` (the `emit_sarif` function)
- Create: `external_validation/_rollout_anchors/_harness/tests/test_sarif_emitter_run_properties.py`

**Why:** Current `emit_sarif` only writes results; D0-19 requires run-level properties (the 10 fields). Extension is additive — existing call sites (the controlled-fixture test) keep working without passing run_properties.

### T3.1: Write failing test

- [ ] **Step 1: Create the new test file**

Create `/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/_harness/tests/test_sarif_emitter_run_properties.py` with content:

```python
"""Tests for sarif_emitter.py's run-level properties extension (D0-19).

Per DECISIONS.md D0-19, harness SARIF artifacts must carry 10 run-level
fields constant per artifact (source, schema_version, three stage shas,
LB sha, four IDs). emit_sarif accepts an optional run_properties dict
and writes it to runs[0].properties. Existing call sites that don't
pass run_properties continue to work (additive change).
"""

from __future__ import annotations

import json
from pathlib import Path

from external_validation._rollout_anchors._harness.sarif_emitter import (
    HarnessResult,
    emit_sarif,
)


def _one_result() -> HarnessResult:
    """Minimal HarnessResult fixture for testing emit_sarif's run-level path."""
    return HarnessResult(
        rule_id="harness:mass_conservation_defect",
        level="note",
        message="raw_value=0.000e+00",
        raw_value=0.0,
        case_study="01-lagrangebench",
        dataset="tgv2d",
        model="segnn",
        ckpt_hash="synthetic_ckpt",
    )


def test_emit_sarif_accepts_run_properties_kwarg(tmp_path: Path) -> None:
    """emit_sarif takes run_properties as a keyword argument and writes
    them to runs[0].properties verbatim.
    """
    out = tmp_path / "out.sarif"
    run_props = {
        "source": "rollout-anchor-harness",
        "harness_sarif_schema_version": "1.0",
        "physics_lint_sha_pkl_inference": "synthetic_inference",
        "physics_lint_sha_npz_conversion": "synthetic_conversion",
        "physics_lint_sha_sarif_emission": "synthetic_emission",
        "lagrangebench_sha": "synthetic_lb",
        "checkpoint_id": "synthetic_ckpt_id",
        "model_name": "segnn",
        "dataset_name": "tgv2d",
        "rollout_subdir": "/vol/rollouts/synthetic/",
    }
    emit_sarif([_one_result()], output_path=out, run_properties=run_props)
    sarif = json.loads(out.read_text())
    assert "properties" in sarif["runs"][0]
    assert sarif["runs"][0]["properties"] == run_props


def test_emit_sarif_omits_run_properties_when_not_passed(tmp_path: Path) -> None:
    """Backwards compatibility: existing call sites that don't pass
    run_properties continue to work and produce a SARIF without
    runs[0].properties (or with an empty dict — the test asserts
    existence-or-empty so either implementation is acceptable, but a
    None-or-missing field MUST NOT crash the writer).
    """
    out = tmp_path / "out.sarif"
    emit_sarif([_one_result()], output_path=out)
    sarif = json.loads(out.read_text())
    # Either no key, or key present with empty dict — both acceptable.
    run_props = sarif["runs"][0].get("properties", {})
    assert run_props == {} or run_props is None or "source" not in run_props


def test_emit_sarif_run_properties_preserved_alongside_results(tmp_path: Path) -> None:
    """Run-level and result-level properties must coexist in the output;
    extending emit_sarif must not regress the existing results-writing path.
    """
    out = tmp_path / "out.sarif"
    run_props = {"source": "rollout-anchor-harness", "harness_sarif_schema_version": "1.0"}
    emit_sarif([_one_result()], output_path=out, run_properties=run_props)
    sarif = json.loads(out.read_text())
    assert sarif["runs"][0]["properties"]["source"] == "rollout-anchor-harness"
    assert len(sarif["runs"][0]["results"]) == 1
    assert sarif["runs"][0]["results"][0]["ruleId"] == "harness:mass_conservation_defect"
```

- [ ] **Step 2: Run the new tests, verify FAIL**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/_harness/tests/test_sarif_emitter_run_properties.py -v
```

Expected: `test_emit_sarif_accepts_run_properties_kwarg` and `test_emit_sarif_run_properties_preserved_alongside_results` FAIL with `TypeError: emit_sarif() got an unexpected keyword argument 'run_properties'`. `test_emit_sarif_omits_run_properties_when_not_passed` PASSES (existing emit_sarif works fine without run_properties).

### T3.2: Extend emit_sarif

- [ ] **Step 1: Edit sarif_emitter.py:emit_sarif**

Open `/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/_harness/sarif_emitter.py`. Replace the entire `emit_sarif` function (lines ~74-105):

```python
def emit_sarif(
    results: list[HarnessResult],
    *,
    output_path: Path | str,
    tool_name: str = "physics-lint-rollout-anchor-harness",
    tool_version: str = "0.1.0",
) -> Path:
    """Write `results` to `output_path` in SARIF v2.1.0 format.

    Returns the absolute path written.
    """
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    sarif: dict[str, Any] = {
        "$schema": _SARIF_SCHEMA_URI,
        "version": _SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": tool_name,
                        "version": tool_version,
                        "informationUri": "https://github.com/tyy0811/physics-lint",
                    }
                },
                "results": [r.to_sarif_result() for r in results],
            }
        ],
    }
    out.write_text(json.dumps(sarif, indent=2, sort_keys=True))
    return out
```

with:

```python
def emit_sarif(
    results: list[HarnessResult],
    *,
    output_path: Path | str,
    tool_name: str = "physics-lint-rollout-anchor-harness",
    tool_version: str = "0.1.0",
    run_properties: dict[str, Any] | None = None,
) -> Path:
    """Write `results` to `output_path` in SARIF v2.1.0 format.

    `run_properties` (D0-19): optional dict written verbatim to
    `runs[0].properties`. When None, runs[0].properties is omitted.
    Callers that emit harness SARIF for rung 4a+ pass the 10 D0-19
    run-level fields here; pre-D0-19 call sites omit and continue to work.

    Returns the absolute path written.
    """
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    run: dict[str, Any] = {
        "tool": {
            "driver": {
                "name": tool_name,
                "version": tool_version,
                "informationUri": "https://github.com/tyy0811/physics-lint",
            }
        },
        "results": [r.to_sarif_result() for r in results],
    }
    if run_properties is not None:
        run["properties"] = run_properties

    sarif: dict[str, Any] = {
        "$schema": _SARIF_SCHEMA_URI,
        "version": _SARIF_VERSION,
        "runs": [run],
    }
    out.write_text(json.dumps(sarif, indent=2, sort_keys=True))
    return out
```

- [ ] **Step 2: Run the three new tests, verify PASS**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/_harness/tests/test_sarif_emitter_run_properties.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 3: Run regression guard on existing harness test suite**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/_harness/tests/ -v
```

Expected: all tests pass. The existing `test_harness_vs_public_api.py` (or whatever calls emit_sarif without run_properties) is unaffected.

- [ ] **Step 4: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/sarif_emitter.py external_validation/_rollout_anchors/_harness/tests/test_sarif_emitter_run_properties.py && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git commit -m "$(cat <<'EOF'
_harness/sarif_emitter: accept run_properties dict (D0-19)

Additive extension: emit_sarif now takes an optional run_properties
keyword argument; when present, written verbatim to runs[0].properties.
Pre-D0-19 call sites that omit it continue to work unchanged
(runs[0].properties simply not emitted in that case).

3 new tests in test_sarif_emitter_run_properties.py: accepts kwarg,
omits when not passed (backwards-compat), preserved alongside results.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: lint_npz_dir.py module + tests (TDD)

**Files:**
- Create: `external_validation/_rollout_anchors/_harness/lint_npz_dir.py`
- Create: `external_validation/_rollout_anchors/_harness/tests/test_lint_npz_dir.py`

**Why:** The bridge between the existing per-rollout defects and the SARIF emission. Reads a directory of npzs, invokes the 3 conservation defects on each, builds HarnessResult rows, attaches `ke_initial` / `ke_final` to `harness:energy_drift` SKIP rows.

### T4.1: Test happy path: empty inputs / directory existence

- [ ] **Step 1: Create test file with the empty-dir test**

Create `/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/_harness/tests/test_lint_npz_dir.py`:

```python
"""Tests for lint_npz_dir.py: generic npz-dir → HarnessResults bridge.

Per DECISIONS.md D0-19, this module reads a directory of
particle_rollout_traj{NN}.npz files, invokes the three conservation
defects (mass_conservation_defect, energy_drift, dissipation_sign_violation)
on each, and returns a list[HarnessResult] suitable for emit_sarif.

Energy_drift SKIP rows get ke_initial / ke_final attached to
extra_properties (the per-row varying values that D0-19's
template-constant skip_reason no longer interpolates).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from external_validation._rollout_anchors._harness.lint_npz_dir import (
    EmptyNpzDirectory,
    lint_npz_dir,
)
from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    ParticleRollout,
    save_rollout_npz,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dissipative_rollout(dataset_name: str = "tgv2d") -> ParticleRollout:
    """Decaying-KE rollout — fires the D0-18 dissipative SKIP path."""
    rng = np.random.default_rng(42)
    n_t, n_p = 8, 4
    dt = 0.01
    positions = np.zeros((n_t, n_p, 2), dtype=float)
    velocities = np.zeros((n_t, n_p, 2), dtype=float)
    v0 = rng.normal(scale=1.0, size=(n_p, 2))
    for t in range(n_t):
        velocities[t] = v0 * np.exp(-0.5 * t * dt)
        positions[t] = 0.5
    return ParticleRollout(
        positions=positions,
        velocities=velocities,
        particle_type=np.zeros(n_p, dtype=np.int32),
        particle_mass=np.ones(n_p, dtype=np.float64),
        dt=dt,
        domain_box=np.array([[0.0, 0.0], [1.0, 1.0]]),
        metadata={"dataset": dataset_name},
    )


def _save_n_npzs(tmp_path: Path, n: int = 3, dataset_name: str = "tgv2d") -> Path:
    """Persist `n` rollouts as particle_rollout_traj{NN}.npz files in tmp_path."""
    for i in range(n):
        rollout = _make_dissipative_rollout(dataset_name=dataset_name)
        save_rollout_npz(rollout, tmp_path / f"particle_rollout_traj{i:02d}.npz")
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Directory presence / emptiness
# ---------------------------------------------------------------------------


def test_lint_npz_dir_raises_on_empty_dir(tmp_path: Path) -> None:
    """An empty directory raises EmptyNpzDirectory rather than returning
    an empty list. Silent empty SARIF is a methodology hazard
    (writeup table renders blank with no error).
    """
    with pytest.raises(EmptyNpzDirectory):
        lint_npz_dir(tmp_path)
```

- [ ] **Step 2: Run, verify FAIL with ImportError**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/_harness/tests/test_lint_npz_dir.py -v
```

Expected: collection FAILs with `ImportError: cannot import name 'lint_npz_dir' from ...`.

### T4.2: Create lint_npz_dir module skeleton with EmptyNpzDirectory

- [ ] **Step 1: Create lint_npz_dir.py with minimal skeleton**

Create `/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/_harness/lint_npz_dir.py`:

```python
"""Generic npz-dir → HarnessResults bridge for harness SARIF emission.

Per DECISIONS.md D0-19, this module is the bridge between per-rollout
defects (computed in particle_rollout_adapter.py) and SARIF result rows
(emitted by sarif_emitter.py). Reads particle_rollout_traj*.npz files
from a directory, invokes the 3 conservation defects on each, builds
HarnessResult rows with the appropriate per-row metadata.

For harness:energy_drift SKIP rows specifically, the per-row varying
KE endpoint values (which D0-19's template-constant skip_reason no
longer interpolates) are recomputed from the rollout and attached to
the HarnessResult's extra_properties as ke_initial / ke_final.

This module knows about the harness defects and the SARIF result
shape; it does NOT know about the case study (model_name, dataset_name,
checkpoint_id, etc. are passed in by the case-study driver). The
case-study driver assembles the run-level properties separately and
calls emit_sarif.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    ParticleRollout,
    dissipation_sign_violation,
    energy_drift,
    kinetic_energy_series,
    load_rollout_npz,
    mass_conservation_defect,
)
from external_validation._rollout_anchors._harness.sarif_emitter import HarnessResult


class EmptyNpzDirectory(Exception):
    """Raised when lint_npz_dir is invoked on a directory containing no
    particle_rollout_traj*.npz files. Silent empty SARIF is a
    methodology hazard.
    """


# Defect functions in their canonical emission order.
# Order matters for downstream SARIF row ordering (deterministic).
_DEFECTS: tuple[tuple[str, Any], ...] = (
    ("harness:mass_conservation_defect", mass_conservation_defect),
    ("harness:energy_drift", energy_drift),
    ("harness:dissipation_sign_violation", dissipation_sign_violation),
)


def lint_npz_dir(
    npz_dir: Path | str,
    *,
    case_study: str = "",
    dataset: str = "",
    model: str = "",
    ckpt_hash: str = "",
) -> list[HarnessResult]:
    """Read all particle_rollout_traj*.npz files from `npz_dir`, invoke
    the 3 conservation defects on each, build HarnessResult rows.

    Per-row varying ke_initial / ke_final attached to harness:energy_drift
    SKIP rows via extra_properties (D0-19).

    Raises EmptyNpzDirectory if no matching files found.
    """
    npz_dir = Path(npz_dir)
    npz_paths = sorted(npz_dir.glob("particle_rollout_traj*.npz"))
    if not npz_paths:
        raise EmptyNpzDirectory(
            f"No particle_rollout_traj*.npz files found in {npz_dir}. "
            f"Expected at least one trajectory; run `modal volume get` to populate."
        )

    results: list[HarnessResult] = []
    for traj_index, npz_path in enumerate(npz_paths):
        rollout = load_rollout_npz(npz_path)
        for rule_id, defect_fn in _DEFECTS:
            defect = defect_fn(rollout)
            extra: dict[str, Any] = {
                "traj_index": traj_index,
                "npz_filename": npz_path.name,
            }
            if rule_id == "harness:energy_drift" and defect.value is None:
                # D0-19: per-row varying KE values move to dedicated properties.
                ke_series = kinetic_energy_series(rollout)
                extra["ke_initial"] = float(ke_series[0])
                extra["ke_final"] = float(ke_series[-1])

            if defect.value is None:
                level = "note"
                message = f"SKIP: {defect.skip_reason or '(no reason)'}"
            else:
                level = "note"
                message = f"raw_value={defect.value:.3e}"

            results.append(
                HarnessResult(
                    rule_id=rule_id,
                    level=level,
                    message=message,
                    raw_value=defect.value,
                    case_study=case_study,
                    dataset=dataset,
                    model=model,
                    ckpt_hash=ckpt_hash,
                    extra_properties=extra,
                )
            )
    return results
```

- [ ] **Step 2: Run T4.1's test, verify PASS**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/_harness/tests/test_lint_npz_dir.py::test_lint_npz_dir_raises_on_empty_dir -v
```

Expected: PASS.

### T4.3: Add happy-path test (multiple npzs → expected row count)

- [ ] **Step 1: Append test to test_lint_npz_dir.py**

Append to `/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/_harness/tests/test_lint_npz_dir.py`:

```python


# ---------------------------------------------------------------------------
# 2. Happy-path row construction
# ---------------------------------------------------------------------------


def test_lint_npz_dir_yields_three_rows_per_npz(tmp_path: Path) -> None:
    """For 3 npz files × 3 defects, expect 9 HarnessResult rows in
    deterministic ordering (sorted by traj_index, then by defect emission
    order from _DEFECTS).
    """
    _save_n_npzs(tmp_path, n=3)
    results = lint_npz_dir(tmp_path)
    assert len(results) == 9

    # Verify ordering: 3 rows per traj_index, ascending traj_index.
    # Row 0..2 from traj 0; row 3..5 from traj 1; row 6..8 from traj 2.
    for traj_idx in range(3):
        rows = results[traj_idx * 3 : (traj_idx + 1) * 3]
        assert rows[0].rule_id == "harness:mass_conservation_defect"
        assert rows[1].rule_id == "harness:energy_drift"
        assert rows[2].rule_id == "harness:dissipation_sign_violation"
        for row in rows:
            assert row.extra_properties["traj_index"] == traj_idx
            assert row.extra_properties["npz_filename"] == f"particle_rollout_traj{traj_idx:02d}.npz"
```

- [ ] **Step 2: Run, verify PASS**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/_harness/tests/test_lint_npz_dir.py::test_lint_npz_dir_yields_three_rows_per_npz -v
```

Expected: PASS.

### T4.4: Test ke_initial / ke_final attached on energy_drift SKIP rows only

- [ ] **Step 1: Append test**

Append to test file:

```python


# ---------------------------------------------------------------------------
# 3. ke_initial / ke_final attachment (D0-19 contract)
# ---------------------------------------------------------------------------


def test_energy_drift_skip_rows_have_ke_initial_and_ke_final(tmp_path: Path) -> None:
    """D0-19: harness:energy_drift SKIP rows MUST carry ke_initial and
    ke_final in extra_properties (the per-row varying values that the
    template-constant skip_reason no longer interpolates).
    """
    _save_n_npzs(tmp_path, n=2, dataset_name="tgv2d")  # tgv2d → dissipative SKIP
    results = lint_npz_dir(tmp_path)
    energy_drift_rows = [r for r in results if r.rule_id == "harness:energy_drift"]
    assert len(energy_drift_rows) == 2
    for row in energy_drift_rows:
        # Defect SKIPped → raw_value is None
        assert row.raw_value is None
        # ke_initial / ke_final present and finite
        assert "ke_initial" in row.extra_properties
        assert "ke_final" in row.extra_properties
        ke_i = row.extra_properties["ke_initial"]
        ke_f = row.extra_properties["ke_final"]
        assert isinstance(ke_i, float) and isinstance(ke_f, float)
        # Dissipative rollout: KE(end) < KE(0)
        assert ke_f < ke_i


def test_non_energy_drift_rows_do_not_have_ke_fields(tmp_path: Path) -> None:
    """ke_initial / ke_final live ONLY on harness:energy_drift rows
    (other rules don't get them per D0-19's result-level field table).
    """
    _save_n_npzs(tmp_path, n=2, dataset_name="tgv2d")
    results = lint_npz_dir(tmp_path)
    non_energy_rows = [r for r in results if r.rule_id != "harness:energy_drift"]
    for row in non_energy_rows:
        assert "ke_initial" not in row.extra_properties
        assert "ke_final" not in row.extra_properties
```

- [ ] **Step 2: Run, verify PASS**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/_harness/tests/test_lint_npz_dir.py::test_energy_drift_skip_rows_have_ke_initial_and_ke_final external_validation/_rollout_anchors/_harness/tests/test_lint_npz_dir.py::test_non_energy_drift_rows_do_not_have_ke_fields -v
```

Expected: both PASS.

### T4.5: Wrapper-preserves-D0-18-signal test (memory-driven)

- [ ] **Step 1: Append test**

Append to test file:

```python


# ---------------------------------------------------------------------------
# 4. Wrapper preserves D0-18 signal end-to-end
# ---------------------------------------------------------------------------


def test_lint_npz_dir_preserves_d0_18_skip_signal(tmp_path: Path) -> None:
    """Wrapper-preserves-contract test: lint_npz_dir doesn't muck up the
    D0-18 SKIP signal as it traverses defect → HarnessResult. A
    dissipative-system npz produces an energy_drift row with skip_reason
    set (not raw_value), confirming the wrapper preserves what the
    defect emitted.
    """
    _save_n_npzs(tmp_path, n=1, dataset_name="tgv2d")  # known dissipative
    results = lint_npz_dir(tmp_path)
    energy_drift_rows = [r for r in results if r.rule_id == "harness:energy_drift"]
    assert len(energy_drift_rows) == 1
    row = energy_drift_rows[0]
    assert row.raw_value is None  # SKIP signal preserved
    # Message documents the SKIP path
    assert "SKIP" in row.message


def test_lint_npz_dir_fires_raw_on_unknown_dataset(tmp_path: Path) -> None:
    """Regression guard: a non-LB dataset name does NOT trigger D0-18
    SKIP — wrapper must not silently classify any synthetic dataset as
    dissipative. raw_value is set; skip path not taken.
    """
    _save_n_npzs(tmp_path, n=1, dataset_name="synthetic-non-lb-name")
    results = lint_npz_dir(tmp_path)
    energy_drift_rows = [r for r in results if r.rule_id == "harness:energy_drift"]
    assert len(energy_drift_rows) == 1
    row = energy_drift_rows[0]
    assert row.raw_value is not None  # raw value emitted
```

- [ ] **Step 2: Run, verify PASS**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/_harness/tests/test_lint_npz_dir.py::test_lint_npz_dir_preserves_d0_18_skip_signal external_validation/_rollout_anchors/_harness/tests/test_lint_npz_dir.py::test_lint_npz_dir_fires_raw_on_unknown_dataset -v
```

Expected: both PASS.

### T4.6: Non-npz files in dir ignored

- [ ] **Step 1: Append test**

Append:

```python


# ---------------------------------------------------------------------------
# 5. Non-npz files ignored
# ---------------------------------------------------------------------------


def test_lint_npz_dir_ignores_non_npz_files(tmp_path: Path) -> None:
    """Files not matching particle_rollout_traj*.npz (e.g., the .pkl
    files persisted alongside, the metrics .pkl, README files) MUST be
    ignored. lint_npz_dir's glob is `particle_rollout_traj*.npz`.
    """
    _save_n_npzs(tmp_path, n=2)
    # Add some red-herring files
    (tmp_path / "rollout_0.pkl").write_text("not an npz")
    (tmp_path / "metrics2026_05_04.pkl").write_text("not an npz")
    (tmp_path / "README.md").write_text("ignored")
    (tmp_path / "particle_rollout_traj99_extra.txt").write_text("doesn't match glob")
    results = lint_npz_dir(tmp_path)
    # Only the 2 npzs × 3 defects = 6 rows.
    assert len(results) == 6
```

- [ ] **Step 2: Run, verify PASS**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/_harness/tests/test_lint_npz_dir.py::test_lint_npz_dir_ignores_non_npz_files -v
```

Expected: PASS.

### T4.7: Run full test suite + commit T4

- [ ] **Step 1: Run full _harness test suite**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/_harness/tests/ -v
```

Expected: all tests pass (the 95+ existing + the 6 new lint_npz_dir tests + the 3 new run_properties tests + the 2 net new D0-18 tests = ~106 green).

- [ ] **Step 2: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/_harness/lint_npz_dir.py external_validation/_rollout_anchors/_harness/tests/test_lint_npz_dir.py && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git commit -m "$(cat <<'EOF'
_harness/lint_npz_dir: generic npz-dir → HarnessResults bridge (D0-19)

New module bridges per-rollout defects (mass_conservation_defect,
energy_drift, dissipation_sign_violation) to SARIF HarnessResult rows.
For harness:energy_drift SKIP rows specifically, recomputes
ke_initial / ke_final from the rollout and attaches to extra_properties
— the per-row varying values that D0-19's template-constant skip_reason
no longer interpolates.

Tests (6): empty-dir raises EmptyNpzDirectory; happy-path 3-rows-per-npz
deterministic ordering; ke_initial/ke_final on energy_drift SKIP rows
only; wrapper-preserves-D0-18-signal (per memory: wrapper-preserves-
contract test pattern); regression guard on unknown dataset; non-npz
files ignored.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 01-lagrangebench/emit_sarif.py case-study driver

**Files:**
- Create: `external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py`

**Why:** Wires the 4a-specific case study (SEGNN-TGV2D + GNS-TGV2D), assembles the 10 D0-19 run-level properties for each stack, calls `lint_npz_dir` and `emit_sarif` twice. Knows about the two stacks; lint_npz_dir doesn't.

This module is intended for command-line invocation against a local mirror of the Modal Volume artifacts. Tests are minimal because integration with Modal Volume / npz files is exercised by manual invocation in T6 and by the fail-loud assertions inside lint_npz_dir.

- [ ] **Step 1: Create emit_sarif.py**

Create `/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py`:

```python
"""Rung 4a case-study driver — emit harness SARIF for SEGNN-TGV2D + GNS-TGV2D.

Per DECISIONS.md D0-19 + D0-20 + the rung-4a design doc at
`methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-design.md`:

Reads the local mirror of the Modal Volume rollout subdirs (populated
by `modal volume get`), invokes lint_npz_dir on each stack, assembles
the 10 D0-19 run-level properties, calls emit_sarif twice — producing
two committed SARIF artifacts for the rung 4a writeup.

USAGE
-----

    # 1. Populate the local mirror (one-shot, ~30 sec per stack):
    modal volume get rollout-anchors-artifacts \\
        /vol/rollouts/lagrangebench/segnn_tgv2d_8c3d080397/ \\
        external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror/segnn_tgv2d_8c3d080397/
    modal volume get rollout-anchors-artifacts \\
        /vol/rollouts/lagrangebench/gns_tgv2d_f48dd3f376/ \\
        external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror/gns_tgv2d_f48dd3f376/

    # 2. Run from physics-lint repo root:
    python external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py

    # 3. Commit the two new SARIFs at outputs/sarif/.

The emission_sha is read from `git rev-parse --short=10 HEAD` at run
time (the current feature/rollout-anchors HEAD), so the SARIF filename
matches the run-level physics_lint_sha_sarif_emission field.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from external_validation._rollout_anchors._harness.lint_npz_dir import (
    EmptyNpzDirectory,
    lint_npz_dir,
)
from external_validation._rollout_anchors._harness.sarif_emitter import emit_sarif


# Pinned shas for the rung 3.5 PASS state on Modal Volume.
# These are the genesis shas for the npz contents — they DO NOT change
# when emit_sarif.py is re-run; the SARIF's physics_lint_sha_sarif_emission
# is a third sha read from git HEAD at emission time.
SEGNN_PKL_INFERENCE_SHA = "8c3d080397"
SEGNN_NPZ_CONVERSION_SHA = "5857144"  # post-D0-17-amendment-1 standalone Modal conversion
GNS_PKL_INFERENCE_SHA = "f48dd3f376"
GNS_NPZ_CONVERSION_SHA = "f48dd3f376"  # P1 inference + conversion in one shot
LAGRANGEBENCH_SHA = "b880a6c84a93792d2499d2a9b8ba3a077ddf44e2"

HARNESS_SARIF_SCHEMA_VERSION = "1.0"

# Local mirror paths (populated by `modal volume get` before this script runs).
REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_MIRROR_ROOT = REPO_ROOT / "external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror"
SARIF_OUTPUT_ROOT = REPO_ROOT / "external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif"


class MissingLocalMirror(Exception):
    """Raised when the local mirror dir does not exist or is empty —
    user must run `modal volume get` first.
    """


def _git_short_sha() -> str:
    """Return short (10-char) sha of the current feature/rollout-anchors HEAD."""
    result = subprocess.run(
        ["git", "rev-parse", "--short=10", "HEAD"],
        capture_output=True,
        check=True,
        cwd=REPO_ROOT,
        text=True,
    )
    return result.stdout.strip()


def _build_run_properties(
    *,
    model_name: str,
    dataset_name: str,
    checkpoint_id: str,
    pkl_inference_sha: str,
    npz_conversion_sha: str,
    sarif_emission_sha: str,
    rollout_subdir_volume_path: str,
) -> dict[str, str]:
    """Assemble the 10 D0-19 run-level fields for one stack."""
    return {
        "source": "rollout-anchor-harness",
        "harness_sarif_schema_version": HARNESS_SARIF_SCHEMA_VERSION,
        "physics_lint_sha_pkl_inference": pkl_inference_sha,
        "physics_lint_sha_npz_conversion": npz_conversion_sha,
        "physics_lint_sha_sarif_emission": sarif_emission_sha,
        "lagrangebench_sha": LAGRANGEBENCH_SHA,
        "checkpoint_id": checkpoint_id,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "rollout_subdir": rollout_subdir_volume_path,
    }


def _emit_for_stack(
    *,
    mirror_subdir: Path,
    sarif_output_path: Path,
    run_properties: dict[str, str],
    case_study_name: str,
    dataset_name: str,
    model_name: str,
    checkpoint_id: str,
) -> Path:
    """Run lint_npz_dir + emit_sarif for one stack."""
    if not mirror_subdir.exists() or not any(mirror_subdir.glob("particle_rollout_traj*.npz")):
        raise MissingLocalMirror(
            f"Local mirror missing or empty at {mirror_subdir}. "
            f"Run `modal volume get rollout-anchors-artifacts /vol/rollouts/lagrangebench/<subdir>/ {mirror_subdir}/` first."
        )
    results = lint_npz_dir(
        mirror_subdir,
        case_study=case_study_name,
        dataset=dataset_name,
        model=model_name,
        ckpt_hash=checkpoint_id,
    )
    return emit_sarif(
        results,
        output_path=sarif_output_path,
        run_properties=run_properties,
    )


def main() -> int:
    sarif_emission_sha = _git_short_sha()
    SARIF_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # SEGNN-TGV2D
    segnn_mirror = LOCAL_MIRROR_ROOT / f"segnn_tgv2d_{SEGNN_PKL_INFERENCE_SHA}"
    segnn_props = _build_run_properties(
        model_name="segnn",
        dataset_name="tgv2d",
        checkpoint_id="segnn_tgv2d",
        pkl_inference_sha=SEGNN_PKL_INFERENCE_SHA,
        npz_conversion_sha=SEGNN_NPZ_CONVERSION_SHA,
        sarif_emission_sha=sarif_emission_sha,
        rollout_subdir_volume_path=f"/vol/rollouts/lagrangebench/segnn_tgv2d_{SEGNN_PKL_INFERENCE_SHA}/",
    )
    segnn_sarif_path = SARIF_OUTPUT_ROOT / f"segnn_tgv2d_{sarif_emission_sha}.sarif"
    out_segnn = _emit_for_stack(
        mirror_subdir=segnn_mirror,
        sarif_output_path=segnn_sarif_path,
        run_properties=segnn_props,
        case_study_name="01-lagrangebench",
        dataset_name="tgv2d",
        model_name="segnn",
        checkpoint_id="segnn_tgv2d",
    )
    print(f"SEGNN SARIF: {out_segnn}")

    # GNS-TGV2D
    gns_mirror = LOCAL_MIRROR_ROOT / f"gns_tgv2d_{GNS_PKL_INFERENCE_SHA}"
    gns_props = _build_run_properties(
        model_name="gns",
        dataset_name="tgv2d",
        checkpoint_id="gns_tgv2d",
        pkl_inference_sha=GNS_PKL_INFERENCE_SHA,
        npz_conversion_sha=GNS_NPZ_CONVERSION_SHA,
        sarif_emission_sha=sarif_emission_sha,
        rollout_subdir_volume_path=f"/vol/rollouts/lagrangebench/gns_tgv2d_{GNS_PKL_INFERENCE_SHA}/",
    )
    gns_sarif_path = SARIF_OUTPUT_ROOT / f"gns_tgv2d_{sarif_emission_sha}.sarif"
    out_gns = _emit_for_stack(
        mirror_subdir=gns_mirror,
        sarif_output_path=gns_sarif_path,
        run_properties=gns_props,
        case_study_name="01-lagrangebench",
        dataset_name="tgv2d",
        model_name="gns",
        checkpoint_id="gns_tgv2d",
    )
    print(f"GNS SARIF: {out_gns}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Sanity-check imports resolve**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" python -c "import importlib.util; spec = importlib.util.spec_from_file_location('m', 'external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py'); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print('imports OK; main symbol:', m.main)"
```

Expected: `imports OK; main symbol: <function main at 0x...>`. The module compiles, all imports resolve.

- [ ] **Step 3: Run with no local mirror, verify MissingLocalMirror raises**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" python external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py
```

Expected: traceback ending with `MissingLocalMirror: Local mirror missing or empty at .../segnn_tgv2d_8c3d080397. Run \`modal volume get ...\` first.` (because we haven't run modal volume get yet — that's T6).

- [ ] **Step 4: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git commit -m "$(cat <<'EOF'
01-lagrangebench/emit_sarif: rung 4a case-study driver (D0-19)

Wires the SEGNN-TGV2D + GNS-TGV2D stacks for harness SARIF emission.
Reads the local mirror dir (populated by `modal volume get` before
invocation), assembles the 10 D0-19 run-level fields per stack — note
the asymmetric sha case for SEGNN (pkl_inference_sha=8c3d080397,
npz_conversion_sha=5857144 from D0-17-amendment-1 standalone Modal
conversion, sarif_emission_sha read from git HEAD at run time) vs the
collapsed case for GNS (single inference+conversion sha at f48dd3f376).

MissingLocalMirror raised loud when the mirror is absent — pointing
at the modal volume get command to fix.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: gitignore + Modal volume get + run + commit SARIFs

**Files:**
- Modify: `.gitignore` (add the local mirror path)
- Create (locally only): `external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror/**` (gitignored)
- Create + commit: `external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/{segnn,gns}_tgv2d_<sha>.sarif`

**Why:** This is the "fire" step — runs the driver against frozen Volume artifacts and commits the produced SARIFs. Modal CLI is required for step 2.

### T6.1: gitignore the local mirror

- [ ] **Step 1: Edit .gitignore**

Open `/Users/zenith/Desktop/physics-lint/.gitignore` and append (under an existing section or as a new section):

```
# rollout-anchor harness local cache (populated by `modal volume get`)
external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror/
```

- [ ] **Step 2: Verify**

```bash
cd /Users/zenith/Desktop/physics-lint && git check-ignore -v external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror/test.txt
```

Expected: line referencing `.gitignore:NN:external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror/	external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror/test.txt`.

- [ ] **Step 3: Commit gitignore change**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add .gitignore && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git commit -m "$(cat <<'EOF'
.gitignore: 01-lagrangebench outputs/_local_mirror/

Local cache for the Modal Volume rollout artifacts populated by
`modal volume get` before running emit_sarif.py. Gitignored — only
the produced SARIF files in outputs/sarif/ are committed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### T6.2: Modal volume get for both stacks

- [ ] **Step 1: Verify Modal CLI available**

```bash
which modal && modal --version
```

Expected: path to modal CLI + version string. If absent, ask the user to install or activate the Modal-aware venv (`! source .venv/bin/activate` then proceed).

- [ ] **Step 2: Pull SEGNN rollout from Volume**

```bash
mkdir -p /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror && cd /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror && modal volume get rollout-anchors-artifacts /vol/rollouts/lagrangebench/segnn_tgv2d_8c3d080397/ ./segnn_tgv2d_8c3d080397/
```

Expected: Modal CLI streams ~221 MB of files (20 npzs + 21 pkls + a metrics .pkl), creates `./segnn_tgv2d_8c3d080397/`. Verify with:

```bash
ls /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror/segnn_tgv2d_8c3d080397/particle_rollout_traj*.npz | wc -l
```

Expected: `20`.

- [ ] **Step 3: Pull GNS rollout from Volume**

```bash
cd /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror && modal volume get rollout-anchors-artifacts /vol/rollouts/lagrangebench/gns_tgv2d_f48dd3f376/ ./gns_tgv2d_f48dd3f376/
```

Expected: Modal CLI streams ~221 MB; create `./gns_tgv2d_f48dd3f376/`. Verify:

```bash
ls /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror/gns_tgv2d_f48dd3f376/particle_rollout_traj*.npz | wc -l
```

Expected: `20`.

### T6.3: Run emit_sarif.py and verify outputs

- [ ] **Step 1: Run the driver**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" python external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py
```

Expected output:
```
SEGNN SARIF: /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/segnn_tgv2d_<sha>.sarif
GNS SARIF: /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/gns_tgv2d_<sha>.sarif
```

where `<sha>` is the 10-char short sha of the current `feature/rollout-anchors` HEAD.

- [ ] **Step 2: Verify SARIF structure**

```bash
SHA=$(git -C /Users/zenith/Desktop/physics-lint rev-parse --short=10 HEAD) && echo "SARIF emission sha: $SHA" && python -c "import json; s = json.load(open('/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/segnn_tgv2d_${SHA}.sarif')); print('runs[0].properties.source =', s['runs'][0]['properties']['source']); print('len(runs[0].results) =', len(s['runs'][0]['results']))"
```

Expected: `runs[0].properties.source = rollout-anchor-harness` and `len(runs[0].results) = 60` (3 defects × 20 trajs).

- [ ] **Step 3: Commit the two SARIFs**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/ && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git commit -m "$(cat <<'EOF'
01-lagrangebench/outputs/sarif: rung 4a SARIF artifacts (D0-19)

Two harness-style SARIF artifacts produced by emit_sarif.py against
the frozen Modal Volume rollouts:

- segnn_tgv2d_<sha>.sarif: 60 result rows (3 defects × 20 trajs).
  pkl inference at 8c3d080397, npz conversion at 5857144 (asymmetric
  three-stage shas, exercising D0-19's may-be-distinct contract).
- gns_tgv2d_<sha>.sarif: 60 result rows. pkl + npz both at f48dd3f376
  (collapsed-shas variant).

Both: schema_version 1.0, source=rollout-anchor-harness, dataset_name
=tgv2d, model_name segnn/gns. mass_conservation_defect=0.0 across all
40 trajs; energy_drift SKIP (D0-18 dissipative path) across all 40
with template-constant skip_reason + per-row ke_initial / ke_final;
dissipation_sign_violation=0.0 across all 40.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Note the commit's sha; this becomes the canonical `physics_lint_sha_sarif_emission` value embedded in the SARIFs.

---

## Task 7: 01-lagrangebench/README.md sibling pointer

**Files:**
- Modify: `external_validation/_rollout_anchors/01-lagrangebench/README.md`

**Why:** Bidirectional findability — a reader who lands on the SARIF artifacts in `outputs/sarif/` should know about the writeup in `methodology/docs/`.

- [ ] **Step 1: Read existing README structure**

```bash
cat /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/README.md
```

Expected: existing content (case study introduction, Modal entrypoints, etc.). Identify a natural insertion point — likely a "Cross-references" or "See also" section near the top or bottom; if absent, add one as a final section.

- [ ] **Step 2: Append/insert the pointer**

Add this section (or a single line under an existing "See also" / "Cross-references" heading):

```markdown
## Cross-references

- **Cross-stack writeup (rung 4a):** [`../methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md`](../methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md) — methodology writeup over the SARIF artifacts in `outputs/sarif/`.
- **Methodology decisions (D0-19, D0-20):** [`../methodology/DECISIONS.md`](../methodology/DECISIONS.md).
```

- [ ] **Step 3: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/01-lagrangebench/README.md && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git commit -m "$(cat <<'EOF'
01-lagrangebench/README: cross-reference pointer to rung 4a writeup

Sibling-relative pointer to methodology/docs/2026-05-04-rung-4a-...md
+ methodology/DECISIONS.md. Bidirectional findability: a reader on the
SARIF artifacts side knows about the writeup; the writeup links back
to the SARIF artifacts (sibling-relative paths).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: methodology/tests/ scaffold + hand-crafted fixtures

**Files:**
- Create: `external_validation/_rollout_anchors/methodology/tests/__init__.py` (empty)
- Create: `external_validation/_rollout_anchors/methodology/tests/fixtures/__init__.py` (empty)
- Create: `external_validation/_rollout_anchors/methodology/tests/fixtures/segnn_tgv2d_fixture.sarif` (asymmetric-shas variant)
- Create: `external_validation/_rollout_anchors/methodology/tests/fixtures/gns_tgv2d_fixture.sarif` (collapsed-shas variant)

**Why:** Hand-crafted synthetic-but-realistic fixtures, never copied from production. Asymmetric-by-default for SEGNN (matches production); collapsed for GNS. `expected_table.md` is generated AFTER the renderer is written (T9.6), not now — paired regeneration with renderer.

- [ ] **Step 1: Create directories and __init__.py files**

```bash
mkdir -p /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/tests/fixtures && touch /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/tests/__init__.py && touch /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/tests/fixtures/__init__.py
```

- [ ] **Step 2: Hand-craft segnn_tgv2d_fixture.sarif**

Create `/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/tests/fixtures/segnn_tgv2d_fixture.sarif` with content (4 trajs × 3 rules = 12 rows, asymmetric three-stage shas, synthetic dataset name):

```json
{
  "$schema": "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    {
      "tool": {
        "driver": {
          "name": "physics-lint-rollout-anchor-harness",
          "version": "0.1.0",
          "informationUri": "https://github.com/tyy0811/physics-lint"
        }
      },
      "properties": {
        "source": "rollout-anchor-harness",
        "harness_sarif_schema_version": "1.0",
        "physics_lint_sha_pkl_inference": "synthetic_inference_sha",
        "physics_lint_sha_npz_conversion": "synthetic_conversion_sha",
        "physics_lint_sha_sarif_emission": "synthetic_emission_sha",
        "lagrangebench_sha": "synthetic_lb_sha",
        "checkpoint_id": "synthetic_segnn_ckpt",
        "model_name": "synthetic_segnn",
        "dataset_name": "synthetic_dissipative_d",
        "rollout_subdir": "/synthetic/vol/segnn_run/"
      },
      "results": [
        {"ruleId": "harness:mass_conservation_defect", "level": "note", "message": {"text": "raw_value=0.000e+00"}, "properties": {"source": "rollout-anchor-harness", "raw_value": 0.0, "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_segnn", "ckpt_hash": "synthetic_segnn_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 0, "npz_filename": "particle_rollout_traj00.npz"}},
        {"ruleId": "harness:energy_drift", "level": "note", "message": {"text": "SKIP: dissipative system"}, "properties": {"source": "rollout-anchor-harness", "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_segnn", "ckpt_hash": "synthetic_segnn_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 0, "npz_filename": "particle_rollout_traj00.npz", "ke_initial": 600.4, "ke_final": 0.067}},
        {"ruleId": "harness:dissipation_sign_violation", "level": "note", "message": {"text": "raw_value=0.000e+00"}, "properties": {"source": "rollout-anchor-harness", "raw_value": 0.0, "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_segnn", "ckpt_hash": "synthetic_segnn_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 0, "npz_filename": "particle_rollout_traj00.npz"}},
        {"ruleId": "harness:mass_conservation_defect", "level": "note", "message": {"text": "raw_value=0.000e+00"}, "properties": {"source": "rollout-anchor-harness", "raw_value": 0.0, "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_segnn", "ckpt_hash": "synthetic_segnn_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 1, "npz_filename": "particle_rollout_traj01.npz"}},
        {"ruleId": "harness:energy_drift", "level": "note", "message": {"text": "SKIP: dissipative system"}, "properties": {"source": "rollout-anchor-harness", "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_segnn", "ckpt_hash": "synthetic_segnn_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 1, "npz_filename": "particle_rollout_traj01.npz", "ke_initial": 598.2, "ke_final": 0.071}},
        {"ruleId": "harness:dissipation_sign_violation", "level": "note", "message": {"text": "raw_value=0.000e+00"}, "properties": {"source": "rollout-anchor-harness", "raw_value": 0.0, "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_segnn", "ckpt_hash": "synthetic_segnn_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 1, "npz_filename": "particle_rollout_traj01.npz"}},
        {"ruleId": "harness:mass_conservation_defect", "level": "note", "message": {"text": "raw_value=0.000e+00"}, "properties": {"source": "rollout-anchor-harness", "raw_value": 0.0, "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_segnn", "ckpt_hash": "synthetic_segnn_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 2, "npz_filename": "particle_rollout_traj02.npz"}},
        {"ruleId": "harness:energy_drift", "level": "note", "message": {"text": "SKIP: dissipative system"}, "properties": {"source": "rollout-anchor-harness", "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_segnn", "ckpt_hash": "synthetic_segnn_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 2, "npz_filename": "particle_rollout_traj02.npz", "ke_initial": 601.1, "ke_final": 0.064}},
        {"ruleId": "harness:dissipation_sign_violation", "level": "note", "message": {"text": "raw_value=0.000e+00"}, "properties": {"source": "rollout-anchor-harness", "raw_value": 0.0, "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_segnn", "ckpt_hash": "synthetic_segnn_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 2, "npz_filename": "particle_rollout_traj02.npz"}},
        {"ruleId": "harness:mass_conservation_defect", "level": "note", "message": {"text": "raw_value=0.000e+00"}, "properties": {"source": "rollout-anchor-harness", "raw_value": 0.0, "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_segnn", "ckpt_hash": "synthetic_segnn_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 3, "npz_filename": "particle_rollout_traj03.npz"}},
        {"ruleId": "harness:energy_drift", "level": "note", "message": {"text": "SKIP: dissipative system"}, "properties": {"source": "rollout-anchor-harness", "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_segnn", "ckpt_hash": "synthetic_segnn_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 3, "npz_filename": "particle_rollout_traj03.npz", "ke_initial": 599.8, "ke_final": 0.069}},
        {"ruleId": "harness:dissipation_sign_violation", "level": "note", "message": {"text": "raw_value=0.000e+00"}, "properties": {"source": "rollout-anchor-harness", "raw_value": 0.0, "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_segnn", "ckpt_hash": "synthetic_segnn_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 3, "npz_filename": "particle_rollout_traj03.npz"}}
      ]
    }
  ]
}
```

(4 trajs is sufficient to exercise the schema; 60 rows would be redundant in fixture form. Renderer's "all-N-identical" detection works at any N ≥ 2.)

- [ ] **Step 3: Hand-craft gns_tgv2d_fixture.sarif (collapsed-shas variant)**

Create `/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/tests/fixtures/gns_tgv2d_fixture.sarif` — same structure but with `model_name=synthetic_gns`, `checkpoint_id=synthetic_gns_ckpt`, and the three sha fields collapsed to a single value (`synthetic_combined_sha` for both inference and conversion; `synthetic_emission_sha` for emission):

```json
{
  "$schema": "https://schemastore.azurewebsites.net/schemas/json/sarif-2.1.0.json",
  "version": "2.1.0",
  "runs": [
    {
      "tool": {
        "driver": {
          "name": "physics-lint-rollout-anchor-harness",
          "version": "0.1.0",
          "informationUri": "https://github.com/tyy0811/physics-lint"
        }
      },
      "properties": {
        "source": "rollout-anchor-harness",
        "harness_sarif_schema_version": "1.0",
        "physics_lint_sha_pkl_inference": "synthetic_combined_sha",
        "physics_lint_sha_npz_conversion": "synthetic_combined_sha",
        "physics_lint_sha_sarif_emission": "synthetic_emission_sha",
        "lagrangebench_sha": "synthetic_lb_sha",
        "checkpoint_id": "synthetic_gns_ckpt",
        "model_name": "synthetic_gns",
        "dataset_name": "synthetic_dissipative_d",
        "rollout_subdir": "/synthetic/vol/gns_run/"
      },
      "results": [
        {"ruleId": "harness:mass_conservation_defect", "level": "note", "message": {"text": "raw_value=0.000e+00"}, "properties": {"source": "rollout-anchor-harness", "raw_value": 0.0, "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_gns", "ckpt_hash": "synthetic_gns_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 0, "npz_filename": "particle_rollout_traj00.npz"}},
        {"ruleId": "harness:energy_drift", "level": "note", "message": {"text": "SKIP: dissipative system"}, "properties": {"source": "rollout-anchor-harness", "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_gns", "ckpt_hash": "synthetic_gns_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 0, "npz_filename": "particle_rollout_traj00.npz", "ke_initial": 600.4, "ke_final": 0.084}},
        {"ruleId": "harness:dissipation_sign_violation", "level": "note", "message": {"text": "raw_value=0.000e+00"}, "properties": {"source": "rollout-anchor-harness", "raw_value": 0.0, "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_gns", "ckpt_hash": "synthetic_gns_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 0, "npz_filename": "particle_rollout_traj00.npz"}},
        {"ruleId": "harness:mass_conservation_defect", "level": "note", "message": {"text": "raw_value=0.000e+00"}, "properties": {"source": "rollout-anchor-harness", "raw_value": 0.0, "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_gns", "ckpt_hash": "synthetic_gns_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 1, "npz_filename": "particle_rollout_traj01.npz"}},
        {"ruleId": "harness:energy_drift", "level": "note", "message": {"text": "SKIP: dissipative system"}, "properties": {"source": "rollout-anchor-harness", "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_gns", "ckpt_hash": "synthetic_gns_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 1, "npz_filename": "particle_rollout_traj01.npz", "ke_initial": 598.2, "ke_final": 0.082}},
        {"ruleId": "harness:dissipation_sign_violation", "level": "note", "message": {"text": "raw_value=0.000e+00"}, "properties": {"source": "rollout-anchor-harness", "raw_value": 0.0, "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_gns", "ckpt_hash": "synthetic_gns_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 1, "npz_filename": "particle_rollout_traj01.npz"}},
        {"ruleId": "harness:mass_conservation_defect", "level": "note", "message": {"text": "raw_value=0.000e+00"}, "properties": {"source": "rollout-anchor-harness", "raw_value": 0.0, "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_gns", "ckpt_hash": "synthetic_gns_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 2, "npz_filename": "particle_rollout_traj02.npz"}},
        {"ruleId": "harness:energy_drift", "level": "note", "message": {"text": "SKIP: dissipative system"}, "properties": {"source": "rollout-anchor-harness", "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_gns", "ckpt_hash": "synthetic_gns_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 2, "npz_filename": "particle_rollout_traj02.npz", "ke_initial": 601.1, "ke_final": 0.085}},
        {"ruleId": "harness:dissipation_sign_violation", "level": "note", "message": {"text": "raw_value=0.000e+00"}, "properties": {"source": "rollout-anchor-harness", "raw_value": 0.0, "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_gns", "ckpt_hash": "synthetic_gns_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 2, "npz_filename": "particle_rollout_traj02.npz"}},
        {"ruleId": "harness:mass_conservation_defect", "level": "note", "message": {"text": "raw_value=0.000e+00"}, "properties": {"source": "rollout-anchor-harness", "raw_value": 0.0, "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_gns", "ckpt_hash": "synthetic_gns_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 3, "npz_filename": "particle_rollout_traj03.npz"}},
        {"ruleId": "harness:energy_drift", "level": "note", "message": {"text": "SKIP: dissipative system"}, "properties": {"source": "rollout-anchor-harness", "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_gns", "ckpt_hash": "synthetic_gns_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 3, "npz_filename": "particle_rollout_traj03.npz", "ke_initial": 599.8, "ke_final": 0.083}},
        {"ruleId": "harness:dissipation_sign_violation", "level": "note", "message": {"text": "raw_value=0.000e+00"}, "properties": {"source": "rollout-anchor-harness", "raw_value": 0.0, "case_study": "01-lagrangebench", "dataset": "synthetic_dissipative_d", "model": "synthetic_gns", "ckpt_hash": "synthetic_gns_ckpt", "harness_validation_passed": null, "harness_vs_public_epsilon": null, "traj_index": 3, "npz_filename": "particle_rollout_traj03.npz"}}
      ]
    }
  ]
}
```

- [ ] **Step 4: Verify both fixtures parse as JSON**

```bash
python -c "import json; print('segnn:', len(json.load(open('/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/tests/fixtures/segnn_tgv2d_fixture.sarif'))['runs'][0]['results'])); print('gns:', len(json.load(open('/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/tests/fixtures/gns_tgv2d_fixture.sarif'))['runs'][0]['results']))"
```

Expected: `segnn: 12` and `gns: 12`.

- [ ] **Step 5: Commit fixtures + scaffolding**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/methodology/tests/ && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git commit -m "$(cat <<'EOF'
methodology/tests: scaffold + hand-crafted SARIF fixtures (D0-20)

methodology/tests/__init__.py + fixtures/__init__.py.

Two hand-crafted synthetic-but-realistic SARIF fixtures for the
renderer's golden tests (per memory: never copy production artifacts
into fixtures):

- segnn_tgv2d_fixture.sarif: ASYMMETRIC three-stage shas
  (synthetic_inference_sha, synthetic_conversion_sha,
  synthetic_emission_sha) reflecting production SEGNN's multi-session
  genesis; exercises D0-19's "shas may be distinct" branch.
- gns_tgv2d_fixture.sarif: COLLAPSED shas (single
  synthetic_combined_sha for inference+conversion, separate
  emission_sha) reflecting production GNS's single-shot run;
  exercises D0-19's "shas may be identical" branch.

4 trajs × 3 rules = 12 result rows per fixture (sufficient to
exercise schema; 60 would be redundant). Synthetic dataset_name
('synthetic_dissipative_d') decouples fixture from LB-specific names.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: methodology/tools/render_cross_stack_table.py with TDD

**Files:**
- Create: `external_validation/_rollout_anchors/methodology/tools/__init__.py` (empty)
- Create: `external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py`
- Create: `external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py`
- Create: `external_validation/_rollout_anchors/methodology/tests/fixtures/expected_table.md` (after T9.6)

**Why:** D0-20's renderer. Generator-vs-consumer separation: this module imports nothing from `_harness/` or `01-lagrangebench/`. Asserts on schema_version, source-tag, run-level field presence; raises loud on mismatch. Aggregates per-traj rows into "all N identical → single cell" view.

### T9.1: Write the version-mismatch test (drives the SchemaVersionMismatch exception class)

- [ ] **Step 1: Create tools/__init__.py and test file**

```bash
mkdir -p /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/tools && touch /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/tools/__init__.py
```

Create `/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py`:

```python
"""Tests for methodology/tools/render_cross_stack_table.py.

Per DECISIONS.md D0-20: renderer asserts schema_version + source-tag +
run-level field presence on every input SARIF; raises loud on
mismatch. Tests use hand-crafted fixtures (per memory: never copy
production artifacts).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from external_validation._rollout_anchors.methodology.tools.render_cross_stack_table import (
    EXPECTED_SCHEMA_VERSION,
    MissingRunLevelField,
    SchemaVersionMismatch,
    SourceTagMismatch,
    render_cross_stack_table,
)


FIXTURES_DIR = Path(__file__).parent / "fixtures"
SEGNN_FIXTURE = FIXTURES_DIR / "segnn_tgv2d_fixture.sarif"
GNS_FIXTURE = FIXTURES_DIR / "gns_tgv2d_fixture.sarif"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(d: dict, path: Path) -> None:
    path.write_text(json.dumps(d, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# 1. Schema-version assertion
# ---------------------------------------------------------------------------


def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    """Bumped harness_sarif_schema_version → SchemaVersionMismatch raises.
    Programmatically-derived from the canonical fixture (per memory:
    don't commit a separate bumped-version fixture file).
    """
    bumped = copy.deepcopy(_load(SEGNN_FIXTURE))
    bumped["runs"][0]["properties"]["harness_sarif_schema_version"] = "99.0"
    bumped_path = tmp_path / "bumped.sarif"
    _write(bumped, bumped_path)

    with pytest.raises(SchemaVersionMismatch):
        render_cross_stack_table([bumped_path, GNS_FIXTURE])
```

- [ ] **Step 2: Run, verify FAIL with ImportError**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py -v
```

Expected: collection FAILs with `ImportError: cannot import name 'render_cross_stack_table' from ...`.

### T9.2: Skeleton the renderer with the assertion classes

- [ ] **Step 1: Create render_cross_stack_table.py minimal skeleton**

Create `/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py`:

```python
"""Render the cross-stack conservation table from harness SARIF artifacts.

Per DECISIONS.md D0-20 + the rung-4a design at
`methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-design.md`:

Reads N harness SARIFs (one per stack), asserts schema_version +
source-tag + run-level field presence, aggregates per-traj rows per
(rule, stack) — detecting "all N identical" specially — and emits a
markdown table to stdout.

Generator-vs-consumer separation: this module imports nothing from
`_harness/` or `01-lagrangebench/`. The SARIF schema is the wire
protocol; harness_sarif_schema_version is asserted on read.

INVOKE FROM REPO ROOT:

    python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py \\
        --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/

The rendered table is what the rung-4a writeup includes via copy-paste,
plus the rederivability footer that records the exact command + sha.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Pinned by D0-19. Bump when SCHEMA.md §3.x bumps.
EXPECTED_SCHEMA_VERSION = "1.0"
EXPECTED_SOURCE_TAG = "rollout-anchor-harness"

# Required run-level fields per D0-19 §3.1.
REQUIRED_RUN_LEVEL_FIELDS: tuple[str, ...] = (
    "source",
    "harness_sarif_schema_version",
    "physics_lint_sha_pkl_inference",
    "physics_lint_sha_npz_conversion",
    "physics_lint_sha_sarif_emission",
    "lagrangebench_sha",
    "checkpoint_id",
    "model_name",
    "dataset_name",
    "rollout_subdir",
)


class SchemaVersionMismatch(Exception):
    """Raised when a SARIF's harness_sarif_schema_version doesn't match
    EXPECTED_SCHEMA_VERSION. The renderer's contract is bound to the
    expected version; mismatch means the renderer might silently emit a
    wrong table on a schema-bumped artifact. Fail loud.
    """


class SourceTagMismatch(Exception):
    """Raised when a SARIF's source-tag is not 'rollout-anchor-harness'.
    Distinguishes harness SARIF from public-API SARIF reaching the
    renderer by accident.
    """


class MissingRunLevelField(Exception):
    """Raised when a SARIF is missing one or more of the 10 required
    D0-19 run-level fields. No defaulting.
    """


def _assert_run_level(sarif: dict[str, Any], src_path: Path) -> dict[str, Any]:
    """Apply the three D0-20 fail-loud assertions on a SARIF.

    Returns the run-level properties dict.
    """
    runs = sarif.get("runs", [])
    if not runs:
        raise MissingRunLevelField(
            f"{src_path}: SARIF has no runs[]; D0-19 requires runs[0] with properties."
        )
    properties = runs[0].get("properties", {})

    missing = [f for f in REQUIRED_RUN_LEVEL_FIELDS if f not in properties]
    if missing:
        raise MissingRunLevelField(
            f"{src_path}: missing required D0-19 run-level fields: {missing}. "
            f"See SCHEMA.md §3.x."
        )

    if properties["source"] != EXPECTED_SOURCE_TAG:
        raise SourceTagMismatch(
            f"{src_path}: expected source={EXPECTED_SOURCE_TAG!r}, "
            f"got {properties['source']!r}."
        )

    if properties["harness_sarif_schema_version"] != EXPECTED_SCHEMA_VERSION:
        raise SchemaVersionMismatch(
            f"{src_path}: expected harness_sarif_schema_version="
            f"{EXPECTED_SCHEMA_VERSION!r}, got {properties['harness_sarif_schema_version']!r}. "
            f"See SCHEMA.md §3.x."
        )

    return properties


def render_cross_stack_table(sarif_paths: Iterable[Path | str]) -> str:
    """Read each SARIF in sarif_paths, assert D0-19 contract, aggregate,
    return a markdown table string.
    """
    paths = [Path(p) for p in sarif_paths]
    if not paths:
        raise MissingRunLevelField("render_cross_stack_table: no SARIF paths provided.")

    stacks: list[tuple[Path, dict[str, Any], list[dict[str, Any]]]] = []
    for path in paths:
        sarif = json.loads(path.read_text())
        run_props = _assert_run_level(sarif, path)
        results = sarif["runs"][0].get("results", [])
        stacks.append((path, run_props, results))

    # Implementation of aggregation + markdown emission lands in T9.4.
    # For T9.2, skeleton just returns the assertions-passed marker.
    return "PLACEHOLDER_T9_2_SKELETON_AGGREGATION_LANDS_IN_T9_4"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sarif-dir",
        type=Path,
        required=True,
        help="Directory containing the harness SARIF files (e.g., outputs/sarif/).",
    )
    args = parser.parse_args(argv)
    sarif_paths = sorted(args.sarif_dir.glob("*.sarif"))
    if not sarif_paths:
        print(f"No .sarif files found in {args.sarif_dir}", file=sys.stderr)
        return 2
    print(render_cross_stack_table(sarif_paths))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

The placeholder string `PLACEHOLDER_T9_2_SKELETON_AGGREGATION_LANDS_IN_T9_4` is intentional and is replaced with real aggregation in T9.4. T9.1's test only exercises the SchemaVersionMismatch path which fires before the aggregation runs, so the placeholder is fine here.

- [ ] **Step 2: Run T9.1's test, verify PASS**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py::test_schema_version_mismatch_raises -v
```

Expected: PASS.

### T9.3: Tests for SourceTagMismatch and MissingRunLevelField

- [ ] **Step 1: Append tests**

Append to `test_render_cross_stack_table.py`:

```python


def test_source_tag_mismatch_raises(tmp_path: Path) -> None:
    """Wrong source field → SourceTagMismatch raises."""
    bad = copy.deepcopy(_load(SEGNN_FIXTURE))
    bad["runs"][0]["properties"]["source"] = "physics-lint-public-api"
    bad_path = tmp_path / "bad_source.sarif"
    _write(bad, bad_path)

    with pytest.raises(SourceTagMismatch):
        render_cross_stack_table([bad_path, GNS_FIXTURE])


def test_missing_run_level_field_raises(tmp_path: Path) -> None:
    """Deleting any required D0-19 run-level field → MissingRunLevelField raises."""
    incomplete = copy.deepcopy(_load(SEGNN_FIXTURE))
    del incomplete["runs"][0]["properties"]["physics_lint_sha_pkl_inference"]
    incomplete_path = tmp_path / "incomplete.sarif"
    _write(incomplete, incomplete_path)

    with pytest.raises(MissingRunLevelField):
        render_cross_stack_table([incomplete_path, GNS_FIXTURE])


def test_no_sarif_paths_raises() -> None:
    """Empty input → MissingRunLevelField (chosen because the renderer
    has no run-level data to operate on; parallel category to missing
    fields).
    """
    with pytest.raises(MissingRunLevelField):
        render_cross_stack_table([])
```

- [ ] **Step 2: Run, verify all PASS**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py -v
```

Expected: 4 PASS.

### T9.4: Aggregation + markdown emission (drives the renderer's logic)

- [ ] **Step 1: Append the asymmetric-shas test and a smoke aggregation test**

Append to `test_render_cross_stack_table.py`:

```python


def test_renderer_handles_asymmetric_shas() -> None:
    """Per D0-19, the three sha fields may be distinct (asymmetric) or
    identical (collapsed). SEGNN fixture has three distinct shas; GNS
    fixture has collapsed shas. The renderer must NOT crash, must NOT
    require equality across stages, and must produce stable output.
    """
    table = render_cross_stack_table([SEGNN_FIXTURE, GNS_FIXTURE])
    # Renderer returns a non-empty string (the markdown table).
    assert isinstance(table, str)
    assert table != ""
    # Both shas appear in the output (asymmetric SEGNN + collapsed GNS shas).
    # Asymmetric: distinct inference and conversion shas both present.
    assert "synthetic_inference_sha" in table
    assert "synthetic_conversion_sha" in table
    # Collapsed: GNS uses the same sha for both stages.
    assert "synthetic_combined_sha" in table


def test_renderer_emits_markdown_table_with_three_rules(tmp_path: Path) -> None:
    """Smoke test: rendered output is a markdown table mentioning the
    three conservation rules.
    """
    table = render_cross_stack_table([SEGNN_FIXTURE, GNS_FIXTURE])
    assert "mass_conservation_defect" in table
    assert "energy_drift" in table
    assert "dissipation_sign_violation" in table


def test_renderer_detects_all_n_identical_aggregation(tmp_path: Path) -> None:
    """Per D0-20: 'all N identical → single cell' detection. All
    mass_conservation_defect rows in segnn_tgv2d_fixture have raw_value
    = 0.0; the rendered cell for that (rule, stack) reports a single
    value, not a min/max range.
    """
    table = render_cross_stack_table([SEGNN_FIXTURE, GNS_FIXTURE])
    # The "all 4 trajs = 0.0" cell renders as a single value.
    # We assert that "0.000e+00" or "0.0" appears at least 4 times
    # (mass=0 × 2 stacks; dissipation=0 × 2 stacks = 4 cells).
    assert table.count("0.0") >= 4 or table.count("0.000e+00") >= 4
```

- [ ] **Step 2: Run, verify all FAIL**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py::test_renderer_handles_asymmetric_shas external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py::test_renderer_emits_markdown_table_with_three_rules external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py::test_renderer_detects_all_n_identical_aggregation -v
```

Expected: 3 FAIL (the placeholder return string from T9.2 doesn't satisfy any of these assertions).

- [ ] **Step 3: Replace the placeholder return with real aggregation + markdown**

In `render_cross_stack_table.py`, replace the placeholder line:

```python
    # Implementation of aggregation + markdown emission lands in T9.4.
    # For T9.2, skeleton just returns the assertions-passed marker.
    return "PLACEHOLDER_T9_2_SKELETON_AGGREGATION_LANDS_IN_T9_4"
```

with the aggregation + emission logic:

```python
    # ----- aggregation + markdown emission -----
    # For each (rule, stack), collapse N rows to a single cell:
    # - if all rows have raw_value AND all values agree (within float-eq) → "value (×N)"
    # - if all rows are SKIP → "SKIP (×N)"
    # - mixed or non-uniform → fall through to summary stats
    rule_ids = (
        "harness:mass_conservation_defect",
        "harness:energy_drift",
        "harness:dissipation_sign_violation",
    )

    cells: dict[tuple[str, str], str] = {}
    stack_labels: list[str] = []
    sha_lines: list[str] = []
    for path, run_props, results in stacks:
        stack_label = f"{run_props['model_name']}-{run_props['dataset_name']}"
        stack_labels.append(stack_label)
        sha_lines.append(
            f"- **{stack_label}**: pkl_inference={run_props['physics_lint_sha_pkl_inference']}, "
            f"npz_conversion={run_props['physics_lint_sha_npz_conversion']}, "
            f"sarif_emission={run_props['physics_lint_sha_sarif_emission']}"
        )
        for rule_id in rule_ids:
            rule_rows = [r for r in results if r["ruleId"] == rule_id]
            if not rule_rows:
                cells[(rule_id, stack_label)] = "(no rows)"
                continue
            n = len(rule_rows)
            raw_values = [r["properties"].get("raw_value") for r in rule_rows]
            skip_present = [r["properties"].get("skip_reason") is not None or r["properties"].get("raw_value") is None for r in rule_rows]
            if all(rv is None for rv in raw_values) and all(skip_present):
                cells[(rule_id, stack_label)] = f"SKIP (×{n}, D0-18)"
            elif all(rv is not None for rv in raw_values):
                vals = [float(rv) for rv in raw_values]
                if all(abs(v - vals[0]) < 1e-15 for v in vals):
                    cells[(rule_id, stack_label)] = f"{vals[0]:.3e} (×{n} identical)"
                else:
                    cells[(rule_id, stack_label)] = (
                        f"min={min(vals):.3e}, max={max(vals):.3e}, n={n}"
                    )
            else:
                # Mixed SKIP / raw — should not happen for a single rule
                # over a single stack, but render explicitly if it does.
                cells[(rule_id, stack_label)] = f"MIXED (n={n})"

    # ----- markdown table -----
    header = ["Rule"] + stack_labels
    rows: list[list[str]] = [header]
    for rule_id in rule_ids:
        short = rule_id.replace("harness:", "")
        row = [f"`{short}`"]
        for label in stack_labels:
            row.append(cells.get((rule_id, label), "(missing)"))
        rows.append(row)

    # Markdown table
    md_lines = [
        "| " + " | ".join(rows[0]) + " |",
        "|" + "|".join(["---"] * len(rows[0])) + "|",
    ]
    for row in rows[1:]:
        md_lines.append("| " + " | ".join(row) + " |")

    # Provenance footer with the three shas per stack.
    md_lines.append("")
    md_lines.append("**Provenance (D0-19 three-sha):**")
    md_lines.append("")
    md_lines.extend(sha_lines)

    return "\n".join(md_lines) + "\n"
```

- [ ] **Step 4: Run all tests, verify PASS**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py -v
```

Expected: all 7 PASS.

### T9.5: Run renderer against fixtures, capture output as expected_table.md (golden test target)

- [ ] **Step 1: Run renderer against fixtures, capture stdout**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py --sarif-dir external_validation/_rollout_anchors/methodology/tests/fixtures/ > /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/tests/fixtures/expected_table.md
```

Expected: file created. Verify content:

```bash
cat /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/tests/fixtures/expected_table.md
```

Expected output (3 rules × 2 stacks markdown table + provenance footer with 6 sha lines).

- [ ] **Step 2: Append the golden test**

Append to `test_render_cross_stack_table.py`:

```python


def test_renderer_golden_output_matches_expected_table(tmp_path: Path) -> None:
    """Golden test: rendering the canonical fixtures produces output
    byte-for-byte identical to expected_table.md. This pins the
    renderer's contract — any non-trivial change in output requires a
    paired update to expected_table.md.
    """
    expected = (FIXTURES_DIR / "expected_table.md").read_text()
    actual = render_cross_stack_table([SEGNN_FIXTURE, GNS_FIXTURE])
    assert actual == expected, (
        f"Renderer output diverged from expected_table.md.\n"
        f"--- expected ---\n{expected}\n"
        f"--- actual ---\n{actual}\n"
        f"Regenerate by re-running:\n"
        f"  python methodology/tools/render_cross_stack_table.py "
        f"--sarif-dir methodology/tests/fixtures/ > methodology/tests/fixtures/expected_table.md"
    )
```

- [ ] **Step 3: Run, verify PASS**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py::test_renderer_golden_output_matches_expected_table -v
```

Expected: PASS (since `expected_table.md` was just generated from the renderer's output and the renderer's output is deterministic for the given fixtures).

### T9.6: Final sweep + commit

- [ ] **Step 1: Run full methodology test suite**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/methodology/tests/ -v
```

Expected: 8 tests pass.

- [ ] **Step 2: Run full _harness test suite as regression guard**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/_harness/tests/ -v
```

Expected: all _harness tests still pass.

- [ ] **Step 3: Commit renderer + tests + golden**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/methodology/tools/ external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py external_validation/_rollout_anchors/methodology/tests/fixtures/expected_table.md && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git commit -m "$(cat <<'EOF'
methodology/tools/render_cross_stack_table: D0-20 renderer + tests

Cross-stack conservation table renderer for rung 4a. Reads N harness
SARIF artifacts, asserts D0-19 contract (schema_version, source-tag,
10 required run-level fields), aggregates per-traj rows per (rule,
stack) — detecting "all N identical" specially — emits markdown table
with 3-sha provenance footer.

Generator-vs-consumer separation enforced: this module imports nothing
from _harness/ or 01-lagrangebench/. Schema version is the wire
protocol.

Tests (8): version-mismatch raises; source-tag mismatch raises;
missing-field raises (programmatic from canonical fixture);
empty-input raises; asymmetric-shas case (renderer makes no equality
assumption across the three sha stages); markdown contains the three
rules; all-N-identical aggregation; golden test (expected_table.md
matches renderer output byte-for-byte; paired-regeneration discipline).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Writeup

**Files:**
- Create: `external_validation/_rollout_anchors/methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md`

**Why:** The 4a deliverable. Frozen headline + rendered table + 5-item NOT list + rederivability footer + integrating-trigger footer.

- [ ] **Step 1: Run renderer against the committed SARIFs (NOT the test fixtures), capture output**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/
```

Expected: markdown table for the real SEGNN-TGV2D + GNS-TGV2D rollouts. Capture this output (e.g., `... > /tmp/rung_4a_table.md`) for inclusion in the writeup.

- [ ] **Step 2: Get the sarif_emission_sha for the rederivability footer**

```bash
SHA=$(git -C /Users/zenith/Desktop/physics-lint rev-parse --short=10 HEAD) && echo "Current sha: $SHA"
```

This is the sha at which the SARIFs were emitted (the T6.3 commit's parent or the T6.3 commit itself, depending on whether T6.3 is HEAD).

- [ ] **Step 3: Create the writeup**

Create `/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md`:

```markdown
# Rung 4a — Cross-stack conservation table (writeup)

**Date:** 2026-05-04
**Predecessor:** rung 3.5 PASS on both stacks (D0-18 amendment 1 implementation at `d03df3e`); npzs frozen on Modal Volume.
**Successor:** rung 4b — equivariance brainstorm session (separate, no code).
**Design doc:** [`./2026-05-04-rung-4a-cross-stack-conservation-design.md`](./2026-05-04-rung-4a-cross-stack-conservation-design.md)
**SARIF artifacts:** [`../../01-lagrangebench/outputs/sarif/`](../../01-lagrangebench/outputs/sarif/)
**Methodology pre-registrations:** [D0-19](../DECISIONS.md#d0-19--2026-05-04--harness-sarif-result-schema-rung-4a-pre-registration), [D0-20](../DECISIONS.md#d0-20--2026-05-04--generator-vs-consumer-separation-architecture-rung-4a-pre-registration)

---

## Headline

physics-lint's harness ran the same conservation rule schema, unmodified, across SEGNN-TGV2D and GNS-TGV2D rollouts of the same dissipative system. Every result row is structurally identical between the two SARIF artifacts (D0-19-enforced); D0-18's dissipative-system skip-with-reason fires identically with the same `skip_reason` string on both — per-stack KE endpoints are recorded in dedicated `properties.ke_initial` / `properties.ke_final` fields, not interpolated into the reason — and points to `dissipation_sign_violation` as the load-bearing alternative. The methodology-evolution machinery — D0-18's skip-with-reason path — is exercised end-to-end against real upstream output.

The "20 identical fires" claim above is schema-enforced, not coincidental: D0-19 §3.4 specifies that for a fixed (rule, stack), all 20 result rows MUST have identical `ruleId`, `level`, `message.text`, plus either identical `properties.raw_value` or identical `properties.skip_reason`.

---

## Cross-stack conservation table

<!--
INSTRUCTION: paste the renderer's stdout from Step 1 here. The
renderer's output looks like:

| Rule | segnn-tgv2d | gns-tgv2d |
|---|---|---|
| `mass_conservation_defect` | 0.000e+00 (×20 identical) | 0.000e+00 (×20 identical) |
| `energy_drift` | SKIP (×20, D0-18) | SKIP (×20, D0-18) |
| `dissipation_sign_violation` | 0.000e+00 (×20 identical) | 0.000e+00 (×20 identical) |

**Provenance (D0-19 three-sha):**

- **segnn-tgv2d**: pkl_inference=8c3d080397, npz_conversion=5857144, sarif_emission=<sha>
- **gns-tgv2d**: pkl_inference=f48dd3f376, npz_conversion=f48dd3f376, sarif_emission=<sha>
-->

(replace this comment block with the actual renderer output from Step 1)

---

## What rung 4a is NOT

1. **Not a SEGNN-vs-GNS model comparison.** Both stacks emit `mass_conservation_defect = 0.0`, both fire D0-18 SKIP on `energy_drift`, both emit `dissipation_sign_violation = 0.0`. Model differentiation lives in equivariance → rung 4b (separate brainstorm, separate session).

2. **Not a GitHub Security-tab integration demo.** Harness-style SARIF emits `level: "note"` rows for PASS-equivalent values; 4a has no findings to populate the Security tab meaningfully. The Security-tab demo is deferred to 4b, where equivariance is expected to produce real warning-level findings (GNS APPROXIMATE band) that exercise the rendering path. An empty Security tab is not a demo of integration.

3. **Not the integrating top-level README.** Composed when 4b's writeup lands; until then `methodology/docs/` carries dated deliverables and is the source of truth.

4. **Not a physics-lint v1.x core change.** The skip-with-reason mechanism, dissipative-system detection, and audit-trail provenance fields all live in the harness layer (`external_validation/_rollout_anchors/_harness/`), not in physics-lint v1.0's public rule path. v1.0's `master`-branch docs are amended as part of 4a to document the dissipative-system limit explicitly alongside the existing PH-BC-001 / PH-RES-001 honest limits, with wording that includes an explicit cross-branch qualifier (the harness layer currently lives on `feature/rollout-anchors` pending merge to `master`). v1.0's behavior on dissipative systems is preserved as-shipped, with the harness-layer skip-with-reason machinery flagged as the v1.x graduation prototype. The graduation itself is a future D-entry, not implied by 4a.

5. **Not a bilateral test of D0-18's mechanism.** TGV2D is dissipative, so 4a exercises the skip-fires path. The opposite path (conservative system, skip does not fire, `energy_drift` evaluates raw_value normally) is not exercised — both 4a stacks are on the same dissipative dataset. Bilateral validation requires a conservative-system anchor (case study 02 if PhysicsNeMo includes a conservative target, or a dedicated future case study). 4a also does not exercise the borderline case — a system that *should* be conservative but is *numerically* dissipating due to a model bug, where D0-18's heuristic (dataset-name primary, KE-monotone-decreasing secondary) could mis-classify as dissipative and silently skip the very PH-CON-002 firing that would catch the bug. Diagnostic gap flagged for future case studies.

---

## Rederivability

Rendered at physics-lint `feature/rollout-anchors` sha `<SARIF_EMISSION_SHA>` via:

```bash
python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py \
    --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/
```

Re-run the command at the same sha with the committed SARIFs at that sha → identical output. The renderer's output is deterministic; any divergence reflects a SARIF artifact change, a renderer change, or both — all three cases are caught by the golden test in `methodology/tests/test_render_cross_stack_table.py`.

---

## Integrating-README trigger

This dated writeup at `methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md` is one of a planned series of dated deliverables under `methodology/docs/`. The integrating top-level README — composing 4a's writeup with rung 4b's equivariance writeup and any subsequent rungs — is composed when rung 4b's writeup lands. Until then, this `docs/` directory is the source of truth in dated-deliverable form.
```

- [ ] **Step 4: Replace the comment block in §"Cross-stack conservation table" with the renderer's actual stdout**

Open the writeup at `/Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md`. Replace the entire HTML comment block (between `<!--` and `-->`) and the `(replace this comment block with the actual renderer output from Step 1)` line with the actual renderer output captured in Step 1.

- [ ] **Step 5: Replace `<SARIF_EMISSION_SHA>` placeholder with the actual sha**

In the same file, replace `<SARIF_EMISSION_SHA>` with the 10-char short sha from Step 2 (the current `feature/rollout-anchors` HEAD at the time T6.3 committed the SARIFs).

- [ ] **Step 6: Verify writeup has no remaining placeholders**

```bash
grep -n "<SARIF_EMISSION_SHA>\|<sha>\|TBD\|TODO\|replace this comment block" /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md
```

Expected: no output (all placeholders replaced).

- [ ] **Step 7: Re-run renderer to confirm rederivability footer command works**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/ > /tmp/regen_check.md && diff /tmp/regen_check.md <(grep -A 1000 "^| Rule " /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md | head -10)
```

Expected: trivial / no diff on the table portion. (The diff command compares the freshly-regenerated table against the table embedded in the writeup.)

- [ ] **Step 8: Commit the writeup**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git commit -m "$(cat <<'EOF'
methodology/docs: rung 4a cross-stack conservation writeup

The rung 4a deliverable. Headline frozen at design time (D0-19-
enforced "20 identical fires" claim, with the parenthetical "per-
stack KE endpoints in dedicated properties, not interpolated into
the reason" doing double duty as load-bearing claim and enforcing
mechanism). Cross-stack table rendered from committed SARIFs at
01-lagrangebench/outputs/sarif/ by methodology/tools/render_cross_stack_table.py.

Five-item "what 4a is NOT" deferral list: SEGNN-vs-GNS model
comparison (rung 4b), Security-tab integration demo (rung 4b),
integrating top-level README (post-4b composition), v1.x core
changes, bilateral D0-18 mechanism test.

Rederivability footer with exact render command + sarif_emission_sha
converts copy-paste from "agree right now" into "reproducible from
commit-pinned source." Integrating-README trigger named explicitly
("composed when rung 4b's writeup lands").

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Post-merge cleanup TODO entry

**Files:**
- Modify: `external_validation/_rollout_anchors/methodology/DECISIONS.md`

**Why:** When `feature/rollout-anchors` merges to `master`, the cross-branch qualifier in master's PH-CON-002 honest-limit entry — "currently on the `feature/rollout-anchors` branch pending merge to master" — becomes stale. Track the cleanup trigger now (per the dated-doc-needs-trigger memory and the cross-branch-citation memory).

- [ ] **Step 1: Find D0-18 amendment 1's footer in methodology/DECISIONS.md**

```bash
grep -n "Realized — amendment 1\|amendment 1.\*\*Realized" /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/DECISIONS.md | head -3
```

Expected: line(s) marking the closing of D0-18 amendment 1.

- [ ] **Step 2: Append the post-merge cleanup line at the end of D0-18 amendment 1's section, before the next `---` or `## D0-19` heading**

In the methodology DECISIONS.md, find D0-18 amendment 1's "**Realized — amendment 1.**" paragraph and append (still within the D0-18 section, before the `---` separator that precedes D0-19):

```markdown

**Post-merge cleanup TODO (rung 4a, 2026-05-04).** The rung-4a v1.0 docs amendment on physics-lint `master` (commit `<MASTER_AMENDMENT_SHA>`, see `methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-design.md` §1.3 deferral #4 + the master README's `## v1.0 known limitations` section) embeds a cross-branch qualifier: *"the harness layer (currently on the `feature/rollout-anchors` branch pending merge to master, at `external_validation/_rollout_anchors/_harness/`)"*. **When `feature/rollout-anchors` merges to `master`, this qualifier becomes stale and must be edited out** — the merge resolves the qualifier's premise. The cleanup is a small README edit + a one-line commit on master post-merge; trigger named here so it gets caught at merge PR review rather than living forever as a fossil.
```

Replace `<MASTER_AMENDMENT_SHA>` with the actual sha of the T0.3 master commit (run `git -C /Users/zenith/Desktop/physics-lint rev-parse master` to get it).

- [ ] **Step 3: Verify entry**

```bash
grep -A 2 "Post-merge cleanup TODO" /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/DECISIONS.md
```

Expected: the TODO line + following content.

- [ ] **Step 4: Commit**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git add external_validation/_rollout_anchors/methodology/DECISIONS.md && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" git commit -m "$(cat <<'EOF'
methodology/DECISIONS.md D0-18 amendment 1: post-merge cleanup TODO

Names the trigger for editing out the cross-branch qualifier in
master's PH-CON-002 honest-limit entry when feature/rollout-anchors
merges to master. Cleanup is a small README edit; trigger captured
here so it gets caught at merge PR review rather than living as a
fossil.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Final verification

After T0–T11 complete, run a final regression sweep:

- [ ] **Step 1: Full test suite**

```bash
cd /Users/zenith/Desktop/physics-lint && PATH="/Users/zenith/Desktop/physics-lint/.venv/bin:$PATH" pytest external_validation/_rollout_anchors/ -v
```

Expected: all tests pass (~115 green: 95+ pre-existing _harness tests + ~6 new _harness tests from T2 (2 net new), T3 (3 new), T4 (6 new) + 8 new methodology tests from T9).

- [ ] **Step 2: Branch state check**

```bash
cd /Users/zenith/Desktop/physics-lint && git log --oneline -16
```

Expected: top 16 commits are the 14 commits from T0–T11 (some tasks have multiple commits) plus the design doc commit `efa7b05` plus pre-design tip.

- [ ] **Step 3: Verify SARIFs present**

```bash
ls /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/*.sarif
```

Expected: two committed `.sarif` files matching the pattern `{segnn,gns}_tgv2d_<sha>.sarif`.

- [ ] **Step 4: Verify writeup has rendered table embedded**

```bash
grep -c "^| Rule\|mass_conservation_defect\|energy_drift\|dissipation_sign_violation" /Users/zenith/Desktop/physics-lint/external_validation/_rollout_anchors/methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md
```

Expected: at least 4 (header + 3 rule rows in the table).

- [ ] **Step 5: Verify master README amendment landed**

```bash
git -C /Users/zenith/Desktop/physics-lint show master:README.md | grep -c "PH-CON-002 evaluates"
```

Expected: at least 1.

If all five verification steps pass, rung 4a is complete. Update the user with: branch state, test count, sha of the SARIF emission, sha of the master README amendment.
