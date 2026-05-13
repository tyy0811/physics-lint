# Case Study 02 — PhysicsNeMo MGN Phase 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compose the case-study-02 writeup (CS02 README expansion + new unified three-column cross-stack consistency table extending rung-4a + v2.1.1 methodology amendment dated artifact), land the Phase 3 boundary Codex prose-cross-review with cell-2 absorption, and close D0-24 + §4.3 acceptance with all artifacts pushed.

**Architecture:**
- **Three artifact tracks** (per Q3 activity-placement triplet):
  - **Case-internal**: `02-physicsnemo-mgn/README.md` expands in place (~150-200 lines, mirroring its existing skeleton structure) — Rule × checkpoint results table + scope-qualifier prose + bridge sentence + "what physics-lint did NOT catch" + Reproducibility + methodology cross-refs. The existing PH-CON-001 routing / structural-identity-reapplication section is preserved.
  - **Cross-stack**: new dated artifact `methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md` holding the unified three-column table (`gns-tgv2d` | `segnn-tgv2d` | `modulus_ns_meshgraphnet-vortex_shedding_2d`) extending rung-4a's two-column table. Renderer-deterministic provenance preserved; renderer minimally extended to ingest both `01-lagrangebench/outputs/sarif/` and `02-physicsnemo-mgn/outputs/sarif/`.
  - **Methodology amendment**: new dated artifact `methodology/docs/physics-lint-validation-plan-v2.1.1.md` holding the four §5.5 items grouped into two categories (framework-behavior observations: pattern-C 4th instance + per-section convergence-pattern; specific-finding flags: D0-22 strictly-conservative falsification + prose-vs-artifact-cross-review-modes). Opening paragraph names the v2.1.1 vs v2.1.2 epistemic distinction (pre-registered vs execution-emergent).
- **Frozen-original-plus-pointer discipline** (Q1+Q2+Q3 refinements): rung-4a cross-stack-table artifact + v2.1.md + CS01 README all receive POINTER-ONLY additions; original content stays unchanged. v2.1.md pointers land at §1.3 + §1.4 + §3 changelog forward-flag (NOT §2 — none of v2.1.1's items resolve a §2 deferral; the verified targets are the pattern-C section, the cross-review-discipline section, and the round-prose-2 changelog forward-flag respectively).
- **Codex cross-review fires after all writeup prose is committed** (Q4b variant i — linear sequence). Findings absorbed per trigger classification (Q4a refinement 2): SARIF data-level findings → CPU re-emission (rollout data cached on Modal Volume); fresh-inference-need findings → one pre-registered A10G fire (~$2 budget; falsification = a second A10G need escalates out-of-Phase-3, NOT auto-fire); out-of-Phase-3 findings → defer to amendment 1 or case study 03.
- **v2.1.1 authored AFTER cross-review absorption settles** (per §5.5 trigger: "Phase 3 cross-review of writeup prose completes"). v2.1.1 stays scoped to the four §5.5 items even if cross-review surfaces a 5th pattern — extras go to v2.1.2 or to a fresh forward-flag, not retrofitted into v2.1.1.

**Tech Stack:**
- Existing renderer at `methodology/tools/render_cross_stack_table.py` + tests at `methodology/tests/test_render_cross_stack_table.py` (already pins LB-shape behavior; bonus plumbing for `inference_run_status` already landed via F1 absorption).
- Existing SARIF artifacts at `01-lagrangebench/outputs/sarif/{gns,segnn}_tgv2d_8e49339469.sarif` (LB-side, rung-4a) and `02-physicsnemo-mgn/outputs/sarif/{gt,mgn}.sarif` (CS02, Phase 2 Tasks 5+7+F1 re-emission). **Verified at plan-write time:** all four SARIFs already satisfy the renderer's current `_assert_run_level` contract (10 required fields present in all four, source tag = `'rollout-anchor-harness'`, schema = `'1.0'` in all four, sentinel `lagrangebench_sha = 'n/a_cs02_physicsnemo'` on CS02 SARIFs). The Q2-anticipated `_assert_run_level` parameterization is YAGNI; the renderer just needs to ingest from both dirs and filter out CS02's GT control-arm SARIF (cross-stack column is MGN-side only).
- `pre-commit` requires `.venv` activated per [[feedback_precommit_venv]]; numpy is pinned `<2` for torch ABI compatibility.
- ASCII hyphens in `.py` files per [[feedback_ascii_hyphens_in_python]] (ruff RUF003 enabled); en/em dashes fine in `.md`.

**Inherited inputs** (from Phase 2 D0-24 refinement-2 forward-flag block at `DECISIONS.md` lines 2555-2565):
- **Verbatim scope-qualifier prose** for §3.3 activity 1 (CS02 README results section) — lands as drop-in text in Task 4.
- **Verbatim floor-bounds-resolution distinction** for §3.3 activity 4 (CS02 README "what physics-lint did NOT catch") — lands as drop-in text in Task 6.
- **Substrate-variability finding** (23/100 in-band; 77/100 out-of-band; canonical = trajectory 44 ranked 80/100 by `strouhal_U_max`) — lands in Task 6.
- **v2.1.2 §1.4 methodology forward-flag** (fourth empirical instance of prose-scope-qualifier walls; cumulative pattern strong enough to formalize) — **out of Phase 3 scope** (v2.1.2 is separate); CS02 README references it as a forward-flag only.

**Brainstorming-decided constraints** (synthesized at this plan's write-time from Q1-Q4 → user approval):
- **Q1 (v2.1.1 landing)** = Option A: new `physics-lint-validation-plan-v2.1.1.md` file; frozen-original-plus-pointer discipline applies; v2.1.1 vs v2.1.2 epistemic distinction named in opening; point-outward (don't re-derive) for items 3 + 4; two-category grouping (framework-behavior vs specific-finding).
- **Q2 (cross-stack table)** = Option A interpretation #2: extend rung-4a renderer; new dated artifact holds the three-column table; rung-4a's file gains a top-of-file `Successor:` pointer + content preserved frozen; renderer refactor SCOPE-REDUCED per pre-flight (no `_assert_run_level` parameterization needed — see Tech Stack).
- **Q3 (CS02 README)** = Option C: case-internal writeup expands CS02 README in place; CS01 README gets 3-5 line back-port with framing language; activity-placement triplet (1/4/5/6 → README · 3 → cross-stack table · 7 → v2.1.1) explicit; outward references to CS02 README sections SHA-pin to README's Phase-3-close commit.
- **Q4a (compute envelope)** = Option B: CPU-only default; ≤1 A10G contingency (~$2) bound by rationale per [[feedback_cap_rationale_not_literal]]; falsification = escalate not auto-fire; trigger classification (SARIF data-level → CPU; fresh-inference need → A10G; out-of-scope → defer).
- **Q4b (sequence)** = Variant i: linear (write → cross-review → absorb → v2.1.1 → close); v2.1.1 stays scoped to four §5.5 items even if cross-review surfaces a 5th pattern; light self-check on v2.1.1 (10-15 min) before commit.

**Compute envelope (Q4a = B, pre-registered):**
- **Default**: CPU-only. Renderer extension + render fires CPU-bound (<1 min each).
- **Pre-registered contingency**: ≤1 A10G fire (~$2 budget) IFF cross-review surfaces a finding that requires fresh MGN inference (checkpoint provenance bug, inference protocol bug, metadata only obtainable from a fresh run).
- **Falsification**: a second A10G need surfaces as an out-of-Phase-3 escalation (amendment 1 / case study 03 candidate). The cap binds by rationale (absorption budget bound to one fire), NOT literal text.
- **Trigger classification** (pre-registered):
  - **SARIF data-level** (sentinel misleading / field mis-emitted / schema violation): CPU re-emission only. Phase 2's F1 absorption precedent — rollout data cached on Modal Volume, re-emission re-reads cached data + re-emits SARIFs.
  - **Fresh-inference need** (checkpoint wrong, inference protocol wrong, metadata only obtainable from fresh run): A10G fire fires (consumes the budget).
  - **Out-of-Phase-3 scope** (e.g., a finding about a substrate not yet probed; a methodology surface for amendment 1 or case study 03): defer; do NOT fire.

**Scope guardrails (Q4b refinement 3):**
- v2.1.1 stays scoped to the four §5.5 items even if the cross-review surfaces a 5th pattern-C instance or methodology issue — extras go to **v2.1.2 or a fresh forward-flag**, NOT retrofitted into v2.1.1.
- Outward references to CS02 README sections SHA-pin to the README's commit at Phase 3 close (provenance equivalence with dated artifacts).
- The substrate-variability finding lands in CS02 README's "what physics-lint did NOT catch" (not the cross-stack table).
- **No active tightening of floor-bounds-resolution via finer-discretization re-fire** — that would be empirical work, properly scoped to v1.1 backlog or a future case study.

---

## Phase 3A — Renderer extension + unified cross-stack table

### Task 1: Pre-flight verify the renderer's current assertion behavior on CS02 SARIFs

**Goal:** Confirm in-test (not just by manual inspection) that the existing `_assert_run_level` contract passes on CS02's `gt.sarif` + `mgn.sarif` without parameterization. This is the YAGNI-justification step: if the test passes as-is, Task 2 reduces to "ingest from both dirs + filter GT"; if it fails, Task 2 expands to per-source parameterization (in which case re-read Q2 and decide which optional-field set to enforce per source).

**Files:**
- Create: `external_validation/_rollout_anchors/methodology/tests/test_render_cs02_ingest.py`

- [ ] **Step 1: Write a failing test that asserts CS02's mgn.sarif passes `_assert_run_level`.**

```python
"""Pre-flight verification: CS02 SARIFs satisfy the existing D0-19 v1.0
required-field contract without per-source parameterization.

If this test ever starts failing (e.g., a future SARIF emission drops one
of the 10 required fields, or a schema bump lands without re-emission),
the renderer extension's "no parameterization needed" premise is broken
and Phase 3 Task 2 should re-open the per-source parameterization decision.
"""

from pathlib import Path

import pytest

from external_validation._rollout_anchors.methodology.tools.render_cross_stack_table import (
    _assert_run_level,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
CS02_SARIF_DIR = REPO_ROOT / "external_validation" / "_rollout_anchors" / "02-physicsnemo-mgn" / "outputs" / "sarif"


@pytest.mark.parametrize("sarif_name", ["gt.sarif", "mgn.sarif"])
def test_cs02_sarifs_pass_existing_assert_run_level(sarif_name: str) -> None:
    """CS02 SARIFs (post-F1-absorption) satisfy the renderer's existing
    10-required-field contract without per-source parameterization. The
    Q2 brainstorming initially scoped a per-source parameterization
    refactor; pre-flight inspection found this is YAGNI because CS02
    SARIFs already carry the LB-side required fields via sentinel
    values (lagrangebench_sha = 'n/a_cs02_physicsnemo').
    """
    import json

    path = CS02_SARIF_DIR / sarif_name
    sarif = json.loads(path.read_text())
    run_props = _assert_run_level(sarif, path)

    assert run_props["source"] == "rollout-anchor-harness"
    assert run_props["harness_sarif_schema_version"] == "1.0"
    assert run_props["lagrangebench_sha"] == "n/a_cs02_physicsnemo"
    assert run_props["dataset_name"] == "vortex_shedding_2d"
```

- [ ] **Step 2: Run the test.**

```bash
source .venv/bin/activate
pytest external_validation/_rollout_anchors/methodology/tests/test_render_cs02_ingest.py::test_cs02_sarifs_pass_existing_assert_run_level -v
```

Expected: **2 passed** (both `gt.sarif` and `mgn.sarif` satisfy the existing contract). If 1+ fail: **STOP** — re-open Q2's parameterization decision; the YAGNI premise is broken; the plan needs amendment before Task 2 fires.

- [ ] **Step 3: Commit.**

```bash
git add external_validation/_rollout_anchors/methodology/tests/test_render_cs02_ingest.py
git commit -m "02-physicsnemo-mgn: Phase 3 Task 1 — pre-flight test verifies CS02 SARIFs satisfy renderer's existing 10-required-field contract (YAGNI confirmation for Task 2 scope reduction)"
```

---

### Task 2: Extend renderer's CLI to ingest from multiple SARIF dirs + filter the CS02 control-arm SARIF

**Goal:** Add support for the renderer to ingest from BOTH `01-lagrangebench/outputs/sarif/` and `02-physicsnemo-mgn/outputs/sarif/` in a single invocation. CS02 emits two SARIFs per Phase 2 (`gt.sarif` = control arm, `mgn.sarif` = the model under test); only `mgn.sarif` becomes a cross-stack column. GT is a per-case-study control, not a cross-stack column.

**Files:**
- Modify: `external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py` (CLI arg handling + arm filter)
- Modify: `external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py` (add tests for multi-dir ingestion + GT-arm filter)

- [ ] **Step 1: Inspect the existing CLI surface to confirm extension shape.**

```bash
grep -n "argparse\|add_argument\|parser\|--sarif-dir\|--include-glob" external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py
```

Expected: an existing `--sarif-dir` argument (used by rung-4a invocation in the docstring). The extension makes it `nargs='+'` (accept multiple paths) or adds a parallel `--sarif-dirs` arg; choose whichever fits the existing arg layout per `argparse` conventions in this file.

- [ ] **Step 2: Write the failing test for multi-dir ingestion.**

Add to `test_render_cross_stack_table.py`:

```python
def test_renderer_ingests_from_multiple_dirs() -> None:
    """Multi-dir ingestion: the renderer reads SARIFs from BOTH
    01-lagrangebench/outputs/sarif/ AND 02-physicsnemo-mgn/outputs/sarif/
    in a single invocation, producing a unified three-column table
    (gns-tgv2d, segnn-tgv2d, modulus_ns_meshgraphnet-vortex_shedding_2d).
    GT-arm SARIF (02-physicsnemo-mgn/outputs/sarif/gt.sarif) is filtered
    out — it is a per-case-study control arm, not a cross-stack column.
    """
    from pathlib import Path

    from external_validation._rollout_anchors.methodology.tools.render_cross_stack_table import (
        render_cross_stack_table,
    )

    repo_root = Path(__file__).resolve().parents[4]
    lb_sarifs = sorted(
        (repo_root / "external_validation" / "_rollout_anchors" / "01-lagrangebench" / "outputs" / "sarif").glob(
            "*tgv2d_8e49339469.sarif"
        )
    )
    cs02_mgn_sarif = repo_root / "external_validation" / "_rollout_anchors" / "02-physicsnemo-mgn" / "outputs" / "sarif" / "mgn.sarif"

    table = render_cross_stack_table([*lb_sarifs, cs02_mgn_sarif])

    assert "gns-tgv2d" in table
    assert "segnn-tgv2d" in table
    assert "modulus_ns_meshgraphnet-vortex_shedding_2d" in table
    # GT-arm should NOT appear as a column — control arm, not cross-stack
    assert "deepmind-cylinder-flow-gt" not in table


def test_renderer_filters_control_arm_sarif_when_present() -> None:
    """If the caller passes CS02's gt.sarif AND mgn.sarif (e.g., via
    a directory glob that picks up both), the renderer must filter the
    control-arm SARIF and use only the model-under-test SARIF for the
    cross-stack column. The filter binds on the run-level `arm` field:
    `arm == 'control'` → filtered out; `arm == 'model'` (or absent for
    LB-side legacy SARIFs) → included.
    """
    from pathlib import Path

    from external_validation._rollout_anchors.methodology.tools.render_cross_stack_table import (
        render_cross_stack_table,
    )

    repo_root = Path(__file__).resolve().parents[4]
    cs02_dir = repo_root / "external_validation" / "_rollout_anchors" / "02-physicsnemo-mgn" / "outputs" / "sarif"
    lb_sarifs = sorted(
        (repo_root / "external_validation" / "_rollout_anchors" / "01-lagrangebench" / "outputs" / "sarif").glob(
            "*tgv2d_8e49339469.sarif"
        )
    )

    # Pass BOTH gt.sarif and mgn.sarif explicitly; renderer must filter gt
    table = render_cross_stack_table(
        [*lb_sarifs, cs02_dir / "gt.sarif", cs02_dir / "mgn.sarif"]
    )

    assert "modulus_ns_meshgraphnet-vortex_shedding_2d" in table
    assert "deepmind-cylinder-flow-gt" not in table
```

- [ ] **Step 3: Run the tests to verify failure.**

```bash
source .venv/bin/activate
pytest external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py::test_renderer_ingests_from_multiple_dirs external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py::test_renderer_filters_control_arm_sarif_when_present -v
```

Expected: BOTH FAIL — `render_cross_stack_table` does not yet filter on `arm`.

- [ ] **Step 4: Add the GT-arm filter to `render_cross_stack_table`.**

Locate the `stacks: list[tuple[Path, dict[str, Any], list[dict[str, Any]]]] = []` block (around line 143 of `render_cross_stack_table.py`). After reading `sarif` + computing `run_props`, add:

```python
        # Filter out per-case-study control-arm SARIFs (CS02-onwards
        # convention). Control-arm SARIFs (`arm == 'control'`) carry
        # ground-truth-vs-rule-floor evidence for the case study itself;
        # they are NOT cross-stack columns. Model-under-test SARIFs
        # (`arm == 'model'`) and LB-side legacy SARIFs (no `arm` field;
        # pre-CS02 convention) flow through to the cross-stack table.
        arm = run_props.get("arm")
        if arm == "control":
            continue
```

- [ ] **Step 5: Run the tests again to verify pass.**

```bash
source .venv/bin/activate
pytest external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py -v
```

Expected: **ALL pass** (existing tests + the 2 new ones). If any pre-existing test now fails, the filter logic placement broke a contract — investigate before committing.

- [ ] **Step 6: Extend the CLI to accept multiple `--sarif-dir` paths.**

Find the existing `--sarif-dir` parser block. Update its `action='append'` / `nargs='+'` so multiple `--sarif-dir` flags are accepted. The resolved dir list flows into `render_cross_stack_table(sorted(itertools.chain.from_iterable(d.glob('*.sarif') for d in dirs)))`.

Add to the file's docstring (top of file, the `INVOKE FROM REPO ROOT:` block):

```python
"""
...

INVOKE FOR THE UNIFIED CS01 + CS02 CROSS-STACK TABLE (Phase 3):

    python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py \\
        --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/ \\
        --include-glob '*_tgv2d_8e49339469.sarif' \\
        --sarif-dir external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/
"""
```

(Keep the existing rung-4a-style invocation in the docstring too — it still works for backwards-compat.)

- [ ] **Step 7: Commit.**

```bash
git add external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py \
        external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py
git commit -m "_rollout_anchors: Phase 3 Task 2 — renderer ingests from multiple SARIF dirs + filters control-arm SARIFs (CS02-onwards convention)"
```

---

### Task 3: Render the unified 2026-05-13 cross-stack table; add `Successor:` pointer to rung-4a artifact

**Goal:** Produce the new dated artifact at `methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md` holding the unified three-column table. Add a top-of-file `Successor:` pointer to `2026-05-04-rung-4a-cross-stack-conservation-table.md` per the frozen-original-plus-pointer discipline (Q2 interpretation #2). Methodology paragraph names the extension discipline (renderer-deterministic provenance preserved across stacks; any divergence reflects a SARIF or renderer change, caught by golden test).

**Files:**
- Create: `external_validation/_rollout_anchors/methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md`
- Modify: `external_validation/_rollout_anchors/methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md` (top-of-file `Successor:` line only; body preserved frozen)

- [ ] **Step 1: Render the unified table from the SARIFs.**

```bash
source .venv/bin/activate
python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py \
    --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/ \
    --include-glob '*_tgv2d_8e49339469.sarif' \
    --sarif-dir external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/ \
    > /tmp/cs02_cross_stack_table.md
cat /tmp/cs02_cross_stack_table.md
```

Expected: a three-column markdown table similar to rung-4a's two-column table, plus a Provenance block with three-sha per stack. The MGN column should show:
- `mass_conservation_defect` → `5.881e-02 (x1)` (single fire; D0-24 v2)
- `energy_drift` → `SKIP (x1, D0-22 + D0-23)` (substrate-class dispatch fires on open-driven-dissipative; D0-24 v3)
- `dissipation_sign_violation` → `SKIP (x1, D0-22 + D0-23)` (same dispatch; D0-24 v4)

(Exact stack-label strings come from the SARIF's `model_name` + `dataset_name`; verify against actual output before committing the writeup.)

- [ ] **Step 2: Write the new dated artifact using rung-4a's layout as a template.**

```bash
cp external_validation/_rollout_anchors/methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md /tmp/template.md
```

Then create `external_validation/_rollout_anchors/methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md` with this structure (mirror rung-4a's headings; replace content):

```markdown
# Case Study 02 — Unified cross-stack conservation table (writeup; extends rung-4a)

**Date:** 2026-05-13
**Predecessor:** [`./2026-05-04-rung-4a-cross-stack-conservation-table.md`](./2026-05-04-rung-4a-cross-stack-conservation-table.md) (LB-only two-column; preserved frozen at its commit date).
**Design doc:** [`./2026-05-11-case-study-02-physicsnemo-mgn-design.md`](./2026-05-11-case-study-02-physicsnemo-mgn-design.md) §3.3 activity 3.
**SARIF artifacts:** [`../../01-lagrangebench/outputs/sarif/`](../../01-lagrangebench/outputs/sarif/) (LB-side, rung-4a state) + [`../../02-physicsnemo-mgn/outputs/sarif/`](../../02-physicsnemo-mgn/outputs/sarif/) (CS02 Phase 2).
**Methodology pre-registrations:** [D0-19](../DECISIONS.md#d0-19), [D0-20](../DECISIONS.md#d0-20), [D0-22](../DECISIONS.md#d0-22), [D0-23](../DECISIONS.md#d0-23), [D0-24](../DECISIONS.md#d0-24).

---

## Headline

physics-lint's harness ran the same conservation rule schema, unmodified, across THREE upstream rollouts of TWO substrate classes: GNS-TGV2D + SEGNN-TGV2D (rung-4a, dissipative-isotropic) and PhysicsNeMo-MGN on cylinder_flow vortex shedding (CS02 Phase 2, open-driven-dissipative). Every result row's structural contract holds across all three columns under D0-19 + D0-20 enforcement; D0-22/D0-23 substrate-class dispatch fires SKIP-with-reason identically on `energy_drift` + `dissipation_sign_violation` across the LB-side (D0-18 dissipative path) and the CS02-side (D0-23 v9 open-driven-dissipative path); `mass_conservation_defect` raw-value renders LB-side at 0.0 (TGV2D's exact mass conservation) and CS02 MGN at 5.881e-02 (the harness-FE-on-P1 floor on the canonical cylinder_flow trajectory; see `02-physicsnemo-mgn/README.md` "What physics-lint did NOT catch" §1 for the floor-bounds-resolution distinction). **Renderer-deterministic provenance is preserved across the extension**: any divergence between the rung-4a-era LB columns in this extended table vs the original rung-4a artifact reflects a SARIF change or a renderer change, caught by the golden test at `methodology/tests/test_render_cross_stack_table.py`.

---

## Unified cross-stack conservation table

<paste the renderer output here verbatim>

---

## What extending the table preserves

1. **Frozen-original discipline.** The rung-4a artifact's two-column table is preserved at its commit date as the LB-only state. This artifact extends with a third column WITHOUT replacing the rung-4a artifact (which retains regression-check value: if the extended renderer accidentally changes the LB columns, the original rung-4a artifact is the comparison baseline).

2. **Renderer-deterministic provenance.** The extended renderer (Phase 3 Task 2) ingests from both `01-lagrangebench/outputs/sarif/` and `02-physicsnemo-mgn/outputs/sarif/`; the `arm == 'control'` filter excludes CS02's GT control-arm SARIF from cross-stack columns. The golden test pins the unified table's shape. Any drift between SARIFs and rendered table is caught.

3. **Cross-substrate schema reuse.** D0-19's run-level + result-level schema (v1.0) was sufficient for the CS02-onwards convention without parameterization. CS02 SARIFs carry the 10 LB-required fields with sentinel values (`lagrangebench_sha = 'n/a_cs02_physicsnemo'`) where LB-specific provenance does not apply; CS02-specific fields (`arm`, `case_study`, `physicsnemo_sha`, `rollout_contract`, `trajectory_index`, `inference_run_status`, etc.) extend the run-level metadata additively per D0-19's optional-field convention. No schema bump was required.

---

## What this table is NOT

1. **Not a cross-stack model comparison on the same substrate.** The LB columns are on TGV2D (dissipative-isotropic); the CS02 column is on cylinder_flow (open-driven-dissipative). The MGN column's `5.881e-02` mass-conservation-defect value is NOT directly comparable to the LB columns' `0.0` values — they bind on different physics regimes. The table's value is the *schema-uniformity* claim, not a model-quality comparison.

2. **Not a CS02-internal writeup.** Case-study-internal results (PH-CON-001 routing per D0-03; scope-qualifier for the single-trajectory regime; floor-bounds-resolution distinction; substrate-variability finding) live at `02-physicsnemo-mgn/README.md`. This dated artifact's scope is cross-stack consistency.

3. **Not the integrating top-level README.** That composes when amendment 1 lands (Ahmed Body + the cross-stack table's fourth column), per the rung-4a precedent.

---

## Rederivability

Rendered at physics-lint `feature/case-study-02-physicsnemo-mgn` sha `<sha at Task 3 commit>` via:

```bash
python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py \
    --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/ \
    --include-glob '*_tgv2d_8e49339469.sarif' \
    --sarif-dir external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/
```

Re-run at the same sha → identical output. Determinism golden-tested at `methodology/tests/test_render_cross_stack_table.py`.

---

## Predecessor → successor pointer

Predecessor `2026-05-04-rung-4a-cross-stack-conservation-table.md` is preserved frozen at its commit date (the LB-only artifact). This dated artifact is the canonical cross-stack table going forward; amendment 1 (Ahmed Body) extends with a fourth column when that case-study row lands.
```

- [ ] **Step 3: Add the `Successor:` pointer to the rung-4a artifact (frozen-original-plus-pointer discipline).**

Modify `external_validation/_rollout_anchors/methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md` — add ONE line near the top (after the existing `**Successor:**` line that points to rung-4b):

```markdown
**Successor (extended table):** [`./2026-05-13-case-study-02-cross-stack-conservation-table.md`](./2026-05-13-case-study-02-cross-stack-conservation-table.md) — unified three-column table (LB-side rung-4a columns preserved + CS02-side MGN-on-vortex-shedding column added). This file remains canonical for the LB-only state at its commit date; the 2026-05-13 file is canonical for the cross-substrate cross-stack table going forward.
```

**Do not modify any other content in this file.** The rung-4a artifact's narrative ("dissipative system, both stacks SKIP via D0-18") is preserved verbatim.

- [ ] **Step 4: Add a golden-test fixture for the unified table shape.**

Pick the existing `test_renderer_emits_markdown_table_with_three_rules` or `test_renderer_golden_output_matches_expected_table` precedent in `test_render_cross_stack_table.py`; add a parallel `test_renderer_unified_cs02_table_golden_output_matches_expected` that pins the new three-column shape (commit the expected output as a string constant in the test file, OR as a `tests/fixtures/cs02_unified_cross_stack_table_golden.md` fixture per the rung-4a precedent).

```bash
source .venv/bin/activate
pytest external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py -v
```

Expected: ALL pass including the new golden test.

- [ ] **Step 5: Commit.**

```bash
git add external_validation/_rollout_anchors/methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md \
        external_validation/_rollout_anchors/methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md \
        external_validation/_rollout_anchors/methodology/tests/test_render_cross_stack_table.py
# Plus the golden-fixture file if you wrote one:
# git add external_validation/_rollout_anchors/methodology/tests/fixtures/cs02_unified_cross_stack_table_golden.md
git commit -m "methodology: Phase 3 Task 3 — unified CS01+CS02 cross-stack conservation table (extends rung-4a; frozen-original-plus-pointer; golden-tested)"
```

---

## Phase 3B — CS02 README expansion

### Task 4: Expand "Rule × checkpoint results" with PH-CON-001/002/003 numbers + scope-qualifier prose

**Goal:** Replace CS02 README's existing `[Populated by Day 2 rollouts. SARIF outputs in outputs/lint.sarif.]` stub with the actual D0-24 PH-CON-001/002/003 results + the verbatim scope-qualifier prose from DECISIONS.md lines 2557-2559 (refinement-2 forward-flag item 1). Per design §3.3 activity 1.

**Files:**
- Modify: `external_validation/_rollout_anchors/02-physicsnemo-mgn/README.md` (Rule × checkpoint results section)

- [ ] **Step 1: Replace the stub with the populated results.**

In `external_validation/_rollout_anchors/02-physicsnemo-mgn/README.md`, replace the existing block:

```markdown
## Rule × checkpoint results

*[Populated by Day 2 rollouts. SARIF outputs in `outputs/lint.sarif`.]*
```

with this expanded section (verbatim, including the drop-in scope-qualifier paragraph):

```markdown
## Rule × checkpoint results

P0 results for `modulus_ns_meshgraphnet` on cylinder_flow vortex shedding (Phase 2 fires on canonical trajectory M = 44 — see scope qualifier below). SARIFs at [`outputs/sarif/`](outputs/sarif/) (`gt.sarif` = ground-truth control arm; `mgn.sarif` = MGN model under test). D0-24 verdict bands pre-registered before fires; all 7 verdicts pinned PASS at [DECISIONS.md D0-24](../methodology/DECISIONS.md#d0-24--2026-05-13--case-study-02-phase-2-audit-verdicts).

| Rule | GT (control arm) | MGN | D0-24 verdict |
|---|---|---|---|
| `PH-CON-001` mass / divergence-free | 5.857e-02 (5.857 %) | 5.881e-02 (5.881 %) | v1 PASS (GT ≤ 6 %); v2 PASS (gap = 0.41 % ≤ 20 %) |
| `PH-CON-002` energy drift | SKIP (open-driven-dissipative, D0-22 + D0-23 v9) | SKIP (same dispatch) | v3 PASS (substrate-class dispatch fires on both arms) |
| `PH-CON-003` dissipation sign violation | SKIP (same dispatch) | SKIP (same dispatch) | v4 PASS (same dispatch) |

See [`methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md`](../methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md) for the unified three-column cross-stack consistency table including the rung-4a LB-side columns (GNS-TGV2D + SEGNN-TGV2D).

### Scope qualifier — single trajectory, in-band subset

Phase 2 results report PH-CON-001 defect on a single cylinder_flow test trajectory (trajectory `M = 44`, Strouhal `St_U_max = 0.192` ∈ design band `[0.16, 0.21]`, cylinder diameter `D = 0.135`, inflow `U_max = 1.502`; Reynolds number not directly captured at Phase 2 — derivable as `Re = U · D / ν` from the cylinder_flow benchmark's `ν` if needed in Phase 3), selected via the Phase 2 pre-fire Strouhal audit (Task 1, 23 / 100 in-band) to be representative of the in-band subset under the pre-registered centerline-convention selection rule (median-`strouhal_U_max` among the 23 in-band trajectories). Trajectory `44` ranks 80th of 100 by `strouhal_U_max` on the full test set (the in-band subset occupies ranks 69-91; out-of-band trajectories sit at lower `strouhal_U_max` values, where the literature-anchored design band would not hold anyway). Coverage-not-statistics framing: physics-lint's value here is rule-firing on a real-world checkpoint, not a distribution over initial conditions. CI-gate threshold derivation from defect-magnitude distributions would require `N > 1` and is deferred (Phase 2 does NOT claim CI-gate calibration or full-distribution representativeness).
```

- [ ] **Step 2: Verify the file parses cleanly.**

```bash
source .venv/bin/activate
python -c "import markdown; markdown.markdown(open('external_validation/_rollout_anchors/02-physicsnemo-mgn/README.md').read())" 2>&1 | head -5
```

Expected: no errors (markdown parses cleanly).

- [ ] **Step 3: Commit.**

```bash
git add external_validation/_rollout_anchors/02-physicsnemo-mgn/README.md
git commit -m "02-physicsnemo-mgn: Phase 3 Task 4 — README Rule × checkpoint results populated with D0-24 PH-CON-001/002/003 numbers + verbatim scope-qualifier prose"
```

---

### Task 5: Add the bridge-sentence paragraph

**Goal:** Add the bridge-sentence paragraph per design §3.3 activity 2. The bridge sentence positions the CS02 results in the context of the rung-4 series's cross-stack-schema-uniformity claim — the same harness ran unmodified across LB-TGV2D and PhysicsNeMo-cylinder-flow, which is the load-bearing claim that the rung-4 methodology lessons generalize beyond rung-4c.

**Files:**
- Modify: `external_validation/_rollout_anchors/02-physicsnemo-mgn/README.md` (insert bridge paragraph between the scope-qualifier section and the existing "PH-CON-001 routing" section)

- [ ] **Step 1: Add the bridge paragraph.**

Insert between the scope-qualifier section (Task 4) and the existing `## PH-CON-001 routing — harness, not public rule` heading:

```markdown
### Bridge to the cross-stack story

The numbers above land in the unified cross-stack table at [`methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md`](../methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md) as the third column (alongside `gns-tgv2d` and `segnn-tgv2d`). The schema-uniformity claim — *the same conservation rule schema runs unmodified across three upstream rollouts of two substrate classes* — is what Case Study 02 supplies to the rung-4 series's methodology trail per [`physics-lint-validation-plan-v2.1.md`](../methodology/docs/physics-lint-validation-plan-v2.1.md) §1.5 (Case Study 02 as a falsification surface). Whether the A + B + C triad generalizes is tested by the Phase 1 + Phase 2 + Phase 3 cross-review findings and triage; see [DECISIONS.md D0-23 + D0-24](../methodology/DECISIONS.md) for the per-pattern verdicts.
```

- [ ] **Step 2: Commit.**

```bash
git add external_validation/_rollout_anchors/02-physicsnemo-mgn/README.md
git commit -m "02-physicsnemo-mgn: Phase 3 Task 5 — README bridge-sentence paragraph linking CS02 results into rung-4 cross-stack story"
```

---

### Task 6: Add "What physics-lint did NOT catch" section

**Goal:** Add a §3.3-activity-4 section to CS02 README covering: (a) floor-bounds-resolution distinction (verbatim from DECISIONS.md refinement-2 forward-flag item 2); (b) substrate-variability finding (item 3); (c) class-level entry on V1-rules-with-input-domain-restrictions (PH-CON-001 routing per D0-03, structural-identity reapplication — this entry was already partially in the existing README; consolidate); (d) PH-NUM-002 multi-resolution → v1.1 backlog; (e) PH-SYM-* not on mesh side; (f) Ahmed Body + PH-RES-001 deferred to amendments; (g) v2.1.2 §1.4 prose-scope-qualifier forward-flag.

**Files:**
- Modify: `external_validation/_rollout_anchors/02-physicsnemo-mgn/README.md` (add new section after PH-CON-001 routing)

- [ ] **Step 1: Add the section.**

Append after the existing `## PH-CON-001 routing — harness, not public rule` section:

```markdown
## What physics-lint did NOT catch

This section names the limits of what Phase 2's PH-CON-001/002/003 fires demonstrate, so the writeup's narrow claims are not over-read into broader ones they do not support.

### 1. Floor-bounds-resolution — PH-CON-001 at this discretization

Phase 2 fires PH-CON-001 with `raw_value = 5.881e-02` (5.881 %) for MGN and `raw_value = 5.857e-02` (5.857 %) for the GT control arm. The harness-FE-on-P1 floor is `≈ 5.8 %` on this trajectory (`≈ 5 %` on Phase 1 substrate-smoke's auto-selected trajectory) and bounds PH-CON-001's discriminating resolution at this discretization (test-trajectory mesh; `N_nodes = 1787`, `N_cells = 3340`; P1 basis). The MGN/GT gap of 0.41 % sits well inside the harness-floor envelope.

What this demonstrates: **"MGN reproduces GT at the floor."**

What this does NOT demonstrate: **"MGN is physically incompressible to 5 %."**

The two claims are distinct. PH-CON-001 at this discretization bounds MGN's deviation from GT-equivalence rather than from physical incompressibility. A tighter discretization would be needed to distinguish whether the `5 %` reflects MGN model error or floor error. Phase 3 holds this distinction explicitly to avoid the over-read; tighter-discretization re-fire is out of Phase 3 scope (would be empirical work, properly scoped to v1.1 backlog or a future case study).

### 2. Substrate-variability — single-trajectory + in-band subset

Phase 2's pre-fire Strouhal audit (Task 1) found 23 / 100 test trajectories land in the literature-anchored design band `[0.16, 0.21]` on the either-or convention check; 77 / 100 land out-of-band on the same check (mostly with `St_U_mean > 0.21` while `St_U_max ∈ [0.16, 0.21]` — the U-convention ambiguity). The canonical trajectory selection (median-`strouhal_U_max` among the in-band 23) is deterministic and reproducible. This is a substrate-variability characterization, NOT a methodology failing — physics-lint's value is rule-firing on a real-world checkpoint, not a distribution over initial conditions. CI-gate threshold derivation from defect-magnitude distributions would require `N > 1` and is deferred.

### 3. PH-CON-001 routing — harness, not public rule (class-level: V1-rules-with-input-domain-restrictions)

PH-CON-001 as shipped in physics-lint v0.0.0.dev0 returns SKIPPED on `pde != "heat"` (per [DECISIONS.md D0-03](../methodology/DECISIONS.md#d0-03)). The mesh case study routes PH-CON-001 through the mesh harness as **structural-identity reapplication** — the structural mass-conservation identity (∫ρ over the domain; ∇·v on incompressible NS) is reapplied by the harness, validated against the analytical mass-conservation fixture at [`_harness/tests/fixtures/mass_conservation_fixture.py`](../_harness/tests/fixtures/mass_conservation_fixture.py). This is NOT "rule ran without modification." The class-level pattern (V1 rules with input-domain restrictions; the harness reapplies the structural identity rather than the v1.0 rule code itself) is the load-bearing methodology claim; see `physics-lint-validation-plan-v3.md` §6 risk-register class-level entry on V1 rules with input-domain restrictions for the cross-rung pattern catalogue.

### 4. PH-NUM-002 multi-resolution → v1.1 backlog

Cylinder_flow comes at a single mesh resolution per trajectory; PH-NUM-002 (which depends on a resolution sweep) is not exercised in Phase 2. Deferred to v1.1 backlog per spec §1.2.

### 5. PH-SYM-* not on mesh side

PH-SYM-001/002/003/004 (rotational / reflection / translation / Galilean equivariance) are scoped particle-side only per spec §1.2; the mesh side does not exercise them. Not deferred, out of scope.

### 6. Ahmed Body + PH-RES-001 deferred to amendments

Ahmed Body + PH-BC-001 (no-slip on car-like geometry) → deferred to amendment 1 per [design §5.1](../methodology/docs/2026-05-11-case-study-02-physicsnemo-mgn-design.md#51-ahmed-body--ph-bc-001--steady-rans--amendment-1) (different physics regime; prime pattern-B candidate; BLOCKING-2 raw-data availability question to resolve). PH-RES-001 (BDO momentum residual) → deferred to amendment 2 OR case study 03 per design §5.2.

### 7. v2.1.2 §1.4 forward-flag — prose-scope-qualifier walls

The floor-bounds-resolution distinction above is the FOURTH empirical instance of physics-lint's prose-scope-qualifier discipline (after round-code-1's three walls — level rendering, file-path location, prose framing). The cumulative pattern is strong enough to formalize as a v2.1.2 §1.4 methodology entry. Tracked separately at `physics-lint-validation-plan-v2.1.2.md` (forthcoming); does NOT block Phase 3 close.
```

- [ ] **Step 2: Commit.**

```bash
git add external_validation/_rollout_anchors/02-physicsnemo-mgn/README.md
git commit -m "02-physicsnemo-mgn: Phase 3 Task 6 — README 'What physics-lint did NOT catch' section (floor-bounds-resolution + substrate-variability + class-level §1.4 entry + deferral catalogue + v2.1.2 forward-flag)"
```

---

### Task 7: Add Reproducibility section + methodology cross-references

**Goal:** Replace CS02 README's existing thin Reproducibility section with a complete one per design §3.3 activities 5 + 6: Modal entrypoints, NGC checkpoint hash + ngc_version, git_sha at Phase 3 close, conda lockfile location, methodology cross-references.

**Files:**
- Modify: `external_validation/_rollout_anchors/02-physicsnemo-mgn/README.md` (replace + expand Reproducibility section)

- [ ] **Step 1: Find the NGC checkpoint hash + ngc_version from the MGN SARIF.**

```bash
python -c "import json; p = json.load(open('external_validation/_rollout_anchors/02-physicsnemo-mgn/outputs/sarif/mgn.sarif'))['runs'][0]['properties']; print('ckpt_sha256:', p['ckpt_sha256']); print('ngc_version:', p['ngc_version']); print('physicsnemo_sha:', p['physicsnemo_sha']); print('checkpoint_id:', p['checkpoint_id'])"
```

Record the four values for the README block.

- [ ] **Step 2: Find the conda lockfile location.**

```bash
find external_validation/_rollout_anchors/02-physicsnemo-mgn -name "*.lock" -o -name "*conda*" -o -name "requirements*.txt" -o -name "environment*.yml" 2>/dev/null | head -5
```

Record the path (likely a `modal_app.py` image-spec pin rather than a literal conda lockfile — name the Modal image-spec sha if so).

- [ ] **Step 3: Replace the existing Reproducibility section.**

Find the existing block in CS02 README:

```markdown
## Reproducibility

Modal entrypoint: `modal_app.py`. Inference script: `run_inference.py`.
Lint driver: `lint_rollouts.py` — invokes
`_rollout_anchors/_harness/mesh_rollout_adapter.py` (per-timestep
materialization) plus the public `physics-lint check` CLI per timestep.

Inference must pass `test_inference_matches_ngc_sample` (max-abs-error
≤ 10⁻³ on velocity components vs NGC's shipped sample) before rollouts
proceed; this is the gate-determining test for Gate D (spec §6).
```

Replace with (filling in the actual values from Steps 1 + 2):

```markdown
## Reproducibility

### Modal entrypoints (Phase 1 + Phase 2 fires)

| Entrypoint | Purpose | Compute |
|---|---|---|
| `02-physicsnemo-mgn/modal_app.py::audit_ngc_sample_reproduction` | Phase 1 Gate D — NGC sample reproduction RMSE vs Pfaff et al. CylinderFlow RMSE-1 baseline (verdict 4 + 5) | A10G, ~10 min |
| `02-physicsnemo-mgn/modal_app.py::audit_gate_a_pyg_to_meshfield` | Phase 1 Gate A — PyG-to-MeshField materialization smoke (verdict 2) | CPU, <1 min |
| `02-physicsnemo-mgn/modal_app.py::smoke_substrate_class_vortex_shedding` | Phase 1 substrate-class smoke (verdicts 6 + 7 — ∫∇·v dV / KE budget / Strouhal) | A10G, ~5 min |
| `02-physicsnemo-mgn/modal_app.py::audit_strouhal_test_trajectories` | Phase 2 Task 1 — Strouhal pre-check across cylinder_flow test trajectories (refinement 1) | CPU, ~10 min |
| `02-physicsnemo-mgn/modal_app.py::cs02_lint_gt_trajectory` | Phase 2 Task 5 — GT-trajectory CPU lint → `gt.sarif` (control arm) | CPU, ~3 min |
| `02-physicsnemo-mgn/modal_app.py::cs02_run_mgn_inference` | Phase 2 Task 6 — MGN inference on canonical trajectory 44 (599 rollout steps) | A10G, ~10 min |
| `02-physicsnemo-mgn/modal_app.py::cs02_lint_mgn_rollout` | Phase 2 Task 7 — MGN-rollout CPU lint → `mgn.sarif` | CPU, ~3 min |

### NGC checkpoint provenance

| Field | Value |
|---|---|
| `checkpoint_id` | `modulus_ns_meshgraphnet` |
| `ngc_version` | `<paste from Step 1>` |
| `ckpt_sha256` | `<paste from Step 1>` |
| `physicsnemo_sha` | `<paste from Step 1>` |

Adapter: `_legacy_checkpoint_name_remap.py` (Phase 1 Gate D fix) — encoder/decoder `.mlp.` → `.model.` rename + parallel-`{edge,node}_blocks` → interleaved-`processor_layers` restructure + edge-MLP first-Linear input-column reorder `[src, dst, edge]` → `[edge, src, dst]` (the bug found at Gate D Band-C re-audit; see [DECISIONS.md D0-23](../methodology/DECISIONS.md#d0-23)).

### Phase-3-close git_sha

`<paste at Task 14 close — final commit sha on feature/case-study-02-physicsnemo-mgn>`

### Image spec / dependency pinning

`02-physicsnemo-mgn/modal_app.py` builds the Modal image with: `physicsnemo @ 1ca85d65ac2ce28ea9762910c09a954c08a37140`, `torch == 2.10.x`, `torch-scatter`, `scikit-fem == 12.0.1`, `dgl` removed (see [DECISIONS.md D0-23](../methodology/DECISIONS.md#d0-23) for the image-build history that resolved the Gate D image-side failures).

### Cross-references — methodology

- **Validation plan v2.1**: [`methodology/docs/physics-lint-validation-plan-v2.1.md`](../methodology/docs/physics-lint-validation-plan-v2.1.md) — §1.5 Case Study 02 as a falsification surface (open prediction; this README closes Phase 3 of that prediction).
- **Validation plan v2.1.1**: [`methodology/docs/physics-lint-validation-plan-v2.1.1.md`](../methodology/docs/physics-lint-validation-plan-v2.1.1.md) — Phase 3 amendment landing the four §5.5 items (pattern-C 4th instance + per-section convergence + D0-22 falsification + prose-vs-artifact-cross-review-modes).
- **Methodology README**: [`methodology/README.md`](../methodology/README.md) — rung-4 series integrating README (composing 4a + 4b + 4c + case-study artifacts).
- **DECISIONS catalogue**: [`methodology/DECISIONS.md`](../methodology/DECISIONS.md) — D0-22 (substrate-class taxonomy) · D0-23 (Phase 1 audit verdicts) · D0-24 (Phase 2 audit verdicts; this case study's authoritative numbers).
- **SCHEMA**: [`_harness/SCHEMA.md`](../_harness/SCHEMA.md) — SARIF run-level + result-level contract; D0-19 v1.0 already accommodates CS02-side optional fields (`arm`, `case_study`, `physicsnemo_sha`, `rollout_contract`, `trajectory_index`, `inference_run_status`, `ckpt_sha256`, `n_rollout_steps`, `ngc_version`, `rollout_npz_path`).

### Inference-gate test

Inference must pass `test_inference_matches_ngc_sample` (max-abs-error ≤ 10⁻³ on velocity components vs NGC's shipped sample) before rollouts proceed; this is the gate-determining test for Gate D (spec §6). Gate D PASSED at Phase 1 with RMSE-1 = 3.19e-3 ≤ 5e-3 (≈ 1.36× Pfaff et al. CylinderFlow RMSE-1 baseline) per [D0-23 verdict 4 recalibrated band](../methodology/DECISIONS.md#d0-23).
```

(Fill in the Step 1 + Step 2 values + the Phase-3-close git_sha at Task 14.)

- [ ] **Step 4: Commit.**

```bash
git add external_validation/_rollout_anchors/02-physicsnemo-mgn/README.md
git commit -m "02-physicsnemo-mgn: Phase 3 Task 7 — README Reproducibility section + methodology cross-references (Modal entrypoint table + NGC checkpoint provenance + image-spec pin + cross-refs)"
```

---

## Phase 3C — Cross-case-study back-port

### Task 8: CS01 README back-port — 3-5 line addition with framing language

**Goal:** Per Q3 refinement 2: CS01 README gains a `## Cross-stack tables` section (or expands the existing `## Cross-references` section) explaining that BOTH rung-4a (LB-only, frozen) and 2026-05-13 (extended, canonical going forward) cross-stack artifacts exist deliberately. Frozen-original-plus-pointer discipline.

**Files:**
- Modify: `external_validation/_rollout_anchors/01-lagrangebench/README.md`

- [ ] **Step 1: Add the back-port block.**

In `external_validation/_rollout_anchors/01-lagrangebench/README.md`, the existing `## Cross-references` section already lists rung-4a as "Cross-stack writeup (rung 4a)". Replace that single bullet with this expanded block:

```markdown
## Cross-references

### Cross-stack tables

- **Original LB-only (rung-4a, frozen):** [`../methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md`](../methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-table.md) — the original two-column cross-stack table over GNS-TGV2D + SEGNN-TGV2D. Preserved frozen at its commit date for regression-check value (LB-side comparison baseline if the extended renderer ever drifts).
- **Extended (CS02, canonical going forward):** [`../methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md`](../methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md) — unified three-column table extending rung-4a with the CS02 MGN-on-vortex-shedding column. The schema-uniformity claim now spans two substrate classes (dissipative-isotropic + open-driven-dissipative). Amendment 1 (Ahmed Body) extends with a fourth column when that row lands.

Both artifacts exist deliberately: the rung-4a artifact preserves the LB-only state at its commit date; the 2026-05-13 artifact extends that state with cross-substrate evidence per the frozen-original-plus-pointer methodology discipline.

### Methodology decisions

- **D0-19, D0-20** (rung-4a pre-registrations): [`../methodology/DECISIONS.md`](../methodology/DECISIONS.md).
- **D0-23, D0-24** (Case Study 02 Phase 1 + Phase 2 audit verdicts): same file.
```

- [ ] **Step 2: Commit.**

```bash
git add external_validation/_rollout_anchors/01-lagrangebench/README.md
git commit -m "01-lagrangebench: Phase 3 Task 8 — README back-port for unified cross-stack table (frozen-original-plus-pointer; both rung-4a + 2026-05-13 artifacts preserved with framing)"
```

---

## Phase 3D — Codex cross-review (prose-cross-review mode)

### Task 9: Codex adversarial review against writeup prose

**Goal:** Fire the Phase 3 boundary cross-review per design §3.3 review-methods + §4.3 acceptance criterion 8. Prose-cross-review mode (parallel to round-prose-1 / round-prose-2 / round-code-1-prose-dimension). Per [[feedback_codex_review_as_task_line_item]]: cross-review is a planned task, not an end-of-session ad-hoc step.

**Files:**
- Create: `external_validation/_rollout_anchors/methodology/docs/2026-05-13-case-study-02-phase-3-cross-review.md` (findings report; Codex output ingested + triaged here)

- [ ] **Step 1: Stage the prose under review.**

The Phase 3 cross-review scope is:
- `external_validation/_rollout_anchors/02-physicsnemo-mgn/README.md` (Tasks 4-7 expansions)
- `external_validation/_rollout_anchors/methodology/docs/2026-05-13-case-study-02-cross-stack-conservation-table.md` (Task 3 artifact)
- `external_validation/_rollout_anchors/01-lagrangebench/README.md` (Task 8 back-port)

Identify the commits at this state:

```bash
git log --oneline 0b97c14..HEAD
```

Record the start sha (`0b97c14` = Phase 2 close per handoff) and end sha (current HEAD).

- [ ] **Step 2: Dispatch the codex-rescue cross-review.**

Per the project's `codex:codex-rescue` skill (used at Phase 2 Task 11; same shape):

```
Codex review of CS02 Phase 3 writeup prose. Files: <three listed above>.

Adopt the application-reviewer voice (parallel to round-prose-2: Stuttgart/Munich
first-time-reviewer; round-code-1: GitHub Security tab user). Findings should
target:

1. Overclaiming — does any prose claim more than the evidence supports? Specific
   focus on the floor-bounds-resolution section (does "MGN reproduces GT at the
   floor" vs "MGN is physically incompressible to 5%" distinction hold under
   close reading? Could a reader miss the distinction?).
2. Hidden assumptions — does any prose implicitly assume a reader knows v2.1 /
   D0-19 / D0-23 / D0-24 framing without naming it?
3. Narrative drift — does the bridge sentence (Task 5) honestly position CS02 as
   a "falsification surface" per v2.1 §1.5, or has it drifted toward
   "validation surface" language?
4. Scope-qualifier completeness — does the scope-qualifier prose (single
   trajectory, in-band subset, coverage-not-statistics framing) hold against an
   adversarial reading? Are there easy over-reads?
5. Cross-stack table prose — is the schema-uniformity claim ("same harness across
   three rollouts of two substrate classes") clear, or does the cross-substrate
   asymmetry (MGN column's 5.881e-02 vs LB columns' 0.0 on `mass_conservation_defect`)
   risk being read as a model-quality comparison rather than a schema claim?

Output: numbered findings list with HIGH / MEDIUM / LOW severity, suggested
absorption shape, and the four-cell triage classification (cell-1 re-discovery,
cell-2 novel-in-scope, cell-3 novel-out-of-scope, cell-4 genuinely-new-framing).
```

(The actual dispatch uses the `codex:codex-rescue` skill; see Phase 2 Task 11 for the precedent invocation pattern.)

- [ ] **Step 3: Write the findings report.**

Create `external_validation/_rollout_anchors/methodology/docs/2026-05-13-case-study-02-phase-3-cross-review.md` mirroring the Phase 2 cross-review report's structure (`docs/2026-05-13-case-study-02-phase-2-cross-review.md`). Capture:
- Codex's voice + adversarial framing
- Each finding's severity + cell classification + recommended absorption
- The diff scope reviewed (start sha → end sha)
- A four-cell triage table at the bottom

- [ ] **Step 4: Commit.**

```bash
git add external_validation/_rollout_anchors/methodology/docs/2026-05-13-case-study-02-phase-3-cross-review.md
git commit -m "methodology: case study 02 Phase 3 prose cross-review findings (round-codex-Phase3)"
```

---

### Task 10: Absorb cross-review findings per trigger classification

**Goal:** Apply the Q4a pre-registered trigger classification to each cross-review finding. Cell-2 findings absorb in-rung per the Phase 2 precedent; cell-3 findings defer to amendment 1 / case study 03 / v2.1.2.

**Files:**
- Modify (per finding): the affected artifact (CS02 README / cross-stack table / CS01 README / v2.1.1 if not yet authored / etc.)
- Modify: `external_validation/_rollout_anchors/methodology/DECISIONS.md` (D0-24 absorption table — same pattern as Phase 2's 5-finding triage table at DECISIONS lines 2501-2511)

- [ ] **Step 1: Triage each finding per pre-registered trigger classification.**

For each Codex finding:

| Trigger | Action |
|---|---|
| **SARIF data-level** (sentinel misleading, field mis-emitted, schema violation) | CPU re-emit relevant SARIFs via `cs02_lint_gt_trajectory` / `cs02_lint_mgn_rollout` Modal entrypoint; commit absorbed-finding fix + re-emitted SARIFs in one commit. NO A10G fire required. Phase 2 F1 precedent. |
| **Fresh-inference need** (checkpoint provenance bug, inference protocol bug, metadata only obtainable from fresh run) | A10G fire fires (consumes the pre-registered ≤1 fire budget). Commit absorbed-finding fix + re-fired inference + re-emitted SARIFs together. **If a SECOND A10G need surfaces in absorption: STOP. Do NOT auto-fire. Escalate to out-of-Phase-3 per Q4a falsification — the cap binds by rationale.** |
| **Out-of-Phase-3 scope** (substrate not yet probed, surface for amendment 1 or case study 03) | Defer to amendment 1 / case study 03 candidate slot. Record the deferral in DECISIONS.md D0-24 absorption table with the explicit deferral target (amendment 1 / amendment 2 / case study 03 / v2.1.2). Do NOT fire compute. |
| **Prose-level absorbable** (overclaim correction, framing sharpening, hidden-assumption naming) | Edit the affected prose file; commit. Most Phase 3 findings expected here (parallel to round-prose-1 / round-prose-2). |

- [ ] **Step 2: Absorb each finding via its own commit (per Phase 2 absorption-commit pattern; multiple findings per commit acceptable only if they share a target file + absorption shape).**

Per Phase 2 precedent (commits `5ad2cb8`, `a6fbd14`, `efcac9c`):
- Group findings by target file when possible
- Commit message format: `02-physicsnemo-mgn: absorb cross-review finding(s) <N> + ... — <one-line summary>`
- Re-emit SARIFs in a follow-up commit if the absorption changes the SARIF schema (per Phase 2 `efcac9c` re-emission commit).

- [ ] **Step 3: Update DECISIONS.md D0-24 absorption table.**

Append a Phase-3 absorption block to the D0-24 entry, parallel to Phase 2's 5-finding table at DECISIONS lines 2501-2511. Format:

```markdown
**Phase 3 boundary cross-review (Task 9-10) — findings triaged.** Codex/GPT-5-CLI adversarial prose review of CS02 Phase 3 writeup (commits `<start_sha>..<end_sha>`) at `docs/2026-05-13-case-study-02-phase-3-cross-review.md`. <N> findings; <triaged per trigger classification>.

| # | Finding | Severity | Cell | Absorption / Defer |
|---|---|---|---|---|
| 1 | ... | ... | ... | Absorbed at `<sha>` — ... |
| 2 | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |

**Totals:** N cell-1 deferred, N cell-2 absorbed, N cell-3 forward-flagged, N cell-4 open.
```

- [ ] **Step 4: Commit the DECISIONS update separately.**

```bash
git add external_validation/_rollout_anchors/methodology/DECISIONS.md
git commit -m "DECISIONS.md: D0-24 Phase 3 cross-review triage summary"
```

---

## Phase 3E — v2.1.1 amendment

### Task 11: Author `physics-lint-validation-plan-v2.1.1.md` (new dated artifact)

**Goal:** Land the four §5.5 items in a new dated artifact at `methodology/docs/physics-lint-validation-plan-v2.1.1.md`. Two-category grouping per Q1 refinement 3 (framework-behavior observations vs specific-finding flags). v2.1.1 vs v2.1.2 epistemic distinction named in opening per Q1 refinement 1. Point-outward shape on items 3 + 4 per Q1 refinement 2.

**Files:**
- Create: `external_validation/_rollout_anchors/methodology/docs/physics-lint-validation-plan-v2.1.1.md`

- [ ] **Step 1: Create the file with the agreed structure.**

```markdown
# physics-lint validation plan v2.1.1 — Phase 3 amendment (Case Study 02)

**Date:** 2026-05-13
**Predecessor:** [`./physics-lint-validation-plan-v2.1.md`](./physics-lint-validation-plan-v2.1.md) (frozen at its commit date; pointers added at §1.3 + §1.4 + §3 changelog forward-flag to mark resolution of v2.1.1's four items).
**Triggered by:** [Case Study 02 Phase 3 design §5.5](./2026-05-11-case-study-02-physicsnemo-mgn-design.md#55-v211-amendment--phase-3-deliverable-33-activity-7) trigger condition ("Phase 3 cross-review of writeup prose completes") — fired at Task 9 + Task 10 close.
**Successor:** `physics-lint-validation-plan-v2.1.2.md` (forthcoming, separate; absorbs execution-emergent discoveries per the v2.1.1 vs v2.1.2 epistemic distinction below).

---

## v2.1.1 vs v2.1.2 — pre-registered vs execution-emergent

v2.1.1 absorbs the **four amendments pre-registered in v2.1 design §5.5** (predicted at design time before Phase 1 fired; landed at Phase 3 close as the design predicted).

v2.1.2 (separate, forthcoming) will absorb **execution-emergent methodology discoveries** — round-code-1 walls 1-2 (level rendering + file-path location) and CS02 Phase 2's fourth empirical prose-scope-qualifier instance (floor-bounds-resolution distinction). These were discovered DURING execution rather than predicted at design time; they warrant separate v2.1.2 framing.

The epistemic distinction matters for the methodology trail's honesty: v2.1.1 says *"we predicted these would land at Phase 3; here they are, as predicted."* v2.1.2 will say *"execution surfaced these; here's what we learned that we didn't anticipate."* Both are legitimate methodology contributions; the separation lets the writeup honestly track *what was foreseen* vs *what was empirically discovered*.

---

## Two categories — framework-behavior observations + specific-finding flags

v2.1.1's four amendments split into two categories:

- **Framework-behavior observations** — observations about how the methodology framework itself evolved across the design-doc rounds. Items 1 + 2.
- **Specific-finding flags** — flags about specific findings from CS02 execution that point outward to dated artifacts holding the full evidence. Items 3 + 4.

---

## §1. Framework-behavior observations

### §1.1 Pattern-C 4th instance documentation (resolves v2.1 §1.3)

The pattern-C catalogue at v2.1 §1.3 currently lists three within-rung-4c instances (rung-4c §9 fold-in round 1 + round 2 + round 3). Round-prose-1's absorption already started counting a 4th cell-2 instance (round-prose-1 itself). CS02 Phase 1's D0-23 + Phase 2's D0-24 + Phase 3's prose cross-review supply additional cell-2 instances; v2.1.1 documents the 4th-instance threshold being crossed and elevates pattern-C from "rung-4c-internal observation" to "validated within at least one cross-rung context (CS02)."

**Outward pointer:** see [DECISIONS.md D0-23 + D0-24](./DECISIONS.md) for the per-case-study triage tables; v2.1 §1.3 receives a pointer to this v2.1.1 entry marking the 4th-instance threshold cross.

### §1.2 Per-section convergence-pattern observation (resolves v2.1 §1.3 + §3 changelog)

Across v2.1's review-round sequence (round 1 → round 2 → round 3 → round-prose-1 → round-prose-2 → round-codex-2/-3/-4 → round-code-1), the round-magnitude of findings has decreased per-section (§1 review → §2.1-§2.3 review → §2.4-§2.6 review → §3 review → §4 + §5 review → §6 review). Each round's findings cluster on the most-recently-added section content; older sections converge to "no new findings." This convergence is measured *per-section*, not *per-doc* — adding new sections re-opens the per-section convergence at the new section.

The observation supports v2.1 §1.3's pattern-C "absorbed → fewer findings on this surface" claim and v2.1 §1.4's "cross-review-gate cost-amortizes across rungs" claim. The per-section granularity is the methodology contribution; without it, the convergence claim could be read at the wrong granularity.

**Outward pointer:** v2.1 §1.3 + §1.4 receive pointers to this v2.1.1 entry.

---

## §2. Specific-finding flags

### §2.1 D0-22 strictly-conservative forward-declaration falsification flag

CS02 design §1 (preflight evidence + substrate-class smoke design) falsifies v2.1's D0-22 strictly-conservative anchor assignment: cylinder_flow is NOT strictly conservative; it's open-driven-dissipative under physicsnemo's mesh-side classification. D0-22's anchor assignment named TGV2D as the dissipative-isotropic anchor and (forward-declared) cylinder_flow as the strictly-conservative anchor for a future case study; CS02 Phase 1 + Phase 2 verdicts confirm cylinder_flow as open-driven-dissipative, falsifying the forward declaration.

v2.1.1 records this flag. The corrective amendment (re-assigning the strictly-conservative anchor to a different upstream dataset) defers to case study 03 candidate identification — assigning speculatively before a strictly-conservative substrate is concretely identified would risk the same forward-declaration error a second time.

**Outward pointer:** see [Case Study 02 design §1](./2026-05-11-case-study-02-physicsnemo-mgn-design.md#1-context) (preflight + substrate-class smoke design) for where the falsification first surfaced; v2.1 §1.3 receives a pointer to this v2.1.1 entry.

### §2.2 Prose-cross-review vs artifact-cross-review modes flag

Round-prose-2's forward-flag (v2.1 §3 changelog, line 329) named the distinction between the two cross-review modes: artifact-cross-review surfaces fail-open paths (round-codex-2/-3/-4 precedent); prose-cross-review surfaces first-impression weaknesses (round-prose-1 + round-prose-2 precedent). Round-code-1 already INSTANTIATED the flag (committed prose + working SARIF upload + Security tab populated); CS02 Phase 3's prose cross-review provides a fifth instance, completing the empirical evidence for the distinction.

v2.1.1 names the flag with a one-paragraph summary and points outward. Full instantiation evidence:
- **Prose-cross-review precedents**: round-prose-1, round-prose-2, round-code-1 prose dimension, CS02 Phase 3 cross-review (this Phase).
- **Artifact-cross-review precedents**: round-codex-2 + round-codex-3 + round-codex-4 (rung-4c PR #9 reviews).

The §1.4 review-discipline triple (smoke + source + cross) does not currently distinguish the two cross-review modes; v2.1.2 §1.4 amendment is the right venue for adding the distinction with worked criteria for when each mode applies.

**Outward pointer:** see v2.1 §3 round-prose-2 changelog forward-flag (line 329) which is now RESOLVED with a pointer to this v2.1.1 entry; full instantiation evidence at [round-code-1 PR #10 merge](../../README.md#hero-physics-lint-in-ci) (level rendering / file-path location / prose framing) + [CS02 Phase 3 cross-review](./2026-05-13-case-study-02-phase-3-cross-review.md).

---

## Acceptance

This file lands the four §5.5 items per their pre-registered scope. Light self-check (Task 13) verifies each item lands + two-category grouping preserved + outward references SHA-pinned.

Any v2.1.1+ expansion (a fifth pattern-C instance, a new framework-behavior observation, a CS02-emergent finding warranting framework-level absorption) goes to v2.1.2 or a fresh forward-flag — NOT retrofitted into v2.1.1.
```

- [ ] **Step 2: Commit v2.1.1 as a single commit (do not commit pointer additions yet — those land in Task 12).**

```bash
git add external_validation/_rollout_anchors/methodology/docs/physics-lint-validation-plan-v2.1.1.md
git commit -m "methodology: physics-lint-validation-plan-v2.1.1 — Phase 3 amendment (4 pre-registered items grouped into framework-behavior + specific-finding categories; v2.1.1 vs v2.1.2 epistemic distinction named)"
```

---

### Task 12: Add pointers in v2.1.md §1.3 + §1.4 + §3 changelog forward-flag

**Goal:** Frozen-original-plus-pointer discipline applied to v2.1.md. Pointer-only additions to §1.3 (items 1, 2, 3 reside here per the framework-behavior category + D0-22 falsification flag's §1.3 home), §1.4 (item 4 resides here per the cross-review-discipline section), and §3 changelog (round-prose-2 forward-flag at line 329 marked RESOLVED with pointer to v2.1.1 §2.2). NO content edits to v2.1.md beyond the pointers.

**Files:**
- Modify: `external_validation/_rollout_anchors/methodology/docs/physics-lint-validation-plan-v2.1.md` (pointers only)

- [ ] **Step 1: Add pointer to v2.1 §1.3 (one or two lines).**

In v2.1.md §1.3, immediately after the closing paragraph of §1.3 (find the end-of-section), insert:

```markdown
**v2.1.1 amendment** — three §1.3-resident items absorbed at [`./physics-lint-validation-plan-v2.1.1.md`](./physics-lint-validation-plan-v2.1.1.md): §1.1 pattern-C 4th-instance threshold cross, §1.2 per-section convergence-pattern observation, §2.1 D0-22 strictly-conservative forward-declaration falsification flag (CS02 Phase 1 + Phase 2 confirmed cylinder_flow as open-driven-dissipative; D0-22's forward-declared anchor falsified; corrective amendment deferred to case study 03). v2.1's §1.3 content unchanged; this pointer is an addition only.
```

- [ ] **Step 2: Add pointer to v2.1 §1.4 (one or two lines).**

In v2.1.md §1.4, immediately after the closing paragraph of §1.4 (find the end-of-section), insert:

```markdown
**v2.1.1 amendment** — §1.4-resident item absorbed at [`./physics-lint-validation-plan-v2.1.1.md`](./physics-lint-validation-plan-v2.1.1.md) §2.2: prose-cross-review vs artifact-cross-review modes flag. CS02 Phase 3's prose cross-review supplies a fifth empirical instance completing the distinction; the formal modes-as-§1.4-content-extension defers to v2.1.2 §1.4 amendment. v2.1's §1.4 content unchanged; this pointer is an addition only.
```

- [ ] **Step 3: Mark the round-prose-2 forward-flag (v2.1 §3 changelog line 329) as RESOLVED.**

Find the existing line in v2.1.md §3 changelog (round-prose-2 absorption entry):

```markdown
**Forward-flag (not absorbed in this round).** Round-prose-2 distinguishes itself from rounds 1-2 and round-codex-2 / -3 / -4 by being cross-review of *prose* rather than *artifacts*; the two cross-review modes have different findings characters (artifacts surface fail-open paths; prose surfaces first-impression weaknesses). §1.4's "smoke + source + cross" review-discipline triple does not currently distinguish these modes. The distinction is a v2.1.1+ §1.4 amendment candidate after case study 02's analogous prose review lands and provides a second instance of the prose-cross-review mode.
```

Replace with (adding a `RESOLVED →` marker at the start, preserving the original text after):

```markdown
**Forward-flag — RESOLVED at v2.1.1 §2.2 (2026-05-13).** Round-prose-2 distinguishes itself from rounds 1-2 and round-codex-2 / -3 / -4 by being cross-review of *prose* rather than *artifacts*; the two cross-review modes have different findings characters (artifacts surface fail-open paths; prose surfaces first-impression weaknesses). §1.4's "smoke + source + cross" review-discipline triple does not currently distinguish these modes. **Resolution:** CS02 Phase 3's prose cross-review supplied the second prose-cross-review-mode instance (and round-code-1's three prose-dimension walls supplied a third); the cumulative pattern is documented at [`./physics-lint-validation-plan-v2.1.1.md`](./physics-lint-validation-plan-v2.1.1.md) §2.2 with outward pointers to all five precedents. Formal mode-distinction-as-§1.4-content-extension defers to v2.1.2 §1.4 amendment per the v2.1.1 vs v2.1.2 epistemic distinction.
```

- [ ] **Step 4: Commit (pointers + RESOLVED marker in one commit).**

```bash
git add external_validation/_rollout_anchors/methodology/docs/physics-lint-validation-plan-v2.1.md
git commit -m "methodology: physics-lint-validation-plan-v2.1 — frozen-original-plus-pointer additions for v2.1.1 (§1.3 + §1.4 pointers; §3 changelog round-prose-2 forward-flag marked RESOLVED)"
```

---

### Task 13: Light self-check on v2.1.1

**Goal:** Per Q4b refinement 3: 10-15 minute authored-content sanity check on v2.1.1 BEFORE the §4.3 acceptance closes. Read v2.1.1 against the four §5.5 pre-registered items. Verify two-category grouping preserved. Verify SHA-pinned outward references correct per Q3 refinement 1. NOT a Codex round; just a manual self-check.

**Files:**
- None modified (read-only check; if defects found, amend in subsequent commits)

- [ ] **Step 1: Verify all four §5.5 items land in v2.1.1.**

Open [CS02 design §5.5](`external_validation/_rollout_anchors/methodology/docs/2026-05-11-case-study-02-physicsnemo-mgn-design.md#55-v211-amendment--phase-3-deliverable-33-activity-7`) and check each item:

| §5.5 item | v2.1.1 location | Check |
|---|---|---|
| 1. Pattern-C 4th-instance documentation | §1.1 | [ ] |
| 2. Per-section convergence-pattern observation | §1.2 | [ ] |
| 3. D0-22 strictly-conservative forward-declaration falsification flag | §2.1 | [ ] |
| 4. Prose-vs-artifact-cross-review-modes flag | §2.2 | [ ] |

If any item is missing or substantially mis-framed: amend v2.1.1 + recommit before Task 14.

- [ ] **Step 2: Verify two-category grouping preserved.**

Read v2.1.1's §1 header ("Framework-behavior observations") and §2 header ("Specific-finding flags"). Verify items 1 + 2 are under §1 and items 3 + 4 are under §2. If items have drifted: re-organize before Task 14.

- [ ] **Step 3: Verify v2.1.1 vs v2.1.2 distinction named in opening.**

Read v2.1.1's opening section. Verify the explicit paragraph stating: v2.1.1 absorbs pre-registered amendments; v2.1.2 will absorb execution-emergent discoveries. If the distinction is absent or fuzzy: add the paragraph before Task 14.

- [ ] **Step 4: Verify SHA-pinned outward references correct.**

Find every outward reference in v2.1.1 (look for `](./` and `](../`). For each:
- If the target is a dated artifact (e.g., `2026-05-11-...-design.md`, `2026-05-13-...-cross-review.md`): no SHA pin needed (the dated filename is the pin).
- If the target is a non-dated file (e.g., `README.md`, `DECISIONS.md`, `physics-lint-validation-plan-v2.1.md`): the reference should either be SHA-pinned at the v2.1.1 commit OR the reference should be to a stable anchor (e.g., a section heading) inside the file. Outward references to CS02 README sections specifically should SHA-pin per Q3 refinement 1.

If SHA pins are missing where they should be: add them at the v2.1.1 commit sha (the most recent commit on this branch after Task 12).

- [ ] **Step 5: Verify outward references' link targets exist.**

```bash
grep -oE '\]\(\./[^)]+\)' external_validation/_rollout_anchors/methodology/docs/physics-lint-validation-plan-v2.1.1.md | sed 's/^](\.\///' | sed 's/)$//' | while read p; do
    if [ ! -e "external_validation/_rollout_anchors/methodology/docs/$p" ]; then
        echo "BROKEN: $p"
    fi
done
```

Expected: no `BROKEN:` lines. If any: fix or remove the reference before Task 14.

- [ ] **Step 6: If amendments fired in Steps 1-5, commit them.**

```bash
git add external_validation/_rollout_anchors/methodology/docs/physics-lint-validation-plan-v2.1.1.md
git commit -m "methodology: physics-lint-validation-plan-v2.1.1 — self-check amendments (see plan Task 13)"
```

(If no amendments fired: skip this commit; the self-check itself is not a code change.)

---

## Phase 3F — Close

### Task 14: Update DECISIONS.md D0-24 Phase 3 close + §4.3 acceptance checklist + commit + push

**Goal:** Close D0-24 with the Phase 3 acceptance checklist + a Phase 3 COMPLETE marker. Verify the §4.3 acceptance criteria are all checked. Push the branch.

**Files:**
- Modify: `external_validation/_rollout_anchors/methodology/DECISIONS.md` (D0-24 Phase 3 close block)
- Modify: `external_validation/_rollout_anchors/02-physicsnemo-mgn/README.md` (Task 7's Phase-3-close git_sha placeholder gets filled)

- [ ] **Step 1: Fill the Phase-3-close git_sha placeholder in CS02 README.**

```bash
git log --oneline | head -1 | awk '{print $1}'
```

Take that sha (it will be the most recent commit on this branch before this Task 14 commit). Edit `02-physicsnemo-mgn/README.md` Task 7 section "Phase-3-close git_sha" to use the sha:

```markdown
### Phase-3-close git_sha

`<paste sha here>`
```

(If you want a more stable reference: use the commit sha AFTER Task 14 lands; that requires two commits — one to land the rest of Task 14, then a second amendment to write the final sha in. The simpler one-commit shape uses the sha just-before-Task-14.)

- [ ] **Step 2: Append Phase 3 close block to D0-24 in DECISIONS.md.**

Append after the Phase 2 absorption block (after the line "**Phase 2 COMPLETE (2026-05-13)**."):

```markdown
**Phase 3 §4.3 acceptance checklist.**

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | `02-physicsnemo-mgn/README.md`: Rule × checkpoint results table populated (vortex shedding row only) | ☑ | Task 4 (`<sha>`) — PH-CON-001/002/003 numbers + verbatim scope-qualifier prose. |
| 2 | Bridge-sentence paragraph drafted | ☑ | Task 5 (`<sha>`) — links CS02 results into rung-4 cross-stack story. |
| 3 | Cross-stack consistency table populated (vortex shedding row; Ahmed Body row marked "deferred to amendment 1") | ☑ | Task 3 (`<sha>`) — new dated artifact `2026-05-13-case-study-02-cross-stack-conservation-table.md` extending rung-4a; Ahmed Body deferral named in the artifact's "What this table is NOT" section. |
| 4 | "What physics-lint did NOT catch" section written | ☑ | Task 6 (`<sha>`) — 7 sub-sections covering floor-bounds-resolution + substrate-variability + PH-CON-001 routing + PH-NUM-002 deferral + PH-SYM-* scope + Ahmed Body/PH-RES-001 deferral + v2.1.2 forward-flag. |
| 5 | Reproducibility section: Modal entrypoints, NGC checkpoint hash, git_sha, conda lockfile | ☑ | Task 7 (`<sha>`) — Modal entrypoint table + NGC checkpoint provenance (`ngc_version`, `ckpt_sha256`, `physicsnemo_sha`) + image-spec pin. |
| 6 | Methodology cross-references to `methodology/README.md` + v2.1.md §1.5 | ☑ | Task 7 (`<sha>`) — Cross-references subsection lists all targets. |
| 7 | v2.1.1 amendment landed (per §5.5) | ☑ | Task 11 (`<sha>`) — `physics-lint-validation-plan-v2.1.1.md` with 4 items in 2 categories; Task 12 (`<sha>`) — v2.1.md §1.3 + §1.4 + §3 changelog pointers added (frozen-original-plus-pointer); Task 13 — light self-check passed. |
| 8 | Phase 3 boundary cross-review complete; findings triaged | ☑ | Task 9 (`<sha>`) — `2026-05-13-case-study-02-phase-3-cross-review.md` with N findings; Task 10 (`<sha>`) — D0-24 absorption table updated. |

Plus the CS01 back-port (cross-stack discoverability symmetry per Q3 refinement 2): Task 8 (`<sha>`) — CS01 README points to both rung-4a (frozen) + 2026-05-13 (canonical going forward) cross-stack artifacts.

**Phase 3 COMPLETE (2026-05-13).** All 8 §4.3 acceptance criteria met. CS02 README expanded to ~<N>-line case-internal writeup; new dated cross-stack table artifact extends rung-4a via the frozen-original-plus-pointer discipline; v2.1.1 dated artifact lands the four pre-registered §5.5 items in two categories; v2.1.md gains pointer-only additions at §1.3, §1.4, and §3 changelog forward-flag (RESOLVED → v2.1.1); CS01 README gets a 3-5 line back-port. Phase 3 boundary cross-review absorbed in-rung per the pre-registered trigger classification (Q4a). The v2.1.2 §1.4 prose-scope-qualifier formalization is the next methodology amendment in the trail; tracked separately, post-Phase-3.

**Case Study 02 COMPLETE.** Phases 1 + 2 + 3 all closed clean. D0-23 (Phase 1) + D0-24 (Phase 2 + Phase 3) carry the full audit trail. Successor work: amendment 1 (Ahmed Body / PH-BC-001 / steady RANS) per design §5.1; case study 03 candidate (strictly-conservative anchor reassignment per design §5.10 + the v2.1.1 §2.1 falsification flag); cover-letter cross-case-study integration per design §5.6 (now unblocked — both CS01 + CS02 writeups exist).
```

- [ ] **Step 3: Commit the D0-24 close.**

```bash
git add external_validation/_rollout_anchors/methodology/DECISIONS.md \
        external_validation/_rollout_anchors/02-physicsnemo-mgn/README.md
git commit -m "DECISIONS.md: D0-24 Phase 3 §4.3 acceptance closed; Phase 3 COMPLETE; Case Study 02 COMPLETE"
```

- [ ] **Step 4: Push the branch.**

```bash
git push -u origin feature/case-study-02-physicsnemo-mgn
```

Expected: push succeeds. If pre-push hooks fail: fix the issue (do NOT `--no-verify`) and re-push.

- [ ] **Step 5: Final verification.**

```bash
# Confirm the branch is at the expected state
git log --oneline 0b97c14..HEAD | wc -l    # Expect 14+ commits (one per Task + absorption commits)
git status                                   # Expect clean
git diff --stat origin/feature/case-study-02-physicsnemo-mgn  # Expect empty (pushed)
```

---

## Acceptance criteria recap (design §4.3)

- [x] `02-physicsnemo-mgn/README.md`: Rule × checkpoint results table populated (Task 4).
- [x] Bridge-sentence paragraph drafted (Task 5).
- [x] Cross-stack consistency table populated; Ahmed Body row marked deferred to amendment 1 (Task 3).
- [x] "What physics-lint did NOT catch" section written (Task 6).
- [x] Reproducibility section: Modal entrypoints, NGC checkpoint hash, git_sha, conda lockfile (Task 7).
- [x] Methodology cross-references to `methodology/README.md` + v2.1.md §1.5 (Task 7).
- [x] v2.1.1 amendment landed per §5.5 (Tasks 11-13).
- [x] Phase 3 boundary cross-review complete; findings triaged (Tasks 9-10).

---

## Out-of-scope (preserved deferrals per design §5)

- **Ahmed Body / PH-BC-001 / steady RANS** → amendment 1 (design §5.1).
- **PH-RES-001 (BDO)** → amendment 2 / case study 03 (design §5.2).
- **PH-NUM-002 resolution sweep** → v1.1 backlog (design §5.3).
- **D0-22 strictly-conservative anchor reassignment** → case study 03 candidate (design §5.10 + v2.1.1 §2.1).
- **Cover-letter cross-case-study integration** → post-Phase-3, across cases (design §5.6 — now unblocked).
- **v2.1.2 §1.4 prose-scope-qualifier formalization** → separate, after v2.1.1 lands.
- **No active tightening of floor-bounds-resolution via finer-discretization re-fire** — would be empirical work; properly scoped to v1.1 backlog or a future case study.

---

## Provenance

- **Phase 1 plan**: [`./2026-05-11-case-study-02-physicsnemo-mgn-phase-1-plan.md`](./2026-05-11-case-study-02-physicsnemo-mgn-phase-1-plan.md) (15 tasks; closed at sha `3c3c8f2`).
- **Phase 2 plan**: [`./2026-05-13-case-study-02-physicsnemo-mgn-phase-2-plan.md`](./2026-05-13-case-study-02-physicsnemo-mgn-phase-2-plan.md) (12 tasks; closed at sha `0b97c14`).
- **Phase 3 plan**: THIS FILE.
- **Design doc (all 3 phases)**: [`./2026-05-11-case-study-02-physicsnemo-mgn-design.md`](./2026-05-11-case-study-02-physicsnemo-mgn-design.md) — §3.3 + §4.3 + §5.5 cover Phase 3 scope.
- **Brainstorming source (this plan's parameters)**: Q1-Q4 alignment session immediately preceding plan-write; synthesis at the plan's Architecture + Brainstorming-decided constraints sections.

**Brainstorming → plan deltas (what this plan refined from brainstorming inputs):**
1. **Renderer parameterization SCOPE-REDUCED**: pre-flight verification (Task 1) confirmed CS02 SARIFs satisfy the existing `_assert_run_level` contract without per-source parameterization. Q2's anticipated parameterization is YAGNI; Task 2 reduces to "ingest from both dirs + filter GT-arm." If pre-flight fails: re-open Q2 + amend plan before Task 2 fires.
2. **v2.1.md pointer targets clarified**: Q1's "RESOLVED → v2.1.1 pointers in v2.1.md §2" mental model was inspected against actual v2.1.md content; none of v2.1.1's four items resolve a §2 deferral. The verified pointer targets are §1.3 (items 1, 2, 3) + §1.4 (item 4) + §3 changelog round-prose-2 forward-flag (item 4 RESOLVED marker). Task 12 implements these targets.
3. **Substrate-variability finding placement**: per Q3 scope guardrails, lands in CS02 README's "what physics-lint did NOT catch" §2 — NOT in the cross-stack table.

**Memory references applicable to plan execution:**
- [[feedback_precommit_venv]] — `source .venv/bin/activate && git commit ...` in physics-lint; numpy<2 pin for torch ABI.
- [[feedback_codex_review_as_task_line_item]] — Codex review is the planned Task 9; absorption is planned Task 10; version close held until findings triaged.
- [[feedback_cap_rationale_not_literal]] — Q4a A10G contingency budget binds by rationale (absorb cell-2 artifact-level findings), not literal text; second A10G need escalates out-of-scope.
- [[feedback_ascii_hyphens_in_python]] — `.py` files use ASCII `-`; `.md` files keep en/em dashes.
- [[feedback_path_invention]] — external-repo paths need filesystem verification before docs/plans cite them. (Phase 3 paths are all in-repo; no external-repo citation risk.)
- [[feedback_readback_gate_layering]] — execution-plan layer (Categories 7 + 8) verification: Task 1's pre-flight covers Category 7 (toolchain precondition: existing assertion on real fixtures); Task 12's pointer verification covers Category 8 (codebase API contract: verify v2.1.md anchors exist before linking).
