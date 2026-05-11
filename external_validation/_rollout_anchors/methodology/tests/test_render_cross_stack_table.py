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
    DuplicateStackLabelError,
    MissingRunLevelFieldError,
    ResultRowInvariantError,
    SchemaVersionMismatchError,
    SourceTagMismatchError,
    render_cross_stack_table,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SEGNN_FIXTURE = FIXTURES_DIR / "segnn_tgv2d_fixture.sarif"
GNS_FIXTURE = FIXTURES_DIR / "gns_tgv2d_fixture.sarif"

# Mirror of the renderer's D-entry regex (single source of truth would be
# ideal but introduces a circular module dependency; the inline copy is a
# deliberate drift-guard — if the renderer's regex in
# external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py
# changes, the regex below must change in lockstep, which is exactly the
# contract these tests pin).
DENTRY_REGEX = r"DECISIONS\.md\s+(D0-\d+(?:\s+\(amendment\s+\d+\))?)"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(d: dict, path: Path) -> None:
    path.write_text(json.dumps(d, indent=2, sort_keys=True))


# ---------------------------------------------------------------------------
# 1. Schema-version assertion
# ---------------------------------------------------------------------------


def test_schema_version_mismatch_raises(tmp_path: Path) -> None:
    """Bumped harness_sarif_schema_version -> SchemaVersionMismatchError raises.
    Programmatically-derived from the canonical fixture (per memory:
    don't commit a separate bumped-version fixture file).
    """
    bumped = copy.deepcopy(_load(SEGNN_FIXTURE))
    bumped["runs"][0]["properties"]["harness_sarif_schema_version"] = "99.0"
    bumped_path = tmp_path / "bumped.sarif"
    _write(bumped, bumped_path)

    with pytest.raises(SchemaVersionMismatchError):
        render_cross_stack_table([bumped_path, GNS_FIXTURE])


def test_source_tag_mismatch_raises(tmp_path: Path) -> None:
    """Wrong source field -> SourceTagMismatchError raises."""
    bad = copy.deepcopy(_load(SEGNN_FIXTURE))
    bad["runs"][0]["properties"]["source"] = "physics-lint-public-api"
    bad_path = tmp_path / "bad_source.sarif"
    _write(bad, bad_path)

    with pytest.raises(SourceTagMismatchError):
        render_cross_stack_table([bad_path, GNS_FIXTURE])


def test_missing_run_level_field_raises(tmp_path: Path) -> None:
    """Deleting any required D0-19 run-level field -> MissingRunLevelFieldError raises."""
    incomplete = copy.deepcopy(_load(SEGNN_FIXTURE))
    del incomplete["runs"][0]["properties"]["physics_lint_sha_pkl_inference"]
    incomplete_path = tmp_path / "incomplete.sarif"
    _write(incomplete, incomplete_path)

    with pytest.raises(MissingRunLevelFieldError):
        render_cross_stack_table([incomplete_path, GNS_FIXTURE])


def test_no_sarif_paths_raises() -> None:
    """Empty input -> MissingRunLevelFieldError (chosen because the renderer
    has no run-level data to operate on; parallel category to missing
    fields).
    """
    with pytest.raises(MissingRunLevelFieldError):
        render_cross_stack_table([])


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
    assert "synthetic_inference_sha" in table
    assert "synthetic_conversion_sha" in table
    assert "synthetic_combined_sha" in table


def test_renderer_emits_markdown_table_with_three_rules() -> None:
    """Smoke test: rendered output is a markdown table mentioning the
    three conservation rules.
    """
    table = render_cross_stack_table([SEGNN_FIXTURE, GNS_FIXTURE])
    assert "mass_conservation_defect" in table
    assert "energy_drift" in table
    assert "dissipation_sign_violation" in table


def test_renderer_detects_all_n_identical_aggregation() -> None:
    """Per D0-20: 'all N identical -> single cell' detection. All
    mass_conservation_defect rows in segnn_tgv2d_fixture have raw_value
    = 0.0; the rendered cell for that (rule, stack) reports a single
    value, not a min/max range.
    """
    table = render_cross_stack_table([SEGNN_FIXTURE, GNS_FIXTURE])
    assert table.count("0.0") >= 4 or table.count("0.000e+00") >= 4


def test_skip_row_missing_skip_reason_raises(tmp_path: Path) -> None:
    """Per D0-19 §3.4 + Codex adversarial review finding: a SKIP row
    must carry properties.skip_reason. Removing it must raise
    ResultRowInvariantError, not silently aggregate to "SKIP (xN, D0-18)".

    This is the regression guard for the bug Codex caught: pre-fix
    artifacts had no skip_reason on SKIP rows and the renderer
    aggregated them anyway via the (raw_value is None) shortcut.
    """
    bad = copy.deepcopy(_load(SEGNN_FIXTURE))
    # Strip skip_reason from one energy_drift SKIP row.
    for r in bad["runs"][0]["results"]:
        if r["ruleId"] == "harness:energy_drift" and "skip_reason" in r["properties"]:
            del r["properties"]["skip_reason"]
            break
    bad_path = tmp_path / "missing_skip_reason.sarif"
    _write(bad, bad_path)

    with pytest.raises(ResultRowInvariantError):
        render_cross_stack_table([bad_path, GNS_FIXTURE])


def test_skip_reason_divergence_raises(tmp_path: Path) -> None:
    """Per D0-19 §3.4: skip_reason is guaranteed-identical across rows
    within a (rule, stack). Two distinct skip_reason values within one
    stack must raise ResultRowInvariantError.
    """
    bad = copy.deepcopy(_load(SEGNN_FIXTURE))
    # Mutate one SKIP row's skip_reason so two distinct values exist.
    mutated = False
    for r in bad["runs"][0]["results"]:
        if r["ruleId"] == "harness:energy_drift" and not mutated:
            r["properties"]["skip_reason"] = "DIVERGENT — should not happen per D0-19 §3.4"
            mutated = True
            break
    assert mutated, "fixture should contain at least one SKIP row to mutate"
    bad_path = tmp_path / "divergent_skip_reason.sarif"
    _write(bad, bad_path)

    with pytest.raises(ResultRowInvariantError):
        render_cross_stack_table([bad_path, GNS_FIXTURE])


def test_renderer_golden_output_matches_expected_table() -> None:
    """Golden test: rendering the canonical fixtures produces output
    byte-for-byte identical to expected_table.md. This pins the
    renderer's contract -- any non-trivial change in output requires a
    paired update to expected_table.md.
    """
    expected = (FIXTURES_DIR / "expected_table.md").read_text()
    actual = render_cross_stack_table([SEGNN_FIXTURE, GNS_FIXTURE])
    assert actual == expected, (
        f"Renderer output diverged from expected_table.md.\n"
        f"--- expected ---\n{expected}\n"
        f"--- actual ---\n{actual}\n"
        f"Regenerate (preserving SEGNN-first column order; CLI's "
        f"alphabetical glob would reverse this) by re-running:\n"
        f"  python external_validation/_rollout_anchors/methodology/tools/"
        f"regenerate_expected_table.py"
    )


# ---------------------------------------------------------------------------
# Round-codex-2 follow-up B: contract test for the D-entry regex extraction.
# The renderer parses each SKIP row's skip_reason via a regex to extract
# the cited D-entry for the cell label. The emitter
# (_harness/particle_rollout_adapter.py) produces skip_reasons in five
# distinct formats; if the regex doesn't match one of them, the cell
# label silently degrades to "?". Drift-guard: feed each actual emitter
# skip_reason format through the regex + assert the expected D-entry is
# extracted.
# ---------------------------------------------------------------------------


# Skip_reason templates copied verbatim from
# external_validation/_rollout_anchors/_harness/particle_rollout_adapter.py.
# Each tuple is (skip_reason_text, expected_d_entry_match). If the emitter's
# skip_reason format changes, this list must be updated in lockstep.
@pytest.mark.parametrize(
    "skip_reason, expected_d_entry",
    [
        # energy_drift KE-rest gate (D0-08)
        (
            "KE(0)=1.234e-12 < 1e-10 (rollout starts at "
            "rest; relative drift undefined; see DECISIONS.md D0-08)",
            "D0-08",
        ),
        # energy_drift dissipative-by-design gate (D0-18)
        (
            "system_class='dissipative' (dataset='tgv2d'); "
            "KE(t) monotone-non-increasing across the rollout; "
            "see properties.ke_initial / ke_final for values; "
            "relative drift is a misfire for dissipative-by-design "
            "systems where the dissipation magnitude IS the physics. "
            "See DECISIONS.md D0-18; consult dissipation_sign_violation "
            "for the load-bearing test on this system class.",
            "D0-18",
        ),
        # energy_drift open-driven gate (D0-22 amendment 1)
        (
            "system_class='open-driven-dissipative' (dataset='dam2d'); "
            "KE grows by orders of magnitude due to physics (gravitational PE → "
            "KE conversion); the strictly-dissipative-or-conservative assumption "
            "underpinning energy_drift does not apply. See DECISIONS.md D0-22 "
            "(amendment 1).",
            "D0-22 (amendment 1)",
        ),
        # dissipation_sign_violation KE-rest gate (D0-08)
        (
            "max(KE)=1.234e-12 < 1e-10 (trajectory "
            "has no kinetic energy; dissipation question undefined; "
            "see DECISIONS.md D0-08)",
            "D0-08",
        ),
        # dissipation_sign_violation open-driven gate (D0-22)
        (
            "system_class='open-driven-dissipative' (dataset='dam2d'); "
            "dE/dt > 0 over a stretch by physics (gravitational PE → KE conversion); "
            "the strictly-dissipative-or-conservative assumption underpinning "
            "dissipation_sign_violation does not apply. See DECISIONS.md D0-22.",
            "D0-22",
        ),
    ],
)
def test_renderer_d_entry_regex_extracts_emitter_skip_reasons(
    skip_reason: str, expected_d_entry: str
) -> None:
    """Drift-guard for the emitter↔renderer skip_reason ↔ D-entry contract.

    The renderer's D-entry extraction regex
    (``r"DECISIONS\\.md\\s+(D0-\\d+(?:\\s+\\(amendment\\s+\\d+\\))?)"``)
    must match each of the five emitter skip_reason formats. If a future
    emitter change alters the citation format (e.g., omits the prefix,
    uses lowercase d0, etc.), this test fails before the renderer
    silently degrades cells to "?". The contract is implicit between
    two source files; this test makes it explicit.
    """
    import re

    match = re.search(DENTRY_REGEX, skip_reason)
    assert match is not None, (
        f"Renderer's D-entry regex failed to match skip_reason: {skip_reason!r}. "
        "Either the emitter's citation format changed (update particle_rollout_adapter.py "
        "or render_cross_stack_table.py to match), or this test's fixture is stale."
    )
    assert match.group(1) == expected_d_entry, (
        f"Renderer's D-entry regex extracted {match.group(1)!r}, expected "
        f"{expected_d_entry!r}, from skip_reason: {skip_reason!r}"
    )


def test_renderer_rejects_duplicate_stack_labels(tmp_path: Path) -> None:
    """Two SARIFs with the same (model_name, dataset_name) must raise.

    Codex round-codex-3 finding 2. Pre-fix, the renderer collected
    stack labels via simple append + indexed by (rule_id, stack_label);
    if the SARIF directory contained two files for the same stack
    (e.g., post-re-emission with old SARIFs still present), the
    rendered table would have duplicate columns whose cells were
    overwritten under one key. The renderer must fail fast on this
    rather than producing an ambiguous table silently.
    """
    # Two SARIFs at different shas but with the same model_name +
    # dataset_name (simulating two SEGNN-TGV2D emissions at different
    # sarif_emission shas left in the same directory).
    segnn_v1 = copy.deepcopy(_load(SEGNN_FIXTURE))
    segnn_v2 = copy.deepcopy(_load(SEGNN_FIXTURE))
    # Vary the sarif_emission_sha so the two files aren't byte-identical,
    # but keep model_name + dataset_name the same (the duplicate-label trigger)
    segnn_v2["runs"][0]["properties"]["physics_lint_sha_sarif_emission"] = "FFFFFFFFFF"

    v1_path = tmp_path / "segnn_tgv2d_AAAAAAAAAA.sarif"
    v2_path = tmp_path / "segnn_tgv2d_FFFFFFFFFF.sarif"
    _write(segnn_v1, v1_path)
    _write(segnn_v2, v2_path)

    # The fixtures use synthetic names; the duplicate-label is whatever
    # the fixture's model_name-dataset_name resolves to.
    with pytest.raises(DuplicateStackLabelError, match="Two SARIFs map to the same stack label"):
        render_cross_stack_table([v1_path, v2_path])


def test_renderer_distinct_stack_labels_still_render() -> None:
    """Sanity: distinct (model_name, dataset_name) labels still work post-fix.

    The canonical fixtures have distinct synthetic labels; post-fix,
    they must still render without raising DuplicateStackLabelError.
    """
    # Should NOT raise — both fixtures have distinct synthetic labels.
    table = render_cross_stack_table([SEGNN_FIXTURE, GNS_FIXTURE])
    assert isinstance(table, str) and len(table) > 0


def test_renderer_d_entry_regex_rejects_incidental_d0_mentions() -> None:
    """The regex must NOT match incidental D0-NN mentions without the prefix.

    Pre-fix, a skip_reason mentioning "D0-19 §3.4" (citing the schema
    invariant) could have been picked up by a looser regex. The
    "DECISIONS.md " prefix requirement prevents this — if it drops,
    cell labels could silently cite the wrong D-entry.
    """
    import re

    # Incidental mention without the DECISIONS.md prefix
    assert re.search(DENTRY_REGEX, "this references D0-19 §3.4 but is not a citation") is None
    # Citation through punctuation that breaks the prefix
    assert (
        re.search(DENTRY_REGEX, "see (DECISIONS.md D0-22) — picked up") is not None
    )  # ok, prefix present
    # Pure mention without prefix
    assert re.search(DENTRY_REGEX, "D0-22 is irrelevant here") is None


# ---------------------------------------------------------------------------
# 7. Optional inference_run_status section (rung-4c §9 review-gate fold-in)
# ---------------------------------------------------------------------------


def test_renderer_renders_inference_run_status_when_all_present(tmp_path: Path) -> None:
    """When both stacks carry the optional `inference_run_status` field,
    the renderer adds a dedicated section after the three-sha provenance
    listing each stack's status. Programmatically derived from the
    canonical fixtures (per memory: don't commit a separate fixture).
    """
    segnn_with_status = copy.deepcopy(_load(SEGNN_FIXTURE))
    segnn_with_status["runs"][0]["properties"]["inference_run_status"] = "from_aborted_inference"
    gns_with_status = copy.deepcopy(_load(GNS_FIXTURE))
    gns_with_status["runs"][0]["properties"]["inference_run_status"] = "from_completed_inference"
    segnn_path = tmp_path / "segnn_with_status.sarif"
    gns_path = tmp_path / "gns_with_status.sarif"
    _write(segnn_with_status, segnn_path)
    _write(gns_with_status, gns_path)

    rendered = render_cross_stack_table([segnn_path, gns_path])

    assert "**Inference run status (rung-4c §9 review-gate fold-in):**" in rendered
    # Each stack's per-stack line carries the explicit value, not a defaulted one.
    assert "synthetic_segnn-synthetic_dissipative_d**: from_aborted_inference" in rendered
    assert "synthetic_gns-synthetic_dissipative_d**: from_completed_inference" in rendered
    # Honest-absence marker should NOT appear when both stacks are present.
    assert "n/a (pre-salvage-tag-schema)" not in rendered


def test_renderer_omits_inference_run_status_when_all_absent() -> None:
    """When NO stack carries `inference_run_status`, the renderer omits
    the optional section entirely so legacy SARIFs (rung-4a/4b
    pre-fold-in) render byte-identically to before. Companion to the
    golden-output test, with an explicit named pin on the absence
    behavior.
    """
    rendered = render_cross_stack_table([SEGNN_FIXTURE, GNS_FIXTURE])
    assert "Inference run status" not in rendered
    assert "from_completed_inference" not in rendered
    assert "from_aborted_inference" not in rendered
    assert "n/a (pre-salvage-tag-schema)" not in rendered


def test_renderer_marks_absent_status_as_n_a_in_mixed_set(tmp_path: Path) -> None:
    """When ONE stack carries `inference_run_status` and the other does
    not (mixed set), the renderer renders the section with the absent
    stack explicitly marked `n/a (pre-salvage-tag-schema)` rather than
    defaulting to a clean classification. Parallel to the gate's
    refuse-by-default posture: the absence of evidence is not evidence
    of absence-of-abort.
    """
    segnn_with_status = copy.deepcopy(_load(SEGNN_FIXTURE))
    segnn_with_status["runs"][0]["properties"]["inference_run_status"] = "from_aborted_inference"
    segnn_path = tmp_path / "segnn_with_status.sarif"
    _write(segnn_with_status, segnn_path)
    # GNS fixture as-is — no inference_run_status field.

    rendered = render_cross_stack_table([segnn_path, GNS_FIXTURE])

    assert "**Inference run status (rung-4c §9 review-gate fold-in):**" in rendered
    assert "synthetic_segnn-synthetic_dissipative_d**: from_aborted_inference" in rendered
    assert "synthetic_gns-synthetic_dissipative_d**: n/a (pre-salvage-tag-schema)" in rendered
