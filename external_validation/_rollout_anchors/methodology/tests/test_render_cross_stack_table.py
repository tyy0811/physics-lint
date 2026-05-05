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
    MissingRunLevelFieldError,
    ResultRowInvariantError,
    SchemaVersionMismatchError,
    SourceTagMismatchError,
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
