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
