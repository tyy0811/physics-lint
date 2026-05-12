"""Tests for sarif_emitter.py's result-level `locations` (round-code-1 Path A).

GitHub Code Scanning rejects SARIF results that carry no location
("locationFromSarifResult: expected at least one location") and only
*displays* results whose location is a file path (a physicalLocation with an
artifactLocation.uri); logical-only locations are not enough. physics-lint
harness findings have no source-line location (they describe model behavior
on a rollout, not a code defect at a file:line), so each result's physical
location points at the *committed harness adapter module that implements the
rule* — a real source file the finding causally originates from — with a
placeholder region (line 1). The per-row detail (case_study / model /
dataset / rule_id / transform / traj) is preserved as a logicalLocation, and
a stable per-row fingerprint (hash of that fully-qualified name) keeps the
results distinct in the Security tab even though many share the same
adapter:1 physical location.
"""

from __future__ import annotations

import hashlib
import json

from external_validation._rollout_anchors._harness.sarif_emitter import (
    HarnessResult,
    emit_sarif,
)

_HARNESS_URI_PREFIX = "external_validation/_rollout_anchors/_harness"


def _eps_result() -> HarnessResult:
    """An eps-shaped HarnessResult (rung-4b lint_eps_dir output shape)."""
    return HarnessResult(
        rule_id="PH-SYM-001",
        level="warning",
        message="eps_pos_rms=4.083e-04 (transform=rotation pi)",
        raw_value=4.083e-4,
        case_study="01-lagrangebench",
        dataset="tgv2d",
        model="gns",
        ckpt_hash="sha256:c1df56",
        extra_properties={
            "transform_kind": "rotation",
            "transform_param": "pi",
            "traj_index": 0,
            "eps_t_npz_filename": "eps_PH-SYM-001_rotation_pi_traj00.npz",
            "eps_pos_rms": 4.083e-4,
        },
    )


def _conservation_result() -> HarnessResult:
    """A conservation-shaped HarnessResult (rung-4a/4c lint_npz_dir output shape)."""
    return HarnessResult(
        rule_id="harness:mass_conservation_defect",
        level="note",
        message="raw_value=0.000e+00",
        raw_value=0.0,
        case_study="01-lagrangebench",
        dataset="tgv2d",
        model="segnn",
        ckpt_hash="sha256:c0be98",
        extra_properties={
            "traj_index": 7,
            "npz_filename": "particle_rollout_traj07.npz",
        },
    )


def _bare_result() -> HarnessResult:
    """A HarnessResult with no extra_properties (controlled-fixture shape)."""
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


def _only_location(result_dict: dict) -> dict:
    assert isinstance(result_dict["locations"], list)
    assert len(result_dict["locations"]) == 1
    return result_dict["locations"][0]


def test_result_has_a_physical_location_pointing_at_committed_repo_file() -> None:
    loc = _only_location(_eps_result().to_sarif_result())
    uri = loc["physicalLocation"]["artifactLocation"]["uri"]
    assert uri.startswith(_HARNESS_URI_PREFIX + "/")
    assert uri.endswith(".py")
    # Region is a placeholder — findings are about the whole adapter, not a line.
    region = loc["physicalLocation"]["region"]
    assert region["startLine"] == 1


def test_physical_location_uri_maps_by_rule_family() -> None:
    sym_uri = _only_location(_eps_result().to_sarif_result())["physicalLocation"][
        "artifactLocation"
    ]["uri"]
    cons_uri = _only_location(_conservation_result().to_sarif_result())["physicalLocation"][
        "artifactLocation"
    ]["uri"]
    assert sym_uri == f"{_HARNESS_URI_PREFIX}/symmetry_rollout_adapter.py"
    assert cons_uri == f"{_HARNESS_URI_PREFIX}/particle_rollout_adapter.py"
    # PH-SYM-003 (SO(2) substrate skip) is also a symmetry-adapter rule.
    skip = HarnessResult(
        rule_id="PH-SYM-003",
        level="note",
        message="SKIP: SO(2) is a continuous substrate symmetry",
        raw_value=None,
        case_study="01-lagrangebench",
        dataset="tgv2d",
        model="gns",
        ckpt_hash="sha256:c1df56",
        extra_properties={
            "transform_kind": "skip",
            "transform_param": "so2_continuous",
            "traj_index": 3,
        },
    )
    skip_uri = _only_location(skip.to_sarif_result())["physicalLocation"]["artifactLocation"]["uri"]
    assert skip_uri == f"{_HARNESS_URI_PREFIX}/symmetry_rollout_adapter.py"


def test_unmapped_rule_id_falls_back_to_a_committed_harness_file() -> None:
    weird = HarnessResult(
        rule_id="PH-FUTURE-999",
        level="note",
        message="raw_value=1.0",
        raw_value=1.0,
        case_study="03-future",
        dataset="ds",
        model="m",
        ckpt_hash="h",
    )
    uri = _only_location(weird.to_sarif_result())["physicalLocation"]["artifactLocation"]["uri"]
    assert uri.startswith(_HARNESS_URI_PREFIX + "/")
    assert uri.endswith(".py")


def test_result_keeps_logical_location_with_detailed_fqn() -> None:
    logical = _only_location(_eps_result().to_sarif_result())["logicalLocations"][0]
    fqn = logical["fullyQualifiedName"]
    assert "01-lagrangebench" in fqn
    assert "gns" in fqn
    assert "tgv2d" in fqn
    assert "PH-SYM-001" in fqn
    assert "rotation" in fqn
    assert "pi" in fqn
    assert "traj00" in fqn
    assert logical["name"] == "traj00"
    # conservation row: rule + traj, no transform segment
    cons_logical = _only_location(_conservation_result().to_sarif_result())["logicalLocations"][0]
    assert "harness:mass_conservation_defect" in cons_logical["fullyQualifiedName"]
    assert "traj07" in cons_logical["fullyQualifiedName"]
    assert cons_logical["name"] == "traj07"


def test_bare_result_still_gets_a_nonempty_logical_location() -> None:
    logical = _only_location(_bare_result().to_sarif_result())["logicalLocations"][0]
    assert "harness:mass_conservation_defect" in logical["fullyQualifiedName"]
    assert logical["fullyQualifiedName"] != ""
    assert logical["name"] != ""


def test_partial_fingerprint_is_present_and_derived_from_the_fqn() -> None:
    result = _eps_result().to_sarif_result()
    fqn = _only_location(result)["logicalLocations"][0]["fullyQualifiedName"]
    fps = result["partialFingerprints"]
    assert isinstance(fps, dict) and fps
    key, value = next(iter(fps.items()))
    # Keyed in the conventional <name>/<version> form, valued by an FQN hash.
    assert "/" in key
    assert value == hashlib.sha256(fqn.encode("utf-8")).hexdigest()[:16]


def test_results_sharing_a_physical_location_have_distinct_partial_fingerprints() -> None:
    """Many results point at the same adapter:1 physical location; distinct
    partialFingerprints keep them from collapsing into one Security-tab alert.
    """
    r1 = HarnessResult(
        rule_id="PH-SYM-001",
        level="error",
        message="m",
        raw_value=0.02,
        case_study="01-lagrangebench",
        dataset="tgv2d",
        model="gns",
        ckpt_hash="h",
        extra_properties={"transform_kind": "rotation", "transform_param": "pi", "traj_index": 0},
    )
    r2 = HarnessResult(
        rule_id="PH-SYM-001",
        level="error",
        message="m",
        raw_value=0.02,
        case_study="01-lagrangebench",
        dataset="tgv2d",
        model="gns",
        ckpt_hash="h",
        extra_properties={"transform_kind": "rotation", "transform_param": "pi", "traj_index": 1},
    )
    d1, d2 = r1.to_sarif_result(), r2.to_sarif_result()
    # Same physical location ...
    assert (
        d1["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
        == d2["locations"][0]["physicalLocation"]["artifactLocation"]["uri"]
    )
    # ... but distinct fingerprints (the traj differs).
    assert d1["partialFingerprints"] != d2["partialFingerprints"]


def test_emit_sarif_writes_physical_and_logical_locations_for_every_result(tmp_path) -> None:
    out = tmp_path / "out.sarif"
    emit_sarif([_eps_result(), _conservation_result(), _bare_result()], output_path=out)
    sarif = json.loads(out.read_text())
    results = sarif["runs"][0]["results"]
    assert len(results) == 3
    seen_fingerprints = set()
    for r in results:
        loc = r["locations"][0]
        assert loc["physicalLocation"]["artifactLocation"]["uri"].endswith(".py")
        assert loc["physicalLocation"]["region"]["startLine"] == 1
        assert loc["logicalLocations"][0]["fullyQualifiedName"] != ""
        seen_fingerprints.add(tuple(sorted(r["partialFingerprints"].items())))
    assert len(seen_fingerprints) == 3
