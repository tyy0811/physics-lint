"""Tests for the shared inference-manifest helper (rung-4c §9 review-gate fold-in round 3).

Pre-round-3, manifest classification was duplicated between
``modal_app.py::_classify_inference_run_status`` (Modal-side gate) and
``emit_sarif.py::_read_inference_manifest_status`` (local-side SARIF
emission), and both implementations collapsed manifest-corruption
into the legacy-absent ``from_unknown_inference`` status, fail-opening
the gate when a corrupt or stale manifest existed on disk (Codex
adversarial review round 3, finding 1).

Round 3 promotes the helpers to a shared module
(``_harness/inference_manifest.py``) and introduces a fourth status —
``manifest_invalid`` — distinct from legacy absence. The gate refuses
``manifest_invalid`` unconditionally (no override flag is appropriate
for corruption); SARIF emission with ``manifest_required=True`` (rung-
4c dam2d post-fold-in stacks) raises rather than silently omitting the
provenance field.

The promotion also resolves plan v2.1 §2.3's deferred shared-helper
candidacy: pre-round-3, two runtime contexts (Modal-side + local-side)
duplicated the logic; round-3's test code is the third runtime context
that pattern B's "single-instance-vs-multi-instance triggers
generalization" rule names as the promotion trigger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from external_validation._rollout_anchors._harness.inference_manifest import (
    INFERENCE_MANIFEST_FILENAME,
    STATUS_FROM_ABORTED_INFERENCE,
    STATUS_FROM_COMPLETED_INFERENCE,
    STATUS_FROM_UNKNOWN_INFERENCE,
    STATUS_MANIFEST_INVALID,
    ManifestInvalidError,
    classify_inference_run_status,
    gate_verdict_for_status,
    persist_inference_manifest_to_rollout_subdir,
    read_inference_manifest_status,
)


def _write_manifest(subdir: Path, payload: dict) -> Path:
    """Write a manifest file for tests, auto-filling rollout_subdir if absent.

    Round-codex-3 made rollout_subdir a required classification field;
    test fixtures that don't care about rollout_subdir get an auto-filled
    value matching the target directory so they classify as expected.
    Tests that exercise the rollout_subdir validation itself (e.g., the
    basename-mismatch attack scenario) set rollout_subdir explicitly.
    """
    target = subdir / INFERENCE_MANIFEST_FILENAME
    if "rollout_subdir" not in payload:
        payload = {**payload, "rollout_subdir": str(subdir)}
    target.write_text(json.dumps(payload))
    return target


def _write_manifest_raw(subdir: Path, payload: dict) -> Path:
    """Write the payload verbatim (no auto-fill). For tests that need to
    simulate manifests with genuinely-missing required fields, including
    a missing rollout_subdir.
    """
    target = subdir / INFERENCE_MANIFEST_FILENAME
    target.write_text(json.dumps(payload))
    return target


# ---------------------------------------------------------------------------
# classify_inference_run_status — exhaustive truth table
# ---------------------------------------------------------------------------


def test_classify_completed_inference(tmp_path: Path) -> None:
    """Manifest present, parseable, returncode=0, aborted_at_step=None."""
    _write_manifest(tmp_path, {"inference_returncode": 0, "aborted_at_step": None})

    status, persisted = classify_inference_run_status(str(tmp_path))

    assert status == STATUS_FROM_COMPLETED_INFERENCE
    assert persisted is not None
    assert persisted["inference_returncode"] == 0


def test_classify_aborted_inference_nonzero_returncode(tmp_path: Path) -> None:
    """Manifest present with non-zero returncode → from_aborted_inference."""
    _write_manifest(tmp_path, {"inference_returncode": -1, "aborted_at_step": "inference"})

    status, persisted = classify_inference_run_status(str(tmp_path))

    assert status == STATUS_FROM_ABORTED_INFERENCE
    assert persisted["aborted_at_step"] == "inference"


def test_classify_aborted_inference_returncode_zero_but_aborted_step(tmp_path: Path) -> None:
    """returncode=0 but aborted_at_step set → still from_aborted_inference (defense-in-depth)."""
    _write_manifest(tmp_path, {"inference_returncode": 0, "aborted_at_step": "conversion"})

    status, _ = classify_inference_run_status(str(tmp_path))

    assert status == STATUS_FROM_ABORTED_INFERENCE


def test_classify_unknown_inference_when_manifest_absent(tmp_path: Path) -> None:
    """No manifest file → from_unknown_inference (legacy / pre-rung-4c)."""
    assert not (tmp_path / INFERENCE_MANIFEST_FILENAME).exists()

    status, persisted = classify_inference_run_status(str(tmp_path))

    assert status == STATUS_FROM_UNKNOWN_INFERENCE
    assert persisted is None


def test_classify_manifest_invalid_corrupt_json(tmp_path: Path) -> None:
    """Manifest file exists but is not valid JSON → manifest_invalid (Codex round-3 finding 1)."""
    target = tmp_path / INFERENCE_MANIFEST_FILENAME
    target.write_text("{not valid json")

    status, persisted = classify_inference_run_status(str(tmp_path))

    assert status == STATUS_MANIFEST_INVALID, (
        "Pre-round-3 collapsed JSON decode failures into from_unknown_inference, "
        "fail-opening the gate. Round-3 distinguishes corruption from legacy absence."
    )
    assert persisted is not None
    assert "_error" in persisted


def test_classify_manifest_invalid_missing_required_fields(tmp_path: Path) -> None:
    """Manifest parseable but missing classification fields → manifest_invalid."""
    # Has rollout_subdir + git_sha but missing inference_returncode + aborted_at_step
    _write_manifest(tmp_path, {"rollout_subdir": "/some/path", "git_sha": "abc123"})

    status, persisted = classify_inference_run_status(str(tmp_path))

    assert status == STATUS_MANIFEST_INVALID
    assert persisted is not None
    assert "_error" in persisted
    assert "inference_returncode" in persisted["_error"]


def test_classify_manifest_invalid_non_dict_root(tmp_path: Path) -> None:
    """Manifest is valid JSON but not a dict (e.g., a list) → manifest_invalid."""
    target = tmp_path / INFERENCE_MANIFEST_FILENAME
    target.write_text(json.dumps([1, 2, 3]))

    status, _ = classify_inference_run_status(str(tmp_path))

    assert status == STATUS_MANIFEST_INVALID


def test_classify_no_side_effects(tmp_path: Path) -> None:
    """Classification must be a pure read; no mutation of the manifest file or directory."""
    _write_manifest(tmp_path, {"inference_returncode": 0, "aborted_at_step": None})
    target = tmp_path / INFERENCE_MANIFEST_FILENAME
    before_mtime = target.stat().st_mtime
    before_listing = sorted(p.name for p in tmp_path.iterdir())

    classify_inference_run_status(str(tmp_path))

    assert target.stat().st_mtime == before_mtime
    assert sorted(p.name for p in tmp_path.iterdir()) == before_listing


# ---------------------------------------------------------------------------
# read_inference_manifest_status — local-side variant for emit_sarif
# ---------------------------------------------------------------------------


def test_read_completed_returns_status_string(tmp_path: Path) -> None:
    _write_manifest(tmp_path, {"inference_returncode": 0, "aborted_at_step": None})

    assert read_inference_manifest_status(tmp_path) == STATUS_FROM_COMPLETED_INFERENCE


def test_read_aborted_returns_status_string(tmp_path: Path) -> None:
    _write_manifest(tmp_path, {"inference_returncode": -1, "aborted_at_step": "inference"})

    assert read_inference_manifest_status(tmp_path) == STATUS_FROM_ABORTED_INFERENCE


def test_read_missing_not_required_returns_none(tmp_path: Path) -> None:
    """Legacy stack: no manifest, not required → None (omit optional SARIF field)."""
    assert read_inference_manifest_status(tmp_path) is None
    assert read_inference_manifest_status(tmp_path, required=False) is None


def test_read_missing_required_raises(tmp_path: Path) -> None:
    """Post-fold-in stack: no manifest but required → raise (Codex round-3 finding 2)."""
    with pytest.raises(FileNotFoundError, match="manifest"):
        read_inference_manifest_status(tmp_path, required=True)


def test_read_invalid_raises_regardless_of_required(tmp_path: Path) -> None:
    """Corruption is not a legacy-absence case; always raise (both required and not)."""
    target = tmp_path / INFERENCE_MANIFEST_FILENAME
    target.write_text("{not valid json")

    with pytest.raises(ManifestInvalidError, match="invalid"):
        read_inference_manifest_status(tmp_path)
    with pytest.raises(ManifestInvalidError, match="invalid"):
        read_inference_manifest_status(tmp_path, required=True)


def test_read_invalid_missing_required_fields_raises(tmp_path: Path) -> None:
    """Parseable but missing classification fields → raise as invalid (not missing)."""
    _write_manifest(tmp_path, {"git_sha": "abc"})  # missing inference_returncode + aborted_at_step

    with pytest.raises(ManifestInvalidError):
        read_inference_manifest_status(tmp_path)


# ---------------------------------------------------------------------------
# persist_inference_manifest_to_rollout_subdir — atomic write contract
# ---------------------------------------------------------------------------


def test_persist_writes_gated_subset(tmp_path: Path) -> None:
    """Persistence writes only INFERENCE_MANIFEST_GATED_FIELDS + _schema_version, not the full manifest."""
    full_manifest = {
        "inference_returncode": 0,
        "aborted_at_step": None,
        "git_sha": "abc",
        "rollout_subdir": str(tmp_path),
        "stdout_excerpt": "x" * 100_000,  # large field that should NOT be persisted
    }
    target = persist_inference_manifest_to_rollout_subdir(full_manifest, str(tmp_path))

    assert target is not None
    persisted = json.loads(Path(target).read_text())
    assert persisted["inference_returncode"] == 0
    assert "stdout_excerpt" not in persisted
    assert persisted["_schema_version"] == "1"


def test_persist_returns_none_when_subdir_missing(tmp_path: Path) -> None:
    """No-op when target directory does not exist (covers early-abort cases)."""
    fake_subdir = tmp_path / "does_not_exist"
    assert not fake_subdir.exists()

    result = persist_inference_manifest_to_rollout_subdir(
        {"inference_returncode": 0, "aborted_at_step": None}, str(fake_subdir)
    )

    assert result is None


def test_persist_returns_none_when_subdir_is_none(tmp_path: Path) -> None:
    result = persist_inference_manifest_to_rollout_subdir(
        {"inference_returncode": 0, "aborted_at_step": None}, None
    )
    assert result is None


def test_persist_then_classify_roundtrip(tmp_path: Path) -> None:
    """End-to-end: persist a manifest, classify reads it back as the expected status."""
    persist_inference_manifest_to_rollout_subdir(
        {"inference_returncode": 0, "aborted_at_step": None}, str(tmp_path)
    )

    status, persisted = classify_inference_run_status(str(tmp_path))

    assert status == STATUS_FROM_COMPLETED_INFERENCE
    assert persisted["inference_returncode"] == 0


def test_persist_atomic_no_partial_file_on_disk(tmp_path: Path) -> None:
    """After successful persist, no .tmp files remain in the subdir."""
    persist_inference_manifest_to_rollout_subdir(
        {"inference_returncode": 0, "aborted_at_step": None}, str(tmp_path)
    )

    tmp_files = [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert tmp_files == []


# ---------------------------------------------------------------------------
# Status constants — exhaustive set, no overlap
# ---------------------------------------------------------------------------


def test_status_constants_distinct() -> None:
    """The four statuses must be distinct strings (no accidental aliasing)."""
    statuses = {
        STATUS_FROM_COMPLETED_INFERENCE,
        STATUS_FROM_ABORTED_INFERENCE,
        STATUS_FROM_UNKNOWN_INFERENCE,
        STATUS_MANIFEST_INVALID,
    }
    assert len(statuses) == 4


def test_manifest_filename_is_dotfile_convention() -> None:
    """Manifest filename starts with underscore (parallels _metadata.json convention)."""
    assert INFERENCE_MANIFEST_FILENAME.startswith("_")
    assert INFERENCE_MANIFEST_FILENAME.endswith(".json")


# ---------------------------------------------------------------------------
# gate_verdict_for_status — truth table for the conversion gate's allow/refuse
# decision based on classification status + flags. Added at v2.1 round-codex-2
# absorption after Codex review surfaced that "delete the manifest to fall
# back to warn-allow" was a documented bypass for post-fold-in stacks. The
# new manifest_required flag closes that path for stacks where the manifest
# is expected to exist.
# ---------------------------------------------------------------------------


def test_gate_completed_always_allows() -> None:
    """Clean completed inference is allow-by-default regardless of flags."""
    allow, _ = gate_verdict_for_status(
        STATUS_FROM_COMPLETED_INFERENCE,
        allow_from_aborted_inference=False,
        manifest_required=False,
    )
    assert allow is True

    allow, _ = gate_verdict_for_status(
        STATUS_FROM_COMPLETED_INFERENCE,
        allow_from_aborted_inference=True,
        manifest_required=True,
    )
    assert allow is True


def test_gate_aborted_refuses_by_default() -> None:
    """Aborted inference refuses without the override flag."""
    allow, reason = gate_verdict_for_status(
        STATUS_FROM_ABORTED_INFERENCE,
        allow_from_aborted_inference=False,
        manifest_required=False,
    )
    assert allow is False
    assert reason == "aborted_inference"


def test_gate_aborted_allows_with_override() -> None:
    """Aborted inference allows when the explicit opt-in flag is set."""
    allow, _ = gate_verdict_for_status(
        STATUS_FROM_ABORTED_INFERENCE,
        allow_from_aborted_inference=True,
        manifest_required=False,
    )
    assert allow is True


def test_gate_unknown_legacy_allows_when_not_required() -> None:
    """Missing manifest on legacy stack (manifest_required=False) → warn-allow."""
    allow, _ = gate_verdict_for_status(
        STATUS_FROM_UNKNOWN_INFERENCE,
        allow_from_aborted_inference=False,
        manifest_required=False,
    )
    assert allow is True


def test_gate_unknown_required_refuses(tmp_path: Path) -> None:
    """Codex round-codex-2 finding: missing manifest on post-fold-in stack must refuse.

    Before this fix, an operator could bypass `--allow-from-aborted-inference`
    on a timed-out dam2d rollout by deleting the manifest — gate would see
    `from_unknown_inference` and warn-allow. The fix: when manifest_required=True
    (passed by post-fold-in entrypoints like convert_pkls_p1_segnn_dam2d),
    missing-manifest fails closed. No override flag — the operator must
    repair/refetch the manifest, not delete it.
    """
    allow, reason = gate_verdict_for_status(
        STATUS_FROM_UNKNOWN_INFERENCE,
        allow_from_aborted_inference=False,
        manifest_required=True,
    )
    assert allow is False
    assert reason == "missing_required_manifest"

    # The override flag for aborted-inference must NOT also bypass the
    # missing-required check — the two are independent gates.
    allow, reason = gate_verdict_for_status(
        STATUS_FROM_UNKNOWN_INFERENCE,
        allow_from_aborted_inference=True,
        manifest_required=True,
    )
    assert allow is False
    assert reason == "missing_required_manifest"


def test_gate_invalid_always_refuses() -> None:
    """manifest_invalid refuses unconditionally — no override flag, regardless of required."""
    for required in (False, True):
        for allow_aborted in (False, True):
            allow, reason = gate_verdict_for_status(
                STATUS_MANIFEST_INVALID,
                allow_from_aborted_inference=allow_aborted,
                manifest_required=required,
            )
            assert allow is False, (
                f"Invalid manifest must refuse (required={required}, allow_aborted={allow_aborted})"
            )
            assert reason == "manifest_invalid"


# ---------------------------------------------------------------------------
# Round-codex-3 finding 1: classifier must bind manifest to rollout directory.
# Pre-fix, a manifest persisted in one rollout directory could be copied/
# moved to another and would still classify as completed/aborted based only
# on inference_returncode + aborted_at_step. This let a stale clean manifest
# from gns_dam2d/ unlock conversion in segnn_dam2d/ (which had actually
# timed out), bypassing the round-codex-2 gate.
# Fix: require persisted rollout_subdir basename to match the directory
# being classified; otherwise MANIFEST_INVALID.
# ---------------------------------------------------------------------------


def test_classify_invalid_when_persisted_rollout_subdir_basename_mismatch(
    tmp_path: Path,
) -> None:
    """Manifest copied from another rollout subdir → MANIFEST_INVALID (round-codex-3 finding 1)."""
    segnn_dir = tmp_path / "segnn_dam2d_e754a4bc2e"
    gns_dir = tmp_path / "gns_dam2d_e754a4bc2e"
    segnn_dir.mkdir()
    gns_dir.mkdir()
    # Write a clean manifest in segnn_dir, but with rollout_subdir
    # pointing at gns_dir (simulating manifest-copied-from-wrong-dir)
    _write_manifest(
        segnn_dir,
        {
            "inference_returncode": 0,
            "aborted_at_step": None,
            "rollout_subdir": str(gns_dir),
        },
    )

    status, persisted = classify_inference_run_status(str(segnn_dir))

    assert status == STATUS_MANIFEST_INVALID, (
        "Manifest copied from a different rollout subdir must classify as "
        "MANIFEST_INVALID; pre-fix this attack bypassed the round-codex-2 gate."
    )
    assert "rollout_subdir" in persisted["_error"]


def test_classify_invalid_when_persisted_rollout_subdir_is_none(tmp_path: Path) -> None:
    """Manifest missing the rollout_subdir field → MANIFEST_INVALID.

    Uses ``_write_manifest_raw`` to bypass the test helper's auto-fill so
    we genuinely write a payload without rollout_subdir; the classifier
    must reject this because rollout_subdir is now a required field.
    """
    _write_manifest_raw(
        tmp_path,
        {"inference_returncode": 0, "aborted_at_step": None},  # no rollout_subdir
    )

    status, persisted = classify_inference_run_status(str(tmp_path))

    assert status == STATUS_MANIFEST_INVALID
    assert "rollout_subdir" in persisted["_error"]


def test_classify_completed_when_basename_matches(tmp_path: Path) -> None:
    """Positive case: persisted rollout_subdir basename matches classified path basename."""
    local_subdir = tmp_path / "segnn_dam2d_e754a4bc2e"
    local_subdir.mkdir()
    modal_path = "/vol/rollouts/lagrangebench/segnn_dam2d_e754a4bc2e"
    _write_manifest(
        local_subdir,
        {
            "inference_returncode": 0,
            "aborted_at_step": None,
            "rollout_subdir": modal_path,
        },
    )

    status, _ = classify_inference_run_status(str(local_subdir))

    assert status == STATUS_FROM_COMPLETED_INFERENCE


def test_classify_aborted_when_basename_matches(tmp_path: Path) -> None:
    """Aborted-inference classification still works when basenames match."""
    local_subdir = tmp_path / "segnn_dam2d_e754a4bc2e"
    local_subdir.mkdir()
    _write_manifest(
        local_subdir,
        {
            "inference_returncode": -1,
            "aborted_at_step": "inference",
            "rollout_subdir": "/vol/rollouts/lagrangebench/segnn_dam2d_e754a4bc2e",
        },
    )

    status, _ = classify_inference_run_status(str(local_subdir))

    assert status == STATUS_FROM_ABORTED_INFERENCE


def test_classify_basename_normalization_handles_trailing_slashes(tmp_path: Path) -> None:
    """Trailing slashes in either path should not break the basename comparison."""
    local_subdir = tmp_path / "segnn_dam2d_X"
    local_subdir.mkdir()
    _write_manifest(
        local_subdir,
        {
            "inference_returncode": 0,
            "aborted_at_step": None,
            "rollout_subdir": "/vol/rollouts/lagrangebench/segnn_dam2d_X/",
        },
    )

    status, _ = classify_inference_run_status(str(local_subdir) + "/")

    assert status == STATUS_FROM_COMPLETED_INFERENCE


def test_persist_auto_fills_rollout_subdir(tmp_path: Path) -> None:
    """persist must auto-fill rollout_subdir from the destination path if missing.

    Post-round-codex-3, rollout_subdir is a required classification field;
    auto-fill keeps minimal-manifest callers (e.g., tests) working without
    re-introducing manual coordination between manifest dict and dest.
    """
    persist_inference_manifest_to_rollout_subdir(
        {"inference_returncode": 0, "aborted_at_step": None}, str(tmp_path)
    )

    target = tmp_path / INFERENCE_MANIFEST_FILENAME
    persisted = json.loads(target.read_text())
    assert persisted["rollout_subdir"] is not None
    assert Path(persisted["rollout_subdir"]).name == tmp_path.name


def test_persist_does_not_overwrite_explicit_rollout_subdir(tmp_path: Path) -> None:
    """If caller passes rollout_subdir, persist keeps it (no silent overwrite)."""
    explicit_path = "/vol/rollouts/lagrangebench/foo_bar"
    persist_inference_manifest_to_rollout_subdir(
        {
            "inference_returncode": 0,
            "aborted_at_step": None,
            "rollout_subdir": explicit_path,
        },
        str(tmp_path),
    )

    target = tmp_path / INFERENCE_MANIFEST_FILENAME
    persisted = json.loads(target.read_text())
    assert persisted["rollout_subdir"] == explicit_path
