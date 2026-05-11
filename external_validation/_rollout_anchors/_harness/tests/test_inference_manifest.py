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
    persist_inference_manifest_to_rollout_subdir,
    read_inference_manifest_status,
)


def _write_manifest(subdir: Path, payload: dict) -> Path:
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
