"""Tests for the rollout subdir isolation helper (round-codex-4 finding 1).

Pre-fix, the four LB rollout entrypoints created
``/vol/rollouts/lagrangebench/<model>_<dataset>_<git_sha>/`` with
``os.makedirs(..., exist_ok=True)`` and proceeded into inference without
checking whether the directory already contained artifacts from a prior
fire. A retry at the same sha could mix stale and fresh artifacts, and
round-codex-3's basename-binding does NOT catch this because the same
directory has the same basename.

The fix is a shared helper that fails closed on a non-empty rollout
subdir unless the explicit ``clean_existing=True`` flag is passed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from external_validation._rollout_anchors._harness.rollout_dir import (
    RolloutDirNotEmptyError,
    prepare_empty_rollout_subdir,
)

# ---------------------------------------------------------------------------
# Creation cases — happy paths
# ---------------------------------------------------------------------------


def test_creates_subdir_when_absent(tmp_path: Path) -> None:
    """Target dir does not exist → helper creates it."""
    target = tmp_path / "segnn_dam2d_abc123"
    assert not target.exists()

    prepare_empty_rollout_subdir(target)

    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_creates_parent_dirs_when_absent(tmp_path: Path) -> None:
    """Parent directories do not exist → helper creates them too."""
    target = tmp_path / "vol" / "rollouts" / "lagrangebench" / "segnn_dam2d_abc"

    prepare_empty_rollout_subdir(target)

    assert target.is_dir()


def test_noop_when_dir_exists_and_empty(tmp_path: Path) -> None:
    """Pre-existing empty dir → helper returns without modification."""
    target = tmp_path / "subdir"
    target.mkdir()

    prepare_empty_rollout_subdir(target)

    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_accepts_string_path(tmp_path: Path) -> None:
    """Helper accepts string paths, not just Path objects (Modal callers pass strings)."""
    target = tmp_path / "subdir"

    prepare_empty_rollout_subdir(str(target))

    assert target.is_dir()


# ---------------------------------------------------------------------------
# Refuse cases — round-codex-4 finding 1 core contract
# ---------------------------------------------------------------------------


def test_raises_when_dir_exists_and_non_empty_default(tmp_path: Path) -> None:
    """Non-empty dir + default flag → refuse with RolloutDirNotEmptyError.

    This is the core round-codex-4 contract: a stale PKL from a prior
    fire (same git_sha, same rollout_subdir) must not be silently mixed
    with fresh artifacts. The basename-binding (round-codex-3) does NOT
    catch same-dir retry contamination.
    """
    target = tmp_path / "segnn_dam2d_abc"
    target.mkdir()
    (target / "rollout_0.pkl").write_bytes(b"stale fire artifact")

    with pytest.raises(RolloutDirNotEmptyError, match=r"rollout_0\.pkl"):
        prepare_empty_rollout_subdir(target)


def test_raises_with_clean_existing_false_explicit(tmp_path: Path) -> None:
    """Explicit ``clean_existing=False`` behaves the same as the default."""
    target = tmp_path / "subdir"
    target.mkdir()
    (target / "stale.npz").touch()

    with pytest.raises(RolloutDirNotEmptyError):
        prepare_empty_rollout_subdir(target, clean_existing=False)


def test_dotfile_counts_as_non_empty(tmp_path: Path) -> None:
    """A leftover ``_inference_manifest.json`` blocks the fresh fire too.

    The manifest itself is a stale-fire artifact; the safety contract
    must treat it as blocking, not as "manifest present so we're OK".
    """
    target = tmp_path / "subdir"
    target.mkdir()
    (target / "_inference_manifest.json").write_text("{}")

    with pytest.raises(RolloutDirNotEmptyError, match=r"_inference_manifest\.json"):
        prepare_empty_rollout_subdir(target)


def test_subdirectory_counts_as_non_empty(tmp_path: Path) -> None:
    """A leftover subdirectory (e.g., from a prior interrupted fire) blocks too."""
    target = tmp_path / "subdir"
    target.mkdir()
    (target / "nested").mkdir()

    with pytest.raises(RolloutDirNotEmptyError, match="nested"):
        prepare_empty_rollout_subdir(target)


def test_error_message_lists_entries_with_truncation(tmp_path: Path) -> None:
    """Error message lists conflicting entries (truncated at 10) for operator triage."""
    target = tmp_path / "subdir"
    target.mkdir()
    for i in range(15):
        (target / f"rollout_{i:02d}.pkl").touch()

    with pytest.raises(RolloutDirNotEmptyError) as excinfo:
        prepare_empty_rollout_subdir(target)

    msg = str(excinfo.value)
    assert "15 entries" in msg
    assert "+5 more" in msg
    assert "rollout_00.pkl" in msg


def test_raises_when_path_exists_but_is_regular_file(tmp_path: Path) -> None:
    """Path exists but is a regular file, not a directory → refuse."""
    target = tmp_path / "file_not_dir"
    target.write_text("oops")

    with pytest.raises(RolloutDirNotEmptyError, match="not a directory"):
        prepare_empty_rollout_subdir(target)


# ---------------------------------------------------------------------------
# Clean-existing salvage opt-in
# ---------------------------------------------------------------------------


def test_clean_existing_true_wipes_files(tmp_path: Path) -> None:
    """``clean_existing=True`` clears regular files from the dir."""
    target = tmp_path / "subdir"
    target.mkdir()
    (target / "rollout_0.pkl").write_bytes(b"stale")
    (target / "rollout_1.pkl").write_bytes(b"stale")

    prepare_empty_rollout_subdir(target, clean_existing=True)

    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_clean_existing_true_wipes_subdirs(tmp_path: Path) -> None:
    """``clean_existing=True`` recursively removes subdirectories too."""
    target = tmp_path / "subdir"
    target.mkdir()
    nested = target / "nested"
    nested.mkdir()
    (nested / "x.pkl").touch()

    prepare_empty_rollout_subdir(target, clean_existing=True)

    assert target.is_dir()
    assert list(target.iterdir()) == []


def test_clean_existing_preserves_symlink_target(tmp_path: Path) -> None:
    """Symlinks inside the dir are unlinked (the link, not the target).

    A subtle bug would be using ``shutil.rmtree`` on a symlink-to-dir,
    which would follow the link and delete the linked-to directory.
    The helper guards against this by checking ``is_symlink()`` before
    rmtree.
    """
    target = tmp_path / "subdir"
    target.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("preserve me")
    (target / "linked").symlink_to(outside)

    prepare_empty_rollout_subdir(target, clean_existing=True)

    assert list(target.iterdir()) == []
    assert outside.exists()
    assert outside.read_text() == "preserve me"


def test_clean_existing_no_op_when_already_empty(tmp_path: Path) -> None:
    """``clean_existing=True`` on an empty dir is a no-op (does not error)."""
    target = tmp_path / "subdir"
    target.mkdir()

    prepare_empty_rollout_subdir(target, clean_existing=True)

    assert target.is_dir()


def test_clean_existing_creates_when_absent(tmp_path: Path) -> None:
    """``clean_existing=True`` on an absent dir creates it (the flag is orthogonal
    to creation)."""
    target = tmp_path / "absent_subdir"
    assert not target.exists()

    prepare_empty_rollout_subdir(target, clean_existing=True)

    assert target.is_dir()
    assert list(target.iterdir()) == []
