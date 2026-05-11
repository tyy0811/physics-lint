"""Rollout subdirectory isolation helper (rung-4c §9 fold-in round-codex-4).

Pre-fix, the four LagrangeBench rollout entrypoints in ``modal_app.py``
created ``/vol/rollouts/lagrangebench/<model>_<dataset>_<git_sha>/`` via
``os.makedirs(..., exist_ok=True)`` and proceeded into inference without
checking whether the directory already contained artifacts from a prior
fire at the same git_sha. A retry at the same sha (after a timeout,
partial conversion, or n_trajs change) could leave stale PKLs/NPZs in
the directory alongside the fresh fire — and the round-codex-3 manifest
basename-binding does NOT catch this case because the basename matches
its own (same) directory.

Conversion paths that walk the rollout subdir for all artifacts could
then mix stale and fresh trajectories and emit a SARIF/table built on
mixed-run data while every manifest gate passed.

The fix: a shared helper that fails closed if the target rollout subdir
already contains entries, with an explicit ``clean_existing=True``
opt-in for documented salvage cases. The opt-in pattern mirrors v2.1
§2's "explicit named-opt-in for salvage paths" governance rule, parallel
to ``allow_from_aborted_inference`` on the standalone-conversion gate.

Codex adversarial review round-codex-4 finding 1 (HIGH) flagged the
underlying retry-isolation gap; the helper closes it.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


class RolloutDirNotEmptyError(Exception):
    """Raised when a rollout subdir contains artifacts from a prior fire.

    The standalone-conversion gate's basename-binding (round-codex-3)
    does not catch same-directory retry contamination — basenames match
    because the directory is the same. The pre-fire emptiness check
    (this error) closes that path: an operator either repairs the
    directory manually or passes ``clean_existing=True`` to wipe before
    re-firing.
    """


def prepare_empty_rollout_subdir(
    rollout_subdir: str | os.PathLike[str],
    *,
    clean_existing: bool = False,
) -> None:
    """Create or verify a rollout subdir is empty before a fresh inference fire.

    Behavior:
    - If ``rollout_subdir`` does not exist, create it (parents included).
    - If it exists and is empty, return (no-op).
    - If it exists and is non-empty AND ``clean_existing=False``, raise
      ``RolloutDirNotEmptyError`` listing up to 10 of the conflicting
      entries so the operator can see what is in the way.
    - If it exists and is non-empty AND ``clean_existing=True``, remove
      all entries from the directory before returning. The directory
      itself is preserved.

    Hidden files (dotfiles, e.g. ``_inference_manifest.json``) and
    subdirectories count as non-empty content — anything in the
    directory blocks a fresh fire by default. Symlinks inside the
    directory are removed via ``unlink`` (the symlink itself, not the
    target).

    Raises ``RolloutDirNotEmptyError`` if the path exists but is not a
    directory (e.g., a stray regular file at the rollout-subdir path).
    """
    target = Path(rollout_subdir)
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
        return
    if not target.is_dir():
        raise RolloutDirNotEmptyError(
            f"rollout_subdir path {str(target)!r} exists but is not a directory"
        )
    entries = sorted(target.iterdir())
    if not entries:
        return
    if not clean_existing:
        listing = ", ".join(repr(p.name) for p in entries[:10])
        more = f" (+{len(entries) - 10} more)" if len(entries) > 10 else ""
        raise RolloutDirNotEmptyError(
            f"rollout_subdir {str(target)!r} already contains "
            f"{len(entries)} entries: {listing}{more}. "
            f"Pre-fire isolation refuses to write into a non-empty rollout "
            f"directory (round-codex-4 finding 1: same-dir retry "
            f"contamination is not caught by manifest basename-binding). "
            f"Repair the directory or pass clean_existing=True to wipe "
            f"and re-fire."
        )
    for entry in entries:
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
