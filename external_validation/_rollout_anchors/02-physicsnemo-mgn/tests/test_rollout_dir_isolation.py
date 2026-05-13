"""Phase 2 Task 7 step 4 — F5 rollout-dir-isolation smoke test.

Phase-1 cross-review Finding 5 absorption: verify that two same-sha
retries of the MGN rollout entrypoint cannot read each other's
CWD-relative stats. ``vortex_shedding_dataset.py`` (lines 103, 141
@ ``1ca85d65``) reads ``edge_stats.json`` and ``node_stats.json`` via
the bare filename, which resolves relative to the current working
directory. If two retries with the same ``git_sha`` prefix shared a
working directory, the second one could see the first one's stale
stats files (or partial writes mid-stage) and silently miscompute.

The production-path absorption is Task 6's
``mgn_rollout_p0_vortex_shedding`` (uses ``tempfile.mkdtemp`` to
create a uniquely-named directory per fire). This test exercises
the same Python-level invariant locally: two ``mkdtemp`` calls with
identical prefixes return distinct paths, and a process chdir'd into
one cannot resolve a sibling's CWD-relative file.

CPU-only; no Modal dependency; runs in standard pytest.
"""

from __future__ import annotations

import os
import tempfile


def test_two_same_sha_retries_get_distinct_rollout_dirs(tmp_path) -> None:
    """``tempfile.mkdtemp`` returns distinct paths even when called with
    the same prefix, so two same-sha retries of the MGN inference
    entrypoint receive isolated working directories. This is the Python-
    level invariant Task 6's rollout-dir isolation depends on.
    """
    prefix = "mgn_rollout_p0_abc123_"  # mirrors Task 6's prefix shape
    dir_a = tempfile.mkdtemp(prefix=prefix, dir=str(tmp_path))
    dir_b = tempfile.mkdtemp(prefix=prefix, dir=str(tmp_path))
    assert dir_a != dir_b, "tempfile.mkdtemp must produce distinct paths even with same prefix"
    assert os.path.isdir(dir_a) and os.path.isdir(dir_b)


def test_chdir_into_retry_dir_cannot_read_sibling_cwd_relative_stats(tmp_path) -> None:
    """A process chdir'd into dir_b cannot read dir_a's stats via the
    bare filename (the CWD-relative access pattern that
    ``vortex_shedding_dataset.py`` uses at sha ``1ca85d65``). This
    asserts the F5 invariant directly: even with same-prefix tempdirs,
    a retry cannot contaminate another's stats read.
    """
    prefix = "mgn_rollout_p0_abc123_"
    dir_a = tempfile.mkdtemp(prefix=prefix, dir=str(tmp_path))
    dir_b = tempfile.mkdtemp(prefix=prefix, dir=str(tmp_path))

    # Stage a stats file in dir_a only.
    with open(os.path.join(dir_a, "edge_stats.json"), "w") as f:
        f.write('{"edge_mean": [0, 0, 0]}')

    # chdir to dir_b and verify the bare filename does NOT resolve to dir_a's file.
    old_cwd = os.getcwd()
    os.chdir(dir_b)
    try:
        assert not os.path.isfile("edge_stats.json"), (
            "F5 violation: dir_b can read dir_a's CWD-relative stats; "
            "rollout-dir isolation is broken on this platform."
        )
        # Also negative: opening the bare filename must raise FileNotFoundError.
        try:
            with open("edge_stats.json"):
                pass
        except FileNotFoundError:
            pass  # expected
        else:
            raise AssertionError("F5 violation: bare-filename open in dir_b succeeded")
    finally:
        os.chdir(old_cwd)


def test_distinct_rollout_dirs_when_dir_arg_is_tmp_default(tmp_path) -> None:
    """Sanity: the same invariant holds when ``dir=None`` (the default
    ``tempfile.mkdtemp`` placement under ``$TMPDIR`` / ``/tmp``). Task
    6's production path uses the default dir (so the rollout_dir lands
    under ``/tmp`` inside the Modal container, not the persistent
    Volume).
    """
    prefix = "mgn_rollout_p0_def456_"
    dir_a = tempfile.mkdtemp(prefix=prefix)
    dir_b = tempfile.mkdtemp(prefix=prefix)
    try:
        assert dir_a != dir_b
        assert os.path.isdir(dir_a) and os.path.isdir(dir_b)
    finally:
        os.rmdir(dir_a)
        os.rmdir(dir_b)
