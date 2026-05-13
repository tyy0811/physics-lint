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

import ast
import os
import pathlib
import re
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


# ---------------------------------------------------------------------------
# round-codex-Phase2 Finding 3 absorption: per-fire persistent path
# uniqueness via the (rollout_key, run_id) tuple.
#
# Verifies the path-construction invariants without importing modal_app
# (which would require the Modal SDK on every test environment). Uses
# AST parsing per the test_modal_app_imports.py pattern to bind on the
# specific f-string format the production code emits, then asserts the
# uniqueness invariants by hand-applying that format.
# ---------------------------------------------------------------------------

_MODAL_APP_PATH = pathlib.Path(__file__).resolve().parent.parent / "modal_app.py"


def _extract_inference_subdir_format() -> str:
    """Parse modal_app.py and pull out the f-string body of
    ``_make_rollout_inference_subdir``'s return statement. The body
    must be a single f-string (one Joined-Str AST node) so the test
    can rebuild it without importing modal.
    """
    tree = ast.parse(_MODAL_APP_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name != "_make_rollout_inference_subdir":
            continue
        # Find the single Return statement; its value must be a JoinedStr
        # (f-string) with the literal+expression-name shape we assert.
        returns = [n for n in node.body if isinstance(n, ast.Return)]
        assert len(returns) == 1, "_make_rollout_inference_subdir must have exactly one Return"
        joined = returns[0].value
        assert isinstance(joined, ast.JoinedStr), (
            "_make_rollout_inference_subdir return must be a single f-string"
        )
        return ast.unparse(joined)
    raise AssertionError("_make_rollout_inference_subdir not found in modal_app.py")


def test_make_rollout_inference_subdir_format_matches_expected_pattern() -> None:
    """Bind the production f-string to the expected format. Catches
    silent drift (e.g., if someone reorders rollout_key and run_id, or
    changes the volume mount prefix).
    """
    unparsed = _extract_inference_subdir_format()
    # The exact f-string canonical form Python's ast.unparse emits.
    expected = "f'/vol/rollouts/physicsnemo/cs02_mgn_inference_{rollout_key}_{run_id}'"
    assert unparsed == expected, (
        f"_make_rollout_inference_subdir format drifted; got {unparsed!r}, "
        f"expected {expected!r}. Update the test if the change is intentional."
    )


def test_inference_subdir_uniqueness_under_same_rollout_key() -> None:
    """Finding 3: two fires with the SAME ``rollout_key`` but different
    ``run_id`` values produce distinct persistent Volume subdirs. This
    is the absorption fix for the prior path scheme
    ``vortex_shedding_<sha>`` which collided on same-sha re-fires.
    Verifies the invariant by hand-applying the format string.
    """
    rollout_key = "4173b32"
    subdir_a = f"/vol/rollouts/physicsnemo/cs02_mgn_inference_{rollout_key}_20260513T154500Z"
    subdir_b = f"/vol/rollouts/physicsnemo/cs02_mgn_inference_{rollout_key}_20260513T154501Z"
    assert subdir_a != subdir_b, (
        f"two same-rollout_key fires with distinct run_ids must produce "
        f"distinct Volume subdirs; got both = {subdir_a}"
    )
    assert subdir_a.startswith(f"/vol/rollouts/physicsnemo/cs02_mgn_inference_{rollout_key}_")


def test_inference_subdir_uniqueness_under_same_run_id() -> None:
    """Finding 3 contrapositive: two fires with the SAME ``run_id`` but
    different ``rollout_key`` values produce distinct subdirs. Defense-
    in-depth — the path-construction must not lose information from
    either component.
    """
    run_id = "20260513T154500Z"
    subdir_a = f"/vol/rollouts/physicsnemo/cs02_mgn_inference_4173b32_{run_id}"
    subdir_b = f"/vol/rollouts/physicsnemo/cs02_mgn_inference_deadbee_{run_id}"
    assert subdir_a != subdir_b


def test_inference_subdir_format_has_no_legacy_vortex_shedding_path_segment() -> None:
    """Finding 3 regression guard: the new path scheme replaces the
    legacy ``vortex_shedding_<sha>`` segment. If a refactor accidentally
    restores the old prefix, the same-sha collision risk returns. The
    production f-string must not contain that segment.
    """
    unparsed = _extract_inference_subdir_format()
    assert "vortex_shedding_" not in unparsed, (
        "_make_rollout_inference_subdir must NOT reuse the legacy "
        "'vortex_shedding_<sha>' segment; that scheme collided on "
        "same-sha re-fires (round-codex-Phase2 Finding 3)."
    )
    # The new segment must be present.
    assert re.search(r"cs02_mgn_inference_", unparsed), (
        "_make_rollout_inference_subdir must use the new "
        "'cs02_mgn_inference_<rollout_key>_<run_id>' scheme."
    )
