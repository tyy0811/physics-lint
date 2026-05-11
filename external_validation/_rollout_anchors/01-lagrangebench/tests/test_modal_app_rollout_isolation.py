"""AST drift-guard: all four LB rollout entrypoints must call the isolation helper.

Round-codex-4 finding 1 (HIGH): pre-fix, each rollout function created
``/vol/rollouts/lagrangebench/<model>_<dataset>_{git_sha}/`` via
``os.makedirs(..., exist_ok=True)`` and proceeded without checking
whether the directory already contained artifacts from a prior fire.
Round-codex-3's manifest basename-binding does NOT catch same-directory
retry contamination — the basenames match because the directory is the
same.

Fix: each rollout function calls ``_prepare_empty_rollout_subdir(...)``
(thin wrapper delegating to ``_harness/rollout_dir.py``) instead of
``os.makedirs(rollout_subdir, exist_ok=True)``. This drift-guard pins
that contract: every LB rollout function must contain a call to the
helper. If a future refactor accidentally drops the call or reverts to
``os.makedirs(..., exist_ok=True)`` for the rollout subdir, this test
fails before the regression ships.

Mirrors the AST-based drift-guard pattern in
``test_modal_app_gpu_class.py`` (D0-13 GPU-class drift-guard).
"""

from __future__ import annotations

import ast
from pathlib import Path

MODAL_APP_PATH = Path(__file__).resolve().parent.parent / "modal_app.py"
ROLLOUT_FUNCTION_NAMES: tuple[str, ...] = (
    "lagrangebench_rollout_p0_segnn_tgv2d",
    "lagrangebench_rollout_p1_gns_tgv2d",
    "lagrangebench_rollout_p1_segnn_dam2d",
    "lagrangebench_rollout_p1_gns_dam2d",
)
EXPECTED_CALL_NAME = "_prepare_empty_rollout_subdir"


def _find_function_def(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _function_body_call_names(func: ast.FunctionDef) -> set[str]:
    """Collect the names of every Call node in the function body.

    For ``foo(...)`` returns ``"foo"``; for ``obj.method(...)`` returns
    ``"method"``; for ``mod.submod.foo(...)`` returns ``"foo"``. Good
    enough for the drift-guard (we only need to detect the helper call,
    not analyze its attribute path).
    """
    names: set[str] = set()
    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


def test_modal_app_path_exists() -> None:
    """Sanity check: the modal_app.py file resolution works from this test's location."""
    assert MODAL_APP_PATH.is_file(), f"modal_app.py not found at {MODAL_APP_PATH}"


def test_all_rollout_functions_call_isolation_helper() -> None:
    """Every LB rollout entrypoint must call ``_prepare_empty_rollout_subdir``.

    Round-codex-4 contract: pre-fire emptiness check is non-optional for
    any rollout function. A future refactor that drops the call or
    reverts to ``os.makedirs(..., exist_ok=True)`` reopens the same-dir
    retry-contamination fail-open.
    """
    tree = ast.parse(MODAL_APP_PATH.read_text(encoding="utf-8"))

    missing: list[str] = []
    for name in ROLLOUT_FUNCTION_NAMES:
        func = _find_function_def(tree, name)
        assert func is not None, (
            f"Expected to find function def {name!r} in {MODAL_APP_PATH.name}; "
            "if it was renamed, update ROLLOUT_FUNCTION_NAMES in this drift-guard."
        )
        called = _function_body_call_names(func)
        if EXPECTED_CALL_NAME not in called:
            missing.append(name)

    assert not missing, (
        f"Rollout function(s) missing the round-codex-4 isolation helper call: "
        f"{missing}. Each LB rollout entrypoint must call "
        f"{EXPECTED_CALL_NAME!r} before writing inference output to the "
        f"rollout subdir, to close the same-dir retry-contamination "
        f"fail-open (Codex round-codex-4 finding 1)."
    )


def test_rollout_functions_do_not_use_makedirs_for_rollout_subdir() -> None:
    """No rollout function may call ``os.makedirs(rollout_subdir, exist_ok=True)``.

    This is the regression pattern that round-codex-4 closes. The
    drift-guard rejects re-introduction of the pattern by string-
    matching the canonical form in each rollout function's source slice.
    String match is fragile in general but the source pattern here is
    standardized across the four rollout functions and the test's job
    is to catch a copy-paste regression, not a deeply-obscured one.
    """
    source = MODAL_APP_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    source_lines = source.splitlines()

    offenders: list[tuple[str, int]] = []
    for name in ROLLOUT_FUNCTION_NAMES:
        func = _find_function_def(tree, name)
        assert func is not None
        assert func.end_lineno is not None
        body = "\n".join(source_lines[func.lineno - 1 : func.end_lineno])
        if "os.makedirs(rollout_subdir, exist_ok=True)" in body:
            offenders.append((name, func.lineno))

    assert not offenders, (
        f"Round-codex-4 regression: rollout function(s) re-introduced "
        f"`os.makedirs(rollout_subdir, exist_ok=True)` instead of calling "
        f"`{EXPECTED_CALL_NAME}`: {offenders}. The makedirs+exist_ok pattern "
        f"allows stale artifacts from a prior fire (same git_sha) to "
        f"silently mix with fresh artifacts; conversion paths walking "
        f"the rollout subdir can then emit a SARIF/table built on "
        f"mixed-run data."
    )


def test_clean_existing_parameter_present_on_rollout_signatures() -> None:
    """Each LB rollout function must accept a ``clean_existing`` parameter.

    The parameter is what threads the operator's explicit salvage opt-in
    through to ``_prepare_empty_rollout_subdir``. Without it, the helper
    can never be passed ``clean_existing=True`` from a Modal CLI fire,
    and the salvage path is closed by accident.
    """
    tree = ast.parse(MODAL_APP_PATH.read_text(encoding="utf-8"))

    missing: list[str] = []
    for name in ROLLOUT_FUNCTION_NAMES:
        func = _find_function_def(tree, name)
        assert func is not None
        arg_names = [a.arg for a in func.args.args]
        if "clean_existing" not in arg_names:
            missing.append(name)

    assert not missing, (
        f"Rollout function(s) missing the `clean_existing` parameter: "
        f"{missing}. Without it, the round-codex-4 salvage opt-in path "
        f"is unreachable from a Modal CLI fire."
    )
