"""Drift-guard: modal_app.py imports cleanly + pins the expected
physicsnemo sha per preflight."""

from __future__ import annotations

import ast
from pathlib import Path

MODAL_APP_PATH = Path(__file__).resolve().parent.parent / "modal_app.py"
EXPECTED_PHYSICSNEMO_SHA = "1ca85d65ac2ce28ea9762910c09a954c08a37140"


def _read_module_string_constant(source_path: Path, name: str) -> str | None:
    """Mirrors 01-lagrangebench/tests/test_modal_app_gpu_class.py pattern."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or target.id != name:
            continue
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def test_modal_app_exists() -> None:
    assert MODAL_APP_PATH.is_file(), f"modal_app.py not found at {MODAL_APP_PATH}"


def test_physicsnemo_sha_pinned() -> None:
    actual = _read_module_string_constant(MODAL_APP_PATH, "PHYSICSNEMO_SHA")
    assert actual == EXPECTED_PHYSICSNEMO_SHA, (
        f"PHYSICSNEMO_SHA = {actual!r} in modal_app.py does not match the "
        f"preflight-pinned sha {EXPECTED_PHYSICSNEMO_SHA!r}. Update the pin "
        f"alongside a DECISIONS.md amendment if the bump is intentional."
    )
