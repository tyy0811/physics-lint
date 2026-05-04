"""Drift-guard test for the GPU-class pre-registered in DECISIONS D0-13.

Mirrors the discipline used in
``test_ke_rest_threshold_matches_pre_registration`` (D0-08) and
``test_mesh_fd_noise_tolerance_matches_pre_registration`` (D0-09):
the constant lives in source code, the pre-registration lives in
DECISIONS.md, and a hard-asserting test pins them together so any
silent drift in either direction fails CI before it ships.

The 01-lagrangebench/ sibling directory has a hyphen and leading
digit, so it is not a valid Python module path and ``modal_app`` is
not importable as a normal module. We AST-parse the file instead of
``import``-ing it; this is robust to the directory naming and avoids
false positives from comment text containing the constant name.
"""

from __future__ import annotations

import ast
from pathlib import Path

D0_13_STAGE_1_GPU_CLASS = "T4"
MODAL_APP_PATH = Path(__file__).resolve().parent.parent / "modal_app.py"


def _read_module_string_constant(source_path: Path, name: str) -> str | None:
    """Return the value of a module-level ``name = "..."`` assignment, or None."""
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


def test_modal_app_gpu_class_matches_d0_13_pre_registration() -> None:
    """The hour-0 / hour-2 micro-gate runs on T4 per D0-13's stage-1 default.

    If a future change wants to move the micro-gate to a different GPU
    class (e.g., L4 for similar price + sm_89), the change must land
    alongside a new DECISIONS sub-entry under D0-13 citing the
    discrepancy — not silently in code.
    """
    assert MODAL_APP_PATH.is_file(), f"modal_app.py not found at {MODAL_APP_PATH}"
    actual = _read_module_string_constant(MODAL_APP_PATH, "MICRO_GATE_GPU_CLASS")
    assert actual == D0_13_STAGE_1_GPU_CLASS, (
        f"MICRO_GATE_GPU_CLASS = {actual!r} in {MODAL_APP_PATH.name} does "
        f"not match D0-13 stage-1 pre-registration "
        f"({D0_13_STAGE_1_GPU_CLASS!r}). Either revert the code change or "
        f"land a new DECISIONS sub-entry refining D0-13."
    )
