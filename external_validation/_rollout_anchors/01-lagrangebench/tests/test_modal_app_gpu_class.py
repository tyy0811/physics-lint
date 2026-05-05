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
D0_13_STAGE_2_GPU_CLASS = "A10G"
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
    """The hour-0 / hour-2 JAX micro-gate runs on T4 per D0-13's stage-1 default.

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


def test_lagrangebench_smoke_gpu_class_matches_d0_13_pre_registration() -> None:
    """The Day-1 §3.2 step-1 LagrangeBench install smoke runs on T4 (D0-13 stage-1).

    D0-13's stage-1 description is "Hour-0 / hour-2 JAX micro-gate"
    with rationale "Smoke test only; cheapest CUDA-JAX path; same
    epistemic content as A100" — the rationale extends naturally to
    the rung-2 LagrangeBench install smoke (also a smoke test, also
    cheapest-CUDA-JAX-path, also same epistemic content as A100). If
    a future change wants to graduate the rung-2 smoke to A10G or
    similar (e.g., because the toy infer needs >16 GB), the change
    must land alongside a DECISIONS sub-entry under D0-13.
    """
    assert MODAL_APP_PATH.is_file(), f"modal_app.py not found at {MODAL_APP_PATH}"
    actual = _read_module_string_constant(MODAL_APP_PATH, "LAGRANGEBENCH_SMOKE_GPU_CLASS")
    assert actual == D0_13_STAGE_1_GPU_CLASS, (
        f"LAGRANGEBENCH_SMOKE_GPU_CLASS = {actual!r} in {MODAL_APP_PATH.name} "
        f"does not match D0-13 stage-1 pre-registration "
        f"({D0_13_STAGE_1_GPU_CLASS!r}). Either revert the code change or "
        f"land a new DECISIONS sub-entry refining D0-13."
    )


def test_rollout_generation_gpu_class_matches_d0_13_pre_registration() -> None:
    """Rung-3 production rollouts run on A10G per D0-13 stage-2.

    D0-13 stage-2 sets A10G as the default for "Day 1 §3.2 step 3
    rollout generation (SEGNN/GNS inference)". The rung-3 production
    rollout function in modal_app.py uses
    ROLLOUT_GENERATION_GPU_CLASS as its gpu= argument; this test
    pins it. If a workload OOMs on A10G the per-D0-13 escalation
    path is to switch *that workload* to A100 with a sub-entry; the
    default value pinned here remains A10G.
    """
    assert MODAL_APP_PATH.is_file(), f"modal_app.py not found at {MODAL_APP_PATH}"
    actual = _read_module_string_constant(MODAL_APP_PATH, "ROLLOUT_GENERATION_GPU_CLASS")
    assert actual == D0_13_STAGE_2_GPU_CLASS, (
        f"ROLLOUT_GENERATION_GPU_CLASS = {actual!r} in {MODAL_APP_PATH.name} "
        f"does not match D0-13 stage-2 pre-registration "
        f"({D0_13_STAGE_2_GPU_CLASS!r}). Either revert the code change or "
        f"land a new DECISIONS sub-entry refining D0-13."
    )
