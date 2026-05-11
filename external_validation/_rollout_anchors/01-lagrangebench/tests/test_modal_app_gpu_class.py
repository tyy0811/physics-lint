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


def test_lagrangebench_eps_entrypoints_use_a10g() -> None:
    """T7 eps entrypoints must use ROLLOUT_GENERATION_GPU_CLASS (A10G).

    Drift-guard for the rung-4b T7 ``lagrangebench_eps_p{0,1}_*_tgv2d``
    Modal entrypoints. Per D0-21 item 10 the eps sweep matches rung-4a's
    A10G GPU class so measurement-noise floors stay calibrated; if a
    future change wants to graduate the eps sweep to a different class,
    the change must land alongside a DECISIONS sub-entry under D0-21.
    """
    import re

    assert MODAL_APP_PATH.is_file(), f"modal_app.py not found at {MODAL_APP_PATH}"
    text = MODAL_APP_PATH.read_text(encoding="utf-8")
    for fn_name in ("lagrangebench_eps_p0_segnn_tgv2d", "lagrangebench_eps_p1_gns_tgv2d"):
        pattern = (
            r"@app\.function\([^)]*?gpu=ROLLOUT_GENERATION_GPU_CLASS[^)]*?\)\s*\ndef "
            + re.escape(fn_name)
        )
        match = re.search(pattern, text, flags=re.DOTALL)
        assert match is not None, (
            f"{fn_name}: expected @app.function decorator with gpu=ROLLOUT_GENERATION_GPU_CLASS"
        )


def test_lagrangebench_dam2d_rollout_entrypoints_use_a10g() -> None:
    """Rung-4c P1 dam2d rollout entrypoints must use ROLLOUT_GENERATION_GPU_CLASS (A10G).

    Drift-guard for the rung-4c ``lagrangebench_rollout_p1_{segnn,gns}_dam2d``
    Modal functions. Same A10G discipline as rung-4a P0/P1 TGV2D rollouts
    (D0-13 stage-2). If a future change wants to graduate the dam2d
    rollout to a different GPU class, the change must land alongside a
    DECISIONS sub-entry under D0-13 or D0-22.
    """
    import re

    assert MODAL_APP_PATH.is_file(), f"modal_app.py not found at {MODAL_APP_PATH}"
    text = MODAL_APP_PATH.read_text(encoding="utf-8")
    for fn_name in (
        "lagrangebench_rollout_p1_segnn_dam2d",
        "lagrangebench_rollout_p1_gns_dam2d",
    ):
        pattern = (
            r"@app\.function\([^)]*?gpu=ROLLOUT_GENERATION_GPU_CLASS[^)]*?\)\s*\ndef "
            + re.escape(fn_name)
        )
        match = re.search(pattern, text, flags=re.DOTALL)
        assert match is not None, (
            f"{fn_name}: expected @app.function decorator with gpu=ROLLOUT_GENERATION_GPU_CLASS"
        )


# ---------------------------------------------------------------------------
# Round-codex-2 drift-guards: manifest_required threading on the standalone-
# conversion gate. Without these, a future refactor could silently drop the
# `manifest_required=True` argument from `convert_pkls_p1_segnn_dam2d` and
# re-open the Codex-flagged delete-to-bypass path (see plan v2.1 §3
# round-codex-2 absorption + DECISIONS.md D0-22 amendment 3 if filed).
# ---------------------------------------------------------------------------


def _read_function_def_source(source_path: Path, function_name: str) -> str | None:
    """Return the raw source slice for a module-level function named ``function_name``.

    Returns None if the function is not found. The slice is from the
    ``def`` keyword through the function body's end. Uses AST line
    numbers for precision (handles decorators above the def).
    """
    import ast

    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    lines = source_path.read_text(encoding="utf-8").splitlines()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            # Use end_lineno (inclusive) for the slice
            end = node.end_lineno or len(lines)
            return "\n".join(lines[node.lineno - 1 : end])
    return None


def test_gate_function_has_manifest_required_parameter() -> None:
    """``lagrangebench_convert_pkls_in_volume`` must accept ``manifest_required``.

    Drift-guard for the round-codex-2 fix: the conversion gate's signature
    must include ``manifest_required: bool = False`` so post-fold-in
    entrypoints can opt into refuse-on-missing behavior. Default is False
    for legacy entrypoints (rung-3.5 / rung-4a/4b convert_pkls_p0_*).
    """
    source = _read_function_def_source(MODAL_APP_PATH, "lagrangebench_convert_pkls_in_volume")
    assert source is not None, (
        "lagrangebench_convert_pkls_in_volume function not found in modal_app.py"
    )
    assert "manifest_required: bool = False" in source, (
        "lagrangebench_convert_pkls_in_volume must accept 'manifest_required: bool = False' "
        "(round-codex-2 fix; closes the delete-to-bypass path for post-fold-in stacks)"
    )


def test_convert_pkls_p1_segnn_dam2d_passes_manifest_required_true() -> None:
    """``convert_pkls_p1_segnn_dam2d`` must pass ``manifest_required=True`` to the gate.

    Drift-guard for the round-codex-2 fix: the rung-4c-specific standalone-
    conversion entrypoint is post-fold-in by definition, so missing manifest
    is a failure mode (stale local mirror, backfill not run, manifest
    deleted), not legacy absence. Operator must repair via
    ``backfill_rung4c_inference_manifests``, not delete to bypass.
    """
    source = _read_function_def_source(MODAL_APP_PATH, "convert_pkls_p1_segnn_dam2d")
    assert source is not None, "convert_pkls_p1_segnn_dam2d function not found in modal_app.py"
    assert "manifest_required=True" in source, (
        "convert_pkls_p1_segnn_dam2d must pass 'manifest_required=True' to "
        "lagrangebench_convert_pkls_in_volume.remote(...) (round-codex-2 fix)"
    )


def test_convert_pkls_p0_segnn_tgv2d_does_not_pass_manifest_required() -> None:
    """``convert_pkls_p0_segnn_tgv2d`` (legacy) must NOT set ``manifest_required=True``.

    Pre-rung-4c rollout subdirs predate the manifest convention; setting
    manifest_required=True would break legacy convert_pkls usage (D0-17
    amendment 1 conversion-bug-recovery case). Default False is correct
    for this entrypoint.
    """
    source = _read_function_def_source(MODAL_APP_PATH, "convert_pkls_p0_segnn_tgv2d")
    assert source is not None, "convert_pkls_p0_segnn_tgv2d function not found in modal_app.py"
    assert "manifest_required=True" not in source, (
        "convert_pkls_p0_segnn_tgv2d must NOT set manifest_required=True — "
        "pre-rung-4c stacks predate the manifest convention and rely on "
        "the legacy warn-allow path (manifest_required=False default)"
    )


def test_gate_has_missing_required_manifest_refuse_branch() -> None:
    """The gate must have a branch refusing ``STATUS_FROM_UNKNOWN_INFERENCE`` when required.

    Drift-guard for the round-codex-2 fix: refuse_reason 'missing_required_manifest'
    must appear in the gate logic with returncode=2 + a specific error message
    naming the bypass-prevention rationale. Without this branch, the
    delete-to-bypass path that Codex flagged remains open.
    """
    source = _read_function_def_source(MODAL_APP_PATH, "lagrangebench_convert_pkls_in_volume")
    assert source is not None
    assert 'refuse_reason == "missing_required_manifest"' in source, (
        "Gate must check for refuse_reason == 'missing_required_manifest' "
        "(round-codex-2 fail-open closure)"
    )
    assert "manifest_required=True" in source, (
        "Gate's missing-required error message must reference manifest_required=True "
        "so operators understand the policy"
    )


def test_gate_invalid_error_does_not_advertise_delete_to_bypass() -> None:
    """The manifest_invalid error message must NOT advertise deletion as a fix path.

    Pre-round-codex-2, the manifest_invalid error said 'repair or delete
    the manifest (deletion falls back to the warn-allow from_unknown_inference
    path)' — a documented bypass. Round-codex-2 removed that language and
    added the missing-required gate. Drift-guard: don't let the bypass
    language re-appear.
    """
    source = _read_function_def_source(MODAL_APP_PATH, "lagrangebench_convert_pkls_in_volume")
    assert source is not None
    forbidden = "deletion falls back to the warn-allow"
    assert forbidden not in source, (
        f"Forbidden phrase {forbidden!r} reappeared in gate error message; "
        "this advertises the round-codex-2 bypass and must stay removed"
    )
