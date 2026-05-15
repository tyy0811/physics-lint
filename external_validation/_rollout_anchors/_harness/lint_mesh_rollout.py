"""Mesh-side single-rollout lint helper.

Mirrors :func:`_harness.lint_npz_dir` for the mesh case: applies the three
``*_on_mesh`` rule mirrors (mass_conservation_defect_on_mesh,
energy_drift_on_mesh, dissipation_sign_violation_on_mesh) to a single
:class:`MeshRollout` and returns ``list[HarnessResult]`` for
:func:`_harness.sarif_emitter.emit_sarif`.

Phase 2 case study 02 uses one rollout per arm (N=1; rung-4c-discipline:
ship at empirically-feasible N), so this helper takes a single rollout
rather than scanning a directory. Task 5 (GT control) and Task 7 (MGN)
both consume it; the trajectory-context metadata (traj_index, n_timesteps)
flows in via ``extra_properties`` rather than being inferred from a
filename pattern.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from external_validation._rollout_anchors._harness.mesh_rollout_adapter import (
    MeshRollout,
    dissipation_sign_violation_on_mesh,
    energy_drift_on_mesh,
    kinetic_energy_series_on_mesh,
    mass_conservation_defect_on_mesh,
)
from external_validation._rollout_anchors._harness.sarif_emitter import HarnessResult

_MESH_DEFECTS = (
    ("harness:mass_conservation_defect", mass_conservation_defect_on_mesh),
    ("harness:energy_drift", energy_drift_on_mesh),
    ("harness:dissipation_sign_violation", dissipation_sign_violation_on_mesh),
)


def lint_mesh_rollout(
    rollout: MeshRollout,
    *,
    case_study: str,
    dataset: str,
    model: str,
    ckpt_hash: str,
    extra_properties: dict[str, Any] | None = None,
) -> list[HarnessResult]:
    """Invoke the three mesh-side conservation defects on ``rollout``,
    return one :class:`HarnessResult` per rule (3 rows total).

    Mirrors :func:`lint_npz_dir`'s structure (rule_id naming, level/message
    conventions, skip_reason placement). Per-row varying ke_initial /
    ke_final attaches to ``harness:energy_drift`` SKIP rows when the FE-KE
    series is computable; substrate-SKIPed rollouts (e.g.
    open-driven-dissipative dispatch firing before KE integration) emit
    KE as None to surface that no computation happened.
    """
    base_extra = dict(extra_properties or {})
    results: list[HarnessResult] = []
    for rule_id, defect_fn in _MESH_DEFECTS:
        defect = defect_fn(rollout)
        row_extra = dict(base_extra)
        if defect.value is None:
            row_extra["skip_reason"] = defect.skip_reason or "(no reason)"
            if rule_id == "harness:energy_drift":
                # ke_series can be NaN-filled (substrate-class SKIP fires
                # before KE integration; or precondition gate fails) —
                # surface as None rather than NaN strings in SARIF.
                ke_series = kinetic_energy_series_on_mesh(rollout)
                ke_initial = float(ke_series[0])
                ke_final = float(ke_series[-1])
                row_extra["ke_initial"] = None if np.isnan(ke_initial) else ke_initial
                row_extra["ke_final"] = None if np.isnan(ke_final) else ke_final
            level: str = "note"
            message = f"SKIP: {defect.skip_reason or '(no reason)'}"
        else:
            level = "note"
            message = f"raw_value={defect.value:.3e}"

        results.append(
            HarnessResult(
                rule_id=rule_id,
                level=level,  # type: ignore[arg-type]
                message=message,
                raw_value=defect.value,
                case_study=case_study,
                dataset=dataset,
                model=model,
                ckpt_hash=ckpt_hash,
                extra_properties=row_extra,
            )
        )
    return results
