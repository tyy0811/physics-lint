"""Generic npz-dir → HarnessResults bridge for harness SARIF emission.

Per DECISIONS.md D0-19, this module is the bridge between per-rollout
defects (computed in particle_rollout_adapter.py) and SARIF result rows
(emitted by sarif_emitter.py). Reads particle_rollout_traj*.npz files
from a directory, invokes the 3 conservation defects on each, builds
HarnessResult rows with the appropriate per-row metadata.

For harness:energy_drift SKIP rows specifically, the per-row varying
KE endpoint values (which D0-19's template-constant skip_reason no
longer interpolates) are recomputed from the rollout and attached to
the HarnessResult's extra_properties as ke_initial / ke_final.

This module knows about the harness defects and the SARIF result
shape; it does NOT know about the case study (model_name, dataset_name,
checkpoint_id, etc. are passed in by the case-study driver). The
case-study driver assembles the run-level properties separately and
calls emit_sarif.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    dissipation_sign_violation,
    energy_drift,
    kinetic_energy_series,
    load_rollout_npz,
    mass_conservation_defect,
)
from external_validation._rollout_anchors._harness.sarif_emitter import HarnessResult


class EmptyNpzDirectoryError(Exception):
    """Raised when lint_npz_dir is invoked on a directory containing no
    particle_rollout_traj*.npz files. Silent empty SARIF is a
    methodology hazard.
    """


# Defect functions in their canonical emission order.
# Order matters for downstream SARIF row ordering (deterministic).
_DEFECTS: tuple[tuple[str, Any], ...] = (
    ("harness:mass_conservation_defect", mass_conservation_defect),
    ("harness:energy_drift", energy_drift),
    ("harness:dissipation_sign_violation", dissipation_sign_violation),
)


def lint_npz_dir(
    npz_dir: Path | str,
    *,
    case_study: str = "",
    dataset: str = "",
    model: str = "",
    ckpt_hash: str = "",
) -> list[HarnessResult]:
    """Read all particle_rollout_traj*.npz files from `npz_dir`, invoke
    the 3 conservation defects on each, build HarnessResult rows.

    Per-row varying ke_initial / ke_final attached to harness:energy_drift
    SKIP rows via extra_properties (D0-19).

    Raises EmptyNpzDirectoryError if no matching files found.
    """
    npz_dir = Path(npz_dir)
    npz_paths = sorted(npz_dir.glob("particle_rollout_traj*.npz"))
    if not npz_paths:
        raise EmptyNpzDirectoryError(
            f"No particle_rollout_traj*.npz files found in {npz_dir}. "
            f"Expected at least one trajectory; run `modal volume get` to populate."
        )

    results: list[HarnessResult] = []
    for traj_index, npz_path in enumerate(npz_paths):
        rollout = load_rollout_npz(npz_path)
        for rule_id, defect_fn in _DEFECTS:
            defect = defect_fn(rollout)
            extra: dict[str, Any] = {
                "traj_index": traj_index,
                "npz_filename": npz_path.name,
            }
            if rule_id == "harness:energy_drift" and defect.value is None:
                # D0-19: per-row varying KE values move to dedicated properties.
                ke_series = kinetic_energy_series(rollout)
                extra["ke_initial"] = float(ke_series[0])
                extra["ke_final"] = float(ke_series[-1])

            if defect.value is None:
                level = "note"
                message = f"SKIP: {defect.skip_reason or '(no reason)'}"
            else:
                level = "note"
                message = f"raw_value={defect.value:.3e}"

            results.append(
                HarnessResult(
                    rule_id=rule_id,
                    level=level,
                    message=message,
                    raw_value=defect.value,
                    case_study=case_study,
                    dataset=dataset,
                    model=model,
                    ckpt_hash=ckpt_hash,
                    extra_properties=extra,
                )
            )
    return results
