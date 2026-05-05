"""Spot-check a single trajectory from a Modal-Volume rollout subdir.

Pulls ``particle_rollout_traj00.npz`` from
``rollout-anchors-artifacts:/rollouts/lagrangebench/<subdir>/`` to a
local cache, loads it via ``particle_rollout_adapter.load_rollout_npz``,
and reports the load-bearing surface (velocity range, PBC audit fields,
KE series, harness defects). The script is the codified version of the
cross-repo contract verification that surfaced the periodic-wraparound
bug (D0-17) and the energy_drift-on-dissipative-systems methodology
gap (D0-18) during the rung-3.5 work.

Run from the physics-lint repo root::

    .venv/bin/python external_validation/_rollout_anchors/01-lagrangebench/scripts/spot_check_rollout.py <subdir_name>

E.g.::

    .venv/bin/python ...scripts/spot_check_rollout.py segnn_tgv2d_8c3d080397
    .venv/bin/python ...scripts/spot_check_rollout.py gns_tgv2d_f48dd3f376

Cache lives at ``/tmp/physics_lint_spot_check/``; existing pulls are
re-used (idempotent — re-running on the same subdir doesn't re-download).

## Reference values (TGV2D specifically)

Pre-D0-17 (f75e22d8dd run, captured in DECISIONS D0-16):
    v range:                    [-24.8, +24.9]   <- spurious wraparound
    v std:                      1.35
    energy_drift:               0.99998 (artifact)
    dissipation_sign_violation: 0.548   (artifact)
    mass_conservation_defect:   0.0     (correct, unaffected by bug)

Post-D0-17 + amendment 1 + D0-18 (8c3d080397 / f48dd3f376):
    v range:                    ~ [-1.0, +1.0]   <- periodic-corrected
    v std:                      ~ 0.17
    energy_drift:               SKIP (D0-18; dissipative-by-design)
    dissipation_sign_violation: 0.0     (model never spuriously gains KE)
    mass_conservation_defect:   0.0     (unchanged, bug-asymmetric)

## v1 scope

Hardcoded assumptions for TGV2D: 2D periodic domain, dissipative,
expects PBC source to be ``"truncated_from_oversize"`` (per LB upstream
convention surfaced in D0-17 amendment 1). Generalize to other LB
datasets (DAM2D wall-bounded, etc.) when those rollouts land. The
script's load-bearing verdict logic (velocity range bounded, PBC
threaded through, harness defects coherent) is dataset-agnostic; only
the printed reference-value comparisons assume TGV2D.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

# Repo root = parents[4] from this script's location:
#   scripts/ -> 01-lagrangebench/ -> _rollout_anchors/ -> external_validation/ -> physics-lint/
_REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO_ROOT))

from external_validation._rollout_anchors._harness.particle_rollout_adapter import (  # noqa: E402
    dissipation_sign_violation,
    energy_drift,
    kinetic_energy_series,
    load_rollout_npz,
    mass_conservation_defect,
)

CACHE_DIR = Path("/tmp/physics_lint_spot_check")
VOLUME_NAME = "rollout-anchors-artifacts"


def _pull_traj00(subdir_name: str) -> Path:
    """Pull traj00.npz from the Modal Volume to local cache; idempotent."""
    CACHE_DIR.mkdir(exist_ok=True)
    target = CACHE_DIR / f"{subdir_name}_traj00.npz"
    if target.exists():
        return target
    print(f"Pulling {subdir_name}/particle_rollout_traj00.npz from Volume...")
    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            VOLUME_NAME,
            f"/rollouts/lagrangebench/{subdir_name}/particle_rollout_traj00.npz",
            str(target),
        ],
        check=True,
    )
    return target


def _fmt_defect(defect) -> str:
    """Render a HarnessDefect as either a numeric string or 'SKIP: <reason>'.

    D0-18 added a skip-with-reason path on energy_drift for
    dissipative-by-design systems (TGV2D, RPF2D, ...) — defect.value
    becomes None on TGV2D rollouts since the harness is now D0-18-aware.
    """
    if defect.value is None:
        return f"SKIP ({defect.skip_reason[:60]}...)"
    return f"{defect.value:.4e}"


def main(subdir_name: str) -> int:
    target = _pull_traj00(subdir_name)
    print(f"\n=== Spot-check: {subdir_name} traj00 ===\n")
    r = load_rollout_npz(target)

    v_min, v_max = float(r.velocities.min()), float(r.velocities.max())
    print(f"  velocities range: [{v_min:.4f}, {v_max:.4f}]")
    print(f"  velocities std:   {r.velocities.std():.4f}")
    print(f"  velocities |max|: {np.abs(r.velocities).max():.4f}")
    print()

    pbc = r.metadata.get("periodic_boundary_conditions")
    pbc_upstream = r.metadata.get("periodic_boundary_conditions_upstream")
    pbc_source = r.metadata.get("periodic_boundary_conditions_source")
    print(f"  metadata.periodic_boundary_conditions:           {pbc}")
    print(f"  metadata.periodic_boundary_conditions_upstream:  {pbc_upstream}")
    print(f"  metadata.periodic_boundary_conditions_source:    {pbc_source!r}")
    print()

    m_def = mass_conservation_defect(r)
    e_def = energy_drift(r)
    d_def = dissipation_sign_violation(r)
    print(f"  mass_conservation_defect:    value={m_def.value}  skip={m_def.skip_reason}")
    print(f"  energy_drift:                value={e_def.value}  skip={e_def.skip_reason}")
    print(f"  dissipation_sign_violation:  value={d_def.value}  skip={d_def.skip_reason}")
    print()

    ke = kinetic_energy_series(r)
    print(f"  KE(0):     {ke[0]:.6e}")
    print(f"  KE(50):    {ke[min(50, len(ke) - 1)]:.6e}")
    print(f"  KE(end):   {ke[-1]:.6e}")
    print(f"  KE max:    {ke.max():.6e}")
    print()

    print("=== Verdict ===")
    bug_signature = abs(v_min) > 5 or abs(v_max) > 5
    if bug_signature:
        print("  -> FAIL: velocities still spurious (|v|>5). D0-17 fix not applied.")
        return 1
    if pbc is None:
        print("  -> FAIL: periodic_boundary_conditions metadata missing.")
        return 1
    print("  -> PASS: velocities physical, metadata threaded through.")
    print(f"     energy_drift:               {_fmt_defect(e_def)}")
    print(f"     dissipation_sign_violation: {_fmt_defect(d_def)}")
    print(f"     mass_conservation_defect:   {_fmt_defect(m_def)}")
    print("     (pre-D0-17 artifacts: energy_drift=0.99998, dissipation_sign_violation=0.548;")
    print("      pre-D0-18 the harness fired energy_drift raw on dissipative systems;")
    print("      now SKIP'd per D0-18 with reason pointing at dissipation_sign_violation.)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} <subdir_name>")
        print("e.g.:  spot_check_rollout.py segnn_tgv2d_8c3d080397")
        print("       spot_check_rollout.py gns_tgv2d_f48dd3f376")
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
