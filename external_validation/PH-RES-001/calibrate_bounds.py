"""Two-configuration calibration of Layer 2 bounds. Not a pytest file.

Characterizes PH-RES-001 on both its code paths (Rev 1.7.2, Path A'):

- **periodic+spectral** emits H^-1. BDO norm-equivalence claim holds; the
  three-perturbation family should give `C_max/c_min < 10`.
- **non-periodic+FD** emits L^2 as a variational fallback (per the rule's
  module docstring: h_minus_one_spectral drops the DC mode, so hD grids
  fall back to L^2 to preserve every-mode residual detection). L^2 is NOT
  H^1-equivalent across frequencies; `rho_k = ||r||_{L^2} / ||p_k||_{H^1}`
  scales linearly with perturbation wavenumber k, by construction.

Writes `fixtures/norm_equivalence_bounds.json` with two top-level sections
(`periodic_spectral` and `nonperiodic_fd`). Re-run only when the rule
implementation or perturbation family changes; re-runs are explicit
commits that update CITATION.md alongside the JSON.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np

from external_validation._harness.mms import mms_perturbation_h1_error
from physics_lint import DomainSpec, GridField
from physics_lint.analytical.poisson import periodic_sin_sin, sin_sin_mms_square
from physics_lint.rules import ph_res_001

N = 64


def _nonperiodic_spec_with_source(source_array: np.ndarray) -> DomainSpec:
    spec = DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [N, N],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    object.__setattr__(spec, "_source_array", source_array)
    return spec


def _periodic_spec_with_source(source_array: np.ndarray) -> DomainSpec:
    two_pi = 2 * math.pi
    spec = DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [N, N],
            "domain": {"x": [0.0, two_pi], "y": [0.0, two_pi]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )
    object.__setattr__(spec, "_source_array", source_array)
    return spec


# Non-periodic perturbation family (sin(k*pi*x)sin(k*pi*y) on [0,1]^2).
def _np_p1(x, y):
    return 0.01 * np.sin(math.pi * x) * np.sin(math.pi * y)


def _np_p2(x, y):
    return 0.01 * np.sin(4 * math.pi * x) * np.sin(4 * math.pi * y)


def _np_p3(x, y):
    return 0.01 * np.sin(math.pi * x) * np.sin(math.pi * y) * x * (1 - x) * y * (1 - y)


# Periodic perturbation family (sin(k*x)sin(k*y) on [0, 2pi]^2).
def _per_p1(x, y):
    return 0.01 * np.sin(x) * np.sin(y)


def _per_p2(x, y):
    return 0.01 * np.sin(2 * x) * np.sin(2 * y)


def _per_p3(x, y):
    return 0.01 * np.sin(3 * x) * np.sin(3 * y)


def _calibrate_nonperiodic() -> dict:
    sol = sin_sin_mms_square()
    h = 1.0 / (N - 1)
    xs = np.linspace(0.0, 1.0, N)
    mesh_x, mesh_y = np.meshgrid(xs, xs, indexing="ij")
    u_exact = sol.u(mesh_x, mesh_y)
    source = sol.source(mesh_x, mesh_y)
    spec = _nonperiodic_spec_with_source(source)

    rhos = []
    last_norm = None
    print("  non-periodic + FD (rule emits L^2 fallback):")
    for label, pert in (
        ("k=1 low-freq", _np_p1),
        ("k=4 mid-freq", _np_p2),
        ("k=1 bdry-resp", _np_p3),
    ):
        u_pert = u_exact + pert(mesh_x, mesh_y)
        field = GridField(u_pert, h=(h, h), periodic=False)
        result = ph_res_001.check(field, spec)
        assert result.status != "SKIPPED", (
            f"nonperiodic+FD {label}: SKIPPED; reason={result.reason!r}"
        )
        r_norm = float(result.raw_value or 0.0)
        h1 = mms_perturbation_h1_error(mesh_x, mesh_y, perturbation=pert, periodic=False)
        rho = r_norm / h1
        print(f"    {label:18s} r_norm={r_norm:.3e}  h1={h1:.3e}  rho={rho:.3e}")
        rhos.append(rho)
        last_norm = result.recommended_norm

    return {
        "rho_values": rhos,
        "rho_labels": ["k=1 low-freq", "k=4 mid-freq", "k=1 bdry-resp"],
        "norm_emitted_by_rule": last_norm,
        "note": (
            "L^2 fallback per ph_res_001.py:123-131; ratio ||r||_{L2} / ||.||_{H1} "
            "scales linearly with perturbation wavenumber k because "
            "||Laplacian f||_{L2} / ||f||_{H1} is proportional to k for "
            "f = sin(k pi x) sin(k pi y). This is a characterization of the "
            "rule's L^2 fallback, not a test of norm-equivalence."
        ),
        "C_max_over_c_min_raw": max(rhos) / min(rhos),
    }


def _calibrate_periodic() -> dict:
    sol = periodic_sin_sin()
    two_pi = 2 * math.pi
    h = two_pi / N  # periodic: endpoint-excluded
    xs = np.linspace(0.0, two_pi, N, endpoint=False)
    mesh_x, mesh_y = np.meshgrid(xs, xs, indexing="ij")
    u_exact = sol.u(mesh_x, mesh_y)
    source = sol.source(mesh_x, mesh_y)
    spec = _periodic_spec_with_source(source)

    rhos = []
    last_norm = None
    print("  periodic + spectral (rule emits H^-1, BDO norm-equivalence):")
    for label, pert in (("k=1", _per_p1), ("k=2", _per_p2), ("k=3", _per_p3)):
        u_pert = u_exact + pert(mesh_x, mesh_y)
        field = GridField(u_pert, h=(h, h), periodic=True, backend="spectral")
        result = ph_res_001.check(field, spec)
        assert result.status != "SKIPPED", (
            f"periodic+spectral {label}: SKIPPED; reason={result.reason!r}"
        )
        r_norm = float(result.raw_value or 0.0)
        h1 = mms_perturbation_h1_error(mesh_x, mesh_y, perturbation=pert, periodic=True)
        rho = r_norm / h1
        print(f"    {label:18s} r_norm={r_norm:.3e}  h1={h1:.3e}  rho={rho:.3e}")
        rhos.append(rho)
        last_norm = result.recommended_norm

    c_min = min(rhos) * 0.5
    c_max = max(rhos) * 2.0
    return {
        "c_min": c_min,
        "C_max": c_max,
        "rho_values": rhos,
        "rho_labels": ["k=1", "k=2", "k=3"],
        "norm_emitted_by_rule": last_norm,
        "C_max_over_c_min": c_max / c_min,
        "note": (
            "H^-1 residual norm (rule's variationally-correct path for "
            "periodic+spectral poisson). BDO norm-equivalence: "
            "||r||_{H-1} is proportional to ||u_pert - u_exact||_{H1} within a "
            "bounded ratio (C_max/c_min < 10) across the three-mode family."
        ),
    }


def main() -> None:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    print(f"Calibrating PH-RES-001 Layer 2 bounds at physics_lint {sha[:8]}")
    print()
    nonperiodic = _calibrate_nonperiodic()
    print()
    periodic = _calibrate_periodic()
    print()

    out = {
        "calibration_date": "2026-04-20",
        "physics_lint_commit": sha,
        "grid_n": N,
        "periodic_spectral": periodic,
        "nonperiodic_fd": nonperiodic,
    }

    fix_dir = Path(__file__).parent / "fixtures"
    fix_dir.mkdir(exist_ok=True)
    (fix_dir / "norm_equivalence_bounds.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {fix_dir / 'norm_equivalence_bounds.json'}")
    print(f"periodic+spectral: C_max/c_min = {periodic['C_max_over_c_min']:.3f} (expect < 10)")
    print(
        f"non-periodic+FD:   C_max/c_min (raw) = "
        f"{nonperiodic['C_max_over_c_min_raw']:.3f} (expect ~k-scaling, ~4x)"
    )


if __name__ == "__main__":
    main()
