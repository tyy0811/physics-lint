"""Week-2/3 physics-lint programmatic self-test smoke script.

Until the ``physics-lint self-test`` CLI subcommand lands in Week 4,
this script is the self-test artifact for release criterion 1 (every
rule on every analytical input lands on PASS, SKIPPED for a declared
reason, or — in the one documented case — a WARN at ``info`` severity
that never moves overall_status).

What we cover:

- Laplace (harmonic polynomial on [0,1]^2 hD fd4)
    PH-RES-001, PH-BC-001, PH-BC-002, PH-POS-002, PH-NUM-002
- Poisson (sin*sin on [0,2pi]^2 periodic spectral)
    PH-RES-001, PH-RES-003
- Heat periodic spectral (cos*cos)
    PH-RES-001, PH-CON-001 (exact-mass), PH-CON-003
- Heat hD fd (eigenfunction decay)
    PH-RES-001, PH-CON-001 (rate-consistency), PH-POS-001
- Wave hD fd (standing wave)
    PH-RES-001, PH-CON-002, PH-VAR-002 (expected WARN/info)
- SYM rules (Week 3):
    PH-SYM-001 on C4-symmetric field -> PASS
    PH-SYM-002 on axis-symmetric field -> PASS
    PH-SYM-003 on radial CallableField -> PASS
    PH-SYM-001 on asymmetric field -> FAIL
    PH-SYM-004 on periodic field -> SKIPPED (V1 stub)

Exit 0 on success, 1 if any rule produced an unexpected status.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from physics_lint import CallableField, DomainSpec, GridField
from physics_lint.analytical import heat as heat_sols
from physics_lint.analytical import laplace as laplace_sols
from physics_lint.analytical import poisson as poisson_sols
from physics_lint.analytical import wave as wave_sols
from physics_lint.rules import (
    ph_bc_001,
    ph_bc_002,
    ph_con_001,
    ph_con_002,
    ph_con_003,
    ph_num_002,
    ph_pos_001,
    ph_pos_002,
    ph_res_001,
    ph_res_003,
    ph_sym_001,
    ph_sym_002,
    ph_sym_003,
    ph_sym_004,
    ph_var_002,
)

# ----- Case builders -------------------------------------------------------


def _laplace_case() -> tuple[GridField, DomainSpec, np.ndarray, np.ndarray]:
    n = 64
    sol = laplace_sols.harmonic_polynomial_square()
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(x, y, indexing="ij")
    u = sol.u(mesh_x, mesh_y)
    h = (1.0 / (n - 1), 1.0 / (n - 1))
    field = GridField(u, h=h, periodic=False, backend="fd")
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [n, n],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    boundary = field.values_on_boundary()
    return field, spec, boundary, boundary.copy()


def _laplace_refined_field() -> GridField:
    n = 128
    sol = laplace_sols.harmonic_polynomial_square()
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    mesh_x, mesh_y = np.meshgrid(x, y, indexing="ij")
    u = sol.u(mesh_x, mesh_y)
    return GridField(u, h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False, backend="fd")


def _poisson_periodic_case() -> tuple[GridField, DomainSpec]:
    n = 64
    sol = poisson_sols.periodic_sin_sin()
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    mesh_x, mesh_y = np.meshgrid(x, y, indexing="ij")
    u = sol.u(mesh_x, mesh_y)
    source = sol.source(mesh_x, mesh_y)
    h = (2 * np.pi / n, 2 * np.pi / n)
    field = GridField(u, h=h, periodic=True, backend="spectral")
    spec = DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [n, n],
            "domain": {"x": [0.0, 2 * np.pi], "y": [0.0, 2 * np.pi]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )
    # Rules that need the Poisson source read it via the runtime-injected
    # _source_array attribute, the same way loader.py plumbs .npz dumps.
    object.__setattr__(spec, "_source_array", source)
    return field, spec


def _heat_periodic_case() -> tuple[GridField, DomainSpec]:
    n, nt = 64, 16
    sol = heat_sols.periodic_cos_cos(kappa=0.01)
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    t = np.linspace(0.0, 0.5, nt)
    mesh_x, mesh_y = np.meshgrid(x, y, indexing="ij")
    pred = np.stack([sol.u(mesh_x, mesh_y, ti) for ti in t], axis=-1)
    field = GridField(
        pred,
        h=(2 * np.pi / n, 2 * np.pi / n, 0.5 / (nt - 1)),
        periodic=True,
        backend="spectral",
    )
    spec = DomainSpec.model_validate(
        {
            "pde": "heat",
            "grid_shape": [n, n, nt],
            "domain": {"x": [0.0, 2 * np.pi], "y": [0.0, 2 * np.pi], "t": [0.0, 0.5]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "diffusivity": 0.01,
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )
    return field, spec


def _heat_hd_case() -> tuple[GridField, DomainSpec]:
    n, nt = 64, 32
    sol = heat_sols.eigenfunction_decay_square(kappa=0.01)
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 0.5, nt)
    mesh_x, mesh_y = np.meshgrid(x, y, indexing="ij")
    pred = np.stack([sol.u(mesh_x, mesh_y, ti) for ti in t], axis=-1)
    field = GridField(
        pred,
        h=(1.0 / (n - 1), 1.0 / (n - 1), 0.5 / (nt - 1)),
        periodic=False,
        backend="fd",
    )
    spec = DomainSpec.model_validate(
        {
            "pde": "heat",
            "grid_shape": [n, n, nt],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.5]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "diffusivity": 0.01,
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    return field, spec


def _wave_hd_case() -> tuple[GridField, DomainSpec]:
    n, nt = 64, 32
    sol = wave_sols.standing_wave_square(c=1.0)
    x = np.linspace(0.0, 1.0, n)
    y = np.linspace(0.0, 1.0, n)
    t = np.linspace(0.0, 0.5, nt)
    mesh_x, mesh_y = np.meshgrid(x, y, indexing="ij")
    pred = np.stack([sol.u(mesh_x, mesh_y, ti) for ti in t], axis=-1)
    field = GridField(
        pred,
        h=(1.0 / (n - 1), 1.0 / (n - 1), 0.5 / (nt - 1)),
        periodic=False,
        backend="fd",
    )
    spec = DomainSpec.model_validate(
        {
            "pde": "wave",
            "grid_shape": [n, n, nt],
            "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 0.5]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "wave_speed": 1.0,
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    return field, spec


# ----- SYM case builders ---------------------------------------------------


def _sym_c4_case() -> tuple[GridField, DomainSpec]:
    """C4-symmetric field: x^2 + y^2 on a centered square grid."""
    n = 64
    x = np.linspace(-0.5, 0.5, n)
    y = np.linspace(-0.5, 0.5, n)
    xg, yg = np.meshgrid(x, y, indexing="ij")
    u = xg**2 + yg**2
    field = GridField(u, h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False)
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [n, n],
            "domain": {"x": [-0.5, 0.5], "y": [-0.5, 0.5]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "symmetries": {"declared": ["C4", "reflection_x", "reflection_y"]},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    return field, spec


def _sym_asymmetric_case() -> tuple[GridField, DomainSpec]:
    """Asymmetric field: x + 0.1*y on a centered square grid."""
    n = 64
    x = np.linspace(-0.5, 0.5, n)
    y = np.linspace(-0.5, 0.5, n)
    xg, yg = np.meshgrid(x, y, indexing="ij")
    u = xg + 0.1 * yg
    field = GridField(u, h=(1.0 / (n - 1), 1.0 / (n - 1)), periodic=False)
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [n, n],
            "domain": {"x": [-0.5, 0.5], "y": [-0.5, 0.5]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "symmetries": {"declared": ["C4"]},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    return field, spec


def _sym_so2_callable_case() -> tuple[CallableField, DomainSpec]:
    """SO(2)-invariant radial CallableField: r^2 on a centered grid."""
    n = 64

    def radial(pts: torch.Tensor) -> torch.Tensor:
        r2 = pts[..., 0] ** 2 + pts[..., 1] ** 2
        return r2.unsqueeze(-1)

    axis = torch.linspace(-0.5, 0.5, n)
    grid = torch.stack(torch.meshgrid(axis, axis, indexing="ij"), dim=-1)
    field = CallableField(radial, sampling_grid=grid, h=(1.0 / (n - 1), 1.0 / (n - 1)))
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [n, n],
            "domain": {"x": [-0.5, 0.5], "y": [-0.5, 0.5]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "symmetries": {"declared": ["SO2"]},
            "field": {"type": "callable", "backend": "fd", "adapter_path": "x"},
        }
    )
    return field, spec


def _sym_periodic_case() -> tuple[GridField, DomainSpec]:
    """Periodic field for PH-SYM-004 SKIPPED test."""
    n = 64
    x = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    y = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    xg, yg = np.meshgrid(x, y, indexing="ij")
    u = np.sin(xg) * np.sin(yg)
    field = GridField(u, h=(2 * np.pi / n, 2 * np.pi / n), periodic=True, backend="spectral")
    spec = DomainSpec.model_validate(
        {
            "pde": "laplace",
            "grid_shape": [n, n],
            "domain": {"x": [0.0, 2 * np.pi], "y": [0.0, 2 * np.pi]},
            "periodic": True,
            "boundary_condition": {"kind": "periodic"},
            "symmetries": {"declared": ["translation_x", "translation_y"]},
            "field": {"type": "grid", "backend": "spectral", "dump_path": "p.npz"},
        }
    )
    return field, spec


# ----- Runner --------------------------------------------------------------

Failure = tuple[str, str, str, str]  # (case, rule, status, reason_or_note)


def _record(
    failures: list[Failure],
    case: str,
    rule_id: str,
    result_status: str,
    result_reason: str | None,
    allowed: set[str],
) -> None:
    if result_status not in allowed:
        failures.append((case, rule_id, result_status, result_reason or ""))


def main() -> int:
    failures: list[Failure] = []
    ok: set[str] = {"PASS", "SKIPPED"}

    # --- Laplace
    field, spec, boundary, target = _laplace_case()
    r = ph_res_001.check(field, spec)
    _record(failures, "laplace", "PH-RES-001", r.status, r.reason, ok)
    r = ph_bc_001.check(field, spec, boundary_target=target)
    _record(failures, "laplace", "PH-BC-001", r.status, r.reason, ok)
    r = ph_bc_002.check(field, spec)
    _record(failures, "laplace", "PH-BC-002", r.status, r.reason, ok)
    r = ph_pos_002.check(field, spec, boundary_values=boundary)
    _record(failures, "laplace", "PH-POS-002", r.status, r.reason, ok)
    r = ph_num_002.check(field, spec, refined_field=_laplace_refined_field())
    _record(failures, "laplace", "PH-NUM-002", r.status, r.reason, ok)

    # --- Poisson
    field, spec = _poisson_periodic_case()
    r = ph_res_001.check(field, spec)
    _record(failures, "poisson periodic", "PH-RES-001", r.status, r.reason, ok)
    r = ph_res_003.check(field, spec)
    _record(failures, "poisson periodic", "PH-RES-003", r.status, r.reason, ok)

    # --- Heat periodic spectral
    field, spec = _heat_periodic_case()
    r = ph_res_001.check(field, spec)
    _record(failures, "heat periodic", "PH-RES-001", r.status, r.reason, ok)
    r = ph_con_001.check(field, spec)
    _record(failures, "heat periodic", "PH-CON-001", r.status, r.reason, ok)
    r = ph_con_003.check(field, spec)
    _record(failures, "heat periodic", "PH-CON-003", r.status, r.reason, ok)

    # --- Heat hD fd
    field, spec = _heat_hd_case()
    r = ph_res_001.check(field, spec)
    _record(failures, "heat hD", "PH-RES-001", r.status, r.reason, ok)
    r = ph_con_001.check(field, spec)
    _record(failures, "heat hD", "PH-CON-001", r.status, r.reason, ok)
    r = ph_pos_001.check(field, spec)
    _record(failures, "heat hD", "PH-POS-001", r.status, r.reason, ok)

    # --- Wave hD fd
    field, spec = _wave_hd_case()
    r = ph_res_001.check(field, spec)
    _record(failures, "wave hD", "PH-RES-001", r.status, r.reason, ok)
    r = ph_con_002.check(field, spec)
    _record(failures, "wave hD", "PH-CON-002", r.status, r.reason, ok)
    # PH-VAR-002 is an info-severity caveat that always fires on wave —
    # design doc §10.2. It WARNs intentionally and must not be counted
    # as a failure.
    r = ph_var_002.check(field, spec)
    if r.rule_id != "PH-VAR-002":
        failures.append(("wave hD", "PH-VAR-002", "???", f"unexpected rule_id {r.rule_id}"))
    elif r.severity != "info":
        failures.append(("wave hD", "PH-VAR-002", r.status, f"severity={r.severity} not 'info'"))
    elif r.status not in {"WARN", "PASS", "SKIPPED"}:
        failures.append(("wave hD", "PH-VAR-002", r.status, r.reason or ""))

    # --- SYM rules (Week 3)
    field, spec = _sym_c4_case()
    r = ph_sym_001.check(field, spec)
    _record(failures, "sym C4", "PH-SYM-001", r.status, r.reason, ok)
    r = ph_sym_002.check(field, spec)
    _record(failures, "sym C4", "PH-SYM-002", r.status, r.reason, ok)

    field, spec = _sym_asymmetric_case()
    r = ph_sym_001.check(field, spec)
    _record(failures, "sym asym", "PH-SYM-001", r.status, r.reason, {"FAIL"})

    field, spec = _sym_so2_callable_case()
    r = ph_sym_003.check(field, spec)
    _record(failures, "sym SO2", "PH-SYM-003", r.status, r.reason, ok)

    field, spec = _sym_periodic_case()
    r = ph_sym_004.check(field, spec)
    _record(failures, "sym periodic", "PH-SYM-004", r.status, r.reason, ok)

    if failures:
        print("FAIL")
        for case, rule, status, note in failures:
            tail = f" — {note}" if note else ""
            print(f"  {case:<18} {rule:<12} {status}{tail}")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
