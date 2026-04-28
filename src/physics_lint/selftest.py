"""physics-lint analytical self-test battery.

This is release criterion 1 incarnate: exit 0 iff every rule hits its
calibrated floor within tolerance on every analytical input.

Invoked by:
- `physics-lint self-test` CLI subcommand (src/physics_lint/cli/self_test.py)
- `scripts/smoke_self_test.py` wrapper (kept for historical / bug-report use)

Both call `main()` below. The logic lives in the package (not in
`scripts/`) so that `physics-lint self-test` works from an installed
wheel / PyPI package, where `scripts/` is not shipped.
"""

from __future__ import annotations

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


def _sym_c4_case() -> tuple[GridField, DomainSpec]:
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


Failure = tuple[str, str, str, str]


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


def run(*, verbose: bool = False) -> tuple[int, str]:
    """Run the full analytical battery.

    Returns (exit_code, stdout_text). exit_code is 0 on PASS, 1 if any
    rule produced an unexpected status.
    """
    failures: list[Failure] = []
    ok: set[str] = {"PASS", "SKIPPED"}
    lines: list[str] = []

    field, spec, _boundary, target = _laplace_case()
    r = ph_res_001.check(field, spec)
    _record(failures, "laplace", "PH-RES-001", r.status, r.reason, ok)
    r = ph_bc_001.check(field, spec, boundary_target=target)
    _record(failures, "laplace", "PH-BC-001", r.status, r.reason, ok)
    r = ph_bc_002.check(field, spec)
    _record(failures, "laplace", "PH-BC-002", r.status, r.reason, ok)
    r = ph_pos_002.check(field, spec, boundary_values=target)
    _record(failures, "laplace", "PH-POS-002", r.status, r.reason, ok)
    r = ph_num_002.check(field, spec, refined_field=_laplace_refined_field())
    _record(failures, "laplace", "PH-NUM-002", r.status, r.reason, ok)

    field, spec = _poisson_periodic_case()
    r = ph_res_001.check(field, spec)
    _record(failures, "poisson periodic", "PH-RES-001", r.status, r.reason, ok)
    r = ph_res_003.check(field, spec)
    _record(failures, "poisson periodic", "PH-RES-003", r.status, r.reason, ok)

    field, spec = _heat_periodic_case()
    r = ph_res_001.check(field, spec)
    _record(failures, "heat periodic", "PH-RES-001", r.status, r.reason, ok)
    r = ph_con_001.check(field, spec)
    _record(failures, "heat periodic", "PH-CON-001", r.status, r.reason, ok)
    r = ph_con_003.check(field, spec)
    _record(failures, "heat periodic", "PH-CON-003", r.status, r.reason, ok)

    field, spec = _heat_hd_case()
    r = ph_res_001.check(field, spec)
    _record(failures, "heat hD", "PH-RES-001", r.status, r.reason, ok)
    r = ph_con_001.check(field, spec)
    _record(failures, "heat hD", "PH-CON-001", r.status, r.reason, ok)
    r = ph_pos_001.check(field, spec)
    _record(failures, "heat hD", "PH-POS-001", r.status, r.reason, ok)

    field, spec = _wave_hd_case()
    r = ph_res_001.check(field, spec)
    _record(failures, "wave hD", "PH-RES-001", r.status, r.reason, ok)
    r = ph_con_002.check(field, spec)
    _record(failures, "wave hD", "PH-CON-002", r.status, r.reason, ok)
    r = ph_var_002.check(field, spec)
    if r.rule_id != "PH-VAR-002":
        failures.append(("wave hD", "PH-VAR-002", "???", f"unexpected rule_id {r.rule_id}"))
    elif r.severity != "info":
        failures.append(("wave hD", "PH-VAR-002", r.status, f"severity={r.severity} not 'info'"))
    elif r.status not in {"WARN", "PASS", "SKIPPED"}:
        failures.append(("wave hD", "PH-VAR-002", r.status, r.reason or ""))

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

    field_dump, spec_dump = _sym_c4_case()
    spec_dump_so2 = DomainSpec.model_validate(
        {
            **spec_dump.model_dump(),
            "symmetries": {"declared": ["SO2"]},
        }
    )
    r = ph_sym_003.check(field_dump, spec_dump_so2)
    _record(failures, "sym SO2 dump", "PH-SYM-003", r.status, r.reason, ok)

    field_poisson, _ = _sym_c4_case()
    spec_poisson = DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [64, 64],
            "domain": {"x": [-0.5, 0.5], "y": [-0.5, 0.5]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet"},
            "symmetries": {"declared": ["C4"]},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )
    r = ph_sym_001.check(field_poisson, spec_poisson)
    _record(failures, "sym C4 poisson", "PH-SYM-001", r.status, r.reason, ok)

    field, spec = _sym_periodic_case()
    r = ph_sym_004.check(field, spec)
    _record(failures, "sym periodic", "PH-SYM-004", r.status, r.reason, ok)

    if failures:
        lines.append("FAIL")
        for case, rule, status, note in failures:
            tail = f" — {note}" if note else ""
            lines.append(f"  {case:<18} {rule:<12} {status}{tail}")
        return 1, "\n".join(lines) + "\n"
    lines.append("PASS")
    return 0, "\n".join(lines) + "\n"


def main() -> int:
    """Console entry point. Prints the report and returns the exit code."""
    code, text = run()
    print(text, end="")
    return code


if __name__ == "__main__":
    import sys

    sys.exit(main())
