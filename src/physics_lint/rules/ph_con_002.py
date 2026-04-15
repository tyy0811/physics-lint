"""PH-CON-002: Wave energy conservation violation.

E(t) = 0.5 * integral(u_t^2 + c^2 |grad u|^2) is conserved under hD/hN/PER.

Implementation: we evaluate the potential term via integration by parts,
0.5 * c^2 * integral(|grad u|^2) = -0.5 * c^2 * integral(u * Laplacian u),
which holds exactly under hD (u=0 on the boundary) and periodic BCs
(no boundary) — exactly the BC classes for which
``spec.boundary_condition.conserves_energy`` is true. This dispatches
the spatial derivative through the field's own backend (spectral on
periodic grids, 4th-order FD on non-periodic grids) instead of
``np.gradient`` with endpoint stencils, which corrupted the seam of
periodic fields in the earlier Week-2 draft (Codex adversarial review,
Finding 3). The quadrature picks rectangle-vs-trapezoidal to match
the endpoint convention of the grid.
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field, GridField
from physics_lint.norms import integrate_over_domain
from physics_lint.report import RuleResult
from physics_lint.rules._helpers import _load_floor, _tristate, ensure_grid_field
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-CON-002"
__rule_name__ = "Energy conservation violation"
__default_severity__ = "error"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-CON-002"
_CITATION = "classical wave equation energy estimate"

_MIN_TIME_STEPS_FOR_GRADIENT = 3


def check(field: Field, spec: DomainSpec) -> RuleResult:
    if spec.pde != "wave":
        return _skip(f"PH-CON-002 applies to wave only; got {spec.pde}")
    if not spec.boundary_condition.conserves_energy:
        return _skip(
            f"BC '{spec.boundary_condition.kind}' does not conserve wave energy; "
            "PH-CON-002 does not apply"
        )

    field = ensure_grid_field(field, spec)

    u = field.values()
    if u.ndim < 3:
        return _skip(f"PH-CON-002 requires a time-dependent field (values shape={u.shape})")
    nt = u.shape[-1]
    if nt < _MIN_TIME_STEPS_FOR_GRADIENT:
        return _skip(
            f"PH-CON-002 needs at least {_MIN_TIME_STEPS_FOR_GRADIENT} time "
            f"samples for a 2nd-order central time derivative; got nt={nt}."
        )

    spatial_h = tuple(float(h) for h in field.h[:-1])
    dt = float(field.h[-1])
    c = spec.wave_speed
    assert c is not None

    u_t = np.gradient(u, dt, axis=-1, edge_order=2)
    energies = np.empty(nt)
    for k in range(nt):
        slice_k = np.take(u, k, axis=-1)
        slice_ut = np.take(u_t, k, axis=-1)
        # Spatial Laplacian via the field's own backend so periodic
        # grids use the FFT path and non-periodic grids use fd4 —
        # matches how every other Week-2 rule handles periodic seams.
        sub_field = GridField(
            slice_k,
            h=spatial_h,
            periodic=spec.periodic,
            backend=field.backend,
        )
        lap_k = sub_field.laplacian().values()
        kinetic_density = 0.5 * slice_ut**2
        # Potential via IBP: 0.5 c^2 |grad u|^2 integrated
        # equals -0.5 c^2 u * Laplacian u integrated, on hD/hN/PER.
        potential_density = -0.5 * (c**2) * slice_k * lap_k
        density = kinetic_density + potential_density
        energies[k] = integrate_over_domain(density, spatial_h, periodic=spec.periodic)

    e0 = float(energies[0])
    denom = max(abs(e0), 1e-12)
    drift = float(np.max(np.abs(energies - e0)) / denom)

    method_key = "fd4" if field.backend == "fd" else field.backend
    floor = _load_floor(
        rule="PH-CON-002",
        pde="wave",
        grid_shape=spec.grid_shape,
        method=method_key,
        norm="relative",
    )
    ratio = drift / floor.value if floor.value > 0 else float("inf")
    status = _tristate(ratio, pass_=floor.tolerance * 10, fail_=floor.tolerance * 100)

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=drift,
        violation_ratio=ratio,
        mode=None,
        reason=None,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="max |E(t) - E(0)| / |E(0)|",
        citation=_CITATION,
        doc_url=_DOC_URL,
    )


def _skip(reason: str) -> RuleResult:
    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status="SKIPPED",
        raw_value=None,
        violation_ratio=None,
        mode=None,
        reason=reason,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="",
        citation=_CITATION,
        doc_url=_DOC_URL,
    )
