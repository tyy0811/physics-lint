"""PH-CON-002: Wave energy conservation violation.

E(t) = 0.5 * integral(u_t^2 + c^2 |grad u|^2) is conserved under hD/hN/PER.
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field
from physics_lint.norms import trapezoidal_integral
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
        gx = np.gradient(slice_k, spatial_h[0], axis=0, edge_order=2)
        gy = np.gradient(slice_k, spatial_h[1], axis=1, edge_order=2)
        kinetic = slice_ut**2
        gradient_sq = gx**2 + gy**2
        density = 0.5 * (kinetic + c**2 * gradient_sq)
        energies[k] = trapezoidal_integral(density, spatial_h)

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
