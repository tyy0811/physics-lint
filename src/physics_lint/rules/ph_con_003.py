"""PH-CON-003: Heat energy dissipation sign violation.

For heat under BC that gives d/dt integral(u^2) <= 0 (hD, hN, PER), report
positive d/dt integral(u^2) values as violations. The raw value is the
maximum positive slope of the L^2 norm squared, normalized by the peak
energy so the ratio is dimensionless.
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field
from physics_lint.norms import integrate_over_domain
from physics_lint.report import RuleResult
from physics_lint.rules._helpers import _load_floor, _tristate, ensure_grid_field
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-CON-003"
__rule_name__ = "Energy dissipation sign violation"
__default_severity__ = "warning"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-CON-003"
_CITATION = "classical parabolic energy estimate"

_MIN_TIME_STEPS_FOR_GRADIENT = 3


def check(field: Field, spec: DomainSpec) -> RuleResult:
    if spec.pde != "heat":
        return _skip(f"PH-CON-003 applies to heat only; got {spec.pde}")
    if not spec.boundary_condition.conserves_energy:
        return _skip(
            f"BC '{spec.boundary_condition.kind}' does not dissipate heat energy; "
            "PH-CON-003 does not apply"
        )

    field = ensure_grid_field(field, spec)

    u = field.values()
    if u.ndim < 3:
        return _skip(f"PH-CON-003 requires a time-dependent field (values shape={u.shape})")
    nt = u.shape[-1]
    if nt < _MIN_TIME_STEPS_FOR_GRADIENT:
        return _skip(
            f"PH-CON-003 needs at least {_MIN_TIME_STEPS_FOR_GRADIENT} time "
            f"samples for a 2nd-order central time derivative; got nt={nt}."
        )

    spatial_h = tuple(float(h) for h in field.h[:-1])
    dt = float(field.h[-1])

    energy = np.array(
        [
            integrate_over_domain(np.take(u, k, axis=-1) ** 2, spatial_h, periodic=spec.periodic)
            for k in range(nt)
        ]
    )
    de_dt = np.gradient(energy, dt, edge_order=2)
    max_growth = float(np.max(de_dt))
    energy_scale = max(float(np.max(energy)), 1e-12)
    violation = max(0.0, max_growth) / energy_scale

    method_key = "fd4" if field.backend == "fd" else field.backend
    floor = _load_floor(
        rule="PH-CON-003",
        pde="heat",
        grid_shape=spec.grid_shape,
        method=method_key,
        norm="relative",
    )
    ratio = violation / floor.value if floor.value > 0 else 0.0
    status = _tristate(ratio, pass_=floor.tolerance * 10, fail_=floor.tolerance * 100)

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=violation,
        violation_ratio=ratio,
        mode=None,
        reason=None if max_growth <= 0 else "energy increases in time (heat should dissipate)",
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="max (dE/dt / max E)",
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
