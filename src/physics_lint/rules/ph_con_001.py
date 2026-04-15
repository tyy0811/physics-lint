"""PH-CON-001: Heat mass conservation.

Design doc §4.4 and §8.1. Branches on BCSpec.conserves_mass:

- True (PER / hN): compute integral of u at each timestep, report absolute
  drift scaled by a characteristic mass (max of |M(0)| and the L^1 norm
  of u at t=0). The L^1 fallback keeps the rule well-defined for periodic
  zero-mean data where M(0) ≈ 0 to quadrature precision.

- False (hD / generic Dirichlet): compute both sides of the identity
    dM/dt = kappa * integral(lap u) dx
  via the divergence theorem, and report relative L^2 error over [0, T]
  between observed and expected time derivative. mode='rate-consistency'.
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field, GridField
from physics_lint.norms import trapezoidal_integral
from physics_lint.report import RuleResult
from physics_lint.rules._helpers import _load_floor, _tristate, ensure_grid_field
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-CON-001"
__rule_name__ = "Mass conservation violation"
__default_severity__ = "error"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-CON-001"
_CITATION = "divergence theorem; design doc §8.1 BC scoping"

_MIN_TIME_STEPS_FOR_GRADIENT = 3


def check(field: Field, spec: DomainSpec) -> RuleResult:
    if spec.pde != "heat":
        return _skipped(f"PH-CON-001 heat-only in V1; got {spec.pde}")

    field = ensure_grid_field(field, spec)

    u = field.values()
    if u.ndim < 3:
        return _skipped(f"PH-CON-001 requires a time-dependent field (values shape={u.shape})")
    nt = u.shape[-1]
    if nt < _MIN_TIME_STEPS_FOR_GRADIENT:
        return _skipped(
            f"PH-CON-001 needs at least {_MIN_TIME_STEPS_FOR_GRADIENT} time "
            f"samples for a 2nd-order central time derivative; got nt={nt}."
        )

    spatial_h = tuple(float(h) for h in field.h[:-1])
    dt = float(field.h[-1])

    mass_series = np.array(
        [trapezoidal_integral(np.take(u, k, axis=-1), spatial_h) for k in range(nt)]
    )

    method_key = "fd4" if field.backend == "fd" else field.backend
    if spec.boundary_condition.conserves_mass:
        return _check_exact_mass(mass_series, u, spatial_h, method_key, spec)
    return _check_rate_consistency(
        u=u,
        mass_series=mass_series,
        dt=dt,
        spatial_h=spatial_h,
        method_key=method_key,
        spec=spec,
        field=field,
    )


def _check_exact_mass(
    mass_series: np.ndarray,
    u: np.ndarray,
    spatial_h: tuple[float, ...],
    method_key: str,
    spec: DomainSpec,
) -> RuleResult:
    m0 = float(mass_series[0])
    drift_abs = float(np.max(np.abs(mass_series - m0)))
    # Scale by the characteristic total mass: |M(0)| alone is unreliable for
    # zero-mean periodic data where the trapezoidal quadrature gives a tiny
    # spurious non-zero. Use the L^1 norm of u(t=0) as a robust fallback so
    # the relative drift stays meaningful across positive, oscillatory, and
    # zero-mean initial conditions.
    l1_scale = float(trapezoidal_integral(np.abs(np.take(u, 0, axis=-1)), spatial_h))
    scale = max(abs(m0), l1_scale, 1e-12)
    relative_drift = drift_abs / scale

    floor = _load_floor(
        rule="PH-CON-001",
        pde="heat",
        grid_shape=spec.grid_shape,
        method=method_key,
        norm="relative",
    )
    ratio = relative_drift / floor.value if floor.value > 0 else float("inf")
    status = _tristate(ratio, pass_=floor.tolerance * 10, fail_=floor.tolerance * 100)
    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=relative_drift,
        violation_ratio=ratio,
        mode="exact-mass",
        reason=None,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="max |M(t) - M(0)| / max(|M(0)|, ||u_0||_1)",
        citation="classical heat equation under PER/hN",
        doc_url=_DOC_URL,
    )


def _check_rate_consistency(
    *,
    u: np.ndarray,
    mass_series: np.ndarray,
    dt: float,
    spatial_h: tuple[float, ...],
    method_key: str,
    spec: DomainSpec,
    field: GridField,
) -> RuleResult:
    kappa = spec.diffusivity
    assert kappa is not None

    dm_dt = np.gradient(mass_series, dt, edge_order=2)

    # Divergence theorem: integral over boundary of (grad u . n) dS
    #                   = integral over domain of Laplacian(u) dV.
    # So the expected dM/dt at time t_k is kappa * integral(Laplacian(u(., t_k)) dV).
    nt = u.shape[-1]
    expected = np.zeros(nt)
    for k in range(nt):
        slice_k = np.take(u, k, axis=-1)
        sub_field = GridField(
            slice_k,
            h=spatial_h,
            periodic=spec.periodic,
            backend=field.backend,
        )
        lap = sub_field.laplacian().values()
        expected[k] = kappa * trapezoidal_integral(lap, spatial_h)

    err = float(np.sqrt(np.sum((dm_dt - expected) ** 2) * dt))
    scale = max(float(np.sqrt(np.sum(expected**2) * dt)), 1e-12)
    relative = err / scale

    floor = _load_floor(
        rule="PH-CON-001",
        pde="heat",
        grid_shape=spec.grid_shape,
        method=method_key,
        norm="relative_L2_over_T",
    )
    ratio = relative / floor.value if floor.value > 0 else float("inf")
    status = _tristate(ratio, pass_=floor.tolerance * 10, fail_=floor.tolerance * 100)
    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=relative,
        violation_ratio=ratio,
        mode="rate-consistency",
        reason=None,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="relative L^2 over [0, T] of dM/dt - kappa * integral(lap u)",
        citation=_CITATION,
        doc_url=_DOC_URL,
    )


def _skipped(reason: str) -> RuleResult:
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
