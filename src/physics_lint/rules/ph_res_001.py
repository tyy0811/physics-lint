"""PH-RES-001: Residual exceeds variationally-correct norm threshold.

Design doc §7.5. Norm selection by PDE class:

- Laplace / Poisson (stationary, elliptic): H^-1 on periodic spectral grids,
  L^2 fallback on non-periodic FD.
- Heat (parabolic): Bochner L^2(0, T; H^-1) when the spatial grid is
  periodic+spectral (variationally correct); Bochner L^2(0, T; L^2) as a
  non-variational fallback otherwise. Trapezoidal time quadrature throughout
  matches the loader's endpoint-inclusive sampling.
- Wave (hyperbolic): same Bochner norm, but the variational framework is
  conjectural — see PH-VAR-002 which always fires on wave.

The non-periodic fallback exists because h_minus_one_spectral drops the DC
mode: a constant-in-space residual (e.g. u_t = 1 from a linear-in-t field)
would be silently reported as zero on an hD grid, producing a physics
false negative. Bochner-L2 keeps every mode at the cost of not being
variationally correct for H^-1 — that tradeoff is documented per-result
via recommended_norm="Bochner-L2" so reports surface it.
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field, GridField
from physics_lint.norms import (
    bochner_l2_fallback,
    bochner_l2_h_minus_one,
    h_minus_one_spectral,
    l2_grid,
)
from physics_lint.report import RuleResult
from physics_lint.rules._helpers import Floor, _load_floor, _tristate
from physics_lint.spec import DomainSpec

# np.gradient(..., edge_order=2) needs at least 3 samples along the axis.
# Fewer than that is a spec error, not a runtime exception — emit SKIPPED.
_MIN_TIME_STEPS_FOR_GRADIENT = 3

__rule_id__ = "PH-RES-001"
__rule_name__ = "Residual exceeds variationally-correct norm threshold"
__default_severity__ = "error"
__input_modes__ = frozenset({"adapter", "dump"})

__default_thresholds__ = {"tol_pass": 10.0, "tol_fail": 100.0}

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-RES-001"

_CITATION = "Bachmayr et al. 2024; Ernst et al. 2025 v3"


def check(field: Field, spec: DomainSpec) -> RuleResult:
    """Compute strong-form residual, measure in H^-1 / Bochner, report."""
    if not isinstance(field, GridField):
        # Callable fields are handled by materializing via the loader before
        # rule dispatch; if we still see one here, raise to surface the bug.
        raise TypeError(
            f"PH-RES-001 requires a GridField; got {type(field).__name__}. "
            "Adapter-mode callables must be materialized by the loader."
        )

    method = field.backend  # "fd" or "spectral"
    method_key = "fd4" if method == "fd" else method

    if spec.pde == "laplace":
        lap = field.laplacian().values()
        residual = -lap
        raw_value, norm_name = _compute_spatial_norm(residual, field, spec)
    elif spec.pde == "poisson":
        source = _resolve_source(spec)
        if source is None:
            return _skipped(
                "PH-RES-001 Poisson path requires a source array on the spec "
                "(set via adapter domain_spec() or stored as 'source' in a .npz dump)."
            )
        lap = field.laplacian().values()
        if source.shape != lap.shape:
            return _skipped(
                f"PH-RES-001 Poisson source shape {source.shape} does not match "
                f"field Laplacian shape {lap.shape}; fix the dump or adapter."
            )
        residual = -lap - source
        raw_value, norm_name = _compute_spatial_norm(residual, field, spec)
    elif spec.pde in {"heat", "wave"}:
        skip = _check_time_axis_preconditions(field, spec)
        if skip is not None:
            return skip
        if spec.pde == "heat":
            raw_value, norm_name = _compute_heat_bochner_residual(field, spec)
        else:
            raw_value, norm_name = _compute_wave_bochner_residual(field, spec)
    else:  # pragma: no cover — pydantic guarantees spec.pde is a literal
        raise ValueError(f"unknown PDE {spec.pde}")

    floor: Floor = _load_floor(
        rule="PH-RES-001",
        pde=spec.pde,
        grid_shape=spec.grid_shape,
        method=method_key,
        norm=norm_name,
    )
    ratio = raw_value / floor.value if floor.value > 0 else float("inf")
    status = _tristate(ratio, pass_=floor.tolerance * 10, fail_=floor.tolerance * 100)

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=raw_value,
        violation_ratio=ratio,
        mode=None,
        reason=None,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm=norm_name,
        citation=_CITATION,
        doc_url=_DOC_URL,
    )


def _compute_spatial_norm(
    residual: np.ndarray,
    field: GridField,
    spec: DomainSpec,
) -> tuple[float, str]:
    """H^-1 spectral when periodic+spectral; L^2 trapezoidal otherwise."""
    if spec.periodic and field.backend == "spectral":
        return float(h_minus_one_spectral(residual, field.h)), "H-1"
    return float(l2_grid(residual, field.h)), "L2"


def _resolve_source(spec: DomainSpec) -> np.ndarray | None:
    """Return the Poisson source array from a runtime-injected spec attribute.

    The loader stashes the source array under spec._source_array via
    object.__setattr__ so pydantic's frozen-model contract stays intact
    while the rule still has access to it. Returns None if no source
    was plumbed — the caller then emits SKIPPED with a clear reason.
    """
    source = getattr(spec, "_source_array", None)
    if source is None and hasattr(spec, "__pydantic_extra__"):
        extras = spec.__pydantic_extra__ or {}
        source = extras.get("_source_array")
    if source is None:
        return None
    return np.asarray(source)


def _check_time_axis_preconditions(field: GridField, spec: DomainSpec) -> RuleResult | None:
    """Return a SKIPPED RuleResult if heat/wave preconditions are unmet, else None.

    Two guards:
    1. Field must actually carry a time axis. `spec.domain.is_time_dependent`
       is true for heat/wave, but a malformed dump could still hand us a
       2D tensor — catch that here rather than letting u.ndim < 3 raise
       deep inside the residual helpers.
    2. np.gradient(edge_order=2) needs at least 3 samples along the axis.
       A heat/wave spec with grid_shape=[N, N, 2] validates through
       pydantic today (DomainSpec only checks the tuple length); without
       this guard the rule would raise ValueError mid-run.
    """
    u = field.values()
    if u.ndim < 3:
        return _skipped(
            f"PH-RES-001 {spec.pde} residual requires a time-axis field; got values shape {u.shape}"
        )
    nt = u.shape[-1]
    if nt < _MIN_TIME_STEPS_FOR_GRADIENT:
        return _skipped(
            f"PH-RES-001 {spec.pde} residual needs at least "
            f"{_MIN_TIME_STEPS_FOR_GRADIENT} time samples for the 2nd-order "
            f"central-difference time derivative; got nt={nt}. "
            "Increase grid_shape[-1] or resample the prediction."
        )
    return None


def _bochner_of(
    residual_series: np.ndarray,
    *,
    field: GridField,
    spec: DomainSpec,
    spatial_h: tuple[float, ...],
    dt: float,
) -> tuple[float, str]:
    """Pick the right Bochner norm based on periodic+spectral availability.

    Periodic spatial grids with the spectral backend get the variationally
    correct Bochner L^2(0,T; H^-1). Everything else (non-periodic hD/hN,
    or FD-backed periodic grids) falls back to Bochner L^2(0,T; L^2). The
    fallback label propagates via recommended_norm so downstream reports
    flag the approximate measurement.
    """
    if spec.periodic and field.backend == "spectral":
        bochner = bochner_l2_h_minus_one(residual_series, spatial_h=spatial_h, dt=dt)
        return float(bochner), "Bochner-H-1"
    bochner = bochner_l2_fallback(residual_series, spatial_h=spatial_h, dt=dt)
    return float(bochner), "Bochner-L2"


def _compute_heat_bochner_residual(
    field: GridField,
    spec: DomainSpec,
) -> tuple[float, str]:
    """Heat residual u_t - kappa * Lap u, measured in Bochner L^2 over time.

    Slice-by-slice spatial Laplacian (reusing the field's backend for
    consistency with the calibration floor); central-difference time
    derivative via np.gradient with edge_order=2. Periodic+spectral path
    uses Bochner-H^-1; everything else uses Bochner-L2 (see module docstring).
    """
    u = field.values()
    spatial_h = field.h[:-1]
    dt = field.h[-1]
    kappa = spec.diffusivity
    assert kappa is not None  # pydantic already validated

    u_t = np.gradient(u, dt, axis=-1, edge_order=2)
    nt = u.shape[-1]
    residual_series = np.empty_like(u)
    for k in range(nt):
        slice_k = np.take(u, k, axis=-1)
        sub_field = GridField(
            slice_k,
            h=spatial_h,
            periodic=spec.periodic,
            backend=field.backend,
        )
        lap_k = sub_field.laplacian().values()
        residual_series[..., k] = np.take(u_t, k, axis=-1) - kappa * lap_k

    return _bochner_of(residual_series, field=field, spec=spec, spatial_h=spatial_h, dt=dt)


def _compute_wave_bochner_residual(
    field: GridField,
    spec: DomainSpec,
) -> tuple[float, str]:
    """Wave residual u_tt - c^2 * Lap u, measured in Bochner L^2 over time.

    Two central-difference passes give u_tt; edge_order=2 keeps 2nd-order
    accuracy at the boundary too. Norm selection is the same as for heat
    (periodic+spectral -> Bochner-H^-1, otherwise Bochner-L2).
    """
    u = field.values()
    spatial_h = field.h[:-1]
    dt = field.h[-1]
    c = spec.wave_speed
    assert c is not None

    u_t = np.gradient(u, dt, axis=-1, edge_order=2)
    u_tt = np.gradient(u_t, dt, axis=-1, edge_order=2)
    nt = u.shape[-1]
    residual_series = np.empty_like(u)
    for k in range(nt):
        slice_k = np.take(u, k, axis=-1)
        sub_field = GridField(
            slice_k,
            h=spatial_h,
            periodic=spec.periodic,
            backend=field.backend,
        )
        lap_k = sub_field.laplacian().values()
        residual_series[..., k] = np.take(u_tt, k, axis=-1) - (c**2) * lap_k

    return _bochner_of(residual_series, field=field, spec=spec, spatial_h=spatial_h, dt=dt)


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
