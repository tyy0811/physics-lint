"""PH-RES-001: Residual exceeds variationally-correct norm threshold.

Design doc §7.5. For Laplace/Poisson the recommended norm is H^-1;
periodic grids use the spectral formula in physics_lint.norms.
Non-periodic Laplace/Poisson on Week 1 falls back to L^2 with a
PH-VAR-001-style caveat (implemented later; Week 1 only emits the
result, not the caveat).
"""

from __future__ import annotations

from physics_lint.field import Field, GridField
from physics_lint.norms import h_minus_one_spectral, l2_grid
from physics_lint.report import RuleResult
from physics_lint.rules._helpers import Floor, _load_floor, _tristate
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-RES-001"
__rule_name__ = "Residual exceeds variationally-correct norm threshold"
__default_severity__ = "error"
__input_modes__ = frozenset({"adapter", "dump"})

__default_thresholds__ = {"tol_pass": 10.0, "tol_fail": 100.0}

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-RES-001"


def check(field: Field, spec: DomainSpec) -> RuleResult:
    """Compute strong-form residual, measure in H^-1 (or L^2 fallback), report."""
    if not isinstance(field, GridField):
        # Callable fields are handled by materializing via the loader before
        # rule dispatch; if we still see one here, raise to surface the bug.
        raise TypeError(
            f"PH-RES-001 requires a GridField; got {type(field).__name__}. "
            "Adapter-mode callables must be materialized by the loader."
        )

    # Compute the residual: R = -Delta u for Laplace; R = -Delta u - f for Poisson
    lap = field.laplacian().values()
    if spec.pde == "laplace":
        residual = -lap
    elif spec.pde == "poisson":
        # Week 1 scope covers Laplace (automatic source=0); Poisson source
        # wiring lands in Week 2 once the loader plumbs spec.source_term.
        raise NotImplementedError(
            "PH-RES-001 for Poisson requires a source term on spec; Week 1 scope "
            "covers Laplace (automatic source=0); Poisson source wiring lands in Week 2."
        )
    elif spec.pde in {"heat", "wave"}:
        raise NotImplementedError(
            f"PH-RES-001 for {spec.pde} lands in Week 2 with the Bochner norm."
        )
    # pydantic guarantees spec.pde is one of the PDEKind literals; no else.

    # Norm selection
    method = field.backend  # "fd" or "spectral"
    if spec.periodic and field.backend == "spectral":
        raw_value = h_minus_one_spectral(residual, field.h)
        norm_name = "H-1"
    else:
        # Non-periodic Week 1: L^2 fallback with PH-VAR-001 documented caveat
        raw_value = l2_grid(residual, field.h)
        norm_name = "L2"

    floor: Floor = _load_floor(
        rule="PH-RES-001",
        pde=spec.pde,
        grid_shape=spec.grid_shape,
        method=("fd4" if method == "fd" else method),
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
        citation="Bachmayr et al. 2024; Ernst et al. 2025 v3",
        doc_url=_DOC_URL,
    )
