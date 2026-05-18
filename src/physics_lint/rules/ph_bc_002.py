"""PH-BC-002: Boundary flux imbalance (divergence theorem).

For Laplace/Poisson: by the divergence theorem applied to ``F = grad u``,
the integral of ``Delta u`` over the domain equals the net outward flux
of ``grad u`` through the boundary. Violation of this identity is a sign
that the learned field is inconsistent with the PDE at a weak-form level
even if the pointwise residual is small.

Both Laplace and Poisson are implemented. Laplace: expected imbalance is
zero. Poisson (-Delta u = f): the imbalance is integral(Delta u) +
integral(f); the source array is read from ``spec._source_array``,
plumbed by the loader. When no source array is present the Poisson arm
emits SKIPPED with a reason rather than guessing f.
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field
from physics_lint.norms import l2_grid, trapezoidal_integral
from physics_lint.report import RuleResult
from physics_lint.rules._helpers import _resolve_source, ensure_grid_field
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-BC-002"
__rule_name__ = "Boundary flux imbalance (divergence theorem)"
__default_severity__ = "warning"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-BC-002"


def check(field: Field, spec: DomainSpec) -> RuleResult:
    if spec.pde not in {"laplace", "poisson"}:
        return RuleResult(
            rule_id=__rule_id__,
            rule_name=__rule_name__,
            severity=__default_severity__,
            status="SKIPPED",
            raw_value=None,
            violation_ratio=None,
            mode=None,
            reason=f"PH-BC-002 applies to laplace/poisson only; got {spec.pde}",
            refinement_rate=None,
            spatial_map=None,
            recommended_norm="",
            citation="divergence theorem",
            doc_url=_DOC_URL,
        )
    # Accept both dump (GridField) and adapter (CallableField) inputs per
    # __input_modes__; ensure_grid_field materializes the callable onto its
    # sampling grid so the downstream divergence-theorem computation uses
    # concrete values and a backend.
    field = ensure_grid_field(field, spec)

    if spec.pde == "poisson":
        # Poisson convention: -Delta u = f. Hence the expected value of the
        # domain integral of Delta u is -integral(f), and the divergence-
        # theorem imbalance is integral(Delta u) + integral(f). The source
        # array is plumbed onto the spec by the loader (loader._resolve_
        # source_term / _load_dump) under the private _source_array
        # attribute; _resolve_source is the shared lookup PH-RES-001 uses.
        source = _resolve_source(spec)
        if source is None:
            return RuleResult(
                rule_id=__rule_id__,
                rule_name=__rule_name__,
                severity=__default_severity__,
                status="SKIPPED",
                raw_value=None,
                violation_ratio=None,
                mode=None,
                reason=(
                    "PH-BC-002 for Poisson needs a source array; provide one "
                    "via an .npz 'source' key or a source_term= config pointer."
                ),
                refinement_rate=None,
                spatial_map=None,
                recommended_norm="",
                citation="classical divergence theorem",
                doc_url=_DOC_URL,
            )
        source_arr = np.asarray(source, dtype=float)
        u_values = field.values()
        if source_arr.shape != u_values.shape:
            return RuleResult(
                rule_id=__rule_id__,
                rule_name=__rule_name__,
                severity=__default_severity__,
                status="SKIPPED",
                raw_value=None,
                violation_ratio=None,
                mode=None,
                reason=(
                    f"PH-BC-002 Poisson source shape {source_arr.shape} does "
                    f"not match field shape {u_values.shape}"
                ),
                refinement_rate=None,
                spatial_map=None,
                recommended_norm="",
                citation="classical divergence theorem",
                doc_url=_DOC_URL,
            )
        lap = field.laplacian().values()
        lap_integral = trapezoidal_integral(lap, field.h)
        source_integral = trapezoidal_integral(source_arr, field.h)
        # -Delta u = f  =>  integral(Delta u) = -integral(f)
        #             =>  imbalance = integral(Delta u) + integral(f).
        imbalance = float(lap_integral + source_integral)
        scale = max(l2_grid(u_values, field.h), 1e-12)
        ratio = abs(imbalance) / scale
        status = "PASS" if ratio < 0.01 else ("WARN" if ratio < 0.1 else "FAIL")
        return RuleResult(
            rule_id=__rule_id__,
            rule_name=__rule_name__,
            severity=__default_severity__,
            status=status,
            raw_value=imbalance,
            violation_ratio=ratio,
            mode=None,
            reason=(
                None
                if status == "PASS"
                else f"Poisson flux imbalance {imbalance:.2e} (ratio {ratio:.2f}; {status})"
            ),
            refinement_rate=None,
            spatial_map=None,
            recommended_norm="integral of Laplacian + source (divergence theorem, Poisson)",
            citation="classical divergence theorem",
            doc_url=_DOC_URL,
        )

    lap = field.laplacian().values()
    u_vol_integral_of_lap = trapezoidal_integral(lap, field.h)
    # Laplace: expected net boundary flux is 0 (f = 0).
    expected = 0.0
    imbalance = float(u_vol_integral_of_lap - expected)
    # Threshold is scale-dependent; compare against the field's L^2 norm.
    scale = max(l2_grid(field.values(), field.h), 1e-12)
    ratio = abs(imbalance) / scale
    status = "PASS" if ratio < 0.01 else ("WARN" if ratio < 0.1 else "FAIL")
    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=imbalance,
        violation_ratio=ratio,
        mode=None,
        reason=None,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="integral of Laplacian (divergence theorem)",
        citation="classical divergence theorem",
        doc_url=_DOC_URL,
    )
