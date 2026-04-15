"""PH-BC-002: Boundary flux imbalance (divergence theorem).

For Laplace/Poisson: the integral of ``-Delta u`` over the domain equals
the net outward boundary flux integral. Violation of this identity is a
sign that the learned field is inconsistent with the PDE at a weak-form
level even if the pointwise residual is small.

Week 1 scope: Laplace only (expected imbalance is zero). The Poisson arm
raises ``NotImplementedError`` to surface the unfinished wiring loudly
instead of silently computing a wrong answer; the source-term integral
lands in Week 2 when the loader plumbs ``spec.source_term``.
"""

from __future__ import annotations

from physics_lint.field import Field, GridField
from physics_lint.norms import l2_grid, trapezoidal_integral
from physics_lint.report import RuleResult
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
    if not isinstance(field, GridField):
        raise TypeError(f"PH-BC-002 requires GridField; got {type(field).__name__}")

    if spec.pde == "poisson":
        # Week 1 scope: source term is not yet plumbed through DomainSpec.
        # Emit SKIPPED rather than raising — the linter must not crash on a
        # valid spec. Source integration (and thus a real imbalance) lands
        # in Week 2.
        return RuleResult(
            rule_id=__rule_id__,
            rule_name=__rule_name__,
            severity=__default_severity__,
            status="SKIPPED",
            raw_value=None,
            violation_ratio=None,
            mode=None,
            reason="PH-BC-002 for Poisson requires source integration; lands in Week 2.",
            refinement_rate=None,
            spatial_map=None,
            recommended_norm="",
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
