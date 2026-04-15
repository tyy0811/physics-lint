"""PH-POS-002: Maximum principle violation for Laplace.

Under a well-posed Dirichlet problem for -Delta u = 0, min/max of u are
attained on the boundary. Violation indicates a spurious interior extremum.
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field
from physics_lint.report import RuleResult
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-POS-002"
__rule_name__ = "Maximum principle violation"
__default_severity__ = "error"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-POS-002"


def check(
    field: Field,
    spec: DomainSpec,
    *,
    boundary_values: np.ndarray,
) -> RuleResult:
    if spec.pde != "laplace":
        return RuleResult(
            rule_id=__rule_id__,
            rule_name=__rule_name__,
            severity=__default_severity__,
            status="SKIPPED",
            raw_value=None,
            violation_ratio=None,
            mode=None,
            reason=f"max principle applies to Laplace only; got {spec.pde}",
            refinement_rate=None,
            spatial_map=None,
            recommended_norm="",
            citation="maximum principle for harmonic functions",
            doc_url=_DOC_URL,
        )

    u = field.values()
    bmin = float(boundary_values.min())
    bmax = float(boundary_values.max())
    below = max(0.0, bmin - float(u.min()))
    above = max(0.0, float(u.max()) - bmax)
    overshoot = max(below, above)

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status="PASS" if overshoot <= 1e-10 else "FAIL",
        raw_value=overshoot,
        violation_ratio=None,
        mode=None,
        reason=(
            None
            if overshoot <= 1e-10
            else f"interior extremum beyond boundary extrema by {overshoot:.3e}"
        ),
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="interior extremum overshoot",
        citation="maximum principle for harmonic functions",
        doc_url=_DOC_URL,
    )
