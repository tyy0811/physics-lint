"""PH-SYM-001: C4 rotation equivariance violation.

**V1 scope:** square grids only (``field.shape[0] == field.shape[1]``).
Non-square C4 requires bilinear interpolation and is deferred to V2.
``np.rot90`` is exact on square grids with no tolerance fudge, which
is why this rule's PASS threshold is tight (1e-10).
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field, GridField
from physics_lint.report import RuleResult
from physics_lint.rules._symmetry_helpers import equivariance_error_np, is_symmetry_declared
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-SYM-001"
__rule_name__ = "C4 rotation equivariance violation"
__default_severity__ = "warning"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-SYM-001"


def check(field: Field, spec: DomainSpec) -> RuleResult:
    if not is_symmetry_declared(spec.symmetries, "C4"):
        return RuleResult(
            rule_id=__rule_id__,
            rule_name=__rule_name__,
            severity=__default_severity__,
            status="SKIPPED",
            raw_value=None,
            violation_ratio=None,
            mode=None,
            reason="C4 not declared in SymmetrySpec",
            refinement_rate=None,
            spatial_map=None,
            recommended_norm="",
            citation="",
            doc_url=_DOC_URL,
        )
    if not isinstance(field, GridField):
        raise TypeError(f"PH-SYM-001 requires GridField; got {type(field).__name__}")

    u = field.values()
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError(f"PH-SYM-001 requires a square 2D field; got shape {u.shape}")

    errs = [equivariance_error_np(np.rot90(u, k=k), u) for k in (1, 2, 3)]
    max_err = max(errs)
    # Tri-state against a small fixed threshold (symmetry is exact on grid so
    # machine precision is the floor).
    if max_err < 1e-10:
        status = "PASS"
    elif max_err < 0.01:
        status = "WARN"
    else:
        status = "FAIL"

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=max_err,
        violation_ratio=max_err / 1e-10 if max_err > 0 else 0.0,
        mode=None,
        reason=None,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="max relative L^2 over k in {1, 2, 3}",
        citation="Helwig et al. 2023; design doc §9.4",
        doc_url=_DOC_URL,
    )
