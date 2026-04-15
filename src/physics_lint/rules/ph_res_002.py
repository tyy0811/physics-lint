"""PH-RES-002: FD-vs-AD residual cross-check.

Adapter-only — requires an AD-capable model. Dump mode emits
SKIPPED with an explicit reason string.

Discrepancy formula (design doc §7.2):
    |R_FD - R_AD| / max(|R_FD|, |R_AD|, epsilon_floor)

Default threshold 0.01. Above: status = "WARN" (severity "warning").
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import CallableField, Field, GridField
from physics_lint.report import RuleResult
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-RES-002"
__rule_name__ = "FD-vs-AD residual cross-check discrepancy"
__default_severity__ = "warning"
__input_modes__ = frozenset({"adapter"})  # dump emits SKIPPED

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-RES-002"


def check(field: Field, spec: DomainSpec) -> RuleResult:
    if not isinstance(field, CallableField):
        return RuleResult(
            rule_id=__rule_id__,
            rule_name=__rule_name__,
            severity=__default_severity__,
            status="SKIPPED",
            raw_value=None,
            violation_ratio=None,
            mode=None,
            reason=(
                "FD-vs-AD cross-check requires a callable model; dump mode "
                "provides only a frozen tensor"
            ),
            refinement_rate=None,
            spatial_map=None,
            recommended_norm="",
            citation="PhysicsNeMo Sym multi-backend pattern",
            doc_url=_DOC_URL,
        )

    # AD path: Laplacian via torch autograd (already done in CallableField.laplacian)
    lap_ad = field.laplacian().values()

    # FD path: materialize the field values on the grid, wrap in a plain GridField
    # with backend="fd", and compute its Laplacian via the FD stencil.
    vals = field.values()
    h = field.h
    fd_field = GridField(vals, h=h, periodic=spec.periodic, backend="fd")
    lap_fd = fd_field.laplacian().values()

    epsilon = 1e-12
    denom = np.maximum(np.maximum(np.abs(lap_fd), np.abs(lap_ad)), epsilon)
    discrepancy_map = np.abs(lap_fd - lap_ad) / denom
    # Interior-only comparison (exclude the outer 2 layers where FD is 2nd-order)
    if vals.ndim == 2:
        interior = discrepancy_map[2:-2, 2:-2]
    elif vals.ndim == 3:
        interior = discrepancy_map[2:-2, 2:-2, 2:-2]
    else:
        interior = discrepancy_map
    max_discrepancy = float(np.max(interior)) if interior.size > 0 else 0.0

    status = "PASS" if max_discrepancy < 0.01 else "WARN"
    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=max_discrepancy,
        violation_ratio=max_discrepancy / 0.01 if max_discrepancy > 0 else 0.0,
        mode=None,
        reason=None,
        refinement_rate=None,
        spatial_map=discrepancy_map,
        recommended_norm="max discrepancy ratio",
        citation="PhysicsNeMo Sym multi-backend pattern",
        doc_url=_DOC_URL,
    )
