"""PH-RES-003: Spectral-vs-FD residual cross-check on periodic grids."""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field, GridField
from physics_lint.report import RuleResult
from physics_lint.rules._helpers import ensure_grid_field
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-RES-003"
__rule_name__ = "Spectral-vs-FD residual discrepancy on periodic grid"
__default_severity__ = "warning"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-RES-003"


def check(field: Field, spec: DomainSpec) -> RuleResult:
    if not spec.periodic:
        return RuleResult(
            rule_id=__rule_id__,
            rule_name=__rule_name__,
            severity=__default_severity__,
            status="SKIPPED",
            raw_value=None,
            violation_ratio=None,
            mode=None,
            reason="PH-RES-003 applies only to periodic domains",
            refinement_rate=None,
            spatial_map=None,
            recommended_norm="",
            citation="Trefethen 2000; Fornberg 1988",
            doc_url=_DOC_URL,
        )
    # Accept both dump (GridField) and adapter (CallableField) inputs;
    # ensure_grid_field materializes the callable so we can rebuild two
    # backend-specific GridFields over the same values.
    field = ensure_grid_field(field, spec)

    vals = field.values()
    spectral_f = GridField(vals, h=field.h, periodic=True, backend="spectral")
    fd_f = GridField(vals, h=field.h, periodic=True, backend="fd")
    lap_spectral = spectral_f.laplacian().values()
    lap_fd = fd_f.laplacian().values()
    diff = lap_spectral - lap_fd
    denom = float(np.max(np.abs(lap_spectral))) or 1.0
    max_rel = float(np.max(np.abs(diff))) / denom

    status = "PASS" if max_rel < 0.01 else "WARN"
    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=max_rel,
        violation_ratio=max_rel / 0.01 if max_rel > 0 else 0.0,
        mode=None,
        reason=None,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="max relative difference",
        citation="Trefethen 2000; Fornberg 1988",
        doc_url=_DOC_URL,
    )
