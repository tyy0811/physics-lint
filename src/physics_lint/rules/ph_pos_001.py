"""PH-POS-001: Positivity violation.

Applies when the BC preserves sign (read via spec.boundary_condition.preserves_sign).
Otherwise emits SKIPPED.
"""

from __future__ import annotations

from physics_lint.field import Field
from physics_lint.report import RuleResult
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-POS-001"
__rule_name__ = "Positivity violation"
__default_severity__ = "error"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-POS-001"


def check(field: Field, spec: DomainSpec, *, floor: float = 0.0) -> RuleResult:
    if not spec.boundary_condition.preserves_sign:
        return RuleResult(
            rule_id=__rule_id__,
            rule_name=__rule_name__,
            severity=__default_severity__,
            status="SKIPPED",
            raw_value=None,
            violation_ratio=None,
            mode=None,
            reason=(
                f"Configured BC '{spec.boundary_condition.kind}' does not preserve sign; "
                "PH-POS-001 does not apply"
            ),
            refinement_rate=None,
            spatial_map=None,
            recommended_norm="",
            citation="heat eigenfunction decay; Poisson positivity under hD with f >= 0",
            doc_url=_DOC_URL,
        )

    u = field.values()
    min_val = float(u.min())
    violation_map = u < floor
    n_violations = int(violation_map.sum())
    violation_fraction = float(violation_map.mean())

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status="PASS" if n_violations == 0 else "FAIL",
        raw_value=min_val,
        violation_ratio=violation_fraction if n_violations > 0 else 0.0,
        mode=None,
        reason=(
            None
            if n_violations == 0
            else f"{n_violations} cells below {floor} (fraction {violation_fraction:.3f})"
        ),
        refinement_rate=None,
        spatial_map=violation_map,
        recommended_norm="min pointwise value",
        citation="design doc §8.6",
        doc_url=_DOC_URL,
    )
