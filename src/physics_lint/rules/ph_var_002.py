"""PH-VAR-002: Hyperbolic norm-equivalence conjectural caveat.

Info-level rule that always fires (WARN) on wave PDEs and is SKIPPED
elsewhere. The Bachmayr-Ernst variational-correctness framework is
parabolic; applying it to hyperbolic problems is a conjecture, not a
theorem. physics-lint still measures a wave residual in Bochner
L^2(0,T; H^-1), but users need to know that "PASS" for wave means
"within the conjectural tolerance", not "certified by theory".

The rule is diagnostic; it doesn't compute anything against the field.
"""

from __future__ import annotations

from physics_lint.field import Field
from physics_lint.report import RuleResult
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-VAR-002"
__rule_name__ = "Hyperbolic norm-equivalence conjectural"
__default_severity__ = "info"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-VAR-002"

_MESSAGE = (
    "Hyperbolic norm-equivalence is not established within the parabolic "
    "Bachmayr-Ernst variational framework; treat the wave residual as "
    "diagnostic, not certification."
)


def check(field: Field, spec: DomainSpec) -> RuleResult:
    if spec.pde != "wave":
        return RuleResult(
            rule_id=__rule_id__,
            rule_name=__rule_name__,
            severity=__default_severity__,
            status="SKIPPED",
            raw_value=None,
            violation_ratio=None,
            mode=None,
            reason=f"PH-VAR-002 applies to wave only; got {spec.pde}",
            refinement_rate=None,
            spatial_map=None,
            recommended_norm="",
            citation="design doc §1.4 + §7.5",
            doc_url=_DOC_URL,
        )
    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status="WARN",
        raw_value=None,
        violation_ratio=None,
        mode=None,
        reason=_MESSAGE,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="Bochner-H-1 (conjectural)",
        citation="design doc §1.4",
        doc_url=_DOC_URL,
    )
