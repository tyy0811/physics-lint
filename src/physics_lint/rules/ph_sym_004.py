"""PH-SYM-004: Translation equivariance (V1 structural stub).

**V1 scope:** this rule is a structural stub that always emits ``SKIPPED``
once past its declared/periodic gates. True translation equivariance is a
*model* property — comparing ``f(roll(x))`` against ``roll(f(x))`` on a
live callable — and requires adapter-mode plumbing that lands in V1.1.

The prior implementation measured the offline quantity
``||roll(u) - u|| / ||roll(u)||``, but ``np.roll`` preserves norm, so the
triangle inequality caps this quantity at 2.0. A PASS-if-<2.0 threshold
rubber-stamped random noise, smooth ramps, and most structured fields;
only a pathologically Nyquist-aligned checkerboard could reach WARN. The
false-pass was removed rather than shipping a metric that cannot fail on
realistic inputs.

The rule keeps its ``declared`` and ``periodic`` gates so the SKIP reasons
explain *which* precondition is missing (not declared, non-periodic, or
the V1-stub deferral).
"""

from __future__ import annotations

from physics_lint.field import Field
from physics_lint.report import RuleResult
from physics_lint.rules._symmetry_helpers import is_symmetry_declared
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-SYM-004"
__rule_name__ = "Translation equivariance violation"
__default_severity__ = "warning"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-SYM-004"


def check(field: Field, spec: DomainSpec) -> RuleResult:
    wants_x = is_symmetry_declared(spec.symmetries, "translation_x")
    wants_y = is_symmetry_declared(spec.symmetries, "translation_y")
    if not (wants_x or wants_y):
        return _skip("no translation_x or translation_y declared")
    if not spec.periodic:
        return _skip(
            "PH-SYM-004 is periodic-only in V1; non-periodic translation "
            "requires interpolation and is deferred to V2"
        )
    return _skip(
        "PH-SYM-004 is a V1 structural stub: true translation equivariance "
        "is a model property (compare f(roll(x)) to roll(f(x))) and requires "
        "adapter-mode plumbing that lands in V1.1. Offline field invariance "
        "is bounded above by 2.0 via triangle inequality and is not a "
        "meaningful check; it has been removed rather than fabricate a PASS."
    )


def _skip(reason: str) -> RuleResult:
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
        citation="",
        doc_url=_DOC_URL,
    )
