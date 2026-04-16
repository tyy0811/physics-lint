"""PH-NUM-001: Quadrature convergence warning (mesh only).

**V1 structural stub.** ``MeshField.integrate`` does not expose a ``qorder``
kwarg in V1, so this rule cannot compare quadrature at orders ``q`` and ``2q``.
It ships as a *structural stub*: the rule module exists, the rule ID is in the
registry, and the CLI surface is stable. V1.1 can plug in the real q-vs-2q check
without breaking any public API. See ``docs/backlog/v1.1.md`` for the backlog
item.

In V1 the rule emits ``PASS`` with a ``reason`` string that says
``'qorder convergence check is a stub until V1.1'`` — it does not fabricate a
convergence claim. The ``raw_value`` is the baseline integral from
``field.integrate()`` (a scalar) so the rule output has a non-None ``raw_value``
that downstream tooling can display, even though it is not a convergence metric.
"""

from __future__ import annotations

from physics_lint.field import Field
from physics_lint.report import RuleResult
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-NUM-001"
__rule_name__ = "Quadrature convergence warning"
__default_severity__ = "warning"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-NUM-001"


def check(field: Field, spec: DomainSpec) -> RuleResult:
    del spec

    try:
        from physics_lint.field import MeshField
    except ImportError:
        return _skipped("PH-NUM-001 requires MeshField")
    if MeshField is None or not isinstance(field, MeshField):
        return _skipped("PH-NUM-001 requires MeshField")

    baseline = field.integrate()

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status="PASS",
        raw_value=baseline,
        violation_ratio=None,
        mode=None,
        reason="qorder convergence check is a stub until V1.1",
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="",
        citation="design doc §8.3",
        doc_url=_DOC_URL,
    )


def _skipped(reason: str) -> RuleResult:
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
