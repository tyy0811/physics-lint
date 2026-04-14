"""Placeholder rule used by the Week 1 Day 3 registry tests.

Deleted in Task 10 once the first real rule (PH-RES-001) lands. Kept here
only so the registry has something to discover before any real rule exists.
"""

__rule_id__ = "PH-PLACEHOLDER-000"
__rule_name__ = "Placeholder (Week 1 Day 3 only)"
__default_severity__ = "info"
__input_modes__ = frozenset({"adapter", "dump"})


def check(field, spec):
    from physics_lint.report import RuleResult

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status="PASS",
        raw_value=0.0,
        violation_ratio=0.0,
        mode=None,
        reason=None,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="",
        citation="",
        doc_url="",
    )
