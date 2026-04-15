"""PH-SYM-004: Translation equivariance (periodic domains only in V1).

**V1 scope:** periodic domains only (``field.periodic == True``).
Non-periodic translation would require zero-padding or other boundary
handling and is deferred to V2. On offline fields this rule degrades to
a "shifted field vs original" sanity bound — the discrete L^2 difference
of ``np.roll(u)`` and ``u``, normalized by ``||u||``. For smooth periodic
sines this is O(shift/N), which sits comfortably below the V1 sanity
threshold of 2.0.

True model-based translation equivariance (comparing ``f(roll(x))`` to
``roll(f(x))`` on a live callable model) is deferred to V1.1 when
``CallableField`` is plumbed through this rule.
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field, GridField
from physics_lint.report import RuleResult
from physics_lint.rules._symmetry_helpers import equivariance_error_np, is_symmetry_declared
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-SYM-004"
__rule_name__ = "Translation equivariance violation"
__default_severity__ = "warning"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-SYM-004"

_SANITY_THRESHOLD = 2.0


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
    if not isinstance(field, GridField):
        raise TypeError(f"PH-SYM-004 requires GridField; got {type(field).__name__}")

    u = field.values()
    if u.ndim != 2:
        raise ValueError(f"PH-SYM-004 requires 2D field; got shape {u.shape}")

    errs: list[float] = []
    if wants_x:
        for shift in (1, u.shape[0] // 4):
            errs.append(equivariance_error_np(np.roll(u, shift=shift, axis=0), u))
    if wants_y:
        for shift in (1, u.shape[1] // 4):
            errs.append(equivariance_error_np(np.roll(u, shift=shift, axis=1), u))

    max_err = max(errs) if errs else 0.0

    # Offline-field sanity bound. For smooth periodic sines, roll(u) - u is
    # O(shift / N) so max_err < 2.0 easily. A raw_value near or above 2.0
    # on an offline field means the stored tensor is pathologically
    # structured (roll doubles its norm) — warn the user.
    status = "PASS" if max_err < _SANITY_THRESHOLD else "WARN"

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=max_err,
        violation_ratio=max_err / _SANITY_THRESHOLD,
        mode=None,
        reason=(
            "offline-field sanity check; model-output translation equivariance "
            "via CallableField lands in V1.1"
            if status == "PASS"
            else f"offline-field sanity bound {_SANITY_THRESHOLD} exceeded "
            f"(max_err {max_err:.2e}); field may be pathologically structured"
        ),
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="max relative L^2 of shifted field vs original",
        citation="design doc §9.2 periodic translation",
        doc_url=_DOC_URL,
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
