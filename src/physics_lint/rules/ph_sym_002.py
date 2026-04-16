"""PH-SYM-002: Reflection equivariance violation (axis-aligned, grid-exact).

**V1 scope:** reflection axes aligned with grid axes only —
``reflection_x`` and ``reflection_y``. Diagonal and arbitrary-axis
reflections deferred to V2 (they would require interpolation).
``np.flip`` is exact along grid axes, same reason as PH-SYM-001.
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field
from physics_lint.report import RuleResult
from physics_lint.rules._helpers import ensure_grid_field
from physics_lint.rules._symmetry_helpers import equivariance_error_np, is_symmetry_declared
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-SYM-002"
__rule_name__ = "Reflection equivariance violation"
__default_severity__ = "warning"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-SYM-002"


def check(field: Field, spec: DomainSpec) -> RuleResult:
    wants_x = is_symmetry_declared(spec.symmetries, "reflection_x")
    wants_y = is_symmetry_declared(spec.symmetries, "reflection_y")
    if not (wants_x or wants_y):
        return _skip("no reflection_x or reflection_y declared")
    field = ensure_grid_field(field, spec)

    u = field.values()
    if u.ndim != 2:
        raise ValueError(f"PH-SYM-002 requires 2D field; got shape {u.shape}")

    errs: list[float] = []
    if wants_x:
        errs.append(equivariance_error_np(np.flip(u, axis=0), u))
    if wants_y:
        errs.append(equivariance_error_np(np.flip(u, axis=1), u))
    max_err = max(errs)

    if max_err <= 1e-10:
        status = "PASS"
    elif max_err <= 0.01:
        status = "WARN"
    else:
        status = "FAIL"

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=max_err,
        # TODO(task-8): switch to _load_floor per invariant 2. See SYM-001.
        violation_ratio=max_err / 0.01,
        mode=None,
        reason=(
            None
            if status == "PASS"
            else f"max reflection equivariance error {max_err:.2e} exceeds "
            f"{'WARN threshold 1e-10' if status == 'WARN' else 'FAIL threshold 1e-2'}"
        ),
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="max relative L^2 over declared reflection axes",
        citation="Helwig et al. 2023; design doc §9.2",
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
