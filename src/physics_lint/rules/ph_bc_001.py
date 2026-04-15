"""PH-BC-001: Boundary condition violation with mode-branched normalization.

Design doc §8.5.

If ``||g|| < abs_threshold``: absolute mode (binary PASS/FAIL against
``abs_tol_fail``). Otherwise: relative mode (tri-state against the
calibrated relative floor). The binary absolute mode is the Rev 4.1 fix
for the homogeneous-Dirichlet footgun: dividing ``||u - g||`` by a
near-zero ``||g||`` explodes the ratio and produces spurious FAIL on
correct solutions.
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field
from physics_lint.report import RuleResult
from physics_lint.rules._helpers import _load_floor, _tristate
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-BC-001"
__rule_name__ = "Boundary condition violation (relative or absolute mode)"
__default_severity__ = "error"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-BC-001"

_DEFAULT_ABS_THRESHOLD = 1e-10
_DEFAULT_ABS_TOL_FAIL = 1e-3


def check(
    field: Field,
    spec: DomainSpec,
    *,
    boundary_target: np.ndarray,
    abs_threshold: float = _DEFAULT_ABS_THRESHOLD,
    abs_tol_fail: float = _DEFAULT_ABS_TOL_FAIL,
) -> RuleResult:
    """Compute ||u - g|| on the boundary; mode-branch on ||g||."""
    u_boundary = field.values_on_boundary()
    if u_boundary.shape != boundary_target.shape:
        raise ValueError(
            f"boundary_target shape {boundary_target.shape} does not match "
            f"field.values_on_boundary() shape {u_boundary.shape}"
        )
    err_values = u_boundary - boundary_target
    # Discrete L^2 on the boundary trace: points are ordered, so divide by
    # sqrt(len) to make the norm independent of boundary sample density.
    err_norm = float(np.linalg.norm(err_values) / np.sqrt(max(len(err_values), 1)))
    g_norm = float(np.linalg.norm(boundary_target) / np.sqrt(max(len(boundary_target), 1)))

    if g_norm < abs_threshold:
        status = "PASS" if err_norm < abs_tol_fail else "FAIL"
        return RuleResult(
            rule_id=__rule_id__,
            rule_name=__rule_name__,
            severity=__default_severity__,
            status=status,
            raw_value=err_norm,
            violation_ratio=None,
            mode="absolute",
            reason=None,
            refinement_rate=None,
            spatial_map=None,
            recommended_norm="boundary L2 (absolute)",
            citation="design doc §8.5",
            doc_url=_DOC_URL,
        )

    relative_value = err_norm / g_norm
    floor = _load_floor(
        rule="PH-BC-001",
        pde=spec.pde,
        grid_shape=spec.grid_shape,
        method="fd4",
        norm="L2-rel",
    )
    ratio = relative_value / floor.value if floor.value > 0 else float("inf")
    status = _tristate(ratio, pass_=floor.tolerance * 10, fail_=floor.tolerance * 100)

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,
        raw_value=relative_value,
        violation_ratio=ratio,
        mode="relative",
        reason=None,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm="boundary L2 (relative)",
        citation="design doc §8.5",
        doc_url=_DOC_URL,
    )
