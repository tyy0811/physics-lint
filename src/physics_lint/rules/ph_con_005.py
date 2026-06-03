"""PH-CON-005: Incompressibility (divergence-free) defect — graph-mesh velocity.

Emit-only diagnostic. For an incompressible flow the mass-conservation identity
is ∇·v = 0; this rule reports the dimensionless Frobenius-normalized defect

    defect = max over t of  ( ∫|∇·v| dV ) / ( ∫‖∇v‖_F dV )

via scikit-fem P1 finite elements (ports the validated harness
`_fe_divergence_defect_max`). It does NOT gate: a discrete velocity field carries
an O(h) divergence defect (CS02 ground truth itself ≈ 5.8%), so an absolute
PASS/FAIL band would fail correct data. The rule emits status=PASS,
severity=info, raw_value=defect (the PH-VAR-002 emit-only pattern) — it never
fails the run or degrades overall_status. The signal is the scalar (and a
ground-truth-vs-model comparison the user runs), not a verdict.
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field
from physics_lint.report import RuleResult
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-CON-005"
__rule_name__ = "Incompressibility (divergence-free) defect"
__default_severity__ = "info"
__input_modes__ = frozenset(
    {"dump"}
)  # mesh_vector is dump-only in v1.2.0 (adapter mode is a follow-on)
__field_types__ = frozenset({"mesh_vector"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-CON-005"
_CITATION = "incompressible NS mass conservation (div v = 0); design doc §3"
_RECOMMENDED_NORM = "max_t integral|div v| / integral||grad v||_F (Frobenius-normalized)"


def check(field: Field, spec: DomainSpec) -> RuleResult:
    del spec  # rule is purely a property of the velocity-on-mesh field
    try:
        from physics_lint.field import MeshVectorField
    except ImportError:
        return _skipped("PH-CON-005 requires MeshVectorField (scikit-fem extra)")
    if MeshVectorField is None or not isinstance(field, MeshVectorField):
        return _skipped(
            f"PH-CON-005 requires a MeshVectorField (graph-mesh velocity); "
            f"got {type(field).__name__}"
        )

    basis = field.basis
    velocity = field.values()  # (T, N, 2)
    defect = _fe_divergence_defect_max(basis, velocity)

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status="PASS",
        raw_value=defect,
        violation_ratio=None,
        mode=None,
        reason=(
            f"incompressibility defect {defect:.3e} "
            f"(max_t integral|div v| / integral||grad v||_F); diagnostic, "
            f"not gated — compare against your reference (a discrete GT carries "
            f"an O(h) defect of its own)"
        ),
        refinement_rate=None,
        spatial_map=None,
        recommended_norm=_RECOMMENDED_NORM,
        citation=_CITATION,
        doc_url=_DOC_URL,
    )


def _fe_divergence_defect_max(basis, velocity: np.ndarray) -> float:  # type: ignore[no-untyped-def]
    """Ports the harness `_fe_divergence_defect_max` (mesh_rollout_adapter.py)."""
    n_t = velocity.shape[0]
    max_rel = 0.0
    for t_idx in range(n_t):
        v_t = velocity[t_idx].astype(np.float64)
        vx_f = basis.interpolate(v_t[:, 0])
        vy_f = basis.interpolate(v_t[:, 1])
        gvx = vx_f.grad  # (2, n_elem, n_qp): [dvx/dx, dvx/dy]
        gvy = vy_f.grad  # [dvy/dx, dvy/dy]
        div = gvx[0] + gvy[1]
        frob = np.sqrt(gvx[0] ** 2 + gvx[1] ** 2 + gvy[0] ** 2 + gvy[1] ** 2)
        int_abs_div = float(np.sum(np.abs(div) * basis.dx))
        int_frob = float(np.sum(frob * basis.dx))
        rel = int_abs_div / max(int_frob, 1e-12)
        if rel > max_rel:
            max_rel = rel
    return max_rel


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
        recommended_norm=_RECOMMENDED_NORM,
        citation=_CITATION,
        doc_url=_DOC_URL,
    )
