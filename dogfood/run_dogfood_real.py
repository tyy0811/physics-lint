"""Real-model dogfood for physics-lint Criterion 3 (A1 scope).

See docs/superpowers/specs/2026-04-17-week-2.5-dogfood-a1-design.md.
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import GridField
from physics_lint.rules import ph_bc_001, ph_pos_002, ph_res_001
from physics_lint.spec import (
    BCSpec,
    DomainSpec,
    FieldSourceSpec,
    GridDomain,
    SymmetrySpec,
)

# Grid spacing: 64-point endpoint-inclusive unit grid → h = 1/63.
# Matches upstream metrics.py:17. Do NOT change — any other h breaks
# PH-RES-001 / pde_residual quadrature comparability.
H = (1.0 / 63.0, 1.0 / 63.0)


def build_a1_spec() -> DomainSpec:
    """DomainSpec for the Week 2½ A1 configuration.

    64x64 unit square, Laplace, non-homogeneous Dirichlet BCs.
    """
    return DomainSpec(
        pde="laplace",
        grid_shape=(64, 64),
        domain=GridDomain(x=(0.0, 1.0), y=(0.0, 1.0)),
        periodic=False,
        boundary_condition=BCSpec(kind="dirichlet"),
        symmetries=SymmetrySpec(declared=[]),
        field=FieldSourceSpec(type="grid", backend="fd", dump_path="unused"),
    )


def apply_rules_to_prediction(
    *,
    prediction: np.ndarray,
    truth: np.ndarray,
    spec: DomainSpec,
) -> dict[str, float]:
    """Run PH-RES-001, PH-BC-001, PH-POS-002 on one (64, 64) prediction.

    Args:
        prediction: (64, 64) float array — model output for one test problem.
        truth: (64, 64) float array — ground-truth solution; used to build
            boundary_target for PH-BC-001.
        spec: DomainSpec from build_a1_spec().

    Returns:
        Mapping rule_id -> raw_value. Each raw_value is a scalar per §6 of
        the design doc; NaN if the rule returned SKIPPED.
    """
    pred_field = GridField(prediction, h=H, periodic=False, backend="fd")
    truth_field = GridField(truth, h=H, periodic=False, backend="fd")
    boundary_target = truth_field.values_on_boundary()
    pred_boundary = pred_field.values_on_boundary()

    res_result = ph_res_001.check(pred_field, spec)
    bc_result = ph_bc_001.check(pred_field, spec, boundary_target=boundary_target)
    pos_result = ph_pos_002.check(pred_field, spec, boundary_values=pred_boundary)

    def _raw(rr):
        return float(rr.raw_value) if rr.raw_value is not None else float("nan")

    return {
        "PH-RES-001": _raw(res_result),
        "PH-BC-001": _raw(bc_result),
        "PH-POS-002": _raw(pos_result),
    }


def check_ordinal_axis(
    *,
    axis_name: str,
    upstream_ranking: list[str],
    physlint_scores: dict[str, dict[str, float]],
    rule_id: str,
) -> dict:
    """Ordinal ranking match between upstream ranking and physics-lint scores.

    Sorts models by physics-lint raw_value ascending (smallest = best per
    upstream's convention on these three metrics) and compares to upstream's
    ranking.
    """
    physlint_ranking = sorted(
        physlint_scores.keys(),
        key=lambda m: physlint_scores[m][rule_id],
    )
    return {
        "axis": axis_name,
        "mode": "ordinal",
        "upstream": list(upstream_ranking),
        "physlint": physlint_ranking,
        "match": physlint_ranking == list(upstream_ranking),
    }


def check_binary_axis(
    *,
    axis_name: str,
    expected_violators: set[str],
    physlint_scores: dict[str, dict[str, float]],
    rule_id: str,
    threshold: float,
) -> dict:
    """Binary-mode check: which models have raw_value > threshold?

    Used for PH-POS-002 where upstream's max_viol splits into {FNO violates,
    others don't}. Under the threshold-mismatch edge case (§6.5), physics-lint
    may report universal violation — this returns match=False and lets the
    caller record the finding in the report.
    """
    physlint_violators = {m for m, s in physlint_scores.items() if s[rule_id] > threshold}
    return {
        "axis": axis_name,
        "mode": "binary",
        "expected_violators": set(expected_violators),
        "physlint_violators": physlint_violators,
        "match": physlint_violators == set(expected_violators),
    }
