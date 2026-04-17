"""Real-model dogfood for physics-lint Criterion 3 (A1 scope).

See docs/superpowers/specs/2026-04-17-week-2.5-dogfood-a1-design.md.
"""

from __future__ import annotations

from pathlib import Path

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


def load_predictions(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load predictions and truth from a .npz dumped by _extract_predictions.py.

    Returns:
        (predictions, truth) each shaped (N, 64, 64) float32.
    """
    data = np.load(npz_path)
    predictions = data["predictions"].astype(np.float32)
    truth = data["truth"].astype(np.float32)
    return predictions, truth


def aggregate_over_problems(
    *,
    predictions: np.ndarray,
    truth: np.ndarray,
    spec: DomainSpec,
) -> dict[str, float]:
    """Apply the three rules per problem and return per-rule mean raw_value.

    SKIPPED rules contribute NaN; the mean treats NaN as missing (nanmean).
    Ranking downstream breaks ties deterministically by model-name sort.
    """
    assert predictions.shape == truth.shape
    assert predictions.shape[1:] == (64, 64)

    per_problem = []
    for i in range(predictions.shape[0]):
        scores = apply_rules_to_prediction(
            prediction=predictions[i],
            truth=truth[i],
            spec=spec,
        )
        per_problem.append(scores)

    rule_ids = ["PH-RES-001", "PH-BC-001", "PH-POS-002"]
    return {rid: float(np.nanmean([p[rid] for p in per_problem])) for rid in rule_ids}


def compute_verdict(*, sanity_match: bool, real_axis_matches: list[bool]) -> str:
    """Four-way Criterion 3 verdict per §7 of the design doc.

    Ordering of branches matters: sanity failure dominates (BUG) even if
    real axes match, because L2-vs-L2 disagreement indicates a discretization
    bug to fix, not a Criterion 3 deferral.
    """
    if not sanity_match:
        return "BUG"
    real_passes = sum(1 for m in real_axis_matches if m)
    if real_passes == len(real_axis_matches):
        return "PASS (scoped)"
    if real_passes >= 1:
        return "PASS (scoped, MIXED)"
    return "FAIL"
