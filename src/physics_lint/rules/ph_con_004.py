"""PH-CON-004: Per-element conservation hotspot (mesh-only, interior-only).

Design doc §8.2 weak-form conservation. For a MeshField ``u`` this rule
computes the elemental residual ``∫_K (Δu)² dx`` where ``Δu`` is taken
from ``MeshField.laplacian_l2_projected_zero_trace()`` — the V1
finite-element operator, *not* a pointwise Laplacian (see
``src/physics_lint/field/mesh.py`` module docstring for the operator's
semantics and ``docs/tradeoffs.md`` entry 2026-04-15 for the rename
rationale). The rule reports the hotspot indicator
``max_elem / mean_elem``. A ratio ≲ 1 means residuals are evenly
distributed; a large ratio indicates one or a few elements dominate the
residual.

Threshold (dimensionless; no floors.toml entry is needed — floors are
for calibrated absolute tolerances):

- ``max_elem / mean_elem ≤ 10``  → PASS
- ``max_elem / mean_elem > 10``  → WARN

**Interior-only by construction.** The V1 zero-trace projection pins
boundary DOFs to 0 as a hard Dirichlet condition on the projection, so
elements that touch the domain boundary carry a residual that reflects
the operator artifact rather than any property of the input field. This
rule excludes boundary-touching elements **structurally** (not via a
docstring caveat): the interior mask removes every element that owns at
least one boundary facet (``mesh.f2t[1] == -1``) before ``max_elem`` and
``mean_elem`` are computed. Downstream callers therefore never see a
value derived from a zero-pinned boundary element. V1.1 may add coverage
of the boundary strip if ``MeshField`` grows a non-zero-trace Laplacian
operator; see ``docs/tradeoffs.md`` entry 2026-04-15.

Skip paths:

- ``field`` is not a ``MeshField`` (GridField / CallableField cleanly
  SKIP; no type-error raises).
- ``skfem`` is not importable at all.
- The interior element set is empty (very coarse mesh where every
  element touches the boundary).
- ``mean_elem`` is numerically zero (constant-ish field where the
  residual is dominated by roundoff — reporting a ratio from that would
  be spurious).
"""

from __future__ import annotations

import numpy as np

from physics_lint.field import Field
from physics_lint.report import RuleResult
from physics_lint.spec import DomainSpec

__rule_id__ = "PH-CON-004"
__rule_name__ = "Per-element conservation hotspot"
__default_severity__ = "warning"
__input_modes__ = frozenset({"adapter", "dump"})

_DOC_URL = "https://physics-lint.readthedocs.io/rules/PH-CON-004"
_CITATION = "design doc §8.2 weak-form conservation"
_RECOMMENDED_NORM = "max elemental residual sq / mean, interior elements only"

# WARN threshold on the dimensionless max/mean ratio. See module docstring.
_HOTSPOT_RATIO_WARN = 10.0
# Below this absolute mean, the residual is numerically zero and the
# ratio is meaningless (constant-ish input). Prefer SKIP over a spurious
# WARN driven by roundoff.
_MEAN_ELEM_ZERO_TOL = 1e-20


def check(field: Field, spec: DomainSpec) -> RuleResult:
    del spec  # unused: rule is purely a property of the FE-backed field

    # Gate 1: MeshField must be importable *and* the field must be one.
    try:
        from physics_lint.field import MeshField
    except ImportError:
        return _skipped("PH-CON-004 requires MeshField (scikit-fem extra)")
    if MeshField is None or not isinstance(field, MeshField):
        return _skipped("PH-CON-004 requires MeshField (scikit-fem extra)")

    # Lazy skfem import — safe because MeshField itself imported OK above.
    from skfem import Functional

    # Pull the V1 zero-trace-projected Laplacian DOFs. NOT the stub
    # `.laplacian()` which raises in V1.
    lap_field = field.laplacian_l2_projected_zero_trace()
    lap_dofs = lap_field.values()

    basis = field._basis  # private access is fine inside the package
    mesh = basis.mesh

    # Elemental residual ∫_K (Δu)² dx, one value per element.
    @Functional
    def residual_sq(w):  # type: ignore[no-untyped-def]
        return w["lap"] ** 2

    elem_res = residual_sq.elemental(basis, lap=basis.interpolate(lap_dofs))
    elem_res = np.asarray(elem_res)

    # Structural interior-element mask. A boundary facet has f2t[1] == -1
    # in skfem; the element owning that facet is a boundary-touching
    # element and its residual is contaminated by the zero-trace artifact.
    f2t = np.asarray(mesh.f2t)
    boundary_facet_mask = f2t[1] == -1
    boundary_element_indices = np.unique(f2t[0, boundary_facet_mask])
    interior_element_mask = np.ones(mesh.nelements, dtype=bool)
    interior_element_mask[boundary_element_indices] = False
    n_interior = int(interior_element_mask.sum())

    if n_interior == 0:
        return _skipped(
            "PH-CON-004 requires at least one interior element; got none on "
            "this mesh (try refining)"
        )

    interior_res = elem_res[interior_element_mask]
    max_elem = float(np.max(interior_res))
    mean_elem = float(np.mean(interior_res))

    if mean_elem < _MEAN_ELEM_ZERO_TOL:
        return _skipped(
            f"PH-CON-004: per-element residual is numerically zero "
            f"(mean={mean_elem:.2e} over {n_interior} interior elements); "
            f"hotspot detection not meaningful for an essentially constant field"
        )

    ratio = max_elem / mean_elem
    # Normalize to the WARN convention used by PH-SYM-001 etc.: > 1 means
    # violation. Threshold is _HOTSPOT_RATIO_WARN on ``ratio``, so the
    # normalized violation_ratio divides by that threshold.
    violation_ratio = ratio / _HOTSPOT_RATIO_WARN

    if ratio <= _HOTSPOT_RATIO_WARN:
        status: str = "PASS"
        reason: str | None = None
    else:
        status = "WARN"
        reason = (
            f"per-element hotspot ratio {ratio:.2f} exceeds {_HOTSPOT_RATIO_WARN:.0f} "
            f"(max elem residual sq {max_elem:.2e}, mean {mean_elem:.2e}, "
            f"over {n_interior} interior elements)"
        )

    return RuleResult(
        rule_id=__rule_id__,
        rule_name=__rule_name__,
        severity=__default_severity__,
        status=status,  # type: ignore[arg-type]
        raw_value=ratio,
        violation_ratio=violation_ratio,
        mode=None,
        reason=reason,
        refinement_rate=None,
        spatial_map=None,
        recommended_norm=_RECOMMENDED_NORM,
        citation=_CITATION,
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
        recommended_norm=_RECOMMENDED_NORM,
        citation=_CITATION,
        doc_url=_DOC_URL,
    )
