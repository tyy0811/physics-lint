"""PH-CON-004 external-validation anchor — per-element conservation hotspot.

Task 10 of the complete-v1.0 plan. PH-CON-004 is a CLEAR-disposition task
per the 2026-04-24 forward-look precheck, but the pre-execution audit
surfaced two substantive scope refinements that both required
user-approved revisions to the Task 10 contract (2026-04-24):

  1. The rule is a *narrower variant* of the Verfürth residual estimator
     — interior volumetric ``||Delta_{L2-proj zero-trace} u||^2`` only,
     no facet-jump term, no source term. F1 cannot claim a guaranteed
     error bound; it claims conservation-defect localization.

  2. The canonical L-shape exact solution u = r^(2/3) sin((2/3) theta)
     is NOT zero on the outer L-shape boundary; the rule's zero-trace
     projection dominates the corner signal with boundary artifacts.
     Replaced with a Gaussian-bump fixture ``u = exp(-20 r^2)`` at the
     re-entrant corner, which vanishes on the boundary at float precision
     and produces clean corner localization (max(r_top5)/h = 1.70 across
     refinements).

F2 splits into two scoped layers, following the Task 9 / Task 12
harness-authoritative + rule-verdict precedent:

  Case A (F2 harness-level, authoritative localization): at n_refine
      in {2, 3, 4}, top-5 hotspot elements on the Gaussian-bump fixture
      land within 2 element-layers of the re-entrant corner. Measured
      max(r_top5)/h = 1.70 +- 0.01 across refinements -- refinement-
      invariant in layer units.

  Case B (rule-verdict contract): rule PASSes on smooth-bubble fixtures
      (ratios 2.17, 3.65, 4.47 at n_refine 2, 3, 4); WARNs on Gaussian-
      bump fixtures (ratios 8.66, 10.85, 75.06) with monotonic growth
      and transition PASS -> WARN across threshold 10 at n_refine 3.

Wording discipline (CITATION.md + README + tests): PH-CON-004 validates
a v1.0 2D mesh-based residual-indicator contract. It does not claim
full adaptive finite-element solver validation or 3D tetrahedral
coverage.

Three-function-labeled stack per complete-v1.0 plan section 1.3:

    F1  Classical residual-based a-posteriori error estimation theory
        (Verfurth 2013 Chs 1-4 chapter-level WARN; Bangerth-Rannacher
        2003 chapter-level WARN; Ainsworth-Oden 2000 chapter-level
        WARN). Five-step proof-sketch with explicit no-guaranteed-
        error-bound claim.

    F2  Two-layer correctness fixture:
        - Case A (harness authoritative): L-shape mesh +
          Gaussian-bump fixture; top-k hotspot localization within
          2 element-layers of re-entrant corner; refinement-invariant
          in layer units.
        - Case B (rule-verdict contract): rule PASSes smooth, WARNs
          singular with monotonic ratio growth.

    F3  Absent with justification. No single-paper reproduction target
        exists for the L-shape effectivity-index benchmark (varies
        across estimator + marker + solver triples per Task 0 Step 5
        F3-hunt). scikit-fem Example 22 is not pip-importable from the
        installed package; L-shape fixture pattern is locally
        reimplemented in _harness/aposteriori.py. No F3-INFRA-GAP
        (F3-absent is structural, not a loader gap). Per 2026-04-24
        user-revised contract: "If scikit-fem Example 22 or related is
        implemented locally and CI-runnable, keep as F3; else demote
        to Supplementary or absent-with-justification." Demoted to
        absent-with-justification.

Rule-verdict contract:
    Rule emits max_K / mean_K of ``integral_K (Delta u)^2 dx`` over
    interior elements (ph_con_004.py:107-164). Threshold 10 -> PASS /
    WARN. Interior mask is DOF-aware (ph_con_004.py:121-126); excludes
    elements with any boundary DOF. Scale-aware numerical-zero guard
    at ph_con_004.py:135-147 skips constant-ish fields whose interior
    residual is dominated by roundoff.

Plan-diffs logged (plan-vs-committed-state drift, section 7.4):
    19. (Task 10) Plan section 18 F1 Verfurth 2013 Thm 1.12 residual
        estimator "upper-/lower-bound" -> scoped to general residual-
        estimator theory. Rule implements only interior volumetric
        ||Delta u||^2 per element; NOT the full Verfurth estimator
        eta^2 = ||h f||^2 + sum_e ||h^(1/2) [nabla u . n_e]||^2.
        F1 claims conservation-defect localization, not guaranteed
        error bound. Per 2026-04-24 user-revised Task 10 contract.
    20. (Task 10) Plan section 18 step 5 "scikit-fem Example 22
        adaptive-Poisson fixture" -> scikit-fem's docs/examples/ex22.py
        is not importable from pip-installed scikit-fem (examples ship
        in repo, not wheel). Replaced with locally-implemented L-shape
        + P2 + Gaussian-bump fixture in _harness/aposteriori.py
        (CI-runnable; tests same localization semantic).
    21. (Task 10) Plan section 18 step 5 "hotspots within 2 element-
        layers of the L-corner" on canonical u = r^(2/3) sin((2/3)
        theta) -> fixture does not vanish on outer L-shape boundary;
        zero-trace projection introduces boundary artifacts that mask
        the corner signal. Replaced with Gaussian-bump
        u = exp(-20 r^2) which vanishes at float precision on boundary
        and gives max(r_top5)/h = 1.70 corner localization across
        refined(2)-refined(4). Tolerance 2.0 element-layers.
    22. (Task 10) Plan section 18 enumerate-the-splits item (c)
        "uniform refinement vs adaptive (both tested)" -> uniform only
        in V1. Physics-lint does not ship an adaptive marker or
        refiner. Anchor tests localization under uniform refinement at
        n_refine in {2, 3, 4}.

Plan-diffs 1-18 are from Tasks 2, 3, 4, 5, 8, 9, 12 (commits 30baf3e,
0cedc7b, 18312b9, 6800d6f, 1112da3, 26ed3bd, 84c7163).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skfem")

from external_validation._harness.aposteriori import (
    characteristic_h,
    corner_distance_layers,
    gaussian_bump_at_corner,
    interior_element_mask,
    l_shape_mesh,
    p2_basis,
    per_element_residual_sq,
    smooth_bubble,
    top_k_hotspot_centroids,
)
from physics_lint import DomainSpec
from physics_lint.field import GridField, MeshField
from physics_lint.rules import ph_con_004

# ---------------------------------------------------------------------------
# Acceptance bands (2026-04-24 precheck-calibrated per user's revised contract)
# ---------------------------------------------------------------------------

CORNER = (0.0, 0.0)  # L-shape re-entrant corner
CASE_A_LAYER_TOL = 2.0  # max(r_top5)/h must be <= this across refinements
CASE_A_REFINEMENTS = (2, 3, 4)
# Smooth-fixture rule-verdict: PASS at all refinements; ratio < 5 is headroom
# over the shipped threshold of 10.
SMOOTH_RATIO_PASS_CAP = 5.0
# Gaussian-bump rule-verdict: PASS at refined(2) (ratio 8.66), WARN at
# refined(3)+ (ratios 10.85 -> 75.06). Enforce WARN status with ratio > 10
# at refined(3) and refined(4).
BUMP_ALPHA = 20.0

SPEC = DomainSpec.model_validate(
    {
        "pde": "poisson",
        "grid_shape": [32, 32],
        "domain": {"x": [-1.0, 1.0], "y": [-1.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "field": {"type": "mesh", "backend": "fd", "adapter_path": "x"},
    }
)


def _build_basis(n_refine: int):
    mesh = l_shape_mesh(n_refine)
    return p2_basis(mesh)


# =========================================================================
# Case A: F2 harness-level authoritative localization.
#
# Uses the rule's exact per-element quantity (reimplemented in the harness
# via per_element_residual_sq) so the localization check is independent of
# the rule's max/mean ratio reduction. If the underlying L^2-projected
# Laplacian operator correctly localizes at the Gaussian-bump concentration
# point, the top-k element centroids must sit near the re-entrant corner.
# =========================================================================


@pytest.mark.parametrize("n_refine", (3, 4))
def test_case_a_gaussian_bump_hotspots_within_two_layers_of_corner(n_refine):
    """Top-5 hotspot elements land within 2 element-layers of the
    re-entrant corner. Measured on 2026-04-24: max(r_top5)/h = 1.70
    across n_refine in {2, 3, 4}; tolerance 2.0 layers absorbs the
    ~0.3-layer safety margin.
    """
    basis = _build_basis(n_refine)
    dofs = gaussian_bump_at_corner(basis, alpha=BUMP_ALPHA, corner=CORNER)
    elem_res = per_element_residual_sq(basis, dofs)
    mask = interior_element_mask(basis)
    centroids = top_k_hotspot_centroids(basis, elem_res, k=5, interior_mask=mask)
    h = characteristic_h(n_refine)
    layers = corner_distance_layers(centroids, corner=CORNER, h=h)
    assert layers.max() <= CASE_A_LAYER_TOL, (
        f"Case A localization at n_refine={n_refine}: top-5 hotspot max "
        f"layer distance {layers.max():.3f} exceeds tolerance "
        f"{CASE_A_LAYER_TOL} (h={h:.4f}, top5 raw distances="
        f"{(layers * h).round(4).tolist()})"
    )


def test_case_a_layer_invariance_across_refinement():
    """max(r_top5)/h is refinement-invariant in layer units — a
    halving of h halves the physical radius. This is the load-bearing
    claim that the rule's operator localizes at a refinement-independent
    element-layer distance. Measured: 1.70 +- 0.01 across
    n_refine in {2, 3, 4}; accept [1.5, 2.0].
    """
    layer_distances = []
    for n_refine in CASE_A_REFINEMENTS:
        basis = _build_basis(n_refine)
        dofs = gaussian_bump_at_corner(basis, alpha=BUMP_ALPHA, corner=CORNER)
        elem_res = per_element_residual_sq(basis, dofs)
        mask = interior_element_mask(basis)
        centroids = top_k_hotspot_centroids(basis, elem_res, k=5, interior_mask=mask)
        h = characteristic_h(n_refine)
        layers = corner_distance_layers(centroids, corner=CORNER, h=h)
        layer_distances.append(float(layers.max()))
    assert all(1.5 <= d <= 2.0 for d in layer_distances), (
        f"Case A layer-invariance: max-layer-distance across "
        f"n_refine={CASE_A_REFINEMENTS} expected in [1.5, 2.0], got "
        f"{layer_distances!r}"
    )
    spread = max(layer_distances) - min(layer_distances)
    assert spread < 0.2, (
        f"Case A layer-invariance: spread across refinements "
        f"{spread:.3f} > 0.2 layers; expected refinement-invariant "
        f"behavior (measured {layer_distances!r})"
    )


# =========================================================================
# Case B: rule-verdict contract on Gaussian-bump vs smooth fixtures.
#
# Gaussian bump: PASS at refined(2) (ratio ~8.66); WARN at refined(3)+ with
# monotonically growing ratio. Smooth bubble: PASS across all refinements
# with ratio < 5. Transition PASS -> WARN happens on the bump fixture
# around the shipped threshold of 10 at refined(3).
# =========================================================================


@pytest.mark.parametrize("n_refine,expected_status", [(3, "WARN"), (4, "WARN")])
def test_case_b_rule_warns_on_gaussian_bump(n_refine, expected_status):
    basis = _build_basis(n_refine)
    dofs = gaussian_bump_at_corner(basis, alpha=BUMP_ALPHA, corner=CORNER)
    field = MeshField(basis=basis, dofs=dofs)
    result = ph_con_004.check(field, SPEC)
    assert result.status == expected_status, (
        f"Case B Gaussian bump at n_refine={n_refine}: expected "
        f"{expected_status}, got {result.status!r} (ratio={result.raw_value!r})"
    )
    assert result.raw_value is not None
    assert result.raw_value > 10.0, (
        f"Case B Gaussian bump at n_refine={n_refine}: ratio "
        f"{result.raw_value!r} <= 10; WARN threshold not exceeded"
    )


@pytest.mark.parametrize("n_refine", CASE_A_REFINEMENTS)
def test_case_b_rule_passes_on_smooth_bubble(n_refine):
    basis = _build_basis(n_refine)
    dofs = smooth_bubble(basis)
    field = MeshField(basis=basis, dofs=dofs)
    result = ph_con_004.check(field, SPEC)
    assert result.status == "PASS", (
        f"Case B smooth bubble at n_refine={n_refine}: expected PASS, "
        f"got {result.status!r} (ratio={result.raw_value!r})"
    )
    assert result.raw_value is not None
    assert result.raw_value < SMOOTH_RATIO_PASS_CAP, (
        f"Case B smooth bubble at n_refine={n_refine}: ratio "
        f"{result.raw_value!r} >= {SMOOTH_RATIO_PASS_CAP}; smooth fixture "
        f"should have ample PASS margin under the shipped threshold 10"
    )


def test_case_b_bump_ratio_grows_monotonically_under_refinement():
    """Indicator responds to refinement: ratio(n+1) > ratio(n) on the
    Gaussian bump across n_refine in {2, 3, 4}. A non-responsive
    indicator would not detect pre-asymptotic concentration.
    """
    ratios = []
    for n_refine in CASE_A_REFINEMENTS:
        basis = _build_basis(n_refine)
        dofs = gaussian_bump_at_corner(basis, alpha=BUMP_ALPHA, corner=CORNER)
        field = MeshField(basis=basis, dofs=dofs)
        result = ph_con_004.check(field, SPEC)
        assert result.raw_value is not None
        ratios.append(float(result.raw_value))
    for i in range(len(ratios) - 1):
        assert ratios[i + 1] > ratios[i], (
            f"Case B monotonicity: ratio {ratios[i + 1]} at n_refine="
            f"{CASE_A_REFINEMENTS[i + 1]} not > {ratios[i]} at n_refine="
            f"{CASE_A_REFINEMENTS[i]}. Full sequence: {ratios!r}"
        )


def test_case_b_smooth_strictly_below_bump_at_same_refinement():
    """At a fixed refinement level, the smooth fixture's ratio must be
    strictly below the Gaussian bump's ratio -- the rule must
    distinguish concentrated-Laplacian input from smooth input.
    """
    n_refine = 4
    basis = _build_basis(n_refine)
    smooth = ph_con_004.check(MeshField(basis=basis, dofs=smooth_bubble(basis)), SPEC)
    bump = ph_con_004.check(
        MeshField(
            basis=basis,
            dofs=gaussian_bump_at_corner(basis, alpha=BUMP_ALPHA, corner=CORNER),
        ),
        SPEC,
    )
    assert smooth.raw_value is not None and bump.raw_value is not None
    assert smooth.raw_value < bump.raw_value, (
        f"Rule fails to distinguish smooth from bump at n_refine={n_refine}: "
        f"smooth ratio={smooth.raw_value!r}, bump ratio={bump.raw_value!r}"
    )


# =========================================================================
# SKIP-path contracts: V1 scope boundary.
# =========================================================================


def test_skipped_on_gridfield_input():
    """Rule requires MeshField. GridField input SKIPs (ph_con_004.py:92-93)."""
    field = GridField(np.zeros((16, 16)), h=(1 / 15, 1 / 15), periodic=False, backend="fd")
    result = ph_con_004.check(field, SPEC)
    assert result.status == "SKIPPED"
    assert "MeshField" in (result.reason or "")


def test_skipped_on_constant_meshfield():
    """Constant field has interior residuals at roundoff; rule SKIPs via
    the scale-aware numerical-zero guard (ph_con_004.py:141-147). Validates
    the well-defined-path contract for inputs below the floor.
    """
    basis = _build_basis(3)
    dofs = np.ones(basis.N)
    field = MeshField(basis=basis, dofs=dofs)
    result = ph_con_004.check(field, SPEC)
    # Constant field may PASS or SKIP depending on mesh discretization; both
    # are well-defined outcomes (tests/rules/test_ph_con_004.py:66 precedent).
    # A WARN would indicate spurious amplification and should fail the test.
    assert result.status in {"PASS", "SKIPPED"}, (
        f"Constant MeshField expected PASS or SKIP, got "
        f"{result.status!r} (raw_value={result.raw_value!r})"
    )


def test_skipped_on_coarse_mesh_all_boundary():
    """On a very coarse mesh where every element touches the boundary,
    the interior mask is empty; rule SKIPs (ph_con_004.py:129-133).
    init_lshaped() at refined(0) has 6 triangles all meeting the
    boundary -> interior set may be empty depending on P2 DOF counts.
    """
    mesh = l_shape_mesh(0)
    basis = p2_basis(mesh)
    dofs = gaussian_bump_at_corner(basis, alpha=BUMP_ALPHA, corner=CORNER)
    field = MeshField(basis=basis, dofs=dofs)
    mask = interior_element_mask(basis)
    if mask.sum() == 0:
        # Hit the all-boundary SKIP path.
        result = ph_con_004.check(field, SPEC)
        assert result.status == "SKIPPED"
        assert "interior" in (result.reason or "").lower()
    else:
        # Mesh happened to have interior DOFs; rule runs normally. This
        # test's goal is to exercise the SKIP path when it applies, not
        # to force a failure; skip with rationale.
        pytest.skip(
            f"init_lshaped refined(0) has {int(mask.sum())} interior "
            f"elements on this scikit-fem version; all-boundary SKIP "
            f"path not reachable at this refinement level"
        )
