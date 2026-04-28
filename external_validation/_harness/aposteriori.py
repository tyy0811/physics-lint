"""A-posteriori residual-based indicator helpers for Task 10 (PH-CON-004).

Consumers: ``external_validation/PH-CON-004/test_anchor.py``. Also expected
to be extended by Task 11 (PH-NUM-001) for its scikit-fem quadrature
fixtures.

Scope (2026-04-24 user-revised Task 10 contract):

- **2D triangulated meshes only.** 3D tetrahedral meshes deferred to v1.2
  per complete-v1.0 plan §0.9 scope boundary.
- **Rule-scope-matched fixture.** PH-CON-004 emits
  ``max_K / mean_K`` of ``∫_K (Δ_{L²-proj zero-trace} u)² dx`` over
  interior elements (`ph_con_004.py:84-164`). It is **not** the classical
  Verfürth residual estimator ``η² = ||hf||² + Σ_e ||h^(1/2) [∇u_h·n_e]||²``
  — no volumetric source term, no facet-jump term, no h-weighting. Helpers
  here stay within the rule's narrower quantity.
- **Localization test target.** Fixtures vanish on the mesh boundary (so
  the zero-trace projection does not introduce artifacts that mask the
  true localization signal) and have their Laplacian magnitude
  concentrated at a known point.

Helpers:

- ``l_shape_mesh(n_refine)``: scikit-fem ``MeshTri.init_lshaped().refined(n_refine)``.
- ``p2_basis(mesh)``: ``Basis(mesh, ElementTriP2())``.
- ``gaussian_bump_at_corner(basis, alpha)``: ``u = exp(-alpha r²)`` centered
  at the re-entrant corner; vanishes on outer L-shape boundary for
  ``alpha ≥ 20``.
- ``smooth_bubble(basis)``: ``u = sin(πx) sin(πy) · max(1 - r²/2, 0)`` on
  the L-shape; smooth everywhere, vanishes on boundary.
- ``interior_element_mask(basis)``: the rule's DOF-aware interior mask
  (elements with no boundary DOF). Duplicates ``ph_con_004.py:121-126``.
- ``per_element_residual_sq(basis, dofs)``: element-wise
  ``∫_K (Δ_{L²-proj zero-trace} u)² dx``. Duplicates the rule's emitted
  per-element quantity so the harness-level layer can do localization
  checks without taking a privileged view into rule internals.
- ``top_k_hotspot_centroids(basis, elem_res, k, interior_mask)``: returns
  the centroids of the top-k interior elements by residual.
- ``corner_distance_layers(basis, centroids, corner, h)``: distance to a
  named corner expressed in element-layer units (``distance / h``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def l_shape_mesh(n_refine: int):
    """scikit-fem L-shape mesh refined `n_refine` times."""
    from skfem import MeshTri

    return MeshTri.init_lshaped().refined(n_refine)


def p2_basis(mesh):
    """P2 basis on the given mesh (continuous Lagrange, quadratic)."""
    from skfem import Basis, ElementTriP2

    return Basis(mesh, ElementTriP2())


def gaussian_bump_at_corner(basis, alpha: float = 20.0, corner=(0.0, 0.0)) -> np.ndarray:
    """u = exp(-alpha * ||x - corner||²).

    Vanishes on the L-shape boundary at alpha >= 20 (``exp(-20) ≈ 2e-9``),
    so the rule's zero-trace projection is consistent with the fixture.
    Laplacian is concentrated at ``corner``; magnitude peaks at r = 0 and
    decays as ``4 * alpha * (alpha * r**2 - 1) * exp(-alpha * r**2)``.
    """
    x = basis.doflocs[0]
    y = basis.doflocs[1]
    cx, cy = corner
    r_sq = (x - cx) ** 2 + (y - cy) ** 2
    return np.exp(-alpha * r_sq)


def smooth_bubble(basis) -> np.ndarray:
    """u = sin(π x) sin(π y) · max(1 - r²/2, 0).

    Smooth everywhere; the bubble factor `max(1-r²/2, 0)` makes u vanish
    on the L-shape outer boundary (r_max = √2). No concentrated Laplacian;
    rule ratio stays PASS under refinement.
    """
    x = basis.doflocs[0]
    y = basis.doflocs[1]
    bubble = np.maximum(1.0 - (x * x + y * y) / 2.0, 0.0)
    return np.sin(np.pi * x) * np.sin(np.pi * y) * bubble


def interior_element_mask(basis) -> np.ndarray:
    """DOF-aware interior mask: True for elements with NO boundary DOF.

    Replicates `ph_con_004.py:121-126`. The facet-only mask
    (`mesh.boundary_facets()`) misses P2 edge-midpoint DOFs on elements
    that share only a vertex with the boundary; this DOF-aware mask
    catches them.
    """
    mesh = basis.mesh
    boundary_dof_set = set(basis.get_dofs().flatten())
    elem_dofs = basis.element_dofs
    has_boundary_dof = np.array(
        [bool(boundary_dof_set & set(elem_dofs[:, e])) for e in range(mesh.nelements)]
    )
    return ~has_boundary_dof


def per_element_residual_sq(basis, dofs: np.ndarray) -> np.ndarray:
    """Element-wise ``∫_K (Δ_{L²-proj zero-trace} u)² dx`` across all elements.

    Returns a length-`mesh.nelements` array. Pair with
    ``interior_element_mask(basis)`` to mask out boundary-touching elements
    (same discipline as the rule).
    """
    from skfem import Functional

    from physics_lint.field import MeshField

    @Functional
    def residual_sq(w):  # type: ignore[no-untyped-def]
        return w["lap"] ** 2

    lap_field = MeshField(basis=basis, dofs=dofs).laplacian_l2_projected_zero_trace()
    return np.asarray(residual_sq.elemental(basis, lap=basis.interpolate(lap_field.values())))


def top_k_hotspot_centroids(
    basis,
    elem_res: np.ndarray,
    *,
    k: int,
    interior_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Centroids of the top-k elements by residual (interior-only by default).

    Returns an array of shape ``(k, 2)``. If ``interior_mask`` is provided,
    elements outside the mask are excluded from the ranking (their residual
    is set to -inf before argsort).
    """
    mesh = basis.mesh
    if interior_mask is not None:
        ranking_res = np.where(interior_mask, elem_res, -np.inf)
    else:
        ranking_res = elem_res
    # argsort ascending; top-k are the last k indices.
    top_idx = np.argsort(ranking_res)[-k:][::-1]
    tri_pts = mesh.p[:, mesh.t]  # (2, 3, n_tri)
    centroids = tri_pts.mean(axis=1).T  # (n_tri, 2)
    return centroids[top_idx]


def corner_distance_layers(
    centroids: np.ndarray, corner: tuple[float, float], h: float
) -> np.ndarray:
    """Distances from `centroids` to `corner`, expressed as element-layer units.

    Element-layer is the characteristic mesh spacing `h`. An output of 1.5
    means the element is ~1.5 mesh-spacing-lengths away from the corner.
    """
    cx, cy = corner
    d = np.sqrt((centroids[:, 0] - cx) ** 2 + (centroids[:, 1] - cy) ** 2)
    return d / h


def characteristic_h(n_refine: int) -> float:
    """Characteristic mesh spacing on the init_lshaped + n refinements.

    `init_lshaped()` has 6 triangles spanning 2 units; each refinement halves
    edge lengths. So `h ≈ 2 / 2^(n_refine + 1)`.
    """
    return 2.0 / (2 ** (n_refine + 1))


def run_refinement_sweep(
    fixture_fn: Callable,
    n_refines: tuple[int, ...],
) -> list[dict]:
    """Build L-shape + P2 basis + fixture at each refinement; compute
    per-element residuals, interior mask, top-5 hotspots, and h.

    Returns a list of dicts (one per refinement) with keys:
    `n_refine, n_elements, n_interior, h, elem_res, interior_mask,
    top5_centroids, ratio_max_over_mean`.
    """
    out = []
    for n in n_refines:
        mesh = l_shape_mesh(n)
        basis = p2_basis(mesh)
        dofs = fixture_fn(basis)
        mask = interior_element_mask(basis)
        elem_res = per_element_residual_sq(basis, dofs)
        interior_res = elem_res[mask]
        if interior_res.size == 0 or float(np.mean(interior_res)) <= 0.0:
            ratio = float("nan")
        else:
            ratio = float(np.max(interior_res) / np.mean(interior_res))
        top5 = top_k_hotspot_centroids(basis, elem_res, k=5, interior_mask=mask)
        out.append(
            {
                "n_refine": n,
                "n_elements": int(mesh.nelements),
                "n_interior": int(mask.sum()),
                "h": characteristic_h(n),
                "elem_res": elem_res,
                "interior_mask": mask,
                "top5_centroids": top5,
                "ratio_max_over_mean": ratio,
            }
        )
    return out
