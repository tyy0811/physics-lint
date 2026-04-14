# scikit-fem spike — Week 1 Day 2

**Outcome:** PASSED on 2026-04-14

## Measured

- **Installation:** `pip install "scikit-fem>=10"` → resolved to `scikit-fem==12.0.1`, pure-Python, no native deps, instant.
- **Assembly of Poisson stiffness** via `laplace.assemble(Basis(MeshTri().refined(r), ElementTriP2()))`: OK.
- **Solve** of Dirichlet-homogeneous Poisson for the manufactured source `f = 2 π² sin(π x) sin(π y)` on the unit square via `condense` + `solve`: OK.
- **Discrete L² errors** at DOF points for refinements `r ∈ {2, 3, 4, 5}` (DOF counts `{81, 289, 1089, 4225}`):

  | r | dofs | \|\|u_h − u\|\|₂ / √N |
  |---|-----:|------------------------:|
  | 2 |   81 |               1.190e-03 |
  | 3 |  289 |               8.668e-05 |
  | 4 | 1089 |               5.771e-06 |
  | 5 | 4225 |               3.708e-07 |

- **Observed convergence rate**: 3.779 (r 2→3), 3.909 (r 3→4), 3.960 (r 4→5). The plan's spike script comment ("expect ~2.0") reflected a P1-elements assumption; P2 elements give O(h³) in continuous L² and exhibit superconvergence (up to O(h⁴)) at nodal / edge-midpoint DOF locations for smooth manufactured solutions. Both the asymptotic rate and the convergence monotonicity comfortably exceed the pass criterion (≥ O(h²)).

- **Total elapsed time:** 0.7 seconds (budget was 4 hours).

## Decision

**MeshField, `PH-CON-004`, `PH-NUM-001` ship in V1.** Week 3 Day 4 will implement the scikit-fem-backed path per design doc §3.4 and §8.2. `pyproject.toml` already declares the `mesh = ["scikit-fem>=10"]` optional-dependency extra from Task 1; no dependency changes are needed for Week 1.

## Follow-ups

- Design doc §2.4 spike-criterion text mentions "O(h²)" as the acceptance floor. P2 elements will actually meet this with comfortable headroom (observed ~O(h⁴) at DOF points), so no doc revision is required — the spike criterion is met and then some.
- Plan §6 ("spike script") comment "expect ~2.0" should be updated in a future plan revision to "expect ≥ 2.0; P2 superconvergence at nodes typically gives ~3-4". Not a Week 1 code change.
- The throwaway `scripts/spike_scikit_fem.py` is deleted per Task 6 Step 5.
