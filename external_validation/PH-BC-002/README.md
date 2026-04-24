# PH-BC-002 external-validation anchor

**Scope separation (read first):** PH-BC-002's external validation
separates **(i) a harness-level Gauss-Green correctness fixture** from
**(ii) the production rule's currently supported Laplace-scope verdict
behavior**. The production rule `src/physics_lint/rules/ph_bc_002.py`
is Laplace-scope only (Week 1 scope; Poisson arm defers to Week 2);
the F1 mathematical anchor — the general Gauss-Green / divergence
theorem — covers arbitrary C¹ vector fields. CITATION.md, README, and
test docstrings do not imply broader production coverage than V1
provides.

Boundary flux imbalance check: for a Laplace-harmonic field `u`, the
rule verifies `∫_Ω Δu dV ≈ 0` (equivalently `∫_{∂Ω} ∂u/∂n dS ≈ 0` by
Gauss-Green) within a relative threshold of 0.01 against `||u||_{L²}`.

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-BC-002/ -v
```

Expected: 12 passed in < 5 s (6 F2-harness parametrized across
{tri, quad} × {4, 8, 16} + 1 invariance summary + 5 rule-verdict-
contract covering PASS/WARN/SKIP paths). Requires `scikit-fem` (already
pinned as optional `[mesh]` extra in `pyproject.toml`).

## Citation stack (three-function-labeled)

See `CITATION.md` for the full function-labeled citation stack.
Summary:

- **F1 Mathematical-legitimacy** (Tier 1 structural-equivalence):
  Evans 2010 Appendix C.2 Gauss-Green Theorem 1 (section-level per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠; theorem number pending
  local copy); Gilbarg-Trudinger 2001 §2.4 (section-level ⚠); 4-step
  proof-sketch embedded in CITATION.md.
- **F2 Correctness-fixture (harness-level, authoritative)**:
  `external_validation/_harness/divergence.py`
  `gauss_green_on_unit_square(mesh_type, n_refine)`. LHS =
  `∫_Ω div F dV` and RHS = `∫_{∂Ω} F·n dS` both equal 2.0 to float64
  roundoff (~4e-16) for `F = (x, y)` on unit square, across both
  triangulation (MeshTri + ElementTriP1) and quadrilateralization
  (MeshQuad + ElementQuad1) at N ∈ {4, 8, 16}.
- **Rule-verdict contract**: exercises the production rule's V1
  Laplace-scope emitted quantity. `u = x² − y²` (degree-2 harmonic)
  gives exact `raw_value = 0` at every N ∈ {16, 32, 64} → PASS.
  `u = x⁵ − 10x³y² + 5xy⁴` (degree-5 harmonic, Re of z⁵) WARNs at
  N=16 due to boundary-FD4 O(h²) error, PASSes at N ∈ {32, 64} with
  ~9× decrease per doubling. SKIP paths on Poisson (Week 2 source
  wiring) and non-laplace/poisson PDEs (scope guard).
- **F3 Borrowed-credibility**: absent with justification. Gauss-Green
  reproduction on MMS fixtures is tautological under the theorem's
  stated preconditions — no borrowed-credibility via published
  numerical baseline applicable.
- **Supplementary calibration context**: LeVeque 2002 FVM §2.1 ISBN
  978-0-521-81087-6 (pedagogical framing flag — divergence-theorem
  rearrangement motivates finite-volume conservation schemes; not a
  reproduction).

## Plan-diffs (6 cumulative across Tier-B execution)

See `test_anchor.py` module docstring for diff 6 (introduced in Task
5). Diffs 1–5 are from Tasks 2 + 3. Summary:

1. (Task 2, commit 30baf3e) Plan §10 FD stencil order `p ∈ {2, 4}` →
   rule interior band is 4th-order only.
2. (Task 2) Plan §10 "F3-absent pre-recorded in Task 0" → decided
   during Task 2 execution per §10 fallback path.
3. (Task 3, commit 0cedc7b) Plan §11 `N = {16, 32, 64}` exp-fit R² >
   0.99 → `SPECTRAL_NS = [8, 10, 12, 14, 16]` pre-floor + rapid-
   collapse fallback.
4. (Task 3) Plan §11 "V1: 1D only" → 2D fixture (DomainSpec
   grid_shape length ≥ 2).
5. (Task 3) F3-hunt audit "Thm 4" → chapter-level per §6.4.
6. **(Task 5, this commit)** Plan §13 "F=(x,y) ... yield LHS = RHS = 2
   within tolerance" recast as harness-level (F2) only. Production
   rule PH-BC-002 is Laplace-scope only, so the rule's emitted
   quantity does not exercise the arbitrary-F fixture. Scope
   separation per 2026-04-24 Path C + V1-stub CRITICAL-task pattern:
   F2 is authoritative via `_harness/divergence.py`; rule-verdict
   contract added on Laplace-harmonic fixture to exercise V1 rule
   scope.
