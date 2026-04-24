# PH-BC-001 external-validation anchor

**Scope separation (read first):** PH-BC-001 validates **Dirichlet-
type boundary trace behavior in the production rule; Neumann/flux
semantics are outside the production validation scope for v1.0**. The
rule's emitted quantity is `||field.values_on_boundary() −
boundary_target||` — a discrete-L² comparison of Dirichlet-type trace
extraction against a caller-supplied target, not a Neumann flux
check. CITATION.md, README, and test docstrings do not imply broader
rule coverage than V1 provides.

Dirichlet-trace violation metric: discrete-L² norm of
`u|_{∂Ω} − g|_{∂Ω}` on the unit square, mode-branched on `||g||`
(absolute for near-zero Dirichlet, relative otherwise). Threshold
0.01 relative (tri-state) / 1e-3 absolute (binary).

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-BC-001/ -v
```

Expected: 13 passed in < 3 s (9 F2 parametrized across 3 Dirichlet
fixtures × 3 refinement levels + 1 perturbation-scaling + 3 rule-
verdict-contract including mode-branch and shape-mismatch). No
mesh / torch / scikit-fem dependency — pure numpy on uniform grid.

## Citation stack (three-function-labeled)

See `CITATION.md` for the full function-labeled citation stack.
Summary:

- **F1 Mathematical-legitimacy** (Tier 2 theoretical-plus-multi-paper):
  Evans 2010 §5.5 Theorem 1 trace theorem (section-level per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠; theorem number pending
  local copy); 4-step proof-sketch from `γ: H¹(Ω) → H^{1/2}(∂Ω)` to
  discrete trace via `values_on_boundary()` to Dirichlet-trace
  mismatch.
- **F2 Correctness-fixture (harness-level)**:
  `external_validation/_harness/trace.py`
  `dirichlet_trace_on_unit_square_grid(u_fn, n)`. Three analytic
  Dirichlet fixtures on unit square:
  - `u = x² − y²` (polynomial, relative mode): raw = 0 exactly at
    every N ∈ {16, 32, 64} → PASS.
  - `u = sin(πx) sin(πy)` (zero Dirichlet, absolute mode): raw = 0
    exactly → PASS.
  - `u = cos(πx) cos(πy)` (nonzero Dirichlet, relative mode): raw =
    0 exactly → PASS.
  Rule liveness: perturbing `u = x² − y²` left edge by δ=1e-3 at
  N=32 gives raw ≈ 8.4e-4 (tested to lie in [1e-4, 5e-3]);
  demonstrates the rule scales with the discrete-L² perturbation
  magnitude as expected.
- **F3 Borrowed-credibility**: **absent with justification**. Live
  PDEBench reproduction requires a dataset loader (adapter-mode
  plumbing for PDEBench HDF5 datasets) not shipped in V1; deferred
  pending V1.x loader infrastructure. Task 0's pinned PDEBench rows
  (Diffusion-sorption, 2D diffusion-reaction, 1D Advection) retained
  in Supplementary calibration context with semantic-equivalence
  derivation.
- **Supplementary calibration context**: PDEBench Takamoto et al.
  2022 `arXiv:2210.07182` Tables 5–6. Pinned rows: Diffusion-
  sorption U-Net 6.1e-3 / FNO 2.0e-3 (Dirichlet-dominant); 2D
  diffusion-reaction U-Net 7.8e-2 / FNO 2.7e-2 (flagged Neumann,
  not V1 rule scope); 1D Advection β=0.1 U-Net 3.8e-2 / FNO 4.9e-3
  (flagged periodic, calibration-adjacent). Reproduction tolerance
  (±2×) documented; not a reproduction claim in V1.

## Plan-diffs (8 cumulative across complete-v1.0 execution)

See `test_anchor.py` module docstring for diffs 7 and 8 (introduced
in Task 4). Diffs 1–6 are from Tasks 2, 3, 5. Summary:

1. (Task 2) FD `p ∈ {2, 4}` → interior band 4th-order only.
2. (Task 2) Plan "F3-absent pre-recorded" → decided in execution.
3. (Task 3) Plan `N = {16, 32, 64}` exp-fit → `SPECTRAL_NS = [8, 10,
   12, 14, 16]` pre-floor with rapid-collapse fallback.
4. (Task 3) Plan "V1: 1D only" → 2D fixture (DomainSpec ≥ 2D).
5. (Task 3) F3-hunt "Thm 4" → chapter-level per §6.4.
6. (Task 5) Plan §13 `F=(x,y)` recast as harness-level F2 only;
   rule-verdict contract on Laplace-harmonic added.
7. **(Task 4, this commit)** Plan §12 "Dirichlet, Neumann, periodic"
   F2 fixtures → **Dirichlet-only**. Rule's
   `values_on_boundary()` is Dirichlet-type value extraction;
   Neumann normal-derivative semantics are outside V1 rule scope.
   Periodic is vacuous on torus (no boundary in the trace sense).
8. **(Task 4, this commit)** Plan §12 F3-PRESENT PDEBench
   reproduction → **F3-absent + PDEBench in Supplementary**. Live
   reproduction requires a PDEBench dataset loader (adapter-mode
   plumbing) not shipped in V1; loader-infrastructure gap missed by
   2026-04-24 precheck. Semantic-equivalence derivation preserved;
   reproduction deferred to V1.x.
