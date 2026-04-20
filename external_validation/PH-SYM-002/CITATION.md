# PH-SYM-002 — Helwig group-equivariant FNO (reflection)

## Citation

- **Paper:** Helwig, Zhang, Fu, Kurtin, Wojtowytsch, Ji, "Group Equivariant Fourier Neural Operators for Partial Differential Equations."
- **Venue:** ICML 2023, PMLR 202:12907–12930.
- **arXiv ID:** 2306.05697.
- **Section:** Table 1 (G-FNO-p4m on the symmetric test set).
- **Artifact:** G-FNO-p4m relative MSE 2.37 ± 0.19.
- **Pinned value:** verdict-based — PH-SYM-002 reports PASS on the `cos(2πx)cos(2πy)` flip-invariant fixture (both axis-0 and axis-1, equivariance error < 1e-12) and WARN/FAIL on the `sin(2πx)sin(2πy)` flip-breaking fixture (`np.flip(u, axis=k) = −u` on each axis, error exactly 2.0). Matches the PH-SYM-001 anchor in structure (same fixtures, same grid, different group element — reflection vs rotation).
- **Verification date:** 2026-04-20.
- **Verification protocol:** two-layer, mirroring PH-SYM-001.
  - **Operator validation (shared harness):** `fft_laplace_inverse` is proven reflection-equivariant on `sin(2πx)sin(2πy)` in `_harness/tests/test_symmetry.py` — the symbol `1/(k_x² + k_y²)` is symmetric under `(k_x, k_y) → (−k_x, k_y)` and `(k_x, k_y) → (k_x, −k_y)`.
  - **Rule anchor (this file):** direct-fixture construction, flip-table sanity tests, then `ph_sym_002.check(GridField, DomainSpec(symmetries={"declared": ["reflection_x", "reflection_y"]}))`.

## Test design

- **Positive fixture:** `u(x,y) = cos(2πx)cos(2πy)` — `np.flip(u, axis=0) ≈ u` and `np.flip(u, axis=1) ≈ u` to float noise.
- **Negative fixture:** `u(x,y) = sin(2πx)sin(2πy)` — `np.flip(u, axis=k) = −u` exactly for `k ∈ {0, 1}`.
- **DomainSpec:** `pde="laplace"`, `symmetries.declared=["reflection_x", "reflection_y"]`, `boundary_condition={kind:"dirichlet"}`, non-periodic.
- **Grid:** 64 × 64.
- **Wall-time budget:** < 15 s.

## Scope note

Same as PH-SYM-001/CITATION.md.
