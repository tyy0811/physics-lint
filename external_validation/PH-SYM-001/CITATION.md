# PH-SYM-001 — Helwig group-equivariant FNO (rotation)

## Citation

- **Paper:** Helwig, Zhang, Fu, Kurtin, Wojtowytsch, Ji, "Group Equivariant Fourier Neural Operators for Partial Differential Equations."
- **Venue:** ICML 2023, PMLR 202:12907–12930.
- **arXiv ID:** 2306.05697.
- **Section:** Table 3 (2D Navier–Stokes rotation test).
- **Artifact:** plain FNO relative MSE 8.41 ± 0.41 (unrotated) → 129.21 ± 3.90 (90°-rotated); G-FNO-p4 test MSE ≈ 4.78 ± 0.39 (equivariant by construction).
- **Pinned value:** verdict-based — PH-SYM-001 reports PASS on the `cos(2πx)cos(2πy)` C₄-symmetric fixture (equivariance error < 1e-12) and WARN/FAIL on the `sin(2πx)sin(2πy)` C₄-breaking fixture (equivariance error exactly 2.0 since `rot90(sin·sin) = −sin·sin`). Helwig's 15× degradation on plain FNO at 90° rotation is the literature anchor for "equivariance violations are empirically detectable by L² difference metrics"; the rule anchor tests that physics-lint's own `equivariance_error_np`-based detector correctly classifies a synthetic fixture at the extreme of that detectability scale.
- **Verification date:** 2026-04-20.
- **Verification protocol:** two-layer.
  - **Operator validation (shared harness, Tier-B support):** `fft_laplace_inverse` is proven C₄-equivariant to float noise on `sin(2πx)sin(2πy)` in `_harness/tests/test_symmetry.py` (the symbol `1/(k_x² + k_y²)` is C₄-invariant on a periodic square grid; zero-mode convention `û(k=0) = 0` closes the `(−Δ)⁻¹` kernel ambiguity). Random-weight `_NonEquivariantCNN` with learned positional embedding is proven non-equivariant on the same input.
  - **Rule anchor (this file):** direct-fixture construction of a C₄-symmetric `cos(2πx)cos(2πy)` and a C₄-breaking `sin(2πx)sin(2πy)` on a 64×64 grid. The fixtures' rotation tables are verified by sanity tests (see `test_anchor.py::test_fixture_*`) before the rule is invoked.

## Test design

- **Positive fixture:** `u(x,y) = cos(2πx)cos(2πy)` on `np.linspace(0,1,64)` — non-trivially C₄-symmetric (not SO(2)-invariant). `np.rot90(u, k) ≈ u` to float noise for `k ∈ {1,2,3}`.
- **Negative fixture:** `u(x,y) = sin(2πx)sin(2πy)` on the same grid — `np.rot90(u, k=1) = −u` exactly.
- **DomainSpec:** `pde="laplace"`, `symmetries.declared=["C4"]`, `boundary_condition={kind:"dirichlet"}`, non-periodic.
- **Grid:** 64 × 64.
- **Wall-time budget:** < 15 s (pure-numpy, no FNO inference, no torch).

## Scope note

Helwig evaluates on 2D Navier–Stokes and NS-SYM, which are outside v1.0's
Laplace/Poisson/heat/wave PDE scope. Rule calibration transfers (the
field-invariance detection mechanism is PDE-agnostic); absolute MSE
magnitudes do not. This anchor verifies that `ph_sym_001`'s
`equivariance_error_np`-based detector correctly classifies a
C₄-symmetric field as PASS and a C₄-breaking field as WARN/FAIL on the
v1.0 PDE family.
