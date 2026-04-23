# PH-SYM-001 — Discrete rotation equivariance (C₄)

## Function-labeled citation stack

Per complete-v1.0 plan §1.3 three-function labeling. Retrofit landed during
Task 1 of the complete-v1.0 plan on 2026-04-23; Tier-A Helwig 2023 content
preserved under Supplementary calibration context with the function-labeled
structure added as the primary organizational layer. Bronstein-GDL demoted
per the 2026-04-22 literature audit Round 3 finding that Bronstein-GDL is
motivational-only, not proof-carrying.

### Mathematical-legitimacy (Tier 1 structural-equivalence)

- **Primary**: Hall, B.C. (2015). *Lie Groups, Lie Algebras, and
  Representations*, 2nd ed. Springer GTM 222. ISBN 978-3-319-13466-6. DOI
  10.1007/978-3-319-13467-3. §2.5 (one-parameter-subgroup family,
  section-level); §3.7 (continuous-to-smooth for matrix Lie group
  homomorphisms, section-level). Theorem numbers pending primary-source
  verification per `../_harness/TEXTBOOK_AVAILABILITY.md` (⚠ after
  complete-v1.0 Task 0 Step 3; secondary-source corroboration via Wikipedia
  "One-parameter group" and "Lie group–Lie algebra correspondence" articles
  and nLab "continuous homomorphisms of Lie groups are smooth").
- **Identity-component generation**: Varadarajan, V.S. (1984). *Lie Groups,
  Lie Algebras, and Their Representations.* Springer GTM 102. ISBN
  978-0-387-90969-1. §2.9–2.10 (section-level; ⚠ per
  `../_harness/TEXTBOOK_AVAILABILITY.md`).
- **ML-paper structural bridge**: Cohen, T.S. & Welling, M. (2016). *Group
  Equivariant Convolutional Networks.* ICML 2016. arXiv:1602.07576. §3–§4
  establish the G-equivariant convolution framework for discrete rotation
  subgroups of SO(2) on 2D grids; p4 (C₄ action on R²) is their paradigm
  example.
- **Structural-equivalence proof-sketch** (C₄ discrete rotation on a 2D
  periodic square grid, section-level framing throughout — no tight
  theorem-number claims per §6.4 of complete-v1.0 plan):
  1. C₄ = {R(kπ/2) : k ∈ {0,1,2,3}} is a finite cyclic subgroup of SO(2),
     realized as the restriction to t ∈ Z of the continuous one-parameter
     subgroup t ↦ exp(t · (π/2) · J) ∈ SO(2), where J = [[0,-1],[1,0]] is
     the so(2) generator. The one-parameter-subgroup correspondence
     underlying this exponential construction is established at Hall §2.5
     (section-level, theorem number pending local copy).
  2. SO(2) is a connected matrix Lie group; Varadarajan §2.9–2.10
     (section-level) establishes that its identity component is generated
     by any neighborhood of the identity. C₄ sits inside SO(2) as a closed
     finite subgroup whose cosets partition SO(2) into four arcs.
  3. For a continuous operator f: L²(T²) → L²(T²) on the 2-torus,
     C₄-equivariance `f(R · u) = R · f(u)` for R ∈ C₄ is a well-defined
     condition on L²(T²). Hall §3.7 (section-level) — continuous
     homomorphisms of finite-dimensional matrix Lie groups are smooth —
     guarantees that continuity plus equivariance with respect to the
     ambient SO(2) action has a well-defined smooth-equivariance
     interpretation, which the discrete C₄ restriction inherits.
  4. The rule's emitted quantity `rotate_test(f, u, k=1) = ||f(R(π/2) · u)
     - R(π/2) · f(u)||_{L²} / ||f(u)||_{L²}` (defined in
     `../_harness/symmetry.py`) numerically measures deviation from this
     C₄-equivariance condition. For operators provably C₄-equivariant by
     construction (e.g., `fft_laplace_inverse`, whose Fourier symbol
     1/(k_x² + k_y²) is C₄-invariant on a periodic square grid), the
     target is zero up to floating-point noise; for random-weight
     non-equivariant operators, the error lands at O(1) relative scale
     (empirically validated in `../_harness/tests/test_symmetry.py`).
- **Verification status** (per `../_harness/TEXTBOOK_AVAILABILITY.md`):
  Hall 2015 §2.5 + §3.7 and Varadarajan 1984 §2.9–2.10 are all ⚠
  secondary-source-confirmed only as of Task 0 Step 3 (2026-04-23);
  Springer GTM 222 / GTM 102 paywall access attempts logged in
  TEXTBOOK_AVAILABILITY.md. Section-level framing is required; tight
  theorem-number framing is mechanically rejected by
  `scripts/check_theorem_number_framing.py`.

### Correctness-fixture (CI-runnable, non-credibility-claiming)

- **Positive fixture**: `u(x,y) = cos(2πx) cos(2πy)` on
  `np.linspace(0,1,64)` — non-trivially C₄-symmetric (not SO(2)-invariant).
  `np.rot90(u, k) ≈ u` to float noise for `k ∈ {1, 2, 3}`. Grid 64 × 64.
- **Negative fixture**: `u(x,y) = sin(2πx) sin(2πy)` on the same grid —
  `np.rot90(u, k=1) = −u` exactly, producing relative-L² equivariance
  error of precisely 2.0.
- **Operator-level correctness**: `fft_laplace_inverse` (in
  `../_harness/symmetry.py`) is proven C₄-equivariant to float noise in
  `../_harness/tests/test_symmetry.py`. The symbol 1/(k_x² + k_y²) is
  C₄-invariant on a periodic square grid; zero-mode convention
  `û(k=0) = 0` closes the `(−Δ)⁻¹` kernel ambiguity. Random-weight
  `_NonEquivariantCNN` (learned positional embedding) is proven
  non-equivariant on the same input.
- **Rule anchor**: `test_anchor.py` directly constructs the two fixtures
  on a 64 × 64 grid and verifies `ph_sym_001.check(...)` returns PASS on
  the positive fixture (equivariance error < 1e-12) and WARN/FAIL on the
  negative fixture (equivariance error exactly 2.0). Fixture rotation
  tables are verified by `test_fixture_*` sanity tests before the rule
  is invoked.

### Borrowed-credibility (external published reproduction layers)

**F3 absent with justification.** No published numerical baseline exists
for discrete-rotation equivariance-error on physics-lint's
`rotate_test`-style emitted quantity. Published equivariance-related
numbers — Helwig 2023 ICML Table 3 (plain vs G-FNO relative MSE on 2D
Navier–Stokes), Weiler-Cesa NeurIPS 2019 (E(2)-CNN test error on
RotMNIST), Cohen-Welling ICML 2016 (P4CNN test error on RotMNIST) —
report downstream-task accuracy or relative MSE, not a directly-
comparable equivariance-error measurement on an author-constructed
analytical fixture. Borrowed-credibility via reproduction is not
available for this rule. Per complete-v1.0 plan §1.2, F3-absent-is-
structural for rules whose correctness layer is analytical and whose
mathematical-legitimacy layer is the Tier-1 structural-equivalence
proof-sketch above.

### Supplementary calibration context

- **Equivariance-scale calibration**: Helwig, T., Zhang, X., Fu, Y.,
  Kurtin, J.S., Wojtowytsch, S., Ji, S. (2023). *Group Equivariant
  Fourier Neural Operators for Partial Differential Equations.* ICML
  2023, PMLR 202:12907–12930. arXiv:2306.05697. Table 3 (2D
  Navier–Stokes rotation test): plain FNO relative MSE 8.41 ± 0.41
  (unrotated) → 129.21 ± 3.90 (90°-rotated), a ~15× degradation;
  G-FNO-p4 test MSE ≈ 4.78 ± 0.39 (equivariant by construction). This
  calibrates the "equivariance violations are empirically detectable by
  L² difference metrics" scale the rule's detector operates on.
  **Calibration-only: not a reproduction claim.** Helwig reports
  downstream 2D Navier–Stokes task relative MSE, not
  `rotate_test`-style equivariance error on an author-constructed
  analytical fixture, and the 2D Navier–Stokes PDE family is outside
  v1.0's Laplace/Poisson/heat/wave scope. The rule's detector classifies
  a C₄-symmetric field as PASS and a C₄-breaking field as WARN/FAIL;
  Helwig Table 3 supplies the scale on which that classification is
  meaningful.

## Citation summary

- **Primary (mathematical-legitimacy)**: Hall 2015 §2.5 + §3.7
  (section-level per `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠);
  Varadarajan 1984 §2.9–2.10 (section-level ⚠).
- **ML-paper structural bridge**: Cohen-Welling ICML 2016,
  arXiv:1602.07576, §3–§4.
- **Calibration (Supplementary)**: Helwig et al. ICML 2023,
  arXiv:2306.05697, Table 3.
- **Pinned value**: verdict-based — PASS on the `cos(2πx) cos(2πy)`
  C₄-symmetric fixture (equivariance error < 1e-12) and WARN/FAIL on
  the `sin(2πx) sin(2πy)` C₄-breaking fixture (equivariance error
  exactly 2.0, since `np.rot90(sin · sin, k=1) = −sin · sin`).
- **Verification date**: 2026-04-23 (Task 1 retrofit). Original Tier-A
  fixture verification 2026-04-20 preserved for fixture construction;
  Task 1 adds the Hall + Varadarajan structural-equivalence F1 layer
  and moves Helwig 2023 to Supplementary calibration context.
- **Verification protocol**: two-layer.
  - **Operator validation (shared harness)**: `fft_laplace_inverse` is
    proven C₄-equivariant to float noise on `sin(2πx) sin(2πy)` in
    `../_harness/tests/test_symmetry.py`. Symbol 1/(k_x² + k_y²) is
    C₄-invariant on a periodic square grid; zero-mode convention
    `û(k=0) = 0` closes the `(−Δ)⁻¹` kernel ambiguity.
  - **Rule anchor (this file)**: direct-fixture construction on 64 × 64
    grid; rotation-table sanity tests before the rule is invoked.

## Pre-execution audit

PH-SYM-001 is a structural-equivalence rule. Per complete-v1.0 plan §6.2
Tier C enumerate-the-splits allocation (0.1 d shared across PH-SYM-001
and PH-SYM-002), the splits audited are: (a) discrete-rotation (Z_n
action; here n=4), (b) discrete-reflection (Z_2 action; covered by
PH-SYM-002), (c) combined D_n dihedral (not split across the two rules).
Outcome: Z_4 is the sole split activated by this rule's fixture and
configuration; D_n dihedral is deferred to v1.2 per the §6.2 audit
outcome. Audit cost absorbed into Task 1 budget.

## Test design

- **Positive fixture:** `u(x,y) = cos(2πx) cos(2πy)` on
  `np.linspace(0,1,64)` — non-trivially C₄-symmetric (not SO(2)-invariant).
  `np.rot90(u, k) ≈ u` to float noise for `k ∈ {1, 2, 3}`.
- **Negative fixture:** `u(x,y) = sin(2πx) sin(2πy)` on the same grid —
  `np.rot90(u, k=1) = −u` exactly.
- **DomainSpec:** `pde="laplace"`, `symmetries.declared=["C4"]`,
  `boundary_condition={kind:"dirichlet"}`, non-periodic.
- **Grid:** 64 × 64.
- **Wall-time budget:** < 15 s (pure-numpy, no FNO inference, no torch).

## Scope note

PH-SYM-001 covers the discrete-rotation-equivariance case (C₄) only.
Continuous-rotation (SO(2)) equivariance is out of v1.0 scope (see Task 6
PH-SYM-003 for the SO(2)-equivariance program). The D_n dihedral case
(rotation + reflection combined) is deferred to v1.2 per complete-v1.0
plan §6.2 enumerate-the-splits audit outcome; PH-SYM-001 and PH-SYM-002
handle the two generators (C₄ rotation, Z_2 reflection) separately in v1.0.

Helwig evaluates on 2D Navier–Stokes and NS-SYM, which are outside v1.0's
Laplace/Poisson/heat/wave PDE scope; the field-invariance detection
mechanism is PDE-agnostic, so rule calibration transfers, but absolute
MSE magnitudes do not.
