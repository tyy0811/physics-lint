# PH-RES-001 — Residual exceeds variationally-correct norm threshold

## Function-labeled citation stack

Per complete-v1.0 plan §1.3 three-function labeling. Retrofit landed during
Task 0 Step 7 of the complete-v1.0 plan; Tier-A four-layer content preserved
with the function-labeled structure added as the primary organizational layer.

### Mathematical-legitimacy (Tier 2 theoretical-plus-multi-paper)

- **Layer 1 interior convergence — Fornberg O(h⁴)**: Fornberg, B. (1988).
  "Generation of Finite Difference Formulas on Arbitrarily Spaced Grids."
  *Mathematics of Computation* 51(184):699–706. DOI 10.2307/2008770. The
  `(-1, 16, -30, 16, -1)/12` coefficients at `src/physics_lint/field/grid.py:32`
  are the standard 5-point 4th-order central second-derivative stencil
  derived in that paper.
- **Layer 1b boundary rationale — Strikwerda boundary closure**: Strikwerda,
  J.C. (2004). *Finite Difference Schemes and Partial Differential
  Equations*, 2nd ed. SIAM. ISBN 978-0-89871-567-5. §3 (FD consistency).
  Section-level framing — Strikwerda's verification status is ⚠
  secondary-source-confirmed in `../_harness/TEXTBOOK_AVAILABILITY.md`.
  When an interior O(h⁴) stencil is combined with one-sided/off-center
  boundary closures whose truncation error is O(h²), the full-domain L²
  error inherits the O(h²) boundary contribution (boundary contribution
  scales as O(h^{5/2}); combined with interior O(h⁴) over a log-log
  regression at N ∈ {16, 32, 64, 128}, produces an empirical slope near 3.5).
- **Layer 2 BDO norm-equivalence — Bachmayr-Dahmen-Oster**: Bachmayr, M.,
  Dahmen, W., Oster, P. (2024 preprint / forthcoming). "A variational
  framework for a posteriori error estimation of parabolic PDEs." Ernst, O.,
  Mugler, A., Starkloff, H.-J., Ullmann, E. (2025 v3). "An analysis of a
  posteriori error estimates for parabolic PDEs." The H⁻¹ spectral residual
  norm is H¹-equivalent on periodic spectral grids by BDO; the L² fallback
  is not H¹-equivalent by construction.

### Correctness-fixture (CI-runnable, non-credibility-claiming)

- **Layer 1 fixture**: `sin_sin_mms_square()` (analytical source
  `f = -Δu = 2π² sin(πx) sin(πy)` on `[0, 1]²`, Dirichlet homogeneous).
  `u_exact = sin(πx) sin(πy)` is closed-form; the residual is computed by
  substitution. `test_anchor.py` sweeps `N ∈ {16, 32, 64, 128}` and fits
  log-log regression.
- **Layer 2a fixture**: `periodic_sin_sin()` on `[0, 2π]²` with three
  perturbations `p_k(x, y) = 0.01 sin(kx) sin(ky)` for `k ∈ {1, 2, 3}`.
  Grid 64×64, `endpoint=False` (periodic convention). Perturbation H¹
  norms are closed-form-derivable from `||p_k||_{H¹} = O(k · ε)` and the
  rule's H⁻¹ residual norm is computed via FFT.
- **Layer 2b fixture**: `sin_sin_mms_square()` on `[0, 1]²` (same as Layer
  1) with perturbations `sin(kπx) sin(kπy)` for `k ∈ {1, 4}`. Closed-form
  k-linear scaling prediction: `||Lap p_k||_{L²} ∝ k²`, `||p_k||_{H¹} ∝ k`
  (gradient-dominated), so `ρ_k = ||r||_{L²} / ||p_k||_{H¹} ∝ k`.
- Pinned bounds stored in `fixtures/norm_equivalence_bounds.json` (Layer 2a);
  Layer 2b is a range-check on the k-linear scaling ratio.

### Borrowed-credibility (external published reproduction layers)

- **Layer 1a — Fornberg O(h⁴) interior reproduction.** log-log slope of
  interior residual on MMS `sin(πx)sin(πy)` at `N ∈ {16, 32, 64, 128}` must
  land in `[3.8, 4.2]` with `R² ≥ 0.99`. **Measured value 3.993 at commit
  c07ba33.** This reproduces Fornberg 1988's O(h⁴) central-difference
  stencil prediction as a measured empirical rate.
- **Layer 2a — BDO `C_max / c_min < 10` norm-equivalence reproduction.**
  On the three-perturbation family (`k ∈ {1, 2, 3}`) on `[0, 2π]²`,
  `ρ_k = ||r||_{H^-1} / ||u_pert - u_exact||_{H¹}` satisfies
  `c_min ≤ ρ_k ≤ C_max` with `C_max / c_min < 10` per BDO.
  **Measured 4.829 at commit e28c493.** Stored in
  `fixtures/norm_equivalence_bounds.json`.
- **Layer 1b — Full-domain characterization (not reproduction).** log-log
  slope in `[3.3, 3.7]`. Measured value 3.494 at commit c07ba33. This is
  *characterization* of the documented boundary-closure regime, not
  reproduction of a specific peer-reviewed rate. Listed here under F3
  because its pinned numerical range is a direct output comparison against
  the Strikwerda §3 closure-theory prediction, but flagged explicitly as
  characterization not reproduction to preserve the §1.2 F3 sharpness.
- **Layer 2b — L² fallback k-linear characterization (not reproduction).**
  `ρ_k=4 / ρ_k=1 ∈ [3.5, 4.5]` on non-periodic+FD path. Measured 4.119 at
  commit e28c493. Characterization of the rule's documented L²-fallback
  regime per closed-form scaling derivation; not a reproduction of a
  specific peer-reviewed table.

### Supplementary calibration context

(None — all external references above are either F1 mathematical-legitimacy
(Fornberg, Strikwerda, BDO papers) or F3 borrowed-credibility (measured
reproductions and characterizations against those theorems' predictions).
No calibration-only or plot-shape references accompany this rule.)

## Citation summary

- **Layer 1 (Fornberg interior O(h⁴)):** Fornberg, "Generation of Finite
  Difference Formulas on Arbitrarily Spaced Grids," *Mathematics of
  Computation* 51(184):699–706, 1988. The (-1, 16, -30, 16, -1)/12
  coefficients at `src/physics_lint/field/grid.py:32` are the standard
  5-point 4th-order central second-derivative stencil derived in that
  paper.
- **Layer 1b boundary rationale:** Strikwerda, *Finite Difference Schemes
  and PDEs* (SIAM Classics 2004), §3.
- **Layer 2 (BDO norm-equivalence):** Bachmayr, Dahmen, Oster, "A
  variational framework for posteriori error estimation of parabolic
  PDEs," 2024 preprint; Ernst, Mugler, Starkloff, Ullmann, "An analysis
  of a posteriori error estimates for parabolic PDEs" (v3, 2025). The
  H⁻¹ spectral residual norm is H¹-equivalent on periodic spectral grids
  by BDO; the L² fallback is not H¹-equivalent by construction.
- **Pinned values:** per-path, measured rather than a priori —
  - **Layer 1a (interior Fornberg):** log-log slope of interior residual
    on MMS `sin(πx)sin(πy)` at N ∈ {16, 32, 64, 128} must land in
    [3.8, 4.2] with R² ≥ 0.99. Measured value 3.993 at commit c07ba33.
  - **Layer 1b (full-domain characterization):** log-log slope of the
    rule's full-domain L² residual on the same fixture must land in
    [3.3, 3.7] (characterization, not a Fornberg reproduction; the
    boundary O(h²) contribution pulls the rate down from 4). Measured
    value 3.494 at commit c07ba33.
  - **Layer 2a (periodic+spectral H⁻¹):** BDO norm-equivalence ratio
    `C_max / c_min < 10` on the three-perturbation family
    (k ∈ {1, 2, 3}) on [0, 2π]². Measured 4.829 at commit e28c493.
    Stored in `fixtures/norm_equivalence_bounds.json`.
  - **Layer 2b (non-periodic+FD L² fallback):** characterization, not
    reproduction. The rule's L² residual norm satisfies
    `rho(k=4) / rho(k=1) ∈ [3.5, 4.5]` on sin(kπx)sin(kπy)
    perturbations, confirming the predicted k-linear scaling
    (the rule's L² path is *not* H¹-equivalent across frequencies by
    construction; see below). Measured 4.119 at commit e28c493.
- **Verification date:** 2026-04-21.
- **Verification protocol:** two-layer reproduction + two-layer
  characterization. See `test_anchor.py` for the four-layer structure.

## Test design

The rule `PH-RES-001` emits **different norms on different configurations**
by design (see `src/physics_lint/rules/ph_res_001.py:14-20` module
docstring): on periodic+spectral grids it emits `H^-1` (variationally
correct for the Poisson/heat strong form); on non-periodic+FD grids it
falls back to L² to preserve every-mode residual detection, because
`h_minus_one_spectral` drops the DC mode. The fallback is a documented
rule-design choice, not a bug, and is surfaced per-result via
`recommended_norm` so downstream reports can flag the approximate
measurement.

This anchor characterizes **both code paths** along **both relevant
dimensions** (spatial convergence rate and norm equivalence):

| Layer | Code path | Dimension | Status | Tests |
|---|---|---|---|---|
| 1a | non-periodic+FD | spatial convergence (interior stencil) | reproduction (Fornberg O(h⁴)) | 3 |
| 1b | non-periodic+FD | spatial convergence (full domain) | characterization | 2 |
| 2a | periodic+spectral | norm equivalence (H⁻¹ vs H¹) | reproduction (BDO C_max/c_min < 10) | 5 |
| 2b | non-periodic+FD | norm behavior (L² under wavenumber sweep) | characterization | 2 |

Total: 12 anchor tests. Layer 2a's pinned bounds (`c_min`, `C_max`) are
serialized to `fixtures/norm_equivalence_bounds.json` from
`calibrate_bounds.py`. Layer 2b's k-linear assertion is a direct
range check in `test_anchor.py` and does not require fixture storage.

### Layer 1 (spatial convergence)

- **Fixture:** `sin_sin_mms_square()` (analytical source
  `f = -Δu = 2π² sin(πx) sin(πy)` on `[0, 1]²`, Dirichlet homogeneous).
- **Grid sweep:** `N ∈ {16, 32, 64, 128}`.
- **Layer 1a residual:** `-Laplacian(u_fd) - source` restricted to the
  [2:-2, 2:-2] interior band where FD4 central stencil applies
  uniformly. L² norm taken on that interior sub-domain.
- **Layer 1b residual:** the rule's `raw_value` (full-domain L² with
  half-weighted boundary trapezoidal quadrature).

### Layer 2 (norm equivalence)

- **Layer 2a fixture:** `periodic_sin_sin()` on `[0, 2π]²` with three
  perturbations `p_k(x, y) = 0.01 sin(kx) sin(ky)` for `k ∈ {1, 2, 3}`.
  Grid 64×64, `endpoint=False` (periodic convention).
- **Layer 2a quantity:** `ρ_k = ||r||_{H^-1} / ||u_pert - u_exact||_{H¹}`.
  BDO theorem asserts `c_min ≤ ρ_k ≤ C_max` with `C_max / c_min < 10`.
- **Layer 2b fixture:** `sin_sin_mms_square()` on `[0, 1]²` (same as
  Layer 1) with perturbations `sin(kπx) sin(kπy)` for `k ∈ {1, 4}`
  (two-point k-scaling check).
- **Layer 2b quantity:** `ρ_k = ||r||_{L²} / ||p_k||_{H¹}`.
  Predicted scaling (closed-form): `||Lap p||_{L²} ∝ k²`,
  `||p||_{H¹} ∝ k` (gradient-dominated), so `ρ_k ∝ k` ⇒
  `ρ_k=4 / ρ_k=1 ≈ 4`.

## Acceptance criteria

### Layer 1a (Fornberg interior reproduction)

- Interior residual log-log regression slope in [3.8, 4.2].
- R² ≥ 0.99 on the interior regression.
- Interior residual monotonically decreases across the four N values.

### Layer 1b (full-domain characterization)

- Full-domain residual log-log regression slope in [3.3, 3.7]. Asserted
  as a characterization range, *not* a Fornberg-derived target — the
  rate is set by the boundary-stencil regime, not by the interior
  stencil's theoretical order.
- Full-domain residual monotonically decreases across N.

### Layer 2a (BDO norm-equivalence reproduction)

- Rule emits `recommended_norm == "H-1"` on periodic+spectral
  configuration (prerequisite — if this fails the BDO claim cannot be
  tested).
- Each `ρ_k` for `k ∈ {1, 2, 3}` lies within the calibrated
  `[c_min, C_max]` bounds.
- `C_max / c_min < 10` on the calibrated family (BDO norm-equivalence
  ratio).

### Layer 2b (L² fallback characterization)

- Rule emits `recommended_norm == "L2"` on non-periodic+FD configuration
  (prerequisite — if this fails, the rule is not on its documented
  fallback path).
- `ρ_k=4 / ρ_k=1 ∈ [3.5, 4.5]` on the non-periodic+FD path, confirming
  the k-linear scaling.

### Suite-level

- Wall-time < 10 s on CPU (measured 6.14 s for the full external-
  validation suite including harness).

## Scope and deliberate out-of-scope

**What this anchor reproduces.** The rule's interior FD4 Laplacian
achieves textbook Fornberg O(h⁴) convergence (Layer 1a). The rule's
periodic+spectral H⁻¹ path satisfies BDO norm-equivalence with a
measured `C_max/c_min = 4.83` on the three-perturbation family, well
below the theoretical requirement `< 10` (Layer 2a).

**What this anchor does NOT reproduce.** The original Tier-A plan
specified a single pinned number `C_max/c_min < 10` on the BDO norm-
equivalence reproduction. Execution surfaced that the rule has two
code paths (periodic+spectral vs non-periodic+FD) emitting different
norms, and the BDO claim holds only on the periodic+spectral path.
The non-periodic+FD L² fallback is *not* H¹-equivalent across
frequencies by construction: `ρ_k` scales linearly with perturbation
wavenumber k because `||Lap p_k||_{L²} / ||p_k||_{H¹}` is proportional
to k for `p_k = sin(kπx) sin(kπy)`. This is a **rule-design limitation,
not a bug** — the L² fallback exists because the spectral H⁻¹ norm
drops the DC mode and would silently report constant-in-space residuals
as zero on non-periodic grids (see `ph_res_001.py:14-20` module
docstring for the full rationale).

**Layer 2b's role.** Rather than hide the fallback or route around it,
Layer 2b positively characterizes the L² path's k-linear scaling as an
assertion the rule's documented behavior must satisfy. A future rule
change that broke the k-linear scaling would be caught by Layer 2b
regardless of whether it produced a plausible-looking single number.
This is the opposite of a self-passing anchor: the L² path's
non-equivalence is named, explained, and turned into a test.

**Rev 1.7.2 Path A′ pattern.** Each layer pairs an external-reference
reproduction on the rule's clean regime (Fornberg interior, spectral
H⁻¹) with a positive characterization of its documented fallback regime
(boundary L² contribution, L² non-periodic+FD k-linear scaling).
Together they surface the rule's two-regime structure honestly rather
than pinning a single number that would hold only on one regime. This
pattern is referenced from `feedback_deviation_pattern_escalation.md`
and is proposed as a template for Tier-B continuous-math rules (see
`docs/backlog/v1.1.md` PH-NUM-002 Post-Tier-A finding for the
Tier-B application).

## Commit trail

The anchor landed across four commits on `external-validation-tier-a`:

- `0ba7ae9 (feat(external-validation): MMS H1-error helper for Task 4 Layer 2)` — initial `mms_sin_sin_h1_error` helper supporting non-periodic MMS H¹ error computation.
- `e28c493 (feat(external-validation): PH-RES-001 Layer 2 two-path norm characterization)` — Layer 2a (periodic+spectral H⁻¹ BDO reproduction) + Layer 2b (non-periodic+FD L² fallback characterization). Also extends `mms.py` with `periodic=True` support (`np.roll`-based central diffs + endpoint-exclusive rectangle rule) and renames `mms_sin_sin_h1_error` → `mms_perturbation_h1_error`.
- `c07ba33 (feat(external-validation): PH-RES-001 Layer 1 Fornberg + full-domain characterization)` — Layer 1a (Fornberg interior O(h⁴) reproduction) + Layer 1b (full-domain boundary-influenced characterization).
- This file's landing commit — adds `CITATION.md` + `README.md` per parallel with the other five Tier-A anchors. The factual gap (shipped main README claimed per-rule `CITATION.md` provenance existed for all six anchors; PH-RES-001 was the one that didn't yet) was caught by Codex adversarial review on 2026-04-21.

## Dependency: rule-source module docstring

Layer 2b's characterization presupposes the rule's L² fallback behaves
as documented at `src/physics_lint/rules/ph_res_001.py:14-20`. If the
fallback path semantics change (e.g., a future rule version emits a
different norm on non-periodic+FD, or the DC-mode handling changes),
Layer 2b's k-linear assertion may fail even when the new behavior is
correct — in which case this CITATION.md and the Layer 2b test
assertions must be revisited together, not patched independently.
