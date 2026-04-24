# PH-SYM-003 external-validation anchor

**Scope separation (read first):** PH-SYM-003 validates **an
infinitesimal Lie-derivative equivariance diagnostic under explicit
SO(2) / smoothness / generator assumptions. It does not prove global
finite equivariance for arbitrary models.**

The v1.0 production rule (`ph_sym_003.py`) is **adapter-mode-only**.
Dump mode emits `SKIPPED` because forward-mode automatic
differentiation on a live callable model is required; a frozen
dumped tensor cannot supply it. In adapter mode, the rule computes
`(L_A f)(x) = d/dθ |_{θ=0} f(R_θ x)` via
`torch.autograd.functional.jvp` and reports the per-point L² norm
against a roundoff floor.

Per 2026-04-24 user-revised Task 6 contract (CRITICAL three-layer
pattern — Tasks 5, 7, 11 precedent `feedback_critical_rule_stub_three_layer_contract.md` —
plus mathematical preflight gate: F1 proof sketch authored before
any test code):

- **F1 Mathematical-legitimacy:** Hall 2015 §2.5 (one-parameter
  subgroup; section-level ⚠) + §3.7 (continuous-to-smooth; section-
  level ⚠) + Varadarajan 1984 §2.9–2.10 (identity-component
  generation; section-level ⚠) + Kondor-Trivedi 2018 compact-group
  equivariance theorem ([arXiv:1802.03690](https://arxiv.org/abs/1802.03690)).
  Six-step structural proof-sketch with explicit assumption statement.
  F1 separates **finite ⇒ infinitesimal** (trivial: differentiate
  along a one-parameter subgroup) from **infinitesimal ⇒ finite**
  (delicate: requires smoothness + connected group + generator
  coverage + exact constraint). The empirical rule measures a
  **single-generator, pointwise-L², scalar-invariant** subset of the
  finite identity; F1 scopes the claim accordingly
  (`feedback_narrower_estimator_than_theorem.md`).

- **F2 harness-level (authoritative):** three case-splits measured
  via `_harness/symmetry.py` `so2_lie_derivative_norm` +
  `finite_small_angle_defect`:

  - Case A positive controls (`identity_scalar_2d`, `radial_scalar`
    with `φ ∈ {exp(−r²), log(1+r²), r², sinc(r²)}`): measured
    `||L_A f||_{per-point L²} = 0.0` exactly (jvp's first-order
    differentiation on a rotation-invariant map produces the
    mathematical zero, not merely roundoff).
  - Case B negative controls (`coord_dependent_scalar_2d` /
    `anisotropic_xx_minus_yy_2d`): measured `||L_A f||` matches the
    closed-form `||−y||_{per-point L²} = 0.5864429587908292` and
    `||−4xy||_{per-point L²} = 1.3756613756613754` to all 16 float64
    digits on a 64 × 64 grid on `[−1, 1]²`.
  - Case C finite-vs-infinitesimal consistency: residual
    `||f(R_ε x) − f(x) − ε · L_A f||_2 / ||ε · L_A f||_2` scales
    linearly in `ε` across `ε ∈ {1e-1, 1e-2, 1e-3, 1e-4}` with a
    coefficient ≈ 0.5 (Taylor second-order remainder), matching the
    theoretical bound.

- **Rule-verdict contract:** adapter-mode `ph_sym_003.check()`
  exercised on five SKIP gates (SO2-not-declared, dump mode,
  non-2D grid, non-origin-centered grid, non-square domain) plus
  live PASS (radial Gaussian wrapped in `CallableField` → rule
  returns PASS with `lie_norm = 0.0`) plus live FAIL
  (`anisotropic_xx_minus_yy_2d` in `CallableField` → rule returns
  FAIL with `lie_norm = 1.376`, matching the closed-form value).
  Differs from Task 7's all-SKIP contract because PH-SYM-003 emits
  live PASS/WARN/FAIL values when gates pass.

- **F3 Borrowed-credibility:** **absent with justification.**
  Codebase grep for `rotmnist|RotMNIST|modal|escnn|e3nn|gruver|
  lie-deriv` returns only a codespell ignore-list row, an unrelated
  Tier-A script, and the rule itself. `pyproject.toml` has no
  `equivariance` optional-dependency group; `.github/workflows/`
  has no Modal trigger or RotMNIST workflow. Plan §14's two-layer
  RotMNIST PR-CI + pre-release Modal A100 validation policy +
  ImageNet-opt-in Gruver reproduction cannot be built in V1 within
  Task 6's budget. Resolution per user's 2026-04-24 revised F3
  contract ("If not, demote to Supplementary"): pre-demoted to
  Supplementary calibration context.

- **Supplementary calibration context:** Cohen-Welling 2016 G-CNN
  ([arXiv:1602.07576](https://arxiv.org/abs/1602.07576); RotMNIST
  2.28%), Weiler-Cesa 2019 E(2)-CNN
  ([arXiv:1911.08251](https://arxiv.org/abs/1911.08251); RotMNIST
  0.705 ± 0.025%), Gruver et al. 2023 LEE
  ([arXiv:2210.02984](https://arxiv.org/abs/2210.02984);
  LEE metric on ImageNet classifiers), Weiler-Forré-Verlinde-
  Welling 2025 GDL monograph — all flagged as theoretical framing,
  **not reproduction**.

**Wording discipline.** Do not write "PH-SYM-003 proves rotation
equivariance" or "PH-SYM-003 tests that a model is SO(2)-
equivariant" or "PH-SYM-003 certifies disconnected-group
symmetries." The rule does none of these. Write: "PH-SYM-003
validates an infinitesimal Lie-derivative equivariance diagnostic
under explicit SO(2) / smoothness / generator assumptions. It does
not prove global finite equivariance for arbitrary models."

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-SYM-003/ -v
```

Expected: 24 passed in < 5 s (1 Case A identity + 4 Case A radial
parametrized + 2 Case B closed-form + 1 Case B distinguishability
+ 4 Case C parametrized + 1 Case C contraction + 1 harness-shape +
5 rule-verdict SKIPs + 3 rule-verdict live PASS/FAIL + 1 wording-
discipline + 1 preflight-audit-exists).

Pure torch on CPU — no Modal, no RotMNIST, no escnn, no e3nn, no
ImageNet.

## Citation stack (three-function-labeled)

See `CITATION.md` for the full function-labeled citation stack.
Summary:

- **F1 Mathematical-legitimacy** (Tier 1 structural-equivalence):
  Hall 2015 §2.5 + §3.7 (section-level ⚠ per
  `../_harness/TEXTBOOK_AVAILABILITY.md`); Varadarajan 1984 §2.9–2.10
  (section-level ⚠); Kondor-Trivedi 2018 compact-group equivariance
  theorem (arXiv:1802.03690). Six-step proof-sketch derives
  `L_A f ≡ 0 ⇔ f` is SO(2)-invariant on the identity component
  (under smoothness + connected + generator-coverage + exact-
  constraint assumptions) via the one-parameter subgroup
  `θ ↦ R_θ = exp(θA)` and the identity-component-generation
  argument. F1 also scopes the rule's claim to the implemented
  narrower-estimator quantity (four subset relationships named:
  infinitesimal-only / scalar-invariant-only / single-generator /
  sampled-grid-only).
- **F2 Correctness-fixture (harness-level, authoritative)**:
  three case-splits on 64 × 64 origin-centered grid on `[−1, 1]²`,
  float64 throughout:
  - Case A positive: `0.0` exactly on identity scalar and four
    radial scalars.
  - Case B negative: `0.586` (`coord_x`) and `1.376` (`x² − y²`)
    matching closed-form `||−y||` and `||−4xy||` to 16 digits.
  - Case C finite-vs-infinitesimal: `defect(ε) / ε ≈ 0.5` constant
    across `ε ∈ {1e-1, ..., 1e-4}` (Taylor second-order coefficient
    of `cos(ε) − 1 ≈ −ε²/2`); successive ratios contract by 10×.
- **Rule-verdict contract**: five SKIP paths + live PASS on radial
  Gaussian + live FAIL on `x² − y²` and `x` maps. Live-path
  `raw_value` numerics match F2 harness measurements to 14+ digits.
- **F3 Borrowed-credibility**: **absent with justification.** No
  CI-executable reproduction target exists for the rule's emitted
  quantity in V1 (no Modal, no RotMNIST, no escnn, no ImageNet).
- **Supplementary calibration context**: Cohen-Welling 2016 /
  Weiler-Cesa 2019 / Gruver 2023 / Weiler-Forré-Verlinde-Welling
  2025 — all theoretical framing flags; **not reproduction**.

## V1 adapter-only scope-truth observation

The production rule's docstring (`ph_sym_003.py:1-15`) and its
gate chain (`ph_sym_003.py:36-68`) document what V1 does NOT cover:

- **Dump mode** (frozen tensor in `GridField`): the single-generator
  Lie derivative requires forward-mode AD on a live callable. The
  rule SKIPs.
- **Non-2D grids, non-origin-centered grids, non-square domains**:
  the rotation action `R_θ` rotates about the origin in 2D; other
  configurations would rotate sample points outside the domain.
  Rule SKIPs.
- **Finite-rotation tests**: V1 evaluates only at `θ = 0`.
- **Non-scalar outputs**: V1 assumes `ρ_Y = identity`. Vector-field
  outputs that rotate with the input would false-PASS.
- **Higher-dim Lie groups** (SO(3), SE(3)): would require multi-
  generator evaluation and BCH compatibility argument.
- **Disconnected groups** (O(2), E(2)): the single-generator
  diagnostic does not detect reflection-component violations.

V1.1 could extend the rule to (a) vector-output equivariance with
representation-tagged outputs, (b) multi-generator Lie groups with a
generator enumeration, (c) finite-rotation sweeps complementing the
infinitesimal diagnostic. These extensions are not in scope for v1.0.

## Pre-execution audit

See `docs/audits/2026-04-24-task-6-preflight.md` for the full
preflight writedown, including:
- Rule-source audit (V1 emitted quantity identified as the per-
  point L² norm of `d/dθ |_{θ=0} f(R_θ x)`; five SKIP gates
  enumerated).
- Finite ⇔ infinitesimal direction separation authored before any
  test code (F1 proof skeleton gate per user's 2026-04-24 contract).
- F3 executability infrastructure audit: `F3-INFRA-GAP`
  classification with resolution = pre-demote to F3-absent +
  Supplementary.
- Expected plan-diffs 30–34 enumerated.

## Plan-diffs (29 cumulative across complete-v1.0 execution before Task 6)

See `test_anchor.py` module docstring for diffs 30–34 (Task 6). Diffs
1–29 are from Tasks 2, 3, 4, 5, 7, 8, 9, 10, 11, 12 (commits
30baf3e, 0cedc7b, 18312b9, 6800d6f, 1112da3, ae1f9a9, 26ed3bd,
84c7163, 87e8a3e, 2ae7d28).

Summary of Task 6 diffs:

30. Plan §14 F3 two-layer RotMNIST CI policy + ImageNet-opt-in
    Gruver reproduction **pre-demoted to F3-absent + Supplementary
    calibration context**. Reason: F3-INFRA-GAP (codebase grep
    returns only a codespell ignore-list row and unrelated Tier-A
    script). Authorized by 2026-04-24 user-revised F3 contract.
31. Plan §14 three-cross-library EMLP + escnn + e3nn correctness
    fixture **replaced with controlled-operator harness** (radial
    positive + coord_x / xx-yy negatives + Case C finite-vs-
    infinitesimal). Same mathematical property with closed-form
    analytical L_A f; avoids unpinned library dependencies (mirrors
    Task 7 plan-diff 23).
32. **CRITICAL three-layer pattern applied** (Tasks 5, 7, 11
    precedent) with mathematical preflight gate. F1 proof skeleton
    authored in `CITATION.md` + `docs/audits/2026-04-24-task-6-
    preflight.md` before any test code. Rule-verdict contract
    exercises live PASS and live FAIL paths (not all-SKIP) because
    ph_sym_003 emits live verdicts when gates pass.
33. **Narrower-estimator-than-theorem scoping applied** (Task 10
    precedent): F1 scope explicitly restricted to infinitesimal-
    LEE-diagnostic-of-scalar-SO(2)-invariance. Four subset
    relationships named (infinitesimal-only / scalar-invariant-
    only / single-generator / sampled-grid-only). F1 does not
    inherit Hall / Varadarajan / Kondor-Trivedi guarantees beyond
    the implemented subset.
34. **Infinitesimal ⇔ finite direction separation** authored
    explicitly in F1 proof skeleton and CITATION.md assumption
    statement per 2026-04-24 user-revised contract. Finite ⇒
    infinitesimal trivial by differentiation; infinitesimal ⇒
    finite delicate under four assumptions (smoothness, connected
    group, generator coverage, exact constraint) and **not claimed
    globally for empirical grid-sampled tests**.
