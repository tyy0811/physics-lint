# PH-SYM-004 external-validation anchor

**Scope separation (read first):** PH-SYM-004 validates **the
mathematical and harness-level translation-equivariance contract for
controlled operators**. The v1.0 production rule validates only its
implemented rule-verdict behavior — which, in V1, is a **structural
stub that always emits `SKIPPED`** once past its declared-symmetry +
periodicity gates (`ph_sym_004.py:36-52`). True translation
equivariance is a *model property* (`f(roll(x)) == roll(f(x))` on a
live callable) and requires adapter-mode plumbing deferred to V1.1.

Per 2026-04-24 user-revised Task 7 contract (CRITICAL three-layer
pattern, Task 5 precedent `feedback_critical_rule_stub_three_layer_contract.md`):

- **F1 Mathematical-legitimacy:** Kondor-Trivedi 2018 compact-group
  equivariance theorem ([arXiv:1802.03690](https://arxiv.org/abs/1802.03690))
  + Li et al. 2021 FNO §2 convolution theorem
  ([arXiv:2010.08895](https://arxiv.org/abs/2010.08895)). Five-step
  structural proof-sketch with explicit assumption statement (periodic
  domain, grid-aligned shifts, same input/output grid, consistent
  translation action, no boundary artifacts unless deliberately tested).
- **F2 harness-level (authoritative):** four controlled operators
  measured via `_harness/symmetry.py` `shift_commutation_error`.
  Identity + circular convolution (1D, 2D) + Fourier multiplier
  (1D, 2D) verified equivariant at float64 roundoff (max 3.75e-16
  over 100 random 2D trials). Coordinate-dependent multiplication
  (1D, 2D) verified non-equivariant (error 9e-02 to 9e-01 across
  100 random 2D trials) — required negative control per the
  2026-04-24 contract.
- **Rule-verdict contract:** V1 stub `ph_sym_004.check()` SKIPs on
  all three gate branches (not-declared / non-periodic / V1-stub);
  anchor verifies each SKIP reason string.
- **F3 Borrowed-credibility:** absent-with-justification. No CI-
  executable reproduction target exists for the rule's emitted
  quantity (which is `SKIPPED` in V1). Per user's 2026-04-24 revised
  F3 contract: "Do not force borrowed credibility by citing theory as
  if it were an executable reproduction." Helwig 2023 §2.2 Lemma 3.1
  moved to Supplementary calibration context per plan §15.
- **Supplementary calibration context:** Helwig 2023 GERNS
  ([arXiv:2306.05697](https://arxiv.org/abs/2306.05697)) + Cohen-Welling
  G-CNN ([arXiv:1602.07576](https://arxiv.org/abs/1602.07576)),
  both flagged as theoretical/pedagogical framing — not reproduction.

**Wording discipline.** Do not write "PH-SYM-004 validates translation
equivariance of FNO layers." The production rule does not; it is a
structural stub. Write: "PH-SYM-004 validates the mathematical and
harness-level translation-equivariance contract for controlled
operators. The v1.0 production rule validates only its implemented
rule-verdict behavior."

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-SYM-004/ -v
```

Expected: 36 passed in < 5 s (12 Case A 1D parametrized + 12 Case A 2D
parametrized + 1 Case A 2D stability + 3 Case B 1D parametrized +
3 Case B 2D parametrized + 1 Case B 2D stability + 3 rule-verdict SKIP
+ 1 rule-verdict invariance).

Pure torch on CPU — no FNO / neuraloperator / pytorch_fno dependency.

## Citation stack (three-function-labeled)

See `CITATION.md` for the full function-labeled citation stack.
Summary:

- **F1 Mathematical-legitimacy** (Tier 1 structural): Kondor-Trivedi
  2018 main theorem (arXiv:1802.03690) — `G`-equivariant linear maps
  between `G`-input/output feature spaces are exactly generalized
  convolutions; specializes to `Z_N` circular convolution on periodic
  grids. Li et al. 2021 FNO §2 (arXiv:2010.08895) — spectral
  convolution `K(u) = F^{-1}(R · F(u))` is translation-equivariant on
  a periodic domain via the convolution theorem. Five-step proof-
  sketch explicitly derives `K(T_s u) = T_s K(u)` from the shift
  theorem `F(T_s u)[k] = e^{-2πi·k·s/N} F(u)[k]`.
- **F2 Correctness-fixture (harness-level, authoritative)**:
  `external_validation/_harness/symmetry.py` additions. Measured max
  `shift_commutation_error` across 100 random 2D trials:
  identity = 0 exactly; circular_convolution = 3.14e-16;
  fourier_multiplier = 3.75e-16; coord_dep_mul = 8.69e-01 (min
  9.17e-02 — always clearly non-zero). Tolerance bands: equivariant
  `≤ 1e-14`, non-equivariant `> 0.05`.
- **Rule-verdict contract**: rule SKIPs on all V1 code paths — anchor
  verifies all three SKIP reasons match `ph_sym_004.py:36-52`.
- **F3 Borrowed-credibility**: **absent with justification** per
  user's 2026-04-24 revised F3 contract. No CI-executable
  reproduction target exists; Helwig 2023 and Li 2021 cited in F1 as
  theoretical backbone + Supplementary as pedagogical framing.
- **Supplementary calibration context**: Helwig 2023 §2.2 Lemma 3.1
  (theoretical framing flag); Cohen-Welling 2016 G-CNN (pedagogical
  framing flag).

## V1 stub scope-truth observation

The production rule's docstring (`ph_sym_004.py:1-21`) explicitly
documents why the V1 offline metric
`||roll(u) − u|| / ||roll(u)||` was **removed** rather than shipped:
`np.roll` preserves norm, so the triangle inequality caps the offline
quantity at 2.0 and a PASS-if-<2.0 threshold would rubber-stamp random
noise, smooth ramps, and most structured fields. The false-pass
metric was removed. The V1 behavior is therefore `SKIPPED` on all
realistic inputs — correct-by-design.

V1.1 will replace the stub with an adapter-mode implementation that
compares `f(roll(x))` against `roll(f(x))` on a live callable, using
the same `shift_commutation_error` primitive that this anchor's F2
layer already exercises. When that lands, this anchor's rule-verdict
contract must be updated in the same commit to switch from "rule
SKIPs with V1-stub reason" to "rule emits numerical equivariance
error that matches harness-level measurement."

## Plan-diffs (26 cumulative across complete-v1.0 execution)

See `test_anchor.py` module docstring for diffs 23–26 (Task 7). Diffs
1–22 are from Tasks 2, 3, 4, 5, 8, 9, 10, 12. Summary of Task 7 diffs:

23. Plan §15 step 3 "random FNO-layer + random input + random
    grid-shift fixture; assert commutation error < 1e-5" → replaced
    with controlled-operator harness (identity + circular conv +
    Fourier multiplier + coord-dependent-multiplication negative
    control). Per 2026-04-24 user-revised Task 7 F2 contract ("Use
    known translation-equivariant operators; also include a
    deliberately non-equivariant operator"). Avoids
    `neuraloperator` / `pytorch_fno` dependency; same mathematical
    property (convolution-theorem-based equivariance) with simpler
    pure-torch operators.
24. Plan §15 step 3 tolerance `< 1e-5` → measured `≤ 1e-14` across
    100 random 2D trials on controlled harness operators (float64
    roundoff at 3.75e-16 max; ~30× safety at 1e-14). Plan's looser
    tolerance was calibrated for float32 FNO-layer testing; harness
    operators easily beat it.
25. CRITICAL three-layer pattern applied (Task 5 precedent): rule-
    verdict contract layer added to verify all three V1-stub SKIP
    branches (not-declared / non-periodic / V1-stub deferral),
    matching `ph_sym_004.py:36-52` reason strings. Anchor document
    explicitly states the rule does not compute
    `shift_commutation_error` in V1.
26. Plan §15 F3 already-absent status reinforced per user's
    2026-04-24 revised F3 contract: Helwig 2023 §2.2 Lemma 3.1
    moved to Supplementary calibration context with explicit
    "theoretical framing, not reproduction" flag. No attempt to
    promote theoretical lemma to reproduction layer.
