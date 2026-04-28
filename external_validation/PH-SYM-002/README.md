# PH-SYM-002 external-validation anchor

Discrete reflection equivariance (Z₂ action on each of two independent axes
of a 2D periodic square grid).

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-SYM-002/ -v
```

Expected: 4 passed in < 15 s (two fixture-sanity tests + one rule PASS + one
rule WARN/FAIL). Operator-level equivariance of `fft_laplace_inverse` is
validated in `../_harness/tests/test_symmetry.py`; this file tests the rule.

## Citation stack (three-function-labeled)

See `CITATION.md` for the full function-labeled citation stack. Summary:

- **F1 Mathematical-legitimacy** (Tier 1 structural-equivalence): Hall 2015
  §2.5 + §3.7 (section-level per `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠);
  Varadarajan 1984 §2.9–2.10 (section-level ⚠); Cohen-Welling ICML 2016
  G-equivariant CNN structural bridge; Z₂-on-O(2) structural-equivalence
  proof-sketch embedded in CITATION.md.
- **F2 Correctness-fixture**: `cos(2πx) cos(2πy)` reflection-invariant
  positive fixture; `sin(2πx) sin(2πy)` reflection-breaking negative
  fixture; operator-level validation of `fft_laplace_inverse` in
  `../_harness/tests/test_symmetry.py`.
- **F3 Borrowed-credibility**: absent with justification — no published
  numerical baseline exists for discrete-reflection equivariance-error on
  physics-lint's `reflect_test`-style emitted quantity. Per complete-v1.0
  plan §1.2, F3-absent-is-structural for analytical discrete-predicate
  rules.
- **Supplementary calibration context**: Helwig et al. ICML 2023 Table 1
  (calibration-only — scale on which equivariance violations are detectable;
  not a reproduction claim).
