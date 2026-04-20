# PH-SYM-001 external-validation anchor

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-SYM-001/ -v
```

Expected: 4 passed in < 15 s (two fixture-sanity tests + one rule PASS + one
rule WARN/FAIL). Operator-level equivariance of `fft_laplace_inverse` is
validated in `../_harness/tests/test_symmetry.py`; this file tests the rule.
