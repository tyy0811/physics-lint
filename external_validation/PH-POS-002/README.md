# PH-POS-002 external-validation anchor

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-POS-002/ -v
```

Expected: 4 passed in < 5 s.

See `CITATION.md` for the external reference, `test_anchor.py` for the
test harness, and `../README.md` for the full-program context.

## Rule reference

PH-POS-002 enforces the maximum principle for harmonic functions:
under a well-posed Dirichlet problem for $-\Delta u = 0$, the minimum
and maximum of $u$ are attained on the boundary. The rule reads the
boundary values supplied alongside the field and compares them against
the interior min/max. The raw value is the interior **overshoot** —
the magnitude by which the interior extremum exceeds the boundary
extrema (zero when the maximum principle holds).

The rule fires (`FAIL`) when the overshoot exceeds a tight tolerance
(`1e-10`). This is a hard check, not a noise-floor-banded one: any
finite-precision-significant overshoot signals a spurious interior
extremum. The rule emits `SKIPPED` when the configured PDE is not
`laplace`, since the maximum principle does not extend to other
operators without restatement.

PH-POS-002 is the rule whose firing is shown in the README hero
screenshot: it catches the FNO baseline on a Dirichlet-homogeneous
Laplace problem in the laplace-uq-bench dogfood corpus.
