# PH-POS-001 external-validation anchor

Evans positivity for Poisson (§2.2.3 Theorem 4 + Positivity corollary, p. 27)
and heat (§2.3.3 Theorem 4). See `CITATION.md` for full provenance and
`AUDIT.md` for the pre-execution enumerate-the-splits verification.

## Run

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-POS-001/ -v
```

Expected: 4 passed in < 10 s.

## Rule reference

PH-POS-001 is a pointwise positivity check: under a boundary condition
that preserves sign (homogeneous Dirichlet with non-negative data; the
heat eigenfunction decay regime), the surrogate's field $u$ must
satisfy $u \ge \mathrm{floor}$ everywhere (default `floor = 0`). The
raw value is the minimum field value; the violation fraction is the
share of cells below the floor.

The rule fires (`FAIL`) when any cell is below the floor. The reported
reason names the cell count and the fraction so the user can
distinguish a single boundary-artefact violation from a systemic
sign-flipping bias. The rule emits `SKIPPED` when the configured
boundary condition does not preserve sign — for example, inhomogeneous
Neumann or a source term that admits sign change.

PH-POS-001's external-validation anchor stack rests on Evans §2.2.3
Theorem 4 (Poisson positivity under homogeneous Dirichlet with $f \ge 0$)
and §2.3.3 Theorem 4 (heat eigenfunction sign preservation). See
`CITATION.md` for the full provenance and `AUDIT.md` for the
pre-execution enumerate-the-splits verification.
