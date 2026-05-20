# PH-RES-001 external-validation anchor

## Rule reference

PH-RES-001 is a two-path characterization of the rule's residual norm
behavior along two dimensions: spatial convergence (on a refined
sequence of grids) and norm equivalence (between physics-lint's emitted
quantity and a reference norm). The rule is calibrated against
Fornberg 1988 (finite-difference stencil rates) and the
Bachmayr-Dahmen-Oster / Ernst et al. variational-correctness framework;
see `CITATION.md` for the full four-layer structure and provenance.

## Run

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-RES-001/ -v
```

Expected: 12 passed in < 10 s.

Recalibrate Layer 2a bounds (stored in
`fixtures/norm_equivalence_bounds.json`) via:

```bash
source .venv/bin/activate && python external_validation/PH-RES-001/calibrate_bounds.py
```

Re-runs are explicit commits that update `CITATION.md` alongside the
JSON.
