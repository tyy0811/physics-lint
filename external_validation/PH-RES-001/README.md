# PH-RES-001 external-validation anchor

Two-path characterization of the rule's residual norm behavior along
two dimensions (spatial convergence and norm equivalence). See
`CITATION.md` for full provenance, external references (Fornberg 1988;
Bachmayr-Dahmen-Oster / Ernst et al.), and the four-layer structure.

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
