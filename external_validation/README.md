# External Validation

One validated, CI-wired external anchor per benchmark-anchorable rule. See
`docs/superpowers/specs/2026-04-20-external-validation-design.md` for the
design rationale and anchor-selection methodology.

## Tier A (v1.0 ship scope)

| Rule        | Anchor type                        | Harness                              | Citation                                                        |
|-------------|------------------------------------|--------------------------------------|-----------------------------------------------------------------|
| PH-POS-002  | Classical theory reproduction       | `PH-POS-002/test_anchor.py`          | Evans §2.2.3 Theorem 4 — see `PH-POS-002/CITATION.md`           |
| PH-CON-003  | Classical theory reproduction       | `PH-CON-003/test_anchor.py`          | Evans §7.1.2 energy-estimate theorem† — see `PH-CON-003/CITATION.md` |
| PH-SYM-001  | Synthetic + literature calibration  | `PH-SYM-001/test_anchor.py`          | Helwig 2023, Table 3 — see `PH-SYM-001/CITATION.md`              |
| PH-SYM-002  | Synthetic + literature calibration  | `PH-SYM-002/test_anchor.py`          | Helwig 2023, Table 1 — see `PH-SYM-002/CITATION.md`              |
| PH-RES-001  | Convergence + norm-equivalence      | `PH-RES-001/test_anchor.py`          | Fornberg 1988 + Bachmayr-Dahmen-Oster 2025 — see `PH-RES-001/CITATION.md` |
| PH-POS-001  | Classical theory reproduction       | `PH-POS-001/test_anchor.py`          | Evans §2.2.4 (Poisson positivity)† + §2.3.3 (heat weak max principle)† — see `PH-POS-001/CITATION.md` |

**Confidence tiers.** Citations without a dagger are pinned at theorem-number
precision and verified against authoritative web sources (Princeton lecture
notes, Stanford handouts, AMS catalog, peer-reviewed preprints). Citations
marked `†` are verified at **section-level concept** but not at the exact
theorem number inside that section — see `_harness/TEXTBOOK_AVAILABILITY.md`
for the verification record and `docs/backlog/v1.1.md` for the tightening
entry (requires a local copy of Evans to close). Tier-A release ships with
this mix honestly documented rather than delaying on theorem-number precision.

## Tier B (v1.1 roadmap)

| Rule        | Anchor type (planned)               | Harness    | Deferral note                                                    |
|-------------|-------------------------------------|------------|------------------------------------------------------------------|
| PH-RES-002  | CAN-PINN AD-vs-FD cross-check       | _(v1.1)_   | See `docs/backlog/v1.1.md` 2026-04-20 external-validation entries |
| PH-RES-003  | Trefethen spectral accuracy         | _(v1.1)_   | "                                                                |
| PH-BC-001   | PDEBench bRMSE                      | _(v1.1)_   | "                                                                |
| PH-BC-002   | Gauss-Green MMS                     | _(v1.1)_   | "                                                                |
| PH-SYM-003  | Gruver LEE (opt-in, ImageNet)       | _(v1.1)_   | "                                                                |
| PH-SYM-004  | G-CNN translation equivariance      | _(v1.1)_   | "                                                                |
| PH-CON-001  | Hansen ProbConserv mass CE          | _(v1.1)_   | "                                                                |
| PH-CON-002  | Wave energy identity                | _(v1.1)_   | "                                                                |
| PH-CON-004  | Bangerth-Rannacher hotspot          | _(v1.1)_   | "                                                                |
| PH-NUM-001  | Ciarlet quadrature convergence      | _(v1.1)_   | "                                                                |
| PH-NUM-002  | Salari-Knupp refinement             | _(v1.1)_   | "                                                                |
| PH-VAR-002  | Hyperbolic norm-equivalence info    | _(v1.1)_   | "                                                                |

## Running locally

Each Tier-A anchor is pytest-collectable (use `--import-mode=importlib` to
avoid the hyphenated-directory import issue). Example:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-POS-002/ -v
```

The full Tier-A suite runs in under 90 s on CPU:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/ -v --ignore=external_validation/_harness/tests
```

## CI

`.github/workflows/external-validation.yml` runs all six Tier-A anchors as a
matrix on every PR to `master`. Tier-B jobs will be added on push to
`master` and release-tag events once v1.1 execution begins.
