# PH-RES-002 external-validation anchor

AD-computed Laplacian vs 4th-order central FD Laplacian: max interior
relative discrepancy ratio on a smooth MMS fixture, asserted to shrink at
O(h⁴) rate on log-log refinement.

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-RES-002/ -v
```

Expected: 5 passed in < 10 s (slope-in-range, monotone decrease, R² ≥ 0.99,
PASS at every N ∈ {16, 32, 64, 128}, SKIPPED on non-callable dump-mode
field). Torch is required — the AD path runs through
`torch.func.hessian ∘ vmap` per `src/physics_lint/field/callable.py`.

## Citation stack (three-function-labeled)

See `CITATION.md` for the full function-labeled citation stack. Summary:

- **F1 Mathematical-legitimacy** (Tier 2 theoretical-plus-multi-paper):
  Griewank-Walther 2008 Ch. 3 (section-level per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠) for reverse-mode AD
  machine-precision accuracy; LeVeque 2007 (section-level ⚠) for
  4th-order FD stencil consistency; Baydin et al. 2018 JMLR
  (arXiv:1502.05767) as structural-bridge secondary corroboration; 4-step
  AD-vs-FD proof-sketch embedded in CITATION.md.
- **F2 Correctness-fixture**: `sin(π x) sin(π y)` MMS on `[0, 1]²`
  Dirichlet homogeneous; refinement sweep at N ∈ {16, 32, 64, 128};
  measured slope 3.997, R² 1.0000.
- **F3 Borrowed-credibility**: absent with justification. Task 0
  literature-pin pass did not pin a directly-comparable CAN-PINN Chiu
  2022 CMAME row (the CAN-PINN papers report final-solution error
  metrics, not AD-vs-FD Laplacian discrepancy); fallback path per plan
  §10 acceptance-criteria moves CAN-PINN to Supplementary calibration
  context.
- **Supplementary calibration context**: CAN-PINN Chiu et al. 2022 CMAME
  arXiv:2110.14432 (calibration-only — motivates cross-checking AD and
  FD derivatives in neural PDE contexts; not a reproduction claim).
