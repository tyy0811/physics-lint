# PH-RES-003 external-validation anchor

Spectral-backend Laplacian vs 4th-order central FD Laplacian on periodic
grids: max relative discrepancy on an analytic periodic MMS fixture,
asserted to exhibit the published rates (exponential spectral collapse +
polynomial FD) and to PASS the 0.01 rule threshold.

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-RES-003/ -v
```

Expected: 6 passed + 1 skipped in < 5 s (the skipped test is the rapid-
collapse fallback pathway, active only when fewer than 4 spectral points
remain above the float64 noise floor; currently 5/5 points are above).

No torch dependency — the rule's check is pure numpy FFT + FD4 on
periodic `GridField`s (unlike PH-RES-002 which exercised the AD path
through `torch.func.hessian`).

## Citation stack (three-function-labeled)

See `CITATION.md` for the full function-labeled citation stack. Summary:

- **F1 Mathematical-legitimacy** (Tier 3 classical-textbook theorem
  reproduction): Trefethen 2000 Chapters 3–4 (chapter-level per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠; theorem number pending
  local copy) for spectral accuracy on analytic periodic functions;
  Fornberg 1988 (DOI 10.1090/S0025-5718-1988-0935077-0) + LeVeque
  2007 (section-level ⚠) for FD4 interior rate; 4-step proof-sketch
  embedded in CITATION.md.
- **F2 Correctness-fixture**: `exp(sin(x) + sin(y))` on `[0, 2π]²`
  periodic. Layer 1 (spectral, `SPECTRAL_NS=[8,10,12,14,16]`): log-
  linear slope ≈ −1.27, R² = 0.9946, above `FLOAT64_FLOOR = 1e-13`.
  Layer 2 (FD, `FD_NS=[16,32,64]`): log-log slope 3.89, R² = 0.9999.
  Layer 3 (rule, `RULE_NS=[16,32,64]`): PASS at every N.
- **F3 Borrowed-credibility**: absent with justification. Pre-recorded
  in Task 0 Step 5 F3-hunt (`docs/audits/2026-04-22-f3-hunt-results.md`,
  §"Task 3 — PH-RES-003 … TERTIARY"). Trefethen's canonical
  `exp(sin x)` spectral-vs-FD demonstration is a plot, not a
  tabulated reproduction target.
- **Supplementary calibration context**: Canuto-Hussaini-Quarteroni-
  Zang 2006 §2.3 convergence curves (curve-shape framing, not
  reproduction) + Trefethen 2000 Program 5 plot (plot-shape framing,
  not reproduction).

## Plan-diffs (5 cumulative across complete-v1.0 execution)

See `test_anchor.py` module docstring for diffs 3, 4, 5 (introduced in
Task 3). Diffs 1 and 2 are from Task 2 (`PH-RES-002`). Summary:

1. (Task 2) Plan §10 FD stencil order `p ∈ {2, 4}` → rule checks
   `p=4` only on interior band.
2. (Task 2) Plan §10 "F3-absent pre-recorded in Task 0 audit" →
   decided in Task 2 execution per plan §10 fallback path.
3. (Task 3) Plan §11 `SPECTRAL_NS = {16, 32, 64}` with R²>0.99 exp-fit
   → tightened to `SPECTRAL_NS = [8, 10, 12, 14, 16]` above documented
   float64 floor; rapid-collapse fallback replaces R² when fewer than
   4 points remain above floor. User-approved 2026-04-24 per §7.3
   n≥2 escalation outcome. FD_NS and RULE_NS held at {16, 32, 64}.
4. (Task 3) Plan §11 pre-execution audit "V1: 1D only" → 2D
   `exp(sin x + sin y)` fixture (DomainSpec constrains `grid_shape`
   length to [2, 3]).
5. (Task 3) F3-hunt audit text "Trefethen 2000 Chs 3–4 Thm 4" →
   CITATION.md uses chapter-level "Chapters 3–4, theorem number
   pending local copy" per §6.4 + enforced by
   `scripts/check_theorem_number_framing.py`.
