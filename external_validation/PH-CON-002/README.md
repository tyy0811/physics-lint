# PH-CON-002 external-validation anchor

**Scope separation (read first):** PH-CON-002 validates the production
rule's ability to measure **wave-energy drift on analytically
controlled conservative snapshots**. It does not certify the accuracy
of a wave-equation time integrator.

The F2 fixture splits into two distinct layers:

- **F2 harness-level (authoritative):** `E(t)` computed directly from
  analytical `u_t, u_x, u_y` — roundoff-only drift.
- **Rule-verdict contract:** analytical `u` snapshots fed to the
  rule; rule computes `u_t` via 2nd-order central FD internally →
  `O(Δt²)` drift with method-dependent tolerance.

Plan §17's "leapfrog time-stepper over 1000 steps with 1e-8
tolerance" is **not implemented** in V1 — PH-CON-002 does not use a
leapfrog stepper (it uses central-FD on supplied `u` snapshots), and
any numerically-evolved fixture would be supplementary with its own
tolerance per user's revised Task 9 contract.

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-CON-002/ -v
```

Expected: 17 passed in < 10 s (11 F2-harness incl. parametrized
sweep `{16,32,64} × {11,21,51}` and E(0) scaling checks + 3 rule-
verdict method-dependent + 3 SKIP-path contracts).

Pure numpy — no mesh assembly, no torch / scikit-fem.

## Citation stack (three-function-labeled)

See `CITATION.md` for the full function-labeled citation stack.
Summary:

- **F1 Mathematical-legitimacy** (Tier 2 theoretical-plus-multi-
  paper): Evans 2010 §2.4.3 energy identity (section-level per
  `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠); Strauss 2007 §2.2
  (section-level ⚠); Hairer-Lubich-Wanner 2006 Ch IX symplectic-
  integrator conservation (section-level ⚠); 4-step proof-sketch
  with IBP derivation of `dE/dt = 0`.
- **F2 Correctness-fixture (harness-level, authoritative)**:
  `external_validation/_harness/energy.py`
  `analytical_wave_snapshot_2d_yindep` + `wave_energy_from_
  analytical_fields`. Measured max relative drift 5.4e-16 across
  full `Nx × Nt` sweep; tolerance `1e-14`. `E(0) = π²c²k²` matches
  analytical value.
- **Rule-verdict contract**: analytical `u` snapshots fed to rule
  give `O(Δt²)` drift (measured slope 1.94; rule PASSes at
  `Δt ≤ π/25` on one wave period at c=k=1). Refinement-invariant
  on spatial `nx`.
- **F3 Borrowed-credibility**: **absent with justification** (pre-
  recorded by Task 0 Step 4 pin audit). PDEBench has no
  wave-equation dataset; shallow-water and compressible-NS rows
  measure mass, not wave-energy. Hansen ProbConserv CE is
  first-order-in-time, not second-order energy functional. Both
  structurally incompatible — no F3-INFRA-GAP (no loader would
  help).
- **Supplementary calibration context**: PDEBench shallow-water
  cRMSE (mass, flagged); Hansen CE (first-order, flagged).

## Plan-diffs (13 cumulative across complete-v1.0 execution)

See `test_anchor.py` module docstring for diffs 12 and 13 (Task 9).
Diffs 1–11 are from Tasks 2, 3, 4, 5, 8. Summary:

1–11. See prior commits.
12.   **(Task 9, this commit)** Plan §17 F2 "`u = sin(kx)cos(ckt)`
      with leapfrog time-stepper over 1000 steps; assert E(t)
      bounded within 1e-8" split into two-layer analytical-snapshot
      F2 + rule-verdict contract per user-approved revised Task 9
      contract. No leapfrog stepper is implemented in V1 (the rule
      itself uses central-FD on supplied `u`; leapfrog would be a
      separate anchor). Tolerance is empirically-measured per layer:
      harness-level `1e-14` (~20× margin over observed 5e-16);
      rule-verdict `O(Δt²)` method-dependent.
13.   **(Task 9, this commit)** Plan §17 F2 fixture "`u = sin(kx)
      cos(ckt)`" (1D) extended to 2D y-independent analog to satisfy
      rule's `ndim ≥ 3 + nt ≥ 3` contract (`ph_con_002.py:51-57`).
      Energy invariance preserved (E factors as `E_x · 2π`).
