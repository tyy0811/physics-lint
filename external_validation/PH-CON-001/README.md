# PH-CON-001 external-validation anchor

**Scope separation (read first):** PH-CON-001 validates the production
rule's ability to measure **integral conservation drift on
analytically controlled source-free snapshots**. It does not certify
the accuracy of a heat-equation time integrator. The F2 fixture is
explicitly analytical-snapshot (`u = cos(2πx) cos(2πy) · exp(−8π²κt)`,
exact solution of 2D periodic heat with zero spatial mean at all t),
not a numerically-evolved FD solution.

Heat mass conservation check: for source-free periodic heat equation,
total mass `∫_Ω u dV` is invariant in `t`. Rule emits
`max_t |M(t) − M(0)| / max(|M(0)|, ‖u_0‖_{L¹})`; shipped-default
relative threshold 1e-4.

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-CON-001/ -v
```

Expected: 17 passed in < 5 s (12 F2 parametrized across Nx ∈ {16, 32,
64, 128} × Nt ∈ {5, 11, 21} + 1 liveness + 1 refinement-invariance
summary + 3 rule-verdict-contract).

Pure numpy — no time-stepper, no mesh assembly, no torch / scikit-fem
dependency.

## Citation stack (three-function-labeled)

See `CITATION.md` for the full function-labeled citation stack.
Summary:

- **F1 Mathematical-legitimacy** (Tier 2 theoretical-plus-multi-paper):
  Evans 2010 §2.3 heat-equation fundamental solution (section-level
  per `../_harness/TEXTBOOK_AVAILABILITY.md` ⚠); Dafermos 2016
  Chapter I balance-law framing (section-level ⚠; DOI
  10.1007/978-3-662-49451-6); 4-step proof-sketch: balance-law
  preconditions → analytical-snapshot fixture → discrete-integral
  invariance on periodic grid → rule's emitted quantity at roundoff.
- **F2 Correctness-fixture (harness-level, analytical-snapshot
  authoritative)**: `external_validation/_harness/energy.py`
  `analytical_heat_snapshot_2d(fixture, nx, nt, kappa, t_end)`.
  Observed mass-drift floor ~1e-18 across full `(Nx, Nt)` sweep;
  acceptance tolerance `1e-15` (1000× safety over observed). Plan's
  original "1e-14" empirically passes but misses the actual physics
  (rule emits sub-epsilon drift, not `O(h^p)` drift). Liveness test:
  `δ = 1e-6` mass injection produces `raw ≈ 2.48e-6`, matching
  theoretical `δ / (2/π)² = 2.47e-6`.
- **F3 Borrowed-credibility**: **absent with justification**. Live
  Hansen ProbConserv reproduction (Hansen 2024 Physica D,
  arXiv:2302.11002, Table 1 ANP row `CE × 10⁻³ = 4.68 (0.10)`)
  requires a `github.com/amazon-science/probconserv` checkpoint
  loader which V1 physics-lint does not ship. Per
  `feedback_precheck_f3_executability_category.md` (2026-04-24), this
  is an F3-INFRA-GAP: pre-demote with semantic-equivalence
  derivation preserved; `external_validation/README.md` already
  scopes PH-CON-001 Hansen reproduction to v1.1.
- **Supplementary calibration context**: Hansen 2024 Physica D
  arXiv:2302.11002 Table 1 diffusion-equation row with
  semantic-equivalence derivation; reproduction deferred to V1.1.

## Plan-diffs (11 cumulative across complete-v1.0 execution)

See `test_anchor.py` module docstring for diffs 9, 10, 11 (introduced
in Task 8). Diffs 1–8 are from Tasks 2, 3, 4, 5. Summary:

1–5. Tasks 2, 3 diffs (see earlier commits).
6.    (Task 5) Plan §13 F=(x,y) → harness-level F2; rule-verdict
      contract on Laplace-harmonic.
7.    (Task 4) Plan §12 "Dirichlet/Neumann/periodic" → Dirichlet-only.
8.    (Task 4) Plan §12 F3-PRESENT PDEBench → F3-absent + PDEBench in
      Supplementary (V1 lacks PDEBench loader).
9.    **(Task 8, this commit)** Plan §16 F3-PRESENT Hansen reproduction
      → F3-absent + Hansen in Supplementary. V1 physics-lint lacks
      ProbConserv loader; gap pre-known per
      `external_validation/README.md` v1.1 tagging; formalized via
      2026-04-24 F3-INFRA-GAP precheck-extension discipline.
10.   **(Task 8, this commit)** Plan §16 "Cosine-IC conservation to
      1e-14" → empirically-measured `ANALYTIC_SNAPSHOT_TOL = 1e-15`.
      Observed floor 1e-18 (3 orders below plan's 1e-14); 1e-15 gives
      ~1000× safety over observed and is 11 orders below the rule's
      1e-4 shipped threshold.
11.   **(Task 8, this commit)** Plan §16 "u_0(x) = cos(2πx) on periodic
      domain" (1D spatial) extended to 2D analog to satisfy rule's
      `ndim ≥ 3 + nt ≥ 3` contract (`ph_con_001.py:52-59`). 2D
      extension preserves the zero-spatial-mean property at every `t`.
