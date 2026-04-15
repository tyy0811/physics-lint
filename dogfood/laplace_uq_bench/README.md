# laplace-uq-bench dogfood harness — fallback D'

Week-2 deliverable for V1 release criterion 3. This directory contains
the physics-lint dump generators for the four dogfood surrogates, plus
the clone + probe script that bootstraps the upstream repo at
[github.com/tyy0811/laplace-uq-bench](https://github.com/tyy0811/laplace-uq-bench).

## Why fallback D' instead of six trained checkpoints

The Week-2 plan (Task 8) originally specified adapters against the six
trained laplace-uq-bench surrogates (U-Net regressor, FNO, deep
ensemble, OT-CFM, improved DDPM, DPS). The Day-4 discovery is that
**the upstream repo does not ship checkpoints in git**: its
`.gitignore` excludes `*.pt`, training runs on Modal T4 GPUs, and the
`experiments/` directory holds JSON result tables rather than model
weights. Running any of the six surrogates requires Modal credentials
and training compute the physics-lint V1 budget does not cover.

Per the Week-2 plan's fallback D clause, we downgraded the dogfood
harness to "train/construct 2–3 small surrogates inline and downgrade
criterion 3 to a ranking table on ≥3 surrogates." Rather than spinning
2–4 hours of CPU training for a synthetic U-Net, we use the repo's own
finite-difference `LaplaceSolver` as the oracle and build three
deliberate defect variants whose relative severity we can reason about
analytically:

| Surrogate | How it is built | Expected severity |
|-----------|-----------------|-------------------|
| `oracle` | `LaplaceSolver(nx=64).solve(...)` on a canonical BC | lowest |
| `coarsened` | Same solver at `nx=17`, bilinear-upsampled to 64×64 | mid |
| `smoothed` | Oracle convolved with a 5×5 Gaussian (sigma=1.5) | mid |
| `noisy` | Oracle + N(0, 5e-3) Gaussian noise (Dirichlet edge restored) | highest |

The Dirichlet edge is restored on `noisy` and `smoothed` so the
boundary-fidelity rules (PH-BC-001/002) stay silent — the dogfood
probes interior residuals only.

## Running the harness

```bash
# 1. Clone the upstream repo (shallow clone; no checkpoint download).
./dogfood/clone_laplace_uq_bench.sh

# 2. Generate the four dumps.
.venv/bin/python -m dogfood.laplace_uq_bench.generate_dumps

# 3. Rank by physics-lint PH-RES-001 residual.
.venv/bin/python dogfood/run_dogfood.py
```

The orchestrator writes
`dogfood/results/week2-dogfood-table.md` and exits 0 if the observed
ordering matches the expected severity bands.

## Interpreting the statuses

Every dogfood surrogate's PH-RES-001 result is `FAIL` against the
calibrated Laplace L² floor (~1e-12). This is not a bug: the floor is
tuned on the analytical harmonic polynomial `x² − y²`, which is a zero
of −Δ to machine precision. Even a correct 4th-order FD Laplacian
applied to a *2nd-order FD-solved* Laplace field on a 64×64 grid with
O(1) Dirichlet BCs has truncation residual ~10¹², not 10⁰ — the
fallback D' dogfood intentionally stresses a regime above the
analytical-battery floor.

**The criterion is the ranking, not the statuses.** When we eventually
have published H¹ numbers from trained laplace-uq-bench surrogates, the
same harness will report ranking-agreement against those numbers too.

## Fallback D ↔ fallback D'

The plan's fallback D (train 2–3 small surrogates inline) produces
"real" neural surrogates at the cost of 2–4 hours of laptop training.
Fallback D' uses the same scaffolding but substitutes the defect
construction for the training step. Both satisfy the modified
criterion 3 ("ranking table on ≥3 surrogates"); fallback D' is strictly
faster and deterministic. If a future session wants to restore
fallback D, it can swap `generate_dumps.py`'s defect routines for
`scripts/train.py` invocations from the upstream repo without touching
`run_dogfood.py`.
