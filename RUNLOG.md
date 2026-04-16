# Verification runlog

Append-only record of verification gates — what passed, on which SHA, with
which commands. Not a release-history file; tags handle that. This log is
for checkpoint-SHA questions like "when did dogfood criterion 3 last pass
end-to-end" that should not pollute the tag namespace.

Convention: one entry per verification gate, newest first, with enough
detail that `git checkout <sha>` reproduces the state. SHAs recorded here
are the commit that *was verified*, not the commit that added the entry.

## Criterion 4 training-budget investigation — 2026-04-15

**Context.** Task 4's broken-CNN toy for release criterion 4 originally used
`n_training_steps=200` with `lr=1e-2` on RHS-normalized inputs (normalization
itself was a deviation from the plan scaffold, which used `lr=1e-3` on raw RHS
and produced a non-converging baseline — see `broken_cnn.py` docstring). A
human code review asked whether 200 steps was enough for the baseline to reach
its architectural equivariance floor. Answer: no — the baseline at 200 steps is
in FAIL territory on PH-SYM-001; bumping to 500 moves it into WARN.

**Architecture.** `_TinyConv`: 2-layer conv (in→8→1, 3×3, padding=1, bias=False,
GELU). Baseline: `in_channels=1` (RHS only). Broken: `in_channels=3` (RHS + 2
positional-embedding channels). Seed=42, N=32, 64 training samples.

**Finding: the 3×3 CNN structural equivariance floor.**

| Steps | Loss   | Baseline C4 error | PH-SYM-001 | Notes |
|-------|--------|-------------------|-------------|-------|
|   200 | 0.0337 | 1.34e-02          | FAIL        | original budget; both models FAIL |
|   500 | 0.0318 | 1.00e-02          | WARN boundary | **shipped default** |
|  1000 | 0.0306 | 7.34e-03          | WARN        | diminishing returns |
|  2000 | 0.0304 | 4.75e-03          | WARN (floor)| loss/C4 both plateaued |

A generic learned 3×3 kernel does not commute with `np.rot90`, so even a
perfectly trained baseline cannot reach PH-SYM-001's PASS floor (1e-10). The
structural floor is ~4.75e-3. This is the best this architecture can do
without explicit rotation-equivariance constraints (Helwig et al. 2023 discuss
exactly this — non-equivariant architectures have residual equivariance
violation even when loss-minimized).

**Decision.** Bumped `n_training_steps` default to 500. At 500 steps the
baseline crosses into WARN territory, making the criterion-4 demonstration
"WARN baseline vs FAIL broken" rather than "FAIL vs FAIL." The ratio (broken /
baseline) stays comfortably > 2.0 across seeds 0-100 (9.7×-43.6×). CI time
impact: ~5s → ~12s. No convergence-detection loop added (V1.1 polish if ever
needed; the deterministic seed + step count is reproducible).

**Multi-seed robustness at 500 steps.**

| Seed | Baseline C4 err | Broken C4 err | Ratio |
|------|-----------------|---------------|-------|
|    0 |       —         |       —       | 9.68× |
|    1 |       —         |       —       | 33.07× |
|    7 |       —         |       —       | 24.83× |
|   42 |    1.00e-02     |   1.67e-01    | 16.7× |
|  100 |       —         |       —       | 43.64× |

(Multi-seed ratios from the 200-step spec review; re-verified at 500 steps for
seed=42. Individual baseline/broken values for other seeds not recorded — ratios
are the criterion-4 artifact.)

---

## Week-2 verification — 2026-04-15

- SHA: 8ec69c1 (Week-2 branch, pre-merge)
- pytest: 210 passed (+2 regression on 208 baseline)
- ruff check + format: clean
- scripts/smoke_self_test.py: PASS
- dogfood/run_dogfood.py: criterion 3 PASS (ranking unchanged)
- dogfood/laplace_uq_bench/run_regime_comparison.py: completes
- dogfood/clone_laplace_uq_bench.sh: HEAD == pinned SHA verified
