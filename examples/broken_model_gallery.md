---
jupyter:
  jupytext:
    formats: md:myst,ipynb
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# physics-lint broken-model gallery

Three cases where MSE ranking and physics-lint ranking disagree, each
highlighting a failure mode physics-lint catches that MSE does not.

- **Case 1:** over-smoothed solution with tiny MSE but boundary violation
  (`PH-BC-001`) — *constructed pathology*.
- **Case 2:** under-trained-style prediction with small MSE but positivity
  violation (`PH-POS-001`) — *constructed pathology*.
- **Case 3:** non-equivariant CNN passing MSE but failing C4 symmetry
  (`PH-SYM-001`) — *real trained model* (`physics_lint.validation.broken_cnn`).

**On the framing.** Cases 1 and 2 are synthetic prediction arrays labelled
after real failure modes seen in trained neural PDE surrogates. Case 1's
construction (Gaussian bump that leaks onto the boundary) stands in for the
over-smoothing behaviour actually observed on FNO in Week 2½'s
laplace-uq-bench dogfood (FNO `bc_err` = 0.2088, ~150× the DDPM baseline —
see `dogfood/dogfood_real_results.md`). Case 2's construction (negative
patch in an interior region) stands in for under-training behaviour that
produces localized positivity violations. Case 3 is a real trained model:
`broken_cnn` trains two 2-layer CNNs on the same loss, one baseline and one
with positional-embedding inputs that break rotational equivariance. MSE is
comparable; `PH-SYM-001` picks up the C4 failure.

Using synthetic constructions for Cases 1-2 keeps the gallery self-contained
(no checkpoint download, no GPU) while still demonstrating the rule
mechanics. Real-model extension is tracked in `docs/backlog/v1.1.md`
(6-surrogate restoration + marketing scatter plot entries).

```{code-cell} ipython3
import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_bc_001, ph_pos_001
```

## Case 1: over-smoothed prediction with tiny MSE but boundary violation

The prediction interpolates the interior accurately but does not respect the
homogeneous Dirichlet BC. A pure-MSE check rates this highly because the
residual against the target is small in the L² sense; `PH-BC-001` in absolute
mode picks up the boundary leak and fails.

```{code-cell} ipython3
# Synthetic "over-smoothed": target + a Gaussian bump that leaks onto the
# boundary. NOT a real FNO output — this is a constructed pathology that
# mirrors the over-smoothing behaviour physics-lint sees on real FNO
# predictions in Week 2½'s dogfood (FNO bc_err = 0.2088 vs DDPM 0.0014).
N = 64
x = np.linspace(0.0, 1.0, N)
X, Y = np.meshgrid(x, x, indexing="ij")
target = np.sin(np.pi * X) * np.sin(np.pi * Y)

center = (0.5, 0.5)
width = 0.3
bump = 0.02 * np.exp(-((X - center[0]) ** 2 + (Y - center[1]) ** 2) / (2 * width**2))
pred = target + bump

mse = np.mean((pred - target) ** 2)
print(f"MSE: {mse:.3e}")

spec = DomainSpec.model_validate(
    {
        "pde": "laplace",
        "grid_shape": [N, N],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
    }
)
field = GridField(pred, h=(1 / (N - 1), 1 / (N - 1)), periodic=False)
boundary_target = np.zeros_like(field.values_on_boundary())

result = ph_bc_001.check(field, spec, boundary_target=boundary_target)
print(f"PH-BC-001 status: {result.status}")
print(f"  mode: {result.mode}")
print(f"  raw_value: {result.raw_value:.3e}")
```

**Interpretation.** MSE on the order of $10^{-5}$ would rank this prediction
near the top of any MSE-based leaderboard. `PH-BC-001` surfaces the specific
failure: the prediction does not vanish on the boundary.

## Case 2: under-trained-style prediction with small MSE but positivity violation

A prediction that is broadly correct but has a small region of negative
values. MSE is small because the negative region is localized; `PH-POS-001`
catches any violation of the physically-mandated $u \geq 0$ constraint.

```{code-cell} ipython3
# Synthetic "under-trained": target with a 5x5 patch near the bottom-left
# corner pushed negative. The patch is placed where the target's amplitude
# is small (sin(pi*x) is small near x=0), so a small subtraction is enough
# to drive values below zero while leaving MSE small. NOT a real U-Net
# output — a constructed pathology that stands in for localized negative-
# values behaviour real models sometimes exhibit during early training.
pred2 = target.copy()
pred2[3:8, 3:8] -= 0.05

mse2 = np.mean((pred2 - target) ** 2)
print(f"MSE: {mse2:.3e}")

spec2 = DomainSpec.model_validate(
    {
        "pde": "poisson",
        "grid_shape": [N, N],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
    }
)
field2 = GridField(pred2, h=(1 / (N - 1), 1 / (N - 1)), periodic=False)
pos_result = ph_pos_001.check(field2, spec2, floor=0.0)
print(f"PH-POS-001 status: {pos_result.status}")
print(f"  min_value (raw): {pos_result.raw_value:.3e}")
if pos_result.violation_ratio is not None:
    print(f"  violation_ratio: {pos_result.violation_ratio:.3f}")
```

**Interpretation.** MSE ~$10^{-4}$ again looks benign on the leaderboard.
`PH-POS-001` detects the negative region and returns FAIL with the minimum
value and a violation ratio that pinpoints how far the prediction crosses
the floor.

## Case 3: non-equivariant CNN passing MSE but failing C4 symmetry

Unlike Cases 1-2, this is a **real trained model**: two 2-layer CNNs trained
from scratch on a rotational Poisson MMS problem. The `baseline` takes the
RHS alone; the `broken` model takes the RHS concatenated with absolute
positional embeddings. Both hit similar MSE during training, because the
loss doesn't know about rotations. The broken model's architecture
*cannot* express a C4-equivariant map, and `PH-SYM-001` picks that up.

```{code-cell} ipython3
from physics_lint.validation.broken_cnn import run_criterion_4_validation

outcome = run_criterion_4_validation(n_training_steps=200, seed=42)
print(f"baseline C4 error: {outcome['baseline_c4_error']:.3e}")
print(f"broken   C4 error: {outcome['broken_c4_error']:.3e}")
print(f"ratio (broken / baseline): {outcome['ratio']:.2f}x")
```

**Interpretation.** Both models achieve comparable loss on the training
objective. The broken model's C4-symmetry error is strictly larger than the
baseline's by a factor well above physics-lint's 2× release-criterion
threshold. MSE ranking would not distinguish the two; `PH-SYM-001` does.

## Takeaway

physics-lint does not replace MSE — it **augments** it. Each case above is a
model (synthetic or real) that a pure-MSE check would green-light but that
has a concrete, specifically-diagnosable physics violation.

Cases 1 and 2 are constructed pathologies for gallery self-containedness.
Case 3 is a real trained model. The v1.1 backlog (marketing scatter plot +
6-surrogate restoration entries in `docs/backlog/v1.1.md`) ships the same
pattern against the full 6-surrogate laplace-uq-bench set and a
distribution-shift OOD test.
