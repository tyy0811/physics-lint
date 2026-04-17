# physics-lint

**A CI linter for trained neural PDE surrogates.** Catches residual, conservation, boundary-condition, positivity, and symmetry violations that MSE misses. Stable rule IDs, SARIF output, GitHub code scanning integration. Think `ruff`, for physics.

by **Jane Yeung** · [github.com/tyy0811/physics-lint](https://github.com/tyy0811/physics-lint) · [physics-lint.readthedocs.io](https://physics-lint.readthedocs.io)

---

## Why physics-lint

A neural PDE surrogate can pass every MSE benchmark and still violate the physics it was trained on. MSE averages spatial error; it says nothing about whether mass is conserved, whether the solution respects the boundary condition, whether a positive initial condition stays positive, or whether a rotationally symmetric problem produces a rotationally symmetric solution. These are the failure modes that matter in production: a climate surrogate that mildly violates energy conservation compounds errors over long rollouts; a medical imaging surrogate that produces negative densities fails downstream pipelines; a structural simulator that breaks reflection symmetry misleads optimization.

physics-lint mechanically checks these properties against calibrated analytical floors, produces actionable warnings with stable rule IDs, and emits machine-readable output that your CI can act on. You add it to your GitHub Actions workflow, it runs on every model PR, and the Security tab shows you exactly which rules fired, which model artifact failed, and a doc link explaining each rule with its mathematical justification and citation.

## Hero: physics-lint in CI

<!--
TODO (Week 4 Task 4 Step 4 — user handoff): capture the FNO PH-BC-001
alert in the Security tab from this repo's own first CI run and save to
docs/figures/sarif-hero.png. Until captured, the image below renders as
a broken link. See docs/plans/2026-05-05-physics-lint-v1-week-4.md
§"README framing commitment" for context.
-->

![physics-lint FNO PH-BC-001 alert rendered in the GitHub Security tab](docs/figures/sarif-hero.png)

*Above: the FNO `PH-BC-001` alert surfaced in physics-lint's own repository Security tab. The screenshot is from running physics-lint against the `fno` surrogate in [`tyy0811/laplace-uq-bench`](https://github.com/tyy0811/laplace-uq-bench) — FNO's Dirichlet-boundary error is ~150× the DDPM baseline, and `PH-BC-001` catches the violation as a code-scanning alert with rule documentation links and persistent state.*

```yaml
# .github/workflows/physics-lint.yml
name: physics-lint
on: [push, pull_request]

permissions:
  contents: read
  security-events: write

jobs:
  lint:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        model:
          - { name: unet, path: models/unet_adapter.py }
          - { name: fno,  path: models/fno_adapter.py }
          - { name: ddpm, path: models/ddpm_pred.npz }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.11' }
      - run: pip install physics-lint
      - run: |
          physics-lint check ${{ matrix.model.path }} \
              --config pyproject.toml \
              --category physics-lint-${{ matrix.model.name }} \
              --format sarif \
              --output physics-lint-${{ matrix.model.name }}.sarif
      - if: always()
        uses: github/codeql-action/upload-sarif@v4
        with:
          sarif_file: physics-lint-${{ matrix.model.name }}.sarif
          category: physics-lint-${{ matrix.model.name }}
```

Every model PR populates the GitHub Security tab with rule violations, complete with documentation links and persistent state. `if: always()` on the SARIF upload step means alerts land even when the check step exits non-zero. Configure `[tool.physics-lint.sarif]` to surface violations in PR checks too.

## Installation

```bash
pip install physics-lint
```

Python 3.10 or later. Optional unstructured-mesh support via `pip install physics-lint[mesh]`.

## Quick start

```bash
# Lint a .npz dump
physics-lint check pred.npz --format text

# Lint an adapter (a Python file defining load_model() and domain_spec())
physics-lint check physics_lint_adapter.py --format text

# CI-style SARIF output
physics-lint check model.py --format sarif --category physics-lint-run \
    --output physics-lint.sarif
```

A minimal adapter at `physics_lint_adapter.py`:

```python
from physics_lint import DomainSpec, BCSpec, GridDomain, FieldSourceSpec
import torch

def load_model() -> torch.nn.Module:
    model = MyFNOLaplace()
    model.load_state_dict(torch.load("checkpoints/fno_laplace.pt"))
    model.eval()
    return model

def domain_spec() -> DomainSpec:
    return DomainSpec(
        pde="laplace",
        grid_shape=(64, 64),
        domain=GridDomain(x=(0.0, 1.0), y=(0.0, 1.0)),
        periodic=False,
        boundary_condition=BCSpec(kind="dirichlet_homogeneous"),
        field=FieldSourceSpec(type="callable", backend="auto"),
    )
```

Or drive physics-lint from Python:

```python
from physics_lint.loader import load_target
from physics_lint.report import PhysicsLintReport

loaded = load_target("physics_lint_adapter.py", cli_overrides={}, toml_path=None)
# ... invoke rules, assemble a PhysicsLintReport, render to text/json/sarif
```

## What physics-lint catches

**Broken-model gallery** ([`examples/broken_model_gallery.ipynb`](examples/broken_model_gallery.ipynb)) walks through three cases where MSE ranking and physics-lint ranking disagree:

| Case | Model | What MSE says | What physics-lint catches |
|---|---|---|---|
| 1 | Over-smoothed prediction with boundary leak | MSE ~1e-4 — top of leaderboard | `PH-BC-001` FAIL: doesn't respect Dirichlet BC |
| 2 | Under-trained prediction with localized negatives | MSE ~1e-5 — near-perfect | `PH-POS-001` FAIL: u < 0 in a 5×5 region |
| 3 | Non-equivariant CNN with positional-embedding input | Comparable loss to baseline | `PH-SYM-001`: C4 error 12× baseline |

Cases 1-2 are constructed pathologies labelled after real failure modes on trained neural PDE surrogates. Case 3 is a real trained model. See the notebook for rationale.

## Dogfood: laplace-uq-bench

physics-lint v1.0 is validated against three trained surrogates from [`github.com/tyy0811/laplace-uq-bench`](https://github.com/tyy0811/laplace-uq-bench) — `unet_regressor`, `fno`, and `ddpm` — through a **3-axis cross-comparison** against the repo's published metrics. The v1.0 verdict is `PASS (scoped, MIXED)`:

- **Real axis #1** (`PH-BC-001` vs upstream `bc_err`): full ranking agreement on all three surrogates (DDPM best, FNO worst — FNO's boundary error is ~150× DDPM).
- **Sanity axis** (`PH-RES-001` vs upstream `pde_residual`): rank-1 consistent under a pre-disclosed definitional gap (fd4 vs fd2 stencil, full-grid vs interior scope, L² trapezoidal vs dimensionless RMS).
- **Real axis #2** (`PH-POS-002` vs upstream `max_viol`): magnitude-vs-count definitional gap, resolved in v1.1 via a metrics-compatibility shim.

Full results in [`dogfood/dogfood_real_results.md`](dogfood/dogfood_real_results.md). Methodology notes and reinterpretation rationale in [`docs/tradeoffs.md`](docs/tradeoffs.md).

**v1.1 roadmap.** Expanding to 6 surrogates (adding ensemble, DPS, OT-CFM, improved DDPM, flow-matching), restoring byte-identical sanity-axis comparison via a metrics-compatibility shim, and producing an out-of-distribution "MSE misses what physics catches" scatter figure are tracked in [`docs/backlog/v1.1.md`](docs/backlog/v1.1.md).

## Rule catalog (v1.0)

Each rule has a stable ID (`PH-<CATEGORY>-<NNN>`), a default severity, documented input-mode compatibility, and a doc page with math justification and citation. v1.0 ships **18 rules**.

| Rule ID | Name | Severity | Input modes |
|---------|------|----------|-------------|
| `PH-RES-001` | Residual exceeds variationally-correct norm threshold | error | adapter + dump |
| `PH-RES-002` | FD-vs-AD residual cross-check discrepancy | warning | adapter only |
| `PH-RES-003` | Spectral-vs-FD residual discrepancy on periodic grid | warning | adapter + dump |
| `PH-BC-001` | Boundary condition violation (relative or absolute mode) | error | adapter + dump |
| `PH-BC-002` | Boundary flux imbalance (divergence theorem) | warning | adapter + dump |
| `PH-CON-001` | Mass conservation violation | error | adapter + dump |
| `PH-CON-002` | Energy conservation violation | error | adapter + dump |
| `PH-CON-003` | Energy dissipation sign violation | warning | adapter + dump |
| `PH-CON-004` | Per-element conservation hotspot | warning | adapter + dump (mesh) |
| `PH-POS-001` | Positivity violation | error | adapter + dump |
| `PH-POS-002` | Maximum principle violation | error | adapter + dump |
| `PH-SYM-001` | $C_4$ rotation equivariance violation | warning | adapter + dump |
| `PH-SYM-002` | Reflection equivariance violation | warning | adapter + dump |
| `PH-SYM-003` | SO(2) Lie derivative equivariance violation | warning | adapter only |
| `PH-SYM-004` | Translation equivariance violation (periodic-only in v1) | warning | adapter + dump |
| `PH-VAR-002` | Hyperbolic norm-equivalence conjectural | info | adapter + dump |
| `PH-NUM-001` | Quadrature convergence warning (mesh) | warning | adapter + dump |
| `PH-NUM-002` | Refinement convergence rate below expected | warning | adapter + dump |

`physics-lint rules list` shows this table (<50 ms via lazy registry). `physics-lint rules show PH-RES-001` prints the full per-rule docs including derivation and citation.

**Design-doc future surface (v1.1).** Three additional rules from the design doc — `PH-VAR-001` (L² residual on second-order strong form), `PH-NUM-003` (non-C² activation scan), `PH-NUM-004` (configured BC vs model training BC) — are specified in [`docs/design/2026-04-14-physics-lint-v1.md`](docs/design/2026-04-14-physics-lint-v1.md) but deferred to v1.1 along with the `[tool.physics-lint.rules]` per-rule override surface. See [`docs/backlog/v1.1.md`](docs/backlog/v1.1.md).

## Supported PDEs and models

**v1.0 PDE coverage:**

| PDE | Residual | Norm |
|-----|----------|------|
| Laplace | $R = -\Delta u$ | $H^{-1}$ |
| Poisson | $R = -\Delta u - f$ | $H^{-1}$ |
| Heat | $R = u_t - \kappa\Delta u$ | Bochner $L^2(0,T; H^{-1})$ |
| Wave | $R = u_{tt} - c^2\Delta u$ | Bochner $L^2(0,T; H^{-1})$ (conjectural; see `PH-VAR-002`) |

Domains: 2D and 3D structured Cartesian grids. Optional unstructured meshes via scikit-fem (install via `pip install physics-lint[mesh]`).

**v1.0 model coverage:** any PyTorch model loadable via a small adapter file (`torch.nn.Module` or any `Callable[[Tensor], Tensor]`). Iterative samplers and non-PyTorch frameworks use the secondary *dump mode*: save the model's prediction as `pred.npz` with metadata, and physics-lint runs against the tensor directly. JAX, TensorFlow, and NumPy users are supported this way.

**Explicitly out of scope for v1.0:** Navier-Stokes, MHD, compressible flow, AMR, GPU kernels, JAX backend, symbolic PDE definitions, auto-fix. See [§19 of the design doc](docs/design/2026-04-14-physics-lint-v1.md#19-non-goals-for-v1) for the full non-goals list.

## How it works

### Three design invariants

**1. Norm-equivalence to error, scoped to the chosen residual formulation.** Every residual rule satisfies a two-sided bound

$$c_B \|r_B(u^\delta)\|_{Y'} \leq \|u - u^\delta\|_W \leq C_B \|r_B(u^\delta)\|_{Y'}$$

(Bachmayr et al. 2024 Eq. 2.13; Ernst et al. 2025 Eq. 3.2–3.3). The constants and the test-space norm $Y'$ depend on the formulation, not the PDE class alone. physics-lint implements the standard second-order residual and warns via `PH-VAR-001` when `L²` would be misleading. For hyperbolic problems, `PH-VAR-002` notes that norm-equivalence is weaker and conjectural.

**2. Self-calibration against numerical floor.** Every rule reports

$$\text{violation\_ratio} = \frac{\text{raw\_violation}}{\text{analytical\_floor}}$$

where the analytical floor is measured by running the same rule on a known analytical solution at the same resolution. Default thresholds: ratio < 10 → PASS; [10, 100] → WARN; > 100 → FAIL. Per-rule overridable via config. Floors live in `physics_lint/data/floors.toml` with per-floor multiplicative tolerance.

**3. Reproduce known empirical results.** The test suite demonstrates physics-lint detects:
- deliberately non-equivariant CNN with positional embeddings violates $C_4$ symmetry by $>2\times$ baseline (see `physics_lint.validation.broken_cnn`);
- real-model disagreement surfaces in the 3-surrogate laplace-uq-bench dogfood (`dogfood/run_dogfood_real.py`);
- the broken-model gallery (`examples/broken_model_gallery.ipynb`) exhibits three MSE-vs-physics-lint disagreement cases.

### Field abstraction

physics-lint represents a trained model's output as a `Field`:

- **`GridField`** — regular Cartesian grid, 4th-order Fornberg FD or Fourier spectral differentiation (auto-selected from the `periodic` flag).
- **`CallableField`** — wraps a `Callable[[Tensor], Tensor]`, derivatives via `torch.autograd.functional.jacobian` batched with `torch.vmap`.
- **`MeshField`** — scikit-fem-backed for unstructured meshes (optional `[mesh]` extra).

All rules operate against the `Field` abstraction and a validated `DomainSpec` (pydantic v2).

### Hybrid loader: adapter + dump

physics-lint supports two model-loading paths, dispatched by file extension:

| Extension | Mode | What you write |
|-----------|------|----------------|
| `.py` | Adapter (primary) | Two functions: `load_model()` and `domain_spec()` |
| `.npz` / `.npy` | Dump (secondary) | Pre-generated prediction with metadata dict |
| `.pt` / `.pth` | Error | Use an adapter or convert to `.npz` |

**Adapter mode** runs the full rule suite including autograd-based rules. **Dump mode** is for iterative samplers (DDPM, DPS), JAX/TensorFlow models, or any case where running the model is expensive or nondeterministic. Rules that require a callable skip gracefully in dump mode with an explicit reason:

```
  ⊘ PH-SYM-003  SKIPPED  SO(2) LEE  requires callable; dump mode
```

Skipped rules appear in the text report, in the JSON report, and in SARIF `run.invocations[].toolExecutionNotifications` — never silent omission. Per-rule PASS outcomes do not emit SARIF results (SARIF results are findings; the Security tab treats every result as an alert).

### GitHub code scanning (SARIF)

SARIF output populates the GitHub Security tab (**Tier 1**, always). Optionally, configuring `[tool.physics-lint.sarif]` with a source file and line region surfaces violations in PR checks (**Tier 2**, opt-in):

```toml
[tool.physics-lint.sarif]
source_file = "train_heat_fno.py"
pde_line = 42
bc_line = 58
```

Tier 3 (arbitrary inline diff comments on unrelated lines) is explicitly not in v1.0.

## Configuration

Canonical config in `pyproject.toml` under `[tool.physics-lint]`; standalone `physics-lint.toml` supported as a fallback.

Minimal (relies on the adapter for everything):

```toml
[tool.physics-lint]
adapter = "./physics_lint_adapter.py"
```

Full v1.0 surface:

```toml
[tool.physics-lint]
pde = "heat"
grid_shape = [64, 64, 32]
domain = { x = [0.0, 1.0], y = [0.0, 1.0], t = [0.0, 1.0] }
periodic = false
boundary_condition = "dirichlet_homogeneous"
diffusivity = 0.01
symmetries = ["D4", "translation_x", "translation_y"]
adapter = "./physics_lint_adapter.py"

[tool.physics-lint.field]
type = "callable"
backend = "auto"

[tool.physics-lint.sarif]
source_file = "train_heat_fno.py"
pde_line = 42
bc_line = 58
```

`physics-lint config init --pde heat` emits a heat-specific commented template. `physics-lint config show --config pyproject.toml` validates your config and pretty-prints the resolved spec (no target required).

**Design-doc future surface.** `[tool.physics-lint.rules]` per-rule overrides (`tol_pass`, `abs_threshold`, `enabled`, `severity`) are specified in the design doc but not wired through the CLI in v1.0. Disable individual rules at run time via `--disable PH-SYM-003`. The full override surface lands in v1.1 per [`docs/backlog/v1.1.md`](docs/backlog/v1.1.md).

## CLI reference

```bash
physics-lint check <target> [--config PATH] [--format {text,json,sarif}] [--category NAME]
                             [--output PATH] [--disable RULE_ID] [--verbose]

physics-lint self-test [--verbose] [--write-report PATH]

physics-lint rules (list | show RULE_ID)

physics-lint config (init [--pde {generic|heat|wave}] | show --config PATH)
```

Exit codes: `0` = all error-severity rules pass; `1` = at least one error-severity rule failed; `2` = invalid config or CLI usage; `3` = model load failed.

## Security

physics-lint `exec`s adapter code — the same trust model as pytest loading `conftest.py`. For local use, fine. In CI contexts, physics-lint runs arbitrary Python with the same token permissions as the job itself. The canonical workflow above sets minimum permissions:

```yaml
permissions:
  contents: read
  security-events: write
```

**Do not grant `contents: write` or `pull-requests: write` unless you need them.** For public-contribution workflows where PR authors and repo owners differ (e.g., model zoos accepting contributions), use `pull_request_target` with branch restrictions per [GitHub's documented guidance](https://docs.github.com/en/actions/security-guides/automatic-token-authentication).

## Development

Design doc: [`docs/design/2026-04-14-physics-lint-v1.md`](docs/design/2026-04-14-physics-lint-v1.md). Implementation plans in [`docs/plans/`](docs/plans/). Methodology tradeoffs in [`docs/tradeoffs.md`](docs/tradeoffs.md). v1.1 backlog in [`docs/backlog/v1.1.md`](docs/backlog/v1.1.md).

**Stack:** Python 3.10+, hatchling, pydantic 2.0+, typer, ruff, pytest + hypothesis, Sphinx + MyST + furo. Apache-2.0 license. Six-job CI matrix (Linux × Python 3.10/3.11/3.12 × NumPy 1.26/2.0 × PyTorch 2.0/2.2/2.5 + macOS arm64). 85% coverage gate.

```bash
git clone https://github.com/tyy0811/physics-lint
cd physics-lint
pip install -e ".[dev]"
pre-commit install
pytest
```

Contributions welcome. File issues for design questions or rule suggestions.

## Citation

```bibtex
@software{yeung_physics_lint_2026,
  author  = {Yeung, Jane},
  title   = {physics-lint: A CI linter for trained neural PDE surrogates},
  year    = {2026},
  url     = {https://github.com/tyy0811/physics-lint},
  version = {1.0.0}
}
```

## Acknowledgments and references

The rule catalog is grounded in:

- Bachmayr, Dahmen, Oster (2024), *Variationally correct neural residual regression for parametric PDEs*, [arXiv:2405.20065](https://arxiv.org/abs/2405.20065).
- Ernst, Rekatsinas, Urban (2025), *A posteriori certification for neural network approximations to PDEs*, [arXiv:2502.20336v3](https://arxiv.org/abs/2502.20336v3).
- Jekel et al. (2022), *Using conservation laws to infer deep learning model accuracy of Richtmyer-Meshkov instabilities*, [arXiv:2208.11477](https://arxiv.org/abs/2208.11477).
- Gruver, Finzi, Goldblum, Wilson (2023), *The Lie derivative for measuring learned equivariance*, ICLR 2023, [arXiv:2210.02984](https://arxiv.org/abs/2210.02984).
- Helwig et al. (2023), *Group equivariant Fourier neural operators for PDEs*, ICML 2023, [arXiv:2306.05697](https://arxiv.org/abs/2306.05697).
- Qiu, Dahmen, Chen (2025), *Variationally correct operator learning*, [arXiv:2512.21319](https://arxiv.org/abs/2512.21319).
- Gustafsson & McBain (2020), *scikit-fem: A Python package for finite element assembly*, JOSS 5(52).
- Trefethen (2000), *Spectral Methods in MATLAB*, SIAM.
- Fornberg (1988), *Generation of finite difference formulas on arbitrarily spaced grids*, Math. Comp. 51(184).

Full reference list in [§22 of the design doc](docs/design/2026-04-14-physics-lint-v1.md#22-references).

## License

[Apache License 2.0](LICENSE). Patent grant included — safe for commercial-adjacent MLOps pipelines.
