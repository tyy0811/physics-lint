# physics-lint

**A CI linter for trained neural PDE surrogates.** Catches residual, conservation, boundary-condition, positivity, and symmetry violations that MSE misses. Stable rule IDs, SARIF output, GitHub code scanning integration. Think `ruff`, for physics.

by **Jane Yeung** · [github.com/tyy0811/physics-lint](https://github.com/tyy0811/physics-lint) · [physics-lint.readthedocs.io](https://physics-lint.readthedocs.io)

> **Status: v1.0 in active development.** Target release: end of Week 4 (late May 2026). The v1.0 design is committed at [`docs/design/2026-04-14-physics-lint-v1.md`](docs/design/2026-04-14-physics-lint-v1.md). Feature set described below reflects the v1.0 target, not the current state of `main`. Follow along or watch the repo for the v1.0 tag.

---

## Why physics-lint

A neural PDE surrogate can pass every MSE benchmark and still violate the physics it was trained on. MSE averages spatial error; it says nothing about whether mass is conserved, whether the solution respects the boundary condition, whether a positive initial condition stays positive, or whether a rotationally symmetric problem produces a rotationally symmetric solution. These are the failure modes that matter in production: a climate surrogate that mildly violates energy conservation compounds errors over long rollouts; a medical imaging surrogate that produces negative densities fails downstream pipelines; a structural simulator that breaks reflection symmetry misleads optimization.

physics-lint mechanically checks these properties against calibrated analytical floors, produces actionable warnings with stable rule IDs, and emits machine-readable output that your CI can act on. You add it to your GitHub Actions workflow, it runs on every model PR, and the Security tab shows you exactly which rules fired, which model artifact failed, and a doc link explaining each rule with its mathematical justification and citation.

## Hero example

```yaml
# .github/workflows/physics-lint.yml
name: physics-lint
on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: '0 6 * * 1'

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
      - name: Run physics-lint
        run: |
          physics-lint check ${{ matrix.model.path }} \
              --config pyproject.toml \
              --category physics-lint-${{ matrix.model.name }} \
              --format sarif \
              --output physics-lint-${{ matrix.model.name }}.sarif
      - name: Upload SARIF
        if: always()
        uses: github/codeql-action/upload-sarif@v4
        with:
          sarif_file: physics-lint-${{ matrix.model.name }}.sarif
          category: physics-lint-${{ matrix.model.name }}
```

Every model PR now populates the GitHub Security tab with rule violations, complete with documentation links and persistent state. Configure `[tool.physics-lint.sarif]` to surface violations in PR checks too.

## Installation

```bash
pip install physics-lint
```

Python 3.10 or later. Optional mesh support via `pip install physics-lint[mesh]` (conditional on the Week-1 scikit-fem spike passing).

## Quick start

> **Week 1 status (current `main`):** `physics-lint` is importable as a Python library with the Field, DomainSpec, loader, and Laplace/Poisson rules in place. There is **no `physics-lint` CLI yet** — the typer-backed CLI shown in the Hero example above lands in Week 4. For today, drive the linter from Python or a dump loader.

**1. Create an adapter** at `physics_lint_adapter.py` in your repo root:

```python
# physics_lint_adapter.py
from physics_lint import DomainSpec, BCSpec, GridDomain, FieldSourceSpec
import torch

def load_model() -> torch.nn.Module:
    model = MyFNOLaplace()
    model.load_state_dict(torch.load("checkpoints/fno_laplace.pt"))
    model.eval()
    return model

def domain_spec() -> DomainSpec:
    # Week 1 ships Laplace/Poisson on 2D/3D spatial grids. Heat/wave land
    # in Week 2 with the Bochner L²(H⁻¹) norm.
    return DomainSpec(
        pde="laplace",
        grid_shape=(64, 64),
        domain=GridDomain(x=(0.0, 1.0), y=(0.0, 1.0)),
        periodic=False,
        boundary_condition=BCSpec(kind="dirichlet_homogeneous"),
        field=FieldSourceSpec(type="callable", backend="auto"),
    )
```

**2. Add `[tool.physics-lint]` to your `pyproject.toml`** (optional — the adapter can specify everything):

```toml
[tool.physics-lint]
adapter = "./physics_lint_adapter.py"

[tool.physics-lint.rules]
"PH-RES-001" = { tol_pass = 10.0, tol_fail = 100.0 }
```

**3. Run physics-lint from Python (Week 1 programmatic API):**

```python
from physics_lint.loader import load_target
from physics_lint.rules import _registry

loaded = load_target("physics_lint_adapter.py", cli_overrides={}, toml_path=None)
for entry in _registry.list_rules():
    check = _registry.load_check(entry)
    try:
        result = check(loaded.field, loaded.spec)
    except TypeError:
        continue  # skip rules that require extra kwargs (boundary_target, …)
    print(f"{result.rule_id:12s} {result.status:7s} raw={result.raw_value}")
```

Each `result` is a `RuleResult` dataclass with `status` (`PASS`/`WARN`/`FAIL`/`SKIPPED`), `raw_value`, `violation_ratio`, `reason`, and provenance fields; see `src/physics_lint/report.py`. Status assembly into a full `PhysicsLintReport` with text/JSON/SARIF output is the Week-4 deliverable.

The `physics-lint check <target>` CLI shown in the Hero example and the rich report format above are the **v1.0 Week-4 target**, not what `main` ships today. Track progress in [`docs/plans/`](docs/plans/).

## Supported PDEs and models

**v1.0 PDE coverage:**

| PDE | Residual | Norm |
|-----|----------|------|
| Laplace | $R = -\Delta u$ | $H^{-1}$ |
| Poisson | $R = -\Delta u - f$ | $H^{-1}$ |
| Heat | $R = u_t - \kappa\Delta u$ | Bochner $L^2(0,T; H^{-1})$ |
| Wave | $R = u_{tt} - c^2\Delta u$ | Bochner $L^2(0,T; H^{-1})$ (conjectural; see PH-VAR-002) |

Domains: 2D and 3D structured Cartesian grids. Optional unstructured meshes via scikit-fem (Week-1 spike-gated).

**v1.0 model coverage:** any PyTorch model loadable via a small adapter file (`torch.nn.Module` or any `Callable[[Tensor], Tensor]`). Iterative samplers and non-PyTorch frameworks use the secondary *dump mode*: save the model's prediction as `pred.npz` with metadata, and physics-lint runs against the tensor directly. JAX, TensorFlow, and NumPy users are supported this way.

**Explicitly out of scope for v1.0:** Navier-Stokes, MHD, compressible flow, AMR, GPU kernels, JAX backend, symbolic PDE definitions, auto-fix. See [§19 of the design doc](docs/design/2026-04-14-physics-lint-v1.md#19-non-goals-for-v1) for the full non-goals list.

## Rule catalog (v1.0, 21 rules)

Each rule has a stable ID (`PH-<CATEGORY>-<NNN>`), a default severity, a documented input-mode compatibility (adapter-only vs adapter+dump), and a doc page with math justification and citation. Rule severities are overridable per-project via config.

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
| `PH-VAR-001` | L² residual on second-order strong-form formulation | info | adapter + dump |
| `PH-VAR-002` | Hyperbolic norm-equivalence conjectural | info | adapter + dump |
| `PH-NUM-001` | Quadrature convergence warning (mesh) | warning | adapter + dump |
| `PH-NUM-002` | Refinement convergence rate below expected | warning | adapter + dump |
| `PH-NUM-003` | Non-$C^2$ activation scan | warning | adapter only |
| `PH-NUM-004` | Configured BC inconsistent with model training BC | warning | adapter + dump |

`physics-lint rules list` shows this table (returning in <50 ms via lazy registry). `physics-lint rules show PH-RES-001` shows the full per-rule docs including derivation and citation.

## How it works

### Three design invariants

**1. Norm-equivalence to error, scoped to the chosen residual formulation.** Every residual rule satisfies a two-sided bound

$$c_B \|r_B(u^\delta)\|_{Y'} \leq \|u - u^\delta\|_W \leq C_B \|r_B(u^\delta)\|_{Y'}$$

(Bachmayr et al. 2024 Eq. 2.13; Ernst et al. 2025 Eq. 3.2–3.3). The constants and the test-space norm $Y'$ depend on the formulation, not the PDE class alone. physics-lint implements the standard second-order residual and warns via `PH-VAR-001` when `L²` would be misleading (suitable first-order reformulations exist for some PDE classes where `L²` is correct). For hyperbolic problems, `PH-VAR-002` notes that norm-equivalence is weaker and conjectural.

**2. Self-calibration against numerical floor.** Every rule reports

$$\text{violation\_ratio} = \frac{\text{raw\_violation}}{\text{analytical\_floor}}$$

where the analytical floor is measured by running the same rule on a known analytical solution at the same resolution. Default thresholds: ratio < 10 → PASS; [10, 100] → WARN; > 100 → FAIL. Per-rule overridable via config. Floors live in `physics_lint/data/floors.toml` with per-floor multiplicative tolerance (pocketfft vs MKL vs Accelerate drift is absorbed here).

**3. Reproduce known empirical results.** The test suite demonstrates physics-lint detects:
- deliberately non-equivariant CNN with positional embeddings violates $C_4$ symmetry by $>2\times$ baseline;
- model trained to violate mass conservation on heat is flagged following Jekel et al. (2022);
- laplace-uq-bench surrogate ranking under physics-lint agrees with the published $H^1$ ranking in top-2 and bottom-2 positions.

### Field abstraction

physics-lint represents a trained model's output as a `Field` — one of three concrete types:

- **`GridField`** — regular Cartesian grid, 4th-order Fornberg FD or Fourier spectral differentiation (auto-selected from the `periodic` flag).
- **`CallableField`** — wraps a `Callable[[Tensor], Tensor]`, derivatives via `torch.autograd.functional.jacobian` batched with `torch.vmap`.
- **`MeshField`** — scikit-fem-backed for unstructured meshes (conditional on the Week-1 Day-2 spike).

All rules operate against the `Field` abstraction and a validated `DomainSpec` (pydantic v2), so rules never touch raw config or backend-specific tensor shapes.

### Hybrid loader: adapter + dump

physics-lint supports two model-loading paths, dispatched by the positional argument's file extension:

| Extension | Mode | What you write |
|-----------|------|----------------|
| `.py` | Adapter (primary) | Two functions: `load_model()` and `domain_spec()` |
| `.npz` / `.npy` | Dump (secondary) | Pre-generated prediction with metadata dict |
| `.pt` / `.pth` | Error | Use an adapter or convert to `.npz`; see docs |

**Adapter mode** runs the full rule suite including autograd-based rules (`PH-RES-002` FD-vs-AD cross-check, `PH-SYM-003` SO(2) LEE). **Dump mode** is for iterative samplers (DDPM, DPS), JAX/TensorFlow models, or any case where running the model is expensive or nondeterministic. Rules that require callables skip gracefully in dump mode with an explicit reason:

```
  ⊘ PH-SYM-003  SKIP  SO(2) LEE  requires callable; dump mode
```

Skipped rules appear in the text report, in the JSON report, and in SARIF `run.invocations[].toolExecutionNotifications` — never silent omission.

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

[tool.physics-lint.rules]
"PH-RES-001" = { tol_pass = 10.0, tol_fail = 100.0 }
"PH-BC-001"  = { abs_threshold = 1e-10, abs_tol_fail = 1e-3 }
"PH-SYM-003" = { enabled = false }
"PH-CON-003" = { severity = "error" }

[tool.physics-lint.sarif]
source_file = "train_heat_fno.py"
pde_line = 42
bc_line = 58
```

`physics-lint config init --pde heat` emits a heat-specific commented template. `physics-lint config show` validates your config and pretty-prints the resolved spec.

## CLI reference

```bash
physics-lint check <target> [--config PATH] [--format {text,json,sarif}] [--category NAME] [--output PATH]
                             [--disable RULE_ID] [--enable-only RULE_ID,...] [--severity RULE_ID=LEVEL]
                             [--pde NAME] [--grid N,N,N] [--periodic] [--bc NAME] [--verbose]

physics-lint self-test [--verbose] [--rule RULE_ID] [--write-report PATH]

physics-lint rules (list | show RULE_ID)

physics-lint config (init [--pde NAME] | show [--config PATH])
```

Exit codes: `0` = all error-severity rules pass; `1` = at least one error-severity rule failed; `2` = invalid config or CLI usage; `3` = model load failed.

## Security

physics-lint `exec`s adapter code — the same trust model as pytest loading `conftest.py`. For local use, fine. In CI contexts, physics-lint runs arbitrary Python with the same token permissions as the job itself. The canonical workflow above sets minimum permissions:

```yaml
permissions:
  contents: read
  security-events: write
```

**Do not grant `contents: write` or `pull-requests: write` unless you need them.** For public-contribution workflows where PR authors and repo owners differ (e.g., model zoos accepting contributions), use `pull_request_target` with branch restrictions per [GitHub's documented guidance](https://docs.github.com/en/actions/security-guides/automatic-token-authentication) on that trigger.

## Development

The v1.0 design is fully documented in [`docs/design/2026-04-14-physics-lint-v1.md`](docs/design/2026-04-14-physics-lint-v1.md). Implementation is tracked in [`docs/plans/`](docs/plans/) once step-by-step plans are generated.

**Stack:** Python 3.10+, hatchling, pydantic 2.0+, typer, ruff, pytest + hypothesis, Sphinx + MyST + furo. Apache-2.0 license. Six-job CI matrix (Linux × Python 3.10/3.11/3.12 × NumPy 1.26/2.0 × PyTorch 2.0/2.2/2.5 + macOS arm64). 85% coverage gate.

Contributions welcome once v1.0 ships. Until then, please file issues for design questions or rule suggestions.

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

---

*physics-lint is being developed by [Jane Yeung](https://github.com/tyy0811) as a 4-week focused project, April–May 2026. The v1.0 design has gone through 8 rounds of architectural review and is documented in full at [`docs/design/`](docs/design/).*
