# physics-lint — V1 Design Document

**Date:** 2026-04-14
**Status:** Planning-phase final; supersedes Rev. 4.3.1 implementation plan
**Baseline:** Rev. 4.3.1 of the physics-lint implementation plan, augmented by five brainstorming Q&A rounds and four per-section design reviews
**Next step:** `superpowers:writing-plans` → step-by-step implementation plan

---

## 0. Executive summary

**physics-lint** is a Python linter for trained neural PDE surrogates. It runs in CI, catches physics violations on every model PR, and produces actionable warnings with stable rule IDs, configurable severity, and machine-readable output. Think `ruff` for FNO, U-Net, DeepONet, PINN, and diffusion-model surrogates: a tool that mechanically checks PDE residuals, conservation laws, boundary conditions, positivity constraints, and symmetry properties, then writes a report your build system can act on.

The library is positioned as **MLOps tooling that knows physics**, not a research utility. The README hero example is a GitHub Actions workflow, not a Python API tour. V1 targets Laplace, Poisson, heat, and wave equations on 2D/3D structured grids, with optional unstructured-mesh support via scikit-fem (conditional on a Week-1 Day-2 half-day spike).

**Primary dogfood target.** V1 validation runs physics-lint on every surrogate in `github.com/tyy0811/laplace-uq-bench`: U-Net, FNO, deep ensemble, OT-CFM, improved DDPM, DPS — six trained models on 2D Laplace with published residual rankings and conformal-calibrated uncertainty. Mixed adapter/dump mode rollout depending on sampler determinism. The repo's own GitHub Actions workflow runs physics-lint on a schedule and produces SARIF output that populates the GitHub Security tab and code-scanning alerts. This is the central marketing artifact.

### Package identity

- **PyPI name:** `physics-lint` (verify availability and register `0.0.0.dev0` placeholder on Week-1 Day-1 AM under `tyy0811`; fallbacks `physicslint`, `pde-lint`, `phylint`)
- **Import name:** `physics_lint`
- **CLI entry point:** `physics-lint`
- **Config file:** `[tool.physics-lint]` in `pyproject.toml` (canonical) or standalone `physics-lint.toml`
- **Docs:** `https://physics-lint.readthedocs.io` (ReadTheDocs; Sphinx + MyST + furo theme)
- **License:** Apache-2.0
- **GitHub owner:** `tyy0811` (professional identity surfaced via display name and authorial metadata rather than handle change)

### Three design invariants

**Invariant 1 — Norm-equivalence to error, scoped to the chosen residual formulation.** Every residual rule must satisfy a two-sided bound

$$c_B \|r_B(u^\delta)\|_{Y'} \leq \|u - u^\delta\|_W \leq C_B \|r_B(u^\delta)\|_{Y'}$$

with respect to the formulation it implements (Bachmayr et al. arXiv:2405.20065 Eq. 2.13; Ernst et al. arXiv:2502.20336v3 Eq. 3.2–3.3). The constants $c_B, C_B$ and the appropriate test-space norm $Y'$ depend on the *formulation*, not the PDE class alone. Suitable first-order (FOSLS-type) reformulations of stationary diffusion and linear elasticity restore $L^2$ norm-equivalence — the issue is the second-order strong-form residual, not "elliptic PDEs in general." V1 implements the standard second-order residual and warns when it would give misleading $L^2$ results (`PH-VAR-001`). For wave (hyperbolic), the analogous norm-equivalence is weaker and conjectural; documented as `PH-VAR-002`.

**Invariant 2 — Self-calibration against numerical floor.** Every rule reports `violation_ratio = raw_violation / analytical_floor`, computed by running the same rule on a known analytical solution at the same resolution. Default thresholds: ratio < 10 → PASS; [10, 100] → WARN; > 100 → FAIL. Per-rule overridable via config. Floors live in `physics_lint/data/floors.toml` with per-floor multiplicative tolerance; see §6.

**Invariant 3 — Reproduce known empirical results.** Test suite must demonstrate physics-lint detects: (a) deliberately non-equivariant CNN with positional embeddings shows $C_4$ violation $> 2\times$ baseline; (b) model trained to violate mass conservation on heat is flagged following Jekel et al.; (c) laplace-uq-bench surrogate ranking under physics-lint matches the published $H^1$ ranking in top-2 and bottom-2 positions (reformulated from Rev 4.3.1's Spearman $\rho > 0.8$ criterion, which is too noisy on $n=6$).

---

## 1. Mathematical foundations per PDE

**Notation for BCs.** Conservation/positivity statements explicitly name the BC class under which they hold. Periodic, homogeneous Neumann, and homogeneous Dirichlet are abbreviated PER, hN, hD.

### 1.1 Laplace equation

**Strong form.** $-\Delta u = 0$ in $\Omega \subset \mathbb{R}^d$, $d \in \{2,3\}$, with $u = g$ on $\partial\Omega$.

**Weak form.** Find $u \in H^1(\Omega)$ with $u|_{\partial\Omega} = g$ such that $(\nabla u, \nabla v)_{L^2(\Omega)} = 0$ for all $v \in H^1_0(\Omega)$.

**Analytical solutions.**
1. Harmonic polynomial on $[0,1]^2$: $u(x,y) = x^2 - y^2$ with $g$ being the trace.
2. Eigenfunction-based on $[0,1]^2$ with sinusoidal Dirichlet trace: $u(x,y) = \sin(n\pi x)\sinh(n\pi y)/\sinh(n\pi)$ giving $u(x,1) = \sin(n\pi x)$, $u = 0$ on the other three sides.
3. Harmonic Gaussian-like: $u(x,y) = e^{x}\cos(y)$ on a box.

> **Periodic Laplace.** Liouville's theorem implies that bounded harmonic functions on the flat torus are constant — there is no nontrivial fully-periodic Laplace solution. Periodic test cases are deferred to Poisson.

**Norms.** Spectral on periodic Poisson-style problems: $\|r\|^2_{H^{-1}} = \sum_{k \neq 0} |\hat{r}_k|^2 / |2\pi k/L|^2$. FE general: $r_h^T K^{-1} r_h$ via stiffness matrix $K$.

**Conservation (BC-qualified).** Under any well-posed BC, the flux integral $\int_{\partial\Omega} \nabla u \cdot \mathbf{n}\, dS = 0$ for harmonic $u$.

**Maximum principle** (rule `PH-POS-002`, valid for any well-posed Dirichlet problem): $\min_{\partial\Omega} g \leq u(x) \leq \max_{\partial\Omega} g$.

**Symmetries.** PDE operator $-\Delta$ is invariant under Euclidean $E(d)$, scaling, and conformal transformations in 2D. Problem-instance symmetry depends on the domain $\Omega$ and the boundary data $g$ — see §9.4.

### 1.2 Poisson equation

**Strong form.** $-\Delta u = f$ in $\Omega$, $u = 0$ on $\partial\Omega$.

**Analytical solutions.**
1. MMS on $[0,1]^2$ with hD: $u = \sin(\pi x)\sin(\pi y)$ gives $f = 2\pi^2 u$.
2. Polynomial: $u = x(1-x)y(1-y)$ gives $f = 2[y(1-y)+x(1-x)]$.
3. Periodic on $[0, 2\pi]^2$: $u = \sin(x)\sin(y)$ gives $f = 2\sin(x)\sin(y)$.

**Norms.** $H^{-1}$ as Laplace. FOSLS alternative (Qiu et al. arXiv:2512.21319, **scoped to stationary diffusion and linear elasticity**) introduces $\sigma = \nabla u$ and measures $\|\sigma - \nabla u\|^2_{L^2} + \|\nabla\cdot\sigma - f\|^2_{L^2}$, which is norm-equivalent to the $H^1$ error in that scope.

**Conservation (BC-qualified).** $\int_{\partial\Omega} \nabla u \cdot \mathbf{n}\, dS = -\int_\Omega f\, dx$ under any well-posed BC.

**Sign constraints.** For $f \geq 0$ with hD: $u \geq 0$ (rule `PH-POS-001`).

### 1.3 Heat equation

**Strong form.** $u_t - \kappa \Delta u = 0$ in $\Omega \times (0,T]$, $u(x,0) = u_0(x)$, with BC per problem.

**Analytical solutions.**
1. Eigenfunction decay on $[0,1]^2$ with hD: $u = \sin(\pi x)\sin(\pi y)\exp(-2\kappa\pi^2 t)$.
2. Periodic mode on $[0, 2\pi]^2$ with PER: $u = \cos(x)\cos(y)\exp(-2\kappa t)$.
3. Free-space Gaussian kernel on $\mathbb{R}^d$ — usable only if the bounded computational domain is large enough that boundary effects are below truncation error at all reported times.

**Norms.** Bochner $L^2(0,T; H^{-1}(\Omega))$ (Ernst et al. v3, Example 3.2):

$$\|r\|^2_{L^2(0,T; H^{-1})} \approx \sum_n \Delta t_n \|r(\cdot, t_n)\|^2_{H^{-1}(\Omega)}$$

**Conservation (BC-qualified).**
- Mass identity (any BC): $\frac{d}{dt}\int_\Omega u\, dx = \kappa \oint_{\partial\Omega} \nabla u \cdot \mathbf{n}\, dS$.
- Mass conservation $\frac{d}{dt}\int_\Omega u\, dx = 0$ holds under PER or hN but not generic Dirichlet.
- Energy dissipation (any BC giving $\int_{\partial\Omega} u\, \nabla u \cdot \mathbf{n}\, dS \leq 0$): $\frac{d}{dt}\int_\Omega u^2\, dx \leq 0$.

**Sign constraints.** $u_0 \geq 0 \Rightarrow u \geq 0$ under hD or PER.

### 1.4 Wave equation

**Scope note.** Retained primarily for energy conservation demonstration (no analog in dissipative heat). Structured-grid only; scikit-fem wave is V2.

**Norm-equivalence caveat.** Hyperbolic Bochner residual is conjectural; `PH-VAR-002` warns: *"Hyperbolic norm-equivalence not rigorously established within the parabolic Ernst-et-al framework; treat residual as diagnostic rather than certification."*

**Strong form.** $u_{tt} - c^2 \Delta u = 0$.

**Analytical solutions.**
1. Standing wave on $[0,1]^2$ with hD: $u = \sin(\pi x)\sin(\pi y)\cos(\pi c\sqrt{2}\, t)$.
2. Periodic traveling wave on $[0, L]^2$ with PER: $u = \cos(2\pi(x - ct)/L)$.

**Conservation (BC-qualified).** Energy $E(t) = \frac{1}{2}\int_\Omega(u_t^2 + c^2|\nabla u|^2)\, dx = \text{const}$ under hD, hN, or PER.

**Symmetries.** Operator admits translations, $SO(d)$ spatial rotation. Problem symmetry instance-dependent. Relativistic/Lorentz-like symmetries are not claimed.

---

## 2. Package identity & tooling stack

### 2.1 Language / runtime

- **Python floor:** 3.10 (CI matrix covers 3.10, 3.11, 3.12)
- **Build backend:** hatchling + PEP 621 `[project]` table
- **License:** Apache-2.0 (patent grant matters for commercial-adjacent MLOps positioning; matches pydantic/FastAPI/HuggingFace/PhysicsNeMo neighborhood)

### 2.2 Runtime dependencies

| Package | Version floor | Role |
|---|---|---|
| `numpy` | 1.26 | Arrays, FFT, trapezoidal integration |
| `scipy` | 1.11 | Sparse linear solves for FE $H^{-1}$ |
| `torch` | 2.0 | CallableField derivatives, SO(2) LEE via `autograd.functional.jvp`, `vmap` |
| `pydantic` | 2.0 | DomainSpec validation, JSON Schema export |
| `typer` | 0.12 | CLI framework with type-hint-driven `--help` |
| `rich` | any recent | CLI output, tables, status glyphs |
| `tomli` | 2.0 | TOML parsing on Python < 3.11; conditional via `python_version<'3.11'` marker |

**Optional:** `scikit-fem>=10` under the `physics-lint[mesh]` extra (conditional on the Week-1 Day-2 spike).

### 2.3 Developer tooling

- **Lint + format:** ruff (format + lint + isort in one tool)
- **Tests:** pytest + pytest-cov + **hypothesis** (Week 1, not deferred)
- **Docs:** Sphinx + MyST + `furo` theme (LaTeX math is load-bearing for rule documentation; MyST gives mkdocs-style prose with Sphinx's math/cross-ref engine)
- **Spelling:** codespell + `.codespellrc` with a scientific-terminology allowlist (Bochner, quadrature, Riesz, FOSLS, etc.)
- **Pre-commit:** in-repo `.pre-commit-config.yaml` runs ruff, codespell, and a scoped hypothesis smoke check
- **Type checker:** none in V1 — pydantic runtime validation + ruff's type-aware checks are sufficient for ~4000 LOC. Reconsider for V2 if surface grows.
- **Coverage gate:** 85% line coverage; CI fails below.
- **Versioning:** SemVer; `0.0.0.dev0` placeholder Week-1 Day-1 → `1.0.0` Week-4 Day-5
- **Changelog:** `CHANGELOG.md` in Keep-a-Changelog format

### 2.4 CI matrix (slim, 6 jobs)

Full cross of Python × NumPy × PyTorch × OS would be 36 cells; boundary sampling gives:

| # | OS | Python | NumPy | PyTorch | Role |
|---|---|---|---|---|---|
| 1 | ubuntu-latest | 3.10 | 1.26 | 2.0 | Floor-of-floors |
| 2 | ubuntu-latest | 3.11 | 1.26 | 2.2 | Middle |
| 3 | ubuntu-latest | 3.12 | 2.0 | 2.5 | Latest everything |
| 4 | ubuntu-latest | 3.12 | 1.26 | 2.5 | NumPy 1.x ceiling check |
| 5 | ubuntu-latest | 3.10 | 2.0 | 2.2 | NumPy 2.x floor check |
| 6 | macos-14 (arm64) | 3.11 | 2.0 | 2.5 | Apple Silicon smoke |

The six boundary cells cover the oldest-everything path, the newest-everything path, both NumPy-axis boundaries, a mid-path interpolation, and an Apple Silicon platform check. Release criterion 1 is "`physics-lint self-test` exits zero on all six jobs."

---

## 3. Field abstraction

### 3.1 Field ABC

```python
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

class Field(ABC):
    """Abstract field over a discretized domain.

    `values()` returns the underlying stored array (no args); `at(x)` evaluates
    at arbitrary coordinates via interpolation or AD.

    Smoothness caveat: physics-lint cannot introspect arbitrary torch.nn.Modules
    for non-C2 activations. PH-NUM-003's detection is a best-effort scan of
    named submodules (nn.ReLU, nn.LeakyReLU, nn.ELU, etc.) and does NOT detect
    F.relu functional calls inside a forward method. For second-order PDEs with
    callable fields, treat PH-NUM-003 as a check against a common class of
    footguns, not a proof of smoothness.
    """

    @abstractmethod
    def values(self) -> np.ndarray: ...
    @abstractmethod
    def at(self, x: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def grad(self) -> "Field": ...
    @abstractmethod
    def laplacian(self) -> "Field": ...
    @abstractmethod
    def integrate(self, weight: Optional["Field"] = None) -> float: ...
    @abstractmethod
    def values_on_boundary(self) -> np.ndarray: ...
```

`values()` and `at(x)` are distinct: `values()` is no-arg and returns the stored array directly; `at(x)` evaluates at user-supplied coordinates.

### 3.2 GridField

Regular Cartesian grid, spacing $h$. Two derivative backends selected by the `periodic: bool` flag at construction.

**4th-order central FD (Fornberg 1988).** First derivative: $f'(x_i) \approx (-f_{i+2} + 8f_{i+1} - 8f_{i-1} + f_{i-2})/(12h)$, $O(h^4)$. Second: $f''(x_i) \approx (-f_{i+2} + 16f_{i+1} - 30f_i + 16f_{i-1} - f_{i-2})/(12h^2)$, $O(h^4)$. 2D/3D Laplacian via tensor-product. Periodic wraps; non-periodic uses 3rd-order one-sided at edges.

**Fourier spectral for periodic.** $k_j = (2\pi/L) \cdot \{j \text{ if } j < N/2, j-N \text{ otherwise}\}$. $\Delta u \approx \mathcal{F}^{-1}(-(k_x^2+k_y^2)\hat{u})$. Zero out $k_{N/2}$ for first derivatives.

**Backend auto-selection:** spectral if `periodic=True`, 4th-order FD otherwise. Smoothness is not runtime-checkable; users with non-smooth periodic data can force `backend="fd"` explicitly.

### 3.3 CallableField

Wraps `Callable[[Tensor], Tensor]`. Derivatives via `torch.autograd.functional.jacobian`, batched via `torch.vmap` (PyTorch 2.0+). Requires $C^2$ activations for second-order PDEs; ReLU-family triggers `PH-NUM-003` as a best-effort scan (see ABC docstring).

### 3.4 MeshField (conditional on Week-1 Day-2 spike)

scikit-fem-backed. NN outputs projected onto FE DOFs via `basis.project(callable)`. **Decision gate:** if the Day-2 spike (assemble Poisson stiffness, solve, verify $O(h^2)$) takes more than half a day, MeshField → V2; V1 ships GridField only.

```python
from skfem import Basis, MeshTri, ElementTriP2
mesh = MeshTri.init_symmetric()
basis = Basis(mesh, ElementTriP2())
nn_dofs = basis.project(lambda x: model(x))
u_interp = basis.interpolate(nn_dofs)
```

---

## 4. DomainSpec — pydantic v2 hierarchy

`DomainSpec` is a **superset** of the user-writable config schema, not 1:1. Config is flat, human-friendly, may have missing fields. DomainSpec is validated, fully populated, contains derived fields (e.g., `adapter_path`) and computed properties (e.g., `conserves_mass`). The merge is one-way: config → DomainSpec with defaulting and derivation.

Lives in `physics_lint/spec.py`. Rules read `DomainSpec`, never raw config. Single validation point: `DomainSpec.model_validate(...)` at the end of the merge in `load_spec()`.

### 4.1 Sub-models

```python
from pydantic import BaseModel, Field as PydField, model_validator
from typing import Literal, Optional

class GridDomain(BaseModel):
    x: tuple[float, float]
    y: tuple[float, float]
    t: Optional[tuple[float, float]] = None

    @property
    def spatial_lengths(self) -> tuple[float, ...]:
        return tuple(hi - lo for lo, hi in (self.x, self.y))

    @property
    def is_time_dependent(self) -> bool:
        return self.t is not None


class BCSpec(BaseModel):
    kind: Literal[
        "periodic",
        "dirichlet_homogeneous",
        "dirichlet",
        "neumann_homogeneous",
        "neumann",
    ]

    @property
    def preserves_sign(self) -> bool:
        return self.kind in {"dirichlet_homogeneous", "periodic"}

    @property
    def conserves_mass(self) -> bool:
        return self.kind in {"periodic", "neumann_homogeneous"}

    @property
    def conserves_energy(self) -> bool:
        return self.kind in {"periodic", "neumann_homogeneous", "dirichlet_homogeneous"}


class SymmetrySpec(BaseModel):
    declared: list[
        Literal[
            "D4",
            "C4",
            "reflection_x",
            "reflection_y",
            "translation_x",
            "translation_y",
            "SO2",
        ]
    ] = []


class FieldSourceSpec(BaseModel):
    type: Literal["grid", "callable", "mesh"]
    backend: Literal["fd", "spectral", "auto"] = "auto"
    adapter_path: Optional[str] = None    # derived, not user-written
    dump_path: Optional[str] = None       # derived, not user-written

    @model_validator(mode="after")
    def exactly_one_source(self):
        if (self.adapter_path is None) == (self.dump_path is None):
            raise ValueError("Exactly one of adapter_path or dump_path must be set")
        return self


class SARIFSpec(BaseModel):
    source_file: Optional[str] = None
    pde_line: Optional[int] = None
    bc_line: Optional[int] = None
    symmetry_line: Optional[int] = None


class DomainSpec(BaseModel):
    pde: Literal["laplace", "poisson", "heat", "wave"]
    grid_shape: tuple[int, ...] = PydField(min_length=2, max_length=3)
    domain: GridDomain
    periodic: bool = False
    boundary_condition: BCSpec
    symmetries: SymmetrySpec = SymmetrySpec()
    field: FieldSourceSpec

    diffusivity: Optional[float] = None
    wave_speed: Optional[float] = None
    source_term: Optional[str] = None
    sarif: Optional[SARIFSpec] = None

    @model_validator(mode="after")
    def pde_params_consistent(self):
        if self.pde == "heat" and self.diffusivity is None:
            raise ValueError("PDE 'heat' requires 'diffusivity'")
        if self.pde == "wave" and self.wave_speed is None:
            raise ValueError("PDE 'wave' requires 'wave_speed'")
        if self.pde in {"heat", "wave"} and not self.domain.is_time_dependent:
            raise ValueError(f"PDE '{self.pde}' requires a time domain 't'")
        return self

    @model_validator(mode="after")
    def symmetries_compatible_with_domain(self):
        if "D4" in self.symmetries.declared or "C4" in self.symmetries.declared:
            lx, ly = self.domain.spatial_lengths[:2]
            if abs(lx - ly) / max(lx, ly) > 1e-6:
                import warnings
                warnings.warn(
                    f"D4/C4 symmetry declared but domain is not square "
                    f"({lx} × {ly}); symmetry rules may produce artifacts"
                )
        return self
```

### 4.2 BCSpec computed properties replace per-rule BC taxonomy

Rev 4.3.1's §4.4 had each conservation rule re-encoding the BC-preservation check (`if pde == "heat" and bc in {"PER", "hN"}`). Those checks now live exactly once in `BCSpec`. Rules read `spec.boundary_condition.conserves_mass` (etc.) as a single boolean, which:

1. Eliminates duplicated BC logic across rules.
2. Makes the rule file itself readable as pure physics: "rate of change of total mass is zero if the BC conserves mass."
3. Forces any future BC addition to update `BCSpec` in one place rather than hunting through every rule.

### 4.3 JSON Schema export for IDE autocomplete

`DomainSpec.model_json_schema()` emits a valid JSON Schema. Committed to `physics_lint/data/config_schema.json` and regenerated in CI whenever `DomainSpec` changes. Users point VS Code's `evenBetterToml` at it for `[tool.physics-lint]` autocomplete:

```json
// .vscode/settings.json
{
  "evenBetterToml.schema.associations": {
    "pyproject.toml": "https://physics-lint.readthedocs.io/_static/config_schema.json"
  }
}
```

~15 LOC of setup; zero maintenance burden.

### 4.4 Rule signature standardized

```python
def check_rule(field: Field, spec: DomainSpec) -> RuleResult: ...
```

Every rule in `physics_lint/rules/` has this signature. Rules never touch raw config.

---

## 5. Model loading — hybrid adapter + dump

physics-lint supports two load paths, dispatched by the positional argument's file extension. Adapter mode is primary (full rule suite); dump mode is secondary (sampler-heavy or JAX/TF models). Rules declare which input modes they accept; rules running in an unsupported mode emit an explicit `SKIPPED` RuleResult with a reason — never silent omission.

### 5.1 Extension dispatch

| Extension | Mode | Loader action |
|---|---|---|
| `.py` | Adapter | `_exec_adapter(path)`; call `load_model()` and `domain_spec()` |
| `.npz` / `.npy` | Dump | `np.load(path)`; read `prediction` array and `metadata` dict |
| `.pt` / `.pth` | Error | Exit 3 with message: *"please use an adapter or convert to .npz; see docs/loading.html"* |

### 5.2 Adapter API

Single file, two functions, no class:

```python
# physics_lint_adapter.py
from physics_lint import DomainSpec
import torch

def load_model() -> torch.nn.Module | Callable[[torch.Tensor], torch.Tensor]:
    """Return a callable mapping grid coords or initial condition to prediction.

    May return either a torch.nn.Module (preferred; enables autograd rules)
    or a plain Python callable (sufficient for FD residual and structural rules).
    """
    ...

def domain_spec() -> DomainSpec:
    """Return the PDE/domain/BC/symmetry spec for this model.

    Overrides pyproject.toml [tool.physics-lint] settings where provided;
    unspecified fields fall through to TOML defaults.
    """
    ...
```

Rationale for two functions over a class (modelled on pytest's `conftest.py`):
1. Functions are easier to test and document; `python -c "from physics_lint_adapter import load_model; m = load_model()"` works.
2. pytest `conftest.py` is the right mental model for users, and pytest uses functions.
3. Two functions is the minimum viable surface. Future lifecycle hooks can promote to a class in V2 with a compatibility shim.

### 5.3 Adapter discovery order

1. `--adapter <path>` CLI flag (highest precedence)
2. `[tool.physics-lint] adapter = "path/to/adapter.py"` in `pyproject.toml`
3. Default `./physics_lint_adapter.py` in cwd (lowest precedence)

Same pattern as `pytest.ini` + `conftest.py`.

### 5.4 Dump mode format

```python
np.savez(
    "pred.npz",
    prediction=pred_array,             # required: the field values
    metadata={                          # required: DomainSpec-compatible dict
        "pde": "heat",
        "grid_shape": [64, 64, 32],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": "dirichlet_homogeneous",
        "symmetries": ["D4"],
        "diffusivity": 0.01,
    },
)
```

physics-lint `np.load`s the file, parses `metadata` into `DomainSpec`, wraps `prediction` in a `GridField`. Works identically for JAX, TF, numpy, or any framework that can write `.npz`.

### 5.5 Rule input-mode declarations

Every rule in the catalog declares its accepted input modes. Three rules are **adapter-only** in V1:

| Rule | Reason |
|---|---|
| `PH-RES-002` (FD-vs-AD cross-check) | AD branch requires a callable |
| `PH-SYM-003` (SO(2) Lie derivative equivariance) | LEE requires autograd through the model |
| `PH-NUM-003` (non-$C^2$ activation scan) | Requires a `torch.nn.Module` to iterate submodules |

When an adapter-only rule runs against a dump input, the rule emits:

```python
RuleResult(
    rule_id="PH-SYM-003",
    status="SKIPPED",
    reason="SO(2) Lie derivative equivariance requires autograd through the model; dump mode provides only a frozen tensor",
    ...
)
```

The text reporter prints these with a `⊘` glyph. The SARIF emitter routes them to `run.invocations[].toolExecutionNotifications` rather than `run.results` (§13.1) — they're diagnostics about the run, not findings about the artifact.

### 5.6 DDPM / DPS implication

Iterative samplers like DDPM and DPS are unergonomic as `Callable[[Tensor], Tensor]`: the "forward pass" means running the full sampling loop, which is slow and stochastic. Trying to adapter-wrap them forces one of two bad choices:
1. Hide sampling stochasticity behind a fixed seed and lie about determinism.
2. Rerun the sampler on every rule call (~minutes per rule; absurd for a CI tool).

Dump mode separates these concerns: the user runs the sampler once to produce `pred.npz`, then physics-lint checks it deterministically and quickly. This is why hybrid E is the right choice rather than "adapter-only with dump as an afterthought" — the dump path is load-bearing for sampler-based surrogates, not a convenience for JAX/TF.

### 5.7 Trust model

physics-lint `exec`s adapter code — same surface as pytest loading `conftest.py`. For most local usage this is fine. In CI contexts specifically, physics-lint runs adapter Python with the same token permissions as the CI job, which can include `GITHUB_TOKEN` write access. Canonical workflow (§14) sets minimum permissions:

```yaml
permissions:
  contents: read
  security-events: write
```

**Do not** grant `contents: write` or `pull-requests: write` unless explicitly needed. For public-contribution workflows, use `pull_request_target` with branch restrictions per GitHub's documented guidance.

---

## 6. Self-test infrastructure

### 6.1 Analytical battery

2 analytical solutions per PDE, chosen to satisfy the stated BCs on bounded domains; self-test asserts violations at/below discretization floor.

### 6.2 Convergence under refinement

Run rule at $N$ and $2N$. Assert $p = \log_2(\|R_N\|/\|R_{2N}\|) \geq 3.5$ for 4th-order FD; exponential for spectral on smooth periodic.

### 6.3 `physics_lint/data/floors.toml`

Hybrid D from the Q3 decision: pytest dev-time assertions + `physics-lint self-test` user-facing CLI + hardcoded floors in a TOML file that both paths load. Floors are overridable via config but not stateful-per-machine.

**Schema (array-of-tables):**

```toml
schema_version = 1

[[floor]]
rule = "PH-RES-001"
pde = "laplace"
grid_shape = [64, 64]
method = "spectral"
norm = "H-1"
value = 1e-14
tolerance = 3.0
analytical_solution = "harmonic_polynomial"
citation = "Trefethen 2000, §3"

[[floor]]
rule = "PH-RES-001"
pde = "laplace"
grid_shape = [64, 64]
method = "fd4"
norm = "H-1"
value = 1e-6
tolerance = 2.0
analytical_solution = "harmonic_polynomial"
citation = "Fornberg 1988"

[[floor]]
rule = "PH-CON-001"
pde = "heat"
grid_shape = [64, 64, 32]
method = "fd4"
norm = "relative"
bc_branch = "per_hn"
value = 1e-5
tolerance = 2.0
analytical_solution = "eigenfunction_decay"
citation = "Ernst 2025 v3, §3.2"

[[floor]]
rule = "PH-CON-001"
pde = "heat"
grid_shape = [64, 64, 32]
method = "fd4"
norm = "relative_L2_over_T"
bc_branch = "hd"
value = 1e-4
tolerance = 2.5
analytical_solution = "eigenfunction_decay"
citation = "Ernst 2025 v3, §3.2"

# ~30 entries total across 4 PDEs × {FD, spectral} × {L2, H-1} × BC branches
```

Per-floor `tolerance` rather than global: spectral floors near FFT backend noise need 3×; FD floors where truncation dominates can use 2×; heat-hD-rate-consistency (the `bc_branch = "hd"` entry, added per Section-2 review) uses 2.5× because it involves both an integral and a time derivative. FFT backend differences (pocketfft vs MKL vs Accelerate) can drift up to $10^{-13}$ on the same input, and torch/numpy reduction-order differences can drift in large grids — both argue against tight additive tolerances.

**Why multiplicative:** the release criterion ("measured ≤ value × tolerance") survives backend drift, platform differences, and minor NumPy/PyTorch version updates without hand-tuning.

### 6.4 Week-1 floor calibration protocol

**Half-day budget, Day 5.** Two-step branch workflow to avoid circularity:

1. **Calibration run.** Create a throwaway branch `chore/floor-calibration`. Author a small script that runs the analytical battery at the specified (rule, pde, grid_shape, method, norm) tuples and records measured values. Run the script in three environments:
   - Local laptop (macOS arm64)
   - Fresh Ubuntu Docker container (`python:3.11-slim` + `pip install -e .`)
   - Throwaway GitHub Actions workflow (one matrix cell per CI environment; ephemeral)
   Record the measured values in a gist (not committed to the repo).

2. **Population PR.** Open a PR against `main` with `floors.toml` populated from the **maximum** observed value across environments — not minimum, not median. The PR's CI validates that the floors are met on the full matrix. If CI passes, merge; if CI fails on a specific cell, investigate (bug in the battery? bug in a rule? genuinely divergent platform?) rather than lowering the floor value.

**Why two steps:** if the floor-calibration PR's CI were the same run that produced the floors, the check would be tautological. Separating calibration (branch + gist) from validation (PR CI) removes the circularity.

**Why maximum, not minimum/median:** preventing "works on my machine" failures at first `pip install`. A user whose floor is slightly above the median won't hit spurious FAILs.

### 6.5 User override path

```toml
[tool.physics-lint.rules."PH-RES-001"]
floor_override = {
    pde = "laplace",
    grid_shape = [64, 64],
    method = "spectral",
    norm = "H-1",
    value = 3e-14,
}
```

At check time, physics-lint loads shipped floors from `physics_lint/data/floors.toml`, then applies any `floor_override` entries from the user's config. The override must match an existing `(rule, pde, grid_shape, method, norm)` tuple; unknown overrides raise a config error rather than silently adding new floors.

### 6.6 Self-test CLI surface

```bash
physics-lint self-test                          # full battery, exit 0/1
physics-lint self-test --verbose                # per-rule measured vs expected table
physics-lint self-test --rule PH-RES-001        # single rule
physics-lint self-test --write-report out.json  # CI artifact / bug-report attachment
```

**Triage benefit:** when a user files "physics-lint is giving me weird ratios on my model," the maintainer's first question becomes "does `self-test` pass?" — if yes, it's adapter/config; if no, it's install/platform. Stateless reproducibility matters more than any calibration sophistication.

### 6.7 Release criterion 1 coupling

The release criterion and the user-facing verification command run the same code. Criterion 1 (§11) reads: "`physics-lint self-test` exits zero on all six CI matrix jobs." Users verifying their install run the same command. The forcing function is that a skeptical reviewer can validate criterion 1 from a clean `pip install` in thirty seconds, not by trusting the maintainer's dev CI.

---

## 7. Residual computation

### 7.1 Residual formulas

| PDE | Residual |
|-----|----------|
| Laplace | $R = -\Delta u$ |
| Poisson | $R = -\Delta u - f$ |
| Heat | $R = u_t - \kappa\Delta u$ |
| Wave | $R = u_{tt} - c^2\Delta u$ |

### 7.2 FD-vs-AD cross-check (`PH-RES-002`)

$$\text{discrepancy} = \frac{|R_{\text{FD}} - R_{\text{AD}}|}{\max(|R_{\text{FD}}|, |R_{\text{AD}}|, \epsilon_{\text{floor}})}$$

Default threshold 0.01. Above: `CheckerDiagnosticWarning`. `grad_method ∈ {"autodiff", "finite_difference", "spectral"}` follows the multi-backend pattern in NVIDIA PhysicsNeMo Sym. **Adapter-only**: dump mode emits `SKIPPED` with reason "FD-vs-AD requires callable."

### 7.3 Spectral differentiation

```python
def spectral_laplacian_2d(u, Lx, Ly):
    Nx, Ny = u.shape
    kx = np.fft.fftfreq(Nx, d=Lx/Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly/Ny) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    u_hat = np.fft.fft2(u)
    return np.real(np.fft.ifft2(-(KX**2 + KY**2) * u_hat))
```

### 7.4 Norm module (`physics_lint/norms.py`)

- **`l2_grid(u, h)`** — trapezoidal $h^d \sum |u|^2$ with half-weight at boundaries.
- **`h_minus_one_spectral(r, L)`** — $\sum_{k \neq 0} |\hat r_k|^2 / |k|^2$.
- **`h_minus_one_fe(r_h, K)`** — solve $K\hat c = r_h$, return $\hat c^T K \hat c$ (Ernst et al. v3, §3.4; conditional on spike).
- **`bochner_l2_h_minus_one(r_series, dt, L)`** — midpoint $\sum_n \Delta t_n \|r(\cdot, t_n)\|^2_{H^{-1}}$.

### 7.5 Variational-correctness rules

| PDE | Recommended norm | Override warning |
|-----|-------------|------------------|
| Laplace/Poisson (2nd-order strong-form) | $H^{-1}$ | `PH-VAR-001` |
| Heat (2nd-order strong-form) | Bochner $L^2(0,T; H^{-1})$ | `PH-VAR-001` |
| Wave (2nd-order strong-form) | Bochner $L^2(0,T; H^{-1})$ | `PH-VAR-001` + `PH-VAR-002` |

**`PH-VAR-001` text:** *"For the chosen second-order strong-form residual, the plain L² norm is not the recommended variationally correct test-space norm. Suitable first-order (FOSLS-type) reformulations exist for some PDE classes (Bachmayr et al. 2024; Qiu et al. 2025, scoped to stationary diffusion and linear elasticity) where L² becomes appropriate. Use the recommended norm or switch formulation."*

**`PH-VAR-002` text:** *"Hyperbolic norm-equivalence is not established within the parabolic Ernst-et-al framework; treat residual as diagnostic, not certification."*

---

## 8. Conservation, boundary, and positivity rules

### 8.1 Structured-grid conservation (`PH-CON-001`, `PH-CON-002`)

Per-cell finite volume:

$$\mathcal{V}_{i,j} = (u^{n+1}_{i,j} - u^n_{i,j})\Delta x \Delta y + \Delta t\big[(F^x_{i+\frac{1}{2},j} - F^x_{i-\frac{1}{2},j})\Delta y + (F^y_{i,j+\frac{1}{2}} - F^y_{i,j-\frac{1}{2}})\Delta x\big]$$

Central face averaging. Boundary flux $\oint \mathbf{F}\cdot\mathbf{n}\, dS$ reported separately.

**BC scoping via `BCSpec`.** `PH-CON-001` reads `spec.boundary_condition.conserves_mass`:
- If True (PER/hN): check $|M(t) - M(0)| / |M(0)|$ against the `bc_branch = "per_hn"` floor.
- Otherwise (hD / generic Dirichlet): check rate-consistency $\frac{dM}{dt} \approx \kappa \oint \nabla u \cdot \mathbf n \, dS$ by computing both sides as time series, reporting relative $L^2$ error over $[0, T]$, and comparing against the `bc_branch = "hd"` floor (§6.3). This is a concrete threshold, not a vague `≈`.

Mismatch between configured BC and adapter's declared BC fires `PH-NUM-004`.

### 8.2 Weak-form conservation (conditional on spike)

```python
from skfem import Functional

@Functional
def divergence_residual_sq(w):
    from skfem.helpers import grad
    div_F = grad(w['Fx'])[0] + grad(w['Fy'])[1]
    return div_F ** 2

elem_violations = divergence_residual_sq.elemental(
    basis, Fx=basis.interpolate(Fx_dofs), Fy=basis.interpolate(Fy_dofs)
)
```

### 8.3 Adaptive quadrature (`PH-NUM-001`)

Default $2\times$ FE polynomial order. `qorder` user-exposed; warning when overridden: "NN integrands non-polynomial; quadrature convergence not guaranteed." Auto-convergence test compares orders $q$ and $2q$; warns if relative change $> 10^{-4}$.

### 8.4 Per-PDE conservation rules

- **Heat.** `PH-CON-001` as above (PER/hN mass or hD rate-consistency). `PH-CON-003`: energy dissipation sign violation when `spec.boundary_condition.conserves_energy`.
- **Wave.** `PH-CON-002` reads `spec.boundary_condition.conserves_energy` → checks $|E(t) - E(0)| / E(0)$; $u_t$ via 4th-order FD.
- **Laplace/Poisson.** Divergence theorem steady-state check under any well-posed BC (`PH-BC-002`).

### 8.5 Boundary condition rule (`PH-BC-001`) — mode-branched

```python
def bc_check(
    field: Field,
    boundary_values: np.ndarray,
    *,
    norm: str = "L2",
    abs_threshold: float = 1e-10,
    abs_tol_fail: float = 1e-3,
) -> RuleResult:
    """Boundary condition violation with mode-branched normalization.

    If ||g|| < abs_threshold, switch to absolute mode (raw boundary error).
    Otherwise, report relative error ||u - g|| / ||g||.
    """
    u_boundary = field.values_on_boundary()
    err_norm = norm_fn(u_boundary - boundary_values, norm)
    g_norm = norm_fn(boundary_values, norm)

    if g_norm < abs_threshold:
        status = "PASS" if err_norm < abs_tol_fail else "FAIL"
        return RuleResult(
            rule_id="PH-BC-001",
            status=status,
            mode="absolute",
            raw_value=err_norm,
            ...
        )
    else:
        ratio = err_norm / g_norm / FLOOR["PH-BC-001-rel"]
        status = _tristate(ratio, pass_=10, fail_=100)
        return RuleResult(
            rule_id="PH-BC-001",
            status=status,
            mode="relative",
            raw_value=err_norm / g_norm,
            violation_ratio=ratio,
            ...
        )
```

**Absolute mode is deliberately binary PASS/FAIL in V1.** No `abs_tol_warn` tier: the natural scale for "intermediate" deviation is problem-specific and cannot be calibrated without domain knowledge from the user. If a project needs a WARN tier in absolute mode, it can promote FAIL to WARN via the `severity` override in config. The WARN state is reserved for relative mode.

The `RuleResult.mode` field is displayed directly on the text report's `PH-BC-001` line (`[absolute mode]` / `[relative mode]`) so users immediately know which branch fired — critical for distinguishing hD from inhomogeneous Dirichlet.

### 8.6 Positivity / maximum principle (`PH-POS-001`, `PH-POS-002`)

```python
def positivity_check(field: Field, spec: DomainSpec, *, floor: float = 0.0) -> RuleResult:
    if not spec.boundary_condition.preserves_sign:
        return RuleResult(
            rule_id="PH-POS-001",
            status="SKIPPED",
            reason="Configured BC does not preserve sign; rule not applicable",
        )
    u = field.values()
    violations = (u < floor)
    return RuleResult(
        rule_id="PH-POS-001",
        status="FAIL" if violations.any() else "PASS",
        raw_value=float(u.min()),
        spatial_map=violations,
        ...
    )


def maximum_principle_check(field: Field, spec: DomainSpec, *, boundary_values) -> RuleResult:
    u = field.values()
    below = max(0.0, boundary_values.min() - u.min())
    above = max(0.0, u.max() - boundary_values.max())
    ...
```

`PH-POS-001` applies under sign-preserving BCs (heat: hD or PER; Poisson: hD with $f \geq 0$). `PH-POS-002` applies to harmonic functions under any well-posed Dirichlet problem.

### 8.7 Jekel caveat (rule docs)

`PH-CON-001` documentation page includes: *"Conservation is necessary but not sufficient for accuracy. A model may conserve mass perfectly while producing spatially incorrect solutions (Jekel et al. 2022)."*

---

## 9. Symmetry / equivariance rules

### 9.1 Scope

V1 operationalizes **isometries only**:

- Finite-transformation tests: `PH-SYM-001` ($C_4$), `PH-SYM-002` (reflections), `PH-SYM-004` (translation, **periodic-only in V1**) — exact on grids via index permutation.
- SO(2) LEE: `PH-SYM-003` — single-generator, ~80 LOC, Gruver et al. Fig. 3 pattern.

V2 deferred: layerwise LEE decomposition; LEE for scaling, Galilean, Kelvin; multi-generator framework; non-periodic translation via interpolation (ducked in V1 to keep every SYM rule that runs exact on its target case).

### 9.2 Finite-transformation testing

```python
def equivariance_error(model, x, T_in, T_out, norm: str = "L2") -> float:
    lhs = model(T_in(x))
    rhs = T_out(model(x))
    return torch.norm(lhs - rhs).item() / max(torch.norm(lhs).item(), 1e-12)
```

$90°/180°/270°$ via `np.rot90`, reflections via `np.flip`, periodic translations via `np.roll` — exact, no interpolation.

### 9.3 SO(2) LEE (`PH-SYM-003`, adapter-only)

$$\text{LEE}(f) = \mathbb{E}_{x}\left[\frac{\|L_X f(x)\|^2}{\dim(V_2)}\right], \quad L_X f(x) = \frac{d}{dt}\bigg|_{t=0} \rho_{21}(\Phi^t_X)[f](x)$$

```python
import torch
import torch.nn.functional as F
from torch.autograd.functional import jvp

def rotation_lie_deriv(model, imgs):
    def rotated_model(theta):
        c, s = torch.cos(theta), torch.sin(theta)
        z0 = torch.zeros_like(theta)
        m = torch.stack([c, s, z0, -s, c, z0]).reshape(1, 2, 3).expand(imgs.shape[0], -1, -1)
        grid = F.affine_grid(m, imgs.size(), align_corners=True)
        x_rot = F.grid_sample(imgs, grid, align_corners=True)
        z = model(x_rot)
        if z.ndim == 4:
            m_inv = m.clone(); m_inv[:, 0, 1] *= -1; m_inv[:, 1, 0] *= -1
            grid_inv = F.affine_grid(m_inv, z.size(), align_corners=True)
            z = F.grid_sample(z, grid_inv, align_corners=True)
        return z

    theta0 = torch.zeros(1, requires_grad=True)
    tangent = torch.ones_like(theta0)
    _, lie_deriv = jvp(rotated_model, (theta0,), v=(tangent,))
    return lie_deriv
```

The `jvp` call passes the tangent vector via `v=` as required by `torch.autograd.functional.jvp(func, inputs, v=None, ...)`.

**V2 migration note.** PyTorch 2.11+ prefers `torch.func.jvp` for forward-mode autodiff: better performance, composes with `torch.vmap`. V1 uses `torch.autograd.functional.jvp` because it is stable across PyTorch 2.0+ without functorch-specific imports. Port in V2 when the layerwise chain-rule LEE decomposition is added, since that design assumes per-layer forward-mode evaluation.

### 9.4 Per-PDE symmetry admissibility — operator-level, problem-instance-dependent

The table below lists symmetries the *PDE operator* admits. Whether a given *problem instance* respects these depends on the computational domain $\Omega$, the source or initial condition, and the BCs. Helwig et al. (2023) emphasize that global symmetries are broken by domain or BC asymmetries even when the operator is symmetric. **Users declare which symmetries apply to their problem instance** via `SymmetrySpec.declared`; physics-lint does not auto-detect.

| PDE | Operator-admissible discrete | Operator-admissible continuous |
|-----|------------------------------|--------------------------------|
| Laplace | $D_4$, translation | $SO(2)$, scaling, conformal (2D) |
| Poisson | $D_4$ (if $f$ symmetric) | $SO(2)$ (if $f$ symmetric) |
| Heat | $D_4$, time/space translation | $SO(2)$, parabolic scaling, Galilean |
| Wave | $D_4$, time translation | $SO(d)$ spatial rotation |

V1 operationalizes only isometries (translations, $D_4$, $SO(2)$). Scaling, Galilean, Kelvin, and any relativistic analogs → V2.

### 9.5 Grid-aware testing

$90°/180°/270°$ exact on square grids. Arbitrary angles would require bilinear interpolation flagged `artifact_prone=True`; deferred to V2 rather than shipping a degraded SYM rule in V1.

### 9.6 Validation target

Criterion 4: deliberately non-equivariant CNN with positional embeddings shows `PH-SYM-001` violation $> 2\times$ baseline (~1 day Week-3 Day-5). Helwig et al. 2023 cited as methodological precedent in rule docs.

---

## 10. Rule catalog

### 10.1 Rule ID scheme

Format: `PH-<CATEGORY>-<NNN>`. Categories:

- **RES** — residual checks
- **BC** — boundary conditions
- **CON** — conservation laws
- **POS** — positivity / maximum principle
- **SYM** — symmetry / equivariance
- **VAR** — variational-correctness meta-warnings
- **NUM** — numerical method meta-warnings

Rule IDs are stable across versions; rule descriptions and thresholds may evolve. Renaming a rule requires a deprecation cycle (one minor version warning, removal in next major).

### 10.2 V1 rule catalog

| Rule ID | Name | Severity | Input modes | Applies to |
|---------|------|----------|---|---|
| `PH-RES-001` | Residual exceeds variationally-correct norm threshold | error | adapter + dump | all PDEs |
| `PH-RES-002` | FD-vs-AD residual cross-check discrepancy | warning | **adapter only** | all PDEs |
| `PH-RES-003` | Spectral-vs-FD residual discrepancy on periodic grid | warning | adapter + dump | periodic domains |
| `PH-BC-001` | Boundary condition violation (relative or absolute mode) | error | adapter + dump | Dirichlet PDEs |
| `PH-BC-002` | Boundary flux imbalance (divergence theorem) | warning | adapter + dump | Poisson, Laplace |
| `PH-CON-001` | Mass conservation violation | error | adapter + dump | heat (PER/hN mass; rate-consistency under hD against `bc_branch = "hd"` floor) |
| `PH-CON-002` | Energy conservation violation | error | adapter + dump | wave (hD/hN/PER) |
| `PH-CON-003` | Energy dissipation sign violation | warning | adapter + dump | heat (hD/hN/PER) |
| `PH-CON-004` | Per-element conservation hotspot | warning | adapter + dump | meshes (if shipped) |
| `PH-POS-001` | Positivity violation | error | adapter + dump | heat ($u_0 \geq 0$, hD/PER), Poisson ($f\geq 0$, hD) |
| `PH-POS-002` | Maximum principle violation | error | adapter + dump | Laplace (well-posed Dirichlet) |
| `PH-SYM-001` | $C_4$ rotation equivariance violation | warning | adapter + dump | declared $C_4$ or $D_4$ |
| `PH-SYM-002` | Reflection equivariance violation | warning | adapter + dump | declared reflections or $D_4$ |
| `PH-SYM-003` | SO(2) Lie derivative equivariance violation | warning | **adapter only** | declared SO(2) |
| `PH-SYM-004` | Translation equivariance violation | warning | adapter + dump | **periodic domains only** |
| `PH-VAR-001` | L² residual on second-order strong-form formulation | info | adapter + dump | user override |
| `PH-VAR-002` | Hyperbolic norm-equivalence conjectural | info | adapter + dump | wave |
| `PH-NUM-001` | Quadrature convergence warning | warning | adapter + dump | meshes (if shipped) |
| `PH-NUM-002` | Refinement convergence rate below expected | warning | adapter + dump | all rules |
| `PH-NUM-003` | Activation function non-$C^2$ for second-order PDE (best-effort scan) | warning | **adapter only** | CallableField |
| `PH-NUM-004` | Configured BC inconsistent with model training BC | warning | adapter + dump | all PDEs |

**Three adapter-only rules** (`PH-RES-002`, `PH-SYM-003`, `PH-NUM-003`) skip gracefully in dump mode: `RuleResult(status="SKIPPED", reason=...)` with an explicit human-readable reason. The text reporter prints `⊘ PH-XYZ-NNN SKIP <reason>`. The SARIF emitter routes them to `run.invocations[].toolExecutionNotifications`, not `run.results`.

### 10.3 Severity levels

- **error** — rule failure produces non-zero exit code; CI fails.
- **warning** — rule failure annotated in report; CI passes.
- **info** — informational; appears in verbose output only.

Severity overridable per rule via config (e.g., promote `PH-CON-003` from warning to error).

### 10.4 Per-rule docs via Sphinx autodoc with lazy registry

Each rule module lives at `physics_lint/rules/ph_xxx_nnn.py` and exports:

```python
# physics_lint/rules/ph_res_001.py
__rule_id__ = "PH-RES-001"
__rule_name__ = "Residual exceeds variationally-correct norm threshold"
__default_severity__ = "error"
__default_thresholds__ = {"tol_pass": 10.0, "tol_fail": 100.0}
__input_modes__ = frozenset({"adapter", "dump"})

"""
Residual check in the variationally-correct norm.

For Laplace and Poisson, the recommended norm is :math:`H^{-1}`. For heat
and wave, Bochner :math:`L^2(0, T; H^{-1})`. See :math:`\text{PH-VAR-001}`
for the narrowing to second-order strong-form residuals.

**Citation:** Bachmayr, Dahmen, Oster (2024).

**Override syntax:**

.. code-block:: toml

   [tool.physics-lint.rules."PH-RES-001"]
   tol_pass = 5.0
   tol_fail = 50.0
"""

def check(field, spec):  # NOT imported at rule-list time
    ...
```

**Lazy registry** — `physics_lint/rules/_registry.py` discovers rules via `importlib.resources` and reads only module-level metadata at registry load time. The `check()` function is imported lazily, only when the rule actually fires. `physics-lint rules list` reads metadata and returns in <50 ms; a full `physics-lint check` imports only the rules that are enabled for the given `DomainSpec`.

Sphinx + `autosummary` + `numpydoc` auto-generates the rule index at `/rules/index.html` from the lazy registry's metadata view. Per-rule pages at `/rules/PH-XXX-NNN.html` are rendered from the module docstring. Zero hand-written markdown per rule; the docstring is the single source of truth, enforced by a pre-commit hook asserting `__rule_id__` matches the filename.

---

## 11. Report schema

```python
from dataclasses import dataclass
from typing import Literal, Optional

Status = Literal["PASS", "WARN", "FAIL", "SKIPPED"]
_STATUS_RANK: dict[Status, int] = {"SKIPPED": 0, "PASS": 0, "WARN": 1, "FAIL": 2}

@dataclass
class RuleResult:
    rule_id: str
    rule_name: str
    severity: Literal["error", "warning", "info"]
    status: Status
    raw_value: Optional[float]              # None when SKIPPED
    violation_ratio: Optional[float]        # None when SKIPPED
    mode: Optional[str]                     # "relative" | "absolute" for PH-BC-001
    reason: Optional[str]                   # human-readable, required when SKIPPED
    refinement_rate: Optional[float]
    spatial_map: Optional["np.ndarray"]
    recommended_norm: str
    citation: str
    doc_url: str

@dataclass
class PhysicsLintReport:
    pde: str
    grid_shape: tuple
    rules: list[RuleResult]
    metadata: dict

    @property
    def overall_status(self) -> Status:
        if not self.rules:
            return "PASS"
        return max(self.rules, key=lambda r: _STATUS_RANK[r.status]).status

    @property
    def status_counts(self) -> dict[Status, int]:
        from collections import Counter
        counts = Counter(r.status for r in self.rules)
        return {s: counts.get(s, 0) for s in ("PASS", "WARN", "FAIL", "SKIPPED")}

    @property
    def exit_code(self) -> int:
        """Non-zero if any error-severity rule failed. SKIPPED never contributes."""
        return int(any(r.status == "FAIL" and r.severity == "error" for r in self.rules))

    def summary(self) -> str: ...
    def plot(self, figsize=(12, 8)) -> "matplotlib.Figure": ...
    def to_dict(self) -> dict: ...
    def to_json(self) -> str: ...
    def to_sarif(self, category: str = "physics-lint") -> dict: ...
```

**`_STATUS_RANK`** is module-level so `overall_status`, `status_counts`, and any future sort/filter operations share one source of truth. SKIPPED has rank 0 (same as PASS) — a skipped rule never moves overall status. `status_counts` always reports all four keys with explicit zeros so the text formatter can render `"2 fail · 1 warn · 4 pass · 3 skip"` predictably.

### 11.1 Text report output

```
physics-lint report — heat on 64×64×32 grid — overall: FAIL (exit 1)
   2 fail · 1 warn · 4 pass · 3 skip

  ✓ PH-RES-001  PASS  Residual [Bochner L²(H⁻¹)]   ratio 2.3   floor 5.1e-05
  ⊘ PH-RES-002  SKIP  FD-vs-AD cross-check         requires callable; dump mode
  ✓ PH-BC-001   PASS  BC violation [absolute mode] value 1.2e-05 (||g||≈0)
  ✗ PH-POS-001  FAIL  Positivity violation         847 cells negative; min=-0.034
                      → physics-lint.readthedocs.io/rules/PH-POS-001
  ✗ PH-CON-001  FAIL  Mass conservation [hD rate]  ratio 240; localized top-left
                      → physics-lint.readthedocs.io/rules/PH-CON-001
                      → Jekel 2022 for interpretation
  ⚠ PH-CON-003  WARN  Energy dissipation sign      ratio 45; local increase at t=0.3
  ⊘ PH-SYM-003  SKIP  SO(2) LEE                    requires callable; dump mode
  ⊘ PH-NUM-003  SKIP  Non-C² activation scan       requires torch.nn.Module
  ✓ PH-SYM-001  PASS  C₄ rotation equivariance     error 0.008
```

The `⊘` glyph distinguishes SKIP from `✓` / `⚠` / `✗`. Mode/branch tags (`[absolute mode]`, `[hD rate]`) inline for clarity.

---

## 12. CLI, configuration, and CI integration

### 12.1 CLI surface

Four top-level subcommands, implemented with typer:

```bash
physics-lint check <target> [options]     # primary entry point
physics-lint self-test [options]           # release criterion 1 verification
physics-lint rules (list | show <id>)      # rule catalog browsing
physics-lint config (init | show)          # config scaffolding and debugging
```

**`check <target>`** — extension-dispatched loader from §5.1.

```bash
physics-lint check ./physics_lint_adapter.py --config pyproject.toml
physics-lint check ./pred.npz --pde heat --grid 64,64,32
physics-lint check ./adapter.py --format sarif --category physics-lint-fno --output out.sarif
physics-lint check ./adapter.py --disable PH-SYM-003 --severity PH-CON-003=error
```

Flags: `--config`, `--pde`, `--grid`, `--domain`, `--periodic`, `--bc`, `--disable <rule_id>`, `--enable-only <rule_id>,...`, `--severity <rule_id>=<level>`, `--format {text,json,sarif}`, `--category <string>`, `--output <path>`, `--verbose`.

**`self-test`** — §6.6 surface (`--verbose`, `--rule`, `--write-report`).

**`rules list`** — reads metadata from the lazy registry; returns in <50 ms. Shows rule ID, name, severity, input modes.

**`rules show <rule_id>`** — prints the rule's module docstring (LaTeX stripped for terminal), citation, default thresholds, override syntax, doc URL.

**`config init [--pde <name>]`** — writes a commented `[tool.physics-lint]` skeleton. With `--pde heat`, emits a heat-specific template with `diffusivity` uncommented and annotated, omits `wave_speed`. With `--pde wave`, the reverse. Bare `config init` emits a generic skeleton with all PDE-specific keys commented out.

**`config show`** — reads + merges + validates the user's config, pretty-prints the resolved `DomainSpec`, exits 0 if valid / 2 if invalid. **Callable without a target** (adapter not required): `physics-lint config show --config pyproject.toml` prints the TOML-only partial view with a note `"(adapter domain_spec() not applied; partial view)"`. Lets users debug `pyproject.toml` in isolation before they have an adapter file.

### 12.2 Exit codes

- `0` — all error-severity rules pass
- `1` — at least one error-severity rule failed
- `2` — invalid config or CLI usage
- `3` — model load failed, PDE unknown, or unsupported file extension (e.g. `.pt` / `.pth` without adapter conversion)

### 12.3 Configuration file

Canonical: `[tool.physics-lint]` in `pyproject.toml`. Fallback: standalone `physics-lint.toml` if no `[tool.physics-lint]` section exists.

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
"PH-VAR-001" = { enabled = false }

[tool.physics-lint.output]
format = "sarif"
verbose = false
plot = false

# Optional: source-map SARIF results for PR-check surfacing (Tier 2).
# Without this block, violations land in the Security tab only.
[tool.physics-lint.sarif]
source_file = "train_heat_fno.py"
pde_line = 42
bc_line = 58
```

### 12.4 Config merge path

Four sources, merged in precedence order (later overrides earlier):

1. **Shipped defaults** in `DomainSpec` field defaults (e.g., `backend="auto"`, `symmetries=[]`)
2. **`pyproject.toml` `[tool.physics-lint]`** — or standalone `physics-lint.toml` fallback
3. **Adapter's `domain_spec()` return** — overrides TOML where specified; unspecified fields fall through. `PH-NUM-004` fires if adapter and TOML disagree on `boundary_condition`.
4. **CLI flags** — highest precedence

```python
def load_spec(
    target: Path,
    config_path: Optional[Path],
    cli_overrides: dict,
) -> DomainSpec:
    raw = _load_toml(config_path)                        # source 2
    if target.suffix == ".py":
        adapter = _exec_adapter(target)
        raw = _merge(raw, adapter.domain_spec().model_dump())  # source 3
    elif target.suffix in (".npz", ".npy"):
        dump_meta = _load_dump_metadata(target)
        raw = _merge(raw, dump_meta)                     # dump metadata acts as "source 3"
    raw = _merge(raw, cli_overrides)                     # source 4
    return DomainSpec.model_validate(raw)                # single pydantic validation point
```

Every config error message the user sees comes from pydantic's field-level validation, not hand-rolled checks.

---

## 13. SARIF output

`PhysicsLintReport.to_sarif(category=...)` emits SARIF 2.1.0.

### 13.1 Structure

- **`run.tool.driver`** — `name: "physics-lint"`, `version`, `rules[]` populated from the lazy registry metadata (no rule `check()` imports needed)
- **`run.automationDetails.id`** = `category` parameter. This identifier is the same one set on the workflow's `codeql-action/upload-sarif@v4` `category:` input — set identical values in both places per GitHub's operational guidance.
- **`run.results[]`** — one result per **non-SKIPPED** rule (PASS/WARN/FAIL)
- **`run.invocations[].toolExecutionNotifications[]`** — SKIPPED rules emit as `level: note` diagnostics here, not as `results`. This is a deliberate choice: the GitHub Security tab shows `results` as alerts, and surfacing "physics-lint did not run this rule" as an alert adds noise without signal. Notifications are the correct SARIF location for run-metadata; any tool reading the SARIF programmatically still sees them.

### 13.2 Location mode branches on config

**Artifact-only (default, Tier 1 guaranteed).** `locations[0].physicalLocation.artifactLocation.uri = <target_path>`, `properties.location_mode = "artifact-only"`. Populates Security tab + code-scanning alert pages. Does not surface in PR checks.

```json
{
  "ruleId": "PH-CON-001",
  "level": "error",
  "message": { "text": "Mass conservation violation: ratio 240; localized top-left" },
  "locations": [{
    "physicalLocation": {
      "artifactLocation": { "uri": "models/heat_fno.pt" }
    }
  }],
  "properties": {
    "violation_ratio": 240.0,
    "raw_value": 0.0017,
    "spatial_hotspot": [12, 58, 5],
    "doc_url": "https://physics-lint.readthedocs.io/rules/PH-CON-001",
    "location_mode": "artifact-only"
  }
}
```

**Source-mapped (opt-in via `[tool.physics-lint.sarif]`, Tier 2).** If `source_file` is set, `locations[0].physicalLocation` carries `artifactLocation.uri = source_file` and `region.startLine = pde_line` (or `bc_line` / `symmetry_line` depending on rule category). Results surface in PR checks when the line intersects the PR diff.

```json
{
  "ruleId": "PH-CON-001",
  "level": "error",
  "message": { "text": "Mass conservation violation: ratio 240; localized top-left" },
  "locations": [{
    "physicalLocation": {
      "artifactLocation": { "uri": "train_heat_fno.py" },
      "region": { "startLine": 42, "endLine": 42 }
    }
  }],
  "properties": {
    "violation_ratio": 240.0,
    "model_artifact": "models/heat_fno.pt",
    "doc_url": "https://physics-lint.readthedocs.io/rules/PH-CON-001",
    "location_mode": "source-mapped"
  }
}
```

**Tier 3 (arbitrary inline comments on PR diff lines unrelated to any configured source mapping)** is not in V1.

### 13.3 Three-tier framing for the README

> physics-lint integrates with GitHub code scanning: every model PR populates the **Security tab** with rule violations, complete with rule documentation links and persistent state (Tier 1, always). When you configure a source file in `[tool.physics-lint.sarif]`, violations also surface in **pull-request checks** against the relevant lines in your training script (Tier 2, opt-in). Arbitrary inline diff comments on unrelated lines are not supported in V1.

---

## 14. GitHub Actions integration

### 14.1 Canonical workflow — single invocation + `if: always()` upload

```yaml
# .github/workflows/physics-lint.yml
name: physics-lint

on:
  push:
    branches: [main]
  pull_request:
  schedule:
    - cron: '0 6 * * 1'   # weekly Monday 06:00 UTC

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
          - { name: ddpm, path: models/ddpm_pred.npz }   # dump mode

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - run: pip install physics-lint

      - name: Run physics-lint on ${{ matrix.model.name }}
        id: lint
        run: |
          physics-lint check ${{ matrix.model.path }} \
              --config pyproject.toml \
              --category physics-lint-${{ matrix.model.name }} \
              --format sarif \
              --output physics-lint-${{ matrix.model.name }}.sarif

      - name: Upload SARIF
        if: always()     # upload even when check failed
        uses: github/codeql-action/upload-sarif@v4
        with:
          sarif_file: physics-lint-${{ matrix.model.name }}.sarif
          category: physics-lint-${{ matrix.model.name }}
```

**Single invocation.** The `check` step writes SARIF to a file via `--output` and exits non-zero on FAIL. The upload step uses `if: always()` to run even when `check` failed. This replaces Rev 4.3.1's `continue-on-error` + re-invocation pattern, which had two problems: (1) doubled CI runtime on the hot path, (2) race condition under nondeterminism where the SARIF-emit run and the exit-code-check run could disagree (sampling, dropout-inference, CUDA/BLAS nondeterminism), producing the credibility-failure symptom "my Security-tab alerts don't match my build status."

`if: always()` is GitHub's documented idiom for "run on failure too." Single invocation, one result, one ground truth.

### 14.2 SARIF category semantics

The category appears in two places:

1. CLI flag `--category physics-lint-<model>` (writes `run.automationDetails.id` in the SARIF file)
2. Workflow input `category: physics-lint-<model>` on `codeql-action/upload-sarif@v4`

These are **the same identifier at two layers**. GitHub's upload-action documentation describes non-overwrite semantics that depend on upload path and varies with edge cases; the safest V1 guidance is **operational, not precedence-based**: set the same value in both places and do not rely on GitHub-side arbitration. Matching values sidestep every edge case.

Per-surrogate categories (`physics-lint-unet`, `physics-lint-fno`, ...) satisfy the July 2025 GitHub policy that disallows multiple SARIF runs sharing the same tool+category in a single file.

### 14.3 Security note

physics-lint `exec`s adapter code, so in CI contexts it runs arbitrary Python with the same token permissions as the job. The canonical workflow sets `permissions: contents: read, security-events: write` at the workflow level — the minimum needed to check out the repo and upload SARIF.

**Do not grant `contents: write` or `pull-requests: write` unless you need them.** For public-contribution workflows where PR authors and repo owners differ (model zoos accepting contributions), use `pull_request_target` with branch restrictions per GitHub's documented guidance on that trigger. The README security notes section states this explicitly.

### 14.4 Pre-commit hook (for downstream users)

Shipped as `.pre-commit-hooks.yaml` in the repo so users can reference it:

```yaml
- id: physics-lint
  name: physics-lint
  entry: physics-lint check
  language: python
  files: '^(models|adapters)/.*\.(py|npz|npy)$'
  pass_filenames: true
```

### 14.5 In-repo pre-commit (for physics-lint development)

```yaml
# .pre-commit-config.yaml (in physics-lint repo itself)
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.x
    hooks:
      - id: ruff
      - id: ruff-format
  - repo: https://github.com/codespell-project/codespell
    rev: v2.x
    hooks:
      - id: codespell
        args: [--toml=pyproject.toml]
  - repo: local
    hooks:
      - id: field-property-tests
        name: Field property tests (hypothesis, ci-quick profile)
        entry: pytest tests/test_field_properties.py -q --hypothesis-profile=ci-quick
        language: system
        files: '^physics_lint/(field|norms)\.py$'
        pass_filenames: false
```

**Pre-commit hypothesis check, not self-test.** Rev 4.3.1 suggested running `self-test` on `field.py`/`norms.py` edits, but self-test on even two rules runs the full analytical battery at multiple resolutions — 5-15 seconds, at the upper bound of developer tolerance. Hypothesis with `--hypothesis-profile=ci-quick` (~3 seconds) covers the "did we break derivative correctness" property without running the full floor battery. Full self-test still runs in CI where the 15-second cost is invisible.

---

## 15. Hypothesis property-based tests

`tests/test_field_properties.py` — **Week 1, not deferred.** ~100 LOC of hypothesis strategies covering five properties:

1. **Polynomial exactness.** Polynomial of degree ≤ stencil order → FD derivative exact to machine precision. Strategy: grid size in `st.integers(8, 1024)`, polynomial degree in `st.integers(0, 3)`.

2. **Sine mode correctness.** $\sin(kx)$ on a periodic grid → spectral derivative is $k\cos(kx)$. Strategy: wavenumber $k$ in `st.integers(1, N/4)`, grid size in powers of two.

3. **Rotation commutativity.** Laplacian commutes with `np.rot90` on square grids — idempotence of $R^4$. This is a cross-check between spectral and FD backends when both are asked to compute the Laplacian of a smooth input: both must satisfy the property to machine precision, so a discrepancy localizes to one backend. Strategy: square grid size, smooth analytical input families.

4. **Refinement convergence.** Residual of an analytical solution converges at the expected rate under refinement $(N, 2N)$. Strategy: refinement factor fixed at 2, grid sizes in a safe range for the analytical solutions.

5. **Integration by parts on periodic domains.** $\int u \Delta v = \int v \Delta u$ for smooth $u, v$ — catches sign errors in the Laplacian that rotation-commutativity misses. Strategy: pairs of smooth periodic functions at various grid sizes.

Each property is 10–20 LOC of hypothesis code that exercises the whole axis-ordering + boundary-handling + backend-selection space. These are exactly the tests that catch the "FD stencil axis-swap bug that sets every floor to $10^{-6}$ instead of $10^{-14}$" failure mode — the kind of bug that would quietly corrupt every rule-report `violation_ratio` downstream.

Hypothesis profiles:

```python
# conftest.py
from hypothesis import settings, HealthCheck

settings.register_profile("ci-quick", max_examples=25, deadline=500)
settings.register_profile("ci", max_examples=200, deadline=2000)
settings.register_profile("dev", max_examples=50, deadline=None)
settings.load_profile("dev")
```

Pre-commit uses `ci-quick` (3-5 seconds), full CI uses `ci` (full coverage), local dev uses `dev`.

---

## 16. V1 release criteria

1. **Self-test passes on the full CI matrix.** `physics-lint self-test` exits zero on all six jobs from §2.4. Every rule hits its calibrated floor within the per-floor multiplicative tolerance from `physics_lint/data/floors.toml`.

2. **Convergence rate.** FD-based rules: $p \geq 3.5$ measured at refinement pair $(N, 2N)$. Spectral: residual $< 10^{-12}$ on $128^2$ smooth periodic Poisson.

3. **laplace-uq-bench dogfood — ranking criterion (reformulated).** physics-lint runs on all six surrogates in `github.com/tyy0811/laplace-uq-bench`. **Top-2 and bottom-2 by $H^{-1}$ residual agree with the published $H^1$ top-2 and bottom-2** positions. Not a Spearman $\rho$ threshold (too noisy on $n=6$ — one rank inversion moves $\rho$ substantially).

   **Fallback D:** if Week-2 Day-4 checkpoint verification reveals any checkpoints don't load cleanly, train 2–3 small surrogates inline on a canonical Laplace problem. Criterion 3 downgrades to "physics-lint produces a ranking table on ≥3 trained surrogates." The published-ranking reproduction is lost but the library's correctness claims stand.

4. **Equivariance detection.** Non-equivariant CNN with positional embeddings shows `PH-SYM-001` violation $> 2\times$ baseline.

5. **Conservation detection.** A model deliberately trained to violate mass on periodic heat is flagged by `PH-CON-001`.

6. **BC and positivity detection.** Gallery includes ≥1 model failing `PH-BC-001` despite small residual, ≥1 failing `PH-POS-001`.

7. **Broken-model gallery.** 3 MSE-vs-physics-lint disagreement pairs: over-smoothed FNO failing `PH-BC-001`; under-trained U-Net failing `PH-POS-001`; non-equivariant model failing `PH-SYM-001`.

8. **Cross-check diagnostic.** `PH-RES-002` fires on non-smooth, silent on smooth. Mode-dispatched correctly: adapter → runs, dump → SKIPPED with reason.

9. **CI integration demonstrated.** physics-lint's own repo runs `.github/workflows/physics-lint.yml` on every PR using the single-invocation + `if: always()` pattern. Per-surrogate categories upload cleanly under `codeql-action/upload-sarif@v4`. Security tab alerts appear with correct rule IDs and doc links. SKIPPED rules surface as `toolExecutionNotifications`, not as Security-tab alerts.

10. **CLI + config functional.** All four subcommands (`check`, `self-test`, `rules`, `config`) work. `pyproject.toml` and `physics-lint.toml` both loadable. Rule overrides (`enabled`, `severity`, `tol_pass`, `tol_fail`, `abs_threshold`, `abs_tol_fail`, `floor_override`) all respected. `rules list` returns in <50 ms via lazy registry. PyPI namespace `physics-lint` registered under `tyy0811`.

---

## 17. Week-by-week implementation schedule

**LOC reconciliation.** Rev 4.3.1 baseline: ~3800 production LOC with MeshField. Augmentations from the Q&A and per-section reviews: +70 pydantic DomainSpec (Q4), +100 hypothesis Week 1 (Q5), +30 lazy rule registry (§10.4), +25 hybrid loader (§5), +25 SKIPPED handling + SARIF notifications + PDE templates (Section 3 review). **Revised total: ~4050 LOC with MeshField / ~3650 without.**

Per-day numbers remain planning maxima with shared-infrastructure double-counting (analytical battery, self-test scaffolding, test utilities, notebook boilerplate appear in multiple days' task descriptions but are authored once). Naive summation overshoots the deduplicated production-code total. Tests and notebooks sit on top.

### Week 1 — Core + DomainSpec + hybrid loader + scikit-fem spike + Laplace/Poisson + BC/positivity

- **Day 1 AM.** Verify `physics-lint` available on PyPI under `tyy0811`; register `0.0.0.dev0` placeholder (fallbacks `physicslint`, `pde-lint`, `phylint`). Initialize git repo. Create `pyproject.toml` with Apache-2.0 license, hatchling backend, dep list from §2.2. ~30 min.
- **Day 1 PM – Day 2 AM.** Field ABC (§3.1). GridField FD + spectral backends (§3.2). Norms module (§7.4): `l2_grid`, `h_minus_one_spectral`. ~550 LOC.
- **Day 2 PM.** scikit-fem spike with half-day decision gate: assemble Poisson stiffness, solve, verify $O(h^2)$. If ≤ half-day → MeshField skeleton; if > → defer to V2 and reclaim Day-4 buffer.
- **Day 3.** DomainSpec pydantic hierarchy in `physics_lint/spec.py`: GridDomain, BCSpec with computed properties, SymmetrySpec with both `"C4"` and `"D4"` literals, FieldSourceSpec with `exactly_one_source` validator, `pde_params_consistent` and `symmetries_compatible_with_domain` cross-validators, JSON Schema export. Config loader `physics_lint/config.py` with merge path (§12.4). Hybrid loader (§5): extension dispatch, `_exec_adapter`, `.npz` reader, rule registry scaffolding (metadata-only iteration; `check()` import stubs). ~450 LOC.
  - **Rollback plan.** Day 3 stacks DomainSpec + loader + lazy registry scaffolding in a single day, which is the densest load-bearing day in Week 1. If Day 3 EOD finds DomainSpec + loader not converged, defer lazy registry plumbing to Week 4 Day 2 (where the CLI work lives anyway); Week 1 ships with eager rule imports. Lazy registry is a perf optimization for `rules list`, not a correctness requirement — `rules list` would be ~500 ms instead of <50 ms in the eager-import fallback, which is still acceptable for V1 release.
- **Day 4.** Residual rules `PH-RES-001`, `PH-RES-002`, `PH-RES-003` for Laplace/Poisson. `PH-BC-001` with absolute/relative branching (§8.5). `PH-BC-002` divergence theorem. `PH-POS-001` and `PH-POS-002`. Each rule honest about its input modes (SKIPPED logic in `PH-RES-002`, reason strings plumbed through `RuleResult`). ~700 LOC.
- **Day 5 AM.** Self-test analytical battery (§6.1): 2 analytical solutions per PDE satisfying stated BCs. Hypothesis property tests `tests/test_field_properties.py` — 5 properties from §15. ~150 LOC hypothesis + ~200 LOC battery.
- **Day 5 PM.** **Floor calibration half-day.** Two-step branch workflow (§6.4): calibration branch + gist records values across 3 environments (local macOS arm64 laptop, fresh Ubuntu Docker `python:3.11-slim`, throwaway GHA workflow). Open PR against `main` with `floors.toml` populated from the **maximum** observed value across envs. Validation CI on the PR confirms floors are met on all six matrix cells.

**Papers referenced Week 1:** Bachmayr 2024, PINO (Li 2021), Fornberg 1988, scikit-fem (spike only).

### Week 2 — Heat + wave + conservation + dogfood prep

- **Day 1–2.** Heat residual, Bochner norm. `PH-CON-001` with BC-scoped branches: PER/hN mass-conservation branch vs hD rate-consistency branch against the `bc_branch = "hd"` floor (§6.3). `PH-CON-003` dissipation sign. Positivity for heat. ~500 LOC.
- **Day 3.** Wave residual. `PH-CON-002` energy under hD/hN/PER via `BCSpec.conserves_energy`. `PH-VAR-002` always-on caveat. ~250 LOC.
- **Day 4 — Dogfood day 1.** Clone `github.com/tyy0811/laplace-uq-bench`. **Verify checkpoints load cleanly** — this is the Q1 flag: commit to the six-surrogate plan only after this verification. Write adapters for U-Net, FNO, deep ensemble (callable-path surrogates). Generate dumps for improved DDPM, DPS via pre-sampling (dump-path surrogates). OT-CFM decision based on measured sampling cost (adapter if ≤ 100 ms/sample, dump otherwise). If any checkpoint fails → invoke fallback D. ~150 LOC.
- **Day 5 — Dogfood day 2.** Run full Laplace rule set across all six surrogates (mixed adapter/dump). Generate the top-2/bottom-2 comparison table (criterion 3). **Also run on Virkkunen 2021 phased-array stress-test** — the README "low MSE ≠ physics-correct" marketing figure. Not a release criterion; central marketing artifact. ~150 LOC + notebook.

**Papers referenced Week 2:** Ernst v3, Jekel 2022, Hansen 2023, Virkkunen 2021.

### Week 3 — Symmetry + FE H⁻¹ (conditional) + broken-model toy

- **Day 1–2.** `PH-SYM-001` ($C_4$), `PH-SYM-002` (reflections), `PH-SYM-004` (translation, **periodic-only in V1**). Full $D_4$ on grids via declared-symmetry gating. Rules applicable only if `SymmetrySpec.declared` lists the relevant group. ~350 LOC.
- **Day 3.** `PH-SYM-003` SO(2) LEE with `jvp(func, inputs, v=(tangent,))` (§9.3). Adapter-only SKIP path with explicit reason. ~150 LOC.
- **Day 4.** If Week-1 spike passed: MeshField implementation, FE $H^{-1}$ via sparse Cholesky, `PH-CON-004` per-element hotspot, `PH-NUM-001` quadrature convergence. Else: buffer day + broken-model toy preparation. ~400 LOC if included.
- **Day 5.** Broken-model toy: non-equivariant CNN with positional embeddings on rotational Poisson problem. Validates criterion 4. ~200 LOC.

**Papers referenced Week 3:** Gruver 2023, Brandstetter 2022, Helwig 2023, e3nn (reference only).

### Week 4 — Report + CLI + config + SARIF + gallery + release

- **Day 1.** Report module: `RuleResult`, `PhysicsLintReport`, module-level `_STATUS_RANK` (SKIPPED=0=PASS), `status_counts` with explicit zero-reporting, `exit_code`, `overall_status`, `plot()`, text formatter with `⊘` glyph and `[mode]` tags. ~400 LOC.
- **Day 2.** **CLI + config** via typer (§12.1): all four subcommands. `config show` callable without target (partial-view mode). `config init --pde heat/wave` PDE-specific templates; bare `config init` generic with commented optionals. Lazy rule registry plumbing finalized — `physics_lint/rules/_registry.py` reads metadata only, imports `check()` on demand. `rules list` must return in <50 ms. ~350 LOC.
- **Day 3.** **SARIF output + GitHub Actions.** `to_sarif(category=...)` with SKIPPED rules routed to `run.invocations[].toolExecutionNotifications` rather than `run.results`. Artifact-only default; source-mapped opt-in via `[tool.physics-lint.sarif]`. Canonical workflow YAML with single-invocation + `if: always()` upload pattern. Security note for minimum `GITHUB_TOKEN` permissions. Verify physics-lint's own repo Security tab populates correctly on a test PR. ~175 LOC + workflow YAML.
- **Day 4.** Broken-model gallery notebook: 3 MSE-vs-physics-lint disagreement cases. Over-smoothed FNO with tiny residual but failing `PH-BC-001`. Under-trained U-Net with small MSE but failing `PH-POS-001`. Non-equivariant model passing MSE but failing `PH-SYM-001`. ~300 LOC + notebook.
- **Day 5.** README with GitHub Actions hero example, six-surrogate dogfood figure (Week 2 output), Virkkunen stress-test comparison. Rule docs auto-generated from docstrings via Sphinx + MyST + `furo`. README CI claim is the honest Tier 1 + Tier 2 framing from §13.3. PyPI release v1.0.0 (replacing `0.0.0.dev0` placeholder). Tag v1.0. **Buffer day** for unknown-unknowns and Week-3 overflow.

**Week 4 dev checks:** full pipeline on 6 laplace-uq-bench models produces 6 SARIF files with distinct categories; gallery shows 3 disagreements; CLI exit codes correct; physics-lint's own GitHub Actions workflow runs green on a test PR; `pip install physics-lint` works on clean env; `physics-lint self-test` exits zero in <60 s on a clean install.

---

## 18. Dogfood plan — laplace-uq-bench

**Source:** `github.com/tyy0811/laplace-uq-bench`.

### 18.1 Mixed-mode surrogate rollout

| Surrogate | Mode | Reason |
|---|---|---|
| U-Net | Adapter | Standard convnet, traces cleanly, supports full rule suite including LEE |
| FNO | Adapter | Spectral convolution, traces cleanly, supports full rule suite |
| Deep ensemble | Adapter | $N$ independent models averaged; adapter loads all $N$ and returns the mean |
| OT-CFM | **Adapter if inference ≤ 100 ms/sample, else dump** | Decided Week-2 Day-4 based on measured sampling cost |
| Improved DDPM | Dump | Iterative sampler; §5.6 reasoning — stochasticity + per-rule rerun absurdity |
| DPS | Dump | Same reasoning as DDPM |

### 18.2 Datasets

1. **In-distribution Laplace test set** — primary criterion-3 comparison (top-2/bottom-2 ranking agreement with published $H^1$).
2. **Virkkunen 2021 phased-array stress-test** — secondary demonstration of "low MSE ≠ physics-correct" under distribution shift. **Not a release criterion; central README marketing figure.** The surrogates in laplace-uq-bench showed AUROC ≈ 0.50 under this shift — a documented case where physics-lint should show the physics ranking degrading while MSE does not.

### 18.3 Fallback D (if checkpoints don't load)

If Week-2 Day-4 checkpoint verification fails for any of the six:
1. Train 2–3 small surrogates inline on the canonical Laplace MMS problem (~2–4 hours on a laptop GPU).
2. Downgrade criterion 3 to "physics-lint produces a ranking table on ≥3 trained surrogates."
3. Lose the published-ranking reproduction (reduces marketing pull) but keep the library's correctness claims.
4. README figure shows whatever surrogate set was available; the framing narrative ("physics-lint ranks models by physics, not just MSE") is unchanged.

Invoke fallback **immediately** on Day 4 if any checkpoint fails — do not spend Day 5 debugging upstream checkpoint issues on a V1 timeline.

---

## 19. Non-goals for V1

- No Navier-Stokes, MHD, compressible flow.
- No AMR, no GPU kernels.
- PyTorch + NumPy only; no JAX backend.
- No symbolic PDE definitions.
- No Galilean, parabolic-scaling, Kelvin, or relativistic symmetry tests.
- No entropy residuals for nonlinear hyperbolic.
- No uncertainty quantification (Hansen ProbConserv → V2).
- No layerwise LEE; SO(2) single-generator only.
- **`PH-SYM-004` is periodic-only in V1.** Non-periodic translation via interpolation is deferred — keeps every V1 SYM rule exact on its target case.
- No $C^2$ activation proof. `PH-NUM-003` is explicitly a best-effort scan of named submodules (`nn.ReLU`, `nn.LeakyReLU`, `nn.ELU`, etc.), does NOT detect `F.relu` functional calls inside a `forward` method, and the rule docs + Field ABC docstring state this explicitly.
- No symmetry auto-detection from PDE + domain + BC. Users declare via `SymmetrySpec`.
- No auto-fix functionality.
- MeshField and rules `PH-CON-004` / `PH-NUM-001` conditional on the Week-1 Day-2 spike.
- **PR-surfacing = Tier 1 (Security tab, always) + Tier 2 (source-mapped PR checks, opt-in via `[tool.physics-lint.sarif]`).** Tier 3 (arbitrary inline diff comments on lines unrelated to the configured source mapping) is not in V1.
- No mypy / pyright in V1 dev stack. Pydantic runtime + ruff's type-aware checks are sufficient. Reconsider for V2.
- No per-machine calibration state. Floors are shipped in `floors.toml` with multiplicative tolerances; user overrides via explicit config entries that must match existing tuples.

---

## 20. Planning-phase ship criteria

This design doc is declared the planning-phase final artifact. Implementation begins Week-1 Day-1 AM.

**Next actions (chronological):**

1. **Planning deliverable.** This doc at `docs/design/2026-04-14-physics-lint-v1.md`. Self-reviewed. User-approved. Committed to git.
2. **Week-1 Day-1 AM.** Verify `physics-lint` available on PyPI under `tyy0811`. Register `0.0.0.dev0` placeholder. Fallbacks `physicslint`, `pde-lint`, `phylint`. Initialize git repo. Create `pyproject.toml` per §2.
3. **Week-1 Day-1 PM.** Begin Field ABC and GridField FD backend per §3.2.
4. **Week-1 Day-2 PM.** scikit-fem spike with half-day decision gate.
5. **Revisions to this design doc** only in response to implementation-driven surprises — scikit-fem API quirks, FFT edge cases, PyTorch version compat, laplace-uq-bench checkpoint surprises. Not in response to further document-only review.

The document has served its purpose when Week-1 implementation begins. Document polish beyond this revision is not the critical path.

---

## 21. Changes from Rev. 4.3.1 baseline (augmentation audit)

This document is Rev. 4.3.1 of the implementation plan plus all augmentations from the brainstorming Q&A rounds and per-section reviews. Full audit:

### From Q1 (laplace-uq-bench dogfood)
- Dogfood repo pinned to `github.com/tyy0811/laplace-uq-bench`.
- Criterion 3 reformulated from "Spearman $\rho > 0.8$" to "top-2 and bottom-2 ranking agreement" (robust on $n=6$).
- Week-2 Day-4 explicit "verify checkpoints load cleanly before committing to six-surrogate plan" gate.
- Virkkunen 2021 stress-test added as secondary marketing demonstration (not a release criterion).
- Fallback D codified: train 2–3 small surrogates inline if checkpoints fail.

### From Q2 (model loading)
- §5 hybrid adapter + dump architecture, extension-dispatched.
- Adapter API: two module-level functions (`load_model`, `domain_spec`), no class.
- Discovery order: CLI flag → config → default `./physics_lint_adapter.py`.
- Dump format: `.npz` with `prediction` array and `metadata` dict.
- `.pt` / `.pth` → error with conversion guidance.
- **Rule input-modes column** in the catalog (§10.2). Three adapter-only rules (`PH-RES-002`, `PH-SYM-003`, `PH-NUM-003`).
- **SKIPPED status** with explicit `reason` field, never silent omission.
- DDPM/DPS as dump mode in the laplace-uq-bench dogfood.
- `pull_request_target` security note in §14.3.

### From Q3 (self-test infrastructure)
- §6 hybrid D: pytest dev-time + `physics-lint self-test` CLI + TOML floors.
- `physics_lint/data/floors.toml` schema as array of tables with per-floor multiplicative tolerance, `analytical_solution`, `citation` metadata.
- Week-1 Day-5 half-day floor calibration half-day with two-step branch workflow (calibration branch + gist, then PR against main with maximum-observed values).
- User override via `floor_override` config key that must match existing tuples.
- `self-test` CLI with `--verbose`, `--rule`, `--write-report`.
- Release criterion 1 reformulated to "`physics-lint self-test` exits zero on all six CI matrix jobs."
- Bug-report triage framing: self-test is a triage artifact.

### From Q4 (DomainSpec type)
- §4 pydantic v2 hierarchy: `GridDomain`, `BCSpec`, `SymmetrySpec`, `FieldSourceSpec`, `SARIFSpec`, `DomainSpec`.
- `BCSpec` computed properties (`preserves_sign`, `conserves_mass`, `conserves_energy`) replace per-rule BC taxonomy duplication.
- `SymmetrySpec.declared` includes both `"C4"` and `"D4"` literals.
- Two cross-validators: `pde_params_consistent`, `symmetries_compatible_with_domain`.
- JSON Schema export for IDE autocomplete via `DomainSpec.model_json_schema()`.
- Rule signature standardized to `check_rule(field: Field, spec: DomainSpec) -> RuleResult`.
- Week-4 Day-2 LOC budget 300 → 350 to account for pydantic glue.

### From Q5 (tooling stack)
- **License Apache-2.0** (not MIT; patent grant for MLOps positioning).
- **Docs Sphinx + MyST + furo** (not mkdocs-material; LaTeX math is load-bearing).
- **Hypothesis moved to Week 1** (not deferred; the Field ABC is too load-bearing to ship without property-based tests).
- **Handle `tyy0811` retained** (professional identity surfaced via display name and authorial metadata; no handle migration).
- **codespell + scientific-terminology allowlist** added to pre-commit.
- Accepted as proposed: hatchling, typer, ruff, pytest-cov, 85% coverage, six-job CI matrix, pre-commit in-repo, SemVer, Keep-a-Changelog.

### From Section 1 review
- CallableField ReLU detection is best-effort submodule scan, documented explicitly.
- `SymmetrySpec` literal adds `"C4"` distinct from `"D4"`.
- §14.3 security note mentions `GITHUB_TOKEN` minimal permissions.
- §6.4 floor calibration via two-step branch workflow to avoid circularity.
- §15 adds integration-by-parts as a fifth hypothesis property.

### From Section 2 review
- `_STATUS_RANK` dict includes `SKIPPED: 0` explicitly (prevents `KeyError` on first dump-mode run).
- Typo fix: three adapter-only rules (`PH-RES-002`, `PH-SYM-003`, `PH-NUM-003`), not two.
- `PH-RES-003` dump-mode interpretation caveat in rule docs.
- `PH-CON-001` hD-branch tolerance via new `floors.toml` entry with `bc_branch = "hd"`.
- `PH-SYM-004` restricted to periodic-only in V1.
- `PH-NUM-003` SKIP logic in dump mode for consistency.
- `physics_lint/rules/_registry.py` lazy discovery — eager metadata, lazy `check()` import. `rules list` < 50 ms.

### From Section 3 review
- **§14.1 single-invocation workflow + `if: always()` upload** — replaces `continue-on-error` + re-invocation. Prevents nondeterminism race.
- **§13.1 SKIPPED rules → `run.invocations[].toolExecutionNotifications`** — not `run.results`. Prevents Security-tab noise.
- `config show` callable without target for config-only debugging.
- `config init --pde <name>` PDE-specific templates; bare `config init` stays generic.
- Pre-commit smoke check uses hypothesis `ci-quick` profile, not `self-test`.

### Unchanged from Rev. 4.3.1
- §1 mathematical foundations (operator admissibility, analytical solutions, BC notation, norm-equivalence caveats).
- §7 residual formulas and FD-vs-AD cross-check mechanics.
- §8.5 `PH-BC-001` absolute/relative mode branching with binary PASS/FAIL for absolute mode.
- §9.4 per-PDE operator-admissible symmetry table.
- §10.3 severity levels.
- §13 SARIF tier framing (Tier 1 + Tier 2 opt-in + Tier 3 out).
- §19 non-goals (V1 scope discipline).

---

## 22. References

1. Bachmayr, M., Dahmen, W., & Oster, T. (2024). Variationally correct neural residual regression for parametric PDEs. arXiv:2405.20065.
2. Qiu, W., Dahmen, W., & Chen, J. (2025). Variationally correct operator learning: Reduced basis neural operator with a posteriori error estimation. arXiv:2512.21319. (Scope: stationary diffusion and linear elasticity.)
3. Ernst, O. G., Rekatsinas, N., & Urban, K. (2025). A posteriori certification for neural network approximations to PDEs. arXiv:2502.20336v3.
4. Jekel, C. F., Sterbentz, D. M., Aubry, S., et al. (2022). Using conservation laws to infer deep learning model accuracy of Richtmyer-Meshkov instabilities. arXiv:2208.11477.
5. Hansen, D., Maddix, D. C., Alizadeh, S., et al. (2023). Learning physical models that can respect conservation laws. *ICML 2023*.
6. Gruver, N., Finzi, M., Goldblum, M., & Wilson, A. G. (2023). The Lie derivative for measuring learned equivariance. *ICLR 2023*. arXiv:2210.02984.
7. Brandstetter, J., Welling, M., & Worrall, D. E. (2022). Lie point symmetry data augmentation for neural PDE solvers. *ICML 2022*. arXiv:2202.07643.
8. Helwig, J., Zhang, X., Fu, C., et al. (2023). Group equivariant Fourier neural operators for PDEs. *ICML 2023*. arXiv:2306.05697.
9. Li, Z., Zheng, H., Kovachki, N., et al. (2021). Physics-informed neural operator for learning PDEs. arXiv:2111.03794.
10. Gustafsson, T., & McBain, G. D. (2020). scikit-fem: A Python package for finite element assembly. *JOSS*, 5(52), 2369.
11. NVIDIA (2024). PhysicsNeMo Sym. github.com/NVIDIA/physicsnemo-sym.
12. Geiger, M., & Smidt, T. (2022). e3nn: Euclidean neural networks. arXiv:2207.09453.
13. Virkkunen, I., et al. (2021). Phased-array ultrasonic testing reference dataset for deep learning. (Used as stress-test distribution-shift dataset for laplace-uq-bench.)
14. Olver, P. J. (1986). *Applications of Lie Groups to Differential Equations*. Springer GTM 107.
15. Fornberg, B. (1988). Generation of finite difference formulas on arbitrarily spaced grids. *Math. Comp.*, 51(184), 699–706.
16. Trefethen, L. N. (2000). *Spectral Methods in MATLAB*. SIAM.
17. OASIS (2019). Static Analysis Results Interchange Format (SARIF) Version 2.1.0. OASIS Standard, July 2019.
