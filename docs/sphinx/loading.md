# Loading models

physics-lint supports two ways to give it your model: an **adapter** (Python
file) or a **dump** (`.npz` file).

## Adapter mode

Write a `physics_lint_adapter.py` next to your model code:

```python
from physics_lint import DomainSpec
import torch

def load_model() -> torch.nn.Module:
    model = build_my_model()
    model.load_state_dict(torch.load("checkpoint.pt"))
    model.eval()
    return model

def domain_spec() -> DomainSpec:
    return DomainSpec.model_validate({
        "pde": "heat",
        "grid_shape": [64, 64, 32],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "t": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": {"kind": "dirichlet_homogeneous"},
        "diffusivity": 0.01,
        "field": {"type": "callable", "backend": "fd", "adapter_path": __file__},
    })
```

Then run:

```bash
physics-lint check physics_lint_adapter.py
```

## Dump mode

For sampler-based models (DDPM, DPS) or models from frameworks other than
PyTorch (JAX, TF), pre-generate a prediction tensor and write it as `.npz`:

```python
import numpy as np
pred = run_my_sampler()          # shape (Nx, Ny[, Nt])
np.savez("pred.npz",
    prediction=pred,
    metadata={
        "pde": "laplace",
        "grid_shape": [64, 64],
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
        "periodic": False,
        "boundary_condition": "dirichlet_homogeneous",
        "field": {"type": "grid", "backend": "fd"},
    },
)
```

Then:

```bash
physics-lint check pred.npz
```

## Which mode is right?

Use **adapter mode** when:

- Your model is a single `torch.nn.Module` and deterministic
- You want the full rule suite (including adapter-only rules like
  `PH-RES-002` FD-vs-AD cross-check and `PH-SYM-003` SO(2) LEE)

Use **dump mode** when:

- Your model is an iterative sampler (DDPM, DPS, score-based generative)
- Your model is in JAX, TensorFlow, or a framework physics-lint can't
  exec directly
- Sampling is expensive and you want to cache results

Adapter-only rules emit `SKIPPED` on dump inputs with an explicit reason —
they are not silently omitted.
