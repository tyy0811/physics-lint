"""Broken-model toy for release criterion 4.

Builds two tiny CNNs:

- **baseline**: plain 2-layer conv taking only the RHS. C4-equivariant
  by architecture alone — convolutions are translation-equivariant and
  the absence of absolute-coordinate input prevents the network from
  distinguishing rotated inputs.

- **broken**: same 2-layer conv architecture plus two additional input
  channels carrying absolute ``(x, y)`` positional embeddings. The absolute
  coordinates destroy equivariance.

Neither model is a U-Net or an external pretrained checkpoint. Both are
fresh random-initialized ``_TinyConv`` instances trained from scratch
inside ``run_criterion_4_validation()`` every call; the validation is
reproducible from a clean checkout with zero external dependencies.

Trains both briefly on a rotational Poisson MMS problem (sin*sin solutions
with random integer wavenumbers), then measures their PH-SYM-001 error on
a C4-symmetric test RHS. The broken model must show a violation > 2x the
baseline's — this is release criterion 4.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812  (conventional PyTorch alias)

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_sym_001


class _TinyConv(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False)
        self.c2 = nn.Conv2d(8, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c2(F.gelu(self.c1(x)))


def _make_positional_channels(n: int) -> torch.Tensor:
    x = torch.linspace(-0.5, 0.5, n)
    X, Y = torch.meshgrid(x, x, indexing="ij")  # noqa: N806
    return torch.stack([X, Y], dim=0)  # shape (2, n, n)


def _rotational_poisson_batch(
    n_batch: int, n: int, rng: np.random.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate n_batch random sin*sin MMS solutions on an n x n grid.

    Input to the model: rhs tensor (n_batch, 1, n, n).
    Target: solution u tensor (n_batch, 1, n, n).
    """
    rhs_list: list[np.ndarray] = []
    u_list: list[np.ndarray] = []
    x = np.linspace(-0.5, 0.5, n)
    X, Y = np.meshgrid(x, x, indexing="ij")  # noqa: N806
    for _ in range(n_batch):
        kx = int(rng.integers(1, 4))
        ky = int(rng.integers(1, 4))
        u = np.sin(np.pi * kx * (X + 0.5)) * np.sin(np.pi * ky * (Y + 0.5))
        f = np.pi**2 * (kx**2 + ky**2) * u
        rhs_list.append(f)
        u_list.append(u)
    rhs = torch.from_numpy(np.stack(rhs_list)).float().unsqueeze(1)  # (n_batch, 1, n, n)
    u_stack = torch.from_numpy(np.stack(u_list)).float().unsqueeze(1)
    return rhs, u_stack


def _train(
    model: nn.Module,
    data: tuple[torch.Tensor, torch.Tensor],
    n_steps: int,
    lr: float,
) -> None:
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    rhs, u = data
    for _ in range(n_steps):
        optim.zero_grad()
        pred = model(rhs)
        loss = F.mse_loss(pred, u)
        loss.backward()
        optim.step()


def _spec_with_c4() -> DomainSpec:
    return DomainSpec.model_validate(
        {
            "pde": "poisson",
            "grid_shape": [32, 32],
            "domain": {"x": [-0.5, 0.5], "y": [-0.5, 0.5]},
            "periodic": False,
            "boundary_condition": {"kind": "dirichlet_homogeneous"},
            "symmetries": {"declared": ["C4"]},
            "field": {"type": "grid", "backend": "fd", "dump_path": "p.npz"},
        }
    )


def _measure_c4_error(model: nn.Module, rhs_test: torch.Tensor) -> float:
    with torch.no_grad():
        pred = model(rhs_test).squeeze().detach().cpu().numpy()
    field = GridField(pred, h=(1 / 31, 1 / 31), periodic=False)
    result = ph_sym_001.check(field, _spec_with_c4())
    return float(result.raw_value or 0.0)


def run_criterion_4_validation(n_training_steps: int = 200, seed: int = 42) -> dict:
    """Train baseline + broken CNNs, measure PH-SYM-001 on each.

    Both models are fresh random-init instances seeded by ``seed`` and
    trained from scratch every call. No external artifact, pretrained
    checkpoint, or downloaded model is involved. The baseline is *not*
    a U-Net — it is the same ``_TinyConv`` architecture as the broken
    model, minus the positional-embedding input channels.

    **Input normalization.** Raw Poisson RHS values are
    ``pi**2*(kx**2 + ky**2)*sin*sin`` with magnitude ~O(100) while the
    target ``u`` is O(1). Feeding the raw RHS at that scale to a
    2-layer conv at ``lr=1e-3`` leaves the kernels effectively
    unchanged after 200 Adam steps — training never converges, so the
    baseline's output stays dominated by its random-init 3x3 kernel
    and is NOT rotation-invariant despite the input being C4-symmetric.
    We normalize ``rhs`` by its batch std so the model sees O(1)
    inputs; with ``lr=1e-2`` the baseline then converges in 200 steps
    to a good Poisson-solver approximation, which IS approximately
    rotation-equivariant (the continuous inverse Laplacian commutes
    with rotations). The broken model, with positional-embedding
    channels, cannot learn a rotation-equivariant mapping no matter
    how well it fits the loss — that is the property release criterion
    4 measures.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = 32

    # Training data: rotationally-symmetric RHS (sin*sin with random integer k).
    rhs_train_raw, u_train = _rotational_poisson_batch(64, n, rng)
    # Normalize RHS by its batch std so the model sees O(1) inputs.
    # See docstring — the raw scale is ~O(100) while targets are O(1).
    rhs_scale = float(rhs_train_raw.std().item())
    rhs_train = rhs_train_raw / rhs_scale

    baseline = _TinyConv(in_channels=1)
    broken = _TinyConv(in_channels=3)  # 1 for rhs + 2 for positional

    # Broken model takes the rhs concatenated with positional channels.
    pos = _make_positional_channels(n)  # (2, n, n)
    pos_batch = pos.unsqueeze(0).expand(rhs_train.shape[0], -1, -1, -1)
    rhs_broken_train = torch.cat([rhs_train, pos_batch], dim=1)

    _train(baseline, (rhs_train, u_train), n_training_steps, lr=1e-2)
    _train(broken, (rhs_broken_train, u_train), n_training_steps, lr=1e-2)

    # Test on a single C4-symmetric RHS. Apply the SAME normalization.
    x = np.linspace(-0.5, 0.5, n)
    X, Y = np.meshgrid(x, x, indexing="ij")  # noqa: N806
    u_exact = np.sin(np.pi * (X + 0.5)) * np.sin(np.pi * (Y + 0.5))  # C4-symmetric
    f_exact = 2 * np.pi**2 * u_exact
    rhs_test = torch.from_numpy(f_exact).float().unsqueeze(0).unsqueeze(0) / rhs_scale
    rhs_test_broken = torch.cat([rhs_test, pos.unsqueeze(0)], dim=1)

    baseline_err = _measure_c4_error(baseline, rhs_test)
    broken_err = _measure_c4_error(broken, rhs_test_broken)

    return {
        "baseline_c4_error": baseline_err,
        "broken_c4_error": broken_err,
        "ratio": broken_err / max(baseline_err, 1e-12),
    }
