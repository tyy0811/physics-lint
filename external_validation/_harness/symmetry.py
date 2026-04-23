"""Shared symmetry harness for Tier-A and Tier-B equivariance anchors.

Primitives:
    rotate_test(model, x, k)      - 90 deg * k rotation equivariance test (C4 subgroup of SO(2))
    reflect_test(model, x, axis)  - axis reflection equivariance test (Z2 subgroup of O(2))
    fft_laplace_inverse           - provably C4- and reflection-equivariant operator
    non_equivariant_cnn           - random-weight CNN with positional embeddings

Zero-mode convention for the FFT Laplace inverse: the Laplacian's kernel on
a periodic square grid is the constant (k = 0) mode, so (-Laplacian)^-1 is
undefined there. We set u_hat(k = 0) = 0 on the inverse output; this makes
the operator fully defined and the equivariance claim operationally complete.
Task 11 in Tier B inherits this convention - DO NOT change the zero-mode
policy without updating Task 11's anchor correspondingly.

Structural-equivalence retrofit note (complete-v1.0 plan Task 1, 2026-04-23):
    rotate_test and reflect_test are the numerical primitives consumed by the
    structural-equivalence proof-sketches embedded in PH-SYM-001/CITATION.md
    (C4 discrete rotation on a 2D periodic square grid) and
    PH-SYM-002/CITATION.md (Z2 discrete reflection). The proof-sketches map
    the rule's emitted equivariance-error quantity to Hall 2015 Lie Groups,
    Lie Algebras, and Representations section 2.5 one-parameter subgroup
    family (section-level) + section 3.7 continuous-to-smooth for matrix
    Lie group homomorphisms (section-level), together with Varadarajan
    1984 section 2.9-2.10 identity-component generation (section-level).
    Hall and Varadarajan citations inherit the WARN-flagged section-level
    framing per external_validation/_harness/TEXTBOOK_AVAILABILITY.md.

    Tasks 6 (PH-SYM-003) and 7 (PH-SYM-004) will extend this harness with
    Lie-group-specific utilities beyond the discrete C4 and Z2 cases:
    continuous SO(2) equivariance for Task 6, translation equivariance
    (R^2 / Z^d) for Task 7. Those extensions land under their respective
    task commits.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = torch.linalg.vector_norm(a).clamp_min(1e-30)
    return float(torch.linalg.vector_norm(a - b) / denom)


def rotate_test(model: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, k: int) -> float:
    """Return relative L2 error between model(rot90(x, k)) and rot90(model(x), k).

    Precondition: x is a 2D or batched-2D tensor with a square last two dims.
    (Non-square C4 would require bilinear interpolation; out of scope here.)
    """
    if x.ndim < 2 or x.shape[-1] != x.shape[-2]:
        raise ValueError(f"rotate_test requires a square 2D tensor; got shape {x.shape}")
    y_rot_then_model = model(torch.rot90(x, k=k, dims=(-2, -1)))
    y_model_then_rot = torch.rot90(model(x), k=k, dims=(-2, -1))
    return _rel_l2(y_rot_then_model, y_model_then_rot)


def reflect_test(
    model: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor, axis: int
) -> float:
    """Return relative L2 error between model(flip(x, axis)) and flip(model(x), axis).

    Precondition: x is a 2D or batched-2D tensor; axis in {-1, -2}.
    """
    if axis not in (-1, -2):
        raise ValueError(f"reflect_test axis must be -1 or -2; got {axis}")
    y_flip_then_model = model(torch.flip(x, dims=(axis,)))
    y_model_then_flip = torch.flip(model(x), dims=(axis,))
    return _rel_l2(y_flip_then_model, y_model_then_flip)


def fft_laplace_inverse(x: torch.Tensor) -> torch.Tensor:
    """Apply (-Laplacian)^-1 on a periodic square grid via FFT.

    Zero-mode convention: the output's k = 0 mode is set to zero. The
    Laplacian's kernel is the constant mode; without a stated convention,
    (-Laplacian)^-1 is undefined there. u_hat(k=0) = 0 makes the operator
    fully defined and is required for the "provably C4-equivariant" claim.
    """
    x = x.to(torch.float64)
    xhat = torch.fft.fftn(x, dim=(-2, -1))
    n1, n2 = x.shape[-2], x.shape[-1]
    kx = torch.fft.fftfreq(n1, d=1.0 / n1) * 2 * math.pi
    ky = torch.fft.fftfreq(n2, d=1.0 / n2) * 2 * math.pi
    kxx, kyy = torch.meshgrid(kx, ky, indexing="ij")
    k2 = kxx**2 + kyy**2
    safe = torch.where(k2 > 0, k2, torch.ones_like(k2))
    yhat = xhat / safe
    yhat[..., k2 == 0] = 0.0
    return torch.fft.ifftn(yhat, dim=(-2, -1)).real.to(torch.float32)


class _NonEquivariantCNN(nn.Module):
    """Random-weight CNN with learned positional embeddings - non-equivariant by design."""

    def __init__(self, n: int = 64) -> None:
        super().__init__()
        self.pos = nn.Parameter(torch.randn(1, n, n))
        self.conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        x = x + self.pos.to(x.dtype)
        x = self.conv(x)
        return x.squeeze(0).squeeze(0) if squeeze else x


def non_equivariant_cnn() -> _NonEquivariantCNN:
    """Fresh random-weight non-equivariant CNN for negative controls."""
    return _NonEquivariantCNN()
