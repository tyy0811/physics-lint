"""Shared symmetry harness for Tier-A and Tier-B equivariance anchors.

Primitives:
    rotate_test(model, x, k)      - 90 deg * k rotation equivariance test (C4 subgroup of SO(2))
    reflect_test(model, x, axis)  - axis reflection equivariance test (Z2 subgroup of O(2))
    fft_laplace_inverse           - provably C4- and reflection-equivariant operator
    non_equivariant_cnn           - random-weight CNN with positional embeddings

    Task 7 (PH-SYM-004 translation equivariance) additions:
    shift_commutation_error(model, x, shifts, dims) - grid-aligned shift commutation error
    identity_op                   - trivially equivariant (error = 0)
    circular_convolution_1d       - periodic-kernel 1D convolution via FFT
    circular_convolution_2d       - periodic-kernel 2D convolution via FFT
    fourier_multiplier_1d         - Fourier-domain multiplier m(k); equivariant by convolution theorem
    fourier_multiplier_2d         - Fourier-domain multiplier m(kx, ky); equivariant
    coord_dependent_multiply_1d   - non-equivariant negative control: u(x) -> w(x) u(x)
    coord_dependent_multiply_2d   - non-equivariant negative control: 2D

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

    Tasks 6 (PH-SYM-003) and 7 (PH-SYM-004) extend this harness with
    Lie-group-specific utilities beyond the discrete C4 and Z2 cases:
    continuous SO(2) equivariance for Task 6, translation equivariance
    (R^2 / Z^d) for Task 7. Task 7's additions (grid-aligned shift
    commutation on periodic domain) follow the CRITICAL three-layer
    pattern per 2026-04-24 user-revised contract: F1 Kondor-Trivedi 2018
    compact-group theorem + Li et al. 2021 FNO convolution theorem;
    F2 harness-authoritative on controlled operators (identity, circular
    convolution, Fourier multiplier) plus a coordinate-dependent-
    multiplication negative control; rule-verdict contract verifies the
    V1 stub SKIPs with the documented reasons (ph_sym_004.py:36-52).

    Task 6 (PH-SYM-003) adds continuous SO(2) Lie-derivative primitives:
        so2_lie_derivative(model, grid)          - jvp infinitesimal generator
        radial_scalar(phi)                        - positive control (L_A f = 0)
        coord_dependent_scalar_2d                 - negative control
        anisotropic_xx_minus_yy_2d                - negative control
        finite_small_angle_defect(model, grid, eps) - Case C finite-vs-infinitesimal

    The so2_lie_derivative primitive is the harness-authoritative version of
    the same jvp computation ph_sym_003.check() performs; F2 validates the
    primitive against closed-form analytical answers, then the rule-verdict
    contract in PH-SYM-003/test_anchor.py wraps the harness primitive's
    positive/negative controls in CallableField and asserts the rule emits
    the expected PASS/WARN/FAIL verdict.
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


# ---------------------------------------------------------------------------
# Task 7 (PH-SYM-004 translation equivariance) primitives.
# ---------------------------------------------------------------------------
#
# Action of the translation group on a periodic grid of shape (..., N) or
# (..., Nx, Ny): T_s u := torch.roll(u, shifts=s, dims=(-1,)) (1D) or
# T_s u := torch.roll(u, shifts=(sx, sy), dims=(-2, -1)) (2D). Integer s
# is "grid-aligned." Continuous / sub-grid shifts are out of V1 scope
# (would require interpolation).
#
# Equivariance definition: K commutes with T_s iff K(T_s u) = T_s K(u) for
# all valid u and s. Verified via shift_commutation_error below.


def shift_commutation_error(
    model: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    shifts,
    dims,
) -> float:
    """Return relative L2 error between model(T_s x) and T_s model(x).

    For a translation-equivariant operator K, model(torch.roll(x, s)) must
    equal torch.roll(model(x), s). The quantity returned is
    ``||model(T_s x) - T_s model(x)||_2 / ||model(T_s x)||_2``, bounded
    below by float-precision roundoff for true equivariance, and growing
    to O(1) for operators that are not translation-equivariant.
    """
    y_shift_then_model = model(torch.roll(x, shifts=shifts, dims=dims))
    y_model_then_shift = torch.roll(model(x), shifts=shifts, dims=dims)
    return _rel_l2(y_shift_then_model, y_model_then_shift)


def identity_op(x: torch.Tensor) -> torch.Tensor:
    """Trivially translation-equivariant operator; error is exactly 0."""
    return x


def circular_convolution_1d(
    kernel: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """1D circular convolution via FFT. Translation-equivariant by the
    convolution theorem: convolution with a fixed kernel commutes with
    shift on a periodic domain.

    `kernel` is zero-padded to the signal length and centered at the
    origin (FFT-convention index 0) via a half-kernel shift.
    """

    def op(x: torch.Tensor) -> torch.Tensor:
        n = x.shape[-1]
        k = kernel.shape[-1]
        padded = torch.zeros(n, dtype=x.dtype)
        padded[:k] = kernel.to(x.dtype)
        padded = torch.roll(padded, -(k // 2))
        xhat = torch.fft.fft(x)
        khat = torch.fft.fft(padded)
        return torch.fft.ifft(xhat * khat).real

    return op


def circular_convolution_2d(
    kernel: torch.Tensor,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """2D circular convolution via FFT. Translation-equivariant by the
    convolution theorem on the periodic 2-torus.
    """

    def op(x: torch.Tensor) -> torch.Tensor:
        nx, ny = x.shape[-2], x.shape[-1]
        kx, ky = kernel.shape[-2], kernel.shape[-1]
        padded = torch.zeros(nx, ny, dtype=x.dtype)
        padded[:kx, :ky] = kernel.to(x.dtype)
        padded = torch.roll(padded, shifts=(-(kx // 2), -(ky // 2)), dims=(-2, -1))
        xhat = torch.fft.fftn(x, dim=(-2, -1))
        khat = torch.fft.fftn(padded, dim=(-2, -1))
        return torch.fft.ifftn(xhat * khat, dim=(-2, -1)).real

    return op


def fourier_multiplier_1d(n: int, *, seed: int = 0) -> Callable[[torch.Tensor], torch.Tensor]:
    """1D Fourier-domain multiplier m(k) with real even symmetry.

    Translation-equivariant by the convolution theorem: multiplication in
    Fourier space corresponds to circular convolution in real space, which
    commutes with shifts on a periodic domain. Even symmetry in `k` plus
    real multiplier ensures the output is real-valued for real input.
    """
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(n, generator=g, dtype=torch.float64)
    # Enforce m(k) = m(-k) so real input -> real output.
    for k in range(1, n // 2 + (0 if n % 2 == 0 else 1)):
        m[n - k] = m[k]

    def op(x: torch.Tensor) -> torch.Tensor:
        xhat = torch.fft.fft(x)
        return torch.fft.ifft(xhat * m.to(x.dtype)).real

    return op


def fourier_multiplier_2d(
    nx: int, ny: int, *, seed: int = 0
) -> Callable[[torch.Tensor], torch.Tensor]:
    """2D Fourier-domain multiplier m(kx, ky), symmetrized for real output.

    Translation-equivariant by the convolution theorem on the 2-torus.
    Symmetrization m = 0.5 * (m + flip(m, (-2, -1))) gives m(-kx, -ky) =
    m(kx, ky), which pairs with real input to produce real output.
    """
    g = torch.Generator().manual_seed(seed)
    m = torch.randn(nx, ny, generator=g, dtype=torch.float64)
    m_flipped = torch.flip(m, dims=(-2, -1)).clone()
    m = 0.5 * (m + m_flipped)

    def op(x: torch.Tensor) -> torch.Tensor:
        xhat = torch.fft.fftn(x, dim=(-2, -1))
        return torch.fft.ifftn(xhat * m.to(x.dtype), dim=(-2, -1)).real

    return op


def coord_dependent_multiply_1d(
    n: int, center: float = 0.5, sigma: float = math.sqrt(0.1)
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Non-equivariant negative control. Multiplies input by a position-
    dependent Gaussian mask w(x) = exp(-(x - center)^2 / sigma^2) on
    x in [0, 1]. Breaks translation equivariance because the mask is
    fixed in space while the shift moves the signal under it.
    """
    x_coord = torch.linspace(0, 1, n, dtype=torch.float64)
    w = torch.exp(-((x_coord - center) ** 2) / (sigma**2))

    def op(x: torch.Tensor) -> torch.Tensor:
        return w.to(x.dtype) * x

    return op


def coord_dependent_multiply_2d(
    nx: int,
    ny: int,
    center: tuple[float, float] = (0.5, 0.5),
    sigma: float = math.sqrt(0.1),
) -> Callable[[torch.Tensor], torch.Tensor]:
    """2D non-equivariant negative control. w(x, y) = exp(-((x-cx)^2 +
    (y-cy)^2) / sigma^2).
    """
    xg = torch.linspace(0, 1, nx, dtype=torch.float64)
    yg = torch.linspace(0, 1, ny, dtype=torch.float64)
    grid_x, grid_y = torch.meshgrid(xg, yg, indexing="ij")
    cx, cy = center
    w = torch.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (sigma**2))

    def op(x: torch.Tensor) -> torch.Tensor:
        return w.to(x.dtype) * x

    return op


# ---------------------------------------------------------------------------
# Task 6 (PH-SYM-003) SO(2) Lie-derivative primitives.
# ---------------------------------------------------------------------------
#
# Single-generator Lie derivative L_A f at theta = 0 along the one-parameter
# subgroup R_theta = exp(theta A) of SO(2), A = [[0,-1],[1,0]]. The generator
# J (imaginary unit in the complex representation) acts on coordinates as
# A (x, y) = (-y, x). Rule PH-SYM-003 consumes the same jvp-based computation;
# the harness exposes it directly so F2 tests the implemented quantity against
# closed-form analytical expressions.
#
# Grid convention: grid tensor of shape (..., 2) with last dim = (x, y);
# origin-centered square domain in [-L/2, L/2] x [-L/2, L/2]. This matches
# ph_sym_003.py's grid contract (line 54-68 gates).


def _origin_centered_square_grid(n: int, half_extent: float = 1.0) -> torch.Tensor:
    """Return an (n, n, 2) origin-centered square grid on [-H, H]^2.

    Used by SO(2) fixtures to satisfy ph_sym_003.py's origin-centered +
    square-domain gates. float64 throughout for jvp numerical accuracy.
    """
    coord = torch.linspace(-half_extent, half_extent, n, dtype=torch.float64)
    gx, gy = torch.meshgrid(coord, coord, indexing="ij")
    return torch.stack([gx, gy], dim=-1)


def _rotate_grid(grid: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Apply SO(2) rotation R_theta to grid points about the origin."""
    c = torch.cos(theta)
    s = torch.sin(theta)
    x = grid[..., 0]
    y = grid[..., 1]
    x_rot = c * x - s * y
    y_rot = s * x + c * y
    return torch.stack([x_rot, y_rot], dim=-1)


def so2_lie_derivative(
    model: Callable[[torch.Tensor], torch.Tensor],
    grid: torch.Tensor,
) -> torch.Tensor:
    """Return the pointwise Lie derivative L_A f on grid, as a tensor.

    Computes d/dtheta |_{theta=0} model(R_theta grid) via forward-mode AD
    (torch.autograd.functional.jvp). This is the same primitive PH-SYM-003's
    check() uses internally (ph_sym_003.py:70-87); exposed here so F2 can
    verify the primitive against closed-form analytical expressions
    independently of the rule's tolerance machinery.

    Shape of return: same as model(grid). For scalar-output models the last
    dim is squeezed by model itself.
    """
    from torch.autograd.functional import jvp

    def rotated(theta_param: torch.Tensor) -> torch.Tensor:
        return model(_rotate_grid(grid, theta_param))

    theta0 = torch.zeros(1, dtype=torch.float64)
    tangent = torch.ones_like(theta0)
    _, lie = jvp(rotated, (theta0,), v=(tangent,))
    return lie


def so2_lie_derivative_norm(
    model: Callable[[torch.Tensor], torch.Tensor],
    grid: torch.Tensor,
) -> float:
    """Per-point L2 norm of the Lie derivative (matches ph_sym_003.py:89)."""
    lie = so2_lie_derivative(model, grid)
    norm = float(torch.linalg.vector_norm(lie))
    denom = max(float(lie.numel()), 1.0) ** 0.5
    return norm / denom


def radial_scalar(
    phi: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[torch.Tensor], torch.Tensor]:
    """SO(2)-invariant positive control: f(x, y) = phi(r), r = sqrt(x^2 + y^2).

    Radial scalar maps are invariant under SO(2) by construction because
    |R_theta (x, y)| = |(x, y)|. L_A f = phi'(r) * d/dtheta |_{theta=0} r(R_theta x)
    and r is rotation-invariant, so L_A f = 0 exactly (up to jvp roundoff).
    """

    def op(coord: torch.Tensor) -> torch.Tensor:
        x = coord[..., 0]
        y = coord[..., 1]
        r = torch.sqrt(x * x + y * y)
        return phi(r)

    return op


def identity_scalar_2d(coord: torch.Tensor) -> torch.Tensor:
    """Scalar-valued constant-one map. Trivially SO(2)-invariant; L_A f = 0."""
    return torch.ones_like(coord[..., 0])


def coord_dependent_scalar_2d(coord: torch.Tensor) -> torch.Tensor:
    """Negative control: f(x, y) = x. Closed-form L_A f = -y.

    Derivation: L_A f = grad(f) . (A (x, y)) = (1, 0) . (-y, x) = -y. So
    ||L_A f||_{per-point L2} equals the per-point L2 norm of -y over the grid.
    """
    return coord[..., 0]


def anisotropic_xx_minus_yy_2d(coord: torch.Tensor) -> torch.Tensor:
    """Negative control: f(x, y) = x^2 - y^2. Closed-form L_A f = -4 x y.

    Derivation: grad(f) = (2x, -2y); A (x, y) = (-y, x); dot product is
    2x * (-y) + (-2y) * x = -4 x y.
    """
    x = coord[..., 0]
    y = coord[..., 1]
    return x * x - y * y


def finite_small_angle_defect(
    model: Callable[[torch.Tensor], torch.Tensor],
    grid: torch.Tensor,
    epsilon: float,
) -> float:
    """Case C finite-vs-infinitesimal consistency primitive.

    Returns ||f(R_eps x) - f(x) - eps * L_A f(x)||_2 / ||eps * L_A f(x)||_2.
    For smooth non-equivariant f, this is the Taylor remainder ratio and
    must scale as O(epsilon) as epsilon -> 0.
    """
    theta = torch.tensor([epsilon], dtype=torch.float64)
    f0 = model(grid)
    f_eps = model(_rotate_grid(grid, theta.squeeze()))
    lie = so2_lie_derivative(model, grid)
    linear_predict = f0 + epsilon * lie
    residual = f_eps - linear_predict
    num = float(torch.linalg.vector_norm(residual))
    denom = float(torch.linalg.vector_norm(epsilon * lie))
    if denom == 0.0:
        return float("inf") if num > 0 else 0.0
    return num / denom
