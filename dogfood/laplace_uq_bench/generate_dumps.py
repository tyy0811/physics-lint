"""Generate the four Week-2 dogfood .npz dumps.

Oracle is the laplace-uq-bench FD solver on a canonical BC. The other
three are degraded variants that imitate the kinds of failure modes
physics-lint is supposed to catch:

- noisy: oracle + small additive Gaussian noise. Interior points drift
  off the harmonic manifold; residual grows with the noise amplitude.
- coarsened: solver run on a 17x17 grid then bilinear-upsampled to
  64x64. Under-resolution on the coarse grid leaks into the FD
  residual at 64x64.
- smoothed: oracle convolved with a 5x5 Gaussian. Smoothing bleeds
  boundary conditions into the interior and breaks the harmonic
  identity over several pixels.

We use the repo's own BC sampler + solver so the oracle dump exercises
external-repo code, not a physics-lint analytical battery re-run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import common


def _gaussian_kernel_2d(sigma: float, radius: int) -> np.ndarray:
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    x, y = np.meshgrid(ax, ax, indexing="ij")
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _convolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # Plain numpy 'same' correlation; dogfood does not need scipy here.
    out = np.zeros_like(image, dtype=np.float64)
    r = kernel.shape[0] // 2
    padded = np.pad(image, r, mode="edge")
    for i in range(kernel.shape[0]):
        for j in range(kernel.shape[1]):
            out += kernel[i, j] * padded[i : i + image.shape[0], j : j + image.shape[1]]
    return out


def _bilinear_upsample(image: np.ndarray, target: int) -> np.ndarray:
    src = image.shape[0]
    src_x = np.linspace(0, 1, src)
    tgt_x = np.linspace(0, 1, target)
    # Column-wise interp to (target, src), then row-wise to (target, target)
    col = np.stack([np.interp(tgt_x, src_x, image[:, j]) for j in range(src)], axis=1)
    return np.stack([np.interp(tgt_x, src_x, col[i, :]) for i in range(target)], axis=0)


def generate_oracle() -> np.ndarray:
    field, _ = common.canonical_test_case()
    return field


def generate_noisy(oracle: np.ndarray, *, sigma: float = 5e-3) -> np.ndarray:
    rng = np.random.default_rng(common.SEED + 1)
    noise = rng.normal(0.0, sigma, size=oracle.shape)
    out = oracle + noise
    # Preserve the Dirichlet boundary so PH-BC-002 does not flag the
    # noise on the edge — we're probing the interior residual, not BC
    # fidelity.
    out[0, :] = oracle[0, :]
    out[-1, :] = oracle[-1, :]
    out[:, 0] = oracle[:, 0]
    out[:, -1] = oracle[:, -1]
    return out


def generate_coarsened() -> np.ndarray:
    common._bootstrap_repo_imports()
    from diffphys.pde.laplace import LaplaceSolver

    # Resample the SAME BC (by seed replay) at a coarser resolution, then
    # bilinear-upsample. The coarse solver gets a 17x17 BC by linear
    # interpolation from the 64x64 BC — the fine-grid BC has no stride
    # that lands exactly on 17 nodes.
    _, bc_bundle = common.canonical_test_case()
    coarse_nx = 17
    src_x = np.linspace(0.0, 1.0, common.NX)
    tgt_x = np.linspace(0.0, 1.0, coarse_nx)
    bc_top = np.interp(tgt_x, src_x, bc_bundle[0])
    bc_bottom = np.interp(tgt_x, src_x, bc_bundle[1])
    bc_left = np.interp(tgt_x, src_x, bc_bundle[2])
    bc_right = np.interp(tgt_x, src_x, bc_bundle[3])
    solver_coarse = LaplaceSolver(nx=coarse_nx)
    coarse_field = solver_coarse.solve(bc_top, bc_bottom, bc_left, bc_right)
    return _bilinear_upsample(coarse_field, common.NX)


def generate_smoothed(oracle: np.ndarray, *, sigma: float = 1.5) -> np.ndarray:
    kernel = _gaussian_kernel_2d(sigma=sigma, radius=2)
    smoothed = _convolve_same(oracle, kernel)
    # Restore Dirichlet edge so PH-BC-002 still sees the right trace.
    smoothed[0, :] = oracle[0, :]
    smoothed[-1, :] = oracle[-1, :]
    smoothed[:, 0] = oracle[:, 0]
    smoothed[:, -1] = oracle[:, -1]
    return smoothed


def main() -> None:
    oracle = generate_oracle()
    dumps: dict[str, np.ndarray] = {
        "oracle": oracle,
        "noisy": generate_noisy(oracle),
        "coarsened": generate_coarsened(),
        "smoothed": generate_smoothed(oracle),
    }
    out_dir = Path(common.HERE)
    for name, prediction in dumps.items():
        path = out_dir / f"{name}_pred.npz"
        common.write_dump(path, prediction)
        print(f"wrote {path} ({prediction.shape})")


if __name__ == "__main__":
    main()
