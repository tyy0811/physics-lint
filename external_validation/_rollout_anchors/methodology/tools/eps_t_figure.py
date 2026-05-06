"""Rung 4b figure-subset ε(t) renderer — 6-trace plot.

Reads the 3 figure-subset eps_*.npz files per stack (each with eps_t shape
(100,) instead of the main-sweep shape (1,)) and renders a 6-trace ε vs
rollout-step plot. Used by the rung 4b cross-stack equivariance writeup
(`methodology/docs/2026-05-07-rung-4b-equivariance-table.md`) to make the
single-step-vs-rollout-horizon distinction explicit.

USAGE
-----

    python external_validation/_rollout_anchors/methodology/tools/eps_t_figure.py \\
        --segnn-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories/segnn_tgv2d_<sha>/ \\
        --gns-dir   external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories/gns_tgv2d_<sha>/ \\
        --out-dir   external_validation/_rollout_anchors/01-lagrangebench/outputs/figures/

Writes ``eps_t_traces_<eps_computation_sha>.{png,pdf}`` keyed off the npz
provenance field. Both formats committed; png for inline writeup display,
pdf for vector-graphics print fidelity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Figure-subset trajectory indices (all PH-SYM-001 rotation π/2 per
# build_figure_sweep_transforms; mirrored across stacks). If the design
# changes the figure subset, update here.
_FIGURE_TRAJ_INDICES = (0, 7, 14)

# Threshold band annotations (matches design §3.3 + render_eps_table verdict):
#   PASS         eps <= 1e-5
#   APPROXIMATE  1e-5 < eps <= 1e-2
#   FAIL         eps > 1e-2
_PASS_CEILING = 1e-5
_APPROXIMATE_CEILING = 1e-2

# Stack-specific colors. SEGNN cool-tones (architectural exact); GNS
# warm-tones (data-aug approximate) — same convention as rung-4a writeup.
_STACK_COLORS = {
    "segnn": "#1f77b4",  # blue
    "gns": "#d62728",  # red
}

# Per-traj line styles so the same traj is identifiable across stacks.
_TRAJ_LINESTYLES = {0: "-", 7: "--", 14: ":"}


def _load_eps_t(eps_dir: Path, traj_index: int) -> tuple[np.ndarray, str]:
    """Load eps_t and the eps_computation sha from a figure-subset npz."""
    npz_path = eps_dir / f"eps_PH-SYM-001_rotation_pi_2_traj{traj_index:02d}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Figure-subset npz missing: {npz_path}. Verify eps_dir holds "
            f"the rung-4b figure-sweep outputs (eps_t shape (100,))."
        )
    with np.load(npz_path, allow_pickle=True) as f:
        eps_t = np.asarray(f["eps_t"])
        eps_computation_sha = str(f["physics_lint_sha_eps_computation"])
    if eps_t.shape != (100,):
        raise ValueError(
            f"{npz_path}: expected eps_t shape (100,) (figure sweep), "
            f"got {eps_t.shape}. Confirm the npz is from the figure-sweep, "
            f"not main-sweep."
        )
    return eps_t, eps_computation_sha


def render_eps_t_figure(
    *,
    segnn_dir: Path,
    gns_dir: Path,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Render the 6-trace ε(t) figure to {out_dir}/eps_t_traces_<sha>.{png,pdf}.

    Returns the (png_path, pdf_path) tuple. Fails loud if eps_computation_sha
    differs across the 6 npzs (would indicate a mid-run sha drift).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    traces: dict[tuple[str, int], np.ndarray] = {}
    shas: set[str] = set()
    for stack_name, eps_dir in (("segnn", segnn_dir), ("gns", gns_dir)):
        for traj in _FIGURE_TRAJ_INDICES:
            eps_t, sha = _load_eps_t(eps_dir, traj)
            traces[(stack_name, traj)] = eps_t
            shas.add(sha)

    if len(shas) != 1:
        raise ValueError(
            f"Inconsistent eps_computation_sha across figure-subset npzs: {shas}. "
            f"All 6 traces must come from a single Modal sweep."
        )
    eps_sha = next(iter(shas))[:10]

    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=150)

    # Band shading: PASS (below 1e-5) + APPROXIMATE (1e-5 to 1e-2).
    # FAIL is unbounded above; visualized as the un-shaded region.
    ax.axhspan(1e-9, _PASS_CEILING, alpha=0.10, color="#2ca02c", label="_PASS band")
    ax.axhspan(
        _PASS_CEILING, _APPROXIMATE_CEILING, alpha=0.08, color="#ff7f0e", label="_APPROX band"
    )
    ax.axhline(_PASS_CEILING, color="#2ca02c", linewidth=0.6, linestyle="-", alpha=0.5)
    ax.axhline(_APPROXIMATE_CEILING, color="#ff7f0e", linewidth=0.6, linestyle="-", alpha=0.5)

    # Six traces.
    rollout_steps = np.arange(100)  # t = 0..99
    for stack_name in ("segnn", "gns"):
        for traj in _FIGURE_TRAJ_INDICES:
            eps_t = traces[(stack_name, traj)]
            ax.plot(
                rollout_steps,
                eps_t,
                color=_STACK_COLORS[stack_name],
                linestyle=_TRAJ_LINESTYLES[traj],
                linewidth=1.4,
                alpha=0.85,
                label=f"{stack_name.upper()} traj{traj:02d}",
            )

    # Band labels at right edge.
    ax.text(99.5, 5e-7, "PASS", fontsize=8, color="#2ca02c", va="center", ha="right")
    ax.text(99.5, 5e-4, "APPROXIMATE", fontsize=8, color="#ff7f0e", va="center", ha="right")
    ax.text(99.5, 5e-1, "FAIL", fontsize=8, color="#888888", va="center", ha="right")

    ax.set_yscale("log")
    ax.set_ylim(1e-9, 1.0)
    ax.set_xlim(0, 99)
    ax.set_xlabel("rollout step  $t$")
    ax.set_ylabel(r"$\epsilon(t)$  (RMS position deviation)")
    ax.set_title(
        "Rung 4b — PH-SYM-001 rotation π/2, figure-subset ε(t)\n"
        "SEGNN (blue) vs GNS (red), 3 trajs each (line style by traj index)"
    )
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, ncol=2, framealpha=0.9)

    fig.tight_layout()

    png_path = out_dir / f"eps_t_traces_{eps_sha}.png"
    pdf_path = out_dir / f"eps_t_traces_{eps_sha}.pdf"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)

    return png_path, pdf_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render rung 4b figure-subset ε(t) plot.")
    parser.add_argument("--segnn-dir", type=Path, required=True)
    parser.add_argument("--gns-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    png_path, pdf_path = render_eps_t_figure(
        segnn_dir=args.segnn_dir,
        gns_dir=args.gns_dir,
        out_dir=args.out_dir,
    )
    print(f"PNG: {png_path}")
    print(f"PDF: {pdf_path}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
