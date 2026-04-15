"""Run the four dogfood surrogates on in-dist vs OOD BC regimes.

This is a pared-down Task-10 artifact. The plan called for a Virkkunen
2021 stress-test scatter plot (MSE vs H^-1 residual) as the central
marketing figure, but:

1. laplace-uq-bench has no Virkkunen dataset — the upstream repo uses
   piecewise-constant BCs as its OOD stress regime, not an external
   dataset. (The plan's "Virkkunen" reference was a guess flagged for
   Day-5 discovery.)
2. The Task-8 fallback-D' surrogates are synthetic defects on the FD
   oracle; their MSE vs the oracle is by construction the defect
   amplitude. A scatter plot would be degenerate (a near-line), not
   the "MSE misses what physics catches" figure the plan wanted.

What we CAN demonstrate honestly with these four surrogates is that
physics-lint's residual changes with BC regime in ways that depend on
the defect class: smoothing and under-resolution degrade much more on
sharp piecewise BCs than on smooth sinusoidal BCs, while additive
noise is regime-invariant. That table is the Week-2 dogfood marketing
artifact; a proper MSE-vs-physics scatter waits for real trained
surrogates.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from physics_lint import DomainSpec, GridField
from physics_lint.rules import ph_res_001

from . import common, generate_dumps

HERE = Path(__file__).parent


def _canonical_case_for(bc_type: str) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild a (64, 64) oracle field + bc_bundle for a named BC family."""
    common._bootstrap_repo_imports()
    from diffphys.pde.boundary import sample_four_edges
    from diffphys.pde.laplace import LaplaceSolver

    rng = np.random.default_rng(common.SEED if bc_type == "sinusoidal" else common.SEED + 9)
    bc_top, bc_bottom, bc_left, bc_right = sample_four_edges(
        rng, allowed_types=[bc_type], nx=common.NX
    )
    solver = LaplaceSolver(nx=common.NX)
    field = solver.solve(bc_top, bc_bottom, bc_left, bc_right).astype(np.float64)
    bc_bundle = np.stack([bc_top, bc_bottom, bc_left, bc_right]).astype(np.float64)
    return field, bc_bundle


def _build_surrogates(oracle: np.ndarray, bc_bundle: np.ndarray) -> dict[str, np.ndarray]:
    """Build the four dogfood predictions for a given oracle + BC pair."""
    common._bootstrap_repo_imports()
    from diffphys.pde.laplace import LaplaceSolver

    # coarsened: re-resolve the same BC at nx=17, then upsample.
    coarse_nx = 17
    src_x = np.linspace(0.0, 1.0, common.NX)
    tgt_x = np.linspace(0.0, 1.0, coarse_nx)
    bc_top = np.interp(tgt_x, src_x, bc_bundle[0])
    bc_bottom = np.interp(tgt_x, src_x, bc_bundle[1])
    bc_left = np.interp(tgt_x, src_x, bc_bundle[2])
    bc_right = np.interp(tgt_x, src_x, bc_bundle[3])
    coarse_field = LaplaceSolver(nx=coarse_nx).solve(bc_top, bc_bottom, bc_left, bc_right)
    coarsened = generate_dumps._bilinear_upsample(coarse_field, common.NX)

    return {
        "oracle": oracle,
        "noisy": generate_dumps.generate_noisy(oracle),
        "coarsened": coarsened,
        "smoothed": generate_dumps.generate_smoothed(oracle),
    }


def _spec() -> DomainSpec:
    return DomainSpec.model_validate(
        common.dump_metadata()
        | {
            "field": {
                "type": "grid",
                "backend": "fd",
                "dump_path": "inline.npz",
            }
        }
    )


def _residual(field_values: np.ndarray) -> tuple[float, float]:
    field = GridField(
        field_values,
        h=(1.0 / (common.NX - 1), 1.0 / (common.NX - 1)),
        periodic=False,
        backend="fd",
    )
    result = ph_res_001.check(field, _spec())
    return float(result.raw_value or 0.0), float(result.violation_ratio or 0.0)


def main() -> None:
    regimes = {
        "in-dist (sinusoidal)": "sinusoidal",
        "OOD (piecewise)": "piecewise",
    }
    surrogate_names = ["oracle", "noisy", "coarsened", "smoothed"]
    table: dict[str, dict[str, float]] = {name: {} for name in surrogate_names}
    mse_table: dict[str, dict[str, float]] = {name: {} for name in surrogate_names}
    for label, bc_type in regimes.items():
        oracle, bc_bundle = _canonical_case_for(bc_type)
        surrogates = _build_surrogates(oracle, bc_bundle)
        for name, pred in surrogates.items():
            raw, _ = _residual(pred)
            table[name][label] = raw
            mse_table[name][label] = float(np.mean((pred - oracle) ** 2))

    print("physics-lint residual by regime x surrogate")
    header = f"{'surrogate':<12}" + "".join(f"{r:<28}" for r in regimes)
    print(header)
    print("-" * len(header))
    for name in surrogate_names:
        row = f"{name:<12}"
        for label in regimes:
            row += (
                f"{table[name][label]:<14.3e}{'(MSE ' + f'{mse_table[name][label]:.2e}' + ')':<14}"
            )
        print(row)

    out_dir = HERE.parent / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "week2-regime-comparison.md"
    lines = [
        "# Week 2 dogfood — regime comparison",
        "",
        "Four synthetic surrogates on two BC regimes. Shows that",
        "smoothing/coarsening defects degrade much more on the sharp",
        "piecewise OOD regime than on smooth in-dist sinusoidal BCs,",
        "while the noise defect is regime-invariant. MSE vs oracle is",
        "reported alongside the physics-lint residual.",
        "",
        "| surrogate | in-dist residual | in-dist MSE | OOD residual | OOD MSE |",
        "|-----------|------------------|-------------|--------------|---------|",
    ]
    for name in surrogate_names:
        t = table[name]
        m = mse_table[name]
        lines.append(
            f"| {name} | {t['in-dist (sinusoidal)']:.3e} | "
            f"{m['in-dist (sinusoidal)']:.3e} | {t['OOD (piecewise)']:.3e} | "
            f"{m['OOD (piecewise)']:.3e} |"
        )
    lines.append("")
    out.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
