"""Convert Week 2½ extraction output into CLI-loader-compatible CI dumps.

`_extract_predictions.py` writes per-model .npz files in the Week 2½ format:
two stacked arrays `predictions` (N, 64, 64) and `truth` (N, 64, 64). That
format is consumed by `run_dogfood_real.py` for the 3-axis cross-comparison,
but the CLI loader at `src/physics_lint/loader.py` expects a single
`prediction` array + `metadata` dict (+ optional `boundary_target`).

This script bridges the two: it picks one sample from the extraction output
and writes a CLI-loader-compatible dump at
`dogfood/laplace_uq_bench/<model>_pred.npz` so that the Task 4 GitHub
Actions workflow can `physics-lint check <dump>` against it.

Usage:

    python -m dogfood.make_ci_dumps \\
        --extraction-dir dogfood/_predictions \\
        --output-dir dogfood/laplace_uq_bench \\
        --models unet_regressor,fno,ddpm \\
        --sample-index 0

Defaults: sample-index 0 (the first sample), models = all three Week 2½
surrogates, output dir = dogfood/laplace_uq_bench.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Problem setup for laplace-uq-bench (64x64 Laplace, unit domain, Dirichlet
# BC — the BC values are shipped per-sample alongside the prediction).
_PROBLEM_METADATA: dict = {
    "pde": "laplace",
    "grid_shape": [64, 64],
    "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0]},
    "periodic": False,
    "boundary_condition": "dirichlet",
    "field": {"type": "grid", "backend": "fd"},
}


def _extract_boundary_values(field_2d: np.ndarray) -> np.ndarray:
    """Return boundary values in the same ordering as GridField.values_on_boundary().

    Mirrors the 2D branch of src/physics_lint/field/grid.py: concatenate
    (left=u[0, :], right=u[-1, :], bottom=u[1:-1, 0], top=u[1:-1, -1]).
    The "left" and "right" rows include their corner entries; "bottom" and
    "top" are the interior columns, so corners are not double-counted.
    """
    left = field_2d[0, :]
    right = field_2d[-1, :]
    bottom = field_2d[1:-1, 0]
    top = field_2d[1:-1, -1]
    return np.concatenate([left, right, bottom, top])


def convert_model(
    extraction_path: Path,
    output_path: Path,
    sample_index: int = 0,
) -> None:
    """Read one sample from extraction output, write CLI-compatible dump."""
    if not extraction_path.is_file():
        raise FileNotFoundError(f"extraction file not found: {extraction_path}")
    data = np.load(extraction_path)
    if "predictions" not in data.files:
        raise KeyError(
            f"{extraction_path}: expected Week 2½ extraction format with "
            f"'predictions' key; got keys {list(data.files)}"
        )
    predictions = data["predictions"]
    if predictions.ndim != 3 or predictions.shape[1:] != (64, 64):
        raise ValueError(
            f"{extraction_path}: predictions shape {predictions.shape} "
            "does not match expected (N, 64, 64)"
        )
    if sample_index >= predictions.shape[0]:
        raise IndexError(
            f"{extraction_path}: sample_index {sample_index} out of range "
            f"(N={predictions.shape[0]})"
        )

    prediction_2d = predictions[sample_index].astype(np.float32)

    # The truth array (if present) gives us the real Dirichlet BC for this
    # problem. physics-lint's PH-BC-001 can then measure |pred_boundary
    # - truth_boundary| and FAIL if the model violates the BC.
    boundary_target: np.ndarray | None = None
    if "truth" in data.files:
        truth_2d = data["truth"][sample_index]
        boundary_target = _extract_boundary_values(truth_2d).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict = {
        "prediction": prediction_2d,
        "metadata": _PROBLEM_METADATA,
    }
    if boundary_target is not None:
        save_kwargs["boundary_target"] = boundary_target

    np.savez_compressed(output_path, **save_kwargs)

    has_bt = "with boundary_target" if boundary_target is not None else "NO boundary_target"
    print(
        f"{extraction_path.name} → {output_path.name} (sample {sample_index}, {has_bt})",
        file=sys.stderr,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extraction-dir",
        type=Path,
        default=Path("dogfood/_predictions"),
        help="Directory containing <model>.npz files from _extract_predictions.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dogfood/laplace_uq_bench"),
        help="Directory to write CLI-compatible <model>_pred.npz dumps",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="unet_regressor,fno,ddpm",
        help="Comma-separated list of model names to convert",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Which sample from the extraction to pick (0 = first)",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for model in models:
        extraction_path = args.extraction_dir / f"{model}.npz"
        output_path = args.output_dir / f"{model}_pred_v1.npz"
        convert_model(extraction_path, output_path, sample_index=args.sample_index)
    return 0


if __name__ == "__main__":
    sys.exit(main())
