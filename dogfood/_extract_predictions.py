"""Subprocess target: load a diffusion-physics checkpoint, run inference on
the first N test problems, dump predictions + truth to .npz.

Runs in .venv-diffphys. Invoked by dogfood/run_dogfood_real.py via
subprocess.run([DIFFPHYS_PYTHON, "dogfood/_extract_predictions.py", ...]).

Outputs an .npz with keys:
    predictions: (N, 64, 64) float32
    truth:       (N, 64, 64) float32

Usage:
    python dogfood/_extract_predictions.py \\
        --model-name unet_regressor \\
        --config $DIFFPHYS_ROOT/configs/unet_regressor.yaml \\
        --checkpoint $DIFFPHYS_ROOT/experiments/unet_regressor/best.pt \\
        --test-npz $DIFFPHYS_ROOT/data/test_in.npz \\
        --max-samples 300 \\
        --output /tmp/unet_regressor_preds.npz
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        required=True,
        choices=["unet_regressor", "fno", "ddpm"],
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-npz", required=True)
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="DDPM only: samples per input for mean",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Imports deferred so --help works even if diffphys is not importable.
    from diffphys.data.dataset import LaplacePDEDataset
    from diffphys.evaluation.evaluate_uq import collect_generative_predictions
    from diffphys.model.trainer import _build_ddpm, build_model, load_config

    cfg = load_config(args.config)
    model = build_model(cfg["model"]).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = LaplacePDEDataset(args.test_npz)
    # Subsample the first max_samples problems.
    indices = list(range(min(args.max_samples, len(ds))))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, indices),
        batch_size=16,
        shuffle=False,
    )

    if args.model_name == "ddpm":
        # Use upstream's helper — handles sample+mean aggregation and the
        # (K, B, 1, H, W) → (N, H, W) shape dance. Verified at plan-writing
        # time: src/diffphys/evaluation/evaluate_uq.py:98-118.
        ddpm = _build_ddpm(model, cfg["ddpm"])
        mean_pred, _std, truth_np = collect_generative_predictions(
            ddpm,
            loader,
            args.device,
            n_samples=args.n_samples,
        )
        predictions = mean_pred.astype(np.float32)
        truth = truth_np.astype(np.float32)
    else:
        # UNet / FNO: deterministic forward pass. No upstream helper for
        # this path (collect_generative_predictions requires .sample()).
        preds_list, truth_list = [], []
        with torch.no_grad():
            for cond, target in loader:
                cond, target = cond.to(args.device), target.to(args.device)
                pred = model(cond)[:, 0]  # (B, H, W)
                preds_list.append(pred.cpu().numpy())
                truth_list.append(target[:, 0].cpu().numpy())
        predictions = np.concatenate(preds_list, axis=0).astype(np.float32)
        truth = np.concatenate(truth_list, axis=0).astype(np.float32)

    assert predictions.shape == truth.shape, (
        f"shape mismatch: predictions {predictions.shape} vs truth {truth.shape}"
    )
    assert predictions.shape[1:] == (64, 64), f"expected 64x64 fields, got {predictions.shape}"

    np.savez_compressed(args.output, predictions=predictions, truth=truth)
    print(
        f"wrote {predictions.shape[0]} predictions to {args.output}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
