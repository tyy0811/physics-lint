"""Rung 4a case-study driver — emit harness SARIF for SEGNN-TGV2D + GNS-TGV2D.

Per DECISIONS.md D0-19 + D0-20 + the rung-4a design doc at
`methodology/docs/2026-05-04-rung-4a-cross-stack-conservation-design.md`:

Reads the local mirror of the Modal Volume rollout subdirs (populated
by `modal volume get`), invokes lint_npz_dir on each stack, assembles
the 10 D0-19 run-level properties, calls emit_sarif twice — producing
two committed SARIF artifacts for the rung 4a writeup.

USAGE
-----

    # 1. Populate the local mirror (one-shot, ~30 sec per stack):
    modal volume get rollout-anchors-artifacts \\
        /vol/rollouts/lagrangebench/segnn_tgv2d_8c3d080397/ \\
        external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror/segnn_tgv2d_8c3d080397/
    modal volume get rollout-anchors-artifacts \\
        /vol/rollouts/lagrangebench/gns_tgv2d_f48dd3f376/ \\
        external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror/gns_tgv2d_f48dd3f376/

    # 2. Run from physics-lint repo root:
    python external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py

    # 3. Commit the two new SARIFs at outputs/sarif/.

The emission_sha is read from `git rev-parse --short=10 HEAD` at run
time (the current feature/rollout-anchors HEAD), so the SARIF filename
matches the run-level physics_lint_sha_sarif_emission field.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Ensure repo root is importable regardless of how this script is invoked
# (plan's documented invocation is `python <path>` from repo root, which
# does not auto-include the repo root in sys.path).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from external_validation._rollout_anchors._harness.lint_npz_dir import lint_npz_dir  # noqa: E402
from external_validation._rollout_anchors._harness.sarif_emitter import emit_sarif  # noqa: E402

# Pinned shas for the rung 3.5 PASS state on Modal Volume.
# These are the genesis shas for the npz contents — they DO NOT change
# when emit_sarif.py is re-run; the SARIF's physics_lint_sha_sarif_emission
# is a third sha read from git HEAD at emission time.
SEGNN_PKL_INFERENCE_SHA = "8c3d080397"
SEGNN_NPZ_CONVERSION_SHA = "5857144"  # post-D0-17-amendment-1 standalone Modal conversion
GNS_PKL_INFERENCE_SHA = "f48dd3f376"
GNS_NPZ_CONVERSION_SHA = "f48dd3f376"  # P1 inference + conversion in one shot
LAGRANGEBENCH_SHA = "b880a6c84a93792d2499d2a9b8ba3a077ddf44e2"

HARNESS_SARIF_SCHEMA_VERSION = "1.0"

# Local mirror paths (populated by `modal volume get` before this script runs).
REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_MIRROR_ROOT = (
    REPO_ROOT / "external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror"
)
SARIF_OUTPUT_ROOT = (
    REPO_ROOT / "external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif"
)


class MissingLocalMirrorError(Exception):
    """Raised when the local mirror dir does not exist or is empty —
    user must run `modal volume get` first.
    """


def _git_short_sha() -> str:
    """Return short (10-char) sha of the current feature/rollout-anchors HEAD."""
    result = subprocess.run(
        ["git", "rev-parse", "--short=10", "HEAD"],
        capture_output=True,
        check=True,
        cwd=REPO_ROOT,
        text=True,
    )
    return result.stdout.strip()


def _build_run_properties(
    *,
    model_name: str,
    dataset_name: str,
    checkpoint_id: str,
    pkl_inference_sha: str,
    npz_conversion_sha: str,
    sarif_emission_sha: str,
    rollout_subdir_volume_path: str,
) -> dict[str, str]:
    """Assemble the 10 D0-19 run-level fields for one stack."""
    return {
        "source": "rollout-anchor-harness",
        "harness_sarif_schema_version": HARNESS_SARIF_SCHEMA_VERSION,
        "physics_lint_sha_pkl_inference": pkl_inference_sha,
        "physics_lint_sha_npz_conversion": npz_conversion_sha,
        "physics_lint_sha_sarif_emission": sarif_emission_sha,
        "lagrangebench_sha": LAGRANGEBENCH_SHA,
        "checkpoint_id": checkpoint_id,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "rollout_subdir": rollout_subdir_volume_path,
    }


def _emit_for_stack(
    *,
    mirror_subdir: Path,
    sarif_output_path: Path,
    run_properties: dict[str, str],
    case_study_name: str,
    dataset_name: str,
    model_name: str,
    checkpoint_id: str,
) -> Path:
    """Run lint_npz_dir + emit_sarif for one stack."""
    if not mirror_subdir.exists() or not any(mirror_subdir.glob("particle_rollout_traj*.npz")):
        raise MissingLocalMirrorError(
            f"Local mirror missing or empty at {mirror_subdir}. "
            f"Run `modal volume get rollout-anchors-artifacts /vol/rollouts/lagrangebench/<subdir>/ {mirror_subdir}/` first."
        )
    results = lint_npz_dir(
        mirror_subdir,
        case_study=case_study_name,
        dataset=dataset_name,
        model=model_name,
        ckpt_hash=checkpoint_id,
    )
    return emit_sarif(
        results,
        output_path=sarif_output_path,
        run_properties=run_properties,
    )


def main() -> int:
    sarif_emission_sha = _git_short_sha()
    SARIF_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # SEGNN-TGV2D
    segnn_mirror = LOCAL_MIRROR_ROOT / f"segnn_tgv2d_{SEGNN_PKL_INFERENCE_SHA}"
    segnn_props = _build_run_properties(
        model_name="segnn",
        dataset_name="tgv2d",
        checkpoint_id="segnn_tgv2d",
        pkl_inference_sha=SEGNN_PKL_INFERENCE_SHA,
        npz_conversion_sha=SEGNN_NPZ_CONVERSION_SHA,
        sarif_emission_sha=sarif_emission_sha,
        rollout_subdir_volume_path=f"/vol/rollouts/lagrangebench/segnn_tgv2d_{SEGNN_PKL_INFERENCE_SHA}/",
    )
    segnn_sarif_path = SARIF_OUTPUT_ROOT / f"segnn_tgv2d_{sarif_emission_sha}.sarif"
    out_segnn = _emit_for_stack(
        mirror_subdir=segnn_mirror,
        sarif_output_path=segnn_sarif_path,
        run_properties=segnn_props,
        case_study_name="01-lagrangebench",
        dataset_name="tgv2d",
        model_name="segnn",
        checkpoint_id="segnn_tgv2d",
    )
    print(f"SEGNN SARIF: {out_segnn}")

    # GNS-TGV2D
    gns_mirror = LOCAL_MIRROR_ROOT / f"gns_tgv2d_{GNS_PKL_INFERENCE_SHA}"
    gns_props = _build_run_properties(
        model_name="gns",
        dataset_name="tgv2d",
        checkpoint_id="gns_tgv2d",
        pkl_inference_sha=GNS_PKL_INFERENCE_SHA,
        npz_conversion_sha=GNS_NPZ_CONVERSION_SHA,
        sarif_emission_sha=sarif_emission_sha,
        rollout_subdir_volume_path=f"/vol/rollouts/lagrangebench/gns_tgv2d_{GNS_PKL_INFERENCE_SHA}/",
    )
    gns_sarif_path = SARIF_OUTPUT_ROOT / f"gns_tgv2d_{sarif_emission_sha}.sarif"
    out_gns = _emit_for_stack(
        mirror_subdir=gns_mirror,
        sarif_output_path=gns_sarif_path,
        run_properties=gns_props,
        case_study_name="01-lagrangebench",
        dataset_name="tgv2d",
        model_name="gns",
        checkpoint_id="gns_tgv2d",
    )
    print(f"GNS SARIF: {out_gns}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
