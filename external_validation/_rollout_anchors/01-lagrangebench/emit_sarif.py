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

import argparse
import subprocess
import sys
from pathlib import Path

# Ensure repo root is importable regardless of how this script is invoked
# (plan's documented invocation is `python <path>` from repo root, which
# does not auto-include the repo root in sys.path).
# REMOVE WHEN PACKAGED: becomes dead code if physics-lint ships
# pip-installable with proper console_scripts / entry_points.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from external_validation._rollout_anchors._harness.inference_manifest import (  # noqa: E402
    read_inference_manifest_status,
)
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

# Rung 4c dam-break shas (Task 8 fire output).
# Both stacks fired at sha e754a4bc2e: SEGNN-dam2d (N=20 attempt) timed
# out at 12/20 trajs; standalone conversion via convert_pkls_p1_segnn_dam2d
# produced 12 npzs. GNS-dam2d (N=12) completed cleanly with 12 npzs.
# Per D0-22 amendment 2, both stacks ship at N=12.
SEGNN_DAM2D_PKL_INFERENCE_SHA = "e754a4bc2e"
SEGNN_DAM2D_NPZ_CONVERSION_SHA = "e754a4bc2e"
GNS_DAM2D_PKL_INFERENCE_SHA = "e754a4bc2e"
GNS_DAM2D_NPZ_CONVERSION_SHA = "e754a4bc2e"

HARNESS_SARIF_SCHEMA_VERSION = "1.0"

# Per the rung 4a writeup's "N identical fires" claim: each stack must
# carry exactly this many trajectories. lint_npz_dir's gap-detection
# already rejects holes; this driver-level assertion catches the case
# where the count is contiguous but wrong (e.g., 19 trajs because one
# never made it to the volume), which lint_npz_dir cannot diagnose.
# Per-case-pair value: rung-4a TGV2D ships at N=20; rung-4c dam2d ships
# at N=12 per D0-22 amendment 2.
EXPECTED_TRAJ_COUNT_TGV2D = 20
EXPECTED_TRAJ_COUNT_DAM2D = 12

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


class UnexpectedTrajCountError(Exception):
    """Raised when a stack's trajectory count does not match
    EXPECTED_TRAJ_COUNT. The writeup's "20 identical fires" claim binds
    on the count; a 19-traj artifact would silently drop one row from
    the table without surfacing the loss.
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


def _read_inference_manifest_status(mirror_subdir: Path, *, required: bool = False) -> str | None:
    """Local-side classification — delegates to the shared helper.

    See ``_harness.inference_manifest.read_inference_manifest_status``.
    Pre-rung-4c §9 fold-in round 3 the logic was duplicated here for
    cross-runtime reasons (modal_app runs inside Modal containers;
    emit_sarif locally). Round 3 promoted to the shared helper after
    Codex review surfaced that both duplicates collapsed manifest-
    corruption into the legacy-absent status, hiding artifact-level
    provenance gaps on stale-mirror or corrupt-backfill scenarios.

    ``required=True`` is passed for post-fold-in stacks (rung-4c dam2d
    onward) where the manifest must exist; missing manifests then
    raise ``FileNotFoundError`` rather than silently producing a
    provenance-stripped SARIF. ``required=False`` (default) preserves
    the legacy-stack behavior (rung-4a/4b tgv2d): missing manifest
    returns None and the optional SARIF property is omitted; the
    renderer surfaces the absence explicitly as "n/a (pre-salvage-tag-
    schema)" rather than defaulting to a clean classification.
    """
    return read_inference_manifest_status(mirror_subdir, required=required)


def _build_run_properties(
    *,
    model_name: str,
    dataset_name: str,
    checkpoint_id: str,
    pkl_inference_sha: str,
    npz_conversion_sha: str,
    sarif_emission_sha: str,
    rollout_subdir_volume_path: str,
    inference_run_status: str | None = None,
) -> dict[str, str]:
    """Assemble the 10 D0-19 run-level fields for one stack.

    The optional 11th field ``inference_run_status`` is populated when
    the local mirror carries a `_inference_manifest.json` (rung-4c §9
    review-gate fold-in). Schema version stays at 1.0 — this is an
    optional additive field, not a required-field bump; renderer's
    fail-loud schema check still passes for legacy SARIFs without it.
    """
    properties: dict[str, str] = {
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
    if inference_run_status is not None:
        properties["inference_run_status"] = inference_run_status
    return properties


def _emit_for_stack(
    *,
    mirror_subdir: Path,
    sarif_output_path: Path,
    run_properties: dict[str, str],
    case_study_name: str,
    dataset_name: str,
    model_name: str,
    checkpoint_id: str,
    expected_traj_count: int,
    manifest_required: bool = False,
) -> Path:
    """Run lint_npz_dir + emit_sarif for one stack.

    Reads ``<mirror_subdir>/_inference_manifest.json`` and merges the
    classification into ``run_properties`` as the optional
    ``inference_run_status`` field (rung-4c §9 review-gate fold-in
    round 2). When ``manifest_required=True`` (rung-4c dam2d and
    post-fold-in stacks), a missing manifest raises FileNotFoundError
    rather than silently omitting the field — emitting a provenance-
    stripped SARIF on a post-fold-in stack hides the salvage tag and
    defeats round 2's artifact-level transparency. When
    ``manifest_required=False`` (rung-4a/4b legacy tgv2d), missing is
    expected and the field is omitted; the renderer surfaces the
    absence as "n/a (pre-salvage-tag-schema)".

    A corrupt manifest (parseable JSON but missing classification
    fields; or unparsable bytes; or non-dict root) raises
    ``ManifestInvalidError`` regardless of ``manifest_required``
    (round 3 absorption — corruption is not a legacy-absence case).
    """
    if not mirror_subdir.exists() or not any(mirror_subdir.glob("particle_rollout_traj*.npz")):
        raise MissingLocalMirrorError(
            f"Local mirror missing or empty at {mirror_subdir}. "
            f"Run `modal volume get rollout-anchors-artifacts /vol/rollouts/lagrangebench/<subdir>/ {mirror_subdir}/` first."
        )
    npz_count = sum(1 for _ in mirror_subdir.glob("particle_rollout_traj*.npz"))
    if npz_count != expected_traj_count:
        raise UnexpectedTrajCountError(
            f"Stack at {mirror_subdir} has {npz_count} trajectories, expected "
            f"{expected_traj_count}. The writeup's "
            f'"{expected_traj_count} identical fires" claim binds on this count.'
        )
    inference_run_status = _read_inference_manifest_status(
        mirror_subdir, required=manifest_required
    )
    if inference_run_status is not None:
        run_properties = {**run_properties, "inference_run_status": inference_run_status}
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stacks",
        choices=["all", "tgv2d", "dam2d"],
        default="all",
        help=(
            "Which rung's stacks to emit. 'tgv2d' = rung-4a (SEGNN-TGV2D + GNS-TGV2D); "
            "'dam2d' = rung-4c (SEGNN-DAM2D + GNS-DAM2D); 'all' = both. Default: all. "
            "Use 'dam2d' when re-emitting only rung-4c (e.g. after a rung-4c-scoped "
            "local-mirror refresh) so rung-4a SARIFs stay at their existing "
            "sarif_emission_sha and the rung-4a writeup's provenance line stays valid."
        ),
    )
    args = parser.parse_args(argv)

    sarif_emission_sha = _git_short_sha()
    SARIF_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if args.stacks not in ("all", "tgv2d"):
        # Skip rung-4a TGV2D stacks.
        pass
    if args.stacks in ("all", "tgv2d"):
        _emit_rung_4a_tgv2d_stacks(sarif_emission_sha)
    if args.stacks in ("all", "dam2d"):
        _emit_rung_4c_dam2d_stacks(sarif_emission_sha)
    return 0


def _emit_rung_4a_tgv2d_stacks(sarif_emission_sha: str) -> None:
    """Emit rung-4a SEGNN-TGV2D + GNS-TGV2D SARIFs at the given emission sha."""
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
        expected_traj_count=EXPECTED_TRAJ_COUNT_TGV2D,
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
        expected_traj_count=EXPECTED_TRAJ_COUNT_TGV2D,
    )
    print(f"GNS SARIF: {out_gns}")


def _emit_rung_4c_dam2d_stacks(sarif_emission_sha: str) -> None:
    """Emit rung-4c SEGNN-DAM2D + GNS-DAM2D SARIFs at the given emission sha.

    Picks up ``inference_run_status`` from each stack's local-mirror
    ``_inference_manifest.json`` (backfilled at the rung-4c §9
    review-gate fold-in). SEGNN-DAM2D's manifest classifies the run as
    ``from_aborted_inference`` (timeout-salvage; D0-22 amendment 2);
    GNS-DAM2D's classifies as ``from_completed_inference`` (clean N=12).
    """
    # SEGNN-DAM2D (rung 4c, D0-22 + amendment 1 + amendment 2)
    segnn_dam2d_mirror = LOCAL_MIRROR_ROOT / f"segnn_dam2d_{SEGNN_DAM2D_PKL_INFERENCE_SHA}"
    segnn_dam2d_props = _build_run_properties(
        model_name="segnn",
        dataset_name="dam2d",
        checkpoint_id="segnn_dam2d",
        pkl_inference_sha=SEGNN_DAM2D_PKL_INFERENCE_SHA,
        npz_conversion_sha=SEGNN_DAM2D_NPZ_CONVERSION_SHA,
        sarif_emission_sha=sarif_emission_sha,
        rollout_subdir_volume_path=f"/vol/rollouts/lagrangebench/segnn_dam2d_{SEGNN_DAM2D_PKL_INFERENCE_SHA}/",
    )
    segnn_dam2d_sarif_path = SARIF_OUTPUT_ROOT / f"segnn_dam2d_{sarif_emission_sha}.sarif"
    out_segnn_dam2d = _emit_for_stack(
        mirror_subdir=segnn_dam2d_mirror,
        sarif_output_path=segnn_dam2d_sarif_path,
        run_properties=segnn_dam2d_props,
        case_study_name="01-lagrangebench",
        dataset_name="dam2d",
        model_name="segnn",
        checkpoint_id="segnn_dam2d",
        expected_traj_count=EXPECTED_TRAJ_COUNT_DAM2D,
        # Rung-4c §9 fold-in round 3: post-fold-in stacks require the
        # manifest. A missing manifest at this point means the local
        # mirror is stale or the backfill was incomplete — both should
        # fail emission rather than produce a provenance-stripped SARIF
        # that silently hides the salvage tag.
        manifest_required=True,
    )
    print(f"SEGNN-DAM2D SARIF: {out_segnn_dam2d}")

    # GNS-DAM2D (rung 4c, D0-22 + amendment 1 + amendment 2)
    gns_dam2d_mirror = LOCAL_MIRROR_ROOT / f"gns_dam2d_{GNS_DAM2D_PKL_INFERENCE_SHA}"
    gns_dam2d_props = _build_run_properties(
        model_name="gns",
        dataset_name="dam2d",
        checkpoint_id="gns_dam2d",
        pkl_inference_sha=GNS_DAM2D_PKL_INFERENCE_SHA,
        npz_conversion_sha=GNS_DAM2D_NPZ_CONVERSION_SHA,
        sarif_emission_sha=sarif_emission_sha,
        rollout_subdir_volume_path=f"/vol/rollouts/lagrangebench/gns_dam2d_{GNS_DAM2D_PKL_INFERENCE_SHA}/",
    )
    gns_dam2d_sarif_path = SARIF_OUTPUT_ROOT / f"gns_dam2d_{sarif_emission_sha}.sarif"
    out_gns_dam2d = _emit_for_stack(
        mirror_subdir=gns_dam2d_mirror,
        sarif_output_path=gns_dam2d_sarif_path,
        run_properties=gns_dam2d_props,
        case_study_name="01-lagrangebench",
        dataset_name="dam2d",
        model_name="gns",
        checkpoint_id="gns_dam2d",
        expected_traj_count=EXPECTED_TRAJ_COUNT_DAM2D,
        manifest_required=True,  # Rung-4c §9 fold-in round 3 (see SEGNN-DAM2D above)
    )
    print(f"GNS-DAM2D SARIF: {out_gns_dam2d}")


if __name__ == "__main__":
    sys.exit(main())
