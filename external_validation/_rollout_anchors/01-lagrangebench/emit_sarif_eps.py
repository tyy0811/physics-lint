"""Rung 4b case-study driver: eps(t) npz dir -> committed SARIF artifact.

Sibling to emit_sarif.py (rung 4a). Assembles run-level v1.1 SARIF
properties from arguments (4-stage sha provenance per SCHEMA.md §1.5
and §3.5), invokes lint_eps_dir to read the eps(t) npzs, and writes the
SARIF via the shared emit_sarif primitive.

The eps(t) npzs themselves are produced by the Modal entrypoints in
modal_app.py (lagrangebench_eps_p0_segnn_tgv2d / _p1_gns_tgv2d); this
driver runs after a `modal volume get` brings them to the local mirror.

USAGE
-----

    # 1. After Modal eps job and `modal volume get`, the local mirror
    #    holds eps_*.npz files at:
    #    outputs/trajectories/{model}_tgv2d_{eps_sha}/eps_*.npz

    # 2. From repo root, run:
    python external_validation/_rollout_anchors/01-lagrangebench/emit_sarif_eps.py \\
        --segnn-eps-dir outputs/trajectories/segnn_tgv2d_<sha>/ \\
        --gns-eps-dir   outputs/trajectories/gns_tgv2d_<sha>/

    # 3. Commit the two new SARIFs at outputs/sarif/.

The emission_sha is read from `git rev-parse --short=10 HEAD` at run
time, matching rung-4a's emit_sarif.py convention.
"""

from __future__ import annotations

import sys
from pathlib import Path

# sys.path bootstrap so `from external_validation...` imports resolve
# regardless of how this script is invoked. Mirrors rung-4a's
# emit_sarif.py pattern.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from external_validation._rollout_anchors._harness.lint_eps_dir import (  # noqa: E402
    lint_eps_dir,
)
from external_validation._rollout_anchors._harness.sarif_emitter import (  # noqa: E402
    emit_sarif,
)

HARNESS_SARIF_SCHEMA_VERSION = "1.1"


def emit_sarif_eps(
    *,
    eps_dir: Path,
    out_sarif_path: Path,
    case_study: str,
    dataset: str,
    model: str,
    ckpt_hash: str,
    ckpt_id: str,
    physics_lint_sha_pkl_inference: str,
    physics_lint_sha_npz_conversion: str,
    physics_lint_sha_eps_computation: str,
    physics_lint_sha_sarif_emission: str,
    lagrangebench_sha: str,
    rollout_subdir: str,
) -> Path:
    """Read eps(t) npzs from eps_dir, emit SARIF v1.1 to out_sarif_path."""
    results = lint_eps_dir(
        eps_dir=eps_dir,
        case_study=case_study,
        dataset=dataset,
        model=model,
        ckpt_hash=ckpt_hash,
    )

    run_properties = {
        "source": "rollout-anchor-harness",
        "harness_sarif_schema_version": HARNESS_SARIF_SCHEMA_VERSION,
        "physics_lint_sha_pkl_inference": physics_lint_sha_pkl_inference,
        "physics_lint_sha_npz_conversion": physics_lint_sha_npz_conversion,
        "physics_lint_sha_eps_computation": physics_lint_sha_eps_computation,
        "physics_lint_sha_sarif_emission": physics_lint_sha_sarif_emission,
        "lagrangebench_sha": lagrangebench_sha,
        "checkpoint_id": ckpt_id,
        "model_name": model,
        "dataset_name": dataset,
        "rollout_subdir": rollout_subdir,
    }

    return emit_sarif(
        results,
        output_path=out_sarif_path,
        run_properties=run_properties,
    )


# =============================================================================
# CLI driver — mirrors rung-4a emit_sarif.py shape.
# =============================================================================
# Pinned shas + ckpt hashes for the rung-4b T9 PASS state on Modal Volume.
# `physics_lint_sha_eps_computation` is the merge-base sha of the entrypoint
# at T9 firing time (must match what the npzs themselves recorded — verified
# by lint_eps_dir's per-file ckpt_hash check + the explicit assert below).

_SEGNN_PKL_INFERENCE_SHA = "8c3d080397"
_SEGNN_NPZ_CONVERSION_SHA = "5857144"
_SEGNN_CKPT_HASH = "sha256:c0be98f9fb59eb4545f05db3d8aa5d31b7c8170b5d4d9634b01749e26598441b"

_GNS_PKL_INFERENCE_SHA = "f48dd3f376"
_GNS_NPZ_CONVERSION_SHA = "f48dd3f376"
_GNS_CKPT_HASH = "sha256:c1df5675d6b29aa7e4b130afc8b88b31f7109ce41dacc9f4e168e5c485a8765e"

# Captured at image-build time; not currently recorded on eps_t.npz (only on
# rung-4a's particle_rollout_traj*.npz). Re-derive when rollout_image is
# rebuilt; the LB clone is `--depth 1` of master, not a sha pin.
_LAGRANGEBENCH_SHA = "b880a6c84a93792d2499d2a9b8ba3a077ddf44e2"

# Per design §3 + §7: 20 PH-SYM-003 SKIP rows + 120 main-sweep entries
# (6 transforms x 20 trajs) = 140 eps_t.npz files per stack. Figure-sweep
# entries overwrite 3 of the main-sweep entries (no count change).
_EXPECTED_NPZ_COUNT = 140

_REPO_ROOT = Path(__file__).resolve().parents[3]
_TRAJECTORIES_ROOT = (
    _REPO_ROOT / "external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories"
)
_SARIF_OUTPUT_ROOT = (
    _REPO_ROOT / "external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif"
)


class MissingLocalMirrorError(Exception):
    """Raised when an expected eps-npz mirror dir is missing or empty.

    Mirrors rung-4a's MissingLocalMirrorError; the user must run
    `modal volume get rollout-anchors-artifacts /trajectories/<subdir>/
    outputs/trajectories/` first.
    """


class UnexpectedNpzCountError(Exception):
    """Raised when a stack's eps_*.npz count does not match _EXPECTED_NPZ_COUNT.

    The rung 4b writeup binds on the (20 SKIP + 120 main) shape; a
    short count would silently drop rows from the table.
    """


def _git_short_sha() -> str:
    """Return short (10-char) sha of current HEAD; matches rung-4a convention."""
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "--short=10", "HEAD"],
        capture_output=True,
        check=True,
        cwd=_REPO_ROOT,
        text=True,
    )
    return result.stdout.strip()


def _emit_for_stack(
    *,
    model: str,
    ckpt_id: str,
    ckpt_hash: str,
    pkl_inference_sha: str,
    npz_conversion_sha: str,
    eps_computation_sha: str,
    sarif_emission_sha: str,
) -> Path:
    """Run lint_eps_dir + emit_sarif for one stack; fail-loud on missing mirror."""
    eps_dir_name = f"{model}_tgv2d_{eps_computation_sha}"
    eps_dir = _TRAJECTORIES_ROOT / eps_dir_name
    if not eps_dir.exists():
        raise MissingLocalMirrorError(
            f"eps-npz mirror missing at {eps_dir}. Run "
            f"`modal volume get rollout-anchors-artifacts "
            f"/trajectories/{eps_dir_name}/ {_TRAJECTORIES_ROOT}/` first."
        )
    npz_count = sum(1 for _ in eps_dir.glob("eps_*.npz"))
    if npz_count != _EXPECTED_NPZ_COUNT:
        raise UnexpectedNpzCountError(
            f"Stack {model} at {eps_dir} has {npz_count} eps_*.npz files, "
            f"expected {_EXPECTED_NPZ_COUNT} (= 20 PH-SYM-003 SKIP + 120 main-sweep). "
            f"Writeup tabular shape binds on this count."
        )

    sarif_path = _SARIF_OUTPUT_ROOT / f"{model}_tgv2d_eps_{sarif_emission_sha}.sarif"
    rollout_subdir = f"/vol/trajectories/{eps_dir_name}/"
    return emit_sarif_eps(
        eps_dir=eps_dir,
        out_sarif_path=sarif_path,
        case_study="01-lagrangebench",
        dataset="tgv2d",
        model=model,
        ckpt_hash=ckpt_hash,
        ckpt_id=ckpt_id,
        physics_lint_sha_pkl_inference=pkl_inference_sha,
        physics_lint_sha_npz_conversion=npz_conversion_sha,
        physics_lint_sha_eps_computation=eps_computation_sha,
        physics_lint_sha_sarif_emission=sarif_emission_sha,
        lagrangebench_sha=_LAGRANGEBENCH_SHA,
        rollout_subdir=rollout_subdir,
    )


def main() -> int:
    """Emit one SARIF v1.1 artifact per stack from local eps-npz mirrors.

    Reads the 4-stage sha provenance from the eps_*.npz files (via
    lint_eps_dir); the 5th sha (`physics_lint_sha_sarif_emission`) is read
    from `git rev-parse --short=10 HEAD` at run time, matching rung-4a's
    emit_sarif.py convention. The SARIF filename embeds the emission sha
    so the artifact identity is reproducible.
    """
    sarif_emission_sha = _git_short_sha()
    _SARIF_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # eps_computation_sha is the same for both stacks at the rung-4b T9 run
    # (both fired from the same entrypoint sha 255af5de8d). Hardcoded here
    # rather than read at runtime from the entrypoint, matching emit_sarif.py's
    # constants pattern; a future re-run at a different eps_computation_sha
    # bumps the constant.
    eps_computation_sha = "255af5de8d"

    out_segnn = _emit_for_stack(
        model="segnn",
        ckpt_id="segnn_tgv2d",
        ckpt_hash=_SEGNN_CKPT_HASH,
        pkl_inference_sha=_SEGNN_PKL_INFERENCE_SHA,
        npz_conversion_sha=_SEGNN_NPZ_CONVERSION_SHA,
        eps_computation_sha=eps_computation_sha,
        sarif_emission_sha=sarif_emission_sha,
    )
    print(f"SEGNN SARIF: {out_segnn}")

    out_gns = _emit_for_stack(
        model="gns",
        ckpt_id="gns_tgv2d",
        ckpt_hash=_GNS_CKPT_HASH,
        pkl_inference_sha=_GNS_PKL_INFERENCE_SHA,
        npz_conversion_sha=_GNS_NPZ_CONVERSION_SHA,
        eps_computation_sha=eps_computation_sha,
        sarif_emission_sha=sarif_emission_sha,
    )
    print(f"GNS SARIF: {out_gns}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
