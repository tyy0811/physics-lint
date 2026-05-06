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
