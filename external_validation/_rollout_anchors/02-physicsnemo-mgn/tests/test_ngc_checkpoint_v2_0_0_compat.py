"""CPU-only NGC checkpoint ↔ PhysicsNeMo v2.0.0 state-dict-key smoke.

Per design §3.1 activity 1 (BLOCKING-1 unblock). Zero-GPU; runs locally.
Test marked skip when NGC checkpoint is absent; Task 4 downloads it, then
this test fires green or red and pins the verdict.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

NGC_CHECKPOINT_PATH = Path(
    os.environ.get(
        "PHYSICS_LINT_NGC_VORTEX_CHECKPOINT",
        # Default to the download location set by Task 4.
        "external_validation/_rollout_anchors/02-physicsnemo-mgn/cache/modulus_ns_meshgraphnet_v0.1/checkpoint.tar",
    )
).resolve()


@pytest.mark.skipif(
    not NGC_CHECKPOINT_PATH.exists(),
    reason=(
        f"NGC vortex-shedding checkpoint not at {NGC_CHECKPOINT_PATH}; "
        "Task 4 (NGC download entrypoint) must run first OR "
        "PHYSICS_LINT_NGC_VORTEX_CHECKPOINT must point to a local copy."
    ),
)
def test_ngc_vortex_shedding_checkpoint_loads_into_v2_0_0_meshgraphnet() -> None:
    """The NGC modulus_ns_meshgraphnet:v0.1 state_dict keys must match
    physicsnemo @ 1ca85d65's MeshGraphNet constructor with the args from
    conf/config.yaml (num_input_features=6, num_edge=3, num_output=3).

    Pre-rename modulus and post-rename physicsnemo may have renamed
    parameter paths or restructured layers; this smoke catches that
    before any Modal compute fires. BLOCKING-1 per design §3.1 / preflight.
    """
    from physicsnemo.models.meshgraphnet import MeshGraphNet

    model = MeshGraphNet(
        input_dim_nodes=6,
        input_dim_edges=3,
        output_dim=3,
    )
    expected_keys = set(model.state_dict().keys())

    ckpt = torch.load(NGC_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    # NGC checkpoint may wrap the state_dict under "model_state_dict" or "state_dict";
    # check both, fail loudly if neither.
    if "model_state_dict" in ckpt:
        ckpt_state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        ckpt_state_dict = ckpt["state_dict"]
    else:
        ckpt_state_dict = ckpt
    actual_keys = set(ckpt_state_dict.keys())

    missing_in_ckpt = expected_keys - actual_keys
    extra_in_ckpt = actual_keys - expected_keys

    assert not missing_in_ckpt and not extra_in_ckpt, (
        f"NGC checkpoint state_dict keys do not match physicsnemo @ 1ca85d65 "
        f"MeshGraphNet constructor. Missing in checkpoint: {sorted(missing_in_ckpt)[:5]}. "
        f"Extra in checkpoint: {sorted(extra_in_ckpt)[:5]}. "
        f"BLOCKING per design §3.1 activity 1; consider an older physicsnemo "
        f"pin or FNO-on-Darcy fallback per Gate D."
    )
