"""CPU-only NGC checkpoint ↔ PhysicsNeMo v2.0.0 state-dict-key smoke.

Per design §3.1 activity 1 (BLOCKING-1 unblock). Zero-GPU; runs locally
when physicsnemo is installed. Test marked skip when NGC checkpoint or
physicsnemo is absent; Task 4 downloads the checkpoint, then this test
fires green or red and pins the verdict.

Verdict path: PASS requires the legacy-modulus → physicsnemo v2.0.0
state_dict name-remap adapter (per D0-23 verdict 1 / `_legacy_checkpoint_name_remap.py`).
Architecture identity (the remap's load-bearing assumption) is empirically
verified by Gate D (`test_inference_matches_ngc_sample`, Task 7).
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch

NGC_CHECKPOINT_PATH = Path(
    os.environ.get(
        "PHYSICS_LINT_NGC_VORTEX_CHECKPOINT",
        # Default to the download location set by Task 4. NGC ships the
        # vortex_shedding_mgn.zip → vortex_shedding_mgn/model.pt; the catalog
        # only tags `latest` (no v0.1 / v0.2 exists for this model).
        "external_validation/_rollout_anchors/02-physicsnemo-mgn/cache/modulus_ns_meshgraphnet_latest/vortex_shedding_mgn/model.pt",
    )
).resolve()

# Hyphenated parent dir blocks normal Python imports; load via importlib.
_ADAPTER_PATH = Path(__file__).resolve().parent.parent / "_legacy_checkpoint_name_remap.py"
_spec = importlib.util.spec_from_file_location("_legacy_checkpoint_name_remap", _ADAPTER_PATH)
assert _spec is not None and _spec.loader is not None
_remap_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_remap_module)
remap_modulus_to_physicsnemo_state_dict = _remap_module.remap_modulus_to_physicsnemo_state_dict

# Skipif gating: physicsnemo is heavy + may not be locally installed. The
# Modal entrypoint `verify_ngc_checkpoint_state_dict_compat` is the source-
# of-truth fire for the BLOCKING-1 verdict (always runs in the pinned image).
_physicsnemo_spec = importlib.util.find_spec("physicsnemo")


@pytest.mark.skipif(
    not NGC_CHECKPOINT_PATH.exists(),
    reason=(
        f"NGC vortex-shedding checkpoint not at {NGC_CHECKPOINT_PATH}; "
        "Task 4 (NGC download entrypoint) must run first OR "
        "PHYSICS_LINT_NGC_VORTEX_CHECKPOINT must point to a local copy."
    ),
)
@pytest.mark.skipif(
    _physicsnemo_spec is None,
    reason=(
        "physicsnemo not installed locally; run the Modal entrypoint "
        "`verify_ngc_checkpoint_state_dict_compat` for the canonical "
        "BLOCKING-1 verdict (always uses the pinned image)."
    ),
)
def test_ngc_vortex_shedding_checkpoint_loads_into_v2_0_0_meshgraphnet() -> None:
    """The NGC modulus_ns_meshgraphnet:latest state_dict keys (after the
    legacy-modulus → physicsnemo v2.0.0 name-remap adapter) must match
    physicsnemo @ 1ca85d65's MeshGraphNet constructor with the args from
    conf/config.yaml (input_dim_nodes=6, input_dim_edges=3, output_dim=3).
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
        raw_state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        raw_state_dict = ckpt["state_dict"]
    else:
        raw_state_dict = ckpt

    ckpt_state_dict = remap_modulus_to_physicsnemo_state_dict(raw_state_dict)
    actual_keys = set(ckpt_state_dict.keys())

    missing_in_ckpt = expected_keys - actual_keys
    extra_in_ckpt = actual_keys - expected_keys

    assert not missing_in_ckpt and not extra_in_ckpt, (
        f"NGC checkpoint state_dict keys (after name-remap adapter) do not "
        f"match physicsnemo @ 1ca85d65 MeshGraphNet constructor. "
        f"Missing in checkpoint: {sorted(missing_in_ckpt)[:5]}. "
        f"Extra in checkpoint: {sorted(extra_in_ckpt)[:5]}. "
        f"BLOCKING per design §3.1 activity 1; rename adapter may need an "
        f"update OR an older physicsnemo pin / FNO-on-Darcy fallback (Gate D)."
    )
