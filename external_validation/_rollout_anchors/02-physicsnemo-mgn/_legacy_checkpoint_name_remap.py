"""Pre-rename modulus → post-rename physicsnemo state_dict key remap.

The NGC `modulus_ns_meshgraphnet` checkpoint (uploaded 2023-05-26) was
trained against pre-rename modulus, which used the parameter-path naming
convention `*.mlp.*` for encoders/decoders and split the message-passing
processor into parallel `processor.edge_blocks[K]` + `processor.node_blocks[K]`
ModuleLists. Post-rename physicsnemo (v2.0.0, sha 1ca85d65) renamed these
submodules to `*.model.*` and restructured the processor into a single
interleaved `processor.processor_layers[2K]=edge, processor.processor_layers[2K+1]=node`
ModuleList. The forward-pass computation is identical (verified by reading
physicsnemo v2.0.0's MeshGraphNetProcessor.__init__: `chain(*zip(edge_blocks,
node_blocks))`); only the parameter paths differ.

This adapter is the response to Pattern A drift detected in BLOCKING-1 per
DECISIONS.md D0-23 verdict 1. **Computational identity is the falsifiable
hypothesis** — Gate D's `test_inference_matches_ngc_sample` (Task 7) within
the Phase-1-pinned tolerance is the empirical verification. If Gate D
fails, the rename-only assumption is falsified and the case study reroutes
to an older physicsnemo pin (Path A) or the FNO-on-Darcy fallback (Path C).

The `device_buffer` extra-key was inspected (Refinement 1 per design
escalation): shape=[0], numel=0, dtype=float32. It is a 0-element
placeholder tensor from the trainer's device-tracking machinery and
contains no learned state, so dropping it cannot produce silent wrong
outputs. This is the only key the adapter explicitly drops.
"""

from __future__ import annotations

import re

# Encoder/decoder rename: `<x>.mlp.<n>.<w_or_b>` → `<x>.model.<n>.<w_or_b>`.
# Matches `edge_encoder.mlp.*`, `node_encoder.mlp.*`, `node_decoder.mlp.*`.
_ENCODER_DECODER_PATTERN = re.compile(
    r"^(edge_encoder|node_encoder|node_decoder)\.mlp\.(\d+)\.(weight|bias)$"
)

# Processor edge-block rename:
# `processor.edge_blocks.<K>.edge_mlp.<N>.<w_or_b>`
# → `processor.processor_layers.<2K>.edge_mlp.model.<N>.<w_or_b>`
_PROCESSOR_EDGE_PATTERN = re.compile(
    r"^processor\.edge_blocks\.(\d+)\.edge_mlp\.(\d+)\.(weight|bias)$"
)

# Processor node-block rename:
# `processor.node_blocks.<K>.node_mlp.<N>.<w_or_b>`
# → `processor.processor_layers.<2K+1>.node_mlp.model.<N>.<w_or_b>`
_PROCESSOR_NODE_PATTERN = re.compile(
    r"^processor\.node_blocks\.(\d+)\.node_mlp\.(\d+)\.(weight|bias)$"
)

# Keys to drop unconditionally (verified safe per Refinement 1 / D0-23).
_DROP_KEYS: frozenset[str] = frozenset({"device_buffer"})


def remap_modulus_to_physicsnemo_state_dict(state_dict: dict) -> dict:
    """Remap an NGC modulus_ns_meshgraphnet state_dict to physicsnemo v2.0.0 names.

    Fails loudly (KeyError) on any key that matches neither a rename rule
    nor the explicit drop-list, so future checkpoint shape changes surface
    as errors rather than silent partial loads.
    """
    remapped: dict = {}
    for k, v in state_dict.items():
        if k in _DROP_KEYS:
            continue
        m = _ENCODER_DECODER_PATTERN.match(k)
        if m:
            module_name, layer_idx, param = m.groups()
            remapped[f"{module_name}.model.{layer_idx}.{param}"] = v
            continue
        m = _PROCESSOR_EDGE_PATTERN.match(k)
        if m:
            block_k, mlp_layer, param = m.groups()
            new_idx = 2 * int(block_k)
            remapped[f"processor.processor_layers.{new_idx}.edge_mlp.model.{mlp_layer}.{param}"] = v
            continue
        m = _PROCESSOR_NODE_PATTERN.match(k)
        if m:
            block_k, mlp_layer, param = m.groups()
            new_idx = 2 * int(block_k) + 1
            remapped[f"processor.processor_layers.{new_idx}.node_mlp.model.{mlp_layer}.{param}"] = v
            continue
        raise KeyError(
            f"Unknown key {k!r} in NGC state_dict — does not match any "
            f"documented rename rule (encoder/decoder .mlp. → .model., "
            f"processor.edge_blocks → processor_layers[2K] edge, "
            f"processor.node_blocks → processor_layers[2K+1] node) "
            f"nor the explicit drop-list ({sorted(_DROP_KEYS)}). The NGC "
            f"checkpoint structure may have changed; re-run "
            f"dump_full_keysets_for_rename_map and update this module."
        )
    return remapped
