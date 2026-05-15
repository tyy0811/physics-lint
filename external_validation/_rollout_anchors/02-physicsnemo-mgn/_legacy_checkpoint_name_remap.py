"""Modulus (legacy) -> physicsnemo v2.0.0 state_dict adapter for the NGC MGN checkpoint.

The NGC ``modulus_ns_meshgraphnet`` checkpoint (uploaded 2023-05-26) was trained
against legacy modulus. Its ``model_state_dict`` keys match
``modulus/models/meshgraphnet/meshgraphnet.py`` at the **v0.1.0 tag** (2023-05-02 ---
the only release predating the upload): the encoders/decoder use the parameter-path
``*.mlp.*`` and the message-passing processor is a pair of *parallel*
``processor.edge_blocks[K]`` / ``processor.node_blocks[K]`` ``ModuleList``s. (v0.2.0
already restructured the processor into a single interleaved ``processor_layers``
``ModuleList`` and renamed ``_EdgeBlock``/``_NodeBlock`` -> ``MeshEdgeBlock``/
``MeshNodeBlock``; the checkpoint has neither, so it is firmly v0.1.0-era. Verified
against the cached ``model.pt``: 262 learned tensors + the 0-element ``device_buffer``;
``processor.edge_blocks.K.edge_mlp.0.weight`` is ``[128, 384]``,
``processor.node_blocks.K.node_mlp.0.weight`` is ``[128, 256]``, K in 0..14.)

Post-rename physicsnemo (v2.0.0, sha 1ca85d65) renamed the encoder/decoder
submodules ``*.mlp.*`` -> ``*.model.*`` and restructured the processor into a single
interleaved ``processor.processor_layers[2K]=edge, processor.processor_layers[2K+1]=node``
``ModuleList`` (``MeshGraphNetProcessor.__init__``: ``chain(*zip(edge_blocks, node_blocks))``).
**This adapter therefore applies two transformations:**

1. **Key rename** -- the cosmetic part:
     * ``<edge_encoder|node_encoder|node_decoder>.mlp.<n>.<w|b>``  ->  ``<...>.model.<n>.<w|b>``
     * ``processor.edge_blocks.<K>.edge_mlp.<n>.<w|b>``  ->  ``processor.processor_layers.<2K>.edge_mlp.model.<n>.<w|b>``
     * ``processor.node_blocks.<K>.node_mlp.<n>.<w|b>``  ->  ``processor.processor_layers.<2K+1>.node_mlp.model.<n>.<w|b>``

2. **Edge-MLP input-column reorder** -- the *non*-cosmetic part, the residual that
   made Gate D (Task 7) fail at E=0.80 with a clean ``load_state_dict(strict=True)``.
   The legacy ``_EdgeBlock`` (modulus v0.1.0) builds each edge update's MLP input as
   ``torch.cat((edges.src["x"], edges.dst["x"], edges.data["x"]), dim=1)`` -- column
   order ``[src_node, dst_node, edge]``, which is exactly DeepMind-meshgraphnets'
   ``tf.concat([sender, receiver, edge], axis=-1)``. physicsnemo v2.0.0's
   ``concat_efeat_pyg`` (``physicsnemo/nn/module/gnn_layers/utils.py`` @ 1ca85d65)
   instead builds it as ``torch.cat((efeat, src, dst), dim=1)`` -- column order
   ``[edge, src_node, dst_node]``. The first ``Linear`` of every processor edge block
   (``processor.edge_blocks.K.edge_mlp.0.weight``, shape ``[D, 3D]`` with
   ``D = hidden_dim_processor = 128``) thus has its ``3D`` input columns laid out
   ``[src(0:D), dst(D:2D), edge(2D:3D)]`` in the checkpoint; to be loaded into v2.0.0
   they must be permuted to ``[edge(2D:3D), src(0:D), dst(D:2D)]``. ``strict=True``
   passes either way -- the ``[128, 384]`` shape is unchanged; only the *meaning* of
   the columns differs -- which is why this surfaced empirically (garbage forward) and
   not at load time.

   Nothing else needs adjusting. ``processor.*.edge_mlp.{2,4}`` are ``Linear(128,128)``
   on the MLP's internal hidden state; the node block's concat ``cat((aggregated_edges,
   self_node))`` -> ``[agg, self]`` is **identical** in modulus v0.1.0
   (``_NodeBlock``) and v2.0.0 (``agg_concat_pyg``); and the encoders/decoder consume
   dataset features whose construction (``cat((velocity, one_hot_node_type))`` ->
   6-D node input; ``cat((disp, disp_norm))`` -> 3-D edge input; ``cat((velocity_diff,
   pressure))`` -> 3-D target) is byte-identical between modulus's ``MGNDataset`` and
   v2.0.0's ``VortexSheddingDataset.__getitem__``. With transformations (1)+(2) the
   per-step block order (edge-then-node, the node block consuming the just-updated edge
   features), the residual connections, and the MLP layer geometry all match between
   the two versions -- so the forward-pass computation is identical. Gate D's
   ``audit_ngc_sample_reproduction`` (Task 7) re-fired against the post-(2) adapter is
   the empirical verification (D0-23 "Band-C refinement").

The 0-element ``device_buffer`` extra-key (shape ``[0]``, ``float32``, no learned
state -- trainer device-tracking machinery) is the only key dropped unconditionally.
"""

from __future__ import annotations

import re

# Encoder/decoder rename: `<x>.mlp.<n>.<w_or_b>` -> `<x>.model.<n>.<w_or_b>`.
# Matches `edge_encoder.mlp.*`, `node_encoder.mlp.*`, `node_decoder.mlp.*`.
_ENCODER_DECODER_PATTERN = re.compile(
    r"^(edge_encoder|node_encoder|node_decoder)\.mlp\.(\d+)\.(weight|bias)$"
)

# Processor edge-block rename:
# `processor.edge_blocks.<K>.edge_mlp.<N>.<w_or_b>`
# -> `processor.processor_layers.<2K>.edge_mlp.model.<N>.<w_or_b>`
_PROCESSOR_EDGE_PATTERN = re.compile(
    r"^processor\.edge_blocks\.(\d+)\.edge_mlp\.(\d+)\.(weight|bias)$"
)

# Processor node-block rename:
# `processor.node_blocks.<K>.node_mlp.<N>.<w_or_b>`
# -> `processor.processor_layers.<2K+1>.node_mlp.model.<N>.<w_or_b>`
_PROCESSOR_NODE_PATTERN = re.compile(
    r"^processor\.node_blocks\.(\d+)\.node_mlp\.(\d+)\.(weight|bias)$"
)

# Keys to drop unconditionally (verified safe per Refinement 1 / D0-23).
_DROP_KEYS: frozenset[str] = frozenset({"device_buffer"})


def _looks_like_tensor(v: object) -> bool:
    """Duck-type a 2-D tensor without importing torch at module scope.

    Keeps this module import-light (unit tests exercise the rename logic with
    plain strings/ints); the column reorder only runs on real checkpoint
    tensors, which carry `.ndim`/`.shape`/`.dtype`.
    """
    return hasattr(v, "ndim") and hasattr(v, "shape")


def _reorder_edge_mlp_input_columns(weight):
    """Permute the first edge-MLP Linear's input columns ``[src, dst, edge]`` -> ``[edge, src, dst]``.

    Converts the legacy-modulus / DeepMind concat order to physicsnemo v2.0.0's.
    ``weight`` is ``processor.edge_blocks.K.edge_mlp.0.weight`` with shape ``[D, 3D]``
    (``D == hidden_dim_processor``; ``D == 128`` for the NGC checkpoint -- the v2.0.0
    /modulus default where ``hidden_dim_processor == hidden_dim_node_encoder ==
    hidden_dim_edge_encoder``). Returns a new tensor; raises loudly on an unexpected
    shape so a checkpoint with a different processor geometry surfaces as an error.
    """
    import torch

    if getattr(weight, "ndim", None) != 2 or weight.shape[1] != 3 * weight.shape[0]:
        raise ValueError(
            f"processor edge-block first-Linear weight has shape {tuple(weight.shape)}; "
            f"expected (D, 3D) with D == hidden_dim_processor (D == 128 for the NGC "
            f"checkpoint, where the node/edge encoder hidden dims equal the processor "
            f"hidden dim). The checkpoint's processor MLP geometry differs from modulus "
            f"v0.1.0's _EdgeBlock; re-derive the edge-MLP concat layout before remapping."
        )
    d = weight.shape[0]
    src = weight[:, 0:d]
    dst = weight[:, d : 2 * d]
    edge = weight[:, 2 * d : 3 * d]
    return torch.cat((edge, src, dst), dim=1).contiguous()


def remap_modulus_to_physicsnemo_state_dict(state_dict: dict) -> dict:
    """Remap an NGC modulus_ns_meshgraphnet state_dict to physicsnemo v2.0.0.

    Renames keys (legacy ``.mlp.`` -> ``.model.``, parallel ``edge_blocks``/
    ``node_blocks`` -> interleaved ``processor_layers``) **and** reorders the
    processor edge-MLP first-Linear input columns from the legacy ``[src, dst, edge]``
    layout to v2.0.0's ``[edge, src, dst]`` -- see the module docstring. Fails loudly
    (KeyError) on any key matching neither a rename rule nor the explicit drop-list,
    so a future checkpoint structure change surfaces as an error rather than a silent
    partial load.
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
            # The first Linear (mlp index 0) consumes the per-edge concat
            # [src, dst, edge] (legacy) -- v2.0.0 expects [edge, src, dst].
            if mlp_layer == "0" and param == "weight" and _looks_like_tensor(v):
                v = _reorder_edge_mlp_input_columns(v)
            remapped[f"processor.processor_layers.{new_idx}.edge_mlp.model.{mlp_layer}.{param}"] = v
            continue
        m = _PROCESSOR_NODE_PATTERN.match(k)
        if m:
            block_k, mlp_layer, param = m.groups()
            new_idx = 2 * int(block_k) + 1
            remapped[f"processor.processor_layers.{new_idx}.node_mlp.model.{mlp_layer}.{param}"] = v
            continue
        raise KeyError(
            f"Unknown key {k!r} in NGC state_dict -- does not match any "
            f"documented rename rule (encoder/decoder .mlp. -> .model., "
            f"processor.edge_blocks -> processor_layers[2K] edge, "
            f"processor.node_blocks -> processor_layers[2K+1] node) "
            f"nor the explicit drop-list ({sorted(_DROP_KEYS)}). The NGC "
            f"checkpoint structure may have changed; re-run "
            f"dump_full_keysets_for_rename_map and update this module."
        )
    return remapped
