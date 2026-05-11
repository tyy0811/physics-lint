"""Unit tests for the modulus → physicsnemo state_dict name-remap adapter.

The hyphenated parent dir `02-physicsnemo-mgn/` blocks normal Python
imports, so we load the module via importlib (same pattern as other
tests in this directory address path resolution via Path(__file__)).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ADAPTER_PATH = Path(__file__).resolve().parent.parent / "_legacy_checkpoint_name_remap.py"
_spec = importlib.util.spec_from_file_location("_legacy_checkpoint_name_remap", _ADAPTER_PATH)
assert _spec is not None and _spec.loader is not None
_remap_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_remap_module)
remap_modulus_to_physicsnemo_state_dict = _remap_module.remap_modulus_to_physicsnemo_state_dict


def test_encoder_rename() -> None:
    sd = {
        "edge_encoder.mlp.0.weight": "W",
        "edge_encoder.mlp.0.bias": "B",
        "node_encoder.mlp.4.weight": "W2",
    }
    out = remap_modulus_to_physicsnemo_state_dict(sd)
    assert out == {
        "edge_encoder.model.0.weight": "W",
        "edge_encoder.model.0.bias": "B",
        "node_encoder.model.4.weight": "W2",
    }


def test_decoder_rename() -> None:
    sd = {"node_decoder.mlp.5.bias": "X"}
    assert remap_modulus_to_physicsnemo_state_dict(sd) == {"node_decoder.model.5.bias": "X"}


def test_processor_edge_block_interleaving() -> None:
    """edge_blocks.K → processor_layers.(2K) per chain(*zip(...)) order."""
    sd = {
        "processor.edge_blocks.0.edge_mlp.0.weight": "E0",
        "processor.edge_blocks.7.edge_mlp.3.bias": "E7",
        "processor.edge_blocks.14.edge_mlp.5.weight": "E14",
    }
    out = remap_modulus_to_physicsnemo_state_dict(sd)
    assert out == {
        "processor.processor_layers.0.edge_mlp.model.0.weight": "E0",
        "processor.processor_layers.14.edge_mlp.model.3.bias": "E7",
        "processor.processor_layers.28.edge_mlp.model.5.weight": "E14",
    }


def test_processor_node_block_interleaving() -> None:
    """node_blocks.K → processor_layers.(2K+1) per chain(*zip(...)) order."""
    sd = {
        "processor.node_blocks.0.node_mlp.0.weight": "N0",
        "processor.node_blocks.7.node_mlp.3.bias": "N7",
        "processor.node_blocks.14.node_mlp.5.weight": "N14",
    }
    out = remap_modulus_to_physicsnemo_state_dict(sd)
    assert out == {
        "processor.processor_layers.1.node_mlp.model.0.weight": "N0",
        "processor.processor_layers.15.node_mlp.model.3.bias": "N7",
        "processor.processor_layers.29.node_mlp.model.5.weight": "N14",
    }


def test_device_buffer_dropped() -> None:
    sd = {"device_buffer": "anything", "edge_encoder.mlp.0.weight": "W"}
    out = remap_modulus_to_physicsnemo_state_dict(sd)
    assert "device_buffer" not in out
    assert out == {"edge_encoder.model.0.weight": "W"}


def test_unknown_key_raises_keyerror() -> None:
    """Fail loudly per design: silent drops mask checkpoint shape drift."""
    with pytest.raises(KeyError, match="does not match any"):
        remap_modulus_to_physicsnemo_state_dict({"foo.bar.baz": "X"})


def test_round_trip_count_preservation() -> None:
    """All non-dropped keys must produce exactly one output key (no merges, no splits)."""
    sd = {
        "edge_encoder.mlp.0.weight": 1,
        "edge_encoder.mlp.0.bias": 1,
        "processor.edge_blocks.0.edge_mlp.0.weight": 1,
        "processor.node_blocks.0.node_mlp.0.weight": 1,
        "device_buffer": 1,  # dropped
    }
    out = remap_modulus_to_physicsnemo_state_dict(sd)
    # 5 input keys minus 1 dropped = 4 output keys.
    assert len(out) == 4
