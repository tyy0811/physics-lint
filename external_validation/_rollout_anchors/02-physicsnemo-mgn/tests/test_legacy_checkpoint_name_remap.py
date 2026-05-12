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


# --- Edge-MLP input-column reorder (Band-C re-audit, D0-23) -----------------
# Legacy modulus / DeepMind concat the per-edge MLP input as [src, dst, edge];
# physicsnemo v2.0.0's concat_efeat_pyg uses [edge, src, dst]. The adapter
# permutes processor.edge_blocks.K.edge_mlp.0.weight's 3D input columns to match.


def test_edge_mlp_first_linear_columns_are_reordered() -> None:
    """[src(0:D) | dst(D:2D) | edge(2D:3D)]  ->  [edge | src | dst]."""
    torch = pytest.importorskip("torch")
    d = 4
    src = torch.full((d, d), 1.0)
    dst = torch.full((d, d), 2.0)
    edge = torch.full((d, d), 3.0)
    w_modulus = torch.cat((src, dst, edge), dim=1)  # shape (d, 3d)
    out = remap_modulus_to_physicsnemo_state_dict(
        {"processor.edge_blocks.0.edge_mlp.0.weight": w_modulus}
    )
    w_v200 = out["processor.processor_layers.0.edge_mlp.model.0.weight"]
    assert w_v200.shape == (d, 3 * d)
    assert torch.equal(w_v200[:, 0:d], edge)
    assert torch.equal(w_v200[:, d : 2 * d], src)
    assert torch.equal(w_v200[:, 2 * d : 3 * d], dst)


def test_only_first_edge_linear_weight_is_reordered() -> None:
    """edge_mlp.{2,4} (internal hidden), biases, and node_mlp.0 are untouched."""
    torch = pytest.importorskip("torch")
    d = 4
    w0 = torch.cat(
        (torch.full((d, d), 1.0), torch.full((d, d), 2.0), torch.full((d, d), 3.0)),
        dim=1,
    )
    w2 = torch.arange(d * d, dtype=torch.float32).reshape(d, d)
    b0 = torch.arange(d, dtype=torch.float32)
    node_w0 = torch.arange(d * 2 * d, dtype=torch.float32).reshape(d, 2 * d)
    out = remap_modulus_to_physicsnemo_state_dict(
        {
            "processor.edge_blocks.3.edge_mlp.0.weight": w0,
            "processor.edge_blocks.3.edge_mlp.0.bias": b0,
            "processor.edge_blocks.3.edge_mlp.2.weight": w2,
            "processor.node_blocks.3.node_mlp.0.weight": node_w0,
        }
    )
    # First edge Linear weight: reordered (block layout changed).
    assert not torch.equal(out["processor.processor_layers.6.edge_mlp.model.0.weight"], w0)
    # Everything else: passed through verbatim.
    assert torch.equal(out["processor.processor_layers.6.edge_mlp.model.0.bias"], b0)
    assert torch.equal(out["processor.processor_layers.6.edge_mlp.model.2.weight"], w2)
    assert torch.equal(out["processor.processor_layers.7.node_mlp.model.0.weight"], node_w0)


def test_reorder_preserves_edge_linear_output() -> None:
    """The permuted weight + v2.0.0 concat order reproduces the legacy forward exactly."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    d, n_edges = 8, 20
    lin_modulus = torch.nn.Linear(3 * d, d)  # the checkpoint's first edge-MLP Linear
    src = torch.randn(n_edges, d)
    dst = torch.randn(n_edges, d)
    edge = torch.randn(n_edges, d)
    # Legacy (modulus / DeepMind): cat((src, dst, edge)).
    y_modulus = lin_modulus(torch.cat((src, dst, edge), dim=1))
    # Remap the weight; feed v2.0.0's cat((edge, src, dst)) with the original bias.
    out = remap_modulus_to_physicsnemo_state_dict(
        {"processor.edge_blocks.0.edge_mlp.0.weight": lin_modulus.weight.detach()}
    )
    w_v200 = out["processor.processor_layers.0.edge_mlp.model.0.weight"]
    y_v200 = torch.nn.functional.linear(
        torch.cat((edge, src, dst), dim=1), w_v200, lin_modulus.bias.detach()
    )
    assert torch.allclose(y_modulus, y_v200, atol=1e-6)


def test_reorder_rejects_unexpected_edge_linear_shape() -> None:
    """A processor edge-MLP first-Linear not shaped (D, 3D) is a hard error, not a silent pass."""
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match=r"expected \(D, 3D\)"):
        remap_modulus_to_physicsnemo_state_dict(
            {"processor.edge_blocks.0.edge_mlp.0.weight": torch.zeros(128, 256)}
        )


def test_non_tensor_edge_linear_weight_passes_through() -> None:
    """String/int fixture values (rename-only unit tests) don't trigger the reorder."""
    out = remap_modulus_to_physicsnemo_state_dict(
        {"processor.edge_blocks.0.edge_mlp.0.weight": "E0"}
    )
    assert out == {"processor.processor_layers.0.edge_mlp.model.0.weight": "E0"}
