"""Tests for dogfood.make_ci_dumps — extraction → CLI-loader-compatible conversion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dogfood.make_ci_dumps import _extract_boundary_values, convert_model


def _write_fake_extraction(
    path: Path, n_samples: int = 4, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    predictions = rng.standard_normal((n_samples, 64, 64)).astype(np.float32)
    truth = rng.standard_normal((n_samples, 64, 64)).astype(np.float32)
    np.savez_compressed(path, predictions=predictions, truth=truth)
    return predictions, truth


def test_boundary_ordering_matches_gridfield(tmp_path: Path):
    """_extract_boundary_values must match GridField.values_on_boundary() exactly —
    otherwise the shipped boundary_target array is element-mismatched with
    the rule's observed boundary, and PH-BC-001 measures noise."""
    from physics_lint.field import GridField

    rng = np.random.default_rng(42)
    field_2d = rng.standard_normal((32, 32)).astype(np.float32)

    expected = GridField(field_2d, h=(1 / 31, 1 / 31), periodic=False).values_on_boundary()
    actual = _extract_boundary_values(field_2d)

    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)


def test_convert_model_writes_loadable_dump(tmp_path: Path):
    """Converted dump must load through physics_lint.loader.load_target."""
    from physics_lint.loader import load_target

    extraction_path = tmp_path / "fno.npz"
    predictions, truth = _write_fake_extraction(extraction_path, n_samples=3, seed=1)

    output_path = tmp_path / "fno_pred.npz"
    convert_model(extraction_path, output_path, sample_index=0)

    loaded = load_target(output_path, cli_overrides={}, toml_path=None)
    assert loaded.spec.pde == "laplace"
    assert tuple(loaded.spec.grid_shape) == (64, 64)
    # The boundary_target must be plumbed through the loader
    assert loaded.boundary_target is not None
    # And must equal the ground-truth boundary of sample 0
    expected_boundary = _extract_boundary_values(truth[0])
    np.testing.assert_array_equal(loaded.boundary_target, expected_boundary)
    # Field values must equal the prediction (float32)
    np.testing.assert_array_equal(loaded.field.values(), predictions[0])


def test_convert_model_sample_index(tmp_path: Path):
    extraction_path = tmp_path / "ddpm.npz"
    predictions, _ = _write_fake_extraction(extraction_path, n_samples=5, seed=7)

    output_path = tmp_path / "ddpm_pred.npz"
    convert_model(extraction_path, output_path, sample_index=3)

    loaded = np.load(output_path)
    np.testing.assert_array_equal(loaded["prediction"], predictions[3])


def test_convert_model_rejects_wrong_shape(tmp_path: Path):
    path = tmp_path / "broken.npz"
    np.savez_compressed(path, predictions=np.zeros((2, 32, 32), dtype=np.float32))
    with pytest.raises(ValueError, match="does not match expected"):
        convert_model(path, tmp_path / "out.npz")


def test_convert_model_rejects_missing_predictions(tmp_path: Path):
    path = tmp_path / "broken.npz"
    np.savez_compressed(path, foo=np.zeros((4, 64, 64), dtype=np.float32))
    with pytest.raises(KeyError, match="predictions"):
        convert_model(path, tmp_path / "out.npz")


def test_convert_model_without_truth_omits_boundary_target(tmp_path: Path):
    """Extraction output without `truth` key still converts; boundary_target
    is simply omitted and the CLI falls back to dirichlet_homogeneous
    auto-extraction (if applicable) or SKIPPED."""
    path = tmp_path / "no_truth.npz"
    preds = np.zeros((2, 64, 64), dtype=np.float32)
    np.savez_compressed(path, predictions=preds)

    output_path = tmp_path / "no_truth_pred.npz"
    convert_model(path, output_path, sample_index=0)

    loaded = np.load(output_path)
    assert "boundary_target" not in loaded.files
