"""Verify _load_floor wildcard matching for PDE-agnostic floor entries."""

from physics_lint.rules._helpers import _load_floor


def test_load_floor_sym001_wildcard_matches_any_pde():
    """SYM-001 floor with pde='*' should match laplace, heat, poisson, wave."""
    for pde in ("laplace", "heat", "poisson", "wave"):
        floor = _load_floor(
            rule="PH-SYM-001",
            pde=pde,
            grid_shape=(64, 64),
            method="rot90",
            norm="max-rel-L2",
        )
        assert floor.source == "calibrated", f"pde={pde} didn't match calibrated entry"
        assert abs(floor.value - 2.895e-16) < 1e-20


def test_load_floor_sym001_wildcard_matches_any_grid():
    """SYM-001 floor with grid_shape='*' should match any grid size."""
    for gs in ((32, 32), (64, 64), (128, 128), (256, 256)):
        floor = _load_floor(
            rule="PH-SYM-001",
            pde="laplace",
            grid_shape=gs,
            method="rot90",
            norm="max-rel-L2",
        )
        assert floor.source == "calibrated", f"grid_shape={gs} didn't match"
        assert abs(floor.value - 2.895e-16) < 1e-20


def test_load_floor_exact_match_beats_wildcard():
    """If both exact and wildcard entries exist, exact should win."""
    # PH-RES-001 has exact entries (pde='laplace', grid_shape=[64,64]).
    # Even if a hypothetical wildcard existed, exact should take priority.
    floor = _load_floor(
        rule="PH-RES-001",
        pde="laplace",
        grid_shape=(64, 64),
        method="fd4",
        norm="L2",
    )
    assert floor.source == "calibrated"
    # This should be the exact entry, not a wildcard
    assert abs(floor.value - 1.003e-12) < 1e-16


def test_load_floor_non_matching_falls_to_default():
    """A rule with no floor entry at all falls back to shipped default."""
    floor = _load_floor(
        rule="PH-FAKE-999",
        pde="laplace",
        grid_shape=(64, 64),
        method="fake",
        norm="fake",
    )
    assert floor.source == "shipped"
    assert floor.value == 1e-5  # global default
