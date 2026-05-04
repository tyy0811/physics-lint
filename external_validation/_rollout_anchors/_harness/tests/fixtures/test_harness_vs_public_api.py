"""Gate B — controlled-fixture harness-vs-public-API validation.

Per spec §4.2 fixture #5 and spec §6 Gate B, this test computes
ε_harness_vs_public on each (rule, fixture) pair and asserts the
verdict-determining inequality. Verdict bands per spec §4.3 and
``SCHEMA.md`` §4.1:

| ε_harness_vs_public | Verdict     |
|---------------------|-------------|
| ≤ 10⁻⁴              | PASS        |
| 10⁻⁴ < ε ≤ 10⁻²     | APPROXIMATE |
| > 10⁻²              | FAIL        |

The PASS threshold is **pre-registered**; this test asserts ε ≤ 10⁻⁴
hard. APPROXIMATE / FAIL paths are not auto-promoted to passing tests
— under the spec, those verdicts trigger DECISIONS.md updates and a
documentation pass, not a soft assertion. If a future divergence
takes the value into the APPROXIMATE band, the right move is to (1)
update DECISIONS.md, (2) update the README's "What we are NOT
claiming" section per spec §1.4, then (3) loosen this test's
assertion explicitly with a comment citing the divergence — not to
silently relax the threshold.

Day-0 scope: PH-SYM-001 + PH-SYM-002 cross-path comparison on
fixtures #1 and #2. PH-CON-001 cross-path comparison is deliberately
out of scope per the rationale in
:mod:`mass_conservation_fixture`; its public-rule path requires a
heat-equation (Nx, Ny, Nt) GridField, which Day-0 fixture
infrastructure does not produce.
"""

from __future__ import annotations

import numpy as np
import pytest

from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    ParticleSnapshot,
    c4_static_defect,
    gridify,
    mass_defect,
    reflection_static_defect,
)
from external_validation._rollout_anchors._harness.tests.fixtures import (
    c4_grid_equivalent,
    c4_invariant_4particle,
    c4_perturbed_4particle,
    mass_conservation_fixture,
)
from physics_lint.rules import ph_sym_001, ph_sym_002

# Pre-registered Gate B PASS tolerance, mirrored from SCHEMA.md §4.1.
GATE_B_PASS_THRESHOLD: float = 1e-4


def _public_api_c4_raw_value(snapshot: ParticleSnapshot) -> float:
    """Run PH-SYM-001 on the gridified fixture and return raw_value.

    Mirrors what Gate B asks for on the public-API path: ``gridify``
    the snapshot with the shared parameters in
    :mod:`c4_grid_equivalent`, wrap the result in a ``GridField``, hand
    it to ``ph_sym_001.check`` with the shared C4 spec, return
    ``result.raw_value`` — the max relative L^2 over the C4 generator
    rotations.
    """
    u = gridify(
        snapshot,
        grid_size=c4_grid_equivalent.GRID_SIZE,
        bandwidth=c4_grid_equivalent.BANDWIDTH,
        domain=c4_grid_equivalent.DOMAIN,
    )
    field = c4_grid_equivalent.make_grid_field(u)
    spec = c4_grid_equivalent.make_c4_spec()
    result = ph_sym_001.check(field, spec)
    assert result.raw_value is not None, "PH-SYM-001 must emit a raw_value"
    return result.raw_value


def _public_api_reflection_raw_value(snapshot: ParticleSnapshot) -> float:
    """Run PH-SYM-002 on the gridified fixture; return raw_value (max-axis)."""
    u = gridify(
        snapshot,
        grid_size=c4_grid_equivalent.GRID_SIZE,
        bandwidth=c4_grid_equivalent.BANDWIDTH,
        domain=c4_grid_equivalent.DOMAIN,
    )
    field = c4_grid_equivalent.make_grid_field(u)
    spec = c4_grid_equivalent.make_c4_spec()
    result = ph_sym_002.check(field, spec)
    assert result.raw_value is not None, "PH-SYM-002 must emit a raw_value"
    return result.raw_value


# ---------------------------------------------------------------------------
# Fixture #1 — exact-C4 4-particle configuration
# ---------------------------------------------------------------------------


def test_c4_invariant_harness_vs_public_api():
    """ε_harness_vs_public on PH-SYM-001 cross-path; expected ≤ 10⁻⁴."""
    snap = c4_invariant_4particle.build_snapshot()
    eps_harness = c4_static_defect(
        snap,
        grid_size=c4_grid_equivalent.GRID_SIZE,
        bandwidth=c4_grid_equivalent.BANDWIDTH,
        domain=c4_grid_equivalent.DOMAIN,
    )
    eps_public = _public_api_c4_raw_value(snap)
    diff = abs(eps_harness - eps_public)
    assert diff <= GATE_B_PASS_THRESHOLD, (
        f"Gate B PH-SYM-001 fixture #1: ε_harness={eps_harness:.6e} "
        f"vs ε_public={eps_public:.6e} (|diff|={diff:.6e}) "
        f"exceeds pre-registered {GATE_B_PASS_THRESHOLD:.0e} threshold"
    )
    # Sanity: both paths should return ≤ 10⁻⁶ on a C4-invariant config
    # (spec §4.2 fixture-construction tolerance, separate from Gate B).
    assert eps_harness <= 1e-6, (
        f"Fixture #1 c4_static_defect ({eps_harness:.6e}) exceeds spec "
        f"§4.2 fixture-construction tolerance 10⁻⁶ — fixture configuration "
        f"may have a hidden symmetry-breaking offset."
    )


def test_reflection_invariant_harness_vs_public_api():
    """ε_harness_vs_public on PH-SYM-002 cross-path; expected ≤ 10⁻⁴."""
    snap = c4_invariant_4particle.build_snapshot()
    # PH-SYM-002 takes max over declared reflection axes (x and y);
    # the harness's reflection_static_defect is per-axis, so to mirror
    # the public emission take the max of the two harness calls.
    eps_harness = max(
        reflection_static_defect(
            snap,
            axis=0,
            grid_size=c4_grid_equivalent.GRID_SIZE,
            bandwidth=c4_grid_equivalent.BANDWIDTH,
            domain=c4_grid_equivalent.DOMAIN,
        ),
        reflection_static_defect(
            snap,
            axis=1,
            grid_size=c4_grid_equivalent.GRID_SIZE,
            bandwidth=c4_grid_equivalent.BANDWIDTH,
            domain=c4_grid_equivalent.DOMAIN,
        ),
    )
    eps_public = _public_api_reflection_raw_value(snap)
    diff = abs(eps_harness - eps_public)
    assert diff <= GATE_B_PASS_THRESHOLD, (
        f"Gate B PH-SYM-002 fixture #1: ε_harness={eps_harness:.6e} "
        f"vs ε_public={eps_public:.6e} (|diff|={diff:.6e}) "
        f"exceeds pre-registered {GATE_B_PASS_THRESHOLD:.0e} threshold"
    )


# ---------------------------------------------------------------------------
# Fixture #2 — same configuration with one particle displaced by delta
# ---------------------------------------------------------------------------


def test_c4_perturbed_harness_vs_public_api():
    """ε_harness_vs_public on perturbed fixture; expected ≤ 10⁻⁴.

    Both paths emit ε = O(delta / bandwidth) on this fixture; what Gate B
    asserts is *agreement between the two paths*, not the absolute ε
    magnitude. The agreement-to-10⁻⁴ claim is trivially met because both
    paths apply identical operations to the same gridified field.
    """
    snap = c4_perturbed_4particle.build_snapshot()
    eps_harness = c4_static_defect(
        snap,
        grid_size=c4_grid_equivalent.GRID_SIZE,
        bandwidth=c4_grid_equivalent.BANDWIDTH,
        domain=c4_grid_equivalent.DOMAIN,
    )
    eps_public = _public_api_c4_raw_value(snap)
    diff = abs(eps_harness - eps_public)
    assert diff <= GATE_B_PASS_THRESHOLD, (
        f"Gate B PH-SYM-001 fixture #2 (perturbed): "
        f"ε_harness={eps_harness:.6e} vs ε_public={eps_public:.6e} "
        f"(|diff|={diff:.6e}) exceeds {GATE_B_PASS_THRESHOLD:.0e}"
    )
    # Sanity: ε is non-trivial — the perturbation actually moved the
    # density field. Catches the failure mode where the perturbation
    # silently doesn't propagate (e.g., dataclass copy bug).
    assert eps_harness > 1e-3, (
        f"Fixture #2 ε_harness={eps_harness:.6e} suspiciously small — "
        f"perturbation delta={c4_perturbed_4particle.DELTA} should "
        f"produce ε of order delta/bandwidth = "
        f"{c4_perturbed_4particle.DELTA / c4_grid_equivalent.BANDWIDTH}."
    )


def test_c4_perturbed_is_order_delta():
    """Defect scales linearly with delta for delta << bandwidth.

    Sanity check on the harness's gridify+rot90 pipeline. Doubling
    delta should approximately double ε in the linear-response regime
    (delta well below bandwidth = 0.04). Not an exact 2x because the
    Gaussian kernel response is delta * exp(...) which is only
    approximately linear, but the slope should be close enough to
    catch a regression to a quadratic or saturating response.
    """
    snap_d = c4_perturbed_4particle.build_snapshot(delta=0.005)
    snap_2d = c4_perturbed_4particle.build_snapshot(delta=0.010)
    eps_d = c4_static_defect(
        snap_d,
        grid_size=c4_grid_equivalent.GRID_SIZE,
        bandwidth=c4_grid_equivalent.BANDWIDTH,
        domain=c4_grid_equivalent.DOMAIN,
    )
    eps_2d = c4_static_defect(
        snap_2d,
        grid_size=c4_grid_equivalent.GRID_SIZE,
        bandwidth=c4_grid_equivalent.BANDWIDTH,
        domain=c4_grid_equivalent.DOMAIN,
    )
    ratio = eps_2d / eps_d
    # Allow 1.7x - 2.4x to absorb sub-linear / super-linear kernel curvature.
    assert 1.7 < ratio < 2.4, (
        f"Doubling delta should give ratio ~2x; got {ratio:.3f} "
        f"(eps_d={eps_d:.3e}, eps_2d={eps_2d:.3e})"
    )


# ---------------------------------------------------------------------------
# Fixture #4 — closed-system mass conservation (harness-only sanity)
# ---------------------------------------------------------------------------


def test_mass_conservation_harness_zero_defect():
    """Closed-system mass_defect must be exactly zero.

    Per the rationale in :mod:`mass_conservation_fixture`, the
    public-API path for PH-CON-001 (which consumes a heat-equation 3D
    GridField) is exercised separately in
    ``external_validation/PH-CON-001/test_anchor.py``; this test
    validates only the harness's mass arithmetic on a closed system.
    """
    t0 = mass_conservation_fixture.build_t0()
    t1 = mass_conservation_fixture.build_t1()
    eps = mass_defect(t0, t1)
    assert eps == 0.0, f"Closed-system mass_defect must be exactly zero; got {eps:.6e}"


# ---------------------------------------------------------------------------
# Cross-fixture sanity
# ---------------------------------------------------------------------------


def test_perturbed_fixture_has_one_displaced_particle():
    """Fixture #2 differs from fixture #1 only in particle 0's x-coord."""
    snap1 = c4_invariant_4particle.build_snapshot()
    snap2 = c4_perturbed_4particle.build_snapshot()
    diff = snap2.positions - snap1.positions
    assert np.allclose(diff[0], [c4_perturbed_4particle.DELTA, 0.0])
    assert np.allclose(diff[1:], 0.0)
    assert np.array_equal(snap1.particle_type, snap2.particle_type)
    assert np.array_equal(snap1.particle_mass, snap2.particle_mass)


def test_invariant_fixture_uses_exact_orbit_cells():
    """Fixture #1's particles must lie on exact GRID_SIZE-cell C4 orbits.

    Catches regressions to a non-orbit configuration — which would
    silently push fixture #1's ε above the 10⁻⁶ machine-precision
    expectation and cause the Gate B PASS assertion to start
    failing for the wrong reason.
    """
    snap = c4_invariant_4particle.build_snapshot()
    n = c4_grid_equivalent.GRID_SIZE
    cells = (snap.positions * n).astype(int)
    # All four particles' cell indices must be integers (they were
    # built that way), and the four cells together must form a discrete
    # rot90 orbit: cell (i, j) -> (j, n-1-i) -> (n-1-i, n-1-j) -> (n-1-j, i).
    i, j = cells[0]
    expected = {
        (int(i), int(j)),
        (int(j), int(n - 1 - i)),
        (int(n - 1 - i), int(n - 1 - j)),
        (int(n - 1 - j), int(i)),
    }
    actual = {tuple(c) for c in cells.tolist()}
    assert actual == expected


# ---------------------------------------------------------------------------
# Verdict reporting (plain stdout — pytest captures and prints under -v)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gate_b_verdict_record(request):
    """Module-scope record of the eight (rule, fixture, ε) tuples for the
    DECISIONS.md Gate B verdict entry. Printed at module teardown so the
    Day-0 commit can quote it.
    """
    record: list[tuple[str, str, float]] = []
    yield record

    print("\n=== Gate B verdict record ===")
    for rule, fixture, eps in record:
        print(f"  {rule:12s} {fixture:35s} ε_harness_vs_public = {eps:.3e}")
    if record:
        worst = max(e for _, _, e in record)
        verdict = (
            "PASS" if worst <= GATE_B_PASS_THRESHOLD else "APPROXIMATE" if worst <= 1e-2 else "FAIL"
        )
        print(f"  worst ε = {worst:.3e}  ->  Gate B verdict: {verdict}")


def test_record_gate_b_eps_values(gate_b_verdict_record):
    """Populate the verdict record. Always passes; reporting only."""
    snap1 = c4_invariant_4particle.build_snapshot()
    snap2 = c4_perturbed_4particle.build_snapshot()
    for fixture_name, snap in (
        ("c4_invariant_4particle", snap1),
        ("c4_perturbed_4particle", snap2),
    ):
        eps_h = c4_static_defect(
            snap,
            grid_size=c4_grid_equivalent.GRID_SIZE,
            bandwidth=c4_grid_equivalent.BANDWIDTH,
            domain=c4_grid_equivalent.DOMAIN,
        )
        eps_p = _public_api_c4_raw_value(snap)
        gate_b_verdict_record.append(("PH-SYM-001", fixture_name, abs(eps_h - eps_p)))
        eps_h_r = max(
            reflection_static_defect(
                snap,
                axis=0,
                grid_size=c4_grid_equivalent.GRID_SIZE,
                bandwidth=c4_grid_equivalent.BANDWIDTH,
                domain=c4_grid_equivalent.DOMAIN,
            ),
            reflection_static_defect(
                snap,
                axis=1,
                grid_size=c4_grid_equivalent.GRID_SIZE,
                bandwidth=c4_grid_equivalent.BANDWIDTH,
                domain=c4_grid_equivalent.DOMAIN,
            ),
        )
        eps_p_r = _public_api_reflection_raw_value(snap)
        gate_b_verdict_record.append(("PH-SYM-002", fixture_name, abs(eps_h_r - eps_p_r)))
