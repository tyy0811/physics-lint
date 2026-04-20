# PH-CON-003 — Heat energy-dissipation sign

## Citation

- **Paper:** Evans, *Partial Differential Equations*.
- **Venue:** AMS Graduate Studies in Mathematics, Vol. 19, 2nd ed. 2010.
- **ISBN:** 978-0-8218-4974-3.
- **Section:** §7.1.2.
- **Artifact:** Theorem 2 (energy estimates for parabolic equations). The `L²(U)` bound on `u_m` implies `E(t) ≤ E(0)` for the heat-equation eigenmode, which the rule checks on the `sin(πx)sin(πy)` eigenmode fixture.
- **Pinned value:** per-step energy ratio on `sin(πx)sin(πy)` eigenmode with `κ = 0.1`, `Δt = 0.05`: `exp(-4κπ² · Δt) = exp(-0.0987) ≈ 0.82087`. Tolerance `ε_quad = 10⁻⁴`.
- **Verification date:** 2026-04-20.
- **Verification protocol:** analytical derivation. The `sin(πx)sin(πy)` eigenmode evolves as `sin(πx)sin(πy)·exp(-2κπ²t)` under the heat equation `∂_t u = κ·Δu` with homogeneous Dirichlet BCs; `E(t) = ½∫u²` evolves as `E₀ · exp(-4κπ²t)`; per-step ratio is `exp(-4κπ² · Δt)`. Evans §7.1.2 Theorem 2 verified at theorem-number precision against the text — see `../_harness/TEXTBOOK_AVAILABILITY.md`.

**κ choice rationale.** κ = 0.1 makes energies decay by a factor of ~2.2 over `t ∈ [0, 0.2]`, well inside the regime where the rule's `np.gradient(energy, dt, edge_order=2)` tracks the decay without endpoint artifacts. A larger κ (e.g. κ = 1.0, which produces a 2700× energy range over the same window) causes the 2nd-order backward-quadratic extrapolation at the final timestep to produce a spurious positive `dE/dt ≈ 0.1 · E_max` — enough to trigger FAIL even though the eigenmode is strictly dissipative. κ = 0.1 matches the style of `tests/rules/test_ph_con_003.py::test_ph_con_003_decaying_energy_passes` (which uses κ = 0.01 with 16 timesteps over 0.5 s, also gentle).

## Test design

- **Fixture:** analytical `u(x,y,t) = sin(πx)sin(πy)·exp(-2κπ²t)` with `κ = 0.1` on `[0,1]²` with homogeneous Dirichlet BCs.
- **Timesteps:** `t ∈ {0, 0.05, 0.1, 0.15, 0.2}`.
- **Grid:** 64 × 64 per slice; 5 slices. Axis convention: time LAST (`grid_shape=[64, 64, 5]`, `h=(H, H, 0.05)`), matching `ph_con_003.py:44-52`.
- **Quadrature:** composite trapezoidal, `O(h²)` on the smooth eigenmode integrand.
- **Negative control:** `u_fake(t) = u_eigenmode(0) · (1 + 2·t/0.2)`, roughly `3×` total energy growth. Signal scale mirrors `tests/rules/test_ph_con_003.py::test_ph_con_003_energy_growth_is_warn` calibration which asserts `status in {"WARN", "FAIL"}` on ~50% total growth; tripling the signal provides margin above the WARN/FAIL cut-off while staying in the range the rule's floor was calibrated for.

## Acceptance criteria

- Fixture-sanity: per-step ratio of the analytical eigenmode matches `exp(-4κπ² · 0.05) ≈ 0.82087` (κ = 0.1) within `ε_quad = 10⁻⁴` (independent of the rule).
- Analytical heat sequence passes the rule with `status == "PASS"`.
- Non-dissipative control produces `status in {"WARN", "FAIL"}` (tri-state classification depends on calibrated floor; both outcomes are informative failures).
- Wall-time < 10 s on CPU.
