# PH-CON-003 — Heat energy-dissipation sign

## Citation

- **Paper:** Evans, *Partial Differential Equations*.
- **Venue:** AMS Graduate Studies in Mathematics, Vol. 19, 2nd ed. 2010.
- **ISBN:** 978-0-8218-4974-3.
- **Section:** §7.1.2.
- **Artifact:** Theorem 2 (energy estimates for parabolic equations). The `L²(U)` bound on `u_m` implies `E(t) ≤ E(0)` for the heat-equation eigenmode, which the rule checks on the `sin(πx)sin(πy)` eigenmode fixture.
- **Pinned value:** per-step energy ratio on `sin(πx)sin(πy)` eigenmode with `κ = 1`, `Δt = 0.05`: `exp(-4π² · 0.05) = exp(-0.2π²) ≈ 0.13888`. Tolerance `ε_quad = 10⁻⁴`.
- **Verification date:** 2026-04-20.
- **Verification protocol:** analytical derivation. The `sin(πx)sin(πy)` eigenmode evolves as `sin(πx)sin(πy)·exp(-2π²t)` under the heat equation `∂_t u = Δu` (κ = 1) with homogeneous Dirichlet BCs; `E(t) = ½∫u²` evolves as `E₀ · exp(-4π²t)`; per-step ratio is `exp(-4π² · Δt)`. Evans §7.1.2 Theorem 2 verified at theorem-number precision against the text — see `../_harness/TEXTBOOK_AVAILABILITY.md`.

## Test design

- **Fixture:** analytical `u(x,y,t) = sin(πx)sin(πy)·exp(-2π²t)` on `[0,1]²` with homogeneous Dirichlet BCs and `κ = 1`.
- **Timesteps:** `t ∈ {0, 0.05, 0.1, 0.15, 0.2}` (5 samples, `Δt = 0.05`). Energies decay by a factor of ~2700× over the window; the rule's forward-difference `dE/dt = np.diff(energy)/dt` tracks this cleanly with no endpoint extrapolation artifact.
- **Grid:** 64 × 64 per slice; 5 slices. Axis convention: time LAST (`grid_shape=[64, 64, 5]`, `h=(H, H, 0.05)`), matching `ph_con_003.py:44-52`.
- **Quadrature:** composite trapezoidal, `O(h²)` on the smooth eigenmode integrand.
- **Negative control:** `u_fake(t) = u_eigenmode(0) · (1 + 2·t/0.2)`, roughly `3×` total energy growth. Signal scale mirrors `tests/rules/test_ph_con_003.py::test_ph_con_003_energy_growth_is_warn` calibration which asserts `status in {"WARN", "FAIL"}` on ~50% total growth; tripling the signal provides margin above the WARN/FAIL cut-off while staying in the range the rule's floor was calibrated for.

## Acceptance criteria

- Fixture-sanity: per-step ratio of the analytical eigenmode matches `exp(-4π² · 0.05) ≈ 0.1389` within `ε_quad = 10⁻⁴` (independent of the rule).
- Analytical heat sequence passes the rule with `status == "PASS"`.
- Non-dissipative control produces `status in {"WARN", "FAIL"}` (tri-state classification depends on calibrated floor; both outcomes are informative failures).
- Wall-time < 10 s on CPU.

## Dependency: rule-source forward-difference primitive

This anchor's κ=1 pinned value presupposes the rule uses a forward-difference `dE/dt` primitive (`np.diff(energy)/dt`). The Rev 1.6 rule used `np.gradient(energy, dt, edge_order=2)`, whose 2nd-order backward-quadratic extrapolation at the last timestep cannot track exponential decay whose range exceeds ~50× over the measurement window, and produced a spurious positive dE/dt at `t = 0.2` on this exact fixture (see the 2026-04-20 Task 2 execution trace). The rule was upgraded to the forward-difference primitive in the same commit window as this anchor; the Rev 1.6 primitive would reject the κ=1 setup above even though it is the textbook Evans 7.1.2 reproduction.
