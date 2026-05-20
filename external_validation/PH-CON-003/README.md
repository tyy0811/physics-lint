# PH-CON-003 external-validation anchor

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-CON-003/ -v
```

Expected: 3 passed in < 10 s.

## Rule reference

PH-CON-003 checks the classical parabolic energy estimate for the heat
equation: under boundary conditions that dissipate (homogeneous Dirichlet,
homogeneous Neumann, or periodic), $\int_\Omega u^2\, dx$ must be
monotonically non-increasing in time. The rule reads $u(t)$ across the
provided time samples, integrates $u^2$ at each step, and computes the
forward-difference slope $dE/dt$. The raw value is the maximum positive
slope, normalized by the peak energy so the reported quantity is
dimensionless.

The rule fires (`FAIL` or `APPROXIMATE` against the calibrated noise
floor) when the surrogate produces a positive $dE/dt$ — typically a
symptom of a non-physically-faithful integrator or a learned residual
that injects energy. Forward differences are used in place of
`np.gradient` because the latter's edge-order-2 endpoint extrapolation
introduces spurious positive $dE/dt$ on strictly dissipative analytical
solutions.

The rule emits `SKIPPED` when the PDE is not heat, when the configured
boundary condition is not energy-dissipating, or when fewer than two
time samples are available (so no forward-difference $dE/dt$ can be
formed).
