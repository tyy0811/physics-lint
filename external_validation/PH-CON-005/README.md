# PH-CON-005 external-validation anchor

## Rule reference

PH-CON-005 is a 2D graph-mesh diagnostic for the incompressibility
(divergence-free) condition of an incompressible Navier-Stokes velocity
field. It operates on `MeshVectorField` inputs from `physics_lint[mesh]`
(scikit-fem `MeshTri` + `ElementTriP1`), built from `node_positions` +
`cells` + a `(T, N, 2)` velocity. For each timestep it computes the
dimensionless Frobenius-normalized divergence defect

    defect = max over t of  ( integral |div v| dV ) / ( integral ||grad v||_F dV )

on the P1 finite-element basis, porting the validated harness
`_fe_divergence_defect_max`. The raw value is this unitless ratio.

**Emit-only — never gates.** The rule emits `status=PASS`, `severity=info`,
`raw_value=defect`. It never trips `exit_code` and never degrades
`overall_status` (the PH-VAR-002 pattern). There is no absolute PASS/FAIL
band: a discrete velocity field carries an O(h) divergence defect of its own
(the CS02 ground truth itself is approximately 5.8% at FE-interpolation
resolution), so an absolute band would fail correct data. The signal is the
scalar — and a ground-truth-vs-model comparison the user runs — not a verdict.
On every surface the scalar travels with the result; in SARIF it is a
`note`-level `toolExecutionNotification` (not a `run.results` finding, so it
never raises a code-scanning alert).

**Calibration is port fidelity, not a band.** PH-CON-005's credibility is the
pre-registered Gate-B check: its `raw_value` reproduces the harness
`_fe_divergence_defect_max` on the same input within epsilon <= 1e-4 (measured
0 on a controlled fixture — same formula, same basis).

The rule emits `SKIPPED` on non-`MeshVectorField` inputs; the central
field-type applicability filter SKIPs it on grid/callable targets before
`check()` runs. 2D only in v1.2.0 (`MeshTri` / `ElementTriP1` / cells (M, 3));
3D (`MeshTet`), the scalar `MeshField` loader, and the energy-drift /
dissipation-sign NS defects are follow-ons. Requires `physics-lint[mesh]`.

## Validation harness

PH-CON-005 ports the graph-mesh divergence kernel from the CS02 research
harness (`external_validation/_rollout_anchors/_harness/mesh_rollout_adapter.py`,
`_fe_divergence_defect_max`). The port-fidelity (Gate-B) test lives at
`external_validation/_rollout_anchors/_harness/tests/test_ph_con_005_vs_harness.py`
and asserts the public rule reproduces the harness within epsilon <= 1e-4 on a
non-trivial fixture. Analytic unit fixtures (linear velocity, where P1 is exact)
anchor the kernel directly: `v = (x, -y)` is divergence-free, so defect is
approximately 0; `v = (x, 0)` has `div v = 1` and `||grad v||_F = 1`, so defect
is exactly 1.0. See `tests/rules/test_ph_con_005.py`.
