# Ahmed Body MeshGraphNet inference-protocol audit (P2.2)

**Date:** 2026-05-19
**Purpose:** P2.2 - determine whether a PH-BC-001 no-slip velocity BC check on
the Ahmed Body MGN checkpoint is non-degenerate, or degenerate like the CS02
cylinder checkpoint.
**Upstream pin:** NVIDIA/physicsnemo @ 1ca85d65ac2ce28ea9762910c09a954c08a37140
(v2.0.0), examples/cfd/external_aerodynamics/aero_graph_net/.

## What the Ahmed Body MGN predicts

The Ahmed Body MeshGraphNet predicts **surface pressure and wall shear stress**
on the body-surface mesh, plus a scalar drag coefficient - it does not predict
a velocity field of any kind.

Evidence from `inference.py` (the inference entrypoint, run with
`+experiment=ahmed/mgn`):

- `inference.py:144` - `pred = self.model(graph.x, graph.edge_attr, graph)`:
  the model produces one prediction tensor over the mesh nodes.
- `inference.py:148` - `num_out_c = gt.shape[1]`: the output-channel count is
  taken from the ground-truth target tensor, and is one of `{1, 3, 4}`.
- `inference.py:149-151` - `if num_out_c in [1, 4]: graph.p_pred = pred[:, 0]`:
  channel 0 is pressure `p`.
- `inference.py:152-154` -
  `if num_out_c in [3, 4]: graph.wallShearStress_pred = pred[:, num_out_c-3:]`:
  the last three channels are wall shear stress `wallShearStress`.
- `inference.py:70-73` - the saved `.vtp` carries exactly
  `p_pred`, `p`, `wallShearStress_pred`, `wallShearStress`; no velocity field.

The model class confirms this. `models.py:43-48` defines `AeroGraphNet`, "A
variant of MeshGraphNet model that also predicts a drag coefficient", and its
`forward` (`models.py:82-94`) returns `{"graph": x, "c_d": c_d}` - per-node
decoder output `x` (the pressure / WSS channels) and a scalar drag coefficient
`c_d` from `c_d_decoder` (`models.py:72-80`, `output_dim=1`).

The Ahmed experiment config agrees. `conf/experiment/ahmed/mgn.yaml:46-68`
declares exactly two visualizers - `mesh_p` (`scalar: p`) and `mesh_wss`
(`scalar: wallShearStress`) - and the test split sets `compute_drag: true`
(`conf/experiment/ahmed/mgn.yaml:41`). The predicted quantities are surface
pressure, wall shear stress, and drag. There is no velocity output.

## No-slip body-surface nodes: predicted, masked, or not velocity

The Ahmed Body inference loop (`inference.py:133-162`) applies no node masking
at all: every mesh node receives a predicted value, with no conditional logic
holding any node-type fixed. This contrasts with the CS02 cylinder checkpoint,
whose rollout loop `mgn_rollout_p0_vortex_shedding` in `modal_app.py` freezes
boundary nodes via `v_diff_masked = torch.where(mask2, pred_i_velo, zeros)`
then `v_next = v_diff_masked + invar[:, 0:2]`, pinning wall nodes at their
step-0 ground-truth velocity (0, no-slip).

But the masking question is moot for the Ahmed Body MGN: the quantity it
predicts on the body surface is **not velocity**. There are no "no-slip
body-surface velocity nodes" to predict or mask - the surrogate represents the
body surface through pressure and wall shear stress, the quantities a steady
aerodynamic drag surrogate needs. A no-slip velocity boundary condition
`||v_wall - 0||` has no predicted field to evaluate against.

## Verdict

Outcome N: the Ahmed Body MGN outputs surface pressure and wall shear stress
(plus a drag coefficient), not a body-surface velocity field, so a
PH-BC-001-style no-slip *velocity* BC check is inapplicable to it - degenerate
for a different structural reason than the cylinder checkpoint's masking.

Consequence for P2.2: the mesh wall-node velocity-BC capability-build has no
home among the current PhysicsNeMo MGN targets - the cylinder checkpoint masks
its wall velocity, the Ahmed Body checkpoint predicts no velocity at all - and
is deferred. A pressure / wall-shear-stress surface check would be a different
rule, outside P2.2 and P4.1 scope.
