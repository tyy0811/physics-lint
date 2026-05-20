# Ahmed Body MeshGraphNet inference-protocol audit (P2.2)

**Date:** 2026-05-19
**Purpose:** P2.2 - determine whether a PH-BC-001 no-slip velocity BC check on
the Ahmed Body MGN checkpoint is non-degenerate, or degenerate like the CS02
cylinder checkpoint.
**Upstream pin:** NVIDIA/physicsnemo @ 1ca85d65ac2ce28ea9762910c09a954c08a37140
(v2.0.0), examples/cfd/external_aerodynamics/aero_graph_net/.

## What the Ahmed Body MGN predicts

The Ahmed Body MeshGraphNet predicts **surface pressure and wall shear stress**
on the body-surface mesh - it does not predict a velocity field of any kind.
The drag coefficient that the experiment reports is computed downstream from
the predicted pressure / WSS together with mesh normals and areas in the test
data pipeline (`inference.py:134` unpacks `normals, areas, coeff` from the
batch, and `conf/experiment/ahmed/mgn.yaml:41` enables `compute_drag: true` on
the test split); it is not a model output.

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

The model wired by `+experiment=ahmed/mgn` confirms this. The Hydra defaults
in `conf/experiment/ahmed/mgn.yaml:23` select `/model: mgn`, which resolves to
`conf/model/mgn.yaml`; that config sets
`_target_: physicsnemo.models.meshgraphnet.MeshGraphNet`
(`conf/model/mgn.yaml:17`) and `output_dim: 4` (`conf/model/mgn.yaml:22`). The
runtime model is plain `MeshGraphNet` producing four channels - one pressure +
three wall-shear-stress components, exactly what `inference.py:149-154`
unpacks. The same example also defines a separate `AeroGraphNet` variant in
`models.py` that adds a drag-coefficient head, but it is not the model
instantiated by `+experiment=ahmed/mgn`; this distinction does not affect the
no-velocity verdict.

The Ahmed experiment config agrees. `conf/experiment/ahmed/mgn.yaml:46-68`
declares exactly two visualizers - `mesh_p` (`scalar: p`) and `mesh_wss`
(`scalar: wallShearStress`); the test split's `compute_drag: true`
(`conf/experiment/ahmed/mgn.yaml:41`) is what wires the downstream drag
computation from those two predicted fields. The predicted quantities are
surface pressure and wall shear stress. There is no velocity output.

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

Outcome N: the Ahmed Body MGN outputs surface pressure and wall shear stress,
not a body-surface velocity field, so a PH-BC-001-style no-slip *velocity* BC
check is inapplicable to it - degenerate for a different structural reason
than the cylinder checkpoint's masking.

Consequence for P2.2: the mesh wall-node velocity-BC capability-build is
**retired from P2.2** - it has no home among the current PhysicsNeMo MGN
targets (the cylinder checkpoint masks its wall velocity, the Ahmed Body
checkpoint predicts no velocity at all). Any future velocity-BC work requires
a new target and a fresh decision entry. A pressure / wall-shear-stress
surface check would be a different rule, outside P2.2 and P4.1 scope.
