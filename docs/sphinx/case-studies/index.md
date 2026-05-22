# Case studies

physics-lint's rule methodology and SARIF schemas are validated against
real published model checkpoints (the "borrowed credibility" methodology —
see [PH-SYM-001](../rules/PH-SYM-001.md) and
[PH-CON-001](../rules/PH-CON-001.md) for the rule-level pattern). Both case
studies below sit on substrates the shipped CLI and Action do not yet
ingest — particle clouds and unstructured meshes — so each runs through a
research harness; CLI/Action support for those substrates is planned for
v1.2.0.

```{toctree}
:maxdepth: 1

01-lagrangebench
02-physicsnemo-mgn
```

## Summary

| Case study | Substrate | Stack | Headline result |
|---|---|---|---|
| [CS01 LagrangeBench](01-lagrangebench.md) | TGV2D, Dam-break, Reverse Poiseuille | JAX, particle-based GNNs (GNS, SEGNN) | Equivariance gap detected on published GNS/SEGNN checkpoints (PH-SYM-001/002 methodology) — particle substrate, run via research harness; native Action support planned for v1.2.0 |
| [CS02 PhysicsNeMo MGN](02-physicsnemo-mgn.md) | Cylinder Flow | PyTorch / PhysicsNeMo, mesh GNN (MGN) | Mass-conservation gap 5.857% GT / 5.881% MGN (PH-CON-001 structural identity, via mesh harness); PH-BC-001 structurally degenerate (D0-27) |
