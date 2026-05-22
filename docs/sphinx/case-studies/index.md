# Case studies

physics-lint runs against real published model checkpoints to validate
that the rule set fires on shipped artifacts (the "borrowed credibility"
methodology — see [PH-SYM-001](../rules/PH-SYM-001.md) and
[PH-CON-001](../rules/PH-CON-001.md) for the rule-level pattern).

```{toctree}
:maxdepth: 1

01-lagrangebench
02-physicsnemo-mgn
```

## Summary

| Case study | Substrate | Stack | Headline result |
|---|---|---|---|
| [CS01 LagrangeBench](01-lagrangebench.md) | TGV2D, Dam-break, Reverse Poiseuille | JAX, particle-based GNNs (GNS, SEGNN) | Equivariance gap detected on published GNS/SEGNN checkpoints (PH-SYM-001/002 methodology) — particle substrate, run via research harness; native Action support in v1.2.0 |
| [CS02 PhysicsNeMo MGN](02-physicsnemo-mgn.md) | Cylinder Flow | PyTorch / PhysicsNeMo, mesh GNN (MGN) | PH-CON-001 mass conservation gap 5.857% GT / 5.881% MGN; PH-BC-001 structurally degenerate (D0-27) |
