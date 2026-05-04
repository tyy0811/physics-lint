# Rollout-domain F3 anchor

Two third-party case studies extending physics-lint's `external_validation/`
framework with a rollout-domain F3 anchor — populating a new "rollout-domain
F3 status" column in the 18-rule anchor matrix for the six v1.0 rules where
the anchor is meaningful (plus `PH-RES-001` as a P2 stretch on the mesh side
only).

The design spec and implementation plan are snapshotted into [`docs/`](docs/)
at merge time per design-spec §1.5 option (b); the merged subdirectory is
self-contained and takes no cross-repo dependency on the working repo
`physics-lint-validation/` where the planning artifacts iterated pre-merge.

## 1. Scope — explicit out-of-scope statement (spec §1.1, verbatim)

> *"This branch does not modify `physics_lint.field.*` public API. Any
> extension to the Field protocol — temporal axis, particle support,
> MeshField widening to non-scikit-fem mesh sources — is out of scope.
> Such extensions, if needed, would land on a separate
> `feature/field-protocol-v1.1` branch sequenced independently after v1.0
> ships. Stub resolution for PH-SYM-004 / PH-BC-002 / PH-NUM-001 is out
> of scope and lives on the parallel `chore/v1.0-stub-resolution`
> branch."*

The negative statement is load-bearing: it stops scope creep on Day 2 when
the temptation to "just widen MeshField a bit" appears.

## 2. Headline claim

*[To be finalised at writeup time per spec §0 / §3.3 / Appendix A; the
trimmed structural-identities-held framing replaces v2's "ran without rule
modification across two completely different stacks" claim.]*

On the mesh side (NVIDIA PhysicsNeMo MeshGraphNet), the existing public
Field/rule API consumes per-timestep materializations of trained
third-party output without rule modification. On the particle side
(LagrangeBench SEGNN/GNS), the rule API does not natively accept particle
clouds; the rule structural identities — finite-group equivariance for
`PH-SYM-001`/`002`, conservation balance for `PH-CON-001`/`002`/`003` — are
reapplied via a thin private harness validated against analytical fixtures
to within ε_harness_vs_public ≤ 10⁻⁴ (Gate B). Both paths emit SARIF in
the same schema.

## 3. Case studies

### 01 — LagrangeBench (TUM, NeurIPS 2023)
*[Day 1 deliverable; see [`01-lagrangebench/README.md`](01-lagrangebench/README.md).]*

### 02 — NVIDIA PhysicsNeMo MeshGraphNet (NGC pretrained)
*[Day 2 deliverable; substituted by FNO-on-Darcy under spec §6 Gate D
fallback. See [`02-physicsnemo-mgn/README.md`](02-physicsnemo-mgn/README.md).]*

## 4. Cross-stack consistency

*[Day 3 deliverable. Table: rule × case study × path (public-API or
harness) × result.]*

## 5. What physics-lint did NOT catch

This section is non-negotiable per plan §5.1; it is the methodological-honesty
signal that matters for Audience A reviewers (Munich/Stuttgart SciML).

- physics-lint's public Field API does not natively accept particle clouds;
  the LagrangeBench path uses a private harness that *reapplies* the rule's
  structural identity, not the public rule itself. Validated against
  analytical fixtures to ε_harness_vs_public ≤ 10⁻⁴, but not equivalent to
  a public-API run.
- The rotation-sweep ε_rot computation does not match `PH-SYM-003`'s
  emitted quantity. `PH-SYM-003` is infinitesimal scalar Lie-derivative;
  the harness computes global-finite multi-output equivariance. The
  harness emits a different quantity than `PH-SYM-003` and labels it
  accordingly via SARIF `properties.source = "rollout-anchor-harness"`.
- `PH-CON-001` as shipped in physics-lint v0.0.0.dev0 returns SKIPPED on
  `pde != "heat"`. On the NS side (PhysicsNeMo MGN vortex shedding), the
  harness reapplies the structural mass-conservation identity via the
  same mechanism used for the particle-side rules — this is a
  structural-identity reapplication, not a public-API rule invocation.
  Extending `PH-CON-001`'s V1 implementation to NS-applicable input
  domains is a separate physics-lint v1.0-resolution task and is out of
  scope for this branch.
- Plasticity / irreversibility rules are not yet implemented (PH-CSH-*
  roadmap, separate issue).
- Contact-non-penetration on deforming meshes is not tested (no public
  checkpoint exists).
- Equivariance tests are statistical (over a finite set of rotation
  angles); they cannot prove equivariance, only fail to disprove it.
- LagrangeBench is fluid SPH, not solid impact — domain transfer is
  implied, not demonstrated.

## 6. Reproducibility

*[Day 3 deliverable. Modal entrypoints, checkpoint hashes, git SHA, conda
lockfile.]*

## 7. Citations

- Toshev et al. — LagrangeBench, NeurIPS 2023.
- Pfaff et al. — MeshGraphNets, ICLR 2021.
- Nabian et al. — arXiv:2510.15201, BMW/GM-style crash with PhysicsNeMo.
- Lahoz Navarro & Jehle et al. — Applied Sciences 2024 (BNN UQ, framed
  here as the deterministic-violation gate complementing UQ).

## 8. Branch / status

- Branch: `feature/rollout-anchors` (this branch).
- Parallel out-of-scope branches per spec §2.4: `chore/v1.0-stub-resolution`,
  conditional `feature/field-protocol-v1.1`.
- Day 0 status: *[populated by `docs/DECISIONS.md` at merge time.]*
