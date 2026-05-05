# In-tree docs snapshots

This directory holds **frozen snapshots** at merge time per design-spec
§1.5 option (b). Until merge, the live documents iterate in the working
repo `physics-lint-validation/`:

- `2026-05-01-rollout-anchor-extension-design.md` — the design spec.
- `2026-05-01-rollout-anchor-extension-plan.md` — the implementation plan.
- `DECISIONS.md` — the in-flight decision log.

At merge time, the as-merged versions of all three are copied here and
become the authoritative reference for anyone reading the merged
subdirectory after the PR lands. The merged artifact is therefore
self-contained — physics-lint takes no cross-repo dependency on
physics-lint-validation's stability or layout.
