# `_rollout_anchors/methodology/`

Methodology trail, design docs, and decision history for the
rollout-anchors validation work — the project of demonstrating
physics-lint's rules against real ML model rollouts (LagrangeBench /
SEGNN, GNS; PhysicsNeMo / MeshGraphNet; ...).

## Files

- **`DECISIONS.md`** — Cross-rung methodology trail (D0-01..D0-18+).
  Single source of truth for *why* each methodology choice was made.
  Physics-lint commits in this subtree reference D-entries by number
  for attribution. Read top-down for chronological story; the
  cumulative-state summary at the bottom ("Day 0+ status") gives the
  current shape.

- **`docs/`** — Plans (`physics-lint-validation-plan-v2.md`), dated
  design docs (`2026-05-01-rollout-anchor-extension-{design,plan}.md`),
  and writeups in progress.

## Scope

This methodology subtree covers the rollout-anchors validation work
across all case studies (`01-lagrangebench/`, future
`02-physicsnemo-mgn/`, ...). DECISIONS entries that touch a specific
case study reference its directory; entries that span multiple case
studies (cross-rung gates, JAX micro-gate, NGC audit, etc.) live here
without case-study attribution.

Per-case-study operational outputs (verdict logs, scripts) live with
the case-study code, not here. E.g., `01-lagrangebench/outputs/verdicts/`
captures Modal-fire audit logs adjacent to `modal_app.py` that
produced them; this README + DECISIONS reference those logs by
relative path.

## Why methodology lives in physics-lint, not in a sibling
"physics-lint-validation" repo

Earlier in the project, methodology + scripts + audit logs lived in a
sibling `physics-lint-validation` repo, while the harness + Modal
entrypoint + rule anchors lived here. That split created recurring
audit-trail-hygiene gaps: DECISIONS would cite paths that crossed the
repo boundary; verdict logs got committed to the wrong side and
referenced by the right side; updates to one repo's contents required
coordinated edits in the other. The split was also misleading about
what physics-lint-validation actually was — the repo's name implied
"a validation library that imports physics-lint," when in fact it was
"the methodology + audit + application materials repo for the project
of using physics-lint."

The migration that brought methodology + scripts + audit logs into
physics-lint co-locates each piece with the harness it documents.
Application materials (cover letters, BMW-thesis context, cross-review
feedback) stay in a small private repo separately, since those are
neither methodology nor technical work and don't belong in a public
linter repo.

## Adding a new D-entry

Same shape as existing entries: D0-XX (or DN-XX for milestone N),
date, one-sentence title, **Question** (what the choice point is),
**Decision** (what was chosen + the pre-registered evidence), other
sections per the existing pattern (Realized, Forward agenda, etc.).
Reference the physics-lint commit SHA where the decision was realized.
Keep amendments inline within their parent D-entry rather than
chaining new D-entries — the scope discriminator is "within-pre-
registered-scope refinement → amendment, out-of-scope discovery → new
entry" (see D0-17 vs D0-15 amendment 5 framing for the canonical
example).
