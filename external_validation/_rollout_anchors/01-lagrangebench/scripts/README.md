# `01-lagrangebench/scripts/`

Operator-side scripts for inspecting, validating, and re-converting
LagrangeBench rollout artifacts on the
``rollout-anchors-artifacts`` Modal Volume.

## Scripts

- **`spot_check_rollout.py <subdir_name>`** — Pulls
  ``particle_rollout_traj00.npz`` from a Volume rollout subdir to a
  local cache, loads it via ``particle_rollout_adapter.load_rollout_npz``,
  reports velocity range / PBC audit fields / KE series / harness
  defects, and emits a PASS/FAIL verdict against the load-bearing
  cross-repo contract (D0-17 periodic correction applied + D0-17
  amendment 1 PBC audit fields populated + D0-18 dissipative-system
  SKIP path firing where applicable). Idempotent; cache at
  ``/tmp/physics_lint_spot_check/``.

  Reference invocations against the rollouts captured in DECISIONS
  D0-16 + D0-18 amendment 1::

      .venv/bin/python ...scripts/spot_check_rollout.py segnn_tgv2d_8c3d080397
      .venv/bin/python ...scripts/spot_check_rollout.py gns_tgv2d_f48dd3f376

## Conventions

Scripts here are operator-side helpers, not part of the test surface
(``tests/`` covers that). They assume Modal CLI is authenticated and
the user has read access to ``rollout-anchors-artifacts``. Caches go
to ``/tmp/`` so the repo doesn't pollute with binary artifacts that
are deterministically re-pullable from the Volume.

Sibling ``02-physicsnemo-mgn/scripts/`` lands when Day 2 work begins.
