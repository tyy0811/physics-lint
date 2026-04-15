# Verification runlog

Append-only record of verification gates — what passed, on which SHA, with
which commands. Not a release-history file; tags handle that. This log is
for checkpoint-SHA questions like "when did dogfood criterion 3 last pass
end-to-end" that should not pollute the tag namespace.

Convention: one entry per verification gate, newest first, with enough
detail that `git checkout <sha>` reproduces the state. SHAs recorded here
are the commit that *was verified*, not the commit that added the entry.

## Week-2 verification — 2026-04-15

- SHA: 8ec69c1 (Week-2 branch, pre-merge)
- pytest: 210 passed (+2 regression on 208 baseline)
- ruff check + format: clean
- scripts/smoke_self_test.py: PASS
- dogfood/run_dogfood.py: criterion 3 PASS (ranking unchanged)
- dogfood/laplace_uq_bench/run_regime_comparison.py: completes
- dogfood/clone_laplace_uq_bench.sh: HEAD == pinned SHA verified
