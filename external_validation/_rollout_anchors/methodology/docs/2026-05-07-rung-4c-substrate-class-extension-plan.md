# Rung 4c — Substrate-class extension to dam-break-2D (implementation plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend physics-lint's harness substrate-detection layer to a second LagrangeBench substrate class (`open-driven-dissipative`) by reclassifying `dam2d` empirically and adding a new SKIP path on `dissipation_sign_violation`. Produce 2 dam-break SARIFs + cross-stack table + writeup, mirroring rung-4a's TGV2D pattern on a new substrate.

**Architecture:** D0-22 lands in DECISIONS.md before any code. Code edits bounded to `_harness/particle_rollout_adapter.py` (taxonomy + SKIP gate) + `01-lagrangebench/modal_app.py` (2 dam2d rollout functions). Renderer untouched (load-bearing evidence of substrate-agnosticism). Pre-flight 5-step checklist gates Modal compute. Total Modal budget: < $0.20 USD.

**Tech Stack:** Python 3.10, JAX 0.4.29 + LagrangeBench (Modal-side, A10G), numpy + pytest (CPU-side), Modal Volume (artifact persistence), SARIF v1.0 (committed schema unchanged).

**Design doc:** [`./2026-05-07-rung-4c-substrate-class-extension-design.md`](./2026-05-07-rung-4c-substrate-class-extension-design.md) — read §1.2 (frozen headline + pre-flight gating-condition), §1.4 (D0-22), §3.3 (pre-flight 5-step), and §6 (test fixtures) before starting.

---

## Branch setup

Working on PR #8's branch tip (`feature/rung-4b-t7-subseq-length-fix` at sha `861c95c`); rebase to `master` post-PR-#8-merge.

- [ ] **Step A1: Verify clean working tree on PR #8 branch tip**

```bash
cd /Users/zenith/Desktop/physics-lint
git status
git rev-parse --abbrev-ref HEAD
```

Expected: working tree clean; branch `feature/rung-4b-t7-subseq-length-fix`.

- [ ] **Step A2: Create rung-4c branch off PR #8 tip**

```bash
cd /Users/zenith/Desktop/physics-lint
git checkout -b feature/rung-4c-substrate-class-extension
git rev-parse --abbrev-ref HEAD
```

Expected: branch `feature/rung-4c-substrate-class-extension`.

- [ ] **Step A3: Verify pytest baseline passes**

```bash
source .venv/bin/activate
pytest --import-mode=importlib external_validation/ -o "addopts=" 2>&1 | tail -5
```

Expected: `447 passed, 1 skipped` (rung-4b T7 PASS state).

---

## Task 1: Pre-register D0-22 in DECISIONS.md

**Files:**
- Modify: `external_validation/_rollout_anchors/methodology/DECISIONS.md` (append D0-22 entry at end)

This task lands the methodology pre-registration **before any code change**, mirroring D0-18/D0-19/D0-21 sequencing. No tests; no code change. Just the append + commit.

- [ ] **Step 1.1: Append D0-22 entry to DECISIONS.md**

Open `external_validation/_rollout_anchors/methodology/DECISIONS.md`. After the existing D0-21 entry (find the last line of the file), append:

````markdown

## D0-22 — 2026-05-07 — Rung 4c substrate-class extension to dam-break-2D (pre-registration)

**Status:** pre-registered before code change.
**Predecessor:** D0-21 (rung-4b cross-stack equivariance), D0-18 (dissipative-system skip-with-reason), D0-19 (harness SARIF result schema), D0-08 (KE-rest skip on `energy_drift`).
**Trigger:** rung-4c design pass (`methodology/docs/2026-05-07-rung-4c-substrate-class-extension-design.md`); source review of `_harness/particle_rollout_adapter.py` surfaced that `LAGRANGEBENCH_DATASET_SYSTEM_CLASS["dam2d"]` was preemptively classified as `"dissipative"` during the D0-18 design pass, before any dam-break-2D rollout was measured.

**Decision.** Extend the harness's substrate-detection layer to a second LagrangeBench substrate class — `open-driven-dissipative` — via:

1. **Taxonomy bump.** `LAGRANGEBENCH_DATASET_SYSTEM_CLASS` shifts from implicit-binary (`"dissipative"` or absent → fire-raw) to explicit-tri-state at minimum: `"dissipative"`, `"open-driven-dissipative"`, absent. Future taxonomy entries (e.g., `"strictly-conservative"` for case study 02 anchor) are forward-compatible.

2. **`dam2d` reclassified empirically** from `"dissipative"` to `"open-driven-dissipative"`. Justification: dam-break-2D's KE rises during gravity-loaded fall (gravitational PE → KE conversion), violating the closed-dissipative class's monotone-non-increasing precondition. Empirical verification via 1-traj Modal smoke at pre-flight step 4 (rung-4c design §3.3); reclassification commits *after* smoke confirms the rise-then-fall shape, not before.

3. **New SKIP path on `dissipation_sign_violation`** for `system_class == "open-driven-dissipative"`. The rule's strictly-dissipative-or-conservative assumption (docstring-stated zero-for-strictly-dissipative-or-conservative behavior) doesn't apply to open-driven systems where `dE/dt > 0` over a stretch by physics. SKIP-reason template:

   ```
   system_class='open-driven-dissipative' (dataset='<name>'); dE/dt > 0
   over a stretch by physics (gravitational PE → KE conversion); the
   strictly-dissipative-or-conservative assumption underpinning
   dissipation_sign_violation does not apply. See DECISIONS.md D0-22.
   ```

4. **Forward-flag two-tier split** for catalogue-wide reclassification:
   - **Almost certainly misclassified, awaiting empirical probe:** `rpf2d` (reverse Poiseuille, forced flow), `ldc2d` (lid-driven cavity, forced flow), `rpf3d`, `ldc3d` (3D variants).
   - **Likely correctly classified but unverified:** `tgv3d` (3D-TGV inherits 2D-TGV physics; closed dissipative system, KE monotone-non-increasing as turbulent KE decays).
   The two-tier split preserves future-rung actionability: a rung exercising rpf or ldc walks into a known-misclassification (and the empirical probe is its first move); a rung exercising tgv3d treats it as inherited-from-validated.

5. **D0-08 (KE-rest IC SKIP) is NOT absorbed into D0-22.** D0-08 gates on `KE(0) < KE_REST_THRESHOLD`, orthogonal to the `system_class` taxonomy. D0-22 gates on `system_class == "open-driven-dissipative"`. The two SKIP gates are sibling members of a "substrate-detection family" — distinct physical preconditions, distinct SKIP reasons. On dam-break-2D specifically, D0-08 fires on `energy_drift` (start-at-rest IC) and D0-22 fires on `dissipation_sign_violation` (open-driven by physics); both fire on the same rollout but on different rules.

6. **(rule, substrate) compatibility matrix forward-flag from D0-21 cited but not promoted.** D0-21 §forward-flag-2 named "the (rule, substrate) compatibility matrix as a future generalization." Rung 4c demonstrates that pattern's empirical instance (`dissipation_sign_violation × open-driven-dissipative` becomes a new SKIP cell) without yet promoting the matrix to a first-class rule-schema field. Promotion is post-rung-4c work.

**"Classify when you exercise" empirical-classification principle.** Named here as a first-class methodology output of the rung-4 series, generalizing the precedent set by rung-4b's PH-SYM-003 PBC-square-SO(2) substrate-incompatibility SKIP (classified only for the substrate measured, not retrospectively over all (rule, substrate) combinations). Pattern reads as: **substrate properties get verdicts only after empirical probing, never on theoretical intuition alone, even when the theoretical guess is almost certainly correct.** Applies to any future rung exercising a previously-unmeasured substrate.

**Source-review-catches-issue-before-compute pattern (now trilateral).**
1. Rung 4b first-pass math correction (TRAIN_PUSHFORWARD_UNROLLS_LAST=3 conflated +1 target with pushforward count) — caught at LB source review at sha `b880a6c`.
2. Rung 4b first-pass latent figure-sweep failure (`valid.h5` hardcoded `subseq_length=10` vs. dynamic upstream value) — caught at the same source review pass.
3. Rung 4c catalogue-misclassification (`dam2d → "dissipative"` was preemptive and wrong; `rpf2d`/`ldc2d` likely the same) — caught at source review of `particle_rollout_adapter.py` during this design pass.

Three instances at $0 Modal cost. To be elevated as a first-class methodology contribution of the rung-4 series in integrating-README composition rather than three scattered amendments. The pattern reads as: **a source-review pre-flight pass between design and execution catches issues that brainstorm-only and execution-only review miss; the cost is hours of source reading, the saving is multiple GPU runs and methodology errors that would otherwise land in writeups.**

**Forward flags carried into D0-22:**
- Bilateral D0-18 still requires a strictly conservative substrate anchor (case study 02 territory); rung 4c does not exercise this.
- PH-SYM substrate-symmetry-SKIP (gravity-direction-pinning breaks SO(2) on dam-break-2D) → rung 4d.
- PH-BC-particle-wall not in v1.0 rule path → out of scope; plan v2.1 amendment removes "PH-BC (wall)" from §3.1 P1.
- Catalogue-wide reclassification (`rpf2d`/`ldc2d`/`rpf3d`/`ldc3d`) deferred to future rungs; D0-22 records the two-tier split for actionability.

**Implementation:** rung-4c plan (`methodology/docs/2026-05-07-rung-4c-substrate-class-extension-plan.md`). Acceptance criteria in design §8.

````

- [ ] **Step 1.2: Verify markdown structure parses cleanly**

```bash
cd /Users/zenith/Desktop/physics-lint
grep -c "^## D0-" external_validation/_rollout_anchors/methodology/DECISIONS.md
```

Expected: 8 (D0-15 through D0-22, with amendments anchored under their parent entries). If the count is unexpected, inspect the file structure manually.

- [ ] **Step 1.3: Commit D0-22 pre-registration**

```bash
cd /Users/zenith/Desktop/physics-lint
git add external_validation/_rollout_anchors/methodology/DECISIONS.md
git commit -m "methodology/DECISIONS: pre-register D0-22 — rung 4c substrate-class extension to dam-break-2D"
```

Expected: clean commit. Post-commit: `git log --oneline -1` shows the new commit.

---

## Task 2: SKIP-reason template documentation in SCHEMA.md

**Files:**
- Modify: `external_validation/_rollout_anchors/_harness/SCHEMA.md` (find the existing harness-rule SKIP-reason templates section; add D0-22 entry)

- [ ] **Step 2.1: Locate the existing SKIP-reason template section**

```bash
grep -n "skip_reason\|SKIP\|D0-18\|D0-08" external_validation/_rollout_anchors/_harness/SCHEMA.md | head -20
```

The SCHEMA.md should have a section documenting the existing D0-18 (energy_drift dissipative SKIP) and D0-08 (energy_drift KE-rest SKIP) templates. Append D0-22 in the same shape.

- [ ] **Step 2.2: Add D0-22 SKIP-reason template entry**

Edit `external_validation/_rollout_anchors/_harness/SCHEMA.md`. After the existing D0-18 template documentation, add:

```markdown

### `dissipation_sign_violation` SKIP-reason template (D0-22)

Fires when `metadata["dataset"]` resolves to `system_class == "open-driven-dissipative"` via `LAGRANGEBENCH_DATASET_SYSTEM_CLASS`. Emitted as `properties.skip_reason` per D0-19 §3.4.

**Template (verbatim):**

```
system_class='open-driven-dissipative' (dataset='<dataset_name>'); dE/dt > 0 over a stretch by physics (gravitational PE → KE conversion); the strictly-dissipative-or-conservative assumption underpinning dissipation_sign_violation does not apply. See DECISIONS.md D0-22.
```

The `<dataset_name>` placeholder is interpolated from `rollout.metadata["dataset"]` at emission time; the rest of the template is identical across (rule, stack) emissions per D0-19 §3.4.

**Trigger gate:** `system_class == "open-driven-dissipative"` AND existing D0-08 KE-rest gate did not fire (max(KE) > KE_REST_THRESHOLD). Orthogonal to KE-shape conditions; the gate is purely `system_class`-conditioned by design (an open-driven system with monotone-decreasing KE due to a model bug should still SKIP — surfacing the bug requires a different rule, not this one).

**Sibling-gate relationship.** D0-22 (dissipation_sign_violation, open-driven) and D0-18 (energy_drift, dissipative-monotone) and D0-08 (energy_drift, KE-rest) are members of a "substrate-detection family" with distinct physical preconditions and distinct SKIP-row outputs. Future generalization to a (rule, substrate) compatibility matrix is forward-flagged in D0-21 §forward-flag-2 and D0-22 §6.
```

- [ ] **Step 2.3: Stage SCHEMA.md (do not commit yet)**

The SCHEMA.md edit will commit alongside the implementation in Task 4. Stage it now:

```bash
cd /Users/zenith/Desktop/physics-lint
git add external_validation/_rollout_anchors/_harness/SCHEMA.md
git status
```

Expected: `external_validation/_rollout_anchors/_harness/SCHEMA.md` shown as staged.

---

## Task 3: TDD red — write D0-22 SKIP-gate tests

**Files:**
- Create: `external_validation/_rollout_anchors/_harness/tests/test_d0_22_open_driven_skip.py`

This task writes the failing tests for D0-22's SKIP gate before any production-code change. Per the test_fixtures_hand_crafted_not_copied discipline, fixtures are hand-crafted with synthetic-but-realistic shapes (rise-then-fall KE for the positive path, monotone for negatives).

- [ ] **Step 3.1: Create test file**

Create `external_validation/_rollout_anchors/_harness/tests/test_d0_22_open_driven_skip.py`:

```python
"""Tests for the D0-22 open-driven-dissipative skip-with-reason gate on dissipation_sign_violation.

Per DECISIONS.md D0-22: `dissipation_sign_violation` is "zero for strictly dissipative
or strictly conservative rollouts" by docstring; on open-driven systems (gravity-loaded
or forced-flow) where dE/dt > 0 over a stretch by physics, the strictly-dissipative-or-
conservative assumption fails and the rule should SKIP-with-reason rather than emit a
spurious raw value.

The gate dispatches purely on `system_class == "open-driven-dissipative"`; KE shape is
NOT a co-condition (open-driven systems with monotone KE due to model bugs should still
SKIP — the surfacing of such bugs is the job of a different rule). Tests cover the gate's
truth table:

1. Positive path: open-driven dataset + rise-then-fall KE → SKIP D0-22.
2. Negative path A: non-open-driven dataset + rise-then-fall KE → fire raw value
   (gate is system_class-conditioned, not KE-shape-conditioned).
3. Negative path B: open-driven dataset + monotone-decreasing KE → SKIP D0-22 anyway
   (gate ignores KE shape).
4. Fail-loud path: typo'd dataset name (not in mapping) → falls through to raw,
   no silent default-to-skip behavior.

All fixtures are hand-crafted, not copied from production rollouts (per
feedback_test_fixtures_hand_crafted_not_copied).
"""

from __future__ import annotations

import numpy as np

from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    LAGRANGEBENCH_DATASET_SYSTEM_CLASS,
    ParticleRollout,
    dissipation_sign_violation,
)


# ---------------------------------------------------------------------------
# Hand-crafted fixtures
# ---------------------------------------------------------------------------


def _build_rise_then_fall_rollout(
    *,
    dataset_name: str,
    n_timesteps: int = 10,
    n_particles: int = 4,
    peak_t: int = 4,
    rise_rate: float = 0.5,
    fall_rate: float = 0.3,
) -> ParticleRollout:
    """Synthetic rollout with KE rising to t=peak_t then falling.

    Mimics dam-break-2D's gravity-loaded PE→KE→dissipation profile in synthetic
    form. Velocities scale linearly during the rise phase and decay exponentially
    during the fall phase. KE = 0.5 * sum(m_i * |v_i|^2) tracks |v|^2 trajectory.

    KE(0) is set above KE_REST_THRESHOLD so the existing D0-08 KE-rest gate does
    not fire — isolates D0-22 specifically.
    """
    rng = np.random.default_rng(20260507)
    dt = 0.01
    positions = np.zeros((n_timesteps, n_particles, 2), dtype=float)
    velocities = np.zeros((n_timesteps, n_particles, 2), dtype=float)
    # v0 above rest threshold so D0-08 KE-rest skip doesn't preempt D0-22
    v0 = rng.normal(scale=1.0, size=(n_particles, 2))
    for t in range(n_timesteps):
        if t <= peak_t:
            scale = 1.0 + rise_rate * t * dt
        else:
            scale = (1.0 + rise_rate * peak_t * dt) * np.exp(-fall_rate * (t - peak_t) * dt)
        velocities[t] = v0 * scale
        positions[t] = 0.5  # not load-bearing
    return ParticleRollout(
        positions=positions,
        velocities=velocities,
        particle_type=np.zeros(n_particles, dtype=np.int32),
        particle_mass=np.ones(n_particles, dtype=np.float64),
        dt=dt,
        domain_box=np.array([[0.0, 0.0], [1.0, 1.0]]),
        metadata={"dataset": dataset_name},
    )


def _build_monotone_decreasing_rollout(
    *,
    dataset_name: str,
    n_timesteps: int = 10,
    n_particles: int = 4,
    decay_rate: float = 0.5,
) -> ParticleRollout:
    """Strictly monotone-decreasing KE rollout. Mirrors test_d0_18's
    `_build_dissipative_rollout` shape but parametrized for D0-22's negative-path-B
    fixture (open-driven dataset name with closed-dissipative KE shape — the gate
    must SKIP regardless of the shape mismatch).
    """
    rng = np.random.default_rng(20260507)
    dt = 0.01
    positions = np.zeros((n_timesteps, n_particles, 2), dtype=float)
    velocities = np.zeros((n_timesteps, n_particles, 2), dtype=float)
    v0 = rng.normal(scale=1.0, size=(n_particles, 2))
    for t in range(n_timesteps):
        velocities[t] = v0 * np.exp(-decay_rate * t * dt)
        positions[t] = 0.5
    return ParticleRollout(
        positions=positions,
        velocities=velocities,
        particle_type=np.zeros(n_particles, dtype=np.int32),
        particle_mass=np.ones(n_particles, dtype=np.float64),
        dt=dt,
        domain_box=np.array([[0.0, 0.0], [1.0, 1.0]]),
        metadata={"dataset": dataset_name},
    )


# ---------------------------------------------------------------------------
# 1. Taxonomy: dam2d post-D0-22 maps to open-driven-dissipative
# ---------------------------------------------------------------------------


def test_dam2d_classified_as_open_driven_dissipative() -> None:
    """D0-22 reclassifies dam2d empirically (post-rollout KE shape verified
    rise-then-fall). The taxonomy map is the source-of-truth for substrate
    class labels.
    """
    assert LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get("dam2d") == "open-driven-dissipative", (
        "D0-22: dam2d must be classified 'open-driven-dissipative' (gravity-loaded SPH)"
    )


# ---------------------------------------------------------------------------
# 2. Gate truth table
# ---------------------------------------------------------------------------


def test_skip_when_open_driven_with_rise_then_fall_ke() -> None:
    """Positive path: open-driven dataset + rise-then-fall KE → SKIP D0-22.

    The headline fixture for D0-22. KE rises during the synthetic gravity-load
    phase (t < peak_t), then decays during the synthetic dissipation phase
    (t > peak_t). Without D0-22, dissipation_sign_violation would fire a non-zero
    raw value (max(0, max(dE/dt)) > 0 during the rise phase) and falsely flag
    the trajectory as a violation. D0-22 SKIPs instead.
    """
    rollout = _build_rise_then_fall_rollout(dataset_name="dam2d")
    result = dissipation_sign_violation(rollout)
    assert result.value is None, "open-driven dataset + rise-then-fall KE must SKIP, not fire raw"
    assert result.skip_reason is not None
    assert "system_class='open-driven-dissipative'" in result.skip_reason
    assert "dataset='dam2d'" in result.skip_reason
    assert "gravitational PE" in result.skip_reason
    assert "D0-22" in result.skip_reason


def test_fire_raw_when_non_open_driven_with_rise_then_fall_ke() -> None:
    """Negative path A: non-open-driven dataset + rise-then-fall KE → fire raw value.

    Proves the gate dispatches on system_class, not KE shape. A "dissipative"-class
    rollout with non-monotone KE is methodologically interesting (possibly a buggy
    supposed-conservative model gaining energy) and should fire raw, not SKIP.
    """
    rollout = _build_rise_then_fall_rollout(dataset_name="tgv2d")
    result = dissipation_sign_violation(rollout)
    assert result.value is not None, (
        "tgv2d (system_class='dissipative') + rise-then-fall KE must fire raw, not SKIP — "
        "the gate must NOT co-condition on KE shape"
    )
    assert result.skip_reason is None
    assert result.value > 0  # non-monotone KE has positive dE/dt somewhere


def test_skip_when_open_driven_regardless_of_ke_shape() -> None:
    """Negative path B: open-driven dataset + monotone-decreasing KE → SKIP D0-22.

    Proves the gate is purely system_class-conditioned. If `dam2d` happens to
    show monotone-decreasing KE on a particular trajectory (unusual but possible —
    e.g., an IC where the fluid column has already fallen), D0-22 still SKIPs
    because the substrate is open-driven by physics and the rule's strictly-
    dissipative-or-conservative assumption still doesn't hold (other dam-break
    trajectories will show non-monotone KE).
    """
    rollout = _build_monotone_decreasing_rollout(dataset_name="dam2d")
    result = dissipation_sign_violation(rollout)
    assert result.value is None, (
        "open-driven dataset must SKIP regardless of KE shape — gate is system_class-"
        "conditioned, not KE-shape-conditioned"
    )
    assert result.skip_reason is not None
    assert "system_class='open-driven-dissipative'" in result.skip_reason


def test_unknown_dataset_falls_through_to_raw() -> None:
    """Fail-loud path: typo'd dataset name not in mapping → falls through to raw.

    Guards against future taxonomy entries with typos creating silent fall-through
    behavior. A dataset name that is neither in `LAGRANGEBENCH_DATASET_SYSTEM_CLASS`
    nor an empty string must NOT default to any SKIP path; it must fire raw value.
    Catches the mistake of `default_skip_class = "open-driven-dissipative"` etc.
    """
    rollout = _build_rise_then_fall_rollout(dataset_name="dam2d_typo_not_in_mapping")
    result = dissipation_sign_violation(rollout)
    assert LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get("dam2d_typo_not_in_mapping") is None, (
        "test precondition: typo'd name must not be in mapping"
    )
    assert result.value is not None, (
        "unknown dataset must fall through to raw, never silent SKIP"
    )
    assert result.skip_reason is None


# ---------------------------------------------------------------------------
# 3. Reason-template invariants (D0-19 §3.4)
# ---------------------------------------------------------------------------


def test_skip_reason_template_constant_across_invocations() -> None:
    """D0-19 §3.4 requires the skip_reason string to be template-constant
    across rows of the same (rule, stack). Two invocations with the same
    `dataset` value must return the same skip_reason string verbatim.
    """
    r1 = _build_rise_then_fall_rollout(dataset_name="dam2d")
    r2 = _build_rise_then_fall_rollout(dataset_name="dam2d", n_particles=8)  # different N
    s1 = dissipation_sign_violation(r1).skip_reason
    s2 = dissipation_sign_violation(r2).skip_reason
    assert s1 == s2, (
        "D0-19 §3.4: skip_reason must be template-constant within (rule, stack); "
        "got divergent strings on the same dataset name"
    )


def test_skip_reason_signposts_d0_22() -> None:
    """The skip_reason must cite D0-22 by name for audit-trail traceability —
    same discipline as D0-18's reason which cites 'D0-18' verbatim.
    """
    rollout = _build_rise_then_fall_rollout(dataset_name="dam2d")
    result = dissipation_sign_violation(rollout)
    assert result.skip_reason is not None
    assert "D0-22" in result.skip_reason
```

- [ ] **Step 3.2: Run tests, expect all to FAIL**

```bash
cd /Users/zenith/Desktop/physics-lint
source .venv/bin/activate
pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_d0_22_open_driven_skip.py -v -o "addopts=" 2>&1 | tail -30
```

Expected: all 6 tests FAIL. The taxonomy test fails because `dam2d` is currently mapped to `"dissipative"`. The gate tests fail because the SKIP path doesn't exist yet.

If any test passes (other than expected failures), the test is wrong — investigate before proceeding. The TDD red phase requires *all* tests to fail for the right reason.

---

## Task 4: TDD green — implement D0-22 SKIP gate + flip dam2d label

**Files:**
- Modify: `external_validation/_rollout_anchors/_harness/particle_rollout_adapter.py:251-259` (taxonomy entry)
- Modify: `external_validation/_rollout_anchors/_harness/particle_rollout_adapter.py:524-558` (`dissipation_sign_violation` function — add D0-22 gate)
- Modify: `external_validation/_rollout_anchors/_harness/tests/test_d0_18_dissipative_skip.py:107` (split dam2d out of the "all 2D SPH dissipative" assertion)

- [ ] **Step 4.1: Flip `dam2d` label in `LAGRANGEBENCH_DATASET_SYSTEM_CLASS`**

Open `external_validation/_rollout_anchors/_harness/particle_rollout_adapter.py`. At line 251-259, the dictionary currently reads:

```python
LAGRANGEBENCH_DATASET_SYSTEM_CLASS: dict[str, str] = {
    "tgv2d": "dissipative",
    "rpf2d": "dissipative",
    "ldc2d": "dissipative",
    "dam2d": "dissipative",
    "tgv3d": "dissipative",
    "rpf3d": "dissipative",
    "ldc3d": "dissipative",
}
```

Change `"dam2d"` line to:

```python
LAGRANGEBENCH_DATASET_SYSTEM_CLASS: dict[str, str] = {
    "tgv2d": "dissipative",
    "rpf2d": "dissipative",  # forward-flag D0-22: forced flow, almost certainly should be open-driven-dissipative; awaiting empirical probe
    "ldc2d": "dissipative",  # forward-flag D0-22: forced flow, almost certainly should be open-driven-dissipative; awaiting empirical probe
    "dam2d": "open-driven-dissipative",  # D0-22: empirically reclassified after rung-4c 1-traj smoke confirmed rise-then-fall KE shape
    "tgv3d": "dissipative",  # likely correct (3D inherits 2D-TGV physics) but unverified
    "rpf3d": "dissipative",  # forward-flag D0-22: forced flow, almost certainly should be open-driven-dissipative; awaiting empirical probe
    "ldc3d": "dissipative",  # forward-flag D0-22: forced flow, almost certainly should be open-driven-dissipative; awaiting empirical probe
}
```

The inline forward-flag comments preserve the two-tier split from D0-22 §4 at the source-of-truth code location.

- [ ] **Step 4.2: Add D0-22 SKIP gate to `dissipation_sign_violation`**

In `external_validation/_rollout_anchors/_harness/particle_rollout_adapter.py`, the existing `dissipation_sign_violation` function starts at line 524. The D0-08 KE-rest gate is at lines 547-555. The D0-22 gate goes **after** D0-08 (D0-08 takes precedence — same ordering as `energy_drift`'s D0-08-then-D0-18 pattern), **before** the raw computation at line 556-558.

Find this block (lines 540-558):

```python
    e_series = kinetic_energy_series(rollout)
    e_max = float(np.max(e_series))
    if e_max < KE_REST_THRESHOLD:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"max(KE)={e_max:.3e} < {KE_REST_THRESHOLD:.0e} (trajectory "
                f"has no kinetic energy; dissipation question undefined; "
                f"see DECISIONS.md D0-08)"
            ),
        )
    de_dt = np.diff(e_series) / rollout.dt
    max_growth = float(np.max(de_dt))
    return HarnessDefect(value=max(0.0, max_growth) / e_max)
```

Replace with:

```python
    e_series = kinetic_energy_series(rollout)
    e_max = float(np.max(e_series))
    if e_max < KE_REST_THRESHOLD:
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"max(KE)={e_max:.3e} < {KE_REST_THRESHOLD:.0e} (trajectory "
                f"has no kinetic energy; dissipation question undefined; "
                f"see DECISIONS.md D0-08)"
            ),
        )
    # D0-22 skip-with-reason gate: open-driven-dissipative substrate class.
    # The rule's strictly-dissipative-or-conservative assumption (zero for
    # strictly dissipative or strictly conservative rollouts) does not apply
    # to gravity-loaded or forced-flow systems where dE/dt > 0 over a stretch
    # by physics. Gate is purely system_class-conditioned by design (an
    # open-driven system with monotone KE due to a model bug should still
    # SKIP — surfacing the bug is the job of a different rule).
    dataset_name = rollout.metadata.get("dataset", "") if rollout.metadata else ""
    system_class = LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get(dataset_name)
    if system_class == "open-driven-dissipative":
        return HarnessDefect(
            value=None,
            skip_reason=(
                f"system_class='open-driven-dissipative' (dataset={dataset_name!r}); "
                "dE/dt > 0 over a stretch by physics (gravitational PE → KE conversion); "
                "the strictly-dissipative-or-conservative assumption underpinning "
                "dissipation_sign_violation does not apply. See DECISIONS.md D0-22."
            ),
        )
    de_dt = np.diff(e_series) / rollout.dt
    max_growth = float(np.max(de_dt))
    return HarnessDefect(value=max(0.0, max_growth) / e_max)
```

Also update the function's docstring (lines 524-540) to mention D0-22 alongside D0-08:

Find:

```python
def dissipation_sign_violation(rollout: ParticleRollout) -> HarnessDefect:
    """max(0, max(dE/dt)) / max(|E_max|, eps), or SKIP if max(KE) below threshold.

    Mirrors PH-CON-003's emitted `violation` form. Zero for strictly
    dissipative or strictly conservative rollouts (dE/dt ≤ 0 or = 0
    everywhere); non-zero for rollouts where the model spuriously gains
    energy at any timestep. SKIPS with reason when ``max(KE) <
    KE_REST_THRESHOLD`` (the trajectory has effectively no kinetic
    energy at any timestep, so the dissipation question is meaningless;
    pre-registered in DECISIONS.md D0-08).

    Uses forward differences ``np.diff(E) / dt`` to match PH-CON-003's
    Week-2 endpoint-pathology fix verbatim — second-order ``np.gradient``
    edge-extrapolation produces spurious positive endpoint slopes on
    fast-decaying signals; forward differences sample at nt - 1 step
    boundaries and have no such pathology.
    """
```

Replace with:

```python
def dissipation_sign_violation(rollout: ParticleRollout) -> HarnessDefect:
    """max(0, max(dE/dt)) / max(|E_max|, eps), or SKIP on substrate-incompatibility.

    Mirrors PH-CON-003's emitted `violation` form. Zero for strictly
    dissipative or strictly conservative rollouts (dE/dt ≤ 0 or = 0
    everywhere); non-zero for rollouts where the model spuriously gains
    energy at any timestep. Two skip-with-reason paths:

    1. **KE-rest** (DECISIONS.md D0-08): SKIPS when ``max(KE) <
       KE_REST_THRESHOLD`` (the trajectory has effectively no kinetic
       energy at any timestep, so the dissipation question is meaningless).

    2. **Open-driven-dissipative** (DECISIONS.md D0-22): SKIPS when
       ``rollout.metadata["dataset"]`` resolves to ``"open-driven-dissipative"``
       via ``LAGRANGEBENCH_DATASET_SYSTEM_CLASS``. The rule's strictly-
       dissipative-or-conservative assumption does not apply to gravity-
       loaded or forced-flow systems where dE/dt > 0 over a stretch by
       physics; emitting a raw value would falsely flag the substrate's
       physics as a model violation.

    Uses forward differences ``np.diff(E) / dt`` to match PH-CON-003's
    Week-2 endpoint-pathology fix verbatim — second-order ``np.gradient``
    edge-extrapolation produces spurious positive endpoint slopes on
    fast-decaying signals; forward differences sample at nt - 1 step
    boundaries and have no such pathology.
    """
```

- [ ] **Step 4.3: Update `test_d0_18_dissipative_skip.py:107` to exclude `dam2d` from "all 2D SPH dissipative" loop**

Open `external_validation/_rollout_anchors/_harness/tests/test_d0_18_dissipative_skip.py`. At line 105-110, the existing test reads:

```python
def test_lagrangebench_dataset_system_class_includes_all_2d_sph() -> None:
    """All five 2D LB SPH datasets must be in the dissipative mapping (D0-18)."""
    for dataset in ("tgv2d", "rpf2d", "ldc2d", "dam2d"):
        assert LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get(dataset) == "dissipative", (
            f"D0-18: {dataset} must be classified 'dissipative' (SPH viscous)"
        )
```

Replace with:

```python
def test_lagrangebench_dataset_system_class_includes_2d_sph_dissipative() -> None:
    """The 2D LB SPH datasets in the closed-dissipative class (post-D0-22).

    `dam2d` was reclassified to 'open-driven-dissipative' per D0-22; see
    `test_d0_22_open_driven_skip.py::test_dam2d_classified_as_open_driven_dissipative`.

    `rpf2d` and `ldc2d` are forward-flagged as almost-certainly-misclassified
    (forced flows by definition) but retain `"dissipative"` pending empirical
    probe in a future rung — D0-22's "classify when you exercise" discipline.
    """
    for dataset in ("tgv2d", "rpf2d", "ldc2d"):
        assert LAGRANGEBENCH_DATASET_SYSTEM_CLASS.get(dataset) == "dissipative", (
            f"D0-18: {dataset} must currently be classified 'dissipative' "
            f"(forward-flagged for empirical probe per D0-22 §4 if forced-flow)"
        )
```

The function rename (`includes_all_2d_sph` → `includes_2d_sph_dissipative`) reflects that the assertion no longer covers the full 2D SPH set; some 2D SPH substrates now have other class labels.

- [ ] **Step 4.4: Run all D0-22 tests, expect PASS**

```bash
cd /Users/zenith/Desktop/physics-lint
source .venv/bin/activate
pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_d0_22_open_driven_skip.py -v -o "addopts=" 2>&1 | tail -20
```

Expected: 6 tests PASS.

- [ ] **Step 4.5: Run D0-18 tests, expect PASS (regression check on the test_d0_18 edit)**

```bash
cd /Users/zenith/Desktop/physics-lint
pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_d0_18_dissipative_skip.py -v -o "addopts=" 2>&1 | tail -25
```

Expected: all D0-18 tests PASS (including the renamed function).

- [ ] **Step 4.6: Run full test suite, expect baseline + 6 new passes**

```bash
cd /Users/zenith/Desktop/physics-lint
pytest --import-mode=importlib external_validation/ -o "addopts=" 2>&1 | tail -5
```

Expected: `453 passed, 1 skipped` (was 447+1; added 6 from `test_d0_22_open_driven_skip.py`).

- [ ] **Step 4.7: Commit D0-22 implementation**

```bash
cd /Users/zenith/Desktop/physics-lint
git add external_validation/_rollout_anchors/_harness/SCHEMA.md \
        external_validation/_rollout_anchors/_harness/particle_rollout_adapter.py \
        external_validation/_rollout_anchors/_harness/tests/test_d0_22_open_driven_skip.py \
        external_validation/_rollout_anchors/_harness/tests/test_d0_18_dissipative_skip.py
git commit -m "_harness: implement D0-22 open-driven-dissipative SKIP gate on dissipation_sign_violation; flip dam2d to open-driven-dissipative"
```

Expected: clean commit. `git log --oneline -2` shows D0-22 entry + this commit.

---

## Task 5: Discover dam2d dataset directory name

**Files:** none (discovery-only step; output captured in Task 6)

LagrangeBench's `download_data.sh dam2d` produces a dataset directory at `datasets/<unknown_name>`. Rung 4a discovered `2D_TGV_2500_10kevery100` for tgv2d via inspection. We need the analogous name for dam2d before Task 6 can populate `LAGRANGEBENCH_DATASET_DIRS`.

- [ ] **Step 5.1: Inspect LagrangeBench upstream config for dam2d dataset.src**

LagrangeBench upstream is at `https://github.com/tumaer/lagrangebench` at sha `b880a6c84a93792d2499d2a9b8ba3a077ddf44e2` (per `01-lagrangebench/emit_sarif.py:58`). The dam2d config is at `configs/dam2d/base.yaml` (consult `configs/dam2d/`). Run a Modal-side discovery function or local clone:

**Option A (local clone, fastest):**

```bash
cd /tmp
git clone --depth 1 https://github.com/tumaer/lagrangebench
grep -rn "src:\|^src:" lagrangebench/configs/dam_2d/ 2>/dev/null | head -5
```

(Note: dam_2d directory may use underscore; configs structure: `configs/<name>/base.yaml` per LB README.)

Expected output: a line like `src: ./datasets/2D_DB_5740_20kevery100` or similar. Record this exact string.

**Option B (Modal CPU-only function):** if local clone fails or is undesirable, add a one-shot CPU function to `modal_app.py` that runs `bash download_data.sh dam2d datasets/` and prints `os.listdir("datasets/")`. Skip if Option A succeeds.

- [ ] **Step 5.2: Record discovered name**

Open a scratch note (or remember for Task 6) the discovered dataset directory name. Throughout Task 6 below, replace `<DAM2D_DIR_NAME>` with the discovered string verbatim. Most likely value: `2D_DB_5740_20kevery100` per LB's naming convention; **verify against upstream config**, do not assume.

---

## Task 6: Modal infrastructure for dam2d rollouts

**Files:**
- Modify: `external_validation/_rollout_anchors/01-lagrangebench/modal_app.py:520-523` (LAGRANGEBENCH_DATASET_DIRS)
- Modify: `external_validation/_rollout_anchors/01-lagrangebench/modal_app.py` (append after line 1014: 2 new rollout functions; append CLI entrypoints)

- [ ] **Step 6.1: Extend `LAGRANGEBENCH_DATASET_DIRS`**

Open `external_validation/_rollout_anchors/01-lagrangebench/modal_app.py`. At lines 520-523:

```python
LAGRANGEBENCH_DATASET_DIRS: dict[str, str] = {
    "tgv_2d": "2D_TGV_2500_10kevery100",
    # extend as P1+ work scales
}
```

Replace with:

```python
LAGRANGEBENCH_DATASET_DIRS: dict[str, str] = {
    "tgv_2d": "2D_TGV_2500_10kevery100",
    "dam_2d": "<DAM2D_DIR_NAME>",  # discovered Task 5; verified against LB sha b880a6c84a93792d2499d2a9b8ba3a077ddf44e2 configs/dam_2d/base.yaml
}
```

Replace `<DAM2D_DIR_NAME>` with the verbatim string from Task 5.

- [ ] **Step 6.2: Add `lagrangebench_rollout_p1_segnn_dam2d` Modal function**

Open `modal_app.py`. Find `lagrangebench_rollout_p1_gns_tgv2d` (line 1021). The new SEGNN-dam2d function mirrors `lagrangebench_rollout_p0_segnn_tgv2d` (line 646) with the differences enumerated as inline comments — same minimal-edit posture rung-4a's P0/P1 used.

Append after the existing P1 GNS-TGV2D function (after line ~1267, before `lagrangebench_eps_p0_segnn_tgv2d`):

```python
# P1 SEGNN-DAM2D: structurally identical to lagrangebench_rollout_p0_segnn_tgv2d
# (line 646), with checkpoint name + dataset name + rollout-subdir prefix
# replaced. Deliberate minimal-edit copy rather than parameterized refactor —
# P0 SEGNN-TGV2D is already well-tested at d03df3e (D0-17 amendment 1) and
# refactoring it carries drift risk. Differences enumerated:
#
#   - ckpt_root         segnn_tgv2d  -> segnn_dam2d
#   - zip_path basename segnn_tgv2d.zip -> segnn_dam2d.zip
#   - LAGRANGEBENCH_CHECKPOINT_GDOWN_IDS["segnn_tgv2d"] -> [...]["segnn_dam2d"]
#   - rollout_subdir    segnn_tgv2d_<git_sha> -> segnn_dam2d_<git_sha>
#   - dataset.src key   tgv_2d -> dam_2d (LAGRANGEBENCH_DATASET_DIRS lookup)
#   - dataset.name CLI  tgv2d -> dam2d
#   - RolloutMetadata.dataset "tgv2d" -> "dam2d"
#
# Per DECISIONS.md D0-22: dam2d post-rung-4c is classified
# 'open-driven-dissipative'; the harness's substrate-detection layer
# dispatches dissipation_sign_violation to the D0-22 SKIP path on these
# rollouts. The Modal-side rollout function is dataset-agnostic (the
# substrate-class dispatch happens consumer-side at SARIF emission).
@app.function(
    image=rollout_image,
    gpu=ROLLOUT_GENERATION_GPU_CLASS,
    volumes={"/vol": rollout_volume},
    timeout=3600,
)
def lagrangebench_rollout_p1_segnn_dam2d(git_sha: str, full_git_sha: str) -> dict:
    """Run SEGNN-dam2d 20-traj rollout per rung-4c plan. Mirrors
    lagrangebench_rollout_p0_segnn_tgv2d's body verbatim with the deltas
    enumerated above.
    """
    # === BEGIN COPY of lagrangebench_rollout_p0_segnn_tgv2d body ===
    # Substitute throughout:
    #   "segnn_tgv2d" -> "segnn_dam2d"
    #   "tgv_2d" -> "dam_2d"   (LAGRANGEBENCH_DATASET_DIRS key)
    #   "tgv2d" -> "dam2d"     (dataset.name CLI value, RolloutMetadata.dataset)
    # All other code is preserved exactly: image, GPU class, timeout, manifest
    # shape, subprocess error handling, conversion handoff, files-walk fallback,
    # rollout_volume.commit() at end.
    # END copy boilerplate.
    raise NotImplementedError(
        "Step 6.2: copy the body of lagrangebench_rollout_p0_segnn_tgv2d "
        "(modal_app.py:646-992) into this function with the 6 string substitutions "
        "listed above. The substitutions are mechanical; do not change any other "
        "logic. After substitution, this docstring should describe the function's "
        "behavior verbatim, not raise NotImplementedError."
    )
```

**The actual implementation** is a mechanical copy of the P0 SEGNN-TGV2D function body (`modal_app.py:646-992`, ~340 lines) with 3 string substitutions and 0 logic changes. The plan-doc-as-spec contract: do not introduce any other deltas; if a difference seems necessary, surface it as a deviation rather than absorb silently.

- [ ] **Step 6.3: Add `lagrangebench_rollout_p1_gns_dam2d` Modal function**

Append immediately after the SEGNN-dam2d function:

```python
# P1 GNS-DAM2D: structurally identical to lagrangebench_rollout_p1_gns_tgv2d
# (line 1021), with dataset replaced. Differences enumerated:
#
#   - ckpt_root         gns_tgv2d  -> gns_dam2d
#   - zip_path basename gns_tgv2d.zip -> gns_dam2d.zip
#   - LAGRANGEBENCH_CHECKPOINT_GDOWN_IDS["gns_tgv2d"] -> [...]["gns_dam2d"]
#   - rollout_subdir    gns_tgv2d_<git_sha> -> gns_dam2d_<git_sha>
#   - dataset.src key   tgv_2d -> dam_2d
#   - dataset.name CLI  tgv2d -> dam2d
#   - RolloutMetadata.dataset "tgv2d" -> "dam2d"
@app.function(
    image=rollout_image,
    gpu=ROLLOUT_GENERATION_GPU_CLASS,
    volumes={"/vol": rollout_volume},
    timeout=3600,
)
def lagrangebench_rollout_p1_gns_dam2d(git_sha: str, full_git_sha: str) -> dict:
    """Run GNS-dam2d 20-traj rollout per rung-4c plan. Mirrors
    lagrangebench_rollout_p1_gns_tgv2d's body verbatim with the deltas
    enumerated above.
    """
    # Same copy-with-substitutions discipline as Step 6.2.
    raise NotImplementedError(
        "Step 6.3: copy the body of lagrangebench_rollout_p1_gns_tgv2d "
        "into this function with the 4 string substitutions listed above."
    )
```

- [ ] **Step 6.4: Add CLI entrypoints `rollout_p1_segnn_dam2d` and `rollout_p1_gns_dam2d`**

Find the existing CLI entrypoints `rollout_p0_segnn_tgv2d` (line ~1917) and `rollout_p1_gns_tgv2d` (line ~2030). Append immediately after the existing pair:

```python
@app.local_entrypoint()
def rollout_p1_segnn_dam2d() -> None:
    """Local entrypoint for rung-4c P1 SEGNN-dam2d rollout. Mirrors
    rollout_p0_segnn_tgv2d / rollout_p1_gns_tgv2d shape: reads git sha,
    invokes the Modal function, prints manifest tail.
    """
    # Copy from rollout_p0_segnn_tgv2d, substituting the function name.
    raise NotImplementedError(
        "Step 6.4: copy rollout_p0_segnn_tgv2d's body, substitute "
        "lagrangebench_rollout_p0_segnn_tgv2d -> lagrangebench_rollout_p1_segnn_dam2d."
    )


@app.local_entrypoint()
def rollout_p1_gns_dam2d() -> None:
    """Local entrypoint for rung-4c P1 GNS-dam2d rollout."""
    # Copy from rollout_p1_gns_tgv2d, substituting the function name.
    raise NotImplementedError(
        "Step 6.4: copy rollout_p1_gns_tgv2d's body, substitute "
        "lagrangebench_rollout_p1_gns_tgv2d -> lagrangebench_rollout_p1_gns_dam2d."
    )
```

After actual implementation, the `raise NotImplementedError` lines should be removed and the body should be the copied function code.

- [ ] **Step 6.5: Verify modal_app.py imports cleanly + GPU-class regression test passes**

```bash
cd /Users/zenith/Desktop/physics-lint
source .venv/bin/activate
python -c "from external_validation._rollout_anchors._01_lagrangebench import modal_app; print('imports OK')" 2>&1 | tail -3
# (If the package path differs, use the python-path-relative form:)
python -c "import sys; sys.path.insert(0, '.'); from external_validation._rollout_anchors import _01_lagrangebench" 2>&1 | tail -3
pytest --import-mode=importlib external_validation/_rollout_anchors/01-lagrangebench/tests/test_modal_app_gpu_class.py -v -o "addopts=" 2>&1 | tail -10
```

Expected: imports clean, GPU-class regression test passes (asserts `ROLLOUT_GENERATION_GPU_CLASS == "A10G"` for new functions too).

If the GPU-class test only iterates over rung-4a functions and doesn't pick up rung-4c functions, extend the test parametrization in the same edit:

```python
# In test_modal_app_gpu_class.py, find the existing function list and append:
("lagrangebench_rollout_p1_segnn_dam2d", "A10G"),
("lagrangebench_rollout_p1_gns_dam2d", "A10G"),
```

- [ ] **Step 6.6: Commit Modal infrastructure**

```bash
cd /Users/zenith/Desktop/physics-lint
git add external_validation/_rollout_anchors/01-lagrangebench/modal_app.py \
        external_validation/_rollout_anchors/01-lagrangebench/tests/test_modal_app_gpu_class.py
git commit -m "01-lagrangebench/modal_app: add P1 SEGNN-dam2d + P1 GNS-dam2d rollout functions and CLI entrypoints (rung 4c)"
```

Expected: clean commit.

---

## Task 7: Pre-flight 5-step checklist + 1-traj smoke per stack

**Files:**
- Create: `preflight/2026-05-07-rung-4c.txt` (preflight log; committed as evidence-of-execution)
- Create (test fixture, optional): `external_validation/_rollout_anchors/_harness/tests/fixtures/synthetic_dam_break_pkl.py` (synthetic conversion fixture if Step 7.2 surfaces a gap)

Per CLAUDE.md global pre-flight checklist + design doc §3.3. Each step has a deliverable; the preflight log records the evidence.

- [ ] **Step 7.1: Create preflight log file with stub structure**

Create `preflight/2026-05-07-rung-4c.txt`:

```
Rung 4c pre-flight log — 2026-05-07
====================================

Branch: feature/rung-4c-substrate-class-extension
Predecessor sha: <commit sha of Task 6 commit>
Compute budget: < $0.20 USD; < 14 min A10G

Step 1 — Data inspection
  Status: <PENDING / PASS / FAIL>
  Modal function: <name>
  Output:
  <captured output>

Step 2 — Conversion round-trip
  Status: <PENDING / PASS / FAIL>
  Fixture: <path>
  Output:
  <captured output>

Step 3 — Rule sanity test
  Status: <PENDING / PASS / FAIL>
  Test: <pytest invocation>
  Output:
  <captured output>

Step 4 — 1-traj smoke per stack
  Status: <PENDING / PASS / FAIL>
  SEGNN-dam2d 1-traj: <result>
  GNS-dam2d 1-traj: <result>
  KE(t) shape verification: <empirical observation>

Step 5 — End-to-end pipeline smoke
  Status: <PENDING / PASS / FAIL>
  Output:
  <captured output>

Decision: <PROCEED / ABORT>
Justification: <one paragraph>
```

- [ ] **Step 7.2: Step 1 — data inspection (Modal CPU-only)**

Run a one-shot Modal CPU function that downloads dam2d dataset and prints metadata. If `lagrangebench_install_smoke` (line 255 of `modal_app.py`) is parametrizable, extend it; otherwise add a small new function:

```bash
cd /Users/zenith/Desktop/physics-lint
modal run external_validation/_rollout_anchors/01-lagrangebench/modal_app.py::lagrangebench_smoke 2>&1 | tee preflight/_step1_data_inspection.log | tail -20
```

If `lagrangebench_smoke` doesn't take a dataset argument, run a minimal inline Modal command (`modal shell` + `bash download_data.sh dam2d datasets/` + `cat datasets/<DAM2D_DIR_NAME>/metadata.json`) or extend the smoke function. Capture the metadata.json contents in the preflight log.

Pass criteria:
- Particle count > 0 and consistent with LB's published dam-break-2D config
- IC velocity stats: mean velocity ≈ 0 (start-at-rest verified)
- `particle_type` includes both fluid and wall particle codes (dam-break has both)
- `dt`, `domain_box`, `periodic_boundary_conditions` all present and well-formed

Update preflight log Step 1 status. If FAIL: investigate dataset shape; do not proceed.

- [ ] **Step 7.3: Step 2 — conversion round-trip on synthetic dam-break fixture (CPU-only, local)**

The conversion path is `lagrangebench_pkl_to_npz` (in `_harness/lagrangebench_pkl_to_npz.py`). Verify it round-trips on a synthetic dam-break-shaped pkl fixture before any Modal compute is spent.

Quick verification: extend an existing test in `_harness/tests/test_lagrangebench_pkl_to_npz.py` (or write a new test) with a dam-break-shaped fixture (mixed fluid/wall particle types):

```bash
cd /Users/zenith/Desktop/physics-lint
pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_lagrangebench_pkl_to_npz.py -v -o "addopts=" 2>&1 | tail -10
```

Pass criteria: existing tests pass; conversion does not error on mixed particle_type. If a new test was needed, it passes too. If FAIL: fix the conversion before proceeding.

Update preflight log Step 2 status.

- [ ] **Step 7.4: Step 3 — rule sanity test (CPU-only, local)**

The D0-22 SKIP gate tests already cover this (Task 3); confirm by re-running them:

```bash
cd /Users/zenith/Desktop/physics-lint
pytest --import-mode=importlib external_validation/_rollout_anchors/_harness/tests/test_d0_22_open_driven_skip.py -v -o "addopts=" 2>&1 | tail -10
```

Pass criteria: 6/6 tests PASS. Capture the output in preflight log Step 3.

- [ ] **Step 7.5: Step 4 — 1-traj smoke per stack on Modal A10G**

Run 1-traj SEGNN-dam2d smoke. Modify the rollout function temporarily (or pass a `n_trajs=1` override CLI arg if the function supports it) to use 1 trajectory instead of 20.

**Option A (CLI override, preferred if supported):** Check whether the rollout function reads `eval.infer.n_trajs` from a parameter. If yes, pass `n_trajs=1` from the CLI entrypoint:

```bash
modal run external_validation/_rollout_anchors/01-lagrangebench/modal_app.py::rollout_p1_segnn_dam2d --n-trajs 1 2>&1 | tee preflight/_step4_segnn_smoke.log | tail -30
modal run external_validation/_rollout_anchors/01-lagrangebench/modal_app.py::rollout_p1_gns_dam2d --n-trajs 1 2>&1 | tee preflight/_step4_gns_smoke.log | tail -30
```

**Option B (temporary code edit):** if the CLI doesn't support `n_trajs`, edit the function to use `eval.infer.n_trajs=1` for the smoke; revert the edit before Task 8's 20-traj fire.

After both smokes complete:

```bash
modal volume get rollout-anchors-artifacts \
    /vol/rollouts/lagrangebench/segnn_dam2d_<smoke_sha>/ \
    /tmp/rung4c_smoke_segnn/
modal volume get rollout-anchors-artifacts \
    /vol/rollouts/lagrangebench/gns_dam2d_<smoke_sha>/ \
    /tmp/rung4c_smoke_gns/
```

Then in Python, verify KE(t) rise-then-fall shape:

```python
import numpy as np
from external_validation._rollout_anchors._harness.particle_rollout_adapter import (
    load_rollout_npz, kinetic_energy_series,
)

for stack, dirpath in [("segnn", "/tmp/rung4c_smoke_segnn/"), ("gns", "/tmp/rung4c_smoke_gns/")]:
    rollout = load_rollout_npz(f"{dirpath}/particle_rollout_traj00.npz")
    ke = kinetic_energy_series(rollout)
    rises = (np.diff(ke) > 0).any()
    peak_t = int(np.argmax(ke))
    print(f"{stack}: KE(0)={ke[0]:.3e}, max(KE)={ke.max():.3e} at t={peak_t}, "
          f"KE(end)={ke[-1]:.3e}, rises_anywhere={rises}")
```

**Pass criteria:** for both stacks, `rises_anywhere == True` AND `peak_t` is in the interior (not 0, not nt-1). This confirms the rise-then-fall shape that justifies the empirical `dam2d → "open-driven-dissipative"` reclassification.

If FAIL (KE is monotone-decreasing or shape is otherwise unexpected): the empirical-classification discipline blocks D0-22's applicability to dam2d. Per design doc §1.2 pre-flight gating-condition: do NOT proceed; revisit the design.

Update preflight log Step 4 with the empirical KE shape numbers and the PASS/FAIL decision.

- [ ] **Step 7.6: Step 5 — end-to-end pipeline smoke (CPU-only, local)**

After the 1-traj smokes return successfully, exercise the full pipeline on the smoke npz to catch loader-contract regressions:

```python
# Run from repo root
import subprocess

# Verify lint_npz_dir reads the smoke npzs without error
from external_validation._rollout_anchors._harness.lint_npz_dir import lint_npz_dir
results = lint_npz_dir("/tmp/rung4c_smoke_segnn/")
print(f"SEGNN-dam2d smoke: {len(results)} HarnessResult rows")
# Expected: 3 rows (one per rule) for 1 trajectory.
# mass_conservation_defect: raw 0.0
# energy_drift: SKIP D0-08 (KE-rest fires because KE(0) ~ 0)
# dissipation_sign_violation: SKIP D0-22 (open-driven)
```

**Pass criteria:** 3 result rows; mass=0.0 raw; energy_drift carries D0-08 reason; dissipation_sign_violation carries D0-22 reason. If any row is shaped wrong: investigate before scaling.

Update preflight log Step 5.

- [ ] **Step 7.7: Final preflight gate — record decision and commit log**

Update `preflight/2026-05-07-rung-4c.txt` final section:

```
Decision: PROCEED
Justification: All 5 pre-flight steps PASS. Empirical KE rise-then-fall shape
confirmed on both stacks at <smoke_sha>. D0-22 reclassification of dam2d to
'open-driven-dissipative' is empirically justified. Pipeline end-to-end clean.
20-traj Modal fire authorized.
```

Or, if any step FAILED:

```
Decision: ABORT
Justification: Step <N> FAIL — <description>. Revisit design before any
20-traj fire.
```

Commit the preflight log:

```bash
cd /Users/zenith/Desktop/physics-lint
git add preflight/2026-05-07-rung-4c.txt
git commit -m "preflight: rung 4c 5-step checklist log (PROCEED|ABORT)"
```

If ABORT: STOP HERE. Resume design pass; do not run Task 8.

---

## Task 8: Production 20-traj Modal rollouts

**Files:** none committed locally; artifacts on Modal Volume.

**Pre-condition:** Task 7 final decision is PROCEED. If ABORT, do not run.

- [ ] **Step 8.1: SEGNN-dam2d 20-traj rollout**

```bash
cd /Users/zenith/Desktop/physics-lint
modal run external_validation/_rollout_anchors/01-lagrangebench/modal_app.py::rollout_p1_segnn_dam2d 2>&1 | tee outputs/rung4c_segnn_dam2d_run.log | tail -50
```

Expected: ~5 min A10G; manifest reports `inference_returncode == 0` and `conversion_returncode == 0`; 20 npz files written to `/vol/rollouts/lagrangebench/segnn_dam2d_<sha>/`.

If FAIL: capture the manifest tail in the run log; investigate rather than retry blindly. Common causes per rung-4b T7 amendment 2: loader-contract issues, dataset.src directory name mismatch.

- [ ] **Step 8.2: GNS-dam2d 20-traj rollout**

```bash
cd /Users/zenith/Desktop/physics-lint
modal run external_validation/_rollout_anchors/01-lagrangebench/modal_app.py::rollout_p1_gns_dam2d 2>&1 | tee outputs/rung4c_gns_dam2d_run.log | tail -50
```

Expected: ~3 min A10G; manifest reports both returncodes 0; 20 npzs at `/vol/rollouts/lagrangebench/gns_dam2d_<sha>/`.

- [ ] **Step 8.3: Verify Volume contents**

```bash
modal volume ls rollout-anchors-artifacts /vol/rollouts/lagrangebench/segnn_dam2d_<sha>/ | wc -l
modal volume ls rollout-anchors-artifacts /vol/rollouts/lagrangebench/gns_dam2d_<sha>/ | wc -l
```

Expected: 20 npz files in each (plus possibly a manifest.json or similar).

- [ ] **Step 8.4: Capture both genesis shas for downstream artifact provenance**

Open a scratch note. Record:
```
SEGNN_DAM2D_PKL_INFERENCE_SHA = "<10-char sha>"   # from inference manifest
SEGNN_DAM2D_NPZ_CONVERSION_SHA = "<10-char sha>"  # from conversion manifest (typically same as inference)
GNS_DAM2D_PKL_INFERENCE_SHA = "<10-char sha>"
GNS_DAM2D_NPZ_CONVERSION_SHA = "<10-char sha>"
```

These shas land in `emit_sarif.py`'s pinned-shas section in Task 9.

- [ ] **Step 8.5: Append run logs to preflight (no commit, run logs are gitignored)**

```bash
cd /Users/zenith/Desktop/physics-lint
ls -la outputs/rung4c_*_run.log
```

(If `outputs/` is gitignored except for `outputs/sarif/` and `outputs/figures/*.{png,pdf}`, leave the run logs uncommitted as ephemeral.)

---

## Task 9: Extend `emit_sarif.py` for dam2d and emit SARIFs

**Files:**
- Modify: `external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py` (add dam2d shas + extend driver loop)
- Create: `external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/segnn_dam2d_<sha>.sarif`
- Create: `external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/gns_dam2d_<sha>.sarif`

- [ ] **Step 9.1: Add dam2d shas to `emit_sarif.py`**

Open `emit_sarif.py`. Find the pinned-sha block (line 50-58) which currently has `SEGNN_PKL_INFERENCE_SHA = "8c3d080397"` etc. Append:

```python
# Rung 4c dam-break shas (Task 8 output).
SEGNN_DAM2D_PKL_INFERENCE_SHA = "<from Step 8.4>"
SEGNN_DAM2D_NPZ_CONVERSION_SHA = "<from Step 8.4>"
GNS_DAM2D_PKL_INFERENCE_SHA = "<from Step 8.4>"
GNS_DAM2D_NPZ_CONVERSION_SHA = "<from Step 8.4>"
```

- [ ] **Step 9.2: Extend the driver's case-pair iteration**

Find the `def main()` function in `emit_sarif.py`. There should be a loop that iterates over (model, dataset, dir, shas) tuples for the existing TGV2D pair. Extend it to include dam2d:

```python
# Pseudo-code; the actual structure depends on emit_sarif.py's existing shape.
# The principle: add 2 more iterations to whatever the existing loop is.

cases = [
    ("segnn", "tgv2d", LOCAL_MIRROR_ROOT / f"segnn_tgv2d_{SEGNN_PKL_INFERENCE_SHA}",
     SEGNN_PKL_INFERENCE_SHA, SEGNN_NPZ_CONVERSION_SHA),
    ("gns", "tgv2d", LOCAL_MIRROR_ROOT / f"gns_tgv2d_{GNS_PKL_INFERENCE_SHA}",
     GNS_PKL_INFERENCE_SHA, GNS_NPZ_CONVERSION_SHA),
    # NEW: rung 4c
    ("segnn", "dam2d", LOCAL_MIRROR_ROOT / f"segnn_dam2d_{SEGNN_DAM2D_PKL_INFERENCE_SHA}",
     SEGNN_DAM2D_PKL_INFERENCE_SHA, SEGNN_DAM2D_NPZ_CONVERSION_SHA),
    ("gns", "dam2d", LOCAL_MIRROR_ROOT / f"gns_dam2d_{GNS_DAM2D_PKL_INFERENCE_SHA}",
     GNS_DAM2D_PKL_INFERENCE_SHA, GNS_DAM2D_NPZ_CONVERSION_SHA),
]

for model, dataset, dir_path, pkl_sha, npz_sha in cases:
    # existing loop body (lint_npz_dir, run-level properties, emit_sarif call)
    ...
```

If `emit_sarif.py` currently hardcodes the case pairs (not in a loop), refactor minimally to introduce the loop, then add the 2 new tuples. Keep the change as small as possible.

- [ ] **Step 9.3: Pull dam2d rollouts to local mirror**

```bash
cd /Users/zenith/Desktop/physics-lint
modal volume get rollout-anchors-artifacts \
    /vol/rollouts/lagrangebench/segnn_dam2d_<sha>/ \
    external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror/segnn_dam2d_<sha>/
modal volume get rollout-anchors-artifacts \
    /vol/rollouts/lagrangebench/gns_dam2d_<sha>/ \
    external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror/gns_dam2d_<sha>/
```

(Note the trailing slash on source dirs per session-handover Modal convention.)

- [ ] **Step 9.4: Run emit_sarif and verify outputs**

```bash
cd /Users/zenith/Desktop/physics-lint
source .venv/bin/activate
python external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py 2>&1 | tail -20
ls -la external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/*dam2d*
```

Expected: 4 SARIFs total (2 existing TGV2D from rung-4a + 2 new dam-break from rung-4c).

Verify dam-break SARIF structure:

```bash
python -c "
import json
for path in ['external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/segnn_dam2d_<sha>.sarif',
             'external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/gns_dam2d_<sha>.sarif']:
    with open(path) as f:
        sarif = json.load(f)
    results = sarif['runs'][0]['results']
    rule_ids = set(r['ruleId'] for r in results)
    print(f'{path}: {len(results)} rows, rule_ids={sorted(rule_ids)}')
    skip_reasons = set(r['properties'].get('skip_reason', '') for r in results if 'skip_reason' in r['properties'])
    print(f'  unique skip_reasons: {len(skip_reasons)}')
    for sr in skip_reasons:
        print(f'    {sr[:120]}...')"
```

Expected:
- 60 rows per SARIF (3 rules × 20 trajs)
- 3 distinct rule_ids: `harness:mass_conservation_defect`, `harness:energy_drift`, `harness:dissipation_sign_violation`
- 2 distinct skip_reasons:
  - One D0-08 (KE-rest IC) on `energy_drift` rows
  - One D0-22 (open-driven-dissipative) on `dissipation_sign_violation` rows
- `mass_conservation_defect` rows are raw (no skip_reason), value 0.0 each

- [ ] **Step 9.5: Run renderer to verify substrate-agnosticism property**

This is the load-bearing test for §5.1 of the design doc — the renderer must handle dam-break SARIFs without modification.

```bash
cd /Users/zenith/Desktop/physics-lint
python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py \
    --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/ 2>&1 | tail -30
```

If the renderer's CLI takes `--sarif-dir` and includes all SARIFs in the dir, it'll produce a 4-stack table (segnn-tgv2d, gns-tgv2d, segnn-dam2d, gns-dam2d). If the CLI takes a pair of paths, run twice — once for the TGV2D pair (sanity check existing behavior unchanged) and once for the dam2d pair (rung-4c output).

Pass criteria: renderer produces output without raising; output shows the dam-break rows with D0-22 SKIPs alongside TGV2D rows with D0-18 SKIPs.

- [ ] **Step 9.6: Commit dam2d SARIFs + emit_sarif.py changes**

```bash
cd /Users/zenith/Desktop/physics-lint
git add external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py \
        external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/segnn_dam2d_<sha>.sarif \
        external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/gns_dam2d_<sha>.sarif
git commit -m "01-lagrangebench: rung 4c dam2d SARIF emission at sha <sha> (D0-22 SKIPs visible)"
```

---

## Task 10: Render dam-break cross-stack table

**Files:**
- Create: `external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/eps_table_dam2d_<sha>.md` *or* `dam2d_table_<sha>.md` (mirror rung-4a's naming convention; rung-4a had `eps_table_<sha>.md` for rung-4b but rung-4a's writeup used the renderer's text output inline)

- [ ] **Step 10.1: Render dam-break cross-stack table to a committed file**

If `render_cross_stack_table.py` supports `--output` flag, use it; otherwise pipe stdout:

```bash
cd /Users/zenith/Desktop/physics-lint
python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py \
    --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/ \
    --filter-dataset dam2d \
    > external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/dam2d_table_<sha>.md
```

If the renderer doesn't support `--filter-dataset`, run it on a dam2d-only subdirectory of SARIFs or post-filter the output.

- [ ] **Step 10.2: Inspect rendered table**

Open `external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/dam2d_table_<sha>.md`. Expected shape (mirroring rung-4a's table):

```markdown
| Rule | gns-dam2d | segnn-dam2d |
|---|---|---|
| `mass_conservation_defect` | 0.000e+00 (x20 identical) | 0.000e+00 (x20 identical) |
| `energy_drift` | SKIP (x20, D0-08) | SKIP (x20, D0-08) |
| `dissipation_sign_violation` | SKIP (x20, D0-22) | SKIP (x20, D0-22) |
```

If the row aggregation differs from rung-4a's shape, reconcile — either the renderer changed (unlikely, design says unchanged) or the SARIF rows aren't structurally identical (inspect; potential D0-19 §3.4 invariant violation).

- [ ] **Step 10.3: Commit rendered table**

```bash
cd /Users/zenith/Desktop/physics-lint
git add external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/dam2d_table_<sha>.md
git commit -m "01-lagrangebench/outputs/sarif: rung 4c dam2d cross-stack table rendered at sha <sha>"
```

---

## Task 11: Write rung-4c table writeup

**Files:**
- Create: `external_validation/_rollout_anchors/methodology/docs/2026-05-07-rung-4c-substrate-class-extension-table.md`

Mirror rung-4a's writeup shape. The writeup is short (~60-80 lines) — the design doc carries the methodology depth; the writeup is the rung's verdict-log artifact.

- [ ] **Step 11.1: Create writeup**

Create `external_validation/_rollout_anchors/methodology/docs/2026-05-07-rung-4c-substrate-class-extension-table.md`:

```markdown
# Rung 4c — Substrate-class extension to dam-break-2D (writeup)

**Date:** 2026-05-07
**Predecessor:** rung 4b T7 PASS on both stacks at sha `255af5de8d` (PR #8 merged at sha `<merge_sha>`); D0-22 pre-registered at `<D0-22 commit sha>`; rung-4c implementation at sha `<sarif_emission_sha>`.
**Successor:** integrating-README composing rung 4a + 4b + 4c (this writeup is the trigger).

**Design doc:** [`./2026-05-07-rung-4c-substrate-class-extension-design.md`](./2026-05-07-rung-4c-substrate-class-extension-design.md)
**Plan doc:** [`./2026-05-07-rung-4c-substrate-class-extension-plan.md`](./2026-05-07-rung-4c-substrate-class-extension-plan.md)
**SARIF artifacts:** [`../../01-lagrangebench/outputs/sarif/segnn_dam2d_<sha>.sarif`](../../01-lagrangebench/outputs/sarif/segnn_dam2d_<sha>.sarif), [`../../01-lagrangebench/outputs/sarif/gns_dam2d_<sha>.sarif`](../../01-lagrangebench/outputs/sarif/gns_dam2d_<sha>.sarif).
**Rendered table:** [`../../01-lagrangebench/outputs/sarif/dam2d_table_<sha>.md`](../../01-lagrangebench/outputs/sarif/dam2d_table_<sha>.md).
**Methodology pre-registrations:** [D0-22](../DECISIONS.md#d0-22--2026-05-07--rung-4c-substrate-class-extension-to-dam-break-2d-pre-registration), [D0-19](../DECISIONS.md#d0-19--2026-05-04--harness-sarif-result-schema-rung-4a-pre-registration), [D0-08](../DECISIONS.md#d0-08).

---

## Headline

physics-lint's harness substrate-detection layer extends to a second LagrangeBench substrate class — `open-driven-dissipative` — via D0-22's new SKIP path on `dissipation_sign_violation`. Rung-4a's TGV2D conservation rule schema (`harness:mass_conservation_defect`, `harness:energy_drift`, `harness:dissipation_sign_violation`) runs unmodified on dam-break-2D rollouts; per-stack rows are emitted in the same v1.0 SARIF schema as rung-4a, with `dam2d → "open-driven-dissipative"` reclassified empirically (KE(t) measured to rise during gravity-loaded fall at pre-flight smoke; commit `<smoke_sha>`) following the *classify when you exercise* discipline that rung-4b's PH-SYM-003 PBC-square-SO(2) substrate-incompatibility SKIP precedented.

The renderer (`methodology/tools/render_cross_stack_table.py`, EXPECTED_SCHEMA_VERSION=1.0) handles dam-break SARIFs **without modification** — load-bearing evidence that the harness's rule schema generalizes across substrate classes at the consumer side, not just at the generator side. The methodology contribution is bounded to two file-level edits (`particle_rollout_adapter.py` taxonomy + new SKIP gate); all downstream consumer code stays schema-compatible.

D0-22 is the third instance of the "source-review-catches-issue-before-compute" pattern in the rung-4 series: rung-4b first-pass math correction, rung-4b first-pass latent figure-sweep failure, rung-4c catalogue-misclassification (`dam2d → "dissipative"` was preemptive and wrong; surfaced at design-pass source review of `particle_rollout_adapter.py:255`). All three caught at $0 Modal cost. To be elevated as a first-class methodology contribution in integrating-README composition.

---

## Cross-stack conservation table — dam-break-2D

| Rule | gns-dam2d | segnn-dam2d |
|---|---|---|
| `mass_conservation_defect` | <PASTE FROM Step 10.2 OUTPUT> | <PASTE FROM Step 10.2 OUTPUT> |
| `energy_drift` | <PASTE> | <PASTE> |
| `dissipation_sign_violation` | <PASTE> | <PASTE> |

**Provenance (D0-19 three-sha):**

- **gns-dam2d**: pkl_inference=<from Step 8.4>, npz_conversion=<from Step 8.4>, sarif_emission=<sarif_emission_sha>
- **segnn-dam2d**: pkl_inference=<from Step 8.4>, npz_conversion=<from Step 8.4>, sarif_emission=<sarif_emission_sha>

**Empirical justification for `dam2d → "open-driven-dissipative"`:**

Pre-flight 1-traj smoke at sha `<smoke_sha>` measured KE(t) on both stacks:
- SEGNN-dam2d: KE(0)=<X>, max(KE)=<Y> at t=<peak_t>, KE(end)=<Z>; rises_anywhere=True
- GNS-dam2d: KE(0)=<X>, max(KE)=<Y> at t=<peak_t>, KE(end)=<Z>; rises_anywhere=True

Both stacks confirm the gravity-loaded rise-then-fall shape that justifies the reclassification. Pre-flight log: [`preflight/2026-05-07-rung-4c.txt`](../../../preflight/2026-05-07-rung-4c.txt).

---

## What rung 4c is NOT

(Verbatim from design §1.3 deferral list — copy-paste at writeup time.)

1. Not a bilateral test of D0-18 — D0-08 (KE-rest IC) fires on dam-break `energy_drift`, not D0-18; the bilateral D0-18 forward-flag from rung-4a §1.3 (5) stays intact.
2. Not a SEGNN-vs-GNS model comparison — both stacks emit identical structural rows.
3. Not the integrating top-level README — composes downstream.
4. Not a wall-non-penetration claim — plan v2.1 amendment removes "PH-BC (wall)" from §3.1 P1.
5. Not a multi-rung renderer — rung 4c stands alone.
6. Not a catalogue-wide reclassification — only `dam2d` reclassified empirically; rpf2d/ldc2d/rpf3d/ldc3d forward-flagged.
7. Not a multi-rule-trigger-axis abstraction — D0-21 §forward-flag-2 named the "(rule, substrate) compatibility matrix" as future work; rung 4c demonstrates its empirical instance without promoting.

---

## Rederivability

Rendered at physics-lint `feature/rung-4c-substrate-class-extension` sha `<sarif_emission_sha>` via:

```bash
python external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py
python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py \
    --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/ \
    --filter-dataset dam2d \
    > external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/dam2d_table_<sha>.md
```

Re-run the commands at the same sha against the committed dam-break SARIFs at that sha → identical output.

---

## Methodology contributions (durable)

1. **Substrate-class extension as a first-class methodology operation.** D0-22's structure (taxonomy entry + new SKIP path + empirical justification + forward-flag two-tier split) is the template future rungs use when extending to additional substrate classes (case study 02 PhysicsNeMo MGN; future LB substrates). The pattern reads as: pre-register the taxonomy entry → measure the substrate empirically → land the new SKIP path with reason template → forward-flag related substrates pending probe.

2. **"Classify when you exercise" empirical-classification principle.** Bilateral now across rung-4b (PH-SYM-003 PBC-square-SO(2) SKIP) and rung-4c (`dam2d` reclassification + rpf/ldc/3D forward-flag). Pattern reads as: substrate properties get verdicts only after empirical probing, never on theoretical intuition alone.

3. **Source-review-catches-issue-before-compute pattern (trilateral).** Three instances at $0 Modal cost in the rung-4 series; pre-flight discipline is paying out. Pattern reads as: a source-review pre-flight pass between design and execution catches issues that brainstorm-only and execution-only review miss.

These three contributions, alongside rung-4a's "schema-uniform-across-stacks" headline and rung-4b's "tripartite-evidence-framing-grouped-by-construction-trivial-vs-architectural-vs-skip" framing, are the durable methodology outputs of the rung-4 series. Composed in integrating-README at `methodology/README.md` (next deliverable).

---

## Integrating-README trigger

This dated writeup is the named-event trigger for integrating-README composition at `methodology/README.md`. Composes rung-4a (cross-stack conservation, TGV-2D) + rung-4b (cross-stack equivariance, TGV-2D) + rung-4c (substrate-class extension, dam-break-2D) into one cross-rung methodology artifact. Foregrounds:

- Rung-4a headline: schema-uniform machinery across SEGNN + GNS on conservation
- Rung-4b §3.2 architecture-claim coupling + obs (4) GNS-translation FP-noise-bounded by LB feature pipeline
- Rung-4c §1.2 substrate-class extension headline + "classify when you exercise" + source-review trilaterality

Composition deliverable lands as a separate task per the design doc §9 anchor; this writeup gates it.
```

Replace `<...>` placeholders with concrete values from previous tasks. The rendered table cell values come from Step 10.2 inspection.

- [ ] **Step 11.2: Commit writeup**

```bash
cd /Users/zenith/Desktop/physics-lint
git add external_validation/_rollout_anchors/methodology/docs/2026-05-07-rung-4c-substrate-class-extension-table.md
git commit -m "methodology/docs: rung 4c substrate-class-extension writeup at sha <sha>"
```

---

## Task 12: Write plan v2.1 amendment

**Files:**
- Create: `external_validation/_rollout_anchors/methodology/docs/physics-lint-validation-plan-v2.1.md`

- [ ] **Step 12.1: Locate plan v2 sections to amend**

Open `external_validation/_rollout_anchors/methodology/docs/physics-lint-validation-plan-v2.md`. Note the line ranges of:
- §3.1 (Targets table)
- §3.2 step 6 (Capture + write up subsection template)
- §5.3 (Application integration / cover letter)
- §6 (Risk register)
- §11 if exists, else end-of-doc for changelog appendix

- [ ] **Step 12.2: Create plan v2.1 as a separate doc**

Create `external_validation/_rollout_anchors/methodology/docs/physics-lint-validation-plan-v2.1.md`:

```markdown
# Physics-lint validation plan — v2.1

**Date:** 2026-05-07
**Status:** v2.1 amendment of `physics-lint-validation-plan-v2.md`. v2 stays frozen at its original path; v2.1 is the corrected document.
**Trigger:** Rung 4c design pass surfaced that v2 §3.1 P1's "PH-BC (wall)" entry assumed an SPH-particle-wall rule that does not exist in physics-lint v1.0. The plan-vs-actual-rule mismatch was caught at source review of the production rule set, before any rung-4c implementation. Plan v2.1 corrects the row honestly.
**Pattern:** Third instance of source-review-catches-issue-before-compute in the rung-4 series (rung-4b math, rung-4b figure-sweep, rung-4c catalogue-misclassification + plan-v2-rule-mismatch).

---

## Diff from v2

### §3.1 P1 row update

**Before (v2):**

```
| P1 | Dam break 2D | GNS | PH-CON-001 (mass), PH-BC (wall) |
```

**After (v2.1):**

```
| P1 | Dam break 2D | GNS + SEGNN | Substrate-class extension to open-driven-dissipative
                                    (D0-22): PH-CON-001 mass ACTIVE +
                                    dissipation_sign_violation SKIP (new) +
                                    energy_drift SKIP (D0-08 KE-rest, existing) |
```

**Rationale:** "PH-BC (wall)" assumed an SPH-particle-wall rule that doesn't exist. The headline-rule column updates to the substrate-class-extension framing per rung-4c design doc §1.2. Architecture column expands `GNS → GNS + SEGNN` to dual-stack scope (the cross-stack uniformity *is* the cross-validation in the substrate-class-extension framing).

### §3.1 P3 row absorption

**Before (v2):**

```
| P3 | Dam break 2D | SEGNN | Cross-validate P1 result |
```

**After (v2.1):** struck. Dual-stack P1 absorbs P3's SEGNN-dam2d cross-validation goal.

### §3.2 step 6 subsection template

**Before (v2):** included a `PH-BC-001` row in the dam-break per-(dataset, model) writeup template.

**After (v2.1):** strikes the `PH-BC-001` row. Dam-break per-stack template:

```markdown
### Dam break 2D — <stack>
- Checkpoint: <stack>_dam2d, best/, SHA-256 <hash>
- Rollout: 20 trajectories × 100 steps
- PH-CON-001 (`harness:mass_conservation_defect`): PASS-equivalent (raw=0.0, x20 identical)
- D0-22 (`harness:dissipation_sign_violation`): SKIP-with-reason (open-driven-dissipative)
- D0-08 (`harness:energy_drift`): SKIP-with-reason (KE-rest IC)
```

### §5.3 cover-letter paragraph

**Before (v2):** mentioned "PH-BC" in the LagrangeBench case-study sentence.

**After (v2.1):** drops the "PH-BC" reference; picks up substrate-class-extension framing as the dam-break headline contribution. Concretely, the cover-letter dam-break sentence becomes:

> "...extending physics-lint's harness substrate-detection layer to a second LagrangeBench substrate class (open-driven-dissipative) via dam-break-2D (D0-22), with the same SARIF schema as TGV-2D conservation and no consumer-side accommodation needed."

### §6 risks register

**v2.1 adds:**

> *Plan-vs-actual-rule mismatch surfaced during implementation.* Plan v2 §3.1 P1 specified "PH-BC (wall)" for dam-break-2D, but no SPH-particle wall rule existed in physics-lint v1.0 (PH-BC-001 in production is Dirichlet boundary trace on a unit square, a mesh-FEM rule). The mismatch was caught at rung-4c design pass via source review of the production rule set, before any implementation. Plan v2.1 corrects the row honestly. Pattern is the third instance of source-review-catches-issue-before-compute in the rung-4 series. Mitigation forward: when planning future rungs that name-reference rule IDs, verify the named rule exists in `external_validation/PH-*/` at design time, not at writeup time.

### Changelog (new §11)

```markdown
## §11. Changelog

### v2.1 — 2026-05-07

- §3.1 P1 row: drop "PH-BC (wall)" (no extant rule); replace with substrate-class-extension framing per D0-22.
- §3.1 P1 architecture: `GNS → GNS + SEGNN` (dual-stack scope absorbs P3).
- §3.1 P3 row: struck (absorbed by dual-stack P1).
- §3.2 step 6 dam-break template: drop PH-BC-001 row.
- §5.3 cover-letter: drop "PH-BC"; pick up substrate-class-extension framing.
- §6 risks: add plan-vs-actual-rule mismatch as meta-risk; record source-review-pattern third instance.

**Source-review-correction acknowledgment.** This v2.1 amendment is the third instance of the source-review-catches-issue-before-compute pattern in the rung-4 series, alongside rung-4b's first-pass math correction and first-pass figure-sweep failure (both surfaced at LB source review at sha b880a6c84a93792d2499d2a9b8ba3a077ddf44e2 between rung-4b first-pass and second-pass fix). All three caught at $0 Modal cost. The pattern paralleling rung-4b amendment 2 §14.6: a source-review pre-flight pass between design and execution catches issues that brainstorm-only and execution-only review miss; the cost is hours of source reading, the saving is multiple GPU runs and methodology errors that would otherwise land in writeups.
```
```

- [ ] **Step 12.2: Commit plan v2.1**

```bash
cd /Users/zenith/Desktop/physics-lint
git add external_validation/_rollout_anchors/methodology/docs/physics-lint-validation-plan-v2.1.md
git commit -m "methodology/docs: physics-lint-validation-plan-v2.1 — drop PH-BC (wall) from §3.1 P1; substrate-class-extension framing per D0-22 (rung 4c)"
```

---

## Final verification

- [ ] **Step F1: Run full test suite**

```bash
cd /Users/zenith/Desktop/physics-lint
source .venv/bin/activate
pytest --import-mode=importlib external_validation/ -o "addopts=" 2>&1 | tail -5
```

Expected: `453 passed, 1 skipped` (447 baseline + 6 new from `test_d0_22_open_driven_skip.py`).

- [ ] **Step F2: Verify acceptance criteria from design §8**

Walk through design doc §8's checkbox list. For each item, point to the commit/file/test that satisfies it. Any unsatisfied item: open as a follow-up before declaring rung 4c PASS.

- [ ] **Step F3: Check git log shape**

```bash
cd /Users/zenith/Desktop/physics-lint
git log --oneline feature/rung-4b-t7-subseq-length-fix..HEAD
```

Expected: ~7-8 commits, each clean and atomic:
1. `methodology/DECISIONS: pre-register D0-22 ...`
2. `_harness: implement D0-22 ...`
3. `01-lagrangebench/modal_app: add P1 SEGNN-dam2d + P1 GNS-dam2d ...`
4. `preflight: rung 4c 5-step checklist log (PROCEED|ABORT)`
5. `01-lagrangebench: rung 4c dam2d SARIF emission ...`
6. `01-lagrangebench/outputs/sarif: rung 4c dam2d cross-stack table ...`
7. `methodology/docs: rung 4c substrate-class-extension writeup ...`
8. `methodology/docs: physics-lint-validation-plan-v2.1 ...`

- [ ] **Step F4: Open PR (only after PR #8 has merged to master)**

```bash
cd /Users/zenith/Desktop/physics-lint
git push -u origin feature/rung-4c-substrate-class-extension
gh pr create --title "rung 4c: substrate-class extension to dam-break-2D (D0-22)" --body "$(cat <<'EOF'
## Summary

- Pre-registers D0-22: substrate-class extension to open-driven-dissipative + new SKIP path on dissipation_sign_violation
- Empirically reclassifies `dam2d` from "dissipative" to "open-driven-dissipative" after pre-flight smoke confirms rise-then-fall KE shape
- Extends Modal infrastructure for SEGNN-dam2d + GNS-dam2d 20-traj rollouts
- Commits 2 dam-break SARIFs + cross-stack table + writeup
- Plan v2.1 amendment drops "PH-BC (wall)" from §3.1 P1 (no extant rule); substrate-class-extension framing per D0-22

## Test plan
- [ ] D0-22 SKIP-gate tests pass (6/6)
- [ ] D0-18 SKIP-gate tests pass (regression check on dam2d label flip)
- [ ] Full test suite passes (453 passed, 1 skipped)
- [ ] Renderer produces dam-break cross-stack table without modification
- [ ] Pre-flight 5-step checklist log committed

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

If PR #8 has not yet merged, hold the rung-4c PR until it does — rung-4c branches off PR #8 and rebasing post-merge is cleaner than mid-flight.

---

## Compute and budget summary

| Stage | Compute | $ (A10G @ $0.86/hr) |
|---|---|---|
| Pre-flight Step 1 (data inspection) | ~30 s CPU Modal | < $0.01 |
| Pre-flight Step 4 SEGNN-dam2d 1-traj | ~1 min A10G | $0.014 |
| Pre-flight Step 4 GNS-dam2d 1-traj | ~1 min A10G | $0.014 |
| Production SEGNN-dam2d 20-traj | ~5 min A10G | $0.072 |
| Production GNS-dam2d 20-traj | ~3 min A10G | $0.043 |
| **Total** | **~10 min A10G + ~30 s CPU** | **~$0.15** |

Margin under design §3.4's $0.20 target: ~$0.05.

---

## Predecessor → successor

- **Predecessor:** rung 4b T7 PASS (PR #8 in flight at sha `255af5de8d`); design doc `2026-05-07-rung-4c-substrate-class-extension-design.md`.
- **This document:** rung 4c implementation plan.
- **Then:** rung 4c execution per the 12 tasks above.
- **Then:** rung 4c writeup commits at Task 11.
- **Then (separate task, named-event-triggered):** integrating-README composition at `methodology/README.md`.
