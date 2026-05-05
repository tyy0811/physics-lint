# Rung 4b — Cross-stack equivariance SARIF + writeup table (design)

**Date:** 2026-05-05
**Repo:** physics-lint
**Branch (primary):** `feature/rung-4b-equivariance`, off `master` at `ba13b45` (post-rung-4a-cleanup; rung 4a's `feature/rollout-anchors` merged at PR #6, commit `68d84c7`).
**Status:** Design — pre-implementation. Pre-registers **D0-21** in `external_validation/_rollout_anchors/methodology/DECISIONS.md` before any code change.
**Predecessor:** Rung 4a — cross-stack conservation (`2026-05-04-rung-4a-cross-stack-conservation-table.md`); SEGNN/GNS-TGV2D rollout npzs frozen on Modal Volume; D0-18, D0-19, D0-20 landed on `master`.
**Successor:** Rung 4b implementation plan (`2026-05-05-rung-4b-equivariance-plan.md`); then execution; then `-table.md`.

---

## 1. Scope and framing

### 1.1 What rung 4b is

Rung 4b extends rung 4a's cross-stack conservation result to equivariance. It reuses the 40 frozen `particle_rollout_traj{NN}.npz` artifacts on Modal Volume from rung 4a (20 SEGNN-TGV2D + 20 GNS-TGV2D) as the reference state $x_0$ source; on A10G, generates first-step ε rows for SARIF classification and a small ε(t) trajectory subset for the writeup figure; packages into committed harness-style SARIF artifacts (schema_version `v1.1`) via the same `_harness/sarif_emitter.py` pipeline; renders a cross-stack equivariance table grouped by tripartite evidence type; writes a dated methodology table writeup at `methodology/docs/2026-05-05-rung-4b-equivariance-table.md`.

### 1.2 Load-bearing claim (frozen headline sentence)

> *"physics-lint's PH-SYM rule schema, with equivariance thresholds set at the float32 numerical-precision floor (ε ≤ 10⁻⁵ PASS, 10⁻⁵ < ε ≤ 10⁻² APPROXIMATE, ε > 10⁻² FAIL), runs unmodified across SEGNN-TGV2D and GNS-TGV2D rollouts; per-stack ε values are emitted in the same SARIF schema as 4a (schema_version v1.1) and reported as observed."*

This sentence is the writeup's lede and is frozen at design time to prevent narrative drift during implementation. The headline is robust to both probe outcomes — SEGNN at the float32 floor (expected, uninteresting confirmation) and GNS at either the floor (TGV2D-specific finding) or in (10⁻⁵, 10⁻²] (consistent-with-architecture-level prior). Neither outcome rewords the headline.

### 1.3 What rung 4b is NOT (explicit deferral list, signposted in the writeup body)

1. **Not a SEGNN-vs-GNS differentiation claim.** If observed ε values place SEGNN at PASS and GNS at APPROXIMATE — the v2-plan-headline outcome — that lands as a *secondary finding* in the writeup body ("consistent with Helwig et al. ICML 2023's architecture-level characterization of GNS as approximately equivariant"), not as the load-bearing claim. The artifact stands or falls on the rule-schema-portability claim, which is independent of the SEGNN/GNS comparison's empirical direction.

2. **Not a cross-framework portability claim.** TGV2D / LagrangeBench only. PhysicsNeMo MGN equivariance (e.g., PH-SYM-002 reflection on Ahmed Body) is a separate rung scoped under case study 02, not bundled here.

3. **Not a multi-dataset claim.** TGV2D only. RPF2D, LDC2D, dam-break, and other LagrangeBench datasets are out of scope.

4. **Not an equivariance-coverage-completeness claim.** PH-SYM-001 tests C₄ at θ ∈ {π/2, π, 3π/2} plus θ = 0 as a construction-trivial smoke test; PH-SYM-002 tests one reflection axis (y-axis through box center); PH-SYM-004 tests one translation vector; PH-SYM-003 ships as substrate-incompatibility SKIP. Exhaustive group coverage is future work.

5. **PH-SYM-003 / PBC-square IS the analogous SKIP mechanism (meta-correction).** The original brainstorm framing claimed PH-SYM rules had no analogue to PH-CON-002's dissipative-system SKIP. The design pass surfaced that PH-SYM-003 (continuous SO(2) rotation) on a periodic-square box *is* structurally unmeasurable — non-{0, π/2, π, 3π/2} rotations don't preserve the periodic geometry; the rotated unit cell doesn't tile with the original. This is exactly analogous to PH-CON-002 / dissipative: substrate makes the rule's classification semantics inapplicable, and the principled response is SKIP-with-reason rather than emit-a-confounded-number. Pulling this into design rather than discovering mid-execution is recorded here as a self-validating instance of the writeup-framing-baked-into-design-before-code discipline.

### 1.4 D-entries 4b creates

- **D0-21** — *Rung 4b pre-registration: SARIF schema_version bump v1.0 → v1.1; PH-SYM rule emissions; tripartite evidence framing; PH-SYM-003 substrate-incompatibility SKIP mechanism; (rule, substrate) compatibility matrix forward-flag generalizing D0-18 beyond system_class detection; trigger-vs-emission separation discipline for SKIP paths; A10G GPU-class consistency generalizing D0-17 amendment 1's matched-stack-consistency principle to GPU class; sibling renderer choice; ε(t) npz schema with 4-stage provenance generalizing D0-19's 3-stage shape; Modal-Volume-only persistence parallel to 4a; rotated state intentionally non-persisted (recovery via re-run).*

D0-21 lands as a single composite entry rather than multiple small entries to keep the rung 4b pre-registration legible as one document. Subsidiary amendments are added under D0-21 footers as implementation surfaces them.

---

## 2. Architecture

### 2.1 Cross-subtree split (post-rung-4a)

```
physics-lint repo (branch: feature/rung-4b-equivariance, off master)
└── external_validation/_rollout_anchors/
    ├── _harness/
    │   ├── symmetry_rollout_adapter.py    [NEW: PH-SYM ε computation primitives]
    │   ├── sarif_emitter.py               [EDIT: schema_version v1.1, PH-SYM extra_properties]
    │   ├── lint_eps_dir.py                [NEW: ε(t) npz dir → HarnessResults, parallel to lint_npz_dir]
    │   ├── SCHEMA.md                      [EDIT: §3.x ε(t) npz schema; §3.y schema_version v1.1]
    │   └── tests/
    │       ├── test_symmetry_rollout_adapter.py    [NEW]
    │       ├── test_lint_eps_dir.py                [NEW]
    │       └── fixtures/
    │           ├── c4_symmetric_4particle.py       [NEW: positive-path]
    │           ├── c4_breaking_4particle.py        [NEW: negative-path]
    │           ├── reflection_symmetric.py         [NEW: positive-path]
    │           └── pbc_square_so2_skip.py          [NEW: SKIP-mechanism fixture]
    ├── 01-lagrangebench/
    │   ├── modal_app.py                   [EDIT: add lagrangebench_eps_p0_segnn_tgv2d, _p1_gns_tgv2d]
    │   ├── emit_sarif_eps.py              [NEW: case-study driver, parallel to emit_sarif.py]
    │   ├── outputs/
    │   │   ├── sarif/
    │   │   │   ├── segnn_tgv2d_eps_<sarif_emission_sha>.sarif   [NEW: committed]
    │   │   │   └── gns_tgv2d_eps_<sarif_emission_sha>.sarif     [NEW: committed]
    │   │   ├── trajectories/                                    [NEW dir; gitignored *.npz]
    │   │   │   ├── .gitkeep                                     [committed]
    │   │   │   ├── segnn_tgv2d_<eps_computation_sha>/           [Modal-Volume-mirrored]
    │   │   │   │   └── eps_traj{NN}_{rule}_{transform}.npz      [gitignored]
    │   │   │   └── gns_tgv2d_<eps_computation_sha>/             [Modal-Volume-mirrored]
    │   │   └── figures/
    │   │       └── eps_t_segnn_vs_gns_pi_2.{png,pdf}            [gitignored]
    │   ├── tests/
    │   │   └── test_emit_sarif_eps.py     [NEW]
    │   └── .gitignore                     [EDIT: add trajectories/*/*.npz]
    └── methodology/
        ├── DECISIONS.md                   [EDIT: append D0-21]
        ├── docs/
        │   ├── 2026-05-05-rung-4b-equivariance-design.md    [THIS DOC]
        │   ├── 2026-05-05-rung-4b-equivariance-plan.md      [next deliverable]
        │   └── 2026-05-05-rung-4b-equivariance-table.md     [post-execution]
        └── tools/
            └── render_eps_table.py        [NEW: sibling renderer for v1.1, not extension]
```

### 2.2 Reuse from rung 4a (unchanged)

- `_harness/sarif_emitter.py` core machinery (HarnessResult dataclass, run-level/result-level property contract, three-stage sha provenance) — extended for v1.1, not rewritten.
- `_harness/particle_rollout_adapter.py` (`load_rollout_npz`, kinetic_energy_series) — re-imported as the reference x_0 source for ε computation.
- 4a's 40 frozen `particle_rollout_traj{NN}.npz` artifacts on Modal Volume — reference x_0 source for the ε computation; not re-generated.
- D0-19's three-stage sha provenance (`pkl_inference`, `npz_conversion`, `sarif_emission`) — extended to four-stage for ε(t) npzs (adds `eps_computation`); 4a's reference rollout shas carried through unchanged.
- D0-20's generator-vs-consumer separation — extended to ε domain: Modal generates ε(t) npzs; consumer (`lint_eps_dir`) reads them and emits SARIF.

### 2.3 New code surface

Five new modules:
- `_harness/symmetry_rollout_adapter.py` — rotation/reflection/translation primitives + ε computation per (rule, transform_param, traj_index)
- `_harness/lint_eps_dir.py` — ε(t) npz dir → HarnessResult rows (parallel to 4a's `lint_npz_dir`)
- `01-lagrangebench/emit_sarif_eps.py` — case-study driver (parallel to 4a's `emit_sarif.py`)
- `methodology/tools/render_eps_table.py` — sibling renderer for schema v1.1 (see §5.1 for the sibling-vs-extend decision)
- New Modal entrypoints in `01-lagrangebench/modal_app.py` for ε generation

---

## 3. Detailed design

### 3.1 Calibration framing — probe-as-calibration with float32 floor

The threshold rationale does not depend on a published-precision citation. The 4b artifact's own measurement on TGV2D *is* the calibration anchor; the threshold band (PASS/APPROXIMATE/FAIL) is defined by an architecture-agnostic float32 numerical-precision argument (§3.3). Helwig et al. ICML 2023 is cited in the writeup body as architecture-level context for interpreting observed GNS values — *contextualizing the result*, not anchoring the threshold. This treats Helwig honestly (the published characterization is architecture-level, primarily on dam-break, not specifically on TGV2D) and lets the threshold reasoning stand on its own foundation.

The original v2-plan framing ("calibrated against Helwig et al.'s published precision on the TGV2D dataset family") was a category mismatch — Helwig does not publish dataset-level precision on TGV2D, and the architecture-level characterization can't be relocalized to a dataset-level calibration claim. The probe-as-calibration framing avoids this and is robust to both probe outcomes without rewording.

### 3.2 Rule set (γ) + tripartite evidence framing

Rung 4b emits four PH-SYM rules in the SARIF, mirroring rung 4a's three active + one SKIP shape. Each rule's ε emits a result row per (model, transform_param, traj_index) with first-step scalar ε plus per-row provenance.

| Rule | Status on TGV2D | Transform | Evidence type |
|---|---|---|---|
| PH-SYM-001 | active at θ ∈ {π/2, π, 3π/2} | rotation about (L/2, L/2) | architectural |
| PH-SYM-001 | smoke-test at θ = 0 | identity | construction-trivial |
| PH-SYM-002 | active | reflection across y-axis through (L/2, ·) | architectural |
| PH-SYM-003 | SKIP-with-reason | continuous SO(2) | substrate-incompatible |
| PH-SYM-004 | active | translation by t = (L/3, L/7) | construction-trivial |

PH-SYM-003 emits with `raw_value = None`, `level = "note"`, `message = "SKIP: PBC-square breaks SO(2) symmetry — rotated cell doesn't tile with original"`, and `extra_properties.skip_reason` = same string. This reuses D0-19 §3.4's skip_reason discipline (guaranteed-identical-across-rows-within-(rule, stack), recomputed per row). The trigger logic ("substrate has periodic boundaries AND rotation angle is non-trivial-symmetry of substrate cell") lives inside `symmetry_rollout_adapter.py`, distinct from the shared D0-19 emission machinery — see §3.6.

The writeup body carries a short "what each PH-SYM rule measures on this substrate" subsection that groups the rule rows by evidence type rather than by rule id:

- **Architectural-evidence rows** (PH-SYM-001 at active angles, PH-SYM-002 reflection): ε reflects the model's per-step equivariance under discrete rotations/reflections; the load-bearing measurement for the headline.
- **Construction-trivial rows** (PH-SYM-001 at θ=0, PH-SYM-004 translation): ε at machine-zero by construction. Identity rotation is trivially equivariant; translation + PBC commute exactly. Both serve as smoke tests (failure indicates a real bug in the rotation/translation mechanic) but do not differentiate models.
- **Substrate-incompatible SKIP** (PH-SYM-003): the rule schema correctly emits SKIP-with-reason rather than reporting a confounded numerical value when the substrate's symmetry doesn't include the rule's transformation group.

This grouping prevents the reader from interpreting four PASSes as the same kind of evidence — the ε values for SEGNN at θ=π/2 (architectural property: per-step C₄ equivariance) and at θ=0 (substrate property: identity is trivially preserved) measure structurally different things, and the writeup needs to make that visible.

### 3.3 Threshold rationale — float32 numerical-precision floor

An exactly E(3)-equivariant network operating in float32 has equivariance error bounded by accumulated round-off, roughly $10^{-7}$ per operation, growing with network depth and condition number. For depth-O(10) networks operating on moderately-conditioned inputs, accumulated round-off lands in the $10^{-7}$–$10^{-6}$ range.

- **PASS:** ε ≤ 10⁻⁵. Above pure numerical noise (so the test isn't measuring nothing) but well below architectural-approximation levels (so an exactly-equivariant network reliably passes with a ~1–2 order-of-magnitude margin).
- **APPROXIMATE:** 10⁻⁵ < ε ≤ 10⁻². Visible architectural approximation at the percent level but not gross failure.
- **FAIL:** ε > 10⁻². Gross failure of the equivariance claim.

The argument is principled (numerical-floor-based), architecture-agnostic, and requires no cross-paper citation at the threshold layer. Helwig et al. ICML 2023's architecture-level characterization moves to writeup body context.

**Forward flag (case study 02):** the float32 floor argument is precision-dependent. If PhysicsNeMo MGN ever runs at float64, the same threshold band would mean different things (float64 floor is ~$10^{-16}$ per op). The threshold rationale needs revisiting at that boundary; D0-21 captures this as a precision-flag against future cross-precision validation.

### 3.4 Reportable ε shape

**SARIF (load-bearing):**
- One scalar ε per (rule, model, transform_param, traj_index), the first-step ε:
  $$ \epsilon = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \| R^{-1} f^1(R x_0)_i - f^1(x_0)_i \|_2^2 } $$
  where $r_i = f^1(x_0)_i$ is the per-particle position at step 1, $R$ is the transform (rotation, reflection, or translation), and the per-particle squared-norm is averaged across $N$ particles before sqrt (RMS-across-particles aggregation, **pinned in `SCHEMA.md` §3.x for PH-SYM**).
- Compared quantity: positions only (P). Velocity equivariance is out of scope; if it ever becomes interesting, it gets a separate rule (PH-SYM-005-ish), not an extra column on PH-SYM-001.
- One scalar per row matches D0-19's clean per-row shape unchanged.

**Artifact tier (uniform; mixed first-step + trajectory):**
- Every (rule, model, transform_param, traj_index) tuple persists one ε(t) npz with shape `(T_steps,)`. For non-figure-subset tuples, `T_steps = 1` (just first-step ε; npz is ~hundreds of bytes). For figure-subset tuples, `T_steps = 100` (full ε(t) trajectory for the writeup figure; npz is ~few KB).
- Single schema, single consumer code path: `lint_eps_dir` reads `eps_t[0]` from every npz to populate the SARIF scalar; the figure renderer filters for npzs with `len(eps_t) > 1` to identify figure traces.
- This preserves rung 4a's single-artifact-tier structure (4a has one type — rollout npzs — and one consumer — `lint_npz_dir`). A second artifact type for figure-only data would break the 4a-parallel claim that the headline's "same SARIF schema as 4a" partly leans on.
- Persisted to `outputs/trajectories/{model}_{dataset}_<eps_computation_sha>/eps_{rule}_{transform_kind}_{transform_param}_traj{NN}.npz`.

**Figure subset:** 1 angle (θ = π/2, the most visually intuitive C₄ rotation) × 3 trajs per stack × 2 stacks = **6 traces total**. Identified at generation time by a fixed selection on `traj_index` (e.g., trajs 0, 7, 14 to span the 20-traj index range); persisted with `T_steps = 100` instead of `T_steps = 1`. Trajectory ε(t) is *not threshold-classified* — classification is on `eps_t[0]` only; the figure shows error evolution as supplementary context. This is the load-bearing-vs-supplementary asymmetry rung 4b adopts, mirroring 4a's pattern: SARIF carries point defects, writeup figure visualizes time-series.

**Why first-step and not trajectory-aggregated:** trajectory-aggregated ε confounds equivariance error with chaotic divergence under non-symplectic integration — two slightly-perturbed trajectories diverge in time even for an exactly-equivariant network. Threshold-classifying on a trajectory aggregate (max_t, mean_t, ε(T)) would tangle architecture with integrator stability, making PASS/FAIL a joint property rather than a clean architectural test. First-step ε measures only the architecture's equivariance claim, applies the float32 floor argument cleanly, and is cheap (sub-minute total compute on A10G).

### 3.5 Mechanic correctness primitives

Four forced choices — pinned to prevent the canonical equivariance-test bugs:

1. **Rotation pivot / reflection axis through box center (L/2, L/2).** Rotations pivot at (L/2, L/2); reflections are across an axis through (L/2, L/2) (e.g., y-axis reflection is across the line x = L/2). Box-center C₄ maps [0, L]² → [0, L]² directly. Origin-rotation requires post-translation bookkeeping that produces a structurally identical answer through a more bug-prone path; pinning to box-center eliminates the class entirely.

2. **Velocities transform with the same R as positions.** $v'_i = R v_i$ for rotation; $v'_x = -v_x$ (and $v'_y$ unchanged) for y-axis reflection; $v'_i = v_i$ for translation (translation matrix is identity on tangent vectors). The canonical bug — rotating $r$ but leaving $v$ in the original frame — measures a hybrid of equivariance violation and rotation misapplication, and is not what the test claims to measure.

3. **PBC `mod L` wrap after every transform.** Box-center C₄ keeps [0, L]² → [0, L]² in principle; floating-point arithmetic at cell boundaries may push positions slightly outside, and `mod L` restores cell membership. Without the wrap, the transformed state isn't a valid input to $f$.

4. **Reference x_0 source: step-0 frames of rung 4a's existing 40 rollout npzs.** Same checkpoint, same x_0, same provenance chain (D0-19's `pkl_inference` and `npz_conversion` shas carried through). No new IC-generation logic.

**Translation vector specifics:** $t = (L/3, L/7)$. Translation + PBC commute exactly is a structural property of the substrate, not a property of the $t$ value, so the choice doesn't affect *whether* PH-SYM-004 passes — any non-zero $t$ gives ε at machine-zero by construction. (L/3, L/7) is non-grid-commensurate (the rationals 1/3 and 1/7 do not share a common factor with typical particle-grid spacings); this avoids any accidental commensurability with structure in $x_0$ that would muddy the diagnostic interpretation if a real bug ever showed up. The "non-grid-commensurate" framing is technically precise; "incommensurate with L" would be mathematically wrong since L/3 and L/7 are rational fractions of L.

**Angle-set specifics for PH-SYM-001 (C₄):** the v2 plan's $\{0, \pi/4, \pi/2, \pi, 3\pi/2\}$ mixed C₄ with one C₈ angle — and π/4 doesn't preserve the periodic-square cell, so its ε would be substrate-confounded just like SO(2). π/4 is dropped from rung 4b entirely. Angle 0 is retained as a construction-trivial smoke-test row (ε = 0 by construction; failure indicates a real rotation-mechanic implementation bug, e.g., wrap-after-rotate breaking at identity).

### 3.6 Pipeline & artifact structure

**Persistence model: Modal-Volume-only (matches 4a's pattern).**
- `outputs/trajectories/*.npz` is gitignored locally; SARIF + verdicts are committed.
- Local `_local_mirror/` directory exists for development convenience, not committed.
- Figure reproducibility is via sub-minute Modal re-run, identical to 4a's reproducibility model. Committing ε(t) npzs would buy "figure reproducible from repo state alone" but at the cost of breaking the 4a-parallel structural claim that the headline's "same SARIF schema as 4a" leans on. Modal-Volume-only is the right answer.

**Generator/consumer split (D0-20 reused unchanged):**
- Modal-side generator: `lagrangebench_eps_p0_segnn_tgv2d` and `_p1_gns_tgv2d` entrypoints in `modal_app.py` load checkpoints, iterate over (rule, transform_param, traj_index), forward 1 step on (x_0) and (transformed x_0), compute ε, persist ε(t) npzs to Modal Volume.
- Consumer-side: `lint_eps_dir(eps_npz_dir, ...) -> list[HarnessResult]` reads ε(t) npzs from disk and emits SARIF rows. Parallel to 4a's `lint_npz_dir`. Lives in `_harness/`, knows nothing about case studies.
- Case-study driver: `01-lagrangebench/emit_sarif_eps.py` assembles run-level properties and invokes `lint_eps_dir`. Parallel to 4a's `emit_sarif.py`.

**Rotated state intentionally non-persisted.** $R x_0$ and $f^1(R x_0)$ live only in Modal-job memory between forward-pass and ε-computation — never written to disk. Audit trail (someone curious about "what does $R x_0$ actually look like for this trajectory") is recovered via re-running a single (model, rule, transform_param, traj_index), which takes sub-minute on A10G. This trades persistent auditability for a clean artifact tier, on the judgment that the recovery path is cheap enough to make persistence redundant.

**Rejected alternative (I) — full rotated-state trajectories.** Would persist 4 angles × 20 trajs × 2 stacks × T=100 step rollouts ≈ 1.6 GB of rotated-state npzs to Modal Volume. Rejected for over-persistence-for-unclear-ROI: the auditability gain is recoverable via sub-minute re-run, and the only artifact-tier consumer (the consumer module) only needs the ε scalar (SARIF) and ε(t) array (figure), not full state. (Note: this rejection is *not* on commit-bulk grounds — under 4a's pattern, rotated-state would live on Modal Volume too, not in the repo.)

**Trigger-vs-emission separation in PH-SYM-003 SKIP:**
- *Trigger* (rule-specific, lives in `symmetry_rollout_adapter.py`): "substrate has periodic boundary conditions AND rotation angle θ ∉ {0, π/2, π, 3π/2}." The trigger is implemented as a check inside the rule's adapter, not as a heuristic in shared code.
- *Emission* (shared, lives in `lint_eps_dir.py` + `sarif_emitter.py`): when the trigger fires, the rule returns a `SkipDefect(value=None, skip_reason="...")` value. Downstream, the same D0-19 §3.4 emission machinery used by PH-CON-002's dissipative SKIP populates `extra_properties.skip_reason` and emits the row.
- This separation gives future rules adding SKIP paths a clean pattern to copy: trigger logic in the rule's own module; emission shape from the shared D0-19 path.

### 3.7 Compute budget + GPU class

**GPU class: A10G.** Locked for D0-17 amendment 1 principle generalized from within-rollout sha consistency to GPU-class consistency.

A100 is overkill for ~880 single-step inferences on TGV2D's ~3000-particle scale — the bottleneck isn't tensor throughput, and the 3–4× cost premium buys no observable speedup. T4 (older Turing architecture) introduces library/CUDA-version drift relative to 4a's Ampere reference rollouts, and at the float32-floor measurement scale, GPU-class FP behavior differences (TF32 on Ampere, FMA scheduling, kernel-fusion variations) sit at the same order as the threshold (~10⁻⁶–10⁻⁷). Matched A10G keeps measurement noise below the floor-vs-architectural-error distinction. Cost difference is cents per run at this duration; methodology consistency dominates.

**Inference count breakdown:**

| Stage | Count |
|---|---|
| Reference forward $f^1(x_0)$ (cached across rules) | 40 |
| PH-SYM-001 active angles × 4 (incl. θ=0) × 20 trajs × 2 stacks × 1 step | 160 |
| PH-SYM-002 reflection × 1 axis × 20 trajs × 2 stacks × 1 step | 40 |
| PH-SYM-004 translation × 1 vector × 20 trajs × 2 stacks × 1 step | 40 |
| PH-SYM-003 (SO(2) SKIP) | 0 |
| Figure-subset ε(t) (1 angle × 3 trajs × 2 stacks × T=100 steps) | 600 |
| **Total** | **~880 single-step inferences** |

Sub-minute on A10G end-to-end (model load + inference + I/O). Dev-time cycle: 1-angle × 3-traj smoke before full sweep (standard practice, not methodology-level pre-flight — under (I)+float32-floor framing, the full execution is itself sub-minute and strictly subsumes any smaller smoke-test as a methodology step).

---

## 4. Schema and provenance

### 4.1 SARIF schema_version bump v1.0 → v1.1

D0-19 v1.0 covers the rung 4a result-row shape with `mass_conservation_defect`, `energy_drift`, `dissipation_sign_violation` rule ids and per-row extra_properties `traj_index`, `npz_filename`, `skip_reason` (D0-19 §3.4), `ke_initial`, `ke_final`.

v1.1 adds:
- New rule ids: `PH-SYM-001`, `PH-SYM-002`, `PH-SYM-003`, `PH-SYM-004`.
- New per-row extra_properties: `eps_pos_rms` (always present for active rules; `None` for SKIP rows where it is replaced by `skip_reason`); `transform_kind` ("rotation" | "reflection" | "translation" | "identity"); `transform_param` (rotation angle in radians as float, reflection axis as string, or translation vector as 2-tuple of floats).
- Existing v1.0 fields unchanged. v1.0 SARIFs remain readable by the v1.1 renderer's schema-version-asserting code path (v1.1 renderer is written; v1.0 renderer is not modified — see §5.1).

The schema_version field on every SARIF row enables the renderer's fail-loud assertion. v1.0 rows with v1.1 schema assertion fail; v1.1 rows with v1.0 schema assertion fail. No silent schema drift.

### 4.2 4-stage provenance for ε(t) npzs

D0-19 v1.0 records 3-stage provenance on rung 4a SARIF rows: `physics_lint_sha_pkl_inference` (LagrangeBench inference run), `physics_lint_sha_npz_conversion` (pkl → npz), `physics_lint_sha_sarif_emission` (consumer → SARIF). Rung 4b's ε(t) artifacts add a fourth stage:

- `physics_lint_sha_pkl_inference` — carried unchanged from rung 4a's reference rollouts (same pkls as 4a).
- `physics_lint_sha_npz_conversion` — carried unchanged.
- `physics_lint_sha_eps_computation` — **new**, recorded at ε(t)-computation time (Modal-side generator). Names the commit at which ε was computed from the reference rollouts.
- `physics_lint_sha_sarif_emission` — recorded at SARIF-emission time (consumer-side, can be later than `eps_computation` if the case-study driver re-runs without re-generating).

The four shas may be identical or distinct; equality is allowed but never required, mirroring D0-19's policy.

ε(t) npz schema is spec'd in `_harness/SCHEMA.md` §3.x (new section), parallel to the rollout-npz schema's existing structure:

```python
{
    "eps_t":           np.ndarray,  # (T,) float32 — ε(t) at each step
    "rule_id":         str,         # "PH-SYM-001" etc.
    "transform_kind":  str,         # "rotation" | "reflection" | "translation"
    "transform_param": object,      # angle (float) | axis_name (str) | vector (tuple)
    "traj_index":      int,         # parsed from filename, validated contiguous (D0-19 §)
    "model_name":      str,
    "dataset_name":    str,
    "ckpt_hash":       str,
    "physics_lint_sha_pkl_inference":  str,
    "physics_lint_sha_npz_conversion": str,
    "physics_lint_sha_eps_computation": str,
}
```

`SCHEMA.md` §3.y documents the schema_version v1.1 SARIF result row structure (extends §3.x for v1.0 already present in the file).

---

## 5. Renderer

### 5.1 Sibling renderer choice (vs extending 4a's `render_cross_stack_table`)

**Decision: sibling renderer.** Rung 4b ships `methodology/tools/render_eps_table.py`, separate from 4a's `methodology/tools/render_cross_stack_table.py`. Each renderer is focused on one schema version (v1.0 and v1.1 respectively).

**Rationale:**

- 4a's renderer is tested, committed, and has passing fixtures. Modifying it to handle v1.1 risks regressions on v1.0 behavior.
- 4b's table layout differs structurally from 4a's: tripartite evidence grouping (architectural / construction-trivial / substrate-incompatible-SKIP) instead of 4a's flat per-rule list. The grouping logic is non-trivial and isn't a small extension to 4a's flow.
- A multi-version-aware single renderer would branch on schema_version at most call sites, increasing per-line complexity and the risk that a change to v1.1 logic accidentally changes v1.0 output.
- Code duplication is bounded: the column formatting and header-emission primitives are simple enough that DRY-ifying them later, if a third schema version arrives, is cheap. Premature DRY-ification of two structurally different table layouts adds complexity for no clear win at n=2.

**Forward-flag:** when rung 4c (or analog) introduces a third schema version, the right move is to extract shared formatting primitives into `methodology/tools/render_lib.py` and have all version-specific renderers compose them. Not pre-emptive at n=2.

### 5.2 Tripartite evidence grouping in the table

The rendered table groups rows by evidence type rather than by rule id:

```
Architectural-evidence rows
| Rule       | Stack | θ/axis/t            | ε              | Verdict      |
| PH-SYM-001 | SEGNN | π/2                 | 4.3e-7         | PASS         |
| PH-SYM-001 | SEGNN | π                   | 4.1e-7         | PASS         |
| PH-SYM-001 | SEGNN | 3π/2                | 4.5e-7         | PASS         |
| PH-SYM-001 | GNS   | π/2                 | 1.2e-2         | APPROXIMATE  |
| ... (the load-bearing rows for the headline)

Construction-trivial rows (smoke tests)
| PH-SYM-001 | SEGNN | 0 (identity)        | 0.0e+0         | PASS (trivial)
| PH-SYM-004 | SEGNN | (L/3, L/7)          | <1e-15         | PASS (trivial)
| ...

Substrate-incompatible SKIP
| PH-SYM-003 | SEGNN | continuous SO(2)    | —              | SKIP: PBC-square breaks SO(2) ...
| PH-SYM-003 | GNS   | continuous SO(2)    | —              | SKIP: PBC-square breaks SO(2) ...
```

The grouping is rendered as three labeled subsections in the markdown output, with section headers identifying the evidence type. The reader sees the load-bearing architectural rows first, then the smoke-tests, then the structural-SKIP — matching the writeup body's "what each PH-SYM rule measures on this substrate" subsection structure.

The renderer asserts schema_version `v1.1` on every SARIF read; mismatch raises `SchemaVersionMismatchError` (loud, fail-fast), parallel to 4a's renderer's v1.0 assertion. v1.0 SARIFs cannot be rendered by the v1.1 renderer; v1.1 SARIFs cannot be rendered by the v1.0 renderer. The two renderers each guard their own version contract.

---

## 6. Test fixtures

Per the "test fixtures hand-crafted, not copied from production" discipline, each PH-SYM rule's mechanic is pinned by synthetic-but-realistic fixtures with paired expected outputs. Fixtures live in `_harness/tests/fixtures/`.

**`c4_symmetric_4particle.py` (positive path, PH-SYM-001):** four particles arranged at corners of a unit square centered at (L/2, L/2). C₄ rotation by π/2 maps (corner_0 → corner_1, corner_1 → corner_2, ...). Expected ε for an exactly-equivariant operator: machine-zero (≤ 10⁻¹⁵). Test asserts ε is machine-zero for the identity operator and stays at the float32 floor for a numerical equivariance-preserving operation; a stub model that returns input-unchanged gives identity-output and is exactly equivariant, so the fixture's expected ε on stub is at the float32 floor.

**`c4_breaking_4particle.py` (negative path, PH-SYM-001):** four particles arranged at non-C₄-equivariant positions — three at C₃-equivariant locations on a circle, one at a position that breaks both C₄ and C₃. C₄ rotation by π/2 produces a different state. A "model" that adds a C₄-breaking constant gives ε well above the FAIL threshold. Test asserts the rotation mechanic catches the break (ε > 10⁻²), failing loud rather than silently producing an apparently-equivariant value.

**`reflection_symmetric.py` (positive path, PH-SYM-002):** four particles arranged y-axis-reflection-symmetric. Reflection across y-axis through box center gives the same state. Stub model gives identity-output → ε at float32 floor. Test asserts ε at floor.

**`pbc_square_so2_skip.py` (SKIP-mechanism fixture, PH-SYM-003):** any rollout npz; the rule's SO(2) trigger fires regardless of particle configuration because the substrate property is what triggers the SKIP. Test asserts:
1. `raw_value is None` (no ε computed)
2. `extra_properties.skip_reason` is populated and non-empty
3. `skip_reason` is identical across all rows within (rule, stack), per D0-19 §3.4
4. SARIF row shape matches PH-CON-002's dissipative-SKIP shape — same emission machinery, different trigger source (the fixture asserts the *contract* matches across SKIP causes)

**Renderer fixture:** golden-fixture test asserting v1.1 SARIF input → expected tripartite-grouped markdown output. Renderer's schema_version assertion catches v1.0-input cases with a clear error message.

**Negative-path renderer fixture:** v1.0 SARIF passed to v1.1 renderer → `SchemaVersionMismatchError` raised; v1.1 SARIF passed to v1.0 renderer → same error class. Both fail-loud.

---

## 7. Forward flags + honest limits captured in D0-21

1. **Float32→float64 threshold-rationale revisit.** The float32 numerical-precision floor argument is precision-dependent. If case study 02 (PhysicsNeMo MGN) runs at float64, the same threshold band would mean different things; D0-21 captures this as a precision-flag against future cross-precision validation.

2. **(rule, substrate) compatibility matrix generalization.** D0-18 detects PH-CON-002 SKIP via dataset-name + system_class heuristics. PH-SYM-003 / PBC-square SKIP triggers on a different (rule, substrate) compatibility axis. The general pattern — "rule's classification semantics depend on substrate properties" — generalizes beyond either. A future rung could formalize a per-rule compatibility matrix in the rule schema rather than per-rule heuristic triggers; D0-21 names the pattern so it doesn't get rediscovered cold each time.

3. **Meta-correction discipline self-validation.** Recorded as a footer note on D0-21: the SO(2)/PBC-square SKIP path was missed at the brainstorm's non-claim-5 stage and surfaced via the design pass (not mid-execution). This is the friction "writeup-framing-baked-into-design-before-code" exists to catch. Recording the correction makes the discipline self-validating rather than self-congratulatory aspiration.

4. **(I)-rejection framing precision.** The full-rotated-state pipeline (Option I) was rejected for over-persistence-for-unclear-ROI under Modal-Volume cost, *not* commit-bulk. D-entry framing in §3.6 captures this so future readers don't think (I) was rejected on commit-bulk grounds (which it wasn't — under 4a's pattern, full state would live on Modal Volume too).

5. **Translation-vector "non-grid-commensurate" wording precision.** L/3 and L/7 are *commensurate* with L (rational fractions); the structural argument (translation + PBC commute exactly) doesn't depend on Diophantine incommensurability. D-entry uses "non-grid-commensurate" to capture the practical anti-accidental-alignment property, *not* "incommensurate with L" which is mathematically wrong.

6. **Sibling-vs-extend renderer forward-flag.** When a third schema version (v1.2 etc.) arrives, extract shared formatting primitives into `methodology/tools/render_lib.py` and have all version-specific renderers compose them. Not pre-emptive at n=2.

---

## 8. Acceptance criteria

Rung 4b is considered passed when all of the following hold:

- [ ] D0-21 committed before any code change.
- [ ] `_harness/symmetry_rollout_adapter.py`, `_harness/lint_eps_dir.py`, `01-lagrangebench/emit_sarif_eps.py`, `methodology/tools/render_eps_table.py` exist with passing tests.
- [ ] `_harness/SCHEMA.md` documents schema_version v1.1 (§3.y) and ε(t) npz schema (§3.x).
- [ ] `01-lagrangebench/outputs/sarif/segnn_tgv2d_eps_<sha>.sarif` and `gns_tgv2d_eps_<sha>.sarif` committed; both schema_version v1.1.
- [ ] Both SARIFs structurally identical row-by-row in (rule_id, transform_kind, transform_param, traj_index): same set of rows present, same skip_reason populated for PH-SYM-003 SKIP path, same emission shape.
- [ ] PH-SYM-001 angle-0 row + PH-SYM-004 row in both SARIFs at machine-zero (smoke-test contract).
- [ ] PH-SYM-003 SKIP row in both SARIFs with identical `skip_reason` string; reuses D0-19 §3.4 emission machinery.
- [ ] ε(t) figure (`outputs/figures/eps_t_segnn_vs_gns_pi_2.png`) generated from 6-trace Modal-Volume-mirrored data; gitignored.
- [ ] `methodology/docs/2026-05-05-rung-4b-equivariance-table.md` writeup committed with frozen headline (§1.2), tripartite-grouped table (§5.2), Helwig body-context paragraph, honest-limits subsection (matches §1.3 deferral list verbatim).
- [ ] Cross-stack equivariance result reported: per-stack ε values, classified per the float32-floor threshold band; secondary-finding sentence on Helwig consistency wording finalized post-execution.

The headline (§1.2) is true under either probe outcome; the secondary-finding sentence on Helwig is the only writeup wording determined by empirical result.

---

## 9. Predecessor → successor → next deliverable

- **Predecessor:** rung 4a (`2026-05-04-rung-4a-cross-stack-conservation-table.md`) — cross-stack conservation result; D0-18, D0-19, D0-20 on master.
- **This document:** rung 4b design — pre-registers D0-21.
- **Next:** `2026-05-05-rung-4b-equivariance-plan.md` — implementation plan with TDD-discipline task breakdown, derived from §§3, 5, 6 above. Generated via the writing-plans skill.
- **Then:** rung 4b execution.
- **Then:** `2026-05-05-rung-4b-equivariance-table.md` — post-execution writeup with frozen headline, tripartite-grouped table, Helwig body-context, honest-limits.
