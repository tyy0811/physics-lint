# Case Study 02 — NVIDIA PhysicsNeMo MeshGraphNet (design)

**Date:** 2026-05-11
**Repo:** physics-lint
**Branch (intended):** new branch `feature/case-study-02-physicsnemo-mgn` off `master` after PR #9 (rung-4c) merges; this design doc + the writing-plans implementation plan land on `feature/rung-4c-substrate-class-extension` as planning-only artifacts (no code paths depending on rung-4c).
**Status:** Design — pre-implementation. Supersedes plan §4 of `2026-05-01-rollout-anchor-extension-plan.md`. Pre-registers Phase 1's audit + code-absorption work; the implementation plan is the successor.
**Predecessor:** rung 4c — substrate-class extension (`2026-05-07-rung-4c-substrate-class-extension-design.md`); v2.1 methodology evolution (`physics-lint-validation-plan-v2.1.md`); CS02 MGN loader-contract preflight (`../../02-physicsnemo-mgn/preflight/mgn_loader_contract.md` at PhysicsNeMo sha `1ca85d65`).
**Successor:** case-study-02 implementation plan (writing-plans output, dated for Phase 1 fire); then Phase 1 execution; then Phase 2; then Phase 3 writeup.

---

## 1. Scope and framing

### 1.1 Headline

> *Case study 02 P0 — `modulus_ns_meshgraphnet` (vortex shedding 2D) — is the first independent test of v2.1's A+B+C methodology triad's cross-rung generalization claim. The design pre-registers what would count as confirmation or disconfirmation per v2.1 §1.5's "readiness, not validation" framing; the execution either holds the triad up or breaks it.*

This headline is frozen at design time. The design itself is *applying* the triad to plan how a future application of the triad will be evaluated — pattern C's verbatim-preservation discipline applied to the framework's own next use.

### 1.2 What this design supersedes

Supersedes `methodology/docs/2026-05-01-rollout-anchor-extension-plan.md` §4 (10 days old, predates v2.1 methodology evolution). v2.1 is the methodology truth-source; this doc is its first application to a mesh-side case study. Plan §4 stays as historical context with the supersession callout at its head.

### 1.3 Targets and rules in scope

**P0 only — `modulus_ns_meshgraphnet`** (vortex shedding 2D, incompressible NS, cylinder wake). PhysicsNeMo `v2.0.0` pinned at sha `1ca85d65` per the preflight doc.

| Rule | Status | Routing |
|---|---|---|
| **PH-CON-001** (mass / divergence-free) | **Definitive** | Mesh harness `mass_conservation_defect_on_mesh` (D0-03 route — public CLI returns SKIP on `pde != "heat"`) |
| **PH-CON-002** (`energy_drift`) | **Provisional pending D0-2X** (Phase 1 verdict: SKIP with reason if open-driven-dissipative confirmed; fire normally otherwise) | Mesh harness `energy_drift_on_mesh` |
| **PH-CON-003** (`dissipation_sign_violation`) | **Provisional pending D0-2X** (Phase 1 verdict: SKIP with reason if open-driven-dissipative confirmed; fire normally otherwise) | Mesh harness `dissipation_sign_violation_on_mesh` |

**Provisional rules — substrate-class rationale.** D0-22's `open-driven-dissipative` SKIP gate (`dissipation_sign_violation`) and D0-22 amendment 1's parallel gate (`energy_drift`) fired only on the particle side. The preflight doc's substrate-class analysis names the natural default for cylinder wake as `open-driven-dissipative` (textbook example: cylinder wake) with three discriminating observables proposed for empirical verification:

1. **∫|∇·v|dV** (mass conservation; the PH-CON-001 signal).
2. **KE budget** — dKE/dt vs inflow-outflow-dissipation balance.
3. **Strouhal number** — St ∈ [0.16, 0.21] for Re ∈ [100, 300] (cylinder-wake-specific signature).

Phase 1 empirically verifies the classification; D0-2X (new entry) introduces the mesh-side substrate-class dispatch + cylinder-wake class label. The preflight doc lists the candidate class explicitly per the rung-4c "classify when you exercise" rule.

**D0-22 forward-declaration falsified (forward-flag).** D0-22 §1 named "strictly-conservative" as the forward-compatible class "for case study 02 anchor." Case study 02 P0 (cylinder wake) is open-driven, not strictly conservative; plan §4's broader scope (vortex shedding + Ahmed Body) never had a strictly-conservative target either. Forward-flag: D0-22 gets a corrective amendment when the strictly-conservative anchor is concretely identified (case study 03 candidate; deferred until then).

**Open-driven sub-class distinction.** "Open-driven" covers two distinct driving mechanisms — *gravity-driven* (dam-break: PE → KE conversion, monotone rise-then-fall) and *boundary-driven* (cylinder wake: inflow / outflow steady-state, dE/dt oscillates around zero rather than monotonically trending). Both are non-strictly-conservative; both are sub-classes of the broader class. Phase 1 empirically verifies whether D0-22's SKIP gate discriminates between them (sub-class-specific → mesh-side needs a new class label) or covers the broader class (sub-class-agnostic → extension is automatic).

**Substrate-class taxonomy gap.** Particle-side has `LAGRANGEBENCH_DATASET_SYSTEM_CLASS`; mesh-side has no equivalent. Phase 1 introduces `MGN_DATASET_SYSTEM_CLASS` as a parallel dispatch table (NOT a stack-agnostic refactor — the duplicate-logic-drift risk is named per round-codex-4 catalogue, not eliminated; stack-agnostic refactor defers to amendment 1 or case study 03 evidence).

### 1.4 §4 supersession callout

The 2026-05-01 plan's §4 receives the following callout at its head:

> **Status:** Superseded by `methodology/docs/2026-05-11-case-study-02-physicsnemo-mgn-design.md`. §4 below is preserved as historical context; it does not reflect v2.1 methodology evolution (pattern A+B+C, smoke/source/cross triple, bilateral schema framing, layered-fail-open observation). Also corrects plan §4 §2.2's rule table, which listed PH-SYM-* as in-scope for vortex shedding contra spec §1.2's particle-side restriction. Read the new design doc for the current case study 02 plan.

### 1.5 What case study 02 is NOT (explicit deferral list)

1. **Not Ahmed Body / PH-BC-001 / steady RANS.** Different rule, different physics, prime pattern-B candidate (wall-node identification per preflight A1-A18). Deferred to amendment 1 — triggered by P0 vortex shedding completing §4.3. Bundling muddles which pattern surfaces which finding.
2. **Not PH-RES-001 (BDO).** Deferred to amendment 2 or case study 03 — BDO is a strong constraint deserving dedicated session.
3. **Not PH-NUM-002 (resolution sweep).** Deferred to v1.1 backlog per spec §1.2 — separate deliverable, not v1.0 scope.
4. **Not PH-SYM-001/002/003/004 on mesh side.** Spec §1.2 in-scope split keeps PH-SYM-* particle-side only. Restated since plan §4 §2.2 drifted from this.
5. **Not the cover-letter cross-case-study paragraph.** Plan §5.3 integrates across BOTH case studies (LB + MGN); post-Phase-3 work after both writeups exist.
6. **Not the stack-agnostic refactor.** `MGN_DATASET_SYSTEM_CLASS` introduced as duplicated route (NOT renamed `DATASET_SYSTEM_CLASS` keyed by stack). Stack-agnostic refactor defers to amendment 1 / case study 03 evidence per rung-4c "don't generalize prematurely" discipline.

---

## 2. Methodology applied to case study 02

This section specializes v2.1's A+B+C triad and smoke/source/cross triple to case study 02 P0. The patterns and triple are the truth-source; this section is their application.

### 2.1 Pattern A — smoke / empirical-vs-prediction

Predicted MGN surfaces where a pre-registered prediction may fail at smoke-time:

- **NGC sample reproduction tolerance.** `test_inference_matches_ngc_sample` predicted to pass within 10⁻³ max-abs-error on velocity (plan §4 step 2 default; Phase 1 audit reads NGC's documented tolerance — if NGC documents a tighter or looser band, the pre-registration narrows accordingly via D-entry amendment before P0 fires). Smoke-fail → D-entry amendment refining the tolerance OR Gate D FAIL → FNO-on-Darcy fallback.
- **Mass conservation drift on cylinder wake.** PH-CON-001's `mass_conservation_defect_on_mesh` predicted bounded by NS incompressibility numerical floor. The concrete bound is a **calibration-based pre-registration** (Phase 1 observes the floor on the NGC-shipped sample, D-entry pins it before Phase 2). Calibration is a *kind* of pre-registration, not a priori. (Optional sharpening: bound from MGN's published training MSE on velocity → expected divergence floor a priori; revisit if MGN's training metrics surface this in Phase 1.)
- **Empirical substrate-class verification.** Phase 1 observes `energy_drift_on_mesh` raw-value behavior on the cylinder-wake rollout. Prediction (from §1.3 physical analysis): boundary-driven sub-class — dE/dt oscillates around zero, not monotone in either direction. If observation matches → classification confirmed, dispatch lands per §2.2. If observation diverges (e.g., dE/dt monotone in some MGN regime), Pattern A drift fires → D-entry amendment captures the surprise. *Classification origin and dispatch response are §2.2's work; this bullet is the empirical-vs-prediction part only.*

**Enabling discipline.** Pre-flight assertions in `mesh_rollout_adapter.py` parallel `particle_rollout_adapter.py`'s "loader-contract assertions" section. The assertions implement pattern A's smoke-time discipline by source method — catches divergence before P0 fires. The preflight doc's V1-V18 (VortexSheddingDataset assertions, file:line at PhysicsNeMo `1ca85d65`) is the source-of-truth for what to assert.

### 2.2 Pattern B — source / loader-contract

Predicted single-artifact-multi-use-case surfaces, with P0-resolvable vs amendment-1-deferred marking:

- **[AMENDMENT-1-DEFERRED] DGL → MeshField materialization.** Highest-confidence pattern-B candidate. One artifact (`mesh_rollout_adapter`'s materialization function), three use-cases (PH-CON-001/002/003 rule kernels) plus Gate A's PASS/PARTIAL/FAIL branches. Hidden assumption: NGC checkpoint emits velocity in a specific topology + dtype + node-ordering. In P0 (single checkpoint), the multi-instance dimension doesn't fire; amendment 1 (Ahmed Body) is the natural trigger.
- **[P0-RESOLVABLE] `_expect_velocity(rollout)` helper.** Single-instance hidden assumption: `node_values["velocity"]` key exists. NGC may emit `u`/`v`/`vel`/`flow_field`/etc. Phase 1 audit detects the actual key name via the preflight's V12 / V14 references; Phase 2 hardcodes the resolution as a D-entry. **No pre-generalization** per §2.2 / rung-4c discipline — predicate-generalization fires only if amendment 1 brings a second naming convention.
- **[P0-RESOLVABLE] Substrate-class dispatch absence on mesh-side + classification + response.** Owns the substrate-class work per §2.1 narrowing. Physical analysis (a priori): cylinder wake is open-driven non-strictly-conservative, boundary-driven sub-class candidate. Mesh-side has no `*_DATASET_SYSTEM_CLASS` dispatch — the rules' strictly-dissipative-or-conservative assumption fails on cylinder wake with nothing to catch it. **Response (P0 scope):** introduce `MGN_DATASET_SYSTEM_CLASS` (parallel to particle-side, NOT stack-agnostic refactor). The duplicate-logic-drift risk is *named* and *registered* per round-codex-4 catalogue, not *eliminated*. Stack-agnostic refactor *triggers* only on amendment 1 / case study 03 evidence.
- **[AMENDMENT-1-DEFERRED] Wall-node identification (Ahmed Body / PH-BC-001).** Prime pattern-B candidate for amendment 1. Loader-contract on DGL wall-tag convention; preflight A2/A4 already identifies the BLOCKING-2 risk that Ahmed-body raw data may be NGC-gated.

§2.3 cross-review checkpoints dispatch only on the P0-resolvable items; amendment-1-deferred items wait for their own cross-review at amendment-1 execution time.

### 2.3 Pattern C — cross / triage-vs-novel

Pre-registered cross-review checkpoints at four phase boundaries:

- **Pre-execution (Phase 0) — this design's brainstorming iteration.** Round-by-round cross-review on the design-doc prose itself. **This iteration fires the strict 4th instance of pattern-C self-application:** review-gate against pattern C's own articulation (this design doc references and applies pattern C), with verbatim preservation already happening — the D0-22 substrate-class extension push-back in §1 review was absorbed nearly verbatim into §1.3's "substrate-class taxonomy gap" paragraph. Documented here per v2.1 §1.3 falsification rule 4 (cell-4 must be earned by articulation, not pre-declared): the articulation is "review-gate fires on framework's own articulation while applying it; preservation discipline holds under the same iteration." Seeds v2.1.1 amendment per §5.5. Predicted prior on remaining Phase-0 findings: cell-2 (novel-in-scope refinements) and cell-1 (re-discovery against prior D-entries — D0-22's strictly-conservative forward-declaration falsification was cell-1, surfaced in §1).
- **End of Phase 1.** Codex review against Gate A/D verdicts + audit findings + code-absorption (per §3.1's expanded cross-review scope). Predicted prior: cell-2 (novel-in-scope MGN findings) + cell-3 (Ahmed-Body-shaped → amendment 1).
- **End of Phase 2.** Codex review against P0 SARIF + rule outputs. Predicted prior: cell-2 (loader-contract / fail-open findings) + cell-1 (re-discovery under rung-4c lens — e.g., mesh-side analog of round-codex-4's retry isolation if MGN inference writes to a persistent volume).
- **End of Phase 3.** Codex review against the writeup prose. Parallel to round-prose-1 / round-prose-2. Predicted prior: cell-2 + cell-4 if MGN-side methodology gaps surface (cell-4 must be earned by articulation per §1.3 falsification rule 4).

### 2.4 Smoke / source / cross triple — applied (looser correspondence)

Smoke / source / cross are **review-method labels, not phase identifiers**. Each phase hosts multiple review methods; the labels apply to the *method* of finding, the patterns apply to the *kind* of finding.

- **Smoke** — empirical-vs-prediction observation against pre-registered thresholds. Fires throughout case study 02: heaviest in Phase 1 (Gate A/D + NGC sample + substrate-class empirical verification), continues in Phase 2 (rule outputs vs pre-registered tolerances), surfaces in Phase 3 (writeup numerical claims). Primary enabling discipline for pattern A.
- **Source** — pre-flight assertions in `mesh_rollout_adapter.py` (source-method-implementing-pattern-A-discipline; written via source method, catches pattern-A divergence at runtime); source review of substrate-class dispatch gap (source-method-implementing-pattern-B, canonical); source review of writeup prose at Phase 3. Implements both A and B depending on application.
- **Cross** — pre-registered Codex passes at phase boundaries 0/1/2/3 per §2.3. Primary enabling discipline for pattern C.

Diagonal correspondence to A/B/C is **observed but not load-bearing** per v2.1 §1.4. Phase-method-pattern mapping is many-to-many. **Operational classification test** for ambiguous overlaps: source surfaces the code assumption (B); empirical run surfaces the divergence (A); cell-1/cell-2 triage handles cases that span both. Each phase ends with a cross-review at its boundary (Phase 0 included); cross is not phase-bound.

### 2.5 Bilateral schema choice

Case study 02 inherits rung-4c's SARIF schema (v1.0 optional `inference_run_status` field). **Does NOT trigger v2.1 §2.1's schema bump 1.0 → 1.0.1 in the predicted path:** NGC checkpoints are static pre-trained models. `inference_run_status` lands as `from_completed_inference` uniformly (or omits via the renderer's `n/a (pre-salvage-tag-schema)` honest-absence convention).

Optional-field state remains canonical; renderer's strict-version assertion holds. Mesh-side SARIFs follow particle-side schema; cross-stack rendering composes uniformly.

**Forward-flag (salvage triggers named, not assumed away):** if MGN inference surfaces a salvage scenario — either (a) a checkpoint requiring partial-retry workaround documented at execution time, or (b) **Modal-side inference timeout requiring partial-rollout SARIF surfacing** — that becomes the §2.1 schema-bump trigger. Long rollout horizons, large rollout counts, or Modal infrastructure transients can fire (b) without (a) ever firing; the two triggers are independent.

### 2.6 Layered fail-open prediction

Round-codex-4 elevated the layered-fail-open observation from "worth flagging" to a named methodology contribution. **Anchoring data:** rung-4c series surfaced HIGH findings on **4 of 4 code-targeting cross-review rounds** (round 3, codex-2, codex-3, codex-4); the predicted layered-fail-open shape held uniformly.

**Anchoring the downward revision:** `mesh_rollout_adapter.py` was built *after* round-codex-4 surfaced its rung-4c findings, so the canonical fail-open shapes (retry isolation, rollout-dir guards, pre-flight assertion patterns) are already absorbed into mesh-side at design time. The preflight doc's 36 enumerated assertions further narrow predicted surfaces. Predicted 2-4 rather than 4-of-4 because residual fail-open surfaces are predicted on MGN-specific code (resampling helper, MGN materialization, cross-precision band) rather than on the rung-4c-canonical surfaces already hardened.

Forward-flag predicted MGN safety surfaces (Phase 2/3 cross-review):

- **Mesh-side analog of rollout-dir retry isolation** (round-codex-4 finding 1). If MGN inference writes to a persistent Modal volume, same-sha-retry contamination is the parallel attack surface. Phase 1 activity 9 commits the verdict.
- **DGL→MeshField materialization under PARTIAL fallback.** Regular-grid resampling path; potential fail-open if resampling silently produces NaN or out-of-bounds nodes without surfacing them. Pre-flight assertion in the resampling helper closes this if Gate A returns PARTIAL.
- **Cross-precision (float32 vs float64) NGC checkpoint compatibility** per rung-4b D0-21. Layered fail-open if MGN runs at a different precision than rung-4c's threshold bands assumed. **Revisit in Phase 1 audit** — concrete trigger (preflight already flagged fp32-default-dtype as a known-unknown).

Predicted: 2-4 cross-review rounds against case-study-02 artifacts will surface these or analogous fail-opens (anchored at rung-4c's 4-of-4). **If 0-1 rounds find HIGH findings, the methodology's prediction was too aggressive — that's a cell-2 finding** (novel-in-scope refinement of the layered-fail-open observation, narrowing its predicted scope or specifying preconditions). **Cell-4 would require a genuinely new framing** — e.g., the observation isn't "layered fail-opens" but a different mechanism deserving its own name. That's a stronger claim than under-prediction; cell-4 must be earned by articulation per §1.3 falsification rule 4.

---

## 3. Execution shape (three phases)

Three execution phases mapping to the user's three-session arc. Each phase ends with a cross-review at its boundary per §2.3. Phase-method-pattern mapping is many-to-many per §2.4.

### 3.1 Phase 1 — Audit (target session 2)

**Goal:** Verify case-study-02-blocking assumptions empirically before any P0 inference fire.

**Activities** (ordered; 1-9 are setup + audit + verification, 10-13 are code-absorption RESPONSIVE to audit findings):

1. **BLOCKING-1 unblock — CPU-only NGC ↔ v2.0.0 state-dict smoke** (cheapest unblock, zero GPU). The preflight doc identifies that `modulus_ns_meshgraphnet` was trained against pre-rename modulus; no upstream doc pins it to a specific physicsnemo commit. Run the state-dict-key smoke against the v2.0.0 MeshGraphNet constructor (`num_input_features=6`, `num_edge=3`, `num_output=3` per `conf/config.yaml`) before any Modal install. **Gate-out:** if state-dict keys don't match → Gate D demotion candidate; consider an older physicsnemo pin or FNO-on-Darcy fallback. **Cited preflight item:** "BLOCKING (P0): NGC checkpoint ↔ v2.0.0 source-compatibility unknown" + the proposed mesh-side analog of LB's `test_inference_matches_ngc_sample`.

2. Modal container install (`nvidia-physicsnemo @ 1ca85d65` + `dgl` + `ngc` CLI). Pinned to the preflight's sha.

3. NGC checkpoint download: `modulus_ns_meshgraphnet:v0.1`. Hash pinned in DECISIONS.md (new D-entry).

4. **Day 2 hour 1 NGC audit (D0-11) — uses preflight V1-V18 as source.** Inspect the NGC-shipped sample timestep against the preflight's pre-enumerated assertions:
   - V1-V18 cover VortexSheddingDataset's loader-contract assumptions at file:line.
   - Resolve: (a) actual velocity-field key in `node_values`; (b) DGL graph topology coercibility (Gate A input); (c) primitive-vs-derived emission. Findings → D0-11 amendment.
   - Watch for the preflight's 5 secondary known-unknowns: CWD coupling (lines 103/141), `noise_std=0.02` split-conditional (line 127), fp32-vs-fp64 default-dtype contract, `meta["trajectory_length"]` vs `num_steps` silent overrun, `node_type ∈ {0, 3, 4, 5, 6}` with `(value - 3)` shift + `num_classes=4` bound.

5. **Gate A verdict** (D0-02 amendment): PASS (DGL → MeshField materialization works) / PARTIAL (GridField resampling fallback) / FAIL (mesh harness SKIPs).

6. **`test_inference_matches_ngc_sample`:** run NGC inference on the shipped sample, compare against shipped expected output within Phase-1-pinned tolerance (plan §4 default 10⁻³; refined per NGC documentation read during audit).

7. **Gate D verdict:** PASS (checkpoint usable) / FAIL (FNO-on-Darcy fallback). The BLOCKING-1 outcome from activity 1 is a strong prior on Gate D.

8. **Empirical substrate-class verification.** Physical analysis already classifies cylinder wake as boundary-driven non-strictly-conservative (a priori). Phase 1 EMPIRICALLY VERIFIES via 1-traj smoke rollout, using the preflight's **three discriminating observables**:
   - ∫|∇·v|dV (mass conservation; PH-CON-001 signal — should be near zero).
   - KE budget dKE/dt vs inflow-outflow-dissipation (should oscillate around zero, NOT monotone-rising/falling).
   - Strouhal St ∈ [0.16, 0.21] for Re ∈ [100, 300] (cylinder-wake-specific signature).

   Verification confirmed → D0-2X amendment lands with `vortex_shedding_2d → "open-driven-dissipative"` (the natural default per preflight). Unexpected pattern → pattern-A drift fires → D-entry amendment captures.

9. **Persistent-volume decision.** Decide whether Modal MGN inference writes to a persistent Modal volume. Default expected (standard Modal pattern); if confirmed, the rollout-dir isolation pattern (round-codex-4) commits to apply in Phase 2 activity 2. If MGN inference uses ephemeral container storage instead, isolation pattern is N/A.

   **Pattern-A drift on substrate-class smoke vs Gate D — disambiguation** (relevant here because the decision feeds both gates):
   - Pattern-A drift in NGC sample reproduction (activity 6) → already subsumed by Gate D FAIL (activity 7); no separate path.
   - Pattern-A drift in substrate-class smoke (activity 8) → D-entry amendment captures the surprise. **NOT** Gate D FAIL — substrate-class divergence is methodology-refinement, not checkpoint-usability failure.

10. **Pattern-B response on `_expect_velocity` helper:** D-entry pinning the actual key name detected in activity 4. No pre-generalization per §2.2 / rung-4c discipline.

11. **`MGN_DATASET_SYSTEM_CLASS` introduction + dispatch on `*_on_mesh` mirrors** per §2.2. Duplicated route (NOT stack-agnostic refactor); risk named per round-codex-4 catalogue.

12. **Pre-flight assertions in `mesh_rollout_adapter.py`** based on Phase 1 audit findings. Source-of-truth: preflight V1-V18 enumerated at file:line at `1ca85d65`. The materializer pre-flight section parallels `particle_rollout_adapter.py`'s "loader-contract assertions" section.

13. **D-entries committed:** new D-entries pinning the BLOCKING-1 verdict, Gate A verdict, Gate D verdict, substrate-class verdict, key-name resolution, and `MGN_DATASET_SYSTEM_CLASS` class label. Amendments to D0-02, D0-11 per audit findings.

**Review methods that fire in Phase 1:**

- **Smoke (primary):** activity 0 (CPU state-dict smoke), 5 (NGC sample reproduction), 7 (substrate-class smoke).
- **Source (secondary):** pre-flight assertions written (source method, pattern-A target); source review of dispatch gap during introduction (source method, pattern-B canonical).
- **Cross (at Phase 1 boundary):** Codex review against Gate A/D verdicts, audit findings, **AND code-absorption** (D-entries, `MGN_DATASET_SYSTEM_CLASS` dispatch, pre-flight assertions, helper key resolution). Single boundary review covers both audit + absorption.

**Gate-out triggers:**

- BLOCKING-1 state-dict mismatch → Gate D demotion candidate (FNO-on-Darcy or older physicsnemo pin).
- Gate D FAIL → FNO-on-Darcy fallback (rename `02-physicsnemo-mgn/` → `02-fno-darcy/`; Phase 2 proceeds with FNO).
- Gate A FAIL → mesh harness SKIPs with reason; cover-letter Appendix A.4 variant fires.

**Exit condition (see §4.1).**

### 3.2 Phase 2 — P0 inference + SARIF (target session 3)

**Goal:** Run vortex-shedding inference end-to-end on Modal; emit PH-CON-001/002/003 SARIFs through the mesh harness.

**Activities:**

1. **Modal entrypoint for MGN inference** (parallel to LB-side `lagrangebench_rollout_p1_*`). Entrypoint includes pre-flight assertions on:
   - Persistent-volume write path (per Phase 1 activity 9 decision).
   - NGC checkpoint hash verification (against DECISIONS.md pin from Phase 1 activity 2).
   - Rollout output schema (DGL graph topology + velocity-field key name match Phase 1 audit findings + preflight V1-V18).
   - CWD discipline (preflight known-unknown: VortexShedding lines 103/141 read stats from CWD via Hydra's chdir:True; entrypoint must reproduce or override).
   - Default dtype set to float32 before dataset construction (preflight known-unknown).
   - Split = "test" for inference (preflight known-unknown: `noise_std=0.02` split-conditional).

   Output to `/vol/rollouts/physicsnemo/vortex_shedding_<git_sha>/` if persistent-volume path is committed.

2. **Apply round-codex-4 rollout-dir isolation pattern** (commits per Phase 1 activity 9 decision). If Phase 1 committed persistent volume → isolation pattern applies (default expected); if Phase 1 committed ephemeral storage → activity drops.

3. **Per-timestep MeshField/GridField materialization** via mesh adapter (Gate A branch decided in Phase 1).

4. **Per-timestep PH-CON-001/002/003 all via harness `*_on_mesh` mirrors** (D0-03 routing committed for PH-CON-001 to bypass its `pde != "heat"` input-domain SKIP; PH-CON-002/003 inherit the routing per §1.3). Per D0-03 and spec §1.4's class-level entry on "V1 rules with documented input-domain restrictions" (added 2026-05-04), this routing means PH-CON-001 on mesh is documented as **"structural-identity reapplication" rather than "rule ran without modification"** — the rule check function is unchanged, but the invocation path is harness-routed. §3.3's "What physics-lint did NOT catch" section inherits this framing.

5. **SARIF emission:** cross-stack consistent schema; `inference_run_status` field per §2.5 (uniformly `from_completed_inference` predicted; salvage triggers fire forward-flag instead).

6. **Smoke check:** do rule outputs match pre-registered tolerances from Phase 1 D-entries? Pattern-A drift fires if not.

**Review methods that fire in Phase 2:**

- **Smoke (primary):** rule outputs vs Phase-1-pre-registered tolerances.
- **Source (secondary):** review of new MGN-side Modal entrypoint for fail-open shapes (rollout-dir guard, materialization helper, resampling NaN propagation per §2.6 forward-flag #2).
- **Cross (at Phase 2 boundary):** Codex review against P0 SARIF + rule outputs.

**Gate-out triggers:**

- Pattern-A drift in rule outputs → D-entry tolerance amendment OR documented as honest-limits.
- Cross-review HIGH finding → in-rung absorption (pattern-B if loader-contract; cell-2 if novel; cell-1 if rung-4c re-discovery → defer).

**Exit condition (see §4.2).**

### 3.3 Phase 3 — Writeup + cross-stack table (target session 3, may bleed to session 4)

**Goal:** Compose case-study-02 writeup and cross-stack consistency artifact; land v2.1.1 amendment.

**Activities:**

1. `02-physicsnemo-mgn/README.md`: populate Rule × checkpoint results table (vortex shedding row only; Ahmed Body row deferred to amendment 1).
2. Bridge-sentence paragraph (plan §4 step 5 template, with case-study-02 actuals).
3. Cross-stack consistency table populated (vortex shedding row only; Ahmed Body row marked "deferred to amendment 1").
4. "What physics-lint did NOT catch" section per spec §1.4. Bullets include:
   - The spec §1.4 class-level entry on V1-rules-with-input-domain-restrictions (PH-CON-001 routing per D0-03; "structural-identity reapplication" framing).
   - PH-NUM-002 multi-resolution → v1.1 backlog.
   - PH-SYM-* not on mesh side.
   - Ahmed Body and PH-RES-001 deferred to amendments.
5. Reproducibility section: Modal entrypoints, NGC checkpoint hash, git_sha, conda lockfile.
6. Methodology cross-references to `methodology/README.md` + v2.1 §1.5.
7. **v2.1.1 amendment** landing per §5.5: pattern-C 4th-instance documentation; per-section convergence-pattern observation; D0-22 strictly-conservative forward-declaration falsification flag; prose-cross-review vs artifact-cross-review modes flag (from round-prose-2 forward-flag at `5cb90cc`).

**Scope boundary:** The case-study-02 writeup at `02-physicsnemo-mgn/README.md` is internal to case study 02. The cover-letter paragraph (plan §5.3) integrates across BOTH case studies (LB + MGN) and is post-Phase-3 work, not part of this design's scope. Tracked separately so the cover-letter integration isn't forgotten in subsequent sessions.

**Review methods that fire in Phase 3:**

- **Smoke (minimal):** numerical-claim verification against Phase 2 outputs.
- **Source (primary on prose):** review for overclaiming, hidden assumptions, narrative drift.
- **Cross (at Phase 3 boundary):** Codex review against writeup prose. Parallel to round-prose-1 / round-prose-2.

**Exit condition (see §4.3).**

---

## 4. Per-phase acceptance criteria

Each phase's exit condition. Measurable thresholds where applicable.

### 4.1 Phase 1 acceptance

- [ ] BLOCKING-1 CPU state-dict smoke complete; verdict recorded in new D-entry.
- [ ] NGC checkpoint `modulus_ns_meshgraphnet:v0.1` downloaded; hash pinned in DECISIONS.md.
- [ ] Day 2 hour 1 NGC audit findings recorded: (a) velocity-field key name in `node_values`; (b) DGL topology coercibility; (c) primitive-vs-derived emission. Findings → D0-11 amendment. The 5 preflight secondary known-unknowns resolved or carried as known-limits.
- [ ] Gate A verdict recorded: PASS / PARTIAL / FAIL. Verdict → D0-02 amendment.
- [ ] Gate D verdict recorded: PASS / FAIL-with-FNO-fallback. Verdict → new D-entry.
- [ ] `test_inference_matches_ngc_sample` passes within Phase-1-pinned tolerance.
- [ ] Empirical substrate-class smoke either verifies the boundary-driven sub-class prediction (via 3 discriminating observables: ∫|∇·v|dV, KE budget, Strouhal St) OR captures pattern-A surprise via D-entry amendment.
- [ ] `MGN_DATASET_SYSTEM_CLASS` introduced; dispatch wired into `*_on_mesh` mirrors. New D-entry.
- [ ] `_expect_velocity` helper key resolution pinned via D-entry; no pre-generalization.
- [ ] Pre-flight assertions written in `mesh_rollout_adapter.py` based on audit findings + preflight V1-V18.
- [ ] Persistent-volume decision (activity 9) recorded.
- [ ] D-entries committed per activity 13.
- [ ] Phase 1 boundary cross-review (Codex pass over verdicts + code-absorption) complete; findings triaged.

### 4.2 Phase 2 acceptance

- [ ] Modal entrypoint for MGN inference committed; pre-flight assertions cover persistent-volume write path, NGC checkpoint hash verification, rollout output schema, CWD discipline, fp32 default-dtype, split="test".
- [ ] Round-codex-4 rollout-dir isolation applied IFF Phase 1 committed persistent volume.
- [ ] P0 vortex-shedding rollout completed end-to-end on Modal (n_trajs per Phase 1 cost estimate; rung-4c discipline — ship at empirically-feasible N, not plan-N).
- [ ] Per-timestep MeshField/GridField materialization works per Gate A branch.
- [ ] PH-CON-001 SARIF committed at `02-physicsnemo-mgn/outputs/sarif/`; values within Phase-1-pre-registered mass-conservation drift bound.
- [ ] PH-CON-002 (`energy_drift_on_mesh`) SARIF committed; behavior consistent with `MGN_DATASET_SYSTEM_CLASS` dispatch (SKIP-with-reason or RAW per dispatch).
- [ ] PH-CON-003 (`dissipation_sign_violation_on_mesh`) SARIF committed; SKIP/RAW outcome per dispatch.
- [ ] `inference_run_status` field lands per §2.5 (uniformly `from_completed_inference` predicted; salvage triggers fire forward-flag instead).
- [ ] Smoke check: rule outputs vs Phase-1-pre-registered tolerances; pattern-A drift absorbed via D-entry amendment if any.
- [ ] Phase 2 boundary cross-review (Codex pass over SARIF + rule outputs) complete; findings triaged.

### 4.3 Phase 3 acceptance

- [ ] `02-physicsnemo-mgn/README.md`: Rule × checkpoint results table populated (vortex shedding row only).
- [ ] Bridge-sentence paragraph drafted (plan §4 step 5 template, with case-study-02 actuals).
- [ ] Cross-stack consistency table populated (vortex shedding row; Ahmed Body row marked "deferred to amendment 1").
- [ ] "What physics-lint did NOT catch" section written per §3.3 activity 4.
- [ ] Reproducibility section: Modal entrypoints, NGC checkpoint hash, git_sha, conda lockfile.
- [ ] Methodology cross-references to `methodology/README.md` + `methodology/docs/physics-lint-validation-plan-v2.1.md` §1.5 (paths from `_rollout_anchors/`).
- [ ] v2.1.1 amendment landed (per §5.5): pattern-C 4th-instance documentation; per-section convergence-pattern observation; D0-22 strictly-conservative forward-declaration falsification flag; prose-cross-review vs artifact-cross-review modes flag.
- [ ] Phase 3 boundary cross-review (Codex pass over writeup prose) complete; findings triaged.

---

## 5. Deferrals (consolidated)

Per v2.1 §2 governance: each deferral named with promotion trigger; bare-naming insufficient.

### 5.1 Ahmed Body / PH-BC-001 / steady RANS → amendment 1

- **Trigger:** P0 vortex shedding completes §4.3.
- **Why deferred:** different rule, different physics, prime pattern-B candidate (wall-node identification per preflight A1-A18). Bundling muddles which pattern surfaces which finding.
- **Companion:** cross-stack consistency table's Ahmed-Body row folds into amendment 1.
- **BLOCKING-2 (from preflight):** Ahmed-body raw data is NGC-gated; the public NGC test subset may lack the `*_info.txt` files A2/A4 require. If confirmed at amendment 1 audit, P1 demotes per case-study-02 README's Gate-D fallback.

### 5.2 PH-RES-001 (BDO) on either substrate → amendment 2 OR case study 03

- **Trigger:** ≥3h buffer at amendment 1 completion (plan §4 condition), OR new case study territory.
- **Why deferred:** BDO is a strong constraint; deserves dedicated session.

### 5.3 PH-NUM-002 resolution sweep → v1.1 backlog

- **Trigger:** v1.1 multi-resolution harness deliverable per spec §1.2.
- **Why deferred:** separate deliverable, not v1.0 scope.

### 5.4 PH-SYM-001/002/003/004 on mesh side → never (out of scope, not deferred)

- Spec §1.2 in-scope split keeps PH-SYM-* particle-side only. Restated since plan §4 §2.2 drifted from this.

### 5.5 v2.1.1 amendment → Phase 3 deliverable (§3.3 activity 7)

- **Trigger:** Phase 3 cross-review of writeup prose completes.
- **Contents:**
  1. Pattern-C 4th-instance documentation per §2.3 (review-gate fires on framework's own articulation; preservation discipline holds; worked example: D0-22 substrate-class extension push-back absorbed nearly verbatim into §1.3).
  2. Per-section convergence-pattern observation (round-magnitude decreasing across §1 → §2.1-§2.3 → §2.4-§2.6 → §3 → §4+§5 → §6; convergence measured per-section, not per-doc).
  3. D0-22 strictly-conservative forward-declaration falsification flag (per §1.3) + corrective-amendment deferral to case study 03 anchor identification.
  4. Prose-cross-review vs artifact-cross-review modes flag (per round-prose-2 forward-flag at `5cb90cc`): the two cross-review modes have different findings characters (artifacts surface fail-open paths; prose surfaces first-impression weaknesses); v2.1.1+ §1.4 amendment candidate to distinguish.

### 5.6 Cover-letter cross-case-study integration (plan §5.3) → post-Phase-3 across cases

- **Trigger:** BOTH case studies (LB + MGN) writeups exist.
- **Why deferred:** integrates across case studies; cannot land until both writeups complete. Scope boundary noted in §3.3 to prevent loss-of-track.

### 5.7 v2.1 §2.1 SARIF schema bump 1.0 → 1.0.1 (B-medium) → continues to defer

- **Trigger** (per §2.5 forward-flag): MGN salvage scenario fires — either (a) checkpoint partial-retry, or (b) Modal-side inference timeout requiring partial-rollout SARIF surfacing.
- **Why continues to defer:** predicted path has no MGN salvage scenarios. Optional-field state remains canonical.

### 5.8 v2.1 §2.2 D0-23 SARIF schema first-class promotion (B-wide) → continues to defer

- **Trigger** (per v2.1 §2.2): case study 02's first salvage scenario.
- **Why continues to defer:** same as §5.7. Optional-field state preserves artifact-level provenance signal.

### 5.9 v2.1 §2.4 Rung-4a/4b SARIF backfill (companion to §2.1) → continues to defer

- **Trigger:** §5.7's promotion.
- **Why continues to defer:** tautological for rung-4a/4b without §2.1 promotion.

### 5.10 D0-22 strictly-conservative anchor reassignment → case study 03 candidate

- **Trigger:** strictly-conservative substrate concretely identified for case study 03.
- **Why deferred:** D0-22's forward-declaration is falsified by case study 02's P0 scope; corrective amendment waits until anchor is concretely identified rather than reassigning speculatively.

### 5.11 Prior-round-reference rollup in design doc → resolved at writeup stage

- **Trigger:** design-doc writeup stage decides between footnote-citation at first mention OR rely on `methodology/README.md` rollup.
- **Why noted here:** prevent the doc from being standalone-opaque on round-prose-1 / round-prose-2 / round-codex-4 references.

**Methodology context — v2.1 §2 deferrals state at this design's date:**

- §2.1 — open with original trigger intact.
- §2.2 — open with original trigger intact.
- §2.3 — RESOLVED at round 3 (`fa0133e`) per v2.1 §2.3 RESOLVED block.
- §2.4 — open as companion to §2.1.
- §2.5 — RESOLVED at round-prose-2 (`5cb90cc`) per v2.1 §2.5 RESOLVED block; `methodology/README.md` §4 updated parallel to §2.3.

---

## 6. Falsification readiness

v2.1 §1.5 names general confirmation / disconfirmation criteria for the A+B+C triad. This section specializes them to case study 02 P0. Case study 02 *as-fired* either confirms or falsifies the triad's claim to cross-rung generalization. Criteria pre-registered before Phase 1 fires.

### 6.1 Pattern A — MGN-specific

- **Confirmation:** at least one empirical-vs-prediction divergence surfaces during Phase 1 (smoke-time) or Phase 2 (rule outputs vs Phase-1-pre-registered tolerances), AND is absorbed via D-entry amendment *before* the affected artifact ships. Concrete candidates: NGC sample reproduction tolerance refinement; mass conservation drift bound calibration; substrate-class smoke verdict; rule-output-vs-tolerance Phase 2 amendment.
- **Disconfirmation:** either (a) divergence absorbed silently (no D-entry amendment), OR (b) Phase 1 smoke passes but Phase 2 surfaces a divergence smoke should have caught (the smoke-gate-fires-before-production discipline didn't generalize).

### 6.2 Pattern B — MGN-specific

- **Confirmation:** at least one single-artifact-multi-use-case hidden assumption surfaces in Phase 1 or Phase 2 with a *generalization* response (not special-case). Concrete P0-resolvable candidates: `_expect_velocity` key resolution; `MGN_DATASET_SYSTEM_CLASS` introduction.

   **Note on P0 confirmation strength:** per §2.2, both P0 instances are *single-instance* D-entries — the strong-form pattern-B generalization (predicate-based helper; stack-agnostic dispatch refactor) defers to amendment 1's multi-instance evidence. So P0 *names* the pattern-B surface without *firing* the full pattern-B response; the response fires at amendment 1.
- **Disconfirmation:** either (a) hidden assumptions surface in MGN but don't fit the single-artifact-multi-use-case shape (genuine multi-instance bugs; cross-runtime drift that isn't duplicate-logic), OR (b) MGN exposes assumption shapes the necessary conditions explicitly fail to capture — pattern's domain is rung-4c-specific.

### 6.3 Pattern C — MGN-specific

- **Confirmation:** at least one Phase-0 / 1 / 2 / 3 boundary cross-review surfaces findings that span ≥2 of the four cells, AND triage classifications survive an independent re-read. **Phase 0 already partially fires this:** D0-22 forward-declaration falsification (cell-1) + substrate-class extension finding (cell-2) both surfaced in §1 review. The other phases' cross-review verdicts populate the rest.
- **Disconfirmation:** either (a) findings fit no cell, OR (b) persistent classification disagreement on MGN findings reveals ambiguous boundaries §1.3 falsification rules can't resolve.

### 6.4 Triad-as-a-whole disconfirmation

Case study 02 surfaces a real and load-bearing methodology lesson that fits NONE of A, B, or C — suggests the triad is not exhaustive over case-study methodology contributions; at least a fourth member needed.

### 6.5 Layered-fail-open prediction (per §2.6 anchoring)

- **Confirmation:** 2-4 cross-review rounds against case-study-02 artifacts surface layered fail-opens of the shape predicted in §2.6 (mesh-side retry isolation, PARTIAL fallback NaN propagation, cross-precision threshold drift).
- **Disconfirmation:** 0-1 rounds find HIGH findings → cell-2 refinement (methodology over-predicts); OR the observation breaks down into a genuinely new mechanism deserving its own name → cell-4 (must be earned per §1.3 falsification rule 4).

### 6.6 Reading the readiness

None of these criteria fire *during* this design doc; they fire *during case-study-02 execution* (Phase 1/2/3). This section is the pre-registration of what would count as confirmation or disconfirmation, per v2.1 §1.5's "readiness, not validation" framing. Case study 02 is a hypothesis test of the cross-rung generalization claim; the §3.3 writeup names the verdicts explicitly per §4.3 acceptance criteria.

---

## 7. Predecessor → successor → next deliverable

- **Predecessor:** rung 4c — substrate-class extension (design + plan + table); v2.1 methodology evolution; CS02 MGN loader-contract preflight (`../../02-physicsnemo-mgn/preflight/mgn_loader_contract.md`).
- **This deliverable:** case-study-02 design (this doc).
- **Successor (immediate):** writing-plans implementation plan dated for **Phase 1 only**, layered on this design. Phase 1's audit verdicts inform Phase 2's plan shape (e.g., Gate A branch chosen; substrate-class verdict known; persistent-volume decision committed); Phase 2 + Phase 3 get a *fresh* writing-plans round after Phase 1 completes — the design covers all three phases at the planning level, but implementation plans are per-phase so verdicts feed forward correctly.
- **Successor (session 2):** Phase 1 execution per §3.1 + §4.1. **COMPLETE** at 2026-05-13 — see DECISIONS.md D0-23 (resolved) for the 10 verdicts and the cross-review findings triage table. Phase 2 inherits two forward-flags from the cross-review: (a) wire `_assert_loader_contract_mgn` into the MGN materializer / lint path's first trusted boundary (Finding 3 — Phase 2 must add a test that the lint path rejects malformed MGN rollouts); (b) reproduce the rollout-dir isolation pattern from the Phase 1 audit entrypoints + carry a smoke assertion that two same-sha retries cannot read each other's CWD-relative stats (Finding 5).
- **Successor (session 3):** Phase 2 execution per §3.2 + §4.2, off a fresh writing-plans round. **COMPLETE** at 2026-05-13 — see DECISIONS.md D0-24 (resolved) for all 7 verdicts (PASS) + the 5-finding cross-review triage table (all cell-2 absorptions). Phase 2 headline: PH-CON-001 GT = 5.857 %, MGN = 5.881 % (0.41 % gap; both within the harness-FE-on-P1 floor envelope on the canonical trajectory); PH-CON-002 / PH-CON-003 SKIP via the v9 substrate-class dispatch on both arms; F3 + F5 forward-flags from Phase 1 closed; the explicit `rollout_contract` metadata key (Finding 4 absorption) is the going-forward production identifier with the legacy `"modulus_*"` prefix retained as a backward-compat fallback. Phase 3 (writeup + cross-stack table + v2.1.1 amendment) gets a *fresh* writing-plans round next, carrying the refinement-2 forward-flag block (scope-qualifier prose + floor-bounds-resolution distinction + v2.1.2 §1.4 methodology entry).
- **Successor (post-Phase-3):** v2.1.1 amendment landing per §5.5; cover-letter cross-case-study integration (plan §5.3) post-Phase-3 once both case studies' writeups exist.
