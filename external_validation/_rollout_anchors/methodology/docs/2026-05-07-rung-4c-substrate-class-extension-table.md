# Rung 4c — Substrate-class extension to dam-break-2D (writeup)

**Date:** 2026-05-07
**Predecessor:** rung 4b cross-stack equivariance writeup at PR #8 merge sha `409bee0` (2026-05-07); rung 4c implementation on `feature/rung-4c-substrate-class-extension`; D0-22 + amendments 1 & 2 pre-registered before fire.
**Successor:** plan v2.1 amendment (next commit); integrating-README composition closing the rung-4 series (§8).

**Design doc:** [`./2026-05-07-rung-4c-substrate-class-extension-design.md`](./2026-05-07-rung-4c-substrate-class-extension-design.md)
**Plan doc:** [`./2026-05-07-rung-4c-substrate-class-extension-plan.md`](./2026-05-07-rung-4c-substrate-class-extension-plan.md)
**SARIF artifacts:** [`../../01-lagrangebench/outputs/sarif/segnn_dam2d_bc3bae929d.sarif`](../../01-lagrangebench/outputs/sarif/segnn_dam2d_bc3bae929d.sarif), [`../../01-lagrangebench/outputs/sarif/gns_dam2d_bc3bae929d.sarif`](../../01-lagrangebench/outputs/sarif/gns_dam2d_bc3bae929d.sarif). *Re-emitted at sha `bc3bae929d` (post-§9-review-gate-fold-in) to surface the optional `inference_run_status` run-level property; pre-fold-in artifacts at sha `69267191a9` superseded.*
**Rendered table:** [`../../01-lagrangebench/outputs/sarif/dam2d_table_bc3bae929d.md`](../../01-lagrangebench/outputs/sarif/dam2d_table_bc3bae929d.md).
**Pre-flight log:** [`../../../../preflight/2026-05-07-rung-4c.txt`](../../../../preflight/2026-05-07-rung-4c.txt).
**Methodology pre-registrations:** [D0-22 + amendments 1, 2](../DECISIONS.md#d0-22--2026-05-07--rung-4c-substrate-class-extension-to-dam-break-2d-pre-registration), [D0-19](../DECISIONS.md#d0-19--2026-05-04--harness-sarif-result-schema-rung-4a-pre-registration), [D0-21](../DECISIONS.md#d0-21--2026-05-05--rung-4b-cross-stack-equivariance-pre-registration), [D0-08](../DECISIONS.md#d0-08).

---

## 1. Headline

physics-lint's harness extends to a second LagrangeBench substrate class — `open-driven-dissipative` — via D0-22's substrate-class dispatch on conservation rules, exercised on dam-break-2D across SEGNN and GNS at N=12 trajectories. D0-22 amendment 1 extends the dispatch from `dissipation_sign_violation` to `energy_drift` after the pre-flight smoke surfaced that the strictly-dissipative-or-conservative assumption fails for both rules under gravitational PE→KE conversion (§5.1). Rung-4a's TGV2D conservation rule schema (`harness:mass_conservation_defect`, `harness:energy_drift`, `harness:dissipation_sign_violation`) runs unmodified on dam-break-2D rollouts; per-stack rows are emitted in the same v1.0 SARIF schema as rung-4a, with `dam2d → "open-driven-dissipative"` reclassified empirically (KE(t) measured to rise during gravity-loaded fall on both stacks at pre-flight smoke, sha `a2119906bb`) following the *classify when you exercise* discipline that rung-4b's PH-SYM-003 PBC-square-SO(2) substrate-incompatibility SKIP precedented.

The cross-stack contrast is **substrate-properties-bind-stacks-equally**: SEGNN-dam2d and GNS-dam2d emit byte-identical structural rows (3 rules × 12 trajectories each = 36 rows per stack), with `mass_conservation_defect` raw=0.0 trivially identical across both stacks (LB SPH preserves particle count), `energy_drift` SKIP via D0-22 amendment 1 on both stacks (the substrate's open-driven physics, not the model's behavior, determines the verdict), and `dissipation_sign_violation` SKIP via D0-22 on both stacks (parallel reason on the same substrate class). This is a complementary cross-stack signature to rung-4b's: rung 4b showed stacks emitting *different* values via uniform schema (architectural-difference-via-uniform-machinery, with PH-SYM-001/002 SEGNN at float32 floor vs GNS bimodal); rung 4c shows stacks emitting *identical* values via uniform schema (substrate-properties-bind-stacks-equally, with conservation rules emitting SKIP rows on both stacks via the same gate). Together they exercise the schema-uniformity claim from both sides — see §5.3.

The "12 trajectories per (rule, stack)" claim binds at the same two enforcement layers parallel to rung 4a/4b's. D0-19 §3.4 specifies that for a fixed (rule, stack), all rows MUST have identical `ruleId`, `level`, `message.text`, plus either identical `properties.raw_value` or identical `properties.skip_reason`. The renderer hard-asserts the SKIP-row half (presence of `properties.skip_reason` on every SKIP row, plus identity within (rule, stack)); raw-row identity is renderer-aggregation-observed. The N=12 (vs rung-4a/4b's N=20) is per D0-22 amendment 2 — see §5.1's smoke-discovered cost overrun and §6 (item 8).

---

## 2. Cross-stack conservation table — dam-break-2D

12 trajectories per (rule, stack) at single-step granularity (per-traj rows in committed SARIFs; condensed per-rule summary below mirrors rung-4a's pattern):

| Rule | gns-dam2d | segnn-dam2d |
|---|---|---|
| `mass_conservation_defect` | 0.000e+00 (x12 identical) | 0.000e+00 (x12 identical) |
| `energy_drift` | SKIP (x12, D0-22 (amendment 1)) | SKIP (x12, D0-22 (amendment 1)) |
| `dissipation_sign_violation` | SKIP (x12, D0-22) | SKIP (x12, D0-22) |

**Provenance (D0-19 three-sha):**

- **gns-dam2d**: pkl_inference=`e754a4bc2e`, npz_conversion=`e754a4bc2e`, sarif_emission=`bc3bae929d`
- **segnn-dam2d**: pkl_inference=`e754a4bc2e`, npz_conversion=`e754a4bc2e`, sarif_emission=`bc3bae929d`

**Inference run status (rung-4c §9 review-gate fold-in):**

- **gns-dam2d**: `from_completed_inference` (clean N=12 fire; `inference_returncode=0`)
- **segnn-dam2d**: `from_aborted_inference` (timeout-salvage; `inference_returncode=-1`, `aborted_at_step="inference"`; D0-22 amendment 2)

The `inference_run_status` field is the artifact-level provenance complement to the writeup-level §6 item 8 transparency on the SEGNN-dam2d salvage path. A reader of just the SARIFs (or the rendered table) sees the per-stack salvage classification without needing the writeup in hand. The field is optional in the harness SARIF schema (still v1.0; additive); legacy rung-4a/4b SARIFs without it render with an explicit `n/a (pre-salvage-tag-schema)` marker rather than a defaulted clean classification, parallel to the gate's refuse-by-default posture.

**Empirical justification for `dam2d → "open-driven-dissipative"`** (pre-flight smoke at sha `a2119906bb`):

| Stack | KE(0) | max(KE) | peak_t | KE(end) | rises_anywhere |
|---|---|---|---|---|---|
| SEGNN-dam2d | 4.703e-01 | 1.304e+03 | 88/105 | 1.239e+03 | True |
| GNS-dam2d | 4.703e-01 | 1.106e+03 | 98/105 | 1.092e+03 | True |

Both stacks confirm the gravity-loaded rise-then-fall KE shape that justifies the reclassification. KE(0)=0.4703 is identical to 4 sig figs across both stacks because the input window is read from the same dataset test split — KE(0) is a dataset property, not a model property. The peak-in-interior ratios (88/105 SEGNN; 98/105 GNS) are the per-stack empirical evidence; both are solidly in-interior, far from either endpoint, ruling out the failure mode where a pre-equilibration phase masks the gravitational PE→KE conversion.

---

## 3. Reading the table — interpretive framing

### 3.1 What is NOT in the evidence chain

Two artifacts that surface and could mislead a reviewer skimming for problems:

**Per-traj `raw_value=0.0` on `mass_conservation_defect` is trivial, not load-bearing.** LB SPH preserves particle count by construction (the inference pipeline's particle array is a fixed-size buffer; the harness reads the buffer's length each frame); a defect of zero is the trivial outcome that any LB-derived rollout produces under the harness's mass rule. The substantive observation is *cross-stack uniformity* of the row shape — both stacks emit the same `0.000e+00 (x12 identical)` cell — not the value itself. Reviewers should read the mass row as *evidence the schema reads dam-break particle counts correctly*, not as a model-quality claim.

**`SKIP (x12, D0-22 (amendment 1))` on `energy_drift` is not a "rule failure".** The methodology-relevant interpretation is "the rule's strictly-dissipative-or-conservative assumption does not apply on this substrate class; the harness recognizes this and emits a SKIP-with-reason instead of a methodologically-meaningless raw value." Pre-amendment-1, rung 4c's pre-flight pipeline smoke fired raw `energy_drift = 2771.39` (SEGNN) and `2349.83` (GNS) — large numbers that would have read as "model violates energy conservation" if shipped without the amendment, when in fact the violation is the substrate's gravitational PE→KE conversion (physics, not model). D0-22 amendment 1 catches this; the SKIP is the load-bearing methodological output. See §5.1 for the smoke-to-amendment loop that surfaced this.

### 3.2 Substrate-class coupling — uniformity-via-substrate vs uniformity-via-architecture

Both rung-4b and rung-4c exercise the schema-uniformity-across-stacks claim, but through different mechanisms. Rung 4b's uniformity is *uniformity-of-machinery-around-different-values*: SEGNN's float32 floor and GNS's bimodal split are *different* per-stack values, but both flow through the same v1.1 SARIF emission, the same renderer, the same band rubric. The cross-stack contrast is the load-bearing observation; uniformity is the substrate enabling that contrast. Rung 4c's uniformity is *uniformity-of-machinery-around-identical-values*: both stacks emit `mass=0.0`, `energy_drift SKIP D0-22a1`, `dissipation_sign_violation SKIP D0-22` — the rows are byte-identical, and the cross-stack invariance under that uniformity is the load-bearing observation.

The mechanism that produces identical-cross-stack-rows in rung 4c is the *substrate-property-determines-verdict* property of the conservation rules on the open-driven-dissipative class. `mass_conservation_defect` is determined by the dataset's particle count (constant per dataset, both stacks read the same input window). `energy_drift` and `dissipation_sign_violation` SKIP on the same substrate-class lookup (`metadata["dataset"] → system_class == "open-driven-dissipative"`) regardless of which model produced the rollout. The verdict is bound to the substrate, not the stack.

This is methodologically valuable in a way distinct from rung-4b's contrast. Rung 4b's evidence shape was: *equivariance is architecturally-class-bound; SEGNN exact, GNS approximate; the schema makes both visible*. Rung 4c's evidence shape is: *conservation gating is substrate-class-bound; both stacks SKIP on the same substrate; the schema uniformly produces SKIP rows that are honest about why the rule doesn't apply*. The two rungs together demonstrate that the same harness machinery handles *both* shapes of cross-stack signature — different-values-uniform-schema and identical-values-uniform-schema — without consumer-side accommodation. See §5.3.

### 3.3 Cross-stack observations (the substantive findings)

**Mass-row triviality is a feature, not a methodology gap.** Future rungs that exercise different conservation rules (e.g., a future PH-CON-NNN for momentum conservation on rotating systems) inherit the property that `mass_conservation_defect = 0.0 (xN identical)` is the trivial-outcome shape — its presence on both stacks is consistent-with-correctness, its absence (e.g., a rule emitting non-zero) would surface a real harness-layer gap.

**KE(0) identity across stacks (4.703e-01 to 4 sig figs) is the dataset-not-model invariant.** The input window is the first 6 frames of the dataset's test trajectory, fed to both models identically; both models produce their first predicted frame from the same input, and the harness's `kinetic_energy_series(rollout)[0]` reads that first predicted frame. Any divergence here would indicate a harness-layer or dataset-pipeline bug (different stacks reading different IC). Identity confirms the harness reads ICs correctly per the SCHEMA.md §1 contract.

**Peak-t separation between stacks (88/105 SEGNN vs 98/105 GNS) is observed but not interpreted in this rung.** Rung 4c's evidence chain is the schema-uniformity claim; per-stack peak-t differences are model-behavior-on-substrate observations that would belong to a different rung exercising model differences (e.g., a hypothetical rung 4e that compares trajectory metric MSE/Sinkhorn cross-stack on the same substrate). Reported here as observed, deliberately not interpreted.

**KE-rest threshold misfire (KE(0)=0.47 >> 1e-10 absolute) was the smoke-discovered drift that motivated D0-22 amendment 1.** The original D0-08 KE-rest gate was designed for normalized-energy systems where peak KE ~ O(1); on dam-break with peak KE ~ O(1000), the absolute threshold doesn't capture "at rest" relative to peak. D0-08 stays unmodified (cross-cutting it would risk regressing rung-4a's TGV2D); D0-22 amendment 1 routes around it with a per-substrate-class dispatch. See §5.1.

---

## 4. Pre-flight smoke evidence (figure-equivalent)

Rung 4c does not ship a separate figure (the cross-stack table is the artifact). The pre-flight smoke at sha `a2119906bb` is the empirical-justification evidence for the `dam2d → "open-driven-dissipative"` reclassification; full trajectory-level KE(t) numbers are recorded in the [pre-flight log](../../../../preflight/2026-05-07-rung-4c.txt). The §2 summary table above condenses the gating evidence; the full log additionally records the conversion round-trip (Step 2), the rule sanity tests (Step 3), and the end-to-end pipeline smoke (Step 5) that gated the production fire.

---

## 5. Methodology lessons

**Three durable methodology outputs from this rung,** all of which generalize beyond rung-4c to future case studies (PhysicsNeMo MGN — case study 02 — and any subsequent neural-physics integration). The three are *paired* in §5.1–5.2 (different-trigger fix-in-rung patterns) and *complementary* in §5.3 (cross-rung schema-uniformity composite).

> **Methodology evolved post-writeup.** This writeup is sha-bound to its committed snapshot (sha `bc3bae929d`); the methodology framing has since been refined in [plan v2.1](./physics-lint-validation-plan-v2.1.md) and the [integrating README](../README.md). Round-prose-1 (2026-05-11) added necessary conditions to pattern B (§1.2 demoted the round-3 "implementation-coordination level" framing — duplicate-logic drift is now treated as adjacent-but-distinct, not a pattern-B instance); round-codex-2 (2026-05-11) added the `manifest_required=True` policy for post-fold-in standalone-conversion entrypoints. Readers tracking the current methodology view should consult v2.1 + the integrating README alongside this writeup; readers reading this writeup for the rung-4c-snapshot view should treat the framings below as authoritative *at sha `bc3bae929d`* but not necessarily *post-rung-4c*.

### 5.1 Smoke-discovered drift → in-rung amendment (paired pattern A)

Empirical observation contradicts plan prediction; the response is to amend the methodology trail (D-entry amendment) recording what was learned. Two within-rung-4c instances:

**D0-22 amendment 1** (commit `e754a4b`, post-Step-5 pipeline smoke). Plan §1.2 + design §1.2 predicted `energy_drift` would SKIP via D0-08 KE-rest gate on dam-break (start-at-rest IC). Empirically false: KE(0)=0.47 is well above `KE_REST_THRESHOLD=1e-10` absolute (the threshold is in absolute energy units; dam-break KE scale is O(1000), so the relative-to-peak ratio is 3.6e-4 but the absolute clears by 9 orders of magnitude). Result: `energy_drift` fired raw with a methodologically meaningless ~2500-2700 value. The same strictly-dissipative-or-conservative assumption that D0-22 catches on `dissipation_sign_violation` ALSO fails for `energy_drift` on the same substrate class. Amendment 1 extends the substrate-class dispatch to cover `energy_drift`; the SKIP-with-reason replaces the misfire.

**D0-22 amendment 2** (commit `6926719`, post-Task-8 production fire). Plan §3.4 estimated ~5min/20-traj A10G; production rollout at N=20 timed out at 2400s subprocess cap with 12/20 trajs converted-pending. Amortized rate ~200s/traj on dam-break SEGNN, ~3.3 min/traj — ~10x optimistic vs the plan's projection. Amendment 2 records the cost-driven N reduction (rung 4c ships at N=12 across both stacks) and the rationale for choosing in-rung correction (Option B) over budget-funded uniformity (Option A): the rung-4 series's smoke-discovered-drift discipline weighs methodology consistency higher than presentational parity with rung-4a/4b's N=20.

**Pattern shape:** trigger is *empirical-vs-prediction divergence*; response is *methodology-trail-amendment recording what the smoke surfaced*; the underlying code changes (extend SKIP gate; canonicalize n_trajs=12) are the consequence of the methodology amendment, not the amendment itself. Future case studies inherit this: when the smoke surfaces a fact the plan didn't predict, the disciplined response is to land an amendment to the relevant pre-registration before shipping the artifact.

### 5.2 Implementation-time hidden assumption → in-rung generalization (paired pattern B)

Code that worked in single-instance use breaks under multi-instance use; the response is to generalize the code to remove the hidden assumption. Both within-rung-4c instances surfaced at Task 10 (SARIF emission + table render):

**Renderer's single-schema-version assumption (`--include-glob` flag).** Pre-rung-4c, `render_cross_stack_table.py` used `--sarif-dir` and globbed `*.sarif`. The fail-loud schema-version assertion (EXPECTED_SCHEMA_VERSION="1.0") worked fine when `outputs/sarif/` contained only conservation SARIFs at v1.0. Once rung 4b landed v1.1 eps SARIFs in the same directory (still rung 4b's single-renderer-instance use), the assumption became visible: any future use that needed to render only the v1.0 subset would trip the assertion on the v1.1 files. Rung 4c was the first cross-version-mixed-dir instance; the fix added `--include-glob` (default `*.sarif`) so callers can filter explicitly. Default behavior unchanged; the schema-strict contract holds within the filtered set.

**Renderer's hardcoded `D0-18` cell label (regex extraction).** Pre-rung-4c, the SKIP cell label was `f"SKIP (x{n}, D0-18)"` because D0-18 was the only SKIP path the schema knew about (rung 4a + 4b both emitted only D0-18 SKIPs on conservation rules). Once rung 4c added D0-22 + amendment 1 SKIP paths, the hardcoded label became wrong (everything still showed `D0-18`). The fix extracts the actual D-entry from the skip_reason string via regex `DECISIONS\.md\s+(D0-\d+(?:\s+\(amendment\s+\d+\))?)`, supporting D0-08, D0-18, D0-22, and D0-22 (amendment 1) cell labels uniformly. Required also adding the `DECISIONS.md D0-18` reference to the synthetic test fixtures' skip_reason — real skip_reasons all carry this prefix; the fixture now matches that convention rather than relying on the renderer's hardcoded fallback.

**Pattern shape:** trigger is *single-instance-vs-multi-instance — code that "happened to work" because there was only one instance breaks when a second instance arrives*; response is *generalize the code to remove the implicit assumption*. Distinct from pattern A in that the trigger is structural (multi-instance use cases are now exercised), not empirical (no plan prediction was contradicted). The renderer changes are renderer-generalizations, not rung-4c-specific patches; future rungs that need to render schema-mixed dirs or D-entries beyond D0-18 inherit them.

**Design-vs-writeup honest amendment.** The design doc §2.1 + §5.1 pre-registered "renderer UNCHANGED" as load-bearing evidence of substrate-class extension at the consumer side; design §1.2's frozen headline echoed this with "no consumer-side accommodation needed." This writeup walks that framing back: the renderer DID change, by adding `--include-glob` and the D-entry regex extraction described above. The substantive claim survives — the rule schema runs unmodified across substrate classes, no substrate-specific consumer code was added — but the framing claim ("no consumer-side change at all") does not. Naming this honestly mirrors the rung-4 series's discipline: pre-registrations that turn out to be over-strong get walked back at writeup time, not glossed. The frozen-headline-vs-reality drift is itself a third instance of paired pattern A within this rung — the design's prediction (renderer untouched) was contradicted by Task 10's renderer changes — and it lands as a writeup-time honest-limits acknowledgment rather than a D-entry amendment because the paired pattern B response (generalize, don't patch) was what the implementation surfaced.

**Bilateral-exercise elevation (post-Task-11 §9 review-gate).** A second pattern-B context surfaced at the §9 cross-review gate (Codex adversarial round): the standalone-conversion entrypoint `convert_pkls_p1_segnn_dam2d` (modal_app.py — added at D0-22 amendment 2 for the rung-4c salvage) lacked run-completion-marker enforcement, so an operator could fire it against a timed-out rollout subdir and get a generic PASS verdict with no signal that the upstream inference was aborted. The same single-instance-vs-multi-instance shape as the renderer fixes — the entrypoint "happened to work" for D0-17 amendment 1's conversion-bug-recovery case (the only instance pre-rung-4c), and the timeout-salvage case D0-22 amendment 2 added reused the same path without distinguishing. The post-review-gate fold-in adds (i) orchestrator-side persistence of the gate-relevant manifest fields to `<rollout_subdir>/_inference_manifest.json` (atomic write, success / inference-timeout / conversion-failure paths uniformly) and (ii) a default-refuse gate in `lagrangebench_convert_pkls_in_volume` keyed off the persisted manifest's `inference_returncode` + `aborted_at_step`, with an opt-in `--allow-from-aborted-inference` flag for the explicit timeout-salvage case (rung-4c's `segnn_dam2d_e754a4bc2e` salvage now requires the explicit flag). Pattern B is now bilaterally exercised within rung-4c at two pipeline layers: Task-10 consumer-side renderer (single-schema-version + hardcoded-D-entry assumptions) and post-review-gate modal-side conversion (single-callable-case assumption); both are pipeline generalizations, not rung-4c-specific patches. Rung-4c artifacts retain self-contained provenance via backfilled `_inference_manifest.json` files at both `segnn_dam2d_e754a4bc2e` and `gns_dam2d_e754a4bc2e` (verification fire `outputs/rung4c_gate_refusal_verification.log` confirms refusal end-to-end). Plan v2.1 (Task 12) carries the longer narrative on cross-review-gate as the parallel discipline to source-review and smoke-review.

### 5.3 Bilateral schema-uniformity composite (rung-4b stacks-differ + rung-4c stacks-identical)

Cross-rung methodology contribution that emerges only when 4b and 4c are read together:

- **Rung 4b:** stacks emit *different* values via *uniform* schema — architectural-difference-via-uniform-machinery. PH-SYM-001/002 active rows separate SEGNN's float32 floor from GNS's bimodal APPROXIMATE+FAIL distribution; the schema is the substrate that *makes the contrast visible*.
- **Rung 4c:** stacks emit *identical* values via *uniform* schema — substrate-properties-bind-stacks-equally. mass=0.0, energy_drift=SKIP, dissipation_sign_violation=SKIP for both stacks; the schema is the substrate that *makes the invariance visible*.

The composite claim is stronger than either rung alone: *the harness's rule schema is uniform-across-stacks both when stacks behave differently (rung 4b) and when they behave identically (rung 4c), depending on what the rule's structure measures*. The schema-uniformity claim is bilaterally exercised — under both shapes of cross-stack signature — within the rung-4 series, and the integrating README composes them as one cross-rung methodology artifact rather than two independent rung-level claims. See §8.

This is a third independent contribution alongside the paired patterns above; it's not a "fix-in-rung" pattern but a cross-rung observation that consolidates after both rungs land.

---

## 6. What rung 4c is NOT

Verbatim from design §1.3 + the additions surfaced during execution (item 8):

1. **Not a bilateral test of D0-18's mechanism.** D0-18 (dissipative-by-design SKIP on `energy_drift`) requires `system_class == "dissipative"` AND `KE(t)` monotone-non-increasing. Dam-break post-D0-22 has `system_class == "open-driven-dissipative"` (not equal), so D0-18 does not fire on dam-break. The bilateral D0-18 forward-flag from rung-4a §1.3 (5) — requiring a *strictly conservative* substrate where `energy_drift` evaluates raw_value normally — stays intact and unfulfilled, pointing at case study 02 (PhysicsNeMo MGN incompressible NS as a candidate conservative anchor) or a future case study.

2. **Not a SEGNN-vs-GNS model comparison.** Both stacks emit byte-identical structural rows. Model differentiation lives in equivariance (rung 4b, already landed). The cross-stack uniformity here is the load-bearing evidence that the harness's rule schema runs unmodified across architectures on a second substrate class, not a probe of model differences.

3. **Not the integrating top-level README.** Composed downstream after rung 4c lands; this writeup is the named-event trigger for that composition (§8). Rung 4c writeup is a dated deliverable under `methodology/docs/`, parallel to rung-4a/4b.

4. **Not a wall-non-penetration claim.** Plan v2 §3.1 P1's "PH-BC (wall)" entry assumed an SPH-particle-wall rule that does not exist in physics-lint v1.0. Plan v2 → v2.1 amendment removes "PH-BC (wall)" from §3.1; rung 4c emits no wall-non-penetration row. Wall-non-penetration as a future SPH-particle rule (PH-BC-NNN with `particle_type == WALL` flag) is post-visa work, not bundled here.

5. **Not a multi-rung renderer.** Rung 4c stands alone — `render_cross_stack_table.py` reads the two dam-break SARIFs (filtered via `--include-glob "*dam2d*.sarif"`) and emits a dam-break-only cross-stack table, mirroring rung-4a's pattern. Cross-rung composition (rung-4a TGV2D + rung-4b equivariance + rung-4c dam-break integrated into one artifact) is the integrating-README's job, not rung 4c's. *Distinct from the within-rung-4c renderer generalizations of §5.2* — those are about multi-schema-version handling and multi-D-entry labeling, not cross-rung composition into one table; the renderer was modified for generalization, not for cross-rung-composition support.

6. **Not a catalogue-wide reclassification.** Only `dam2d` is reclassified empirically. `rpf2d`, `ldc2d`, `rpf3d`, `ldc3d`, `tgv3d` retain their pre-D0-22 `"dissipative"` labels. The *classify when you exercise* discipline rules out preemptive reclassification without empirical probing; a future rung exercising rpf or ldc walks into a known-misclassification (D0-22 §4 two-tier split) and the empirical probe is its first move.

7. **Not a multi-rule-trigger-axis abstraction.** D0-21 §forward-flag-2 named the "(rule, substrate) compatibility matrix" as a future generalization. Rung 4c's D0-22 + amendment 1 demonstrates that pattern's empirical instance — `dissipation_sign_violation × open-driven-dissipative` and `energy_drift × open-driven-dissipative` both become new SKIP cells — without yet promoting the matrix to a first-class rule-schema field. Promotion is post-rung-4c work.

8. **Not a cross-rung-N-uniform artifact.** Rung 4a/4b ship at N=20 trajectories per (rule, stack); rung 4c ships at N=12 per D0-22 amendment 2 (smoke-discovered cost overrun). The structural claims hold at N≥2 and are unaffected; cross-rung N parity is the cost paid for the smoke-discovered-drift-gets-in-rung-correction discipline (§5.1). The integrating README's cover-letter sentence picks this up explicitly: "rung-4a TGV2D at N=20 trajs; rung-4b TGV2D at N=20 trajs; rung-4c dam-break-2D at N=12 trajs (per-traj inference cost ~10x estimate; D0-22 amendment 2)."

   **Standalone-conversion provenance addendum (§9 review-gate fold-in).** The SEGNN-dam2d N=12 NPZs were produced via the standalone-conversion path (`convert_pkls_p1_segnn_dam2d`) against a timed-out N=20 inference subdir; the writeup's headline (§2) and the §7 provenance shas describe this honestly, but per the bilateral-exercise elevation in §5.2, the standalone-conversion path itself is now gated against silent reuse of timed-out artifacts — operators must pass `--allow-from-aborted-inference` to invoke the salvage path, and the gate verdict explicitly tags the NPZs as `from_timeout_salvage` rather than a generic `PASS`. Rung-4c artifacts retain self-contained provenance via backfilled `_inference_manifest.json` files at both `segnn_dam2d_e754a4bc2e` (`inference_returncode=-1`, `aborted_at_step="inference"`) and `gns_dam2d_e754a4bc2e` (`inference_returncode=0`); the gate refusal was end-to-end-verified post-backfill (`outputs/rung4c_gate_refusal_verification.log`).

---

## 7. Rederivability + provenance

The full pipeline from Modal Volume artifacts to this writeup's table is reproducible via three scripted steps (assumes Modal auth + the Volume state from the rung-4c Task 8 fire):

```bash
# 1. Mirror the dam2d rollout subdirs locally (~30 sec per stack):
modal volume get rollout-anchors-artifacts \
    /rollouts/lagrangebench/segnn_dam2d_e754a4bc2e \
    external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror/
modal volume get rollout-anchors-artifacts \
    /rollouts/lagrangebench/gns_dam2d_e754a4bc2e \
    external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror/

# 2. Emit the four SARIFs (TGV2D pair re-emitted at current sha; DAM2D pair freshly emitted):
python external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py

# 3. Render the dam-break cross-stack table (filter to dam2d-only via --include-glob):
python external_validation/_rollout_anchors/methodology/tools/render_cross_stack_table.py \
    --sarif-dir external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/ \
    --include-glob "*dam2d*.sarif" \
    > external_validation/_rollout_anchors/01-lagrangebench/outputs/sarif/dam2d_table_<sarif_emission_sha>.md
```

Re-run at the same sha with the committed SARIFs at that sha → identical output. The renderer's output is deterministic; any divergence reflects a SARIF artifact change, a renderer change, or both — all caught by `methodology/tests/test_render_cross_stack_table.py`.

**3-stage sha provenance** (per D0-19, applied uniformly to both stacks):

| Stage | SEGNN-dam2d | GNS-dam2d |
|---|---|---|
| pkl_inference | `e754a4bc2e` | `e754a4bc2e` |
| npz_conversion | `e754a4bc2e` | `e754a4bc2e` |
| sarif_emission | `bc3bae929d` | `bc3bae929d` |

Both stacks share the inference + conversion sha because both were fired at the same physics-lint sha (`e754a4bc2e` — the post-D0-22-amendment-1 commit). SEGNN's npz_conversion came via the standalone `convert_pkls_p1_segnn_dam2d` entrypoint after the production inference timed out at 12/20 trajs (D0-22 amendment 2 §rationale); GNS's came via the in-fire conversion at n_trajs=12. Both produce N=12 schema-conformant npzs at the rung-4c canonical N.

**Pinned external dependencies:**

- LagrangeBench sha (captured at image-build, `--depth 1` clone of master): `b880a6c84a93792d2499d2a9b8ba3a077ddf44e2` (verified at Task 5 dataset-directory discovery; identical to rung-4a/4b's pinned LB sha).
- Dataset directory: `2D_DAM_5740_20kevery100` (discovered at Task 5; LB's name per `configs/dam_2d/base.yaml` at the pinned sha).
- SEGNN-DAM2D checkpoint gdown ID: `1_6rHxK81vzrdIMPtJ7rIkeoUgsTeKmSn` (from LB upstream's pretrained-models table; fetched once via gdown, cached on Volume).
- SEGNN-DAM2D checkpoint sha256-namespaced (computed by `_hash_directory` over the unpacked `/best/` dir; recorded in npz metadata): `eb482dd8b4469f2cdc1333b8fc62256123eea0a400f517c2dbf01617c381a3bf`.
- GNS-DAM2D checkpoint gdown ID: `16bJz3VfSMxOG1II8kCg5DlzGhjvdip2p` (same provenance shape).
- GNS-DAM2D checkpoint sha256-namespaced: `cb0db8bcc11bb5d1fb6a070b87ce69caf981fb9a5bf196deee128c6c3fe758ba` (parity with rung-4b's sha256 listing for Stuttgart/Munich-reviewer reproducibility audit).

**Total Modal compute spent across rung 4c:**

- Pre-flight Step 4 SEGNN-dam2d 1-traj smoke: ~5 min A10G (dataset download dominated; checkpoint download + 1 traj inference): ~$0.07
- Pre-flight Step 4 GNS-dam2d 1-traj smoke: ~2 min A10G (cached dataset; ckpt download + 1 traj): ~$0.03
- Production SEGNN-dam2d N=20 attempt: 40 min A10G (subprocess timeout; 12/20 converted-pending): ~$0.57
- Production SEGNN-dam2d standalone conversion: ~30 sec CPU Modal: ~$0.01
- Production GNS-dam2d N=12: 15 min A10G: ~$0.21
- **Total: ~$0.89** (~5x the per-rung $0.20 estimate, ~3% of the ~$30 total-validation budget ceiling)

The cost overrun was the smoke-discovered drift from §5.1 paying in compute rather than in cycles-saved (vs rung 4b's source-review pattern which saved compute). Both shapes are line items in the methodology budget worth their margin: rung 4b's pattern catches issues *before* compute (saving ~$0.50 in re-fires); rung 4c's pattern catches issues *during* compute (paying ~$0.57 for the timeout discovery, then routing around it via amendment 2). Methodology lessons compound: rung-4 series total compute remains under $5 cumulative, well under the $30 ceiling.

---

## 8. Integrating-README composition trigger

This dated writeup is the trigger that composes the integrating top-level README per rung-4b's writeup §8 (which deferred composition until rung 4c's writeup landed). All three rung-4 dated writeups (4a, 4b, 4c) are now committed; the integrating-README composition can land as a follow-up commit.

**Composition path:** `external_validation/_rollout_anchors/methodology/README.md` (overwrite the predecessor README, whose current content was the rung-4a-pre-4b-pre-4c state).

**Composition shape (TBD as a follow-up commit, likely after plan v2.1 amendment + PR review):**

- **One-paragraph summary** linking rung 4a (cross-stack conservation, PH-CON rule schema, TGV2D), rung 4b (cross-stack equivariance, PH-SYM rule schema, TGV2D), and rung 4c (substrate-class extension, PH-CON rule schema on a second substrate, dam-break-2D) under one narrative thread: *physics-lint's harness machinery runs the same rule schemas across architecturally distinct neural-physics stacks AND across substrate classes within one architecture, with per-stack rows emitted in a unified SARIF schema (v1.0 conservation, v1.1 equivariance) and substrate-class dispatch handled at the harness layer rather than the consumer layer.*

- **Three independent cross-rung methodology contributions** (composed from rung-4c §5):
  - **Bilateral schema-uniformity composite (rung-4b stacks-differ + rung-4c stacks-identical).** The schema is uniform-across-stacks both when stacks behave differently (architectural-class-bound, rung 4b) and when they behave identically (substrate-property-bound, rung 4c). A stronger composite claim than either rung alone (rung-4c §5.3).
  - **"Smoke-discovered drift → in-rung amendment" pattern (paired pattern A from rung-4c §5.1).** Two within-rung-4c instances (D0-22 amendments 1 and 2); generalizes to: when the smoke surfaces a fact the plan didn't predict, the disciplined response is to land an amendment to the relevant pre-registration before shipping the artifact. Cited as the natural extension of rung-4b's "loader-contract pre-flight" pattern (rung-4b §5.1) — both are smoke-driven discipline patterns; rung-4b catches *before* compute, rung-4c catches *during* compute.
  - **"Implementation-time hidden assumption → in-rung generalization" pattern (paired pattern B from rung-4c §5.2).** Two within-rung-4c instances (renderer's `--include-glob` for schema-version filtering; D-entry extraction in cell labels). Distinct trigger from pattern A (structural, not empirical); same fix-in-rung response shape; generalizes the harness machinery beyond rung-4c's specific needs. Renderer is now multi-schema-version-mixed-dir-aware and multi-D-entry-aware.

- **"Classify when you exercise" empirical-classification principle.** Trilateral across the rung-4 series: rung 4b's PH-SYM-003 PBC-square-SO(2) substrate-incompatibility SKIP, rung 4c's `dam2d` empirical reclassification, and rung 4c's catalogue-wide forward-flag two-tier split. Substrate properties get verdicts only after empirical probing.

- **"Source-review-catches-issue-before-compute" pattern (now bilateral within rung-4 + extended).** Three rung-4 instances: rung-4b first-pass math correction (TRAIN_PUSHFORWARD_UNROLLS_LAST), rung-4b first-pass figure-sweep failure (valid.h5 hardcoded subseq_length), rung-4c catalogue-misclassification (dam2d preemptive `dissipative` label). All three caught at $0 Modal cost. Cited alongside rung-4c §5.1's "smoke-during-compute" pattern as complementary discipline modes.

- **Forward-flag for case study 02 (PhysicsNeMo MGN):** all five contributions inherit. The materializer's pre-flight assertions section gets a sibling "MGN loader-contract assertions" alongside the existing "LB loader-contract assertions"; the substrate-class taxonomy gains entries as MGN's substrates are empirically probed; the amendment-layered DECISIONS.md pattern carries forward unchanged. The integrating README is the named-event durable artifact that future case studies read.

- **N-consistency forward-flag.** Rung-4a/4b at N=20, rung-4c at N=12 (D0-22 amendment 2). Future rungs touching dam-break-2D at N=20 require explicit timeout refactor (subprocess `timeout=2400` → `5400+` minimum + corresponding function timeout); recorded in D0-22 amendment 2 §forward-flag.

The integrating README composition is the rung-4 series's closure deliverable. Rung-4c's next adjacent step (Task 12 — plan v2.1 amendment) lands first, then the integrating README composes after PR review surfaces any final framing adjustments.
