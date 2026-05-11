# Physics-lint validation plan — v2.1

**Date:** 2026-05-10 (initial) / 2026-05-11 (round-3 absorption appended)
**Status:** v2.1 amendment of [`physics-lint-validation-plan-v2.md`](physics-lint-validation-plan-v2.md). v2 stays frozen at its original path; v2.1 is the corrected and absorption-extended document.
**Predecessor:** rung-4 series (4a + 4b + 4c) committed; rung-4c §9 review-gate fold-in rounds 1+2 closed at sha `3037209` (2026-05-10); round 3 appended at 2026-05-11 against the v2.1-committed sha `9952170`.

## Trigger — two layers

**Layer 1 — source-review trigger (rung-4c design pass, 2026-05-07).** Plan v2 §3.1 P1's "PH-BC (wall)" entry assumed an SPH-particle wall rule that does not exist in physics-lint v1.0 (PH-BC-001 in production is Dirichlet boundary trace on a unit square — a mesh-FEM rule, not particle-wall). The mismatch was caught during rung-4c design-pass source review of the production rule set, before any rung-4c implementation began. v2.1 corrects the row honestly. Third instance of source-review-catches-issue-before-compute in the rung-4 series (rung-4b math + rung-4b figure-sweep + rung-4c plan-rule mismatch); see rung-4b T7 design §14.6 for the precedent acknowledgment shape.

**Layer 2 — cross-review trigger (rung-4c §9 review-gate fold-in rounds 1+2+3, 2026-05-08 to 2026-05-11).** The rung-4c writeup's §9 cross-review gate prescribed Codex adversarial review of the as-shipped artifacts. Three rounds fired:

- Round 1 (commit `bc3bae9`) — surfaced the standalone-conversion entrypoint's missing run-completion-marker enforcement (operators could fire `convert_pkls_p1_segnn_dam2d` against a timed-out subdir and get a generic PASS verdict). Absorbed in-rung as the modal-side standalone-conversion gate.
- Round 2 (commit `3037209`) — surfaced (A) untracked `outputs/*.log` automation-pickup risk and (B) gap between the bc3bae9 conversion-side gate and SARIF artifact-level provenance. Absorbed in-rung as `/outputs/` gitignore + optional `inference_run_status` SARIF run-property + renderer salvage-tag section + 3 new renderer tests.
- Round 3 (committed at the same sha as this v2.1 round-3 absorption) — surfaced two HIGH findings that the gate added at round 1 and the SARIF salvage-tag added at round 2 **fail open in the exact corrupted/stale-manifest scenarios they are supposed to make auditable**. Both `_classify_inference_run_status` (Modal-side) and `_read_inference_manifest_status` (local-side) collapsed manifest-corruption (JSON decode failure, missing required fields, non-dict root) into the generic legacy-absent status (`from_unknown_inference` → warn-allow at the gate; `None` → silently-omit at SARIF emission). Absorbed in-rung as: (i) a fourth status `manifest_invalid` distinct from legacy absence; (ii) the standalone-conversion gate refuses `manifest_invalid` unconditionally with no override flag (corruption is structural, not policy); (iii) `_emit_for_stack` gains `manifest_required: bool = False` and raises `FileNotFoundError` / `ManifestInvalidError` on missing-or-corrupt for post-fold-in (rung-4c dam2d) stacks; (iv) classification + persistence helpers promoted to shared `_harness/inference_manifest.py` after Codex review surfaced that the duplicate implementations both carried the same fail-open bug — which is exactly the failure mode v2.1 §2.3 named as the promotion trigger. Plus 21 new tests covering corrupt-JSON, missing-required-fields, non-dict-root, missing-when-required, and persist-then-classify-roundtrip cases.

The rung-4 series's methodology lessons crystallized through these fold-ins into a paired-pattern triad (A + B + C) that v2.1's methodology section uses as its central organizing structure.

## Pattern (combined)

The PH-BC mismatch is **pattern A** (smoke/source-discovered drift; here surfaced at design-pass instead of post-fire) → in-rung amendment via this v2.1. The §9 fold-in absorptions are **pattern C** (review-gate finding → triage → conditional absorption): one re-discovery (cell 1, defer-with-marker; round-2's "re-fire SEGNN at N=20") and several novel-in-scope (cell 2, in-rung absorption; the standalone-conversion gate, the SARIF salvage-tag, the gitignore). v2.1 records both the absorptions and the pattern triad they made visible.

---

## §0. Diff from v2 — substantive edits

### §3.1 P1 row update

**Before (v2):**

```
| P1 | Dam break 2D | GNS | PH-CON-001 (mass), PH-BC (wall) |
```

**After (v2.1):**

```
| P1 | Dam break 2D | GNS + SEGNN | Substrate-class extension to open-driven-dissipative
                                    (D0-22 + amendments 1, 2):
                                      • PH-CON-001 mass: raw = 0.0 (×N identical, both stacks)
                                      • dissipation_sign_violation: SKIP (D0-22)
                                      • energy_drift: SKIP (D0-22 amendment 1; replaces v2's
                                                       D0-08 KE-rest assumption — empirically
                                                       false on dam-break, see writeup §5.1)
                                      • N = 12 trajs per stack (D0-22 amendment 2; cost-driven
                                                       reduction from rung-4a/4b's N = 20) |
```

**Rationale.** Three independent corrections to v2's row, all surfaced during rung-4c execution and captured in committed D-entries:

1. **"PH-BC (wall)" struck.** Assumed an SPH-particle wall rule that does not exist in physics-lint v1.0; caught at rung-4c design-pass source review. Wall-non-penetration as a future SPH-particle rule (PH-BC-NNN with `particle_type == WALL` flag) is post-visa work, not bundled here.
2. **Architecture column expanded `GNS → GNS + SEGNN`.** Dual-stack scope absorbs v2's separate P3 row (see §3.1 P3 row absorption below). Cross-stack uniformity *is* the cross-validation in the substrate-class-extension framing.
3. **Headline-rule column rewritten.** Substrate-class-extension framing per rung-4c design §1.2; energy_drift's SKIP path was retargeted from D0-08 KE-rest (v2's prediction) to D0-22 amendment 1 (post-smoke reality; D0-08's absolute threshold misfires on dam-break's KE = O(1000) scale; see writeup §5.1). N = 12 reflects D0-22 amendment 2's cost-driven reduction (per-traj inference cost ~10× plan estimate; in-rung methodology-consistency over presentational uniformity).

### §3.1 P3 row absorption

**Before (v2):**

```
| P3 | Dam break 2D | SEGNN | Cross-validate P1 result |
```

**After (v2.1):** struck. Dual-stack P1 absorbs P3's SEGNN-dam2d cross-validation goal; the byte-identical structural rows across SEGNN-dam2d and GNS-dam2d (writeup §2) *are* the cross-validation.

### §3.2 step 6 dam-break-per-stack template

**Before (v2):** included a `PH-BC-001` row.

**After (v2.1):** strikes the `PH-BC-001` row; uses the rule-id naming the harness actually emits.

```markdown
### Dam break 2D — <stack>
- Checkpoint: <stack>_dam2d, best/, SHA-256 <hash>
- Rollout: 12 trajectories × 105 steps (rollout horizon = 100; +5-frame input window per LB convention)
- `harness:mass_conservation_defect`: raw = 0.0 (×12 identical) — trivial-outcome (LB SPH preserves particle count by construction; cross-stack uniformity is the load-bearing observation, not the value)
- `harness:dissipation_sign_violation`: SKIP × 12, `DECISIONS.md D0-22` (open-driven-dissipative substrate; substrate-class dispatch)
- `harness:energy_drift`: SKIP × 12, `DECISIONS.md D0-22 (amendment 1)` (same substrate-class dispatch extended to energy_drift after pre-flight smoke; replaces v2's D0-08 KE-rest assumption)
- Inference run status (optional, post-§9-fold-in): `from_completed_inference` | `from_aborted_inference` | `from_unknown_inference`
```

### §5.3 cover-letter dam-break sentence

**Before (v2):** mentioned "PH-BC" in the LagrangeBench case-study sentence; no reference to substrate-class extension or to N-per-rung.

**After (v2.1):**

> "...extending physics-lint's harness substrate-detection layer to a second LagrangeBench substrate class (open-driven-dissipative) via dam-break-2D (D0-22 + amendments 1, 2; rung-4c at N=12 per-stack vs rung-4a/4b at N=20 — D0-22 amendment 2 records the cost-driven N reduction transparently), with the same SARIF schema as TGV-2D conservation and no consumer-side accommodation needed beyond two renderer generalizations (multi-schema-version filtering and D-entry extraction; see plan v2.1 §1.2 pattern B)."

The N-per-rung explicit citation is the writeup §6 item 8 framing carried into the application-facing artifact. Reviewers who read the cover letter and are concerned by the N=12 vs N=20 asymmetry have an immediate pointer to the methodology-consistency rationale.

### §6 risks register additions

Three new entries layered on top of v2's table:

> *Plan-vs-actual-rule mismatch surfaced during implementation.* Plan v2 §3.1 P1 specified "PH-BC (wall)" for dam-break-2D, but no SPH-particle wall rule existed in physics-lint v1.0 (PH-BC-001 in production is Dirichlet boundary trace on a unit square — a mesh-FEM rule, not particle-wall). The mismatch was caught at rung-4c design-pass source review of the production rule set, before any implementation. Plan v2.1 §3.1 corrects the row honestly. **Mitigation forward.** When planning future rungs that name-reference rule IDs, verify the named rule exists in `external_validation/PH-*/` at design time, not at writeup time. Pattern is the third instance of source-review-catches-issue-before-compute in the rung-4 series (paralleling rung-4b T7 design §14.6).

> *Plan-vs-actual N-per-rung mismatch surfaced during compute.* Plan v2 §3.4 estimated ~5 min/20-traj A10G for dam-break inference; production fire timed out at 2400s subprocess cap with 12/20 trajs converted-pending (per-traj rate ~3.3 min/traj on dam-break SEGNN, ~10× optimistic vs the plan's projection). D0-22 amendment 2 records the in-rung methodology-consistency-over-presentational-parity choice (Option B over Option A at amendment time; see writeup §6 item 8 for the verbatim reasoning trail). **Mitigation forward.** Future rungs touching dam-break-2D at N≥20 require explicit subprocess-timeout refactor (`timeout=2400 → 5400+ minimum` + corresponding function-timeout); recorded in D0-22 amendment 2 §forward-flag.

> *Standalone-conversion entrypoint silently re-used timed-out artifacts.* Pre-§9-review-gate, the modal_app `convert_pkls_p1_segnn_dam2d` entrypoint (added at D0-22 amendment 2 for the rung-4c salvage path) accepted any rollout subdir with PKL files and emitted a generic conversion PASS regardless of whether the upstream inference completed or was aborted. Surfaced by Codex round-1 cross-review of the as-shipped artifact set. **Mitigation forward.** Standalone-conversion entrypoints must persist a run-completion manifest (`_inference_manifest.json`) atomically alongside the rollout subdir on every fire-path (success / inference-timeout / conversion-failure), and downstream conversion entrypoints must default-refuse on aborted-inference status with an explicit `--allow-from-aborted-inference` opt-in for documented salvage cases. Implemented in rung-4c §9 fold-in round 1 (`bc3bae9`); both rung-4c subdirs carry the manifest via post-hoc backfill. Pattern shape is the bilateral-exercise elevation of pattern B (see §1.2 below).

### §11 Changelog (new section, appended)

(Full changelog at §3 below; this is the in-section pointer.)

---

## §1. Methodology — paired-pattern triad as central organizing structure

The rung-4 series's methodology lessons resolve into three patterns with structural symmetry. Each names a divergence type, a response shape, and the enabling discipline. Together they form the v2.1 methodology section's central organizing structure; case study 02 (PhysicsNeMo MGN) is the bilateral-validation surface that will exercise A + B + C from a different framework integration's perspective.

### §1.1 Pattern A — smoke-discovered drift → in-rung amendment

- **Trigger:** empirical-vs-prediction divergence. The plan / design / pre-registration predicted some property of the data, the rule, the rollout, the cost, or the schema; empirical execution contradicted it.
- **Response shape:** amend the relevant pre-registration with a dated D-entry amendment recording what the smoke surfaced. The underlying code changes (extend a SKIP gate; canonicalize an N reduction) are the *consequence* of the amendment, not the amendment itself. The amendment is the methodology output; the code is the implementation of that output.
- **Enabling discipline:** the smoke gate fires before production. Pre-flight 5-step checklist + design-time source review (when the consumer is open-source, see §1.4) constitutes the smoke layer; the amendment lands in the gap between smoke and production fire.
- **Rung-4 instances:**
  - D0-22 amendment 1 (`e754a4b`) — extend SKIP gate from `dissipation_sign_violation` to `energy_drift` after pre-flight smoke surfaced KE(0) = 0.47 vs `KE_REST_THRESHOLD = 1e-10` (absolute threshold misfires at dam-break's KE = O(1000) scale).
  - D0-22 amendment 2 (`6926719`) — canonicalize rung-4c at N = 12 trajs after production fire timed out at 12/20 with ~10× optimistic per-traj cost; choose in-rung correction (Option B) over budget-funded uniformity (Option A) for methodology-consistency.
  - PH-BC plan-row mismatch (this v2.1) — design-pass source review of production rule set surfaced that PH-BC-001 is a mesh-FEM rule, not a particle-wall rule; v2.1 §3.1 corrects honestly.
  - Renderer-untouched framing walked back at writeup time (writeup §5.2 "Design-vs-writeup honest amendment") — pre-registration-vs-writeup drift treated as a third pattern-A instance within rung-4c.
- **Forward-applicability.** Future case studies inherit: when the smoke (or design-pass source review, or writeup-time honesty pass) surfaces a fact the plan didn't predict, the disciplined response is to land an amendment to the relevant pre-registration before shipping the artifact. The amendment is the load-bearing methodology output.

### §1.2 Pattern B — implementation-time hidden assumption → in-rung generalization (bilaterally exercised within rung-4c)

- **Trigger:** single-instance-vs-multi-instance divergence. Code that "happened to work" because there was only one instance of a use-case breaks when a second instance arrives. Distinct from pattern A in that the trigger is structural (multi-instance use is now exercised), not empirical (no plan prediction was contradicted).
- **Response shape:** generalize the code to remove the implicit assumption. The fix is a generalization of existing machinery, not a rung-specific patch. Future use cases inherit the generalization without further modification.
- **Enabling discipline:** multi-instance use surfaces assumptions. Add a second instance of an existing pattern (a second SARIF schema version landing in the same directory; a second standalone-conversion case beyond the conversion-bug-recovery one); the assumptions become visible.
- **Rung-4c instances (bilaterally exercised at two pipeline layers + round-2 artifact-level propagation):**
  - **Consumer-side renderer (Task 10) — layer 1.** Two within-rung-4c hidden assumptions surfaced:
    - Renderer's single-schema-version assumption (`render_cross_stack_table.py` globbed `*.sarif`, fail-loud-asserted `EXPECTED_SCHEMA_VERSION = "1.0"`). Once rung 4b landed v1.1 eps SARIFs in the same directory, any consumer needing only v1.0 would trip the assertion. Rung 4c's first cross-version-mixed-dir use surfaced the assumption; fix added `--include-glob` (default `*.sarif`) so callers can filter explicitly.
    - Renderer's hardcoded `D0-18` cell label (`f"SKIP (x{n}, D0-18)"`). Pre-rung-4c, D0-18 was the only SKIP path; rung 4c's D0-22 + amendment 1 SKIP paths made the hardcode wrong (everything labeled D0-18). Fix extracts the actual D-entry from `skip_reason` via regex; supports D0-08, D0-18, D0-22, and D0-22 (amendment 1) uniformly.
  - **Modal-side standalone-conversion (post-§9-review-gate fold-in round 1, `bc3bae9`) — layer 2.** Codex adversarial round 1 surfaced that the standalone-conversion entrypoint `convert_pkls_p1_segnn_dam2d` (added at D0-22 amendment 2) lacked run-completion-marker enforcement — operators could fire it against a timed-out subdir and get a generic PASS verdict. Same single-instance-vs-multi-instance shape as the renderer fixes: the entrypoint "happened to work" for D0-17 amendment 1's conversion-bug-recovery case (the only pre-rung-4c instance), and rung-4c's D0-22 amendment 2 timeout-salvage case re-used the same path without distinguishing. Fix: (i) `_persist_inference_manifest_to_rollout_subdir` helper called from all four LB rollout orchestrators (atomic write of gate-relevant fields on success / timeout / conversion-failure paths); (ii) `lagrangebench_convert_pkls_in_volume` gated with `allow_from_aborted_inference: bool = False` + `inference_run_status` classification (`from_completed_inference` / `from_aborted_inference` / `from_unknown_inference`); (iii) `backfill_rung4c_inference_manifests` Modal entrypoint, fired post-hoc — both rung-4c subdirs now carry explicit `_inference_manifest.json`. Gate refusal verified end-to-end at `outputs/rung4c_gate_refusal_verification.log`.
  - **SARIF artifact-level salvage-tag propagation (post-§9-review-gate fold-in round 2, `3037209`) — propagation, not a third layer.** Codex adversarial round 2 surfaced that the bc3bae9 gate tags conversion verdicts but the SARIFs themselves carried uniform provenance with no artifact-level signal of segnn-dam2d's salvage status. The fix propagates round 1's `inference_run_status` classification through to the SARIF run-property level (and renderer's optional "Inference run status" section), rather than introducing an independent pattern-B exercise. Distinct from a third pipeline-layer in that the *classification scheme* (round 1's contribution) is unchanged; round 2 threads it through the artifact-emission path. Fix: `emit_sarif.py::_read_inference_manifest_status` reads local-mirror manifest + threads `inference_run_status` through `_emit_for_stack`; `--stacks={all,tgv2d,dam2d}` flag for scoped re-emission; renderer's optional "Inference run status" section with honest-absence (`n/a (pre-salvage-tag-schema)`) for legacy stacks; 3 new renderer tests (all-present + all-absent + mixed). Schema version stays at v1.0 (additive optional field; renderer's strict-version assertion still passes for legacy SARIFs).
- **Bilateral-exercise observation.** Pattern B fires *twice within one rung* at *two distinct pipeline layers* (consumer-side renderer at Task 10; modal-side conversion at §9 fold-in round 1). Round 2's artifact-level salvage-tag is a propagation of round 1's classification scheme through to the SARIF emission path, not an independent third-layer exercise — the *classification scheme* contribution is round 1's; round 2 makes that contribution legible at the artifact level. The same shape — single-instance-vs-multi-instance assumption surfaces, generalize-not-patch response — fires uniformly across the two layers. This is methodologically non-obvious: a single rung exercising a pattern at multiple layers gives stronger evidence the pattern is real than a single rung exercising it once would. The bilateral elevation (with round-2's artifact-level propagation as the legibility complement) is itself the methodology-lesson contribution; the fixes are the implementations of the lesson.
- **Round-3 meta-instance: duplicate-implementations are themselves a single-instance-vs-multi-instance assumption.** Round 3 surfaced a pattern-B exercise at a different level: the *fact that the classification logic was duplicated* between modal_app.py (Modal-side runtime) and emit_sarif.py (local-side runtime) was itself a structural assumption — namely, "two duplicate implementations will stay in sync because they're simple enough." That assumption failed: both copies carried the same fail-open bug (corruption-collapses-to-legacy-absence), and Codex round-3 surfaced both findings against the same logical defect. The round-3 fix promoted the helpers to a shared `_harness/inference_manifest.py` module — the generalization response — which v2.1 §2.3 had explicitly pre-named as a deferred candidacy with trigger "third runtime context needs the classification." Round-3's third runtime context is the test code; the trigger fired exactly as v2.1 §2.3 predicted. This is pattern B operating at the *implementation-coordination level* rather than the *pipeline-layer level* — same shape, different abstraction layer. Worth flagging: anticipated-deferral-promotions ARE pattern-B exercises when their trigger condition surfaces, not lower-stakes refactorings; the deferral text named the trigger, the trigger fired, the promotion landed.
- **Forward-applicability.** Future case studies inherit: when implementation surfaces a single-instance-vs-multi-instance assumption (whether at the consumer-side renderer / artifact-level provenance / loader-contract / harness-emitter / modal-side orchestration layer), generalize rather than patch. Renderer is now multi-schema-version-mixed-dir-aware and multi-D-entry-aware; modal_app is now standalone-conversion-aborted-inference-aware; emit_sarif is now inference-status-provenance-aware. All three generalizations carry forward unchanged.

### §1.3 Pattern C — review-gate finding → triage-vs-novel classification → conditional absorption

- **Trigger:** re-discovery-vs-novel divergence. A review-gate (cross-review by an external agent or reviewer) surfaces findings that overlap with prior decisions; the question is whether each finding is genuinely novel or a re-discovery of a prior decision under a different lens.
- **Response shape:** triage each finding into one of four cells (see four-cell triage framework below); absorb in-rung only the novel-in-scope findings; defer the others with explicit citations to the discipline-markers.
- **Enabling discipline:** **verbatim memorialization** of decision-markers in handoff docs. Without preserving exact text — verbatim quotes of decision-markers, not paraphrases — triage degrades to guesswork at re-fire time. The discipline-marker is what makes cell-1 triage (re-discovery → defer) reliable.

#### Pattern C four-cell triage framework

(Preserved verbatim per pattern C's own self-application requirement: this framework's load-bearing-ness is the precise wording, and modifying it implicitly re-litigates prior decisions. Future readers should preserve this table verbatim as well.)

| Finding type | Response |
|---|---|
| Re-discovery of prior finding under prior scope | Defer to prior decision; cite the discipline-marker |
| Novel finding under current scope | In-rung absorption (pattern A or B as appropriate) |
| Novel finding outside current scope | Forward-flag to future rung |
| Genuinely new framing of prior finding | Re-examine prior decision with new information |

**Cell 4 is the only cell that legitimately reopens prior decisions, and the bar for cell 4 is "genuinely new framing or information surfaced," not "re-asked under a different lens."** That bar matters because cell 4 is also where blind absorption is most tempting: the new framing makes it *feel* novel, but the underlying decision-logic might be unchanged. Disciplined cell 4 invocation requires articulating what's new beyond framing — without that, default to cell 1.

- **Rung-4c instances (§9 review-gate fold-in rounds 1+2):**
  - Round 1 was Codex adversarial review of `outputs/rung4c_segnn_dam2d_run.log`. Two findings:
    - Recommendation 1 ("re-fire SEGNN-DAM2D at N=20 with adequate timeout") — **cell 1 (re-discovery under prior scope).** D0-22 amendment 2 had pre-registered Option A vs Option B with the user loop-in transparency; Option B (ship-at-N=12 with smoke-discovered-drift discipline) was canonical. The "N=12 canonical / no SEGNN re-fire" memory acted as the discipline-marker; deferred-with-citation to writeup §6 item 8 + DECISIONS.md D0-22 amendment 2 + bc3bae9 commit-message reasoning chain.
    - Recommendation 2 ("standalone-conversion entrypoint should refuse-by-default on aborted inference") — **cell 2 (novel-in-scope).** No prior decision; in-rung absorption as the bc3bae9 gate.
  - Round 2 was Codex adversarial review of post-bc3bae9 working-tree (untracked `outputs/*.log` files). Findings:
    - "Re-fire SEGNN at N=20" re-surfaced — **cell 1.** Same discipline-marker; same defer-with-citation. Round-2 demonstrated the discipline-marker's load-bearingness: when working-tree scope narrows, the same Codex agent re-surfaces prior-round findings; without the verbatim "N=12 canonical / no SEGNN re-fire" memory, round-2 triage would have wasted time re-litigating.
    - Untracked `outputs/*.log` automation-pickup risk — **cell 2 (novel-in-scope, A-narrow).** No prior decision on logs vs deliverables convention; in-rung absorption as `/outputs/` gitignore (anchored leading slash matters; un-anchored pattern would have ignored the lagrangebench SARIF deliverable too — caught + fixed pre-commit).
    - SARIF artifact-level salvage-tag gap — **cell 2 (novel-in-scope, B-narrow).** No prior decision on artifact-level provenance for salvage cases; in-rung absorption as the optional `inference_run_status` field. Two larger-scope alternatives proposed and explicitly deferred:
      - B-medium (required field + schema bump 1.0 → 1.0.1 + rung-4a/4b backfill + SCHEMA.md amendment) — deferred to v2.1 §2.1 below.
      - B-wide (D0-23 entry + full SCHEMA.md amendment for inference-status provenance as a first-class schema concept) — deferred to v2.1 §2.2 below.
  - Round 3 (working-tree review of rung-4c branch at sha `9952170` against base `409bee0`; 21 files / 6328 insertions). Two HIGH findings, both **cell 2 (novel-in-scope)** — neither prior round addressed manifest-corruption as a distinct case:
    - Finding 1: **modal_app.py gate fails open on corrupt/malformed manifests** (`_classify_inference_run_status` collapsed JSON decode failures into `from_unknown_inference`, which the gate warn-allowed). Absorbed by introducing the `manifest_invalid` status (distinct from legacy absence) and a third gate branch that refuses unconditionally with no override flag. The pre-round-3 behavior was a fail-open in the exact scenario the round-1 gate was designed to prevent (silent reuse of timed-out / corrupted artifacts).
    - Finding 2: **SARIF emission silently omits salvage status on missing/corrupt dam2d manifests** (`_read_inference_manifest_status` returned `None` on both legacy absence and corruption; `_emit_for_stack` then omitted the optional `inference_run_status` field; renderer hid the section). Absorbed by adding `manifest_required: bool = False` to `_emit_for_stack`, passing `True` for the dam2d emission paths, and raising `FileNotFoundError` / `ManifestInvalidError` on missing-or-corrupt for required stacks.
    - Helper promotion (round-3 § §2.3 deferral resolution): the duplicate `_classify_inference_run_status` (modal_app) + `_read_inference_manifest_status` (emit_sarif) implementations both carried the same fail-open bug, demonstrating the failure mode v2.1 §2.3 had named for the deferral's trigger. Promotion to shared `_harness/inference_manifest.py` landed in the same commit as findings 1 + 2.
  - **Cell-4-bar self-application during v2.1 drafting (mid-task, 2026-05-10).** Worth recording: the v2.1 draft itself surfaced a potentially-novel "trilateral / three layers" framing of pattern B that on triage resolved to cell 1 (re-discovery of the canonical bilateral-plus-round-2-propagation framing). The cell-4 bar held: the new framing didn't introduce new information beyond "one might count differently," and the underlying decision-logic was unchanged. Pattern C self-applied to its own documentation's construction; preserving this observation here closes the recursion-loop.

- **Forward-applicability.** Future case studies inherit pattern C verbatim. The four-cell framework + the discipline of verbatim memorialization carry forward unchanged. Case study 02's own cross-review cycle will exercise pattern C from a different framework integration's perspective, the same way it'll exercise A and B for its own loader-contract and rendering-assumption surfaces. The triad is bilateral-validation-ready (see §1.5 below).

### §1.4 Cross-review-gate as parallel discipline (third leg of the review triple)

The rung-4 series demonstrates that review-discipline is not one mechanism but a triple of mechanisms operating at different stages:

| Layer | Cost | What it catches | Where in rung-4 it fired |
|---|---|---|---|
| Smoke-review (pre-flight 5-step) | ~30 min CPU + writedown | Empirical-vs-prediction drift; cost overruns; numerical misfires that look reasonable | D0-22 amendment 1 (energy_drift KE-rest threshold misfire); D0-22 amendment 2 (timeout cost overrun) |
| Source-review (pre-compute, $0 Modal) | ~hours of source reading | Plan-vs-actual-rule mismatches; loader-contract assumptions; subseq-length math errors | Rung-4b math correction; rung-4b figure-sweep failure; rung-4c PH-BC plan-row mismatch + dam2d catalogue misclassification |
| Cross-review (post-shipping, external agent) | ~hours of agent-driven adversarial reading | Implementation-time hidden assumptions that survived smoke + source review; novel framings of prior decisions | Rung-4c §9 fold-in round 1 (standalone-conversion gate); round 2 (SARIF salvage-tag + outputs/ gitignore) |

The three layers are *complementary*, not redundant. Smoke catches empirical-vs-prediction; source catches consumer-API-contract; cross catches what survived both. Each layer's fixed costs (writedown infrastructure for smoke; source-reading habit for source-review; cross-review-gate prescription in plan §9 for cross-review) are paid once per case study and amortize across all rungs in the series.

The cross-review layer is the youngest of the three in the rung-4 series's procedural toolkit; rung-4c is the first rung to exercise it as a load-bearing gate (§9 review-gate fold-in rounds 1+2). The triple as a whole — smoke + source + cross — is the rung-4 series's review-discipline deliverable. Future case studies inherit the triple and the exercises that surface its instances.

### §1.5 Forward-applicability to case study 02 (PhysicsNeMo MGN) — bilateral validation

Each pattern's forward-applicability stanza names what case study 02 inherits. In aggregate, case study 02 is the bilateral-validation surface for the triad: a different framework integration (PyTorch/DGL Eulerian mesh-based MGN, vs LB JAX/Haiku Lagrangian particle SPH) exercising A + B + C in distinct concrete instances. The triad's robustness as a methodology contribution depends on it firing in case study 02 with the same shape as it fired in case study 01.

Concrete inheritance items, by pattern:

- **Pattern A.** MGN materializer's pre-flight assertions section gets a sibling "MGN loader-contract assertions" alongside the existing "LB loader-contract assertions." Substrate-class taxonomy gains entries as MGN's substrates are empirically probed (vortex-shedding, Ahmed Body — both candidate "conservative" anchors per writeup §6 item 1's bilateral D0-18 forward-flag). The amendment-layered DECISIONS.md pattern carries forward unchanged.
- **Pattern B.** MGN integration will surface its own single-instance-vs-multi-instance assumptions at consumer-side (renderer; emit_sarif) and modal-side (orchestrator; standalone-conversion). Generalize-not-patch response carries forward unchanged. The harness machinery is now bilaterally-validation-ready: rung-4c's renderer + modal_app + emit_sarif generalizations are MGN-agnostic by design.
- **Pattern C.** Case study 02's cross-review gate (§9-equivalent) is prescribed in the same plan-§9 language; the four-cell framework carries forward verbatim. Case study 02 is the first opportunity to exercise pattern C's robustness against a *different* set of framework conventions; the triad's claim that pattern C generalizes beyond rung-4c stands or falls on case study 02's exercise.

The integrating README at `methodology/README.md` (rung-4 series's closure deliverable, implicit Task 13) elevates the triad to a cross-rung output rather than a within-rung-4c observation. Case study 02 reads the integrating README as the named-event durable artifact for the rung-4 series's methodology contributions.

---

## §2. Deferrals — items v2.1 records but does not implement

Each item is named explicitly so that v2.1 readers see both what was absorbed in-rung and what was triaged for future absorption. Cell-3 (novel-out-of-scope) findings from §9 fold-in round 2 land here as forward-flags rather than dropped without record.

### §2.1 SARIF schema bump 1.0 → 1.0.1 (B-medium)

**What.** Promote `inference_run_status` from optional run-property (current v1.0 additive field) to a required run-level property; schema-version bump 1.0 → 1.0.1; rung-4a/4b SARIFs backfilled with classification (likely uniformly `from_completed_inference` since no salvage scenarios occurred); SCHEMA.md amendment documenting the new required field + the n/a-on-pre-1.0.1 honest-absence convention.

**Why deferred.** Optional-field absorption (current state) preserves backward compatibility (legacy SARIFs without the field render with `n/a (pre-salvage-tag-schema)` rather than failing the schema-version assertion). Required-field absorption forces a re-emission pass on rung-4a/4b artifacts that don't have salvage scenarios to record — tautological work for those rungs. The optional-field state is the cheapest absorption that preserves the artifact-level provenance signal where it matters (rung-4c salvage case); the required-field promotion is justified only when a future rung's downstream consumer (case study 02; integrating README; CI gate) relies on the field's presence as a hard contract.

**Trigger for promotion.** Whenever a downstream consumer's contract becomes "every rung's SARIFs must declare inference_run_status," promote to required. Until then, optional-field state is canonical.

### §2.2 D0-23 entry — inference-status provenance as first-class schema concept (B-wide)

**What.** A new D-entry pre-registering `inference_run_status` (and adjacent provenance fields like `aborted_at_step`, `inference_returncode`) as first-class schema concepts in `_harness/sarif_emitter.py`'s contract; D0-23 sits alongside D0-19 (harness SARIF result schema) and references D0-19 §3.4 for the row-level identity contract. SCHEMA.md amendment documenting the run-level inference-status section parallel to D0-19's result-level documentation.

**Why deferred.** D0-23 promotes the salvage-tag from a rung-specific addition (current state, optional field threaded through emit_sarif) to a first-class schema concern. Promotion is justified only if a downstream consumer (case study 02; integrating README; CI gate) treats inference-status provenance as load-bearing. Currently the rung-4c writeup §6 item 8 carries the textual transparency, the `inference_run_status` optional field carries the artifact-level transparency, and the bc3bae9 gate carries the runtime refuse-by-default — three layers of inference-status discipline without yet promoting it to first-class schema status. Promotion is post-rung-4 work.

**Trigger for promotion.** Whenever case study 02's first salvage-or-failed-inference scenario fires — i.e., when the second instance of the inference-status provenance pattern arrives, paralleling pattern B's "single-instance-vs-multi-instance triggers generalization" rule.

### §2.3 `_classify_inference_run_status` shared helper promotion — **RESOLVED at round 3 (2026-05-11)**

**What (original deferral, preserved verbatim).** The `_classify_inference_run_status` function is currently duplicated between `01-lagrangebench/modal_app.py` (Modal-container side; reads manifest fields written during inference) and `01-lagrangebench/emit_sarif.py` (local side; reads the same manifest fields from the local-mirror SARIF emission flow). Promote to a shared module, e.g. `external_validation/_rollout_anchors/_harness/inference_manifest.py`, that both runtime contexts can import.

**Why originally deferred.** The duplication was *intentional* for cross-runtime reasons — modal_app.py runs inside Modal containers (no access to local repo paths) and emit_sarif.py runs locally (no access to Modal volume state). A shared module that both can import is cleaner only if both runtime contexts can resolve the same import path; under current Modal image-build conventions, that required either (i) bundling `_harness/inference_manifest.py` into the Modal image as part of the LB-runtime install, or (ii) inlining the helper at both sites and treating duplication as load-bearing for runtime independence.

**Trigger that fired.** Codex round 3 surfaced two HIGH findings that both stemmed from the duplicate implementations carrying the same fail-open bug (corruption-collapses-to-legacy-absence). Test code became the third runtime context: comprehensive unit tests on the classifier required a single import target, and writing the round-3 fix against duplicate implementations would have re-introduced the drift risk. The "third runtime context" trigger fired exactly as named.

**Resolution path (committed at the same sha as this absorption).** Helpers promoted to `external_validation/_rollout_anchors/_harness/inference_manifest.py`. modal_app.py mounts the file into the rollout image via `_HARNESS_T7_MODULES` (the existing `add_local_file` pattern); its `_classify_inference_run_status` and `_persist_inference_manifest_to_rollout_subdir` now delegate via a dual-path import lookup (local `external_validation` package vs. container-side bare-name import). emit_sarif.py imports the shared helper directly. 21 new tests at `_harness/tests/test_inference_manifest.py` cover the classifier truth table including the new `manifest_invalid` status; full test suite passes at 796 / 1 skipped (775 baseline + 21 new).

### §2.4 Rung-4a/4b SARIF backfill of `inference_run_status` (companion to §2.1)

**What.** Re-emit rung-4a + rung-4b SARIFs at a future sha to add the optional `inference_run_status = from_completed_inference` property uniformly (no rung-4a/4b run was a salvage scenario; all classify trivially as completed). Companion to §2.1's required-field promotion; without §2.1's promotion, the rung-4a/4b backfill is purely tautological and adds no information beyond the renderer's `n/a (pre-salvage-tag-schema)` honest-absence already provides.

**Why deferred.** Tautological for rung-4a/4b (no salvage scenarios). Worth doing only as a companion to §2.1's promotion, when the required-field schema makes the absence non-tautological.

### §2.5 Cross-rung N=12 vs N=20 framing in §3.1 P1 + §5.3 cover letter

**What.** The current §0 §3.1 P1 row update carries the N=12 reduction as a parenthetical (D0-22 amendment 2). The current §0 §5.3 cover-letter sentence carries it explicitly. Both treatments may need sharpening if Stuttgart/Munich reviewers read the N-asymmetry as a methodological gap rather than as a methodology-consistency-honoring choice.

**Why deferred.** The current treatment is honest-by-default (the asymmetry is named, the rationale is named); whether *additional* framing is needed is a function of cross-review-gate findings (ChatGPT cross-review, ≥3 rounds, prescribed in plan §9 and not yet exercised at the v2.1 level — the §9 fold-in rounds 1+2 were Codex cross-review of *artifacts*, not of v2.1 *prose*). v2.1's deferral here records the question; a v2.1.1 amendment after ChatGPT cross-review of the v2.1 document itself can sharpen if the review surfaces a concrete framing concern.

---

## §3. Changelog

### v2.1 round-3 absorption — 2026-05-11

§9 fold-in round 3 (Codex adversarial review of rung-4c branch at sha `9952170`, target diff vs `409bee0`; 21 files / 6328 insertions). Two HIGH findings, both **pattern-C cell 2 (novel-in-scope)**:

- **Finding 1: modal_app.py gate fails open on corrupt/malformed manifests.** Pre-round-3, `_classify_inference_run_status` collapsed JSON decode failures and missing required fields into `from_unknown_inference` (legacy-absent), which the gate warn-allowed. A corrupted or stale `_inference_manifest.json` therefore let an aborted-inference subdir be converted without `allow_from_aborted_inference=True` — fail-opening the exact discipline round-1's gate was designed to enforce. Absorbed by introducing `STATUS_MANIFEST_INVALID` (parseable but missing required classification fields; or unparsable bytes; or non-dict root), distinct from `STATUS_FROM_UNKNOWN_INFERENCE`, and adding a gate branch that refuses `STATUS_MANIFEST_INVALID` unconditionally with no override flag (corruption is structural, not policy).
- **Finding 2: SARIF emission silently omits salvage status for missing/corrupt dam2d manifests.** Pre-round-3, `_read_inference_manifest_status` returned `None` on both legacy absence and corruption; `_emit_for_stack` then omitted the optional `inference_run_status` field; the renderer hid the section. For rung-4c dam2d (post-fold-in stack), a stale local mirror or corrupt backfilled manifest could emit a valid-looking SARIF with no aborted-inference tag — defeating round 2's artifact-level transparency. Absorbed by adding `manifest_required: bool = False` to `_emit_for_stack` (default for legacy rung-4a/4b tgv2d stacks), passing `True` for the dam2d emission paths, and raising `FileNotFoundError` / `ManifestInvalidError` on missing-or-corrupt for required stacks.
- **Helper promotion (§2.3 deferral resolution).** Both findings stemmed from duplicate classifier implementations carrying the same fail-open bug — the failure mode v2.1 §2.3 had named as the deferral's promotion trigger. Classification + persistence helpers promoted to shared `external_validation/_rollout_anchors/_harness/inference_manifest.py`. modal_app.py mounts the file into the rollout image via `_HARNESS_T7_MODULES`; its `_classify_inference_run_status` + `_persist_inference_manifest_to_rollout_subdir` now delegate via a dual-path import lookup (local `external_validation` package vs. container-side bare-name import after sys.path insertion). emit_sarif.py imports the shared helper directly. v2.1 §2.3 marked RESOLVED.
- **Tests.** 21 new tests at `_harness/tests/test_inference_manifest.py` covering: (i) classifier truth table — completed / aborted (returncode≠0) / aborted (returncode=0 + aborted_at_step) / unknown-legacy / invalid-corrupt-json / invalid-missing-fields / invalid-non-dict-root / pure-read-no-side-effects; (ii) `read_inference_manifest_status` — completed → status string / aborted → status string / missing-not-required → None / missing-required → FileNotFoundError / invalid → ManifestInvalidError (regardless of required); (iii) `persist_inference_manifest_to_rollout_subdir` — writes gated subset / returns None when subdir missing / returns None when subdir is None / persist→classify roundtrip / atomic-no-partial-file; (iv) status-constants-distinct + manifest-filename-dotfile-convention.
- **Full test suite:** 796 passed, 1 skipped (775 baseline + 21 new). No regressions.

### v2.1 — 2026-05-10

**§0 substantive edits (diff from v2):**

- §3.1 P1 row: drop "PH-BC (wall)" (no extant SPH-particle wall rule in physics-lint v1.0); replace with substrate-class-extension framing per D0-22 + amendments 1, 2; expand architecture column to dual-stack `GNS + SEGNN`.
- §3.1 P3 row: struck (absorbed by dual-stack P1).
- §3.2 step 6 dam-break template: drop PH-BC-001 row; add inference-run-status optional field per §9 fold-in round 2.
- §5.3 cover-letter dam-break sentence: drop "PH-BC"; pick up substrate-class-extension framing + N-per-rung explicit citation + renderer-generalizations cross-reference.
- §6 risks register: add three new entries — plan-vs-actual-rule mismatch (PH-BC); plan-vs-actual N-per-rung mismatch (timeout cost overrun); standalone-conversion silent-reuse risk (§9 fold-in round 1).

**§1 methodology section (new, paired-pattern triad as central organizing structure):**

- §1.1 Pattern A — smoke-discovered drift → in-rung amendment. Four within-rung-4c instances catalogued; forward-applicability to case study 02 named.
- §1.2 Pattern B — implementation-time hidden assumption → in-rung generalization. **Bilaterally exercised within rung-4c at two pipeline layers + round-2 artifact-level provenance propagation** (consumer-side renderer at Task 10; modal-side standalone-conversion at §9 fold-in round 1; round 2's SARIF salvage-tag is propagation of round 1's classification scheme into the artifact-emission path, not an independent third-layer exercise). The bilateral elevation is itself the methodology-lesson contribution.
- §1.3 Pattern C — review-gate finding → triage-vs-novel classification → conditional absorption. **Pattern C is named at session-close 2026-05-10**, post-§9-fold-in; the four-cell triage framework is preserved verbatim per pattern C's own self-application requirement. Round-1 + round-2 instances catalogued by cell.
- §1.4 Cross-review-gate as parallel discipline (third leg of the smoke + source + cross review triple). Each layer's cost / catches / where-in-rung-4-it-fired tabulated.
- §1.5 Forward-applicability to case study 02 — bilateral validation. The triad's claim that A + B + C generalize beyond rung-4c stands or falls on case study 02's exercise.

**§2 deferrals (new):**

- §2.1 SARIF schema bump 1.0 → 1.0.1 (B-medium); §2.2 D0-23 entry (B-wide); §2.3 shared helper promotion; §2.4 rung-4a/4b backfill (companion to §2.1); §2.5 cross-rung N framing sharpening (post-ChatGPT-cross-review, if surfaced).

### Source-review-correction acknowledgment

This v2.1 amendment is the third instance of the source-review-catches-issue-before-compute pattern in the rung-4 series, alongside rung-4b's first-pass math correction (`TRAIN_PUSHFORWARD_UNROLLS_LAST` derivation) and first-pass figure-sweep failure (valid.h5 hardcoded subseq_length). All three caught at $0 Modal cost. The pattern paralleling rung-4b T7 design §14.6's acknowledgment shape: a source-review pre-flight pass between design and execution catches issues that brainstorm-only and execution-only review miss; the cost is hours of source reading, the saving is multiple GPU runs and methodology errors that would otherwise land in writeups (or worse, in the cover letter).

### Cross-review-gate-absorption acknowledgment

This v2.1 amendment is also the first instance of pattern C (review-gate finding → triage → conditional absorption) being named explicitly in the rung-4 series. The pattern was implicitly exercised earlier (rung-4b T7's amendment 2 §14.6 was a self-driven re-review with similar absorption shape); rung-4c's §9 fold-in rounds 1+2 made the pattern explicit by exercising it against an external agent (Codex) twice consecutively, surfacing the discipline-marker's load-bearing role in cell-1 triage. v2.1's central methodology section structures around the A + B + C triad with C as the third-named member; integrating README composition (implicit Task 13) elevates the triad to a cross-rung output.

---

## §4. Definition-of-done for v2.1

This v2.1 amendment is **done** when all of the following are true:

1. `physics-lint-validation-plan-v2.1.md` is committed at this path; v2 stays frozen.
2. §0 substantive edits are individually citeable (each edit names the source D-entry / commit / writeup section it absorbs).
3. §1 methodology section structures around the A + B + C triad with the four-cell framework preserved verbatim.
4. §2 deferrals are named explicitly (no silent drops); each names trigger-for-promotion.
5. §3 changelog includes both the source-review-correction acknowledgment (pattern A) and the cross-review-gate-absorption acknowledgment (pattern C).
6. PR review of the rung-4c branch surfaces no v2.1 framing changes that would force a v2.1.1 amendment. (If it does, v2.1.1 lands as a separate companion doc, parallel to v2 → v2.1.)
7. Integrating README composition (implicit Task 13) reads v2.1 as its primary methodology source for the rung-4 series's cross-rung contributions.
