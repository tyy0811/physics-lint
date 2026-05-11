# `_rollout_anchors/methodology/`

Methodology trail, design docs, and decision history for the
rollout-anchors validation work — the project of demonstrating
physics-lint's rules against real ML model rollouts (LagrangeBench /
SEGNN, GNS; PhysicsNeMo / MeshGraphNet; ...).

This README is the **rung-4 series closure deliverable**: it composes
the rung-4a, rung-4b, and rung-4c writeups into a single cross-rung
synthesis, lifts the methodology contributions to a cross-rung output,
and frames case study 02 as the next falsification surface for the
contributions that emerged within rung-4. For per-rung detail, follow
the links into `docs/`; for the operational methodology trail,
`DECISIONS.md` is the single source of truth.

---

## 1. Headline (rung-4 series, closed)

The rung-4 series validates that physics-lint's harness machinery runs
**uniform SARIF rule schemas across architecturally-distinct and
substrate-class-distinct neural-physics integrations**, with no
consumer-side accommodation beyond two renderer generalizations
(multi-schema-version filtering; D-entry extraction).

- **Rung 4a** ([writeup](docs/2026-05-04-rung-4a-cross-stack-conservation-table.md))
  ran the same conservation rule schema (PH-CON, v1.0) unmodified
  across SEGNN-TGV2D + GNS-TGV2D rollouts. Both stacks emit byte-
  identical rows (mass = 0.0; energy_drift SKIP via D0-18 dissipative-
  system gate; dissipation_sign_violation = 0.0). The
  methodology-evolution machinery (D0-18's skip-with-reason) is
  exercised end-to-end against real upstream output.
- **Rung 4b** ([writeup](docs/2026-05-07-rung-4b-equivariance-table.md))
  ran the same equivariance rule schema (PH-SYM, v1.1) across the
  same two stacks at single-step inference, surfacing the architectural
  contrast as the load-bearing cross-stack signature: SEGNN's E(2)-
  equivariance is exact-by-construction (ε at the float32 noise floor
  ~2.3e-7); GNS's is approximate-by-training (bimodal APPROXIMATE +
  FAIL bands ~3.6e-4 to 3.5e-2). The same machinery emits different
  values when the stacks behave differently.
- **Rung 4c** ([writeup](docs/2026-05-07-rung-4c-substrate-class-extension-table.md))
  extended the harness's substrate-detection layer to a second
  LagrangeBench substrate class (open-driven-dissipative) via
  dam-break-2D, with D0-22 + amendments 1, 2 routing the substrate-
  class dispatch. Both stacks (SEGNN-DAM2D + GNS-DAM2D, N=12 trajs
  per amendment 2's cost-driven reduction) emit byte-identical
  structural rows. The same machinery emits identical values when the
  substrate determines the verdict.

Together rung-4b + rung-4c bilaterally exercise the schema-uniformity
claim: the harness machinery handles **both** shapes of cross-stack
signature — different-values-uniform-schema (rung 4b) and identical-
values-uniform-schema (rung 4c) — without consumer-side accommodation.

The current **plan-amendment view** is
[`docs/physics-lint-validation-plan-v2.1.md`](docs/physics-lint-validation-plan-v2.1.md);
the original frozen plan is `docs/physics-lint-validation-plan-v2.md`.
v2.1 carries the post-rung-4c methodology-current view (including
the §9 cross-review fold-in rounds 1 + 2 + 3 + round-prose-1
absorptions); per-rung writeups are sha-bound to their respective
rung-snapshot views.

---

## 2. Cross-rung methodology contributions

Five contributions emerge from the rung-4 series and are intended to
carry forward into case study 02 (PhysicsNeMo MGN) and subsequent
integrations. Two are validated within rung-4 alone; three are
generalization claims that case study 02 will test (see §3 below).

### 2.1 Bilateral schema-uniformity composite (validated within rung-4)

The schema-uniformity claim is stronger when rung-4b and rung-4c are
read together than either alone. The two rungs exercise opposite
shapes of cross-stack signature through the same machinery:

- **Rung 4b — stacks behave differently, schema is uniform.**
  Architectural-class-bound contrast: SEGNN's float32-floor vs GNS's
  bimodal APPROXIMATE+FAIL split is what the rule schema *makes
  visible*. The schema is the substrate for the contrast.
- **Rung 4c — stacks behave identically, schema is uniform.**
  Substrate-property-bound invariance: both stacks emit `mass=0.0`,
  `energy_drift SKIP D0-22a1`, `dissipation_sign_violation SKIP D0-22`.
  The schema is the substrate for the invariance.

A schema that handles only one signature shape (only contrasts, or
only invariances) would be a less powerful machinery. Rung-4 has
evidence for both. **Open question for case study 02**: does a
third signature shape exist (e.g., stacks behave differently *and*
the substrate determines part of the verdict) that the current
schema cannot accommodate? Surface that, and the schema needs a
generalization; absent that surfacing, the bilateral composite is
the load-bearing cross-rung output.

### 2.2 Paired-pattern triad A + B + C (within-rung evidence; cross-rung generalization is open)

Three patterns for methodology-lesson absorption emerged in rung-4c
and were named explicitly in plan v2.1 §1 (with necessary conditions
and falsification rules added at round-prose-1). Each names a
divergence type + a response shape + an enabling discipline. Detailed
catalogue + necessary conditions are in
[plan v2.1 §1.1–§1.3](docs/physics-lint-validation-plan-v2.1.md#1-methodology--paired-pattern-triad-as-central-organizing-structure);
short form below.

#### Pattern A — smoke-discovered drift → in-rung amendment

- **Trigger:** empirical-vs-prediction divergence (smoke / source
  review / writeup-honesty pass surfaces a fact the pre-registration
  didn't predict).
- **Response:** amend the relevant pre-registration with a dated
  D-entry; the code changes are the consequence, not the lesson.
- **Enabling discipline:** smoke / source / writeup-honesty layer
  fires before production fire.
- **Rung-4c instances:** D0-22 amendments 1 + 2; PH-BC plan-row
  mismatch (caught at design-pass source review); renderer-untouched
  framing walked back at writeup-time.

#### Pattern B — implementation-time hidden assumption → in-rung generalization

- **Trigger:** single-instance-vs-multi-instance divergence; one
  piece of code (single artifact) worked for one use-case, breaks
  when a second use-case arrives.
- **Necessary conditions** (added at v2.1 round-prose-1):
  1. Single artifact, multi-instance use (one piece of code, multiple
     use-cases — *not* multiple copies of the same code).
  2. Hidden assumption about the first use-case that wasn't named as
     a contract.
  3. Generalization response (remove the assumption, not special-case
     the new use-case).
- **Rung-4c instances (canonical):** renderer's single-schema-version
  + hardcoded-D-entry assumptions surfaced at Task 10 (multi-version
  mixed dir; multiple SKIP D-entries); modal-side standalone-conversion
  gate at §9 fold-in round 1; round 2's artifact-level salvage-tag is
  propagation of round 1's classification scheme into SARIF emission
  (not an independent third instance).
- **Not pattern B:** *duplicate-implementation drift* (round 3's helper
  promotion to `_harness/inference_manifest.py`) is an adjacent-but-
  distinct discipline — two copies of the same logic sharing a bug,
  consolidated to a shared module. The disciplines look similar but
  the necessary conditions differ (round 3's case is two artifacts
  that should have been one, not one artifact handling two use-cases).
  Counting helper-promotion refactors as pattern B would broaden the
  pattern enough to absorb any duplication-removal, weakening its
  discriminating power.

#### Pattern C — review-gate finding → triage-vs-novel classification → conditional absorption

- **Trigger:** cross-review surfaces findings that may overlap with
  prior decisions; the question is novel vs re-discovery.
- **Response:** triage each finding into one of four cells; absorb
  only novel-in-scope findings; defer the others with citations to
  the discipline-markers.
- **Enabling discipline:** verbatim memorialization of decision-
  markers in handoff docs; without preserving exact text, triage
  degrades to guesswork.
- **Four-cell framework** (preserved verbatim per pattern C's own
  self-application; see [plan v2.1 §1.3](docs/physics-lint-validation-plan-v2.1.md#13-pattern-c--review-gate-finding--triage-vs-novel-classification--conditional-absorption)
  for the table + falsification conditions): re-discovery (defer),
  novel-in-scope (absorb), novel-out-of-scope (forward-flag),
  genuinely-new-framing (re-examine).
- **Falsification conditions** (added at round-prose-1): finding-fits-
  no-cell; classification disagreement; cell-1-dominance-without-
  cell-4-reopeners; cell-4-invocation-without-articulation. The
  framework's cells are preserved verbatim *within their effective
  life*; amendment is permitted when failure modes surface.
- **Rung-4c instances:** §9 fold-in round 1 (cell-1 + cell-2);
  round 2 (cell-1 + 2× cell-2); round 3 (2× cell-2 + helper
  promotion); round-prose-1 (4× cell-2 + the first cell-4 invocation
  in the rung-4 series — the unfalsifiability critique against the
  framework itself).

### 2.3 Review-discipline triple — smoke + source + cross (validated within rung-4 as procedural toolkit)

Three review layers operate at different stages of the pipeline:

| Layer | Cost | What it catches in rung-4 | First load-bearing exercise |
|---|---|---|---|
| Smoke-review (pre-flight 5-step) | ~30 min CPU + writedown | Empirical-vs-prediction drift; cost overruns; numerical misfires | D0-22 amendment 1 (energy_drift threshold misfire) |
| Source-review (pre-compute, $0 Modal) | ~hours of source-reading | Plan-vs-actual-rule mismatches; loader-contract assumptions; subseq-length math errors | Rung-4b T7 math correction + figure-sweep failure; rung-4c PH-BC + dam2d misclassification |
| Cross-review (post-shipping, external agent) | ~hours of agent-driven adversarial reading | Implementation-time hidden assumptions that survived smoke + source; novel framings | Rung-4c §9 fold-in rounds 1 + 2 + 3 + round-prose-1 |

The three layers are **complementary**, not redundant — each catches
what the others missed. Each layer's fixed cost amortizes across all
rungs in the case study.

**On the smoke/source/cross ↔ A/B/C diagonal correspondence.** The
"what it catches" column lines up suggestively with the A+B+C triad
in rung-4c instances (smoke ~ pattern A drift; source ~ pattern B
hidden assumption; cross ~ pattern C novel framing). Per plan v2.1
§1.4, this is **observed correspondence, not load-bearing structural
claim** — a given review type *can* surface any pattern. The triple
and the triad are orthogonal in principle; their rung-4c diagonal
alignment is empirical regularity rather than theorem.

### 2.4 "Classify when you exercise" empirical-classification principle (trilateral)

Substrate properties get verdicts only after empirical probing, not
preemptively from dataset-name conventions or paper claims. Trilateral
across rung-4:

- Rung 4b: PH-SYM-003 PBC-square-SO(2) substrate-incompatibility
  surfaced *during exercise*, captured as a SKIP-with-reason at the
  schema layer.
- Rung 4c: dam2d empirically reclassified from "dissipative" to
  "open-driven-dissipative" after pre-flight smoke confirmed the
  rise-then-fall KE shape (KE(0)=0.47, peak ~1100, mid-trajectory
  peak) — the catalogue's pre-D0-22 label was a paper-derived guess.
- Rung 4c forward-flag: remaining LB datasets (`rpf2d`, `ldc2d`,
  `rpf3d`, `ldc3d`, `tgv3d`) retain their pre-D0-22 labels and walk
  into known-misclassifications at first exercise; the empirical
  probe is each rung's first move.

### 2.5 "Source-review-catches-issue-before-compute" pattern (trilateral within rung-4)

Three rung-4 instances caught at $0 Modal cost:

1. Rung-4b T7 design §14.6 — `TRAIN_PUSHFORWARD_UNROLLS_LAST`
   derivation math error caught at LB source review.
2. Rung-4b T7 design §14.6 — figure-sweep latent-failure
   (`valid.h5` hardcoded subseq_length) caught at the same review.
3. Rung-4c plan v2.1 §3.1 P1 — "PH-BC (wall)" assumed an SPH-particle
   wall rule that doesn't exist in physics-lint v1.0; caught at
   design-pass source review of the production rule set.

The pattern's cost-benefit ratio is decisive: hours of source-reading
buys preventing multi-GPU compute cycles + methodology errors landing
in writeups. When the consumer's loader / rule set is open-source,
source review pre-flight is essentially free relative to its alternative.

---

## 3. Case study 02 as the next falsification surface

Per [plan v2.1 §1.5](docs/physics-lint-validation-plan-v2.1.md#15-case-study-02-physicsnemo-mgn-as-a-falsification-surface--readiness-not-validation),
**case study 02 (PhysicsNeMo MGN) is a pending falsification surface
for the cross-rung generalization claims above, not an already-
exercised validation**. The rung-4 series validates the contributions
*within rung-4*; whether A + B + C and the schema-uniformity composite
generalize beyond rung-4 is open and depends on case study 02's
empirical exercise.

This README's claims about case study 02 are **predictions with
explicit confirmation + disconfirmation criteria**, not forecasts-as-
facts. Readers writing or reading case study 02 should treat the
contributions in §2 as hypotheses to be tested, not a framework to be
applied uncritically.

Concrete case-study-02 confirmation / disconfirmation criteria per
pattern are in plan v2.1 §1.5; short form:

- **Pattern A confirmed** if case study 02 surfaces ≥1 empirical-vs-
  prediction divergence + absorbs it via amendment before shipping.
  **Disconfirmed** if smoke fails to catch a production-exposed
  divergence (the discipline didn't fire).
- **Pattern B confirmed** if case study 02 surfaces ≥1 single-artifact-
  multi-use-case hidden assumption (per the necessary conditions) +
  responds with generalization. **Disconfirmed** if hidden assumptions
  surface but fit none of the necessary conditions (pattern is
  rung-4c-specific).
- **Pattern C confirmed** if cross-review fires, surfaces findings
  spanning ≥2 cells, and triage classifications survive independent
  re-read. **Disconfirmed** if findings fit no cell or persistent
  classification disagreement reveals ambiguous boundaries.
- **Triad-as-a-whole disconfirmed** if case study 02 surfaces a real
  load-bearing methodology lesson that fits **none** of A, B, C —
  evidence the triad is not exhaustive over case-study methodology
  contributions.

The harness generalizations (renderer's multi-schema-version filtering
+ D-entry extraction; modal_app's standalone-conversion-aborted-
inference-aware gate; emit_sarif's inference-status-provenance
threading; shared `_harness/inference_manifest.py`) are **MGN-agnostic
by design intent**, not by tested guarantee. Case study 02 tests
whether the design intent holds.

---

## 4. Open items deferred to post-rung-4 (from plan v2.1 §2)

Five named deferrals carried forward, each with a trigger-for-promotion
governed by the v2.1 §2-prelude rule (substance-over-surface match;
original-rationale impact; explicit narrative; tests are not a
production runtime context):

- **§2.1 SARIF schema bump 1.0 → 1.0.1** (B-medium) — required-field
  `inference_run_status` + rung-4a/4b backfill. Trigger: downstream
  consumer treats the field as a hard contract.
- **§2.2 D0-23 entry** (B-wide) — inference-status provenance as
  first-class schema concept. Trigger: case study 02's first salvage-
  or-failed-inference scenario.
- **§2.4 Rung-4a/4b SARIF backfill of `inference_run_status`**
  (companion to §2.1).
- **§2.5 Cross-rung N=12 vs N=20 framing sharpening** — only if
  Stuttgart/Munich review reads the N-asymmetry as a methodology gap
  rather than as methodology-consistency-honoring (D0-22 amendment 2).

§2.3 (`_classify_inference_run_status` shared helper promotion) was
**resolved at §9 fold-in round 3** when the duplicate-logic-drift
failure mode manifested concretely (both copies carrying the same
fail-open bug); see plan v2.1 §2.3 for the revised resolution
narrative.

---

## 5. Operational sections

### Files

- **`DECISIONS.md`** — Cross-rung methodology trail (D0-01..D0-22+).
  Single source of truth for *why* each methodology choice was made.
  Physics-lint commits in this subtree reference D-entries by number
  for attribution. Read top-down for chronological story; the
  cumulative-state summary at the bottom gives the current shape.
- **`docs/`** — Plans (`physics-lint-validation-plan-v2.md` frozen +
  `physics-lint-validation-plan-v2.1.md` post-rung-4c absorption-
  extended), dated design docs, and per-rung writeups (rung-4a,
  rung-4b, rung-4c at `2026-05-04-*` / `2026-05-05-*` / `2026-05-07-*`).
- **`tools/`** — Cross-rung renderers (`render_cross_stack_table.py`
  for conservation v1.0; `render_eps_table.py` for equivariance v1.1).
- **`tests/`** — Renderer regression tests + fixtures.

### Scope

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

### Why methodology lives in physics-lint, not in a sibling repo

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

### Adding a new D-entry

Same shape as existing entries: D0-XX (or DN-XX for milestone N),
date, one-sentence title, **Question** (what the choice point is),
**Decision** (what was chosen + the pre-registered evidence), other
sections per the existing pattern (Realized, Forward agenda, etc.).
Reference the physics-lint commit SHA where the decision was realized.
Keep amendments inline within their parent D-entry rather than
chaining new D-entries — the scope discriminator is "within-pre-
registered-scope refinement → amendment, out-of-scope discovery → new
entry" (see D0-17 vs D0-15 amendment 5 framing for the canonical
example, and D0-22 + amendments 1, 2 for the rung-4c example with
two amendments to one parent entry).

### Adding a new rung writeup

Per the rung-4 series's established shape: dated writeup at
`docs/YYYY-MM-DD-rung-NX-<topic>-table.md`, paired with a `-design.md`
+ `-plan.md` predecessor pair, sha-bound to a specific commit (the
rung's headline sha). After landing, the integrating README is the
durable cross-rung artifact; individual writeups stay sha-bound to
their snapshot views and are not edited after their rung closes.
