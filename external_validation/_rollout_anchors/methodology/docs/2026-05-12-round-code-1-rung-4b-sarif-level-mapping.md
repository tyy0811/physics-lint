# Round-code-1 — rung-4b SARIF band-to-level mapping (prose-vs-code drift absorption)

**Date:** 2026-05-12
**Predecessor:** rung-4 series closure (`tag:rung-4-closure` at master merge `7e1bd9d`); PR #9 merged.
**Successor:** rung-4b §6 item 3 forward-flag closed by this absorption (see §10); P2 RPF2D pre-registration (D0-24) proceeds independently.

**Branch:** `worktree-rung-5-rpf2d-and-rung-4b-security` (isolated worktree, branched from master `7e1bd9d` to avoid colliding with the in-flight CS02 Phase 1 work on `feature/case-study-02-physicsnemo-mgn`).

**Commits:**
- **Commit A** (`483e9831ad`) — `_harness/lint_eps_dir.py` band-to-level mapping + 11 TDD tests (level-rendering half).
- **Commit B** (`69f0fc0`) — re-emitted rung-4b eps SARIFs at commit A's sha + Security-tab upload workflow + this writeup + v2.1 §3 changelog entry.
- **Second-wall commits** (`aeb4a53e87`, `96ce8ed0eb`, `387f617`, + this writeup-update commit) — GitHub Code Scanning rejected the location-less harness SARIFs; `_harness/sarif_emitter.py` extended with physical + logical `locations` + `partialFingerprints`, all six SARIFs re-emitted, workflow goes all-green. See §13.

**Methodology pre-registrations touched:** none (no new D-entry; the amendment *is* the audit trail). Cross-references: rung-4b design §3.3 (the band thresholds), rung-4b writeup §6 item 3 (the claim being made-true), plan v2.1 §1.4 (the prose-vs-code drift mode forward-flag, now instantiated), plan v2.1 §2.4 (the hybrid-Pattern refinement in §13).

---

## 1. Headline

The rung-4b writeup §6 item 3 forward-flag — *"GNS APPROXIMATE-band rows produce `level: "warning"` SARIF entries… GNS FAIL-band rows produce `level: "error"`, exercising the error rendering path. Full Security-tab integration screenshots and PR-comment integration are deferred to a separate deliverable."* — was found, on a smoke pass before building that deliverable, to be a writeup-vs-code drift: `_harness/lint_eps_dir.py` hardcoded `level: str = "note"` for every active row, and all six committed LB SARIFs (rung 4a/4b/4c) were `level: note` only. The claim was aspirational ("the design enables…") framed as descriptive ("the artifact does…").

Round-code-1 absorbs the drift the same way round-prose-2 absorbed the §5.3 cover-letter under-framing: it amends the surface AND fixes the underlying machinery. Commit A adds `_band_to_level(eps)` per design §3.3 (PASS ≤ 1e-5 → note; APPROXIMATE 1e-5 < ε ≤ 1e-2 → warning; FAIL > 1e-2 → error). Commit B re-emits the two rung-4b eps SARIFs at commit A's sha — the GNS-TGV2D eps SARIF now carries **37 warning + 43 error + 60 note**, matching the rung-4b writeup §3.3 observation (2) exactly ("37 of 80 APPROXIMATE / 43 of 80 FAIL" + 20 SKIP + 20 identity + 20 translation = 60 note). SEGNN-TGV2D stays 140 note (exact E(2)-equivariance, all PASS at the float32 floor). A new workflow (`.github/workflows/rung-4-sarif-upload.yml`) uploads all six LB SARIFs to the GitHub Security tab, closing the §6 item 3 forward-flag substantively.

Round-code-1 is also the **first prose-vs-code cross-review instance** in the rung-4 series — the `round-prose-N` rounds were prose-vs-prose (cross-review of the v2.1 plan text against itself), the `round-codex-N` rounds were artifact-vs-artifact (cross-review of code + committed artifacts against each other). This is the first where the review crossed the prose/code boundary: a writeup claim checked against the code that was supposed to back it. Plan v2.1 §1.4 had this as a forward-flag ("observed correspondence, not load-bearing structural claim"); round-code-1 is the instantiated discipline. See §8.

---

## 2. The discovery

The Security-tab demo was scoped as a cheap forward-flag close (§6 item 3): add a GitHub Actions step uploading the rung-4b eps SARIFs, verify the Security tab renders the warning + error rows, capture screenshots. The pre-flight check before writing the workflow was to confirm the SARIFs actually carry warning/error levels — the load-bearing content of the demo.

```bash
$ for f in .../outputs/sarif/*.sarif; do
    jq -r '[.runs[0].results[].level] | group_by(.) | map({(.[0]): length}) | add' $f
  done
{ "note": 36 }   # gns_dam2d
{ "note": 60 }   # gns_tgv2d (conservation)
{ "note": 140 }  # gns_tgv2d_eps  ← expected: note + warning + error
{ "note": 36 }   # segnn_dam2d
{ "note": 60 }   # segnn_tgv2d (conservation)
{ "note": 140 }  # segnn_tgv2d_eps
```

All `note`. The eps SARIFs — which the rung-4b writeup §6 item 3 specifically named as carrying the warning + error rendering paths — were note-only. Tracing to source:

```python
# _harness/lint_eps_dir.py (pre-round-code-1, line 71)
level: str = "note"
...
results.append(HarnessResult(rule_id=rule_id, level=level, ...))  # always note
```

`lint_eps_dir` set `level = "note"` unconditionally for every active row; the `else:` branch computing `eps_first` (line 79) never touched `level`. The `HarnessResult` dataclass declares `level: Literal["note", "warning", "error"]` — the schema *supported* the levels — but the consumer never produced anything but `note`. The `# type: ignore[arg-type]` on the `level=level` line was the only trace that the author had noticed the loose typing.

So the §6 item 3 claim — written as a present-tense description of SARIF behavior — described behavior the code never had. Uploading the SARIFs as-is would have put 472 informational notes on the Security tab: exactly the state the rung-4a writeup §3.2 dismissed ("Harness-style SARIF emits `level: 'note'` rows for PASS-equivalent values; 4a has no findings to populate the Security tab meaningfully. An empty Security tab is not a demo of integration."). The demo could not close §6 item 3 without first making §6 item 3 true.

---

## 3. Root cause

Two layers:

1. **Code layer:** `lint_eps_dir.py` was written with the SARIF *schema* in mind (the `HarnessResult.level` field exists) but the *band-to-level mapping was never wired*. The renderer (`render_eps_table.py`) classifies the band from `raw_value` at render time, so the rendered table showed PASS/APPROXIMATE/FAIL correctly — the gap was invisible unless you looked at the SARIF `level` field directly, which nothing in the rung-4b pipeline did (the table render reads `raw_value`, not `level`).

2. **Prose layer:** the rung-4b writeup §6 item 3 was drafted while the writeup author was reasoning about *what the design enables* (a band-to-level mapping is the natural design; the SARIF spec has note/warning/error; the bands are 3-valued; the mapping is obvious) and wrote it as *what the artifact does* ("GNS APPROXIMATE-band rows produce `level: 'warning'` SARIF entries"). The two are different claims, and the writeup conflated them. See §6 for the methodology implication.

Neither layer was caught by the rung-4 series's prior cross-review rounds because:
- The `round-codex-N` artifact reviews looked at code + committed artifacts *against each other* — and the code (note-only) and the artifacts (note-only) were *consistent* with each other. There was no internal contradiction.
- The `round-prose-N` reviews looked at the v2.1 plan text *against itself* — and the §6 item 3 claim lives in the rung-4b *writeup*, not the v2.1 plan, so it was out of the prose-review scope.

The drift sat in the gap: a writeup claim vs the code that should back it, with no review mode that crossed that boundary. Round-code-1 is that mode. See §8.

---

## 4. Resolution

### Commit A (`483e9831ad`) — band-to-level mapping + tests

`_harness/lint_eps_dir.py`:
- Module-level constants `EPS_PASS_THRESHOLD = 1e-5`, `EPS_APPROXIMATE_THRESHOLD = 1e-2` (per rung-4b design §3.3).
- `_band_to_level(eps: float) -> str`: `eps ≤ 1e-5 → "note"`; `1e-5 < eps ≤ 1e-2 → "warning"`; `eps > 1e-2 → "error"`.
- `lint_eps_dir`: the active-row branch now sets `level = _band_to_level(eps_first)` after computing `eps_first`. SKIP rows keep `level = "note"` (no eps to classify; `skip_reason` carries the methodology signal).

Tests (`test_lint_eps_dir.py`, 11 new, TDD): 9-case parametrized band-to-level table including boundary cases at exactly 1e-5 (still PASS → note) and exactly 1e-2 (still APPROXIMATE → warning); a threshold-constant pin (`EPS_PASS_THRESHOLD == 1e-5`, `EPS_APPROXIMATE_THRESHOLD == 1e-2`) so a future amendment must update the design doc in lockstep; a SKIP-stays-note test. The pre-existing first test already asserted `level == "note"` for a 4.3e-7 PASS-band value — that assertion stays correct post-change. Full harness suite green, no regressions.

### Commit A's sha → re-emitted SARIFs

`emit_sarif_eps.py` reads `git rev-parse --short=10 HEAD` for `physics_lint_sha_sarif_emission`. Run after commit A landed, it produced:
- `outputs/sarif/segnn_tgv2d_eps_483e9831ad.sarif` (supersedes `…_255af5de8d.sarif`)
- `outputs/sarif/gns_tgv2d_eps_483e9831ad.sarif` (supersedes `…_255af5de8d.sarif`)

The eps_t.npzs themselves were re-pulled from Modal Volume (`modal volume get rollout-anchors-artifacts /trajectories/{segnn,gns}_tgv2d_255af5de8d/`) — the `eps_computation` sha is unchanged at `255af5de8d` (same rung-4b T9 measurements), only the `sarif_emission` sha advances to `483e9831ad`. So this is a faithful re-emission: identical eps values, identical `raw_value` properties, identical run-level provenance fields except `physics_lint_sha_sarif_emission`; the only *new* content is the band-derived `level` field. The old `…_255af5de8d.sarif` files are removed (superseded).

### Commit B (this commit) — workflow + writeup + changelog

- `.github/workflows/rung-4-sarif-upload.yml`: matrix-upload of all six LB SARIFs to the Security tab (`security-events: write`), distinct `category` per stack per the `physics-lint.yml` dogfood-workflow convention. Triggers: push to master + pull_request.
- This writeup.
- v2.1 §3 changelog entry: `round-code-1 absorption — 2026-05-12`.

---

## 5. The new evidence

Post-round-code-1 SARIF level distributions:

| SARIF | note | warning | error | Reading |
|---|---|---|---|---|
| `segnn_tgv2d_8e49339469` (4a cons.) | 60 | 0 | 0 | mass=0.0 PASS; energy_drift SKIP D0-18; dissipation_sign_violation=0.0 PASS — all-PASS-equivalent |
| `gns_tgv2d_8e49339469` (4a cons.) | 60 | 0 | 0 | same |
| `segnn_tgv2d_eps_483e9831ad` (4b equiv.) | 140 | 0 | 0 | exact E(2)-equivariance; all PASS at float32 floor (~2.3e-7) |
| `gns_tgv2d_eps_483e9831ad` (4b equiv.) | **60** | **37** | **43** | bimodal: 20 SKIP + 20 identity + 20 translation = 60 note; APPROXIMATE lower mode = 37 warning; FAIL upper mode (0.02, √2·0.02, √3·0.02) = 43 error |
| `segnn_dam2d_bc3bae929d` (4c cons.) | 36 | 0 | 0 | mass=0.0 PASS; energy_drift SKIP D0-22a1; dissipation_sign_violation SKIP D0-22 |
| `gns_dam2d_bc3bae929d` (4c cons.) | 36 | 0 | 0 | same |

The GNS-TGV2D eps row's `60 note / 37 warning / 43 error` is the substantive demo content: it is *byte-consistent with* the rung-4b writeup §2's condensed summary ("the APPROXIMATE/FAIL split is 37/43 across the four active-symmetry rules… SEGNN: 80/80 PASS, monomodally at float32 floor. GNS: 0/80 PASS") and §3.3 observation (2) ("Lower mode (37 of 80 rows): APPROXIMATE band — … Upper mode (43 of 80 rows): … FAIL band, with quantized magnitudes at 2.000e-2, 2.828e-2 (=√2·0.02), 3.463e-2 (=√3·0.02)"). The band-to-level mapping was never wrong in the *rendered table* — only absent from the *SARIF level field*. Round-code-1 closes that gap; the SARIF level now carries the same band signal the renderer derives from `raw_value`.

The Security-tab integration the §6 item 3 forward-flag named is now actual: the GNS eps SARIF, once uploaded, produces 37 `warning`-level code-scanning alerts (APPROXIMATE-band equivariance ε) and 43 `error`-level alerts (FAIL-band), each carrying the `raw_value` ε scalar and the `transform_kind` / `transform_param` / `traj_index` properties; the SEGNN eps SARIF and the rung-4a/4c conservation SARIFs upload as informational notes (no alerts), which is the correct rendering for all-PASS-equivalent rows and confirms the demo is honest about which rows generate alerts and which do not.

---

## 6. Prose-discipline sub-finding — aspirational vs descriptive

The drift exists because §6 item 3 wrote a forward-looking claim ("the design enables a band-to-level mapping; therefore the SARIFs carry warning/error levels") in the grammatical mood of a descriptive one ("GNS APPROXIMATE-band rows produce `level: 'warning'` SARIF entries"). The writeup author was reasoning about the design's *capability* and wrote it as the artifact's *behavior*. The two are different:

- **Descriptive claim:** "the artifact at sha X does Y" — verifiable by reading the artifact.
- **Aspirational/design claim:** "the design enables Y" or "the schema supports Y" — verifiable by reading the design, but says nothing about whether any artifact has been built that does Y.

A reviewer who reads "GNS APPROXIMATE-band rows produce `level: 'warning'`" reasonably assumes the SARIFs do that. They didn't. The methodology implication, beyond this one item:

> **Writeup prose should distinguish aspirational from descriptive claims.** "The artifact does X" is a checkable assertion about a specific committed artifact; "the design enables X" / "the schema supports X" is an assertion about the design surface. Conflating them — writing a design-capability claim in descriptive mood — creates exactly the prose-vs-code drift round-code-1 absorbs. When a writeup makes a behavioral claim about an artifact, that claim should either (a) be true of the committed artifact at the writeup's sha, or (b) be explicitly marked as design-enabled-but-not-yet-implemented ("the schema supports a band-to-level mapping; rung 4b ships note-only and defers the mapping to a follow-up").

This sub-finding seeds a future methodology entry (v2.1.2 or v2.2) on prose-discipline distinctions — a sibling to the `round-prose-N` / `round-codex-N` / `round-code-N` taxonomy that names *modes of review*, this would name *modes of claim* (descriptive / aspirational / design-enabled / observed-as-of-sha). CS02's writeups inherit the distinction: when CS02 documents what its MGN materializer or its mesh adapter *does*, that should be a descriptive claim verifiable at the CS02 writeup's sha, not a design-capability claim in descriptive mood.

---

## 7. Cross-session heads-up for CS02

The CS02 Phase 1 work (on `feature/case-study-02-physicsnemo-mgn`, in a parallel session) should be told this, with the framing below carried forward into CS02's design/plan as a sentence in the Pattern A enabling-discipline section:

> v2.1 §1.4's prose-vs-code drift mode forward-flag now has its first empirical instance — rung-4b §6 item 3 claimed warning/error SARIF levels but actual code emitted note-only across all bands. Documented as round-code-1 absorption at `external_validation/_rollout_anchors/methodology/docs/2026-05-12-round-code-1-rung-4b-sarif-level-mapping.md`. CS02 Pattern A predictions in §2.1 should anticipate this drift mode: empirical-vs-prediction divergence isn't just about runtime metrics, it's also about writeup-claim vs code-reality. Worth a sentence in CS02's §2.1 enabling-discipline section noting that pre-flight assertions should verify writeup claims against code behavior, not just code behavior in isolation.
>
> Update (round-code-1 closed): the forward-flag is no longer aspirational — round-code-1 instantiated it (the level-mapping drift), refined it through a second wall (the GitHub-Code-Scanning locations gap), and closed cleanly without triggering the third-wall escalation. The full trail is in this writeup §13 + the v2.1 §3 changelog (the §1.4 *section-body* amendment still waits for round-code-2 to give a second prose-vs-code instance). So CS02's §2.1 section now has concrete prior art to point at, not just a forward-flag — and a second item to anchor on: §2.4 now allows a hybrid Pattern-A-surfacing / Pattern-B-response classification (the locations finding was Pattern-A-surfaced via an empirical workflow run but Pattern-B in response shape — modify a shared multi-consumer artifact).

The CS02 session can fold this in immediately, defer it as a known follow-up, or push back if there's a reason it doesn't apply. (Propagation is the CS02 session's call — round-code-1 does not touch the CS02 branch from this worktree, per branch isolation.)

---

## 8. Methodology contribution — prose-vs-code as a third cross-review mode

The rung-4 series's cross-review machinery now has three instantiated modes, with distinct findings characters:

| Mode | What it cross-reviews | Findings character | Rung-4 instances |
|---|---|---|---|
| `round-codex-N` (artifact) | code + committed artifacts against each other | fail-open paths; internal contradictions; defensive-validation gaps | §9 fold-in rounds 1, 2, 3; round-codex-2, -3, -4 |
| `round-prose-N` (prose) | the plan text (v2.1) against itself | first-impression weaknesses; framing that under/over-claims; ambiguous taxonomy boundaries | round-prose-1, round-prose-2 |
| `round-code-N` (prose-vs-code) | writeup claims against the code that should back them | aspirational-claimed-as-descriptive drift; documentation that outran implementation | **round-code-1 (this absorption)** |

Plan v2.1 §1.4 had the `round-codex` ↔ `round-prose` distinction noted with a diagonal-mapping disclaimer ("observed correspondence, not load-bearing structural claim"). Round-code-1 instantiates the third mode — and importantly, it's the mode that catches drift the other two structurally can't: artifact-review needs an internal contradiction (code and artifact were consistent here, both note-only), prose-review needs the claim to live in the reviewed text (the §6 item 3 claim lives in the rung-4b writeup, out of the v2.1-prose-review scope). The three modes are complementary, not redundant — same shape as the smoke/source/cross review-discipline triple in §1.4.

This strengthens v2.1 §1.4 from "smoke + source + cross" (where "cross" was implicitly artifact-cross-review) to a finer taxonomy: cross-review has at least three sub-modes (artifact / prose / prose-vs-code), and a complete cross-review pass should exercise all three. The v2.1 §3 changelog entry for round-code-1 records this; a future v2.1.2 §1.4 amendment can fold the sub-mode taxonomy into the section body once CS02 provides a second prose-vs-code instance (round-code-2) — at which point the mode is bilaterally validated, not single-instance.

---

## 9. What round-code-1 is NOT

1. **Not a broader rung-4b prose-vs-code audit.** §6 item 3 had this drift; other claims in rung-4b's (or rung-4a's, or rung-4c's) writeups might too. A systematic sweep of all "the artifact does X" claims against code reality would take ~30–60 min + 0–N more amendments. Deferred deliberately: round-code-1 establishes the discipline; future cross-reviews catch similar drift naturally; pre-emptive retrospective auditing risks consuming discovery cycles for nothing this late in the visa timeline. If a second prose-vs-code drift surfaces organically (round-code-2), *then* consider a broader audit.

2. **Not a new rung.** "Rung-5" would imply a new workstream alongside the rung-4 series. Round-code-1 is a retrospective amendment *inside* rung-4 closure work, sibling to round-prose-2's amendment of the rung-4c writeup framing — named `round-code-1` to fit the `round-{prose,codex,code}-N` taxonomy, not `rung-5`.

3. **Not a re-derivation of the rung-4b ε measurements.** The eps_t.npzs are unchanged (re-pulled from Modal Volume at the same `eps_computation` sha `255af5de8d`). The `raw_value` ε scalars in the re-emitted SARIFs are byte-identical to the `…_255af5de8d.sarif` versions. The only new content is the band-derived `level` field; the only changed provenance field is `physics_lint_sha_sarif_emission` (`255af5de8d` → `483e9831ad`).

4. **Not a renderer change.** `render_eps_table.py` already derived the band from `raw_value`; it does not read the SARIF `level` field. The rendered table is unchanged. Round-code-1 only affects the SARIF `level` field — i.e., what GitHub Code Scanning sees, not what the methodology table shows.

5. **Not a schema-version bump.** The SARIF document is still SARIF 2.1.0; the harness `harness_sarif_schema_version` run-property is unchanged (v1.1 for eps SARIFs). The `level` field was always part of the SARIF 2.1.0 spec and the `HarnessResult` dataclass; round-code-1 just wires the mapping that was always supported but never connected.

6. **Not a change to the band thresholds.** PASS ≤ 1e-5, APPROXIMATE ≤ 1e-2, FAIL otherwise — inherited verbatim from rung-4b design §3.3. The new `EPS_PASS_THRESHOLD` / `EPS_APPROXIMATE_THRESHOLD` constants codify those values; the threshold-pin test fails loudly if a future amendment changes them without updating the design doc.

---

## 10. Rung-4b §6 item 3 forward-flag — CLOSED

The rung-4b writeup (`2026-05-07-rung-4b-equivariance-table.md`) is sha-bound to its committed snapshot per the v2.1 amendment convention (matching how the rung-4c writeup's §6 item 8 framing was walked back in v2.1 §2.5 without touching the sha-bound writeup itself). So the rung-4b writeup's §6 item 3 prose stays as-authored — the closure is recorded *here* and in the v2.1 §3 changelog, not by editing the frozen writeup.

**Status of §6 item 3 ("Not a GitHub Security-tab integration demo at saturation… deferred to a separate deliverable"):** FULLY CLOSED at round-code-1 (2026-05-12). Both halves:

- **Level-rendering half** — `_harness/lint_eps_dir.py` band-to-level mapping (commit A `483e9831ad`) makes the warning/error rendering paths actual; the re-emitted GNS-TGV2D eps SARIF carries 37 warning + 43 error rows.
- **Security-tab-integration half** — `.github/workflows/rung-4-sarif-upload.yml` runs all six LB SARIF uploads green on PR #10 (run `25744425002` at commit `387f617`); GitHub Code Scanning ingested all 472 results (60+60 conservation tgv2d, 140+140 eps, 36+36 conservation dam2d), and the GNS eps category (`rung-4b-equivariance-gns-tgv2d`) produces exactly **37 `warning`-level + 43 `error`-level code-scanning alerts** — verified via `GET /repos/.../code-scanning/alerts?ref=refs/pull/10/merge` — matching the rung-4b writeup §3.3 obs (2) split across the four active PH-SYM-* equivariance rules (PH-SYM-001: 27 warn / 33 err; PH-SYM-002: 10 warn / 10 err). The all-PASS-equivalent SARIFs (SEGNN eps; both conservation pairs) upload as informational notes with no alerts, which is the honest rendering for those rows.

Getting the SARIFs *accepted* by GitHub Code Scanning required a second fix beyond the level mapping — GitHub rejected the location-less harness SARIFs (`locationFromSarifResult: expected at least one location`) and additionally only *displays* results that carry a file-path location, so the emitter had to be extended with physical + logical `locations` + per-row `partialFingerprints`. That is the "second wall" of round-code-1; see §13. (The "PR-comment integration" the §6 item 3 prose also mentioned — inline code-scanning annotations on PR diffs — is a GitHub-side rendering of the uploaded alerts when the changed files overlap the alert locations; it is not a separate deliverable.)

**One honest caveat carried forward:** the rung-4b writeup §6 item 3 was, at the writeup's sha (`255af5de8d`-era), describing behavior the code did not have. Round-code-1 makes it true going forward but does not retroactively make the `…_255af5de8d.sarif` files carry warning/error levels (they're superseded, not edited). A reader of the rung-4b writeup at its frozen sha should read §6 item 3 as the design-capability-stated-as-descriptive that it was; the v2.1 §3 round-code-1 changelog entry is the methodology-current pointer.

---

## 11. Rederivability + provenance

```bash
# 1a. eps SARIFs (rung-4b) — pull the eps-npz mirrors (unchanged at eps_computation sha 255af5de8d):
modal volume get rollout-anchors-artifacts /trajectories/segnn_tgv2d_255af5de8d/ \
    external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories/
modal volume get rollout-anchors-artifacts /trajectories/gns_tgv2d_255af5de8d/ \
    external_validation/_rollout_anchors/01-lagrangebench/outputs/trajectories/
# 1b. conservation SARIFs (rung-4a/4c) — pull the rollout-npz mirrors. NOTE: `modal volume get`
#     (CLI ≥ 1.4.0) bundles a *directory* download into one opaque file; pull the per-traj npz files
#     individually (and `_inference_manifest.json` for the dam2d stacks):
for sub in segnn_tgv2d_8c3d080397 gns_tgv2d_f48dd3f376 segnn_dam2d_e754a4bc2e gns_dam2d_e754a4bc2e; do
  modal volume ls rollout-anchors-artifacts "/rollouts/lagrangebench/$sub" \
    | grep -E 'particle_rollout_traj|_inference_manifest' \
    | sed "s#rollouts/lagrangebench/$sub/##" \
    | while read f; do
        modal volume get rollout-anchors-artifacts "/rollouts/lagrangebench/$sub/$f" \
          "external_validation/_rollout_anchors/01-lagrangebench/outputs/_local_mirror/$sub/$f"
      done
done

# 2a. (At the locations-fix commit aeb4a53e87 or later) re-emit the eps SARIFs:
python external_validation/_rollout_anchors/01-lagrangebench/emit_sarif_eps.py
# → segnn_tgv2d_eps_<HEAD-sha>.sarif (140 note)
# → gns_tgv2d_eps_<HEAD-sha>.sarif   (60 note + 37 warning + 43 error)
# 2b. re-emit the conservation SARIFs:
python external_validation/_rollout_anchors/01-lagrangebench/emit_sarif.py
# → segnn/gns_tgv2d_<HEAD-sha>.sarif (60 note each), segnn/gns_dam2d_<HEAD-sha>.sarif (36 note each)

# 3. Verify level distributions + that every result carries locations + a partialFingerprint:
for f in .../outputs/sarif/*.sarif; do
    jq -r '[.runs[0].results[].level] | group_by(.) | map({(.[0]): length}) | add' $f
    jq -e 'all(.runs[0].results[]; (.locations|length>=1) and (.partialFingerprints|length>=1))' $f >/dev/null \
      && echo "  locations+fingerprints OK" || echo "  MISSING locations/fingerprints"
done
```

Run at the same code sha with the committed NPZs → identical SARIF output (deterministic). Any divergence reflects a code change, an NPZ change, or both.

**Provenance:**
- physics-lint commits: band-to-level mapping `483e9831ad` (commit A); locations fix `aeb4a53e87`; eps re-emit `96ce8ed0eb`; conservation re-emit `387f617…`.
- eps_computation sha (unchanged): `255af5de8d` (rung-4b T9 PASS state); conservation pkl_inference shas (unchanged): tgv2d `8c3d080397` / `f48dd3f376`, dam2d `e754a4bc2e`.
- sarif_emission shas: eps `255af5de8d` → `483e9831ad` → `aeb4a53e87`; conservation `8e49339469` / `bc3bae929d` → `96ce8ed0eb`.
- LagrangeBench sha (image-build pin, unchanged): `b880a6c84a93792d2499d2a9b8ba3a077ddf44e2`
- SEGNN-TGV2D checkpoint: `sha256:c0be98f9fb59eb4545f05db3d8aa5d31b7c8170b5d4d9634b01749e26598441b`
- GNS-TGV2D checkpoint: `sha256:c1df5675d6b29aa7e4b130afc8b88b31f7109ce41dacc9f4e168e5c485a8765e`
- Modal compute spent: $0 (data transfer only — `modal volume get` of pre-computed npzs)

---

## 12. Integrating-trail update

The integrating README (`methodology/README.md`) and plan v2.1 are the methodology-current view; the per-rung writeups are sha-bound snapshots. Round-code-1's effects on the integrating trail:

- **v2.1 §3 changelog** gains a `round-code-1 absorption — 2026-05-12` entry naming the prose-vs-code drift mode, the §6 item 3 closure, the prose-discipline sub-finding (with the v2.1.2 §1.4 sub-mode-taxonomy forward-flag), and (in the second-wall sub-block) the locations fix + the v2.1 §2.4 hybrid-pattern refinement.
- **Methodology README §2.3** (review-discipline triple) could note in a future touch that "cross-review" has at least three sub-modes (artifact / prose / prose-vs-code); deferred to the v2.1.2 §1.4 amendment after round-code-2 provides a second instance — same defer-until-bilateral discipline the rest of the series uses.
- **rung-4b §6 item 3** is FULLY CLOSED (this writeup §10 + §13); no edit to the sha-bound rung-4b writeup.
- **v2.1 §1.4** picks up two small refinements from §13: (a) when a finding has no natural source-line location, pointing the SARIF physical location at the *rule's implementation file* (committed, causally-honest, sensible click-target) is the defensible third option — neither a fabricated path nor a gitignored artifact; (b) any design that points multiple results at the same physical location must add stable per-result `partialFingerprints` (GitHub Code Scanning's dedup is location-keyed) — the location choice and the fingerprint are a co-design pair, not separable concerns. Both fold into the v2.1.2 §1.4 amendment alongside the prose-vs-code sub-mode taxonomy.
- **v2.1 §2.4** (Pattern A/B/C operational classification) picks up the hybrid-case refinement from §13: a finding can be Pattern-A in *surfacing* (an empirical workflow run surfaced it) yet Pattern-B in *response shape* (modify a shared multi-consumer artifact). §2.4's "classify by what's directly surfaced" test still resolves it to Pattern A; the surfacing→response asymmetry is real and the classification handles it cleanly.

---

## 13. The second wall — GitHub Code Scanning needs a file-path location

### 13.1 What surfaced

§10's level-rendering half closed, the eps SARIFs carried 37 warning + 43 error rows, the upload workflow was in place — but the workflow's *first* run failed all six jobs:

```
##[error]Code Scanning could not process the submitted SARIF file:
  locationFromSarifResult: expected at least one location  (× every result)
```

The harness emitter (`_harness/sarif_emitter.py:HarnessResult.to_sarif_result`) wrote `ruleId` + `level` + `message` + `properties` but **no `locations`** — valid SARIF v2.1.0 (`result.locations` is optional in the spec) but GitHub-incompatible: GitHub Code Scanning requires every `result` to carry ≥ 1 `location`, and (per its SARIF-support docs) only *displays* results whose location is a file-path `physicalLocation.artifactLocation.uri`. A logical-only location (`logicalLocations` with a `fullyQualifiedName`) satisfies neither the "must display" requirement nor — empirically — clears the ingest. So this is a *second* harness-SARIF-emitter gap of the same shape as the band-to-level one: the emitter was built for the methodology-renderer consumer (which reads `raw_value` and ignores both `level` and `locations`) and never exercised against the GitHub-Code-Scanning consumer (which needs `level` + a file-path `locations` + per-row fingerprints).

### 13.2 The fix — physical + logical locations + partialFingerprints

physics-lint harness findings have no natural source-line location: they describe model *behavior* under a physics rule on a particular rollout trajectory, not a defect at a `file:line`. The handoff's first instinct was a logical-only location (the FQN), reasoning that any physical location would be either a fabricated source path or a pointer at the gitignored rollout NPZs. But there is a defensible third physical target: **the committed harness adapter module that *implements* the rule** — a real, in-repo source file, the file the finding causally originates from, and a sensible click-target in the Security tab ("here is how this check works"). The emitter now maps each `ruleId` to its adapter (`PH-SYM-*` → `_harness/symmetry_rollout_adapter.py`; `harness:mass_conservation_defect` / `harness:energy_drift` / `harness:dissipation_sign_violation` and `PH-CON-*` → `_harness/particle_rollout_adapter.py`; `PH-MESH-*` / `mesh:*` → `_harness/mesh_rollout_adapter.py`; anything unmapped → `_harness/sarif_emitter.py` itself) and writes, for every result, one `location` with **both**:

- `physicalLocation` → `{ artifactLocation: { uri: "<adapter module, repo-root-relative>" }, region: { startLine: 1, startColumn: 1, endColumn: 2 } }`. The `region` is a **placeholder**: the finding is about the whole adapter, not line 1 — line 1 is just where the SARIF spec requires *some* region anchor. (Honest disclosure, in the prose-discipline spirit of §6: this `region` does NOT claim the finding is localized to that line; it is the minimal valid anchor for a finding whose true "location" is "the behavior of this rule on this trajectory", which SARIF has no first-class slot for.)
- `logicalLocations` → `[{ name: "<leaf segment>", fullyQualifiedName: "<case_study>/<model>/<dataset>/<rule_id>[/<transform_kind>_<transform_param>]/traj<NN>" }]`. This carries the per-row detail that the physical location (the same `adapter:1` for every result of a rule) cannot.

Because every result of a given rule shares the same `adapter:1` physical location, GitHub would collapse them into one alert per `(ruleId, location)` — losing the 140-row resolution that makes the GNS eps demo a demo. So each result also gets a stable **`partialFingerprints`** entry — `{ "physicsLintResultFqnHash/v1": <16-hex of sha256(fully_qualified_name)> }` — derived from the per-row FQN (which is unique within a SARIF). The `github/codeql-action/upload-sarif` action adds its own `primaryLocationLineHash` etc. on top but does not remove this key, so GitHub's dedup keys on the full fingerprint map and keeps the rows distinct. (If a future GitHub change made `partialFingerprints` insufficient and the rows still collapsed, that would be a new wall to surface, not absorb — see §13.4.)

TDD: `_harness/tests/test_sarif_emitter_locations.py` asserts the physical-URI rule-family mapping, the placeholder region, the logical-location FQN detail, and that two results sharing a physical location have distinct fingerprints; the new assertions were watched failing (`KeyError: 'physicalLocation'` / `'partialFingerprints'`) before the emitter change. (An earlier commit on this branch shipped a *logical-only* version of the emitter + tests; it was superseded — the GitHub-display requirement is a file-path location, which the logical-only design missed.) Full `_rollout_anchors` suite green.

### 13.3 What the second run showed

Re-emitting all six SARIFs at the locations-fix sha (eps at `aeb4a53e87` via `emit_sarif_eps.py`; conservation at `96ce8ed0eb` via `emit_sarif.py`, after `modal volume get` of the four rollout-npz mirrors — see §11 note on the per-file pull) and re-running the workflow: **all six jobs green** (run `25744425002` on commit `387f617`). GitHub ingested all 472 results; the `rung-4b-equivariance-gns-tgv2d` category produced exactly 37 `warning` + 43 `error` code-scanning alerts (PH-SYM-001: 27/33; PH-SYM-002: 10/10) — the rung-4b §3.3 obs (2) split, now live on the Security tab. The first run's three *other* job failures were a transient `codeload` 429 (Too Many Requests) downloading the `github/codeql-action` archive when all six jobs hit the CDN at once — unrelated to the SARIF format; the workflow now caps `max-parallel: 2` to stagger that download.

### 13.4 Methodology — three refinements + the stop-condition check

**(1) Third physical-location target — refinement to v2.1 §1.4.** When a SARIF-emitting tool produces findings that have no source-line location, the choices are not just "fabricate a path" or "point at a non-committed artifact": pointing the `physicalLocation` at the *committed code that implements the check* is defensible because (a) the file exists and is committed, (b) the finding causally originates from that code, (c) the click-through UX is sensible ("how this rule works"). This generalises the "what counts as an honest location" question; folds into the v2.1.2 §1.4 amendment.

**(2) Hybrid Pattern-A/Pattern-B case — refinement to v2.1 §2.4.** The locations finding is a *hybrid*: Pattern-A in surfacing (an empirical workflow run — not a prose review, not an artifact-cross-review — surfaced the divergence between the §6 item 3 demo's intent and the artifact's GitHub-compatibility) but Pattern-B in response shape (the response modifies a shared, multi-consumer artifact — the emitter that both the methodology renderer and GitHub consume — rather than amending a pre-registration). v2.1 §2.4's operational test ("classify by what is *directly* surfaced") still resolves it cleanly to Pattern A; the point is that the surfacing→response shape can be asymmetric, and the classification scheme accommodates that. (This pairs with the §6 prose-discipline finding's "generalises Pattern A to writeup-claim-vs-code-reality" point: round-code-1's two walls together stretched Pattern A on two axes — *what* the smoke checks, and the surfacing-vs-response asymmetry.)

**(3) Placeholder-region honesty.** The `region` on every physical location is `startLine: 1` — a placeholder, not a claim. Recorded here and in `sarif_emitter.py`'s docstring so a reader who clicks through to `symmetry_rollout_adapter.py:1` and finds an unrelated docstring line knows the line number is structural, not semantic.

**(4) partialFingerprints is a co-design decision with shared physical locations — refinement to v2.1 §1.4.** GitHub Code Scanning's deduplication is location-keyed by default: two results with the same `(ruleId, physicalLocation)` collapse into one alert unless their `partialFingerprints` differ. So *any* design that points multiple results at the same physical location — which the "physical → rule-implementation module" choice (refinement 1) does by construction (every result of a rule shares that rule's `adapter:1`) — must add a stable per-result fingerprint, or the multi-row resolution that makes the demo a demo is lost. The two are a co-design pair, not separable concerns; the locations decision should be made *with* the fingerprint requirement in view, not have it discovered afterward. (This was surfaced at execution time, not foreseen in the design recommendation — recorded here so the next emitter design that adopts the rule-implementation-module pattern budgets the fingerprint up front.) Folds into the v2.1.2 §1.4 amendment alongside refinement 1.

**Stop-condition check.** The round-code-1 stop-condition was: a *third wall* (a new GitHub-Code-Scanning requirement category — rule-definition completeness, severity mapping, runId conventions) → escalate, don't silently absorb. What happened: the locations fix surfaced no new requirement category — GitHub accepted the re-emitted SARIFs as-is (no `tool.driver.rules` entries, no explicit `securitySeverity`, no custom runId). The `partialFingerprints` addition was a *known consequence* of the physical-location design (shared `adapter:1` location → collapse risk), reasoned about and documented above, not a separate empirical wall — it is in the same spirit as the placeholder-region disclosure, not a fourth wall. So round-code-1 closes at two walls (level rendering + locations), which fits the §6 item 3 narrative exactly ("the deferred deliverable had two unbacked sub-claims: warning/error level *rendering* AND Security-tab *integration*"). If a genuine third requirement category surfaces in future (a `rules`-completeness rejection, a severity-mapping requirement, a runId convention), *that* crosses the line — escalate, scope-reset (round-code-2 absorbs it as its own instance), don't silently absorb.
