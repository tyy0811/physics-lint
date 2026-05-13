# physics-lint validation plan v2.1.1 — Phase 3 amendment (Case Study 02)

**Date:** 2026-05-13
**Predecessor:** [`./physics-lint-validation-plan-v2.1.md`](./physics-lint-validation-plan-v2.1.md) (frozen at its commit date; pointers added at §1.3 + §1.4 + §3 changelog forward-flag to mark resolution of v2.1.1's four items).
**Triggered by:** [Case Study 02 Phase 3 design §5.5](./2026-05-11-case-study-02-physicsnemo-mgn-design.md) trigger condition (*"Phase 3 cross-review of writeup prose completes"*) — fired at Task 9 + Task 10 close.
**Successor:** `physics-lint-validation-plan-v2.1.2.md` (forthcoming, separate; absorbs execution-emergent discoveries per the v2.1.1 vs v2.1.2 epistemic distinction below).

---

## v2.1.1 vs v2.1.2 — pre-registered vs execution-emergent

v2.1.1 absorbs the **four amendments pre-registered in v2.1 design §5.5** (predicted at design time before Phase 1 fired; landed at Phase 3 close as the design predicted).

v2.1.2 (separate, forthcoming) will absorb **execution-emergent methodology discoveries** — round-code-1 walls 1-2 (level rendering + file-path location) and CS02 Phase 2's fourth empirical prose-scope-qualifier instance (floor-bounds-resolution distinction). These were discovered DURING execution rather than predicted at design time; they warrant separate v2.1.2 framing.

The epistemic distinction matters for the methodology trail's honesty: v2.1.1 says *"we predicted these would land at Phase 3; here they are, as predicted."* v2.1.2 will say *"execution surfaced these; here's what we learned that we didn't anticipate."* Both are legitimate methodology contributions; the separation lets the writeup honestly track *what was foreseen* vs *what was empirically discovered*.

---

## Two categories — framework-behavior observations + specific-finding flags

v2.1.1's four amendments split into two categories:

- **Framework-behavior observations** — observations about how the methodology framework itself evolved across the design-doc rounds. Items 1 + 2.
- **Specific-finding flags** — flags about specific findings from CS02 execution that point outward to dated artifacts holding the full evidence. Items 3 + 4.

---

## §1. Framework-behavior observations

### §1.1 Pattern-C 4th instance documentation (resolves v2.1 §1.3)

The pattern-C catalogue at v2.1 §1.3 currently lists three within-rung-4c instances (rung-4c §9 fold-in round 1 + round 2 + round 3). Round-prose-1's absorption already started counting a 4th cell-2 instance (round-prose-1 itself). CS02 Phase 1's D0-23 + Phase 2's D0-24 + Phase 3's prose cross-review supply additional cell-2 instances; v2.1.1 documents the 4th-instance threshold being crossed and elevates pattern-C from "rung-4c-internal observation" to "validated within at least one cross-rung context (CS02)."

**Outward pointer:** see [DECISIONS.md D0-23 + D0-24](../DECISIONS.md) for the per-case-study triage tables; v2.1 §1.3 receives a pointer to this v2.1.1 entry marking the 4th-instance threshold cross.

### §1.2 Per-section convergence-pattern observation (resolves v2.1 §1.3 + §3 changelog)

Across v2.1's review-round sequence (round 1 → round 2 → round 3 → round-prose-1 → round-prose-2 → round-codex-2/-3/-4 → round-code-1), the round-magnitude of findings has decreased per-section (§1 review → §2.1-§2.3 review → §2.4-§2.6 review → §3 review → §4 + §5 review → §6 review). Each round's findings cluster on the most-recently-added section content; older sections converge to "no new findings." This convergence is measured *per-section*, not *per-doc* — adding new sections re-opens the per-section convergence at the new section.

The observation supports v2.1 §1.3's pattern-C "absorbed → fewer findings on this surface" claim and v2.1 §1.4's "cross-review-gate cost-amortizes across rungs" claim. The per-section granularity is the methodology contribution; without it, the convergence claim could be read at the wrong granularity.

**Outward pointer:** v2.1 §1.3 + §1.4 receive pointers to this v2.1.1 entry.

---

## §2. Specific-finding flags

### §2.1 D0-22 strictly-conservative forward-declaration falsification flag

CS02 design §1 (preflight evidence + substrate-class smoke design) falsifies v2.1's D0-22 strictly-conservative anchor assignment: cylinder_flow is NOT strictly conservative; it's open-driven-dissipative under physicsnemo's mesh-side classification. D0-22's anchor assignment named TGV2D as the dissipative-isotropic anchor and (forward-declared) cylinder_flow as the strictly-conservative anchor for a future case study; CS02 Phase 1 + Phase 2 verdicts confirm cylinder_flow as open-driven-dissipative, falsifying the forward declaration.

v2.1.1 records this flag. The corrective amendment (re-assigning the strictly-conservative anchor to a different upstream dataset) defers to case study 03 candidate identification — assigning speculatively before a strictly-conservative substrate is concretely identified would risk the same forward-declaration error a second time.

**Outward pointer:** see [Case Study 02 design §1](./2026-05-11-case-study-02-physicsnemo-mgn-design.md) (preflight + substrate-class smoke design) for where the falsification first surfaced; v2.1 §1.3 receives a pointer to this v2.1.1 entry.

### §2.2 Prose-cross-review vs artifact-cross-review modes flag

Round-prose-2's forward-flag (v2.1 §3 changelog) named the distinction between the two cross-review modes: artifact-cross-review surfaces fail-open paths (round-codex-2/-3/-4 precedent); prose-cross-review surfaces first-impression weaknesses (round-prose-1 + round-prose-2 precedent). Round-code-1 already INSTANTIATED the flag (committed prose + working SARIF upload + Security tab populated); CS02 Phase 3's prose cross-review provides a fifth instance (ordering / forward-cite hygiene; reader-first cross-reference placement; see [CS02 Phase 3 cross-review](./2026-05-13-case-study-02-phase-3-cross-review.md) meta-summary), completing the empirical evidence for the distinction.

v2.1.1 names the flag with a one-paragraph summary and points outward. Full instantiation evidence:

- **Prose-cross-review precedents** (first-impression weaknesses; ordering / forward-cite hygiene): round-prose-1, round-prose-2, round-code-1 prose dimension (level rendering / file-path location / prose framing walls), CS02 Phase 3 cross-review (this Phase — six findings clustered on Headline-ordering and shorthand-before-glossary).
- **Artifact-cross-review precedents** (fail-open paths; data-level invariants): round-codex-2 + round-codex-3 + round-codex-4 (rung-4c PR #9 reviews), CS02 Phase 2 cross-review (one HIGH SARIF-data finding — sentinel field — plus four MEDIUM data-level prose-or-comment findings).

The §1.4 review-discipline triple (smoke + source + cross) does not currently distinguish the two cross-review modes; v2.1.2 §1.4 amendment is the right venue for adding the distinction with worked criteria for when each mode applies.

**Outward pointer:** see v2.1 §3 round-prose-2 changelog forward-flag which is now RESOLVED with a pointer to this v2.1.1 entry; full instantiation evidence at [round-code-1 PR #10 merge](../../../../README.md#hero-physics-lint-in-ci) (level rendering / file-path location / prose framing) + [CS02 Phase 3 cross-review](./2026-05-13-case-study-02-phase-3-cross-review.md).

---

## Acceptance

This file lands the four §5.5 items per their pre-registered scope. Light self-check (Task 13) verifies each item lands + two-category grouping preserved + outward references SHA-pinned.

Any v2.1.1+ expansion (a fifth pattern-C instance, a new framework-behavior observation, a CS02-emergent finding warranting framework-level absorption) goes to v2.1.2 or a fresh forward-flag — NOT retrofitted into v2.1.1.
