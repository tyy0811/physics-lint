# Textbook availability — 2026-04-20

Recorded by Task 0 Step 17 (named substep; brainstorming decision 2026-04-20).
Consumed by per-rule CITATION.md files that cite textbooks.

## Status

| Textbook | Local copy? | Web verification | Consumer tasks |
|----------|-------------|------------------|----------------|
| Evans, *Partial Differential Equations* (AMS, 2nd ed. 2010), ISBN 978-0-8218-4974-3 | ❌ no | Partial — see below | Tasks 1 (§2.2.3 Thm 4), 2 (§7.1.2 Thm 2), 5 (§2.2.4 Thm 13, §2.3.3 Thm 8) |
| Ciarlet, *FEM for Elliptic Problems* (SIAM Classics 2002), ISBN 978-0-89871-514-9 | ❌ no | Not attempted (Tier-B only) | Tier-B Task 15 (§4.1 Thms 4.1.2–4.1.6) |
| Trefethen, *Spectral Methods in MATLAB* (SIAM 2000), ISBN 978-0-89871-465-4 | ❌ no | Not attempted (Tier-B only) | Tier-B Task 7 (Ch. 3) |

## Evans theorem-number verification (2026-04-20 web pass)

Web sources consulted: Princeton math department notes (web.math.princeton.edu/~const/maxhar.pdf), Stanford Math 220B handouts (web.stanford.edu/class/math220b/handouts/greensfcns.pdf), MSU lecture notes on Evans (users.math.msu.edu/users/yanb/849-full-note.pdf), AMS Graduate Studies Vol. 19 TOC (ams.org/books/gsm/019/).

- **§2.2.3 Theorem 4 (harmonic maximum principle) — VERIFIED.** Multiple sources confirm Evans §2.2 Theorem 4 combines the weak and strong maximum principles for harmonic functions: "(i) max_{Ω̄} u = max_{∂Ω} u (weak); (ii) if U is connected and u attains its max in the interior, u is constant (strong)." Task 1's anchor targets assertion (i).
- **§7.1.2 Theorem 2 (parabolic energy estimates) — SECTION-LEVEL ONLY.** Web sources confirm §7.1 covers second-order parabolic equations and §7.1.2 specifically addresses existence of weak solutions via energy estimates in the Galerkin framework. Exact theorem number (2 vs adjacent) not confirmed from web sources; Task 2's anchor cites "Evans §7.1.2 energy-estimate theorem" as the concept pointer.
- **§2.2.4 Theorem 13 (Green's function / positivity for Poisson) — SECTION-LEVEL ONLY.** §2.2.4 is confirmed to cover Green's function and representation formulas for Poisson's equation. Some web sources index a Lemma 13 in this section on Green's function symmetry (G(x,y)=G(y,x)), not positivity — the exact theorem number for the positivity-of-solution-with-f≥0 statement is not confirmed. Task 5's anchor cites "Evans §2.2.4 (positivity of the Poisson solution with f≥0 via Green's function)" as the concept pointer.
- **§2.3.3 Theorem 8 (heat equation weak max principle) — SECTION-LEVEL ONLY.** §2.3.3 is confirmed to cover properties of heat-equation solutions including the weak maximum principle; multiple lecture notes reference it. Exact theorem number not independently confirmed. Task 5's anchor cites "Evans §2.3.3 weak maximum principle for heat" as the concept pointer.

## Fallback rule applied (per input spec §9 item 3(a) + memory feedback_textbook_web_verify.md)

For the three Evans references not confirmed at theorem-number precision by web search, the citing rule's `CITATION.md` uses the verified **section + concept** as the anchor, with explicit acknowledgment that the exact theorem number inside that section is pending local-copy verification. No fabricated theorem numbers; no silent guesses.

The concept pointer is strong enough for a reviewer holding any edition of Evans to find the right theorem by reading the section. If a reviewer requests tightening, a future PR can bump these to exact theorem numbers with source URL evidence, with a matching `docs/backlog/v1.1.md` entry if that turns out to change the rule's verdict interpretation (it will not — the rule does not depend on the theorem number, only on the mathematical content).

## When to update this file

- On successful verification of a previously-unavailable textbook (local copy acquired, or a more authoritative web source surfaces), flip the relevant row to ✅, update the date at the top, and remove the "section-level only" caveat from the citing `CITATION.md`.
- Commit message: "docs(external-validation): textbook <name> verified; caveats cleared in <rule_id>/CITATION.md".
