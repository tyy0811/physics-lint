# Textbook availability — 2026-04-20

Recorded by Task 0 Step 17 (named substep; brainstorming decision 2026-04-20).
Consumed by per-rule CITATION.md files that cite textbooks.

## Status

| Textbook | Local copy? | Verification | Consumer tasks |
|----------|-------------|--------------|----------------|
| Evans, *Partial Differential Equations* (AMS, 2nd ed. 2010), ISBN 978-0-8218-4974-3 | ❌ no | ✅ Full text fetched + pages visually inspected via pymupdf → PNG → Claude vision | Tasks 1 (§2.2.3 Thm 4), 2 (§7.1.2 Thm 2), 5 (§2.2.3 Thm 4 positivity corollary + §2.3.3 Thm 4) |
| Ciarlet, *FEM for Elliptic Problems* (SIAM Classics 2002), ISBN 978-0-89871-514-9 | ❌ no | Not yet required (Tier-B only) | Tier-B Task 15 (§4.1 Thms 4.1.2–4.1.6) |
| Trefethen, *Spectral Methods in MATLAB* (SIAM 2000), ISBN 978-0-89871-465-4 | ❌ no | Not yet required (Tier-B only) | Tier-B Task 7 (Ch. 3) |

## Evans theorem-number verification (2026-04-20 tightening pass)

Source: Evans PDF mirrored at `https://wms.mat.agh.edu.pl/~lusapa/pl/Evans.pdf` (AGH Kraków math department). The PDF is image-only (no OCR layer), so verification was done by rendering relevant pages to PNG via `pymupdf` and visually inspecting each page through Claude's multimodal vision at DPI 140. Page numbers below refer to **book pages**, not PDF pages. Book-to-PDF offset = 15 (PDF page = book page + 15).

### ✅ §2.2.3 Theorem 4 — Strong maximum principle for harmonic (book p. 27)

Direct quote from Evans:

> **THEOREM 4 (Strong maximum principle).** *Suppose u ∈ C²(U) ∩ C(Ū) is harmonic within U.*
> *(i) Then* max_{Ū} u = max_{∂U} u.
> *(ii) Furthermore, if U is connected and there exists a point x₀ ∈ U such that u(x₀) = max_{Ū} u, then u is constant within U.*

Consumed by PH-POS-002 (Task 1). Assertion (i) is the weak maximum principle the rule checks; assertion (ii) is the strong version that guarantees constant solutions.

**Also on p. 27 (directly following Theorem 4):** "**Positivity.** The strong maximum principle states in particular that if U is connected and u ∈ C²(U) ∩ C(Ū) satisfies {-Δu = 0 in U, u = g on ∂U}, where g ≥ 0, then u is positive everywhere in U if g is positive somewhere on ∂U."

This *Positivity corollary* (not a separately-numbered theorem) is consumed by PH-POS-001 (Task 5) for the Poisson case.

### ✅ §7.1.2 Theorem 2 — Energy estimates for parabolic (book p. 376)

Direct quote from Evans:

> **THEOREM 2 (Energy estimates).** *There exists a constant C, depending only on U, T and the coefficients of L, such that*
>
> max_{0≤t≤T} ‖u_m(t)‖_{L²(U)} + ‖u_m‖_{L²(0,T; H¹₀(U))} + ‖u'_m‖_{L²(0,T; H⁻¹(U))} ≤ C (‖f‖_{L²(0,T; L²(U))} + ‖g‖_{L²(U)})
>
> *for m = 1, 2, …*

Consumed by PH-CON-003 (Task 2). The L² energy bound on `u_m` implies the E(t) ≤ E(0) dissipation property that the rule checks on the heat-equation eigenmode fixture.

### ❌ §2.2.4 Theorem 13 — Symmetry of Green's function (book p. 35), NOT positivity

The design-spec's Rev 1.6 §2 Task 5 citation of "Evans §2.2.4 Theorem 13 (positivity for Poisson with f ≥ 0, homogeneous Dirichlet)" was **incorrect**.

Direct quote from Evans:

> **THEOREM 13 (Symmetry of Green's function).** *For all x, y ∈ U, x ≠ y, we have* G(y, x) = G(x, y).

This is the symmetry theorem, not positivity. **The correct citation for the Poisson positivity property is §2.2.3 Theorem 4 + its Positivity corollary paragraph on p. 27 (documented above).**

**Correction applied:** Task 5's `PH-POS-001/CITATION.md` will cite Evans §2.2.3 Theorem 4 (with p. 27 Positivity corollary) for the Poisson case, not §2.2.4 Theorem 13. This supersedes the Rev 1.6 spec's attribution.

### ❌ §2.3.3 Theorem 8 — does not exist; the heat max principle is §2.3.3 Theorem 4 (book p. 55)

The design-spec's Rev 1.6 §2 Task 5 citation of "Evans §2.3.3 Theorem 8 (weak maximum principle for heat)" was **incorrect in theorem number**. Evans restarts theorem numbering per section, so both §2.2.3 and §2.3.3 have local Theorem 4s — the spec conflated the global numbering.

Direct quote from Evans:

> **THEOREM 4 (Strong maximum principle for the heat equation).** *Assume u ∈ C²₁(U_T) ∩ C(Ū_T) solves the heat equation in U_T.*
> *(i) Then* max_{Ū_T} u = max_{Γ_T} u.
> *(ii) Furthermore, if U is connected and there exists a point (x₀, t₀) ∈ U_T such that u(x₀, t₀) = max_{Ū_T} u, then u is constant on Ū_{t₀}.*

Where Γ_T is the *parabolic boundary* = bottom cap + lateral sides. Consumed by PH-POS-001 (Task 5) for the heat case.

**Correction applied:** Task 5's `PH-POS-001/CITATION.md` will cite Evans §2.3.3 Theorem 4 (heat case), not §2.3.3 Theorem 8. This supersedes the Rev 1.6 spec's attribution.

## Spec-correction provenance

The two corrections (§2.2.4 Theorem 13 → §2.2.3 Theorem 4 Positivity corollary; §2.3.3 Theorem 8 → §2.3.3 Theorem 4) were uncovered during the 2026-04-20 tightening-pass execution of the named Task 0 Step 17 substep. This is the first instance where the Rev 1.6 design spec was proven to contain a factual error — the read-back gate's six categories did not catch it because the gate reasons about the plan's internal consistency, not about the external truth of its citations.

The corrections land ONLY in the per-task CITATION.md files, NOT in the committed design spec (`docs/superpowers/specs/2026-04-20-external-validation-design.md` at `78d4cba`). The spec is preserved verbatim; the CITATION.md files document the corrected references with a cross-link to this TEXTBOOK_AVAILABILITY.md for audit.

A v1.1 backlog entry will record this methodology lesson: "Spec citations to textbook theorem numbers should be verified against the text during writing-plans, not during execution — this tightening pass could have run one layer earlier and avoided the dagger-footnote compromise entirely."

## When to update this file

- On successful verification of Ciarlet or Trefethen (needed for Tier B), flip the relevant row and record direct quotes.
- If future edits to a cited theorem's surrounding context change the interpretation, update the quoted passage here and the cross-linking CITATION.md in the same commit.
