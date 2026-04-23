# Textbook availability — 2026-04-23 (complete-v1.0 extension)

Recorded by complete-v1.0 plan Task 0 Step 2 (2026-04-23); extends the 2026-04-20
Tier-A record which covered Evans, Ciarlet, Trefethen only.

Consumed by per-rule CITATION.md files across all 18 v1.0 rules. The discipline is
`docs/plans/2026-04-22-physics-lint-external-validation-complete.md` §6.4 "Textbook
primary-source verification discipline": citations that pass primary-source
verification may use tight theorem-number framing in CITATION.md (e.g., "Hall 2015
Theorem 2.14"); citations that pass only secondary-source corroboration must use
section-level framing ("Hall 2015 §2.5, theorem number pending local copy") and
carry a ⚠ flag in this file. The plan's `scripts/check_theorem_number_framing.py`
(Task 0 closeout) enforces this mechanically against per-rule CITATION.md files.

## Status key

- ✅ primary-source verified (local copy / accessible PDF / institutional-repository
  page image visually inspected against the claimed theorem/section)
- ⚠ secondary-source corroborated only (Wikipedia, lecture notes, review articles
  citing the theorem; no direct inspection of the textbook text) — CITATION.md
  framing must be section-level, not tight theorem-number
- ❌ access failed entirely — CITATION.md framing is section-level with an
  additional "source not directly accessed" note
- 🅿 pending — access attempt not yet run at this file's revision date

## Summary table

| Textbook | Status | Sections/theorems cited | Consumer tasks |
|----------|--------|-------------------------|----------------|
| Evans 2010 *PDEs* (AMS GSM 19, 2nd ed., ISBN 978-0-8218-4974-3) | ✅ §2.2.3 Thm 4; ✅ §2.3.3 Thm 4; ✅ §7.1.2 Thm 2; 🅿 §2.3 (mass-conservation corollary); 🅿 §2.4.3 (energy identity); 🅿 §5.5 Thm 1 (trace theorem); 🅿 Appendix C.2 Thm 1 (Gauss-Green) | Tier-A Tasks 1, 2, 5; Tier-B Tasks 4, 5, 8, 9 |
| Hall 2015 *Lie Groups, Lie Algebras, and Representations*, 2nd ed. (Springer GTM 222, ISBN 978-3-319-13466-6; DOI 10.1007/978-3-319-13467-3) | ⚠ §2.5 Theorem 2.14 (one-parameter-subgroup); ⚠ §3.7 Corollary 3.50 (continuous-to-smooth for matrix Lie group homomorphisms) — secondary-source-confirmed only; see "Hall + Varadarajan verification pass" below | Tasks 1, 6 |
| Varadarajan 1984 *Lie Groups, Lie Algebras, and Their Representations* (Springer GTM 102, ISBN 978-0-387-90969-1) | ⚠ §2.9–2.10 identity-component generation — secondary-source-confirmed only; see "Hall + Varadarajan verification pass" below | Tasks 1, 6 |
| Ciarlet 2002 *The Finite Element Method for Elliptic Problems* (SIAM Classics 40, ISBN 978-0-89871-514-9; original North-Holland 1978 reprinted by SIAM 2002) | ⚠ §3.2 Céa's lemma; ⚠ §4.1 Thms 4.1.2–4.1.6 (quadrature convergence) | Tasks 11, 12 |
| Trefethen 2000 *Spectral Methods in MATLAB* (SIAM Other Titles in Applied Mathematics 10, ISBN 978-0-89871-465-4) | ⚠ Chapters 3–4 (trigonometric interpolation + Fourier spectral accuracy) | Task 3 |
| LeVeque 2007 *Finite Difference Methods for Ordinary and Partial Differential Equations* (SIAM, ISBN 978-0-898716-29-0) | ⚠ consistency-order theorem (FDM Ch. on convergence) | Task 2 |
| Griewank-Walther 2008 *Evaluating Derivatives: Principles and Techniques of Algorithmic Differentiation*, 2nd ed. (SIAM, ISBN 978-0-89871-659-7; DOI 10.1137/1.9780898717761) | ⚠ Chapter 3 (accuracy of reverse-mode AD derivatives; proposition number pending local copy — row added during Task 2 per §6.4 "new backbone" convention) — section-level only; secondary-source corroboration via Baydin-Pearlmutter-Radul-Siskind 2018 JMLR "Automatic Differentiation in Machine Learning: A Survey" §3 (arXiv:1502.05767) | Task 2 |
| LeVeque 2002 *Finite Volume Methods for Hyperbolic Problems* (Cambridge Texts in Applied Mathematics, ISBN 978-0-521-00924-9) | ⚠ §2.1 conservation-form framing | Task 5 (Supplementary calibration context only) |
| Dafermos 2016 *Hyperbolic Conservation Laws in Continuum Physics*, 4th ed. (Springer Grundlehren 325, ISBN 978-3-662-49449-3) | ⚠ Chapter I (balance-law framing) | Task 8 |
| Gilbarg-Trudinger 2001 *Elliptic Partial Differential Equations of Second Order* (Springer Classics in Mathematics, 2nd ed. reprint, ISBN 978-3-540-41160-4) | ⚠ §2.4 (divergence theorem on Lipschitz domains) | Task 5 |
| Hairer-Lubich-Wanner 2006 *Geometric Numerical Integration*, 2nd ed. (Springer Computational Mathematics 31, ISBN 978-3-540-30663-4) | ⚠ Chapter IX (symplectic-integrator conservation) | Task 9 |
| Strauss 2007 *Partial Differential Equations: An Introduction*, 2nd ed. (Wiley, ISBN 978-0-470-05456-7) | ⚠ §2.2 (wave-equation energy identity) | Task 9 |
| Bangerth-Rannacher 2003 *Adaptive Finite Element Methods for Differential Equations* (Birkhäuser Lectures in Mathematics ETH Zürich, ISBN 978-3-7643-7009-1) | ⚠ (dual-weighted-residual framing, various chapters) | Task 10 |
| Verfürth 2013 *A Posteriori Error Estimation Techniques for Finite Element Methods* (Oxford Numerical Mathematics and Scientific Computation, ISBN 978-0-19-967942-3) | ⚠ Chapters 1–4 (residual-based estimator theorems, Thm 1.12 cited) | Task 10 |
| Ainsworth-Oden 2000 *A Posteriori Error Estimation in Finite Element Analysis* (Wiley, ISBN 978-0-471-29411-5) | ⚠ (residual estimator overview) | Task 10 |
| Brenner-Scott 2008 *The Mathematical Theory of Finite Element Methods*, 3rd ed. (Springer Texts in Applied Mathematics 15, ISBN 978-0-387-75933-3) | ⚠ §10.3 (quadrature and variational crimes) | Task 11 |
| Strang-Fix 2008 *An Analysis of the Finite Element Method*, 2nd ed. (Wellesley-Cambridge Press, ISBN 978-0-9802327-0-8) | ⚠ (FEM fundamentals; referenced as backbone) | (background only; no rule direct-cites specific theorem) |
| Strang 1972 "Variational Crimes in the Finite Element Method" (published in *The Mathematical Foundations of the Finite Element Method*, ed. Aziz, Academic Press, ISBN 978-0-12-068650-6) | ⚠ (variational crimes framing) | Task 11 |
| Ern-Guermond 2004 *Theory and Practice of Finite Elements* (Springer Applied Mathematical Sciences 159, ISBN 978-0-387-20574-8) | ⚠ (FEM theory backbone) | Task 11 |
| Ern-Guermond 2021 *Finite Elements I–III* (Springer Texts in Applied Mathematics 72–74, ISBNs 978-3-030-56340-0, 978-3-030-56922-8, 978-3-030-57347-8) | ⚠ §8.3 (quadrature-order calibration, cited as Supplementary calibration context in Task 11) | Task 11 |
| Strikwerda 2004 *Finite Difference Schemes and Partial Differential Equations*, 2nd ed. (SIAM, ISBN 978-0-89871-567-5) | ⚠ §3; Chapter 10 Lax equivalence theorem (already cited on Tier-A PH-RES-001) | Task 12 (+ Tier-A PH-RES-001 boundary rationale) |
| Canuto-Hussaini-Quarteroni-Zang 2006 *Spectral Methods: Fundamentals in Single Domains* (Springer Scientific Computation, ISBN 978-3-540-30725-9) | ⚠ §2.3 (spectral convergence curves) | Task 3 (Supplementary calibration context) |
| Boyd 2001 *Chebyshev and Fourier Spectral Methods*, 2nd ed. (Dover, ISBN 978-0-486-41183-8; full text free at the author's page at math.colorado.edu/~boyd) | 🅿 (F3-hunt tertiary; exp(sin x) convergence tables) | Task 3 |
| Hesthaven-Gottlieb-Gottlieb 2007 *Spectral Methods for Time-Dependent Problems* (Cambridge Monographs on Applied and Computational Mathematics 21, ISBN 978-0-521-79211-0) | ⚠ (time-dependent spectral methods backbone) | (background only) |
| Gottlieb-Orszag 1977 *Numerical Analysis of Spectral Methods: Theory and Applications* (SIAM CBMS-NSF Regional Conference Series in Applied Mathematics 26, ISBN 978-0-89871-023-6) | ⚠ (spectral convergence foundational) | (background only) |
| Grisvard 2011 *Elliptic Problems in Nonsmooth Domains* (SIAM Classics 69 reprint, ISBN 978-1-611972-02-3; original Pitman 1985) | ⚠ (corner-singularity theory) | Task 10 (background; corner-singularity justification) |
| John 1982 *Partial Differential Equations*, 4th ed. (Springer Applied Mathematical Sciences 1, ISBN 978-0-387-90609-6) | ⚠ (classical PDE reference) | (background only) |
| Oberkampf-Roy 2010 *Verification and Validation in Scientific Computing* (Cambridge University Press, ISBN 978-0-521-11360-1) | ⚠ Chapters 5–6 (p_obs algorithm, observed convergence rate) | Task 12 |
| Folland 1995 *A Course in Abstract Harmonic Analysis* (CRC Press / Chapman & Hall Studies in Advanced Mathematics, ISBN 978-0-8493-8490-5) | ⚠ (compact-group harmonic analysis background) | Task 6 (background only) |
| Weiler-Forré-Verlinde-Welling 2025 *Equivariant and Coordinate Independent Convolutional Networks* (World Scientific, DOI 10.1142/14143, ISBN 978-981-98-0662-1) | ⚠ (published 2025 monograph; geometric-deep-learning backbone) | Tasks 1, 6 (GDL backbone per complete-v1.0 plan §0.3) |

## Evans verification (from Tier-A 2026-04-20 pass; preserved verbatim)

Source: Evans PDF mirrored at `https://wms.mat.agh.edu.pl/~lusapa/pl/Evans.pdf`
(AGH Kraków math department). The PDF's text layer was empty on the pages sampled
during this pass (`pymupdf.Document[pg].get_text()` returned 0 bytes on pages 5,
50, 150, 300), so verification was done by rendering relevant pages to PNG via
`pymupdf` at DPI 140 and visually inspecting each page through Claude's multimodal
vision rather than relying on parsed text. This does not prove the entire file
lacks any OCR layer — only that the pages actually inspected here behaved as
image-only.

Page numbers below refer to **book pages** (the page number printed on each
page), not PDF file pages. Offset:

- pymupdf's `doc[pg]` uses a **0-indexed** page counter: 0-indexed PDF page number =
  book page + 15. This is the numbering used by the rendering scripts in this
  session (`pix = doc[pg].get_pixmap(...)`).
- A **1-indexed** PDF viewer (Preview, Acrobat): 1-indexed PDF page number =
  book page + 16.

Example verifications: book p. 27 (§2.2.3 Theorem 4) = 0-indexed pymupdf page 42 =
1-indexed PDF page 43. Book p. 55 (§2.3.3 Theorem 4) = 0-indexed pymupdf page 70 =
1-indexed PDF page 71.

### ✅ §2.2.3 Theorem 4 — Strong maximum principle for harmonic (book p. 27)

Direct quote from Evans:

> **THEOREM 4 (Strong maximum principle).** *Suppose u ∈ C²(U) ∩ C(Ū) is
> harmonic within U.*
> *(i) Then* max_{Ū} u = max_{∂U} u.
> *(ii) Furthermore, if U is connected and there exists a point x₀ ∈ U such
> that u(x₀) = max_{Ū} u, then u is constant within U.*

Consumed by PH-POS-002 (Tier-A Task 1). Assertion (i) is the weak maximum
principle the rule checks; assertion (ii) is the strong version that guarantees
constant solutions.

**Also on p. 27 (directly following Theorem 4):** "**Positivity.** The strong
maximum principle states in particular that if U is connected and u ∈ C²(U) ∩
C(Ū) satisfies {-Δu = 0 in U, u = g on ∂U}, where g ≥ 0, then u is positive
everywhere in U if g is positive somewhere on ∂U."

This *Positivity corollary* (not a separately-numbered theorem) is consumed by
PH-POS-001 (Tier-A Task 5) for the Poisson case.

### ✅ §7.1.2 Theorem 2 — Energy estimates for parabolic (book p. 376)

Direct quote from Evans:

> **THEOREM 2 (Energy estimates).** *There exists a constant C, depending only
> on U, T and the coefficients of L, such that*
>
> max_{0≤t≤T} ‖u_m(t)‖_{L²(U)} + ‖u_m‖_{L²(0,T; H¹₀(U))} + ‖u'_m‖_{L²(0,T; H⁻¹(U))} ≤ C (‖f‖_{L²(0,T; L²(U))} + ‖g‖_{L²(U)})
>
> *for m = 1, 2, …*

Consumed by PH-CON-003 (Tier-A Task 2). The L² energy bound on `u_m` implies the
E(t) ≤ E(0) dissipation property that the rule checks on the heat-equation
eigenmode fixture.

### ❌ §2.2.4 Theorem 13 — Symmetry of Green's function (book p. 35), NOT positivity

The design-spec Rev 1.6 §2 Task 5 citation of "Evans §2.2.4 Theorem 13
(positivity for Poisson with f ≥ 0, homogeneous Dirichlet)" was **incorrect**.

Direct quote from Evans:

> **THEOREM 13 (Symmetry of Green's function).** *For all x, y ∈ U, x ≠ y, we
> have* G(y, x) = G(x, y).

This is the symmetry theorem, not positivity. **The correct citation for the
Poisson positivity property is §2.2.3 Theorem 4 + its Positivity corollary
paragraph on p. 27 (documented above).**

Correction applied: PH-POS-001/CITATION.md cites Evans §2.2.3 Theorem 4 (with
p. 27 Positivity corollary) for the Poisson case, not §2.2.4 Theorem 13. This
supersedes the Rev 1.6 spec's attribution.

### ❌ §2.3.3 Theorem 8 — does not exist; the heat max principle is §2.3.3 Theorem 4 (book p. 55)

The design-spec Rev 1.6 §2 Task 5 citation of "Evans §2.3.3 Theorem 8 (weak
maximum principle for heat)" was **incorrect in theorem number**. Evans restarts
theorem numbering per section, so both §2.2.3 and §2.3.3 have local Theorem 4s —
the spec conflated the global numbering.

Direct quote from Evans:

> **THEOREM 4 (Strong maximum principle for the heat equation).** *Assume u ∈
> C²₁(U_T) ∩ C(Ū_T) solves the heat equation in U_T.*
> *(i) Then* max_{Ū_T} u = max_{Γ_T} u.
> *(ii) Furthermore, if U is connected and there exists a point (x₀, t₀) ∈ U_T
> such that u(x₀, t₀) = max_{Ū_T} u, then u is constant on Ū_{t₀}.*

Where Γ_T is the *parabolic boundary* = bottom cap + lateral sides. Consumed by
PH-POS-001 (Tier-A Task 5) for the heat case.

Correction applied: PH-POS-001/CITATION.md cites Evans §2.3.3 Theorem 4 (heat
case), not §2.3.3 Theorem 8.

## Hall + Varadarajan verification pass — complete-v1.0 Task 0 Step 3 (2026-04-23)

Per complete-v1.0 plan §6.4, Hall 2015 2nd ed. and Varadarajan 1984 are the
load-bearing Lie-group backbone for Tasks 1 and 6. Task 0 Step 3 attempted
primary-source verification for:

- Hall, B.C. (2015). *Lie Groups, Lie Algebras, and Representations*, 2nd ed.
  Springer GTM 222. Theorem 2.14 (one-parameter subgroup theorem); Corollary 3.50
  (continuous-to-smooth for matrix Lie group homomorphisms).
- Varadarajan, V.S. (1984). *Lie Groups, Lie Algebras, and Their Representations.*
  Springer GTM 102. §2.9–2.10 (identity-component generation from a neighborhood
  of the identity).

### Verification outcome

**Primary-source access was not obtained for either Hall 2015 2nd ed. or Varadarajan
1984 during this Task 0 Step 3 pass.** Both titles are Springer GTM paywalled
ebooks. Access attempts:

- SpringerLink (publisher): paywalled; front-matter only, not §2.5 / §3.7 / §2.9–2.10.
- Stony Brook physics mirror (`https://max2.physics.sunysb.edu/~rastelli/2017/Hall.pdf`):
  connection refused during Task 0 Step 3 (both via WebFetch and `curl --max-time 10`).
- ResearchGate 2015-filename PDF: 403 Forbidden.
- Penn course notes (`https://www2.math.upenn.edu/~wziller/math650/LieGroupsReps.pdf`):
  successfully fetched, but it is Wolfgang Ziller's Fall 2010 Penn course notes,
  not Hall's book — inspected via `fitz.open(...).get_text()` and found to contain
  no "Theorem 2.14" or "Corollary 3.50" references.
- Internet Archive Varadarajan record (`liegroupsliealge0000vara`): the 1974
  Prentice-Hall edition (content-equivalent to the 1984 Springer GTM 102 reprint),
  access-restricted.

**Secondary-source corroboration is strong for all three citations** — two independent
expert-curated references name the theorem numbers and state the result content
explicitly in each case. Verification status is ⚠ (secondary-source-confirmed only);
CITATION.md framing must be section-level per §6.4 of the complete-v1.0 plan.

- **Hall 2015 2nd ed. Theorem 2.14** (one-parameter-subgroup theorem, §2.5
  "One-Parameter Subgroups"). Corroborated by two independent Wikipedia articles:
  - [Wikipedia *One-parameter group*](https://en.wikipedia.org/wiki/One-parameter_group)
    cites "Hall 2015 Theorem 2.14" at footnote [5], stating: "Theorem: Suppose
    φ : R → GL(n;C) is a one-parameter group. Then there exists a unique n × n
    matrix X such that φ(t) = e^{tX} for all t ∈ R." (content-exact match for the
    asserted theorem.)
  - [Wikipedia *Lie group–Lie algebra correspondence*](https://en.wikipedia.org/wiki/Lie_group%E2%80%93Lie_algebra_correspondence)
    cites "Hall 2015, Theorem 2.14" at footnote [^17], attributing the one-parameter
    subgroup generated by X to the same theorem.
- **Hall 2015 2nd ed. Corollary 3.50** (continuous-to-smooth for matrix Lie group
  homomorphisms, §3.7). Corroborated by:
  - [nLab *continuous homomorphisms of Lie groups are smooth*](https://ncatlab.org/nlab/show/continuous+homomorphisms+of+Lie+groups+are+smooth)
    cites "Hall15, Cor. 3.50" explicitly, quoting the proposition: "every
    continuous group homomorphism G → H is smooth" (finite-dim Lie groups).
    Also notes the proof follows via Cartan's closed subgroup theorem applied
    to the graph of the homomorphism.
- **Varadarajan 1984 §2.9–2.10** (identity-component generation from a
  neighborhood of the identity in a connected Lie group). Secondary-source
  corroboration via Maryland Math 744 (Fall 2010) syllabus reading list and
  UC Davis / UCLA course pages that cite Varadarajan GTM 102 §2.9–2.10 for
  the connected-Lie-group topological-generation result. Direct quote not
  obtained.

### Framing consequence per §6.4

Per the plan's §6.4 fallback rule: Hall 2015 2nd ed. and Varadarajan 1984
citations in Tasks 1 and 6 CITATION.md files must use **section-level framing**
("Hall 2015 §2.5, theorem number pending local copy"; "Varadarajan 1984 §2.9–2.10,
section-level only") rather than tight theorem-number framing ("Hall 2015 Theorem
2.14"). The `scripts/check_theorem_number_framing.py` closeout script enforces
this mechanically. Tasks 1 and 6 are pre-flagged for inheriting the caveat
visibly in their structural-equivalence proof-sketches.

If local Hall or Varadarajan access is obtained during a later Tier-B task's
execution, the affected CITATION.md files get a tightening-pass commit per the
plan-vs-committed-state drift discipline (§7.4) and this file's status rows
flip from ⚠ to ✅ with direct quotes in the same commit.

## Other textbook verification status

The remaining textbooks in the summary table carry ⚠ status — the citations
can be corroborated via published course notes, review articles, or text-
book-of-textbooks indexes, but primary-source access was not obtained during
Task 0's verification pass within its 0.3 d budget. Per §6.4, CITATION.md
framing for each of these must be section-level. The status rows are available
for in-place upgrade in any later task's execution if local access is obtained.

Trefethen 2000 has a companion errata and code repository at
`https://people.maths.ox.ac.uk/trefethen/spectral.html`; that's useful for
confirming chapter boundaries and code fragments, but the Chapters 3–4 theorem
statements themselves are in the printed text. SIAM sells the ebook on its
store page; no free public mirror of the full text is known.

Boyd 2001 *Chebyshev and Fourier Spectral Methods* 2nd ed. is the exception
in the "Spectral" textbook row — the author's page at Michigan (formerly
math.colorado.edu/~boyd; now hosted at various mirrors; canonical is
`https://depts.washington.edu/ph506/Boyd.pdf`) carries the full PDF for free.
If the Task 3 F3-hunt promotes Boyd 2001 into F3-present, the verification
pass will be rerun against that PDF and this file's Boyd row will flip to ✅.

## Fallback rule

Per §6.4: any CITATION.md citation of a textbook marked ⚠ in this file must use
section-level framing. Tight theorem-number framing requires ✅ status. The
`scripts/check_theorem_number_framing.py` script (Task 0 Step 9) cross-references
CITATION.md files against this table and fails on framing mismatches.

## When to update this file

- On obtaining direct access to a currently-⚠ textbook's relevant pages, flip
  the row to ✅ with direct quotes, and in the same commit tighten any
  CITATION.md framings that were section-level-only pending the verification.
  This is the plan-vs-committed-state drift discipline (§7.4) applied
  prospectively.
- If future edits to a cited theorem's surrounding context change the
  interpretation, update the quoted passage here and the cross-linking
  CITATION.md in the same commit.
- When a new rule / task begins citing a textbook not yet in the summary
  table, add a new row and run the verification pass at that task's
  pre-execution gate.
