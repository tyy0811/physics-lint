# PH-POS-002 — Evans weak maximum principle

## Citation

- **Paper:** Evans, *Partial Differential Equations*.
- **Venue:** AMS Graduate Studies in Mathematics, Vol. 19, 2nd ed. 2010.
- **ISBN:** 978-0-8218-4974-3.
- **Section:** §2.2.3.
- **Artifact:** Theorem 4 (strong maximum principle for harmonic functions — the weak statement `max_Ū u = max_∂U u` is assertion (i)).
- **Cross-reference:** Protter–Weinberger, *Maximum Principles in Differential Equations*, Dover 1999, ISBN 978-0-486-41302-3, Chapter 2 §1 Theorem 1.
- **Pinned value:** verdict-based — the rule must report PASS on three harmonic polynomials (`x²−y²`, `xy`, `x³−3xy²`) and FAIL on a non-harmonic control (injected interior spike with zero boundary).
- **Verification date:** 2026-04-20.
- **Verification protocol:** analytical derivation from the cited theorem; theorem number + quote verified against Evans 2010 (AGH Kraków mirror, pymupdf → Claude vision, Task 0 Step 17). See `../_harness/TEXTBOOK_AVAILABILITY.md`.

## Test design

- **Fixtures:** three harmonic polynomials on `[0,1]²`: `u(x,y) = x²−y²`, `u(x,y) = xy`, `u(x,y) = x³−3xy²` (real part of `(x+iy)³`).
- **Grid:** 64 × 64.
- **Verdict criterion (from the rule):** the rule's current implementation flags any positive `overshoot = max(bmin−min(u), max(u)−bmax)` above a float64 floor of `1e-10`. All three harmonic polynomials are exactly representable on the grid (corner extrema match boundary extrema), so overshoot is 0 and PASS is mechanically secure.
- **Negative control:** `x²+y²` is the spec's pedagogical symbol for a non-harmonic test field, but on `[0,1]²` its extrema also live on the boundary, so the value-based rule would still PASS. The operational negative control mirrors `tests/rules/test_ph_pos_002.py::test_ph_pos_002_interior_overshoot_fails`: `u = 0` on boundary with a spike of `5.0` at the centre cell, which exercises the "interior extremum beyond boundary extrema" branch.

## Acceptance criteria

- Three harmonic polynomials produce `status == "PASS"`.
- Negative control produces `status == "FAIL"` with `raw_value > 0`.
- Wall-time < 5 s on CPU.
