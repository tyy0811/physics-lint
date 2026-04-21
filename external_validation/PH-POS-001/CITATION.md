# PH-POS-001 - Evans positivity for Poisson and heat

## Citation

- **Paper:** Evans, *Partial Differential Equations*.
- **Venue:** AMS Graduate Studies in Mathematics, Vol. 19, 2nd ed. 2010.
- **ISBN:** 978-0-8218-4974-3.
- **Sections:**
  - Section 2.2.3 Theorem 4 (strong maximum principle for harmonic functions)
    together with its "Positivity" corollary on book p. 27 ("if U is
    connected and u satisfies `{-Lap u = 0 in U, u = g on dU}` with
    g >= 0, then u is positive everywhere in U if g is positive somewhere
    on dU"). Consumed here for the Poisson case with f >= 0, homogeneous
    Dirichlet.
  - Section 2.3.3 Theorem 4 (strong maximum principle for the heat
    equation): `max_{Ubar_T} u = max_{Gamma_T} u`, where Gamma_T is the
    parabolic boundary.
- **Cross-reference:** Protter-Weinberger, *Maximum Principles in
  Differential Equations*, Dover 1999, ISBN 978-0-486-41302-3, Chapters 2
  and 3.
- **Pinned value:** verdict-based - PASS on both analytical fixtures, FAIL
  on the injected negative spike.
- **Verification date:** 2026-04-20.
- **Verification protocol:** analytical derivation. For Poisson:
  `u = x(1-x)y(1-y)` is factor-wise non-negative on `[0,1]^2` and
  `-Lap u = 2[y(1-y) + x(1-x)] >= 0` satisfies the Positivity corollary's
  hypothesis (f >= 0 on the interior, g = 0 on dU; the corollary gives
  u >= 0 throughout). For heat:
  `u(x,y,t) = 1 + 0.5 * exp(-8 pi^2 t) sin(2 pi x) sin(2 pi y)`
  has `u >= 0.5` at all `t` (since `|sin * sin| <= 1` and the constant
  offset is 1.0); this is consistent with the heat max principle's
  positivity-preservation corollary (max on the parabolic boundary, with
  non-negative initial and boundary data).

Evans theorem numbers verified at theorem-level precision against the text
(AGH Krakow mirror, pymupdf + Claude vision pass on 2026-04-20) and recorded
verbatim in `../_harness/TEXTBOOK_AVAILABILITY.md`. The Rev 1.6 spec's
attributions of "Section 2.2.4 Theorem 13" (Poisson) and "Section 2.3.3
Theorem 8" (heat) are factually wrong in Evans - Theorem 13 in Section 2.2.4
is the symmetry of the Green's function, and Evans restarts theorem numbering
per section so "Theorem 8" in Section 2.3.3 does not exist. Both corrections
land in this per-rule CITATION.md only; the committed design spec is
preserved verbatim at commit 78d4cba.

## Pre-execution audit

See `AUDIT.md` in this directory. PH-POS-001 is a pure discrete-predicate
rule (pointwise `u.min() >= floor` post-BC-gate); the enumerate-the-splits
audit was run per `feedback_deviation_pattern_escalation.md` after the
n=3 escalation on PH-RES-001 and came back clean.

## Test design

- **Poisson fixture:** `u(x,y) = x(1-x)y(1-y)` on unit square, 64x64.
- **Heat fixture:** `u(x,y,t) = 1 + 0.5 * exp(-8 pi^2 t) sin(2 pi x) sin(2 pi y)`,
  periodic, at `t in {0, 0.02, 0.05, 0.1}`, 64x64. Axis convention: time
  LAST (`grid_shape=[64, 64, 4]`, `h=(H, H, 0.02)`). diffusivity=1.0.
- **Negative control:** Poisson fixture with a `-0.8` patch injected on a
  5x5 interior region.

## Acceptance criteria

- Poisson polynomial PASSES; `min(u) >= 0` on every grid point.
- Heat fixture PASSES at every timestep.
- Injected-spike negative control FAILS with `raw_value < 0`.
- Wall-time < 10 s on CPU.
