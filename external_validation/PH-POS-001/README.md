# PH-POS-001 external-validation anchor

Evans positivity for Poisson (§2.2.3 Theorem 4 + Positivity corollary, p. 27)
and heat (§2.3.3 Theorem 4). See `CITATION.md` for full provenance and
`AUDIT.md` for the pre-execution enumerate-the-splits verification.

## Run

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-POS-001/ -v
```

Expected: 4 passed in < 10 s.
