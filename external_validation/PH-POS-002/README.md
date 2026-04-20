# PH-POS-002 external-validation anchor

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-POS-002/ -v
```

Expected: 4 passed in < 5 s.

See `CITATION.md` for the external reference, `test_anchor.py` for the
test harness, and `../README.md` for the full-program context.
