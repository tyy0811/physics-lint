# PH-SYM-002 external-validation anchor

Run:

```bash
source .venv/bin/activate && pytest --import-mode=importlib external_validation/PH-SYM-002/ -v
```

Expected: 4 passed in < 15 s (two fixture-sanity tests + one rule PASS + one
rule WARN/FAIL).
