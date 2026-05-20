# SARIF schema reference

physics-lint emits SARIF v2.1.0 results in **scalar mode** (one result per
non-PASS rule firing). This page documents the scalar schema and the
fields' meanings; for the full SARIF v2.1.0 specification, see the
[OASIS spec](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html).

## What is covered by semver

The scalar schema is part of the public surface (see
[Stability policy](stability.md)). Specifically: the `result` object
shape, `ruleId`, `level`, and the `properties` block (`raw_value`,
`violation_ratio`, `mode`, `doc_url`, `location_mode`). Internal SARIF
fields outside the result-and-notifications spine — for example, the
exact contents of `tool.driver.rules` descriptors — are not part of
the covered surface.

## Result schema

Each non-PASS rule firing emits one `result` object:

```json
{
  "ruleId": "PH-POS-002",
  "level": "error",
  "message": {
    "text": "Maximum principle violation; raw=7.800e-02; ratio=1.78; mode=absolute"
  },
  "locations": [
    {
      "physicalLocation": {
        "artifactLocation": {
          "uri": "models/fno_adapter.py"
        }
      }
    }
  ],
  "properties": {
    "violation_ratio": 1.78,
    "raw_value": 0.078,
    "doc_url": "https://physics-lint.readthedocs.io/rules/PH-POS-002",
    "mode": "absolute",
    "location_mode": "artifact-only"
  }
}
```

When the adapter provides source mapping (a Python file with a known
PDE/BC line), `locations[0].physicalLocation` swaps `artifactLocation.uri`
to the source file and adds a `region` with `startLine` / `endLine`;
`properties.location_mode` becomes `"source-mapped"` and
`properties.model_artifact` records the original target path.

## Field reference

| Field | Type | Meaning |
|---|---|---|
| `ruleId` | string | Stable rule identifier `PH-<CATEGORY>-<NNN>`. See [rule catalog](rules/index.md) |
| `level` | enum | `error`, `warning`, or `note`. Maps from the rule's `default_severity` (modifiable via config) |
| `message.text` | string | Human-readable summary including the rule name, raw value, ratio, mode, and reason (semicolon-joined) |
| `locations[].physicalLocation.artifactLocation.uri` | string | Path to the model artifact, or the source file in source-mapped mode |
| `locations[].physicalLocation.region` | object | `{startLine, endLine}` — present only in source-mapped mode |
| `properties.raw_value` | number | The rule's emitted quantity (e.g., $L^2$-norm of BC residual, equivariance deviation) |
| `properties.violation_ratio` | number | `raw_value / threshold`; >1 indicates violation |
| `properties.mode` | enum | `absolute` or `relative` (rule-dependent; documented per-rule on the rule's page) |
| `properties.doc_url` | string | URL to the rule's documentation |
| `properties.location_mode` | enum | `artifact-only` or `source-mapped` |
| `properties.model_artifact` | string | Source-mapped mode only: the original target the user invoked physics-lint on |

## Status values

A rule can produce one of four `status` values internally; the SARIF
emission maps these as follows:

| Internal status | SARIF emission |
|---|---|
| `PASS` | **No result emitted.** PASS rules do not appear in `runs[].results` — GitHub code scanning treats every result as an alert regardless of `level`, so a PASS with `level: error` would surface as a false positive. PASS is visible in `text` / `json` output. |
| `APPROXIMATE` | `level: warning` in `runs[].results` |
| `FAIL` | `level: error` in `runs[].results` (or `warning` / `note` per the rule's `default_severity`) |
| `SKIPPED` | **Routed to `runs[0].invocations[0].toolExecutionNotifications`**, not `results`. Each notification has `level: note`, a `message.text` of the form `<rule_id> skipped: <reason>`, and a `descriptor: {id: rule_id}`. This keeps SKIP rows out of the Security tab (where they would be noise) while still recording them for diagnostic tooling. |

## Categories

GitHub code scanning groups results by `category`. The CLI accepts
`--category <name>` (e.g., `physics-lint-fno`) so distinct model runs
emit distinguishable groups in the Security tab. See the
[GitHub Action](action.md) for the matrix-style pattern.

## Example: SARIF in CI

```yaml
- run: |
    physics-lint check models/fno.py \
      --format sarif \
      --category physics-lint-fno \
      --output physics-lint-fno.sarif

- if: always()
  uses: github/codeql-action/upload-sarif@v4
  with:
    sarif_file: physics-lint-fno.sarif
    category: physics-lint-fno
```

`if: always()` is important: the SARIF upload runs even if the previous
step exited non-zero (which happens when error-severity rules fire).
