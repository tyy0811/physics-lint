# SARIF schema reference

physics-lint emits SARIF v2.1.0 results in **scalar mode** (one result per
non-PASS rule firing). This page documents the scalar schema and the
fields' meanings; for the full SARIF v2.1.0 specification, see the
[OASIS spec](https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html).

## What is covered by semver

The scalar schema is part of the public surface (see
[Stability policy](stability.md)). Specifically: the `result` schema,
`ruleId`, `level`, and the `properties` block (`raw_value`,
`violation_ratio`, `mode`, `reason`). Internal SARIF fields (`tool`
metadata, `runs[].invocations`) are not part of the covered surface.

## Result schema

Each non-PASS rule firing emits one `result` object:

```json
{
  "ruleId": "PH-POS-002",
  "level": "error",
  "message": {
    "text": "Maximum principle violation: interior extremum exceeds boundary by 0.078 in a Dirichlet-homogeneous problem."
  },
  "locations": [
    {
      "physicalLocation": {
        "artifactLocation": {
          "uri": "models/fno_adapter.py"
        }
      },
      "logicalLocations": [
        {
          "fullyQualifiedName": "physics_lint.rules.PH-POS-002",
          "kind": "rule"
        }
      ]
    }
  ],
  "partialFingerprints": {
    "ruleId/v1": "PH-POS-002"
  },
  "properties": {
    "raw_value": 0.078,
    "violation_ratio": 1.78,
    "mode": "absolute",
    "reason": "interior extremum exceeds Dirichlet boundary by 0.078 (threshold: 0.044 absolute)"
  }
}
```

## Field reference

| Field | Type | Meaning |
|---|---|---|
| `ruleId` | string | Stable rule identifier `PH-<CATEGORY>-<NNN>`. See [rule catalog](rules/index.md) |
| `level` | enum | `error`, `warning`, or `note`. Maps from the rule's `default_severity` (modifiable via config) |
| `message.text` | string | Human-readable summary including the raw value and threshold |
| `locations[].physicalLocation` | object | Path to the model artifact being linted |
| `locations[].logicalLocations` | array | The rule's fully-qualified name, used by GitHub code scanning to group results |
| `partialFingerprints` | object | Used by GitHub code scanning to deduplicate results across runs |
| `properties.raw_value` | number | The rule's emitted quantity (e.g., L²-norm of BC residual, equivariance deviation) |
| `properties.violation_ratio` | number | `raw_value / threshold`; >1 indicates violation |
| `properties.mode` | enum | `absolute` or `relative` (rule-dependent; documented per-rule on the rule's page) |
| `properties.reason` | string | A short prose explanation of which threshold was crossed and how |

## Status values

A rule can produce one of four `status` values internally; the SARIF
emission maps these as follows:

| Internal status | SARIF emission |
|---|---|
| `PASS` | No result emitted |
| `APPROXIMATE` | `level: warning` |
| `FAIL` | `level: error` (or `warning` / `note` per config) |
| `SKIP` | `level: note`, `properties.mode: "skip"`, `properties.reason` explains why the rule did not run on this input |

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
