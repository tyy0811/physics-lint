"""Regenerate src/physics_lint/data/config_schema.json from DomainSpec.

Run this in CI whenever DomainSpec changes so the committed schema stays
in sync with the runtime definition.
"""

import json
from pathlib import Path

from physics_lint.spec import DomainSpec


def main() -> None:
    schema = DomainSpec.model_json_schema()
    out = Path(__file__).parent.parent / "src" / "physics_lint" / "data" / "config_schema.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
