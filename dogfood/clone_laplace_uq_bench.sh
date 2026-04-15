#!/usr/bin/env bash
# Clone the laplace-uq-bench repo under dogfood/laplace-uq-bench and probe
# the checkpoint layout. Invoked as the first step of the Week-2 dogfood
# discovery per the V1 Week-2 plan Task 8.
#
# Exit codes:
#   0  clone succeeded; at least one checkpoint loaded cleanly
#   1  no checkpoints found, or all checkpoint loads failed — caller
#      should invoke fallback D (train 2-3 small surrogates inline)

set -euo pipefail

DOGFOOD_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_URL="https://github.com/tyy0811/laplace-uq-bench.git"
CLONE_DIR="$DOGFOOD_DIR/laplace-uq-bench"

if [ ! -d "$CLONE_DIR" ]; then
    echo "Cloning laplace-uq-bench into $CLONE_DIR"
    git clone --depth 1 "$REPO_URL" "$CLONE_DIR"
else
    echo "Repo already cloned at $CLONE_DIR (not pulling; pin to commit)"
fi

echo
echo "Repo layout summary:"
(cd "$CLONE_DIR" && ls -la 2>/dev/null | head -30)

echo
echo "Searching for checkpoint files (*.pt, *.pth, *.ckpt, *.safetensors):"
found=0
while IFS= read -r -d '' ckpt; do
    rel="${ckpt#$CLONE_DIR/}"
    echo "  $rel"
    found=1
done < <(find "$CLONE_DIR" \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.safetensors" \) -print0 2>/dev/null)

if [ "$found" -eq 0 ]; then
    echo "  (none)"
    echo
    echo "No checkpoint files found — caller should invoke fallback D."
    exit 1
fi

echo
echo "Probing each checkpoint with torch.load(weights_only=False):"
PY_BIN="${PYTHON:-python}"
status=0
while IFS= read -r -d '' ckpt; do
    rel="${ckpt#$CLONE_DIR/}"
    "$PY_BIN" - "$ckpt" <<'PY' || status=1
import sys
import torch

path = sys.argv[1]
try:
    state = torch.load(path, map_location="cpu", weights_only=False)
except Exception as exc:
    print(f"  FAIL {path}: {exc}")
    sys.exit(1)
if isinstance(state, dict):
    keys = list(state.keys())
    print(f"  OK   {path}: dict with {len(keys)} keys ({keys[:5]}...)")
elif hasattr(state, "state_dict"):
    print(f"  OK   {path}: module-like object of type {type(state).__name__}")
else:
    print(f"  OK?  {path}: unknown type {type(state).__name__}")
PY
done < <(find "$CLONE_DIR" \( -name "*.pt" -o -name "*.pth" -o -name "*.ckpt" -o -name "*.safetensors" \) -print0)

if [ "$status" -ne 0 ]; then
    echo
    echo "At least one checkpoint failed to load. Caller should invoke fallback D."
    exit 1
fi

echo
echo "All checkpoints probed cleanly."
