#!/usr/bin/env bash
# Clone the laplace-uq-bench repo at a pinned commit under
# dogfood/laplace-uq-bench/. Invoked as the first step of the Week-2
# dogfood discovery per the V1 Week-2 plan Task 8.
#
# Security posture: this clone is a trust boundary. Fallback D' uses
# ONLY the repo's pure-numpy LaplaceSolver and boundary sampler
# (src/diffphys/pde/{laplace,boundary}.py); it never loads PyTorch
# checkpoints. To keep the threat surface minimal we:
#
# 1. Pin to a specific upstream commit SHA and refuse to proceed if
#    the clone is at a different commit.
# 2. Do NOT call `torch.load(..., weights_only=False)` on anything in
#    the tree. Any future probe of an untrusted *.pt file MUST use
#    `weights_only=True` (and ideally a checksum check on top).
#
# Reviewer note (Codex adversarial review, Week 2 Day 5): the previous
# version of this script called `torch.load(weights_only=False)` on
# every checkpoint found under the clone, which is arbitrary-code-
# execution if the repo or any checkpoint path were compromised.
# Removed.

set -euo pipefail

DOGFOOD_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_URL="https://github.com/tyy0811/laplace-uq-bench.git"
CLONE_DIR="$DOGFOOD_DIR/laplace-uq-bench"

# Pinned upstream commit. Bumping this value is a conscious security
# decision — verify the new SHA belongs to the expected history and
# that only the files we import (src/diffphys/pde/*.py) changed in a
# way that does not expand the trust surface.
PINNED_SHA="4c2113a5b51cfca38cbd609be0739ca43757e93f"

if [ ! -d "$CLONE_DIR" ]; then
    echo "Cloning laplace-uq-bench into $CLONE_DIR"
    git clone "$REPO_URL" "$CLONE_DIR"
    git -C "$CLONE_DIR" checkout --detach "$PINNED_SHA"
else
    echo "Repo already cloned at $CLONE_DIR"
fi

actual_sha="$(git -C "$CLONE_DIR" rev-parse HEAD)"
if [ "$actual_sha" != "$PINNED_SHA" ]; then
    echo "ERROR: $CLONE_DIR is at $actual_sha but the pinned SHA is $PINNED_SHA."
    echo "Either reset the clone or bump PINNED_SHA in this script after review."
    exit 2
fi
echo "Verified HEAD == $PINNED_SHA"

echo
echo "Repo layout summary:"
(cd "$CLONE_DIR" && ls -la 2>/dev/null | head -30)

echo
echo "Note: this script does NOT probe PyTorch checkpoints. Fallback D'"
echo "(see dogfood/laplace_uq_bench/README.md) uses only the repo's"
echo "pure-numpy LaplaceSolver and BC sampler. Loading untrusted .pt"
echo "files with weights_only=False is prohibited by this script's"
echo "security posture; see the header comment."
