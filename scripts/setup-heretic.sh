#!/usr/bin/env bash
set -euo pipefail

EXPECTED_COMMIT="6ea3b8d778d047b4b3b7c5b843e21c5bea98ee8d"
SUBMODULE_PATH="third_party/heretic"

echo "Initializing pinned Heretic submodule..."
git submodule update --init --recursive -- "$SUBMODULE_PATH"

ACTUAL_COMMIT="$(git -C "$SUBMODULE_PATH" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "Heretic commit mismatch. Expected $EXPECTED_COMMIT, got $ACTUAL_COMMIT" >&2
  exit 1
fi

echo "Heretic pin verified: $ACTUAL_COMMIT"
echo "Upstream environment requirements are defined inside third_party/heretic."
