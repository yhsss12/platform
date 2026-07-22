#!/usr/bin/env bash
set -euo pipefail

ISAACSIM_ROOT="${1:-}"
if [ -z "$ISAACSIM_ROOT" ]; then
  echo "Usage: bash scripts/run_official_pick_place.sh /path/to/isaacsim"
  exit 1
fi

"$ISAACSIM_ROOT/python.sh" "$(dirname "$0")/../expert/official_franka_pick_place_adapter.py" --test --output-dir "$(pwd)/runs/isaacsim_franka_pick_place"
