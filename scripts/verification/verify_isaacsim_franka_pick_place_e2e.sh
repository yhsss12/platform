#!/usr/bin/env bash
# Isaac Sim / Franka Pick and Place real-environment E2E acceptance.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT_DIR/.env"
  set +a
fi

export TERM="${TERM:-xterm}"

JOB_ID="verify_isaacsim_franka_pick_place"
JOB_DIR="$ROOT_DIR/runs/data_generation/jobs/$JOB_ID"
EPISODE_ID="ep_000001"
ADAPTER_REL="integrations/IsaacSimFrankaPickPlace/expert/official_franka_pick_place_adapter.py"
ADAPTER_PATH="$ROOT_DIR/$ADAPTER_REL"
VIDEO_PATH="$JOB_DIR/videos/${EPISODE_ID}.mp4"
PREVIEW_PATH="$JOB_DIR/videos/${EPISODE_ID}_preview.png"
COLLECT_PY="$ROOT_DIR/scripts/verification/verify_isaacsim_franka_pick_place_collect.py"
DIAGNOSE_PY="$ROOT_DIR/scripts/diagnostics/diagnose_isaac_runtime.py"
DIAG_JSON="$JOB_DIR/diagnose_runtime.json"

mkdir -p "$JOB_DIR/videos" "$JOB_DIR/episodes/$EPISODE_ID" "$JOB_DIR/logs" "$JOB_DIR/metadata" "$JOB_DIR/results"

write_skipped_acceptance() {
  local reason="$1"
  python3 "$COLLECT_PY" \
    --job-dir "$JOB_DIR" \
    --write-skipped \
    --skip-reason "$reason" \
    --diagnose-json "$DIAG_JSON"
}

if [[ ! -f "$ADAPTER_PATH" ]]; then
  echo "[verify] missing adapter: $ADAPTER_PATH"
  exit 1
fi

echo "[verify] diagnosing Isaac Lab / Isaac Sim runtime..."
python3 "$DIAGNOSE_PY" --json > "$DIAG_JSON"

DIAGNOSIS="$(python3 - <<'PY' "$DIAG_JSON"
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
print(report.get("diagnosis", "runtime_not_detected"))
PY
)"

SKIP_REASON="$(python3 - <<'PY' "$DIAG_JSON"
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
print(report.get("skip_reason") or "")
PY
)"

ISAAC_LAB_AVAILABLE="$(python3 - <<'PY' "$DIAG_JSON"
import json, sys
print("true" if json.load(open(sys.argv[1], encoding="utf-8")).get("isaac_lab_runtime_available") else "false")
PY
)"

FRANKA_AVAILABLE="$(python3 - <<'PY' "$DIAG_JSON"
import json, sys
print("true" if json.load(open(sys.argv[1], encoding="utf-8")).get("can_import_franka_pick_place") else "false")
PY
)"

echo "[verify] diagnosis=$DIAGNOSIS isaac_lab_runtime_available=$ISAAC_LAB_AVAILABLE franka_pick_place=$FRANKA_AVAILABLE"
python3 - <<'PY' "$DIAG_JSON"
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
runner = report.get("recommended_runner") or {}
print("[verify] recommended_runner:", runner.get("label") or "(none)")
if report.get("franka_pick_place_import_path"):
    print("[verify] franka_pick_place_import_path:", report["franka_pick_place_import_path"])
PY

if [[ "$DIAGNOSIS" != "ready" ]]; then
  echo "[verify] SKIPPED: ${SKIP_REASON:-E2E acceptance skipped.}"
  write_skipped_acceptance "${SKIP_REASON:-E2E acceptance skipped.}"
  if [[ "$DIAGNOSIS" == "controller_unavailable" ]]; then
    exit 3
  fi
  exit 2
fi

echo "[verify] running official FrankaPickPlace adapter via diagnosed runner..."
python3 "$DIAGNOSE_PY" --exec "$ADAPTER_PATH" \
  --output-dir "$JOB_DIR" \
  --video-path "$VIDEO_PATH" \
  --episode-id "$EPISODE_ID" \
  --seed 0

echo "[verify] organizing platform job artifacts..."
python3 "$COLLECT_PY" --job-dir "$JOB_DIR" --episode-id "$EPISODE_ID" --diagnose-json "$DIAG_JSON"

echo "[verify] extracting preview frame and writing acceptance record..."
set +e
python3 "$COLLECT_PY" \
  --job-dir "$JOB_DIR" \
  --episode-id "$EPISODE_ID" \
  --extract-preview \
  --write-acceptance \
  --diagnose-json "$DIAG_JSON"
collect_status=$?
set -e

required_files=(
  "$JOB_DIR/status.json"
  "$JOB_DIR/dataset_manifest.json"
  "$JOB_DIR/results/aggregate_metrics.json"
  "$JOB_DIR/episodes/$EPISODE_ID/episode_manifest.json"
  "$JOB_DIR/episodes/$EPISODE_ID/metrics.json"
  "$JOB_DIR/episodes/$EPISODE_ID/trajectory.json"
  "$VIDEO_PATH"
  "$JOB_DIR/ACCEPTANCE.md"
)

echo "[verify] checking required artifacts..."
missing=0
for file_path in "${required_files[@]}"; do
  if [[ -f "$file_path" ]]; then
    echo "  [OK] $file_path"
  else
    echo "  [MISSING] $file_path"
    missing=1
  fi
done

if [[ -f "$PREVIEW_PATH" ]]; then
  echo "  [OK] $PREVIEW_PATH"
else
  echo "  [WARN] preview not generated: $PREVIEW_PATH"
fi

if [[ "$missing" -ne 0 || "$collect_status" -ne 0 ]]; then
  echo "[verify] REJECTED: E2E acceptance failed."
  exit 1
fi

echo "[verify] ACCEPTED: E2E acceptance passed."
echo "[verify] acceptance record: $JOB_DIR/ACCEPTANCE.md"
exit 0
