#!/usr/bin/env bash
# NutAssembly PINN/P8/P9 job cleanup DRY-RUN (Round 2)
# Generated: 2026-07-08 — NO deletions performed
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

echo "========================================"
echo " NutAssembly jobs cleanup DRY-RUN"
echo " NO files will be deleted."
echo "========================================"

echo
echo "[KEEP — do not delete]"
KEEP=(
  "runtime_outputs/nut_assembly/jobs/na_gen_official_p7_50_20260703_171055_cafd"
  "runtime_outputs/nut_assembly/jobs/na_gen_20260703_175437_537b"
  "runtime_assets/models/pinn"
  "runtime_assets/mimicgen/nut_assembly/source"
)
for p in "${KEEP[@]}"; do
  if [[ -e "$p" ]]; then
    echo "  KEEP: $p ($(du -sh "$p" 2>/dev/null | cut -f1))"
  fi
done
echo "  KEEP: workspace_jobs DB rows pointing to above paths (update DB before deleting 175437 job)"

echo
echo "[REVIEW-BEFORE-DELETE]"
REVIEW=(
  "runtime_outputs/nut_assembly/jobs/na_gen_p8_pinn_20260703_181533_97c8"
)
for p in "${REVIEW[@]}"; do
  if [[ -d "$p" ]]; then
    echo "  REVIEW: $p ($(du -sh "$p" | cut -f1)) — legacy P8 input; current builder uses EAI_DATA_ROOT"
  fi
done
echo "  REVIEW: 7 other workspace_jobs na_gen_* dirs (141635, 171705, 170958, 162448, 161222, 154646, 154412)"

echo
echo "[SAFE-TO-DELETE — after confirmation]"
# P9 duplicate ablation runs (~1.0G)
SAFE_P9=(
  runtime_outputs/nut_assembly/jobs/na_gen_p9_torch_*
  runtime_outputs/nut_assembly/jobs/na_gen_p9_heur*
  runtime_outputs/nut_assembly/jobs/na_gen_p9_heuristic_*
  runtime_outputs/nut_assembly/jobs/na_gen_p9_no_repair_*
  runtime_outputs/nut_assembly/jobs/na_gen_p9_norepair_*
)
TOTAL=0
for g in "${SAFE_P9[@]}"; do
  for p in $g; do
    [[ -d "$p" ]] || continue
    [[ "$p" == *official_p7_50* ]] && continue
    [[ "$p" == *175437_537b* ]] && continue
    sz=$(du -sb "$p" 2>/dev/null | cut -f1)
    echo "  WOULD DELETE: $p ($(du -sh "$p" | cut -f1))"
    TOTAL=$((TOTAL + sz))
  done
done

# Early smoke / failed jobs not in critical path
SAFE_SMOKE=(
  runtime_outputs/nut_assembly/jobs/na_gen_p2_test
  runtime_outputs/nut_assembly/jobs/na_gen_p3_*
  runtime_outputs/nut_assembly/jobs/na_gen_p4_accept
  runtime_outputs/nut_assembly/jobs/na_gen_p5_*
  runtime_outputs/nut_assembly/jobs/na_gen_p6_*
  runtime_outputs/nut_assembly/jobs/na_gen_env_fix_*
  runtime_outputs/nut_assembly/jobs/na_gen_official_accept*
  runtime_outputs/nut_assembly/jobs/na_gen_official_p7_20_20260703_170645_4778
  runtime_outputs/nut_assembly/jobs/na_gen_test_p1
  runtime_outputs/nut_assembly/jobs/na_gen_20260702_164636_8a8e
)
for p in "${SAFE_SMOKE[@]}"; do
  for d in $p; do
    [[ -d "$d" ]] || continue
    sz=$(du -sb "$d" 2>/dev/null | cut -f1)
    echo "  WOULD DELETE: $d ($(du -sh "$d" | cut -f1))"
    TOTAL=$((TOTAL + sz))
  done
done

echo
echo "Estimated reclaim (safe candidates only): $(echo "scale=2; $TOTAL/1073741824" | bc 2>/dev/null || echo '~1.0G+')"
echo "DRY-RUN complete."
