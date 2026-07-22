#!/usr/bin/env bash
# PINN / PhyGen cleanup DRY-RUN script
# Generated: 2026-07-08
# IMPORTANT: This script performs NO deletions. It only echoes intended actions.
# Review cleanup_pinn_inventory_report.md (or chat report) before running any real cleanup.

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

echo "========================================"
echo " PINN / PhyGen cleanup DRY-RUN"
echo " Project root: $ROOT"
echo " NO files will be deleted."
echo "========================================"
echo

# ---------------------------------------------------------------------------
# SAFE-TO-DELETE: runtime experiment outputs (~6.9G total)
# Risk: low — regenerated from scripts + official MimicGen source
# ---------------------------------------------------------------------------
echo "[SAFE-TO-DELETE] runtime_outputs/phygen_* experiment trees"

SAFE_DIRS=(
  "runtime_outputs/phygen_coffee_theta_sweep_v2"      # ~3.0G — v2 theta sweep rollouts (40 runs)
  "runtime_outputs/phygen_coffee_online_rollout"      # ~2.4G — online rollout + oracle diagnosis (288 rollouts)
  "runtime_outputs/phygen_coffee_theta_sweep"         # ~1.3G — v1 theta sweep rollouts (16 runs)
  "runtime_outputs/phygen_coffee_official"            # ~323M — prepared source copies + official smoke datagen
  "runtime_outputs/phygen_smoke"                      # ~332K — StackThree smoke splits + tiny checkpoints
  "runtime_outputs/phygen_coffee_smoke"               # ~172K — Coffee smoke feedback + checkpoint
  "runtime_outputs/phygen_coffee_true_rollout"        # ~140K — proxy true rollout smoke
)

for d in "${SAFE_DIRS[@]}"; do
  if [[ -e "$d" ]]; then
    echo "  WOULD DELETE DIR: $d  ($(du -sh "$d" 2>/dev/null | cut -f1))"
  else
    echo "  SKIP (not found): $d"
  fi
done

echo
echo "[SAFE-TO-DELETE] Python bytecode caches (PINN-related only)"
SAFE_PYC=(
  "scripts/legacy/__pycache__/phygen_model.cpython-313.pyc"
  "phygen/adapters/mimicgen/__pycache__/coffee_repair.cpython-310.pyc"
  "phygen/adapters/mimicgen/__pycache__/coffee_repair.cpython-313.pyc"
  "phygen/__pycache__"
  "phygen/core/__pycache__"
  "phygen/adapters/__pycache__"
  "scripts/__pycache__/train_phygen.cpython-313.pyc"
  "scripts/__pycache__/generate_coffee_preparation_feedback.cpython-313.pyc"
  "scripts/__pycache__/convert_mimicgen_coffee_to_phygen_feedback.cpython-310.pyc"
  "scripts/__pycache__/run_coffee_preparation_theta_sweep.cpython-310.pyc"
  "scripts/__pycache__/split_phygen_feedback.cpython-310.pyc"
  "scripts/__pycache__/evaluate_phygen_selector_on_feedback.cpython-310.pyc"
  "scripts/__pycache__/run_coffee_preparation_theta_sweep_v2.cpython-310.pyc"
  "scripts/__pycache__/run_phygen_generalization_eval.cpython-310.pyc"
  "scripts/__pycache__/run_coffee_oracle_pool_diagnosis.cpython-310.pyc"
)
for p in "${SAFE_PYC[@]}"; do
  if [[ -e "$p" ]]; then
    echo "  WOULD DELETE: $p"
  fi
done

echo
echo "[SAFE-TO-DELETE] small nut_assembly PINN training scratch"
echo "  WOULD DELETE DIR: runtime_outputs/nut_assembly/pinn_training  (~72K)"
echo "  WOULD DELETE FILE: runtime_outputs/nut_assembly/debug/p8_pinn_candidate_and_repair_report.md"
echo "  WOULD DELETE FILE: runtime_outputs/nut_assembly/debug/p9_pinn_model_training_and_validation_report.md"

# ---------------------------------------------------------------------------
# REVIEW-BEFORE-DELETE: may contain code, models, or platform-linked assets
# ---------------------------------------------------------------------------
echo
echo "[REVIEW-BEFORE-DELETE] experiments/phys_demo_gen (~222M)"
echo "  Contains NutAssembly PINN research code + outputs + trained models."
echo "  WOULD ARCHIVE OR DELETE (after confirmation): experiments/phys_demo_gen"

echo
echo "[REVIEW-BEFORE-DELETE] runtime_assets/models/pinn (~1.0M)"
echo "  Contains nut_assembly_pinn_v1/model.pt — may be referenced by platform."
echo "  WOULD ARCHIVE OR DELETE (after confirmation): runtime_assets/models/pinn"

echo
echo "[REVIEW-BEFORE-DELETE] configs/experiments/nut_assembly/pinn (~4K)"
echo "  PINN config snippets for NutAssembly integration."
echo "  WOULD DELETE (after confirmation): configs/experiments/nut_assembly/pinn"

echo
echo "[REVIEW-BEFORE-DELETE] CoffeePreparation experiment scripts (project root + scripts/)"
REVIEW_SCRIPTS=(
  "phygen/adapters/mimicgen/coffee_repair.py"
  "scripts/legacy/phygen_model.py"
  "scripts/train_phygen.py"
  "experiments/phygen/scripts/split_phygen_feedback.py"
  "experiments/phygen/scripts/evaluate_phygen_selector_on_feedback.py"
  "experiments/phygen/scripts/run_phygen_generalization_eval.py"
  "experiments/phygen/scripts/convert_mimicgen_coffee_to_phygen_feedback.py"
  "experiments/phygen/scripts/generate_coffee_preparation_feedback.py"
  "experiments/phygen/scripts/audit_coffee_preparation_source_replay.py"
  "experiments/phygen/scripts/run_coffee_preparation_theta_sweep.py"
  "experiments/phygen/scripts/run_coffee_preparation_theta_sweep_v2.py"
  "experiments/phygen/scripts/run_coffee_online_selected_rollout_validation.py"
  "experiments/phygen/scripts/run_coffee_oracle_pool_diagnosis.py"
)
for s in "${REVIEW_SCRIPTS[@]}"; do
  if [[ -f "$s" ]]; then
    echo "  WOULD DELETE (if abandoning PhyGen): $s"
  fi
done

echo
echo "[REVIEW-BEFORE-DELETE] experiments/phys_demo_gen/dist/nut_assembly_pinn_experiment_code.tar.gz"
echo "  Packaged experiment code archive — confirm before delete."

# ---------------------------------------------------------------------------
# KEEP — do not delete
# ---------------------------------------------------------------------------
echo
echo "[KEEP] Official MimicGen repository and source datasets"
echo "  KEEP: third_party/mimicgen/ (entire repo)"
echo "  KEEP: third_party/mimicgen/datasets/source/*.hdf5 (~450M official source data)"
echo "  KEEP: third_party/mimicgen/mimicgen/datagen/waypoint.py (includes local bugfix)"

echo
echo "[KEEP] PhyGen-Core source (if retaining feature development)"
echo "  KEEP: phygen/"
echo "  KEEP: phygen/fixtures/stack_three_smoke_feedback.jsonl"

echo
echo "[KEEP] Platform NutAssembly PINN integration"
echo "  KEEP: integrations/NutAssemblyMimicGen/utils/pinn_*.py"
echo "  KEEP: integrations/NutAssemblyMimicGen/scripts/run_p8_pinn_candidate_and_repair.py"
echo "  KEEP: integrations/NutAssemblyMimicGen/scripts/run_p9_pinn_model_training_and_validation.py"
echo "  KEEP: integrations/NutAssemblyMimicGen/scripts/run_pinn_repair_rollout.py"

echo
echo "[KEEP] Non-PINN platform runtime outputs"
echo "  KEEP: runtime_outputs/cable_threading/"
echo "  KEEP: runtime_outputs/nut_assembly/ (except pinn_training + debug pinn reports if approved)"
echo "  KEEP: runtime_outputs/training/"
echo "  KEEP: runtime_outputs/evaluation/"
echo "  KEEP: runtime_outputs/isaac_lab/"
echo "  KEEP: runtime_outputs/asset_pipeline/"
echo "  KEEP: backend/ frontend/ docs/ (platform main code)"

echo
echo "[KEEP] Not found — no SoftYamBag / yam_bag experiment artifacts in scan"
echo "  (No paths matching soft_yam / yam_bag / SoftYamBag were discovered.)"

echo
echo "========================================"
echo " DRY-RUN complete. Estimated reclaim from SAFE dirs: ~6.9G"
echo " To execute for real, create a separate script with rm -rf after review."
echo "========================================"
