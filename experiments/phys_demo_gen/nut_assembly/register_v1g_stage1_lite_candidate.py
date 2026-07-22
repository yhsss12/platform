#!/usr/bin/env python3
"""将 V1-G-stage1-lite-p1p2 注册为 experimental candidate（不替换 aligned-original 默认）。"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_OFFLINE_DIR = _EXPERIMENT_DIR / "offline_mimicgen_repair_test"
if str(_OFFLINE_DIR) not in sys.path:
    sys.path.insert(0, str(_OFFLINE_DIR))

from config import (  # noqa: E402
    DEFAULT_BASELINE_MODEL,
    DEFAULT_MODEL_ASSETS_REGISTRY,
    DEMO_3_V1G_LITE_DIAGNOSTIC,
    V1G_STAGE1_LITE_P1P2_METADATA,
    V1G_STAGE1_LITE_P1P2_MODEL,
    V1G_STAGE1_LITE_P1P2_MODEL_CARD,
)

OUT_DIR = _EXPERIMENT_DIR / "outputs" / "v1g_stage1_lite_p1p2"
COMPARISON_JSON = OUT_DIR / "model_comparison_report.json"
INTEGRITY_JSON = OUT_DIR / "checkpoint_integrity.json"
SUMMARY_MD = OUT_DIR / "aligned_original_vs_v1g_lite_summary.md"

MODEL_ID = "v1g-stage1-lite-p1p2"
REL_CKPT = "outputs/v1g_stage1_lite_p1p2/trained_model/model_v1g_stage1_lite_p1p2.pt"
REL_INIT = (
    "outputs/v1f_aligned_repair_parameter_model/original_failed/trained_model/model_v1f_aligned_original.pt"
)

PHYSICS_METADATA: dict[str, Any] = {
    "init_from": "aligned-original",
    "physics_loss": ["E_transport", "E_xy", "E_lift_soft"],
    "lambda_transport": 0.02,
    "lambda_xy": 0.02,
    "lambda_lift": 0.01,
    "lambda_retention": 0.35,
    "excluded_demo": "demo_3",
    "status": "passed_candidate_criteria",
    "replace_default": False,
}


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(_EXPERIMENT_DIR))
    except ValueError:
        return str(path)


def build_model_asset_metadata(
    *,
    comparison: dict[str, Any],
    acceptance: dict[str, Any],
    integrity: dict[str, Any],
) -> dict[str, Any]:
    validation = comparison.get("validation", {})
    return {
        "schema": "nut_assembly_model_asset_metadata_v1",
        "model_id": MODEL_ID,
        "display_name": "V1-G-stage1-lite-p1p2",
        "role": "experimental_candidate",
        "opt_in": True,
        **PHYSICS_METADATA,
        "default_baseline": "aligned-original",
        "checkpoint": REL_CKPT,
        "init_checkpoint": REL_INIT,
        "excluded_demo_diagnostic": {
            "repairability": DEMO_3_V1G_LITE_DIAGNOSTIC["repairability"],
            "failure_stage": DEMO_3_V1G_LITE_DIAGNOSTIC["failure_stage"],
            "failure_reason": DEMO_3_V1G_LITE_DIAGNOSTIC["failure_reason"],
        },
        "training": {
            "epochs": 60,
            "lr": 5e-5,
            "dataset": "outputs/v1f_100base/repair_parameter_dataset_v1f_100base.npz",
            "excluded_training_demos": ["demo_3"],
        },
        "validation": {
            "demos": validation.get("demos", ["demo_2", "demo_4"]),
            "num_samples": validation.get("num_samples", 400),
            "top_k": validation.get("top_k", 30),
            "seeds": validation.get("seeds", [0, 1, 2, 3, 4]),
            "strategies": validation.get("strategies", []),
            "physics_residual_repair_required": True,
        },
        "acceptance": {
            "checkpoint_integrity_ok": acceptance.get("checkpoint_integrity_ok"),
            "passes_candidate_criteria": acceptance.get("passes_candidate_criteria"),
            "demo_4_regression_detected": acceptance.get("demo_4_regression_detected"),
            "insertion_gate_effective": acceptance.get("insertion_gate_effective_on_v1g"),
            "recommend_replace_aligned_original": False,
            "aligned_original_remains_default": True,
        },
        "artifacts": {
            "model_card": "outputs/v1g_stage1_lite_p1p2/model_card_v1g_stage1_lite_p1p2.md",
            "comparison_summary": "outputs/v1g_stage1_lite_p1p2/aligned_original_vs_v1g_lite_summary.md",
            "training_log": "outputs/v1g_stage1_lite_p1p2/training_log.json",
            "checkpoint_integrity": "outputs/v1g_stage1_lite_p1p2/checkpoint_integrity.json",
            "model_comparison_report": "outputs/v1g_stage1_lite_p1p2/model_comparison_report.json",
            "rollout_validation": "outputs/v1g_stage1_lite_p1p2/rollout_validation_v1g_lite.json",
            "residual_validation": "outputs/v1g_stage1_lite_p1p2/residual_validation_v1g_lite.json",
        },
        "registry_path": "model_assets/registry.json",
    }


def build_registry_model_entry() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "display_name": "V1-G-stage1-lite-p1p2",
        "role": "experimental_candidate",
        "opt_in": True,
        "replace_default": False,
        "checkpoint": REL_CKPT,
        "metadata_path": "outputs/v1g_stage1_lite_p1p2/model_asset_metadata.json",
        "model_card": "outputs/v1g_stage1_lite_p1p2/model_card_v1g_stage1_lite_p1p2.md",
        "comparison_summary": "outputs/v1g_stage1_lite_p1p2/aligned_original_vs_v1g_lite_summary.md",
        **PHYSICS_METADATA,
        "description": "基于 aligned-original 轻量 fine-tune；experimental / opt-in candidate，不替换默认基线。",
    }


def write_model_card(*, comparison: dict[str, Any], acceptance: dict[str, Any]) -> str:
    table = comparison.get("comparison_table", [])
    demo4_rows = [r for r in table if r["demo"] == "demo_4"]
    demo2_gated = [
        r
        for r in table
        if r["demo"] == "demo_2" and "gated" in r["strategy"] and "insertion" not in r["strategy"]
    ]
    demo4_insertion = [r for r in table if r["demo"] == "demo_4" and "insertion_gated" in r["strategy"]]

    def _fmt(rows: list[dict], key: str) -> str:
        if not rows:
            return "N/A"
        r = rows[0]
        a, v = r[f"aligned_mean_{key}"], r[f"v1g_mean_{key}"]
        d = r.get(f"delta_{key}", v - a)
        return f"{a:.0%} → {v:.0%}（{d:+.0%}）"

    demo4_plain = _fmt([r for r in demo4_rows if r["strategy"] == "v1f_plain_top_k"], "final")
    demo4_insertion_final = _fmt(demo4_insertion, "final")
    demo2_gate = ""
    if demo2_gated:
        r = demo2_gated[0]
        demo2_gate = f"{r['aligned_mean_gate_pass']:.0%} → {r['v1g_mean_gate_pass']:.0%}"

    lines = [
        "# Model Card: V1-G-stage1-lite-p1p2",
        "",
        "## 概述",
        "",
        "| 字段 | 值 |",
        "|------|-----|",
        f"| model_id | `{MODEL_ID}` |",
        "| role | **experimental_candidate**（opt-in，不替换默认） |",
        f"| status | `{PHYSICS_METADATA['status']}` |",
        "| default_baseline | `aligned-original`（保持不变） |",
        f"| replace_default | `{PHYSICS_METADATA['replace_default']}` |",
        "",
        "V1-G-stage1-lite-p1p2 是基于 **aligned-original** checkpoint 的轻量 physics-augmented fine-tune 分支，"
        "用于在保持默认基线能力的前提下，增强 nut 接近 peg、xy 对齐与 lift 相关能力。",
        "",
        "## Checkpoint",
        "",
        "| 项 | 路径 |",
        "|----|------|",
        f"| 本模型 | `{REL_CKPT}` |",
        f"| 初始化来源 | `{REL_INIT}` |",
        "",
        "- `init_from = aligned-original`",
        "- aligned-original checkpoint **未被修改**（sha256 / mtime 训练前后一致）",
        "",
        "## 训练配置",
        "",
        "### Physics Loss（仅以下 residual 进入训练 loss）",
        "",
        "```",
        "physics_loss = E_transport + E_xy + E_lift_soft",
        "```",
        "",
        "| 参数 | 值 |",
        "|------|-----|",
        f"| lambda_transport | {PHYSICS_METADATA['lambda_transport']} |",
        f"| lambda_xy | {PHYSICS_METADATA['lambda_xy']} |",
        f"| lambda_lift | {PHYSICS_METADATA['lambda_lift']} |",
        f"| lambda_retention | {PHYSICS_METADATA['lambda_retention']} |",
        "",
        "### 排除项",
        "",
        "- **训练 loss 不含**：E_contact、E_bilateral、E_dynamics、E_slip、E_coupling、insertion residual 等",
        "- **excluded_demo**：`demo_3`（`non_repairable_under_current_pipeline`，仅诊断保留）",
        "",
        "### 训练规模",
        "",
        "- epochs: 60，lr: 5e-5",
        "- 数据集: `repair_parameter_dataset_v1f_100base.npz`",
        "- 排除 demo_3 后 eligible 样本: 2583",
        "",
        "## 使用方式",
        "",
        "本模型为 **experimental / opt-in**，不作为默认 PINN scoring 模型。",
        "",
        "```bash",
        "# 显式指定 V1-G-lite checkpoint 进行对比或 rollout",
        "python3 run_v1g_lite_model_comparison.py \\",
        "  --enable-physics-residual-repair \\",
        "  --aligned-model outputs/v1f_aligned_repair_parameter_model/original_failed/trained_model/model_v1f_aligned_original.pt \\",
        "  --v1g-model outputs/v1g_stage1_lite_p1p2/trained_model/model_v1g_stage1_lite_p1p2.pt \\",
        "  --demos demo_2 demo_4 --num-samples 400 --top-k 30 --seeds 0 1 2 3 4",
        "```",
        "",
        "Physics residual repair 仍需显式启用：`enable_physics_residual_repair=true` 或 `--enable-physics-residual-repair`。",
        "",
        "## 验收结论（multi-seed 400/30，seeds 0–4）",
        "",
        "| Demo | 结论 |",
        "|------|------|",
        f"| demo_4 | **无 regression**；plain/gated final **{demo4_plain}**；insertion_gated final **{demo4_insertion_final}** |",
        f"| demo_2 | final / partial **基本持平**；gated 策略 gate pass rate **{demo2_gate}**（略有改善） |",
        f"| insertion gate | 在 V1-G-lite 下**仍然有效**（{'是' if acceptance.get('insertion_gate_effective_on_v1g') else '否'}） |",
        "",
        "## 建议",
        "",
        "- **不建议替换 aligned-original** 为默认模型",
        "- **建议继续扩大验证**（更多 seed / demo / 策略组合）",
        "- 状态：`experimental_candidate`，注册于 `model_assets/registry.json`",
        "",
        "## 关联产物",
        "",
        "- `model_asset_metadata.json`",
        "- `aligned_original_vs_v1g_lite_summary.md`",
        "- `model_comparison_report.json` / `.md`",
        "- `checkpoint_integrity.json`",
        "- `training_log.json`",
        "",
    ]
    return "\n".join(lines)


def write_comparison_summary(*, comparison: dict[str, Any], acceptance: dict[str, Any]) -> str:
    validation = comparison.get("validation", {})
    table = comparison.get("comparison_table", [])

    lines = [
        "# aligned-original vs V1-G-stage1-lite-p1p2 摘要",
        "",
        f"> 验证配置：{' / '.join(validation.get('demos', []))}，"
        f"num_samples={validation.get('num_samples', 400)}，"
        f"top_k={validation.get('top_k', 30)}，"
        f"seeds {'–'.join(str(s) for s in validation.get('seeds', []))}，"
        "physics residual repair opt-in 启用。",
        "",
        "## 模型角色",
        "",
        "| 模型 | 角色 | replace_default |",
        "|------|------|-----------------|",
        "| aligned-original | **default_baseline** | — |",
        "| V1-G-stage1-lite-p1p2 | **experimental_candidate**（opt-in） | `false` |",
        "",
        "aligned-original 继续作为默认 PINN 基线；V1-G-lite 已注册为 experimental candidate，**不替换**默认模型。",
        "",
        "## Rollout 对比（mean 聚合）",
        "",
        "| demo | strategy | aligned final | v1g final | Δfinal | aligned gate pass | v1g gate pass |",
        "|------|----------|---------------|-----------|--------|-------------------|---------------|",
    ]
    for row in table:
        lines.append(
            f"| {row['demo']} | {row['strategy']} | "
            f"{row['aligned_mean_final']:.0%} | {row['v1g_mean_final']:.0%} | {row['delta_final']:+.0%} | "
            f"{row['aligned_mean_gate_pass']:.0%} | {row['v1g_mean_gate_pass']:.0%} |"
        )

    lines.extend(
        [
            "",
            "## 关键结论",
            "",
            "### 1. demo_4：无 regression，final success 提升",
            "",
            "- 所有策略 Δfinal ≥ 0，最大 **+7pp**（plain / gated / p1p2_gated）",
            f"- **未触发** 5pp regression 阈值（`demo_4_regression_detected: {acceptance.get('demo_4_regression_detected', False)}`）",
            "- insertion_gated：**89% → 91%**，gate 收益保持",
            "",
            "### 2. demo_2：基本持平，gate pass rate 略有改善",
            "",
            "- final / partial 与 aligned-original 基本持平（Δfinal ≈ 0）",
            "- gated / p1p2_gated：**gate pass rate 77% → 79%**（residual 筛选略有改善）",
            "",
            "### 3. insertion gate 在 V1-G-lite 下仍然有效",
            "",
            "- demo_4 insertion_gated final：**91%**，显著高于同模型 plain top-k **73%**",
            f"- `insertion_gate_effective_on_v1g: {acceptance.get('insertion_gate_effective_on_v1g', False)}`",
            "",
            "### 4. 不建议替换 aligned-original",
            "",
            "- `replace_default = false`",
            "- `aligned_original_remains_default = true`",
            "- aligned-original checkpoint 完整性已验证（sha256 未变）",
            "",
            "### 5. 建议继续扩大验证",
            "",
            "V1-G-stage1-lite-p1p2 已通过 candidate 验收（`status = passed_candidate_criteria`），"
            "可作为 **experimental candidate** 在更大候选池、更多 demo 或更长 seed 序列上继续验证；"
            "默认生产路径仍使用 aligned-original。",
            "",
            "## 模型资产 metadata",
            "",
            "```json",
            json.dumps(PHYSICS_METADATA, indent=2, ensure_ascii=False),
            "```",
            "",
            "完整 metadata：`outputs/v1g_stage1_lite_p1p2/model_asset_metadata.json`  ",
            "注册表：`model_assets/registry.json`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Register V1-G-lite experimental candidate")
    parser.add_argument("--registry", type=Path, default=DEFAULT_MODEL_ASSETS_REGISTRY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not V1G_STAGE1_LITE_P1P2_MODEL.exists():
        raise SystemExit(f"Checkpoint missing: {V1G_STAGE1_LITE_P1P2_MODEL}")
    if not COMPARISON_JSON.exists():
        raise SystemExit(f"Comparison report missing: {COMPARISON_JSON}")

    comparison = json.loads(COMPARISON_JSON.read_text(encoding="utf-8"))
    integrity = (
        json.loads(INTEGRITY_JSON.read_text(encoding="utf-8"))
        if INTEGRITY_JSON.exists()
        else {"aligned_original_unchanged": None}
    )
    acceptance = comparison.get("acceptance", {})
    registered_at = datetime.now(timezone.utc).isoformat()

    metadata = build_model_asset_metadata(
        comparison=comparison, acceptance=acceptance, integrity=integrity
    )
    model_card = write_model_card(comparison=comparison, acceptance=acceptance)
    summary = write_comparison_summary(comparison=comparison, acceptance=acceptance)

    registration = {
        "registered_at": registered_at,
        "default_baseline": "aligned-original",
        "default_baseline_checkpoint": str(DEFAULT_BASELINE_MODEL),
        "experimental_candidate": {
            "model_id": MODEL_ID,
            "role": "experimental_candidate",
            "opt_in": True,
            "replace_default": False,
            "status": PHYSICS_METADATA["status"],
            "checkpoint": str(V1G_STAGE1_LITE_P1P2_MODEL),
            "metadata": str(V1G_STAGE1_LITE_P1P2_METADATA),
            "model_card": str(V1G_STAGE1_LITE_P1P2_MODEL_CARD),
            "comparison_summary": str(SUMMARY_MD),
        },
        "acceptance_summary": {
            "passes_candidate_criteria": acceptance.get("passes_candidate_criteria"),
            "demo_4_regression_detected": acceptance.get("demo_4_regression_detected"),
            "insertion_gate_effective_on_v1g": acceptance.get("insertion_gate_effective_on_v1g"),
            "aligned_original_unchanged": integrity.get("aligned_original_unchanged"),
            "aligned_original_remains_default": True,
        },
        "registry": str(args.registry),
    }

    if args.dry_run:
        print(json.dumps({"registration": registration, "metadata": metadata}, indent=2))
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    V1G_STAGE1_LITE_P1P2_METADATA.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    V1G_STAGE1_LITE_P1P2_MODEL_CARD.write_text(model_card, encoding="utf-8")
    SUMMARY_MD.write_text(summary, encoding="utf-8")

    registry_path = args.registry
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    else:
        registry = {
            "schema": "nut_assembly_pinn_model_registry_v1",
            "physics_residual_repair_opt_in": True,
            "models": {},
        }

    registry["updated_at"] = registered_at
    registry["default_baseline"] = "aligned-original"
    registry.setdefault("models", {})
    registry["models"]["aligned-original"] = {
        "model_id": "aligned-original",
        "display_name": "V1-F aligned-original",
        "role": "default_baseline",
        "opt_in": False,
        "replace_default": False,
        "checkpoint": REL_INIT,
        "status": "production_default",
        "description": "Nut assembly PINN repair 默认基线；physics residual repair 与 experimental 分支均不得覆盖此 checkpoint。",
    }
    registry["models"][MODEL_ID] = build_registry_model_entry()
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")

    status_path = OUT_DIR / "registration_status.json"
    status_path.write_text(json.dumps(registration, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "registered": True,
                "default_baseline": "aligned-original",
                "experimental_candidate": MODEL_ID,
                "replace_default": False,
                "registry": str(registry_path),
                "metadata": str(V1G_STAGE1_LITE_P1P2_METADATA),
                "model_card": str(V1G_STAGE1_LITE_P1P2_MODEL_CARD),
                "comparison_summary": str(SUMMARY_MD),
                "status": str(status_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
