#!/usr/bin/env python3
"""Audit evaluation jobs for evaluation type label consistency."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "backend"))


def _old_label(evaluation_mode: str | None) -> str:
    mode = (evaluation_mode or "").strip()
    if mode == "trained_model_evaluation":
        return "模型评测"
    if mode in {"expert_policy_evaluation", "expert_policy", "policy_evaluation"}:
        return "专家策略评测"
    if mode == "episode_stability":
        return "稳定性评测"
    if mode in {"dataset_evaluation", "dataset_offline"}:
        return "数据集评测"
    if mode == "robustness_evaluation":
        return "鲁棒性评测"
    return "评测任务"


def main() -> None:
    from app.core.database import SessionLocal
    from app.models.workspace_job import WorkspaceJob
    from app.services.evaluation.evaluation_type import resolve_evaluation_type_from_sources

    rows: list[dict] = []
    with SessionLocal() as db:
        jobs = (
            db.query(WorkspaceJob)
            .filter(WorkspaceJob.job_type == "evaluation", WorkspaceJob.status != "deleted")
            .order_by(WorkspaceJob.created_at.desc())
            .all()
        )

        for job in jobs:
            meta = dict(job.metadata_json or {}) if isinstance(job.metadata_json, dict) else {}
            metrics = dict(job.metrics_json or {}) if isinstance(job.metrics_json, dict) else {}
            eval_request = meta.get("evaluationRequest") if isinstance(meta.get("evaluationRequest"), dict) else {}
            evaluation_mode = str(
                metrics.get("evaluationMode") or eval_request.get("evaluationMode") or meta.get("evaluationMode") or ""
            )
            current = _old_label(evaluation_mode)
            resolution = resolve_evaluation_type_from_sources(
                evaluation_object=eval_request.get("evaluationObject") or meta.get("evaluationObject"),
                evaluation_mode=evaluation_mode,
                product_evaluation_mode=eval_request.get("productEvaluationMode") or meta.get("productEvaluationMode"),
                model_asset_id=metrics.get("modelAssetId") or eval_request.get("modelAssetId"),
                model_asset_name=metrics.get("modelName") or eval_request.get("modelName"),
                dataset_id=metrics.get("datasetId") or eval_request.get("datasetId"),
                dataset_name=metrics.get("datasetName") or eval_request.get("datasetName"),
                task_type=job.task_type,
                runner=job.runner,
                task_name=job.task_name,
                metadata=meta,
                metrics=metrics,
                evaluation_request=eval_request,
            )
            expected = resolution["evaluationTypeLabel"]
            rows.append(
                {
                    "job_id": job.job_id,
                    "task_name": job.task_name or "",
                    "current": current,
                    "expected": expected,
                    "basis": resolution["basis"],
                    "confidence": resolution["confidence"],
                    "match": current == expected,
                }
            )

    total = len(rows)
    matched = sum(1 for row in rows if row["match"])
    mismatched = total - matched

    report_path = ROOT / "docs" / "reports" / "evaluation_type_audit_report.md"
    lines = [
        "# 评测类型历史任务审计报告",
        "",
        f"- 总任务数：{total}",
        f"- 修复前标签匹配：{matched}",
        f"- 修复前标签不匹配：{mismatched}",
        "",
        "主要错误类型：",
        "- `episode_stability` 被旧逻辑映射为「稳定性评测」，应统一为「专家策略评测」",
        "- 缺少 evaluationMode / evaluationObject 的历史任务被旧逻辑映射为「评测任务」",
        "",
        "| job_id | task_name | 当前显示 | 应显示 | 判断依据 | 置信度 | 是否匹配 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        task_name = str(row["task_name"]).replace("|", "\\|")
        lines.append(
            f"| {row['job_id']} | {task_name} | {row['current']} | {row['expected']} | {row['basis']} | {row['confidence']} | {'是' if row['match'] else '否'} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"TOTAL={total} MATCHED={matched} MISMATCHED={mismatched}")


if __name__ == "__main__":
    main()
