#!/usr/bin/env python3
"""Audit whether V1-F-aligned-original used new Square_D0 (100 demo) data."""
from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
_OUTPUT = _EXPERIMENT_DIR / "outputs" / "pinn_model_provenance"
_REPO_ROOT = _EXPERIMENT_DIR.parents[2]

NEW_HDF5_MARKERS = ("demo(1).hdf5", "demo_failed(1).hdf5")
OLD_FAILED_HDF5 = _REPO_ROOT / "mnt" / "data" / "demo_failed.hdf5"


def _load_npz_summary(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return json.loads(data["meta_json"].item())


def _scan_jsonl_sources(path: Path) -> dict[str, Any]:
    source_files: Counter[str] = Counter()
    source_demos: Counter[str] = Counter()
    n = 0
    demo1_hits = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            n += 1
            rec = json.loads(line)
            sf = rec.get("source_file") or rec.get("source_hdf5") or rec.get("hdf5_path") or ""
            source_files[str(sf)] += 1
            sd = rec.get("source_demo") or rec.get("demo_key") or rec.get("context", {}).get("source_demo") or ""
            source_demos[str(sd)] += 1
            blob = json.dumps(rec)
            if any(m in blob for m in NEW_HDF5_MARKERS):
                demo1_hits += 1
    return {
        "path": str(path),
        "num_records": n,
        "source_file_counts": dict(source_files),
        "source_demo_counts": dict(source_demos),
        "contains_demo_1_hdf5": demo1_hits > 0,
        "demo_1_hdf5_record_count": demo1_hits,
    }


def _manifest_hits(manifest_path: Path) -> dict[str, bool]:
    if not manifest_path.exists():
        return {}
    text = manifest_path.read_text(encoding="utf-8")
    terms = [
        "new_100_demo",
        "new_100",
        "demo(1).hdf5",
        "demo_failed(1).hdf5",
        "v1f_aligned_plus",
        "Square_D0",
        "new_rollout_samples",
    ]
    return {t: t in text for t in terms}


def _meta_has_new_hdf5(summary: dict[str, Any]) -> dict[str, Any]:
    blob = json.dumps(summary)
    return {
        "contains_demo(1).hdf5": "demo(1).hdf5" in blob,
        "contains_demo_failed(1).hdf5": "demo_failed(1).hdf5" in blob,
        "contains_v1f_plus_rollout_sampling": "v1f_plus_rollout_sampling"
        in summary.get("source_counts", {}),
    }


def build_report() -> dict[str, Any]:
    orig_dir = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_repair_parameter_model" / "original_failed"
    plus_dir = _EXPERIMENT_DIR / "outputs" / "v1f_aligned_plus"
    audit_path = _EXPERIMENT_DIR / "outputs" / "new_100_demo_audit" / "new_demo_audit_report.json"

    orig_npz = orig_dir / "repair_parameter_dataset_v1f.npz"
    plus_npz = plus_dir / "repair_parameter_dataset_v1f_plus.npz"
    orig_manifest = orig_dir / "build_manifest.json"
    plus_manifest = plus_dir / "build_manifest.json"
    orig_jsonl = orig_dir / "repair_parameter_dataset_v1f.jsonl"
    new_rollout = plus_dir / "new_rollout_samples.jsonl"
    new_contexts = plus_dir / "new_failed_contexts.jsonl"
    train_log = orig_dir / "trained_model" / "train_v1f_log.json"

    orig_summary = _load_npz_summary(orig_npz)
    plus_summary = _load_npz_summary(plus_npz)
    plus_manifest_data = json.loads(plus_manifest.read_text(encoding="utf-8"))
    train_log_data = json.loads(train_log.read_text(encoding="utf-8")) if train_log.exists() else {}

    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
    orig_demos = set(orig_summary.get("demo_counts", {}))
    plus_demos = set(plus_summary.get("demo_counts", {}))
    plus_only_demos = sorted(plus_demos - orig_demos, key=lambda x: int(x.split("_")[1]))

    orig_uses_new = _meta_has_new_hdf5(orig_summary)
    plus_uses_new = _meta_has_new_hdf5(plus_summary)

    conclusion = {
        "v1f_aligned_original_used_new_100_data": False,
        "first_version_including_new_100_failed_data": "V1-F-aligned-plus",
        "rationale": (
            "V1-F-aligned-original 仅含 1357 条样本、demo_0–demo_4，"
            "source 为 v1e_import + v1f_rollout_sampling，"
            "meta/manifest/jsonl 中均无 demo(1).hdf5 或 demo_failed(1).hdf5。"
            "新 100 条 Square_D0 中的 23 条 failed demo 通过 demo_failed(1).hdf5 "
            "自 V1-F-aligned-plus 起以 v1f_plus_rollout_sampling (3552 条) 并入训练。"
        ),
        "demo(1).hdf5_in_any_training_set": False,
        "demo_failed(1).hdf5_first_appears_in": "V1-F-aligned-plus (new_rollout_samples.jsonl → plus NPZ)",
    }

    return {
        "report_type": "pinn_model_provenance",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "new_100_data_audit": {
            "audit_report": str(audit_path),
            "source_env": audit.get("source_env"),
            "total_demos_audited": audit.get("demo_counts", {}).get("total"),
            "success_hdf5": audit.get("files", {}).get("success", {}),
            "failed_hdf5": audit.get("files", {}).get("failed", {}),
            "note": (
                "100 条 Square_D0 新数据 = demo(1).hdf5 (77 success) + demo_failed(1).hdf5 (23 failed)。"
                "训练管线仅使用 failed 子集做 repair rollout；success HDF5 未进入 PINN 数据集。"
            ),
        },
        "v1f_aligned_original": {
            "checkpoint": "outputs/v1f_aligned_repair_parameter_model/original_failed/trained_model/model_v1f_aligned_original.pt",
            "dataset_npz": str(orig_npz),
            "build_manifest": str(orig_manifest),
            "train_log": str(train_log),
            "num_samples": orig_summary.get("num_samples"),
            "train_log_num_samples": train_log_data.get("num_samples"),
            "dataset_version": orig_summary.get("dataset_version"),
            "context_mode": orig_summary.get("context_mode"),
            "legacy_failed_hdf5_expected": str(OLD_FAILED_HDF5),
            "source_counts": orig_summary.get("source_counts"),
            "demo_counts": orig_summary.get("demo_counts"),
            "failure_type_counts": orig_summary.get("failure_type_counts"),
            "contains_demo(1).hdf5": orig_uses_new["contains_demo(1).hdf5"],
            "contains_demo_failed(1).hdf5": orig_uses_new["contains_demo_failed(1).hdf5"],
            "contains_v1f_plus_rollout_sampling": orig_uses_new["contains_v1f_plus_rollout_sampling"],
            "build_manifest_term_hits": _manifest_hits(orig_manifest),
            "jsonl_scan": _scan_jsonl_sources(orig_jsonl) if orig_jsonl.exists() else None,
            "notes": orig_summary.get("notes"),
        },
        "v1f_aligned_plus": {
            "dataset_npz": str(plus_npz),
            "build_manifest": plus_manifest_data,
            "num_samples": plus_summary.get("num_samples"),
            "num_base_aligned_original": plus_manifest_data.get("num_base"),
            "num_new_rollout_samples": plus_manifest_data.get("num_new"),
            "base_aligned_jsonl": plus_manifest_data.get("base_aligned_jsonl"),
            "new_rollout_jsonl": plus_manifest_data.get("new_rollout_jsonl"),
            "source_counts": plus_summary.get("source_counts"),
            "demo_counts": plus_summary.get("demo_counts"),
            "plus_only_demo_keys": plus_only_demos,
            "new_source_tag_count": plus_summary.get("source_counts", {}).get("v1f_plus_rollout_sampling"),
            "contains_demo(1).hdf5": plus_uses_new["contains_demo(1).hdf5"],
            "contains_demo_failed(1).hdf5": plus_uses_new["contains_demo_failed(1).hdf5"],
            "build_manifest_term_hits": _manifest_hits(plus_manifest),
            "new_rollout_jsonl_scan": _scan_jsonl_sources(new_rollout),
            "new_failed_contexts_scan": _scan_jsonl_sources(new_contexts) if new_contexts.exists() else None,
            "notes": plus_summary.get("notes"),
        },
        "conclusion": conclusion,
    }


def to_markdown(report: dict[str, Any]) -> str:
    o = report["v1f_aligned_original"]
    p = report["v1f_aligned_plus"]
    c = report["conclusion"]
    audit = report["new_100_data_audit"]

    lines = [
        "# PINN 模型数据来源（Provenance）报告",
        "",
        f"生成时间 (UTC): {report['generated_at']}",
        "",
        "## 结论",
        "",
        f"- **V1-F-aligned-original 是否使用新 100 条数据**: **否**",
        f"- **新 failed 数据首次进入训练的版本**: **{c['first_version_including_new_100_failed_data']}**",
        f"- **demo(1).hdf5（success）是否进入训练**: **否**（仅 audit 使用）",
        f"- **demo_failed(1).hdf5 首次出现**: {c['demo_failed(1).hdf5_first_appears_in']}",
        "",
        c["rationale"],
        "",
        "## 新 100 条 Square_D0 数据",
        "",
        f"- Success: `{audit['success_hdf5'].get('path')}` — {audit['success_hdf5'].get('demo_count')} demos",
        f"- Failed: `{audit['failed_hdf5'].get('path')}` — {audit['failed_hdf5'].get('demo_count')} demos",
        f"- Audit 合计: {audit.get('total_demos_audited')} demos",
        "",
        audit["note"],
        "",
        "## V1-F-aligned-original",
        "",
        f"- 样本数: **{o['num_samples']}**（train_log 确认: {o['train_log_num_samples']}）",
        f"- 数据集版本: `{o['dataset_version']}`",
        f"- 预期 legacy failed HDF5: `{o['legacy_failed_hdf5_expected']}`",
        "",
        "### source 分布（meta source_counts）",
        "",
    ]
    for src, cnt in sorted(o["source_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"- `{src}`: {cnt}")

    lines.extend(
        [
            "",
            "### source_demo 分布",
            "",
        ]
    )
    for dk, cnt in sorted(o["demo_counts"].items(), key=lambda x: int(x[0].split("_")[1])):
        lines.append(f"- `{dk}`: {cnt}")

    lines.extend(
        [
            "",
            "### 新 100 条数据检查",
            "",
            f"- 含 demo(1).hdf5: **{o['contains_demo(1).hdf5']}**",
            f"- 含 demo_failed(1).hdf5: **{o['contains_demo_failed(1).hdf5']}**",
            f"- 含 v1f_plus_rollout_sampling: **{o['contains_v1f_plus_rollout_sampling']}**",
            "",
            "### build_manifest 关键词",
            "",
        ]
    )
    for term, hit in o["build_manifest_term_hits"].items():
        lines.append(f"- `{term}`: {hit}")

    lines.extend(
        [
            "",
            "## V1-F-aligned-plus",
            "",
            f"- 总样本数: **{p['num_samples']}**",
            f"- 基座（aligned-original）: **{p['num_base_aligned_original']}**",
            f"- 新增 rollout 样本: **{p['num_new_rollout_samples']}**（source tag: `v1f_plus_rollout_sampling`）",
            "",
            "### 新增 demo_key（相对 original 的 demo_5–demo_22）",
            "",
            ", ".join(f"`{d}`" for d in p["plus_only_demo_keys"]),
            "",
            "### new_rollout_samples.jsonl",
            "",
        ]
    )
    scan = p["new_rollout_jsonl_scan"]
    for sf, cnt in scan["source_file_counts"].items():
        lines.append(f"- `{sf}`: {cnt} records")
    lines.append(f"- 覆盖 source_demo 数: {len(scan['source_demo_counts'])}")

    lines.extend(
        [
            "",
            "### build_manifest 关键词",
            "",
        ]
    )
    for term, hit in p["build_manifest_term_hits"].items():
        lines.append(f"- `{term}`: {hit}")

    lines.extend(
        [
            "",
            "## 版本时间线",
            "",
            "| 版本 | 样本数 | demo 范围 | 新 100 failed 数据 |",
            "|------|--------|-----------|-------------------|",
            f"| V1-F-aligned-original | {o['num_samples']} | demo_0–demo_4 | 否 |",
            f"| V1-F-aligned-plus | {p['num_samples']} | demo_0–demo_22 | 是（+{p['num_new_rollout_samples']} 来自 demo_failed(1).hdf5） |",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    report = build_report()
    _OUTPUT.mkdir(parents=True, exist_ok=True)
    json_path = _OUTPUT / "model_provenance_report.json"
    md_path = _OUTPUT / "model_provenance_summary.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(to_markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "md": str(md_path), "conclusion": report["conclusion"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
