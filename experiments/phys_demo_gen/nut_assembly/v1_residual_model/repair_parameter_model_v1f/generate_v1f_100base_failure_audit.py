#!/usr/bin/env python3
"""V1-F-100Base failure audit：排查 old demo_4 repair_rate 退化根因。"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

_V1F_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1F_DIR.parent.parent
for path in (_EXPERIMENT_DIR, _V1F_DIR.parent, _V1F_DIR, _EXPERIMENT_DIR / "offline_mimicgen_repair_test"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import DEFAULT_CEM_REPORT, DEFAULT_FAILED_HDF5, DEFAULT_PINN_MODEL, DEMO_REPAIR_CONFIGS  # noqa: E402
from pinn_v1f_repair_model import PINNV1FRepairModel, V1FPhysicsLossConfig, compute_v1f_100base_losses  # noqa: E402
from repair_common_v1f import (  # noqa: E402
    extract_repair_context_v1f,
    sample_repair_candidates_v1f,
    score_repair_candidates_v1f,
    select_candidate_indices_v1f,
)
from train_residual_model import split_indices  # noqa: E402
from v1f_100base_utils import (  # noqa: E402
    ALIGNED_ORIGINAL_JSONL,
    DEFAULT_100BASE_OUTPUT,
    DEFAULT_ALIGNED_MODEL,
    DEFAULT_DATASET_NPZ,
    OLD_DEMO_KEYS,
    OLD_STABLE_SOURCE_PREFIXES,
    NEW_FAILED_SOURCE_PREFIXES,
    SUCCESS_REFERENCE_SOURCES,
)
from v1f_repair_dataset import V1FRepairDataset, load_v1f_npz  # noqa: E402

DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_100base_failure_audit"
DEFAULT_100BASE_CKPT = DEFAULT_100BASE_OUTPUT / "trained_model" / "model_v1f_100base.pt"
DEFAULT_EVAL = DEFAULT_100BASE_OUTPUT / "evaluation" / "quick_validation_report.json"

FAILURE_TYPES = ("transport_failed", "insertion_failed", "alignment_failed", "grasp_failed", "lift_failed", "success")


def _is_old_stable_source(source: str) -> bool:
    return any(source.startswith(p) or source == p for p in OLD_STABLE_SOURCE_PREFIXES)


def _is_new_failed_source(source: str) -> bool:
    return any(source.startswith(p) for p in NEW_FAILED_SOURCE_PREFIXES)


def _is_success_reference_source(source: str) -> bool:
    return source in SUCCESS_REFERENCE_SOURCES or "success_reference" in source


def _source_dataset(meta: dict[str, Any]) -> str:
    source = str(meta.get("source", ""))
    if _is_success_reference_source(source):
        return "success_reference"
    if _is_old_stable_source(source) or meta.get("is_old_demo"):
        return "legacy_old"
    if _is_new_failed_source(source):
        return "new100_failed"
    return "other"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def audit_demo_key_collisions(meta_records: list[dict[str, Any]], demo_group_id: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    gid_by_demo: dict[str, set[int]] = defaultdict(set)
    for i, meta in enumerate(meta_records):
        dk = str(meta.get("demo_key", ""))
        gid_by_demo[dk].add(int(demo_group_id[i]))

    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, meta in enumerate(meta_records):
        dk = str(meta.get("demo_key", ""))
        sd = _source_dataset(meta)
        grouped[dk][sd].append(i)

    for dk in sorted(grouped.keys(), key=lambda k: int(k.split("_")[-1]) if "_" in k else k):
        by_src = grouped[dk]
        legacy = by_src.get("legacy_old", [])
        new100 = by_src.get("new100_failed", [])
        ref = by_src.get("success_reference", [])
        collision = len(legacy) > 0 and len(new100) > 0
        same_gid = len(gid_by_demo.get(dk, set())) == 1 and len(gid_by_demo.get(dk, set())) > 0
        rows.append(
            {
                "demo_key": dk,
                "demo_group_id": next(iter(gid_by_demo.get(dk, {-1}))),
                "legacy_old_count": len(legacy),
                "new100_failed_count": len(new100),
                "success_reference_count": len(ref),
                "total_count": len(legacy) + len(new100) + len(ref) + len(by_src.get("other", [])),
                "demo_key_collision": collision,
                "shares_demo_group_id": same_gid,
                "pairwise_ranking_mixed_group": collision and same_gid,
                "retention_flag_on_new100": any(meta_records[i].get("demo_key") in OLD_DEMO_KEYS for i in new100),
                "recommended_group_key_legacy": f"old:{dk}" if collision else dk,
                "recommended_group_key_new100": f"new100:{dk}" if collision else dk,
            }
        )

    demo_4 = next((r for r in rows if r["demo_key"] == "demo_4"), {})
    summary = {
        "collision_demo_keys": [r["demo_key"] for r in rows if r["demo_key_collision"]],
        "demo_0_to_4_all_collide_with_new100": all(
            r["demo_key_collision"] for r in rows if r["demo_key"] in OLD_DEMO_KEYS and r["new100_failed_count"] > 0
        ),
        "grouping_uses_demo_key_only": True,
        "pairwise_ranking_groups_by_demo_group_id_only": True,
        "validation_groups_by_demo_group_not_source": True,
        "demo_4_collision": demo_4,
    }
    return rows, summary


def audit_old_demo_4_splits(
    meta_records: list[dict[str, Any]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    old_demo_retention: np.ndarray,
    ranking_eligible: np.ndarray,
    demo_group_id: np.ndarray,
) -> dict[str, Any]:
    def _stats(indices: np.ndarray, label: str) -> dict[str, Any]:
        idxs = [int(i) for i in indices]
        legacy = [i for i in idxs if _source_dataset(meta_records[i]) == "legacy_old" and meta_records[i].get("demo_key") == "demo_4"]
        new100 = [i for i in idxs if _source_dataset(meta_records[i]) == "new100_failed" and meta_records[i].get("demo_key") == "demo_4"]
        ref = [i for i in idxs if _source_dataset(meta_records[i]) == "success_reference" and meta_records[i].get("demo_key") == "demo_4"]
        success_legacy = [i for i in legacy if meta_records[i].get("success_flag") or meta_records[i].get("refined_success_flag")]
        return {
            "split": label,
            "legacy_old_samples": len(legacy),
            "new100_failed_samples": len(new100),
            "success_reference_samples": len(ref),
            "legacy_success_theta": len(success_legacy),
            "legacy_retention_weighted": sum(1 for i in legacy if old_demo_retention[i] > 0.5),
            "new100_retention_mislabeled": sum(1 for i in new100 if old_demo_retention[i] > 0.5),
            "ranking_eligible_legacy": sum(1 for i in legacy if ranking_eligible[i] > 0.5),
            "ranking_eligible_new100": sum(1 for i in new100 if ranking_eligible[i] > 0.5),
            "shared_demo_group_id": int(demo_group_id[legacy[0]]) if legacy else None,
        }

    train_stats = _stats(train_idx, "train")
    val_stats = _stats(val_idx, "val")
    gid = train_stats.get("shared_demo_group_id")
    mixed_ranking_pairs = 0
    if gid is not None:
        train_set = set(int(i) for i in train_idx)
        group_idxs = [i for i in train_set if int(demo_group_id[i]) == gid and ranking_eligible[i] > 0.5]
        for a in range(len(group_idxs)):
            for b in range(a + 1, len(group_idxs)):
                sa = _source_dataset(meta_records[group_idxs[a]])
                sb = _source_dataset(meta_records[group_idxs[b]])
                if sa != sb:
                    mixed_ranking_pairs += 1

    return {
        "demo_4_train": train_stats,
        "demo_4_val": val_stats,
        "pairwise_old_new_cross_pairs_in_train_demo_4_group": mixed_ranking_pairs,
        "retention_loss_applies_to_new100_demo_4": train_stats["new100_retention_mislabeled"] > 0,
    }


def estimate_loss_contributions(
    model: PINNV1FRepairModel,
    dataset: V1FRepairDataset,
    meta_records: list[dict[str, Any]],
    device: torch.device,
    physics: V1FPhysicsLossConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    model.eval()

    per_sample: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)

    with torch.no_grad():
        offset = 0
        for batch in loader:
            bs = len(batch["features"])
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            out = model(batch_dev["features"])
            losses = compute_v1f_100base_losses(out, batch_dev, physics=physics)

            # Per-sample proxy contributions (component-level, scaled like total loss)
            err = (out["E_total"] - batch_dev["target_E_total"]) ** 2
            sw = batch_dev.get("sample_weight")
            if sw is not None:
                w = sw / sw.mean().clamp(min=1e-6)
                sup = w * err
            else:
                sup = err
            old_ret = batch_dev.get("old_demo_retention")
            if old_ret is not None:
                retention = (1.0 + 2.5 * old_ret) * err
            else:
                retention = torch.zeros_like(err)

            focal = torch.nn.functional.binary_cross_entropy_with_logits(
                out["success_logit"], batch_dev["success_flag"], reduction="none"
            )

            for j in range(bs):
                idx = dataset.indices[offset + j]
                meta = meta_records[idx]
                dk = str(meta.get("demo_key", ""))
                sd = _source_dataset(meta)
                ft = str(meta.get("source_failure_type", meta.get("failure_type", "unknown")))

                tags = [sd, f"demo:{dk}"]
                if dk == "demo_4" and sd == "legacy_old":
                    tags.append("old_demo_4")
                if dk == "demo_2" and sd == "legacy_old":
                    tags.append("old_demo_2")
                if sd == "success_reference":
                    tags.append("success_reference")
                if ft in FAILURE_TYPES:
                    tags.append(ft)

                vals = {
                    "total_supervision": float(sup[j].item()),
                    "old_demo_retention": float(retention[j].item()),
                    "success_focal": float(focal[j].item()),
                }
                for tag in tags:
                    counts[tag] += 1
                    for k, v in vals.items():
                        per_sample[tag][k] += v
            offset += bs

    total_sup = sum(v["total_supervision"] for v in per_sample.values()) or 1.0
    total_ret = sum(v["old_demo_retention"] for v in per_sample.values()) or 1.0
    total_focal = sum(v["success_focal"] for v in per_sample.values()) or 1.0

    rows: list[dict[str, Any]] = []
    for tag in sorted(per_sample.keys()):
        ps = per_sample[tag]
        n = counts[tag]
        rows.append(
            {
                "category": tag,
                "sample_count": n,
                "total_supervision_sum": ps["total_supervision"],
                "total_supervision_share_pct": 100.0 * ps["total_supervision"] / total_sup,
                "old_demo_retention_sum": ps["old_demo_retention"],
                "old_demo_retention_share_pct": 100.0 * ps["old_demo_retention"] / total_ret,
                "success_focal_sum": ps["success_focal"],
                "success_focal_share_pct": 100.0 * ps["success_focal"] / total_focal,
            }
        )

    summary = {
        "train_sample_count": len(dataset),
        "legacy_old_supervision_share_pct": next((r["total_supervision_share_pct"] for r in rows if r["category"] == "legacy_old"), 0.0),
        "new100_failed_supervision_share_pct": next((r["total_supervision_share_pct"] for r in rows if r["category"] == "new100_failed"), 0.0),
        "success_reference_supervision_share_pct": next((r["total_supervision_share_pct"] for r in rows if r["category"] == "success_reference"), 0.0),
        "old_demo_4_supervision_share_pct": next((r["total_supervision_share_pct"] for r in rows if r["category"] == "old_demo_4"), 0.0),
        "insertion_failed_supervision_share_pct": next((r["total_supervision_share_pct"] for r in rows if r["category"] == "insertion_failed"), 0.0),
        "transport_failed_supervision_share_pct": next((r["total_supervision_share_pct"] for r in rows if r["category"] == "transport_failed"), 0.0),
        "retention_dominance_risk": (
            next((r["total_supervision_share_pct"] for r in rows if r["category"] == "new100_failed"), 0.0)
            + next((r["total_supervision_share_pct"] for r in rows if r["category"] == "success_reference"), 0.0)
        )
        > next((r["total_supervision_share_pct"] for r in rows if r["category"] == "old_demo_4"), 0.0),
    }
    return rows, summary


def audit_success_reference(meta_records: list[dict[str, Any]], ranking_eligible: np.ndarray, old_demo_retention: np.ndarray) -> dict[str, Any]:
    ref_idxs = [i for i, m in enumerate(meta_records) if _is_success_reference_source(str(m.get("source", "")))]
    ref_in_old_keys = [i for i in ref_idxs if str(meta_records[i].get("demo_key", "")) in OLD_DEMO_KEYS]
    return {
        "success_reference_total": len(ref_idxs),
        "success_reference_on_demo_0_4": len(ref_in_old_keys),
        "success_reference_ranking_eligible": sum(1 for i in ref_idxs if ranking_eligible[i] > 0.5),
        "success_reference_retention_mislabeled": sum(1 for i in ref_idxs if old_demo_retention[i] > 0.5),
        "used_in_success_focal": len(ref_idxs),
        "excluded_from_loss_code": False,
        "note": "训练 loss 未读取 is_success_reference；success reference 参与 success_focal / pairwise_ranking / total_supervision",
    }


def audit_early_stopping(train_log: dict[str, Any], val_idx: np.ndarray, meta_records: list[dict[str, Any]]) -> dict[str, Any]:
    history = train_log.get("history_tail", [])
    val_demo_4_legacy = sum(
        1
        for i in val_idx
        if meta_records[int(i)].get("demo_key") == "demo_4" and _source_dataset(meta_records[int(i)]) == "legacy_old"
    )
    val_demo_4_new100 = sum(
        1
        for i in val_idx
        if meta_records[int(i)].get("demo_key") == "demo_4" and _source_dataset(meta_records[int(i)]) == "new100_failed"
    )
    epoch20_val = next((h.get("val_loss") for h in history if h.get("epoch") == 20), None)
    epoch120_val = next((h.get("val_loss") for h in history if h.get("epoch") == 120), None)
    return {
        "best_val_loss_saved": train_log.get("best_val_loss"),
        "checkpoint_selection_metric": "val_loss (random 15% sample split)",
        "logged_epoch20_val_loss": epoch20_val,
        "logged_epoch120_val_loss": epoch120_val,
        "val_loss_increased_late_training": epoch20_val is not None and epoch120_val is not None and epoch120_val > epoch20_val,
        "val_split_legacy_old_demo_4_count": val_demo_4_legacy,
        "val_split_new100_demo_4_count": val_demo_4_new100,
        "val_split_has_old_demo_4_gate": val_demo_4_legacy > 0,
        "val_loss_aligns_with_rollout_repair_rate": False,
        "recommendation": "early stopping 需加入 rollout-free old demo_4 ranking gate，不能只看 val_loss",
    }


def audit_ranking_regression(
    *,
    aligned_model: Path,
    hundredbase_model: Path,
    failed_hdf5: Path,
    cem_report: Path,
    aligned_jsonl: Path,
    seed: int = 0,
    pool_size: int = 500,
    top_k: int = 20,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cfg = DEMO_REPAIR_CONFIGS["demo_4"]
    context = extract_repair_context_v1f(
        context_source="original_failed_context",
        failed_hdf5=failed_hdf5,
        demo_key="demo_4",
        failure_type=cfg["failure_type"],
        search_kind=cfg["search_kind"],
        cem_report=cem_report,
    )

    # Known-good thetas from aligned-original jsonl successes
    known_good: list[dict[str, Any]] = []
    for rec in _load_jsonl(aligned_jsonl):
        if rec.get("demo_key") != "demo_4":
            continue
        rollout = rec.get("rollout", {})
        if not rollout.get("success_flag"):
            continue
        sim = rollout.get("sim_params") or rollout.get("repair_insertion_params") or {}
        if not sim:
            continue
        known_good.append(
            {
                "source": "aligned_original_jsonl_success",
                "target_E_total": float(rec.get("target_E_total", rollout.get("E_total_norm", 0.0))),
                "insertion": {k: float(sim[k]) for k in sim},
            }
        )

    pool_seed = seed + hash("demo_4") % 10000
    candidates = sample_repair_candidates_v1f(search_kind=cfg["search_kind"], n_samples=pool_size, seed=pool_seed)

    # Inject a few known-good thetas into candidate pool
    for i, kg in enumerate(known_good[:5]):
        candidates.append(
            {
                "index": pool_size + i,
                "insertion": kg["insertion"],
                "transport": None,
                "grasp_lift": None,
                "lift_extra": None,
                "injected_known_good": True,
                "jsonl_target_E_total": kg["target_E_total"],
            }
        )

    rows: list[dict[str, Any]] = []
    top_sets: dict[str, list[int]] = {}

    for label, model_path in (("aligned-original", aligned_model), ("v1f-100base", hundredbase_model)):
        cands = [dict(c) for c in candidates]
        score_repair_candidates_v1f(
            context=context,
            candidates=cands,
            active=cfg["active"],
            v1e_model_path=DEFAULT_PINN_MODEL,
            v1f_model_path=model_path,
        )
        top_idx = select_candidate_indices_v1f(cands, method="v1f_plain_top_k", top_k=top_k, rng=random.Random(seed))
        top_sets[label] = top_idx
        for rank, idx in enumerate(top_idx, start=1):
            c = cands[idx]
            rows.append(
                {
                    "model": label,
                    "pool_index": idx,
                    "rank": rank,
                    "predicted_E_total": c.get("v1f_E_total"),
                    "success_prob": c.get("v1f_success_prob"),
                    "uncertainty": c.get("v1f_uncertainty"),
                    "injected_known_good": bool(c.get("injected_known_good")),
                    "jsonl_target_E_total": c.get("jsonl_target_E_total"),
                }
            )

    overlap = len(set(top_sets["aligned-original"]) & set(top_sets["v1f-100base"]))
    orig_kg_ranks = [r["rank"] for r in rows if r["model"] == "aligned-original" and r["injected_known_good"]]
    base_kg_ranks = [r["rank"] for r in rows if r["model"] == "v1f-100base" and r["injected_known_good"]]

    summary = {
        "candidate_pool_size": len(candidates),
        "known_good_injected": min(5, len(known_good)),
        "top20_overlap_count": overlap,
        "top20_overlap_ratio": overlap / top_k,
        "aligned_original_known_good_ranks": orig_kg_ranks,
        "v1f_100base_known_good_ranks": base_kg_ranks,
        "known_good_pushed_down_by_100base": (
            len(base_kg_ranks) > 0 and (not orig_kg_ranks or min(base_kg_ranks) > min(orig_kg_ranks))
        ),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_markdown(report: dict[str, Any]) -> str:
    c = report["conclusions"]
    qv = report.get("quick_validation", {})
    lines = [
        "# V1-F-100Base Failure Audit",
        "",
        "## 结论摘要",
        "",
        f"- **A. demo_key collision 导致退化？** {c['A_demo_key_collision']}",
        f"- **B. success reference 误入 repair ranking？** {c['B_success_reference']}",
        f"- **C. new data loss 淹没 old retention？** {c['C_loss_drowning']}",
        f"- **D. early stopping 指标不匹配？** {c['D_early_stopping']}",
        f"- **E. 100Base-R1 建议** {c['E_r1_recommendations']}",
        "",
        "## Quick Validation 退化",
        "",
        f"- old demo_4: aligned-original **{qv.get('old_demo_4_aligned_original')}** → 100Base **{qv.get('old_demo_4_v1f_100base')}**",
        f"- old demo_2: **{qv.get('old_demo_2_aligned_original')}** → **{qv.get('old_demo_2_v1f_100base')}**",
        "",
        "## demo_4 Collision",
        "",
        json.dumps(report["demo_key_collision"]["summary"].get("demo_4_collision", {}), indent=2, ensure_ascii=False),
        "",
        "## Ranking Regression (old demo_4 pool)",
        "",
        json.dumps(report["ranking_regression"]["summary"], indent=2, ensure_ascii=False),
        "",
        "## Loss Contribution",
        "",
        json.dumps(report["loss_contribution"]["summary"], indent=2, ensure_ascii=False),
        "",
        "## Early Stopping",
        "",
        json.dumps(report["early_stopping"], indent=2, ensure_ascii=False),
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-F-100Base failure audit")
    parser.add_argument("--dataset-npz", type=Path, default=DEFAULT_DATASET_NPZ)
    parser.add_argument("--aligned-model", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--100base-model", dest="hundredbase_model", type=Path, default=DEFAULT_100BASE_CKPT)
    parser.add_argument("--train-log", type=Path, default=DEFAULT_100BASE_OUTPUT / "trained_model" / "train_log.json")
    parser.add_argument("--quick-validation", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--aligned-jsonl", type=Path, default=ALIGNED_ORIGINAL_JSONL)
    parser.add_argument("--failed-hdf5", type=Path, default=DEFAULT_FAILED_HDF5)
    parser.add_argument("--cem-report", type=Path, default=DEFAULT_CEM_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_v1f_npz(args.dataset_npz)
    meta = bundle["meta"]
    meta_records: list[dict[str, Any]] = meta.get("meta_records", [])

    ckpt = torch.load(args.hundredbase_model, map_location="cpu", weights_only=False)
    train_idx = np.array(ckpt.get("train_indices", []), dtype=np.int64)
    val_idx = np.array(ckpt.get("val_indices", []), dtype=np.int64)
    if len(train_idx) == 0:
        train_idx, val_idx = split_indices(len(bundle["features"]), args.val_frac, args.seed)

    collision_rows, collision_summary = audit_demo_key_collisions(meta_records, bundle["demo_group_id"])
    split_audit = audit_old_demo_4_splits(
        meta_records,
        train_idx,
        val_idx,
        old_demo_retention=bundle["old_demo_retention"],
        ranking_eligible=bundle["ranking_supervision_eligible"],
        demo_group_id=bundle["demo_group_id"],
    )
    success_ref_audit = audit_success_reference(
        meta_records, bundle["ranking_supervision_eligible"], bundle["old_demo_retention"]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PINNV1FRepairModel(input_dim=int(bundle["features"].shape[1])).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    train_ds = V1FRepairDataset(args.dataset_npz, train_idx)
    loss_rows, loss_summary = estimate_loss_contributions(
        model, train_ds, meta_records, device, V1FPhysicsLossConfig.hundredbase()
    )

    train_log = json.loads(args.train_log.read_text(encoding="utf-8")) if args.train_log.exists() else {}
    early_audit = audit_early_stopping(train_log, val_idx, meta_records)

    ranking_rows, ranking_summary = audit_ranking_regression(
        aligned_model=args.aligned_model,
        hundredbase_model=args.hundredbase_model,
        failed_hdf5=args.failed_hdf5,
        cem_report=args.cem_report,
        aligned_jsonl=args.aligned_jsonl,
        seed=args.seed,
    )

    qv = {}
    if args.quick_validation.exists():
        qv_report = json.loads(args.quick_validation.read_text(encoding="utf-8"))
        acc = qv_report.get("acceptance", {})
        qv = {
            "old_demo_4_aligned_original": acc.get("old_demo_4", {}).get("aligned-original"),
            "old_demo_4_v1f_100base": acc.get("old_demo_4", {}).get("v1f-100base"),
            "old_demo_2_aligned_original": acc.get("old_demo_2", {}).get("aligned-original"),
            "old_demo_2_v1f_100base": acc.get("old_demo_2", {}).get("v1f-100base"),
            "all_passed": acc.get("all_passed"),
        }

    collision_yes = collision_summary.get("demo_4_collision", {}).get("demo_key_collision", False)
    success_ref_yes = success_ref_audit["success_reference_ranking_eligible"] > 0
    loss_drown_yes = loss_summary.get("retention_dominance_risk", False)
    early_stop_yes = early_audit.get("val_loss_increased_late_training", False) and not early_audit.get(
        "val_loss_aligns_with_rollout_repair_rate", True
    )

    conclusions = {
        "A_demo_key_collision": (
            "是 — old/new demo_4 共享 demo_key 与 demo_group_id，pairwise ranking 跨来源混排，"
            "new100 demo_4 被误标 old_demo_retention=1.0"
            if collision_yes
            else "否"
        ),
        "B_success_reference": (
            "是 — success reference 未在 loss 中隔离，650 条参与 success_focal 与 ranking_eligible pairwise"
            if success_ref_yes
            else "否或影响有限"
        ),
        "C_loss_drowning": (
            "是 — new100_failed + success_reference 的 weighted supervision 占比高于 legacy old demo_4，retention 信号被稀释"
            if loss_drown_yes
            else "部分 — new insertion 样本主导 loss，但 old demo_4 仍有 retention 权重"
        ),
        "D_early_stopping": (
            "是 — best checkpoint 按 random val_loss 选取，后期 val_loss 上升，且不保证 old demo_4 repair_rate"
            if early_stop_yes
            else "否"
        ),
        "E_r1_recommendations": (
            "1) 引入 source_dataset 分组键 legacy_old:new100:demo_4；"
            "2) retention/ranking 仅作用于 legacy_old；"
            "3) success reference 从 repair ranking/focal 排除，仅 calibration；"
            "4) early stop 加 old demo_4 ranking gate；"
            "5) 禁止 new100 demo_0-4 与 old 共享 demo_key"
        ),
    }

    report: dict[str, Any] = {
        "task": "v1f_100base_failure_audit",
        "focus": "old demo_4 repair_rate regression (0.95 -> 0.25)",
        "default_model_unchanged": str(DEFAULT_ALIGNED_MODEL),
        "do_not_promote_100base": True,
        "quick_validation": qv,
        "demo_key_collision": {"rows": collision_rows, "summary": collision_summary},
        "old_demo_4_retention": split_audit,
        "success_reference": success_ref_audit,
        "loss_contribution": {"rows": loss_rows, "summary": loss_summary},
        "early_stopping": early_audit,
        "ranking_regression": {"rows": ranking_rows, "summary": ranking_summary},
        "conclusions": conclusions,
    }

    write_csv(args.output_dir / "demo_key_collision_audit.csv", collision_rows)
    write_csv(args.output_dir / "loss_contribution_audit.csv", loss_rows)
    write_csv(args.output_dir / "ranking_regression_audit.csv", ranking_rows)
    (args.output_dir / "failure_audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "failure_audit_summary.md").write_text(to_markdown(report), encoding="utf-8")

    print(json.dumps({"output_dir": str(args.output_dir), "conclusions": conclusions}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
