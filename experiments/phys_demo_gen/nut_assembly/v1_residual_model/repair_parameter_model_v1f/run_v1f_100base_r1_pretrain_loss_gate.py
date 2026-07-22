#!/usr/bin/env python3
"""V1-F-100Base-R1 pre-train loss contribution gate（训练前必须通过）。"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

_V1F_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1F_DIR.parent.parent
for path in (_EXPERIMENT_DIR, _V1F_DIR.parent, _V1F_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pinn_v1f_repair_model import PINNV1FRepairModel, V1FPhysicsLossConfig, compute_v1f_100base_r1_losses  # noqa: E402
from v1f_100base_utils import OLD_DEMO_KEYS  # noqa: E402
from v1f_100base_r1_utils import (  # noqa: E402
    DEFAULT_ALIGNED_MODEL,
    DEFAULT_DATASET_NPZ,
    DEFAULT_LOSS_GATE_REPORT,
    SOURCE_LEGACY_OLD,
    SOURCE_NEW100_FAILED,
    SOURCE_SUCCESS_REF,
    audit_demo_uid_collisions,
    make_demo_uid,
    split_indices_by_demo_uid,
)
from v1f_repair_dataset import V1FRepairDataset, load_v1f_npz  # noqa: E402

FAILURE_TYPES = ("transport_failed", "insertion_failed", "alignment_failed", "grasp_failed", "lift_failed")


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _audit_retention_labeling(meta_records: list[dict[str, Any]], old_demo_retention: np.ndarray) -> dict[str, Any]:
    legacy_ret = [
        i
        for i, m in enumerate(meta_records)
        if old_demo_retention[i] > 0.5 and str(m.get("source_dataset", "")) == SOURCE_LEGACY_OLD
    ]
    new100_mislabeled = [
        i
        for i, m in enumerate(meta_records)
        if old_demo_retention[i] > 0.5 and str(m.get("source_dataset", "")) == SOURCE_NEW100_FAILED
    ]
    success_ref_mislabeled = [
        i
        for i, m in enumerate(meta_records)
        if old_demo_retention[i] > 0.5 and str(m.get("source_dataset", "")) == SOURCE_SUCCESS_REF
    ]
    legacy_demo_4 = [
        i
        for i in legacy_ret
        if str(meta_records[i].get("demo_key", "")) == "demo_4"
    ]
    return {
        "legacy_old_retention_count": len(legacy_ret),
        "new100_retention_mislabeled_count": len(new100_mislabeled),
        "success_ref_retention_mislabeled_count": len(success_ref_mislabeled),
        "legacy_old_demo_4_retention_count": len(legacy_demo_4),
        "passed": len(new100_mislabeled) == 0 and len(success_ref_mislabeled) == 0,
    }


def _audit_success_reference_isolation(
    meta_records: list[dict[str, Any]],
    ranking_eligible: np.ndarray,
    is_success_reference: np.ndarray,
) -> dict[str, Any]:
    ref_idxs = [i for i, m in enumerate(meta_records) if is_success_reference[i] > 0.5]
    ref_ranking = sum(1 for i in ref_idxs if ranking_eligible[i] > 0.5)
    return {
        "success_reference_total": len(ref_idxs),
        "success_reference_ranking_eligible": ref_ranking,
        "passed": ref_ranking == 0,
    }


def _audit_pairwise_groups(meta_records: list[dict[str, Any]], demo_group_id: np.ndarray) -> dict[str, Any]:
    gid_to_uids: dict[int, set[str]] = defaultdict(set)
    for i, meta in enumerate(meta_records):
        gid_to_uids[int(demo_group_id[i])].add(str(meta.get("demo_uid", "")))
    cross_namespace = [
        {"demo_group_id": gid, "demo_uids": sorted(uids)}
        for gid, uids in gid_to_uids.items()
        if len(uids) > 1
    ]
    return {
        "num_groups": len(gid_to_uids),
        "cross_namespace_groups": cross_namespace,
        "cross_namespace_group_count": len(cross_namespace),
        "passed": len(cross_namespace) == 0,
    }


def _structural_loss_contributions(
    meta_records: list[dict[str, Any]],
    *,
    ranking_eligible: np.ndarray,
    old_demo_retention: np.ndarray,
    is_success_reference: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[str, Any]:
    """按样本权重与 eligibility 估算各监督通道的结构占比（无需 forward）。"""
    buckets: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    counts: dict[str, int] = defaultdict(int)

    for i, meta in enumerate(meta_records):
        sd = str(meta.get("source_dataset", "unknown"))
        dk = str(meta.get("demo_key", ""))
        tags = [sd, f"demo:{dk}"]
        if sd == SOURCE_LEGACY_OLD and dk == "demo_4":
            tags.append("legacy_old_demo_4_retention")
        if sd == SOURCE_SUCCESS_REF:
            tags.append("success_reference")
        ft = str(meta.get("source_failure_type", meta.get("failure_type", "")))
        if ft in FAILURE_TYPES:
            tags.append(ft)

        w = float(sample_weight[i])
        ranking_w = w if ranking_eligible[i] > 0.5 and is_success_reference[i] <= 0.5 else 0.0
        focal_w = w if is_success_reference[i] <= 0.5 else 0.0
        retention_w = w * (1.0 + 2.5 * old_demo_retention[i]) if old_demo_retention[i] > 0.5 else 0.0
        sup_w = w

        for tag in tags:
            counts[tag] += 1
            buckets[tag]["total_supervision"] += sup_w
            buckets[tag]["pairwise_ranking_eligible_weight"] += ranking_w
            buckets[tag]["success_focal_eligible_weight"] += focal_w
            buckets[tag]["old_demo_retention_weight"] += retention_w

    def _share(key: str, field: str) -> float:
        total = sum(b[field] for b in buckets.values()) or 1.0
        return 100.0 * buckets.get(key, {}).get(field, 0.0) / total

    rows = []
    for tag in sorted(buckets.keys()):
        b = buckets[tag]
        rows.append(
            {
                "category": tag,
                "sample_count": counts[tag],
                "total_supervision_weight": b["total_supervision"],
                "pairwise_ranking_eligible_weight": b["pairwise_ranking_eligible_weight"],
                "success_focal_eligible_weight": b["success_focal_eligible_weight"],
                "old_demo_retention_weight": b["old_demo_retention_weight"],
            }
        )

    return {
        "rows": rows,
        "success_reference_ranking_share_pct": _share("success_reference", "pairwise_ranking_eligible_weight"),
        "success_reference_focal_share_pct": _share("success_reference", "success_focal_eligible_weight"),
        "legacy_old_demo_4_retention_share_pct": _share("legacy_old_demo_4_retention", "old_demo_retention_weight"),
        "new100_failed_retention_share_pct": _share(SOURCE_NEW100_FAILED, "old_demo_retention_weight"),
        "legacy_old_supervision_share_pct": _share(SOURCE_LEGACY_OLD, "total_supervision"),
        "new100_failed_supervision_share_pct": _share(SOURCE_NEW100_FAILED, "total_supervision"),
    }


def _model_loss_contributions(
    model: PINNV1FRepairModel,
    dataset: V1FRepairDataset,
    meta_records: list[dict[str, Any]],
    device: torch.device,
    physics: V1FPhysicsLossConfig,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    model.eval()
    per_tag: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    with torch.no_grad():
        offset = 0
        for batch in loader:
            bs = len(batch["features"])
            batch_dev = {k: v.to(device) for k, v in batch.items()}
            out = model(batch_dev["features"])
            losses = compute_v1f_100base_r1_losses(out, batch_dev, physics=physics)

            err = (out["E_total"] - batch_dev["target_E_total"]) ** 2
            is_ref = batch_dev.get("is_success_reference", torch.zeros_like(batch_dev["success_flag"]))
            repair_mask = (is_ref <= 0.5).float()
            ranking_eligible = batch_dev.get("ranking_supervision_eligible", torch.ones_like(batch_dev["success_flag"]))
            ranking_eligible = ranking_eligible * repair_mask
            old_ret = batch_dev.get("old_demo_retention", torch.zeros_like(batch_dev["success_flag"]))
            focal = torch.nn.functional.binary_cross_entropy_with_logits(
                out["success_logit"], batch_dev["success_flag"], reduction="none"
            )
            focal_w = focal * repair_mask

            for j in range(bs):
                idx = dataset.indices[offset + j]
                meta = meta_records[idx]
                sd = str(meta.get("source_dataset", ""))
                dk = str(meta.get("demo_key", ""))
                tags = [sd]
                if sd == SOURCE_SUCCESS_REF:
                    tags.append("success_reference")
                if sd == SOURCE_LEGACY_OLD and dk == "demo_4":
                    tags.append("legacy_old_demo_4")

                vals = {
                    "success_focal": float(focal_w[j].item()) if repair_mask[j] > 0.5 else 0.0,
                    "old_demo_retention": float(((1.0 + 2.5 * old_ret[j]) * err[j] * (old_ret[j] > 0.5).float()).item()),
                    "ranking_eligible": float(ranking_eligible[j].item()),
                }
                for tag in tags:
                    for k, v in vals.items():
                        per_tag[tag][k] += v
            offset += bs

    ref_ranking = per_tag.get("success_reference", {}).get("ranking_eligible", 0.0)
    ref_focal = per_tag.get("success_reference", {}).get("success_focal", 0.0)
    total_ranking = sum(v.get("ranking_eligible", 0.0) for v in per_tag.values()) or 1.0
    total_focal = sum(v.get("success_focal", 0.0) for v in per_tag.values()) or 1.0

    return {
        "init_checkpoint_proxy": {
            "success_reference_ranking_eligible_sum": ref_ranking,
            "success_reference_focal_sum": ref_focal,
            "success_reference_ranking_share_pct": 100.0 * ref_ranking / total_ranking,
            "success_reference_focal_share_pct": 100.0 * ref_focal / total_focal,
            "legacy_old_demo_4_retention_sum": per_tag.get("legacy_old_demo_4", {}).get("old_demo_retention", 0.0),
        },
        "aggregate_losses_on_full_dataset": {k: float(v.item()) for k, v in losses.items()},
    }


def run_loss_gate(
    *,
    dataset_npz: Path,
    init_checkpoint: Path,
    output_report: Path,
    val_frac: float = 0.15,
    seed: int = 42,
    skip_model_proxy: bool = False,
) -> dict[str, Any]:
    bundle = load_v1f_npz(dataset_npz)
    meta = bundle["meta"]
    meta_records: list[dict[str, Any]] = meta.get("meta_records", [])

    demo_group_id = bundle["demo_group_id"]
    ranking_eligible = bundle.get("ranking_supervision_eligible", np.ones(len(demo_group_id)))
    old_demo_retention = bundle.get("old_demo_retention", np.zeros(len(demo_group_id)))
    is_success_reference = bundle.get("is_success_reference", np.zeros(len(demo_group_id)))
    sample_weight = bundle.get("sample_weight", np.ones(len(demo_group_id)))

    collision = audit_demo_uid_collisions(meta_records, demo_group_id)
    retention_audit = _audit_retention_labeling(meta_records, old_demo_retention)
    ref_audit = _audit_success_reference_isolation(meta_records, ranking_eligible, is_success_reference)
    pairwise_audit = _audit_pairwise_groups(meta_records, demo_group_id)
    structural = _structural_loss_contributions(
        meta_records,
        ranking_eligible=ranking_eligible,
        old_demo_retention=old_demo_retention,
        is_success_reference=is_success_reference,
        sample_weight=sample_weight,
    )

    train_idx, val_idx = split_indices_by_demo_uid(demo_group_id, val_frac=val_frac, seed=seed)
    split_audit = {
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "train_demo_uids": sorted(
            {str(meta_records[i].get("demo_uid")) for i in train_idx}
        ),
        "val_demo_uids": sorted({str(meta_records[i].get("demo_uid")) for i in val_idx}),
    }

    model_proxy: dict[str, Any] = {}
    if not skip_model_proxy and init_checkpoint.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        physics = V1FPhysicsLossConfig.hundredbase()
        model = PINNV1FRepairModel(input_dim=int(bundle["features"].shape[1])).to(device)
        ckpt = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        ds = V1FRepairDataset(dataset_npz, np.arange(len(bundle["features"])))
        model_proxy = _model_loss_contributions(model, ds, meta_records, device, physics)

    gate_checks = {
        "demo_uid_collision_zero": collision["demo_uid_collision_count"] == 0
        and collision["demo_group_id_cross_namespace_count"] == 0,
        "success_reference_ranking_share_zero": structural["success_reference_ranking_share_pct"] == 0.0,
        "success_reference_focal_share_zero": structural["success_reference_focal_share_pct"] == 0.0,
        "new100_retention_mislabeled_zero": retention_audit["new100_retention_mislabeled_count"] == 0,
        "legacy_old_demo_4_retention_visible": retention_audit["legacy_old_demo_4_retention_count"] > 0,
        "new100_does_not_drown_old_retention": (
            structural["legacy_old_demo_4_retention_share_pct"]
            >= structural["new100_failed_retention_share_pct"]
        ),
        "pairwise_groups_demo_uid_isolated": pairwise_audit["passed"],
        "success_reference_isolation_flags": ref_audit["passed"],
        "retention_labeling": retention_audit["passed"],
    }
    if model_proxy:
        proxy = model_proxy.get("init_checkpoint_proxy", {})
        gate_checks["model_proxy_success_ref_ranking_zero"] = (
            proxy.get("success_reference_ranking_share_pct", 0.0) == 0.0
        )
        gate_checks["model_proxy_success_ref_focal_zero"] = (
            proxy.get("success_reference_focal_share_pct", 0.0) == 0.0
        )

    training_allowed = all(gate_checks.values())
    block_reasons = [k for k, v in gate_checks.items() if not v]

    report: dict[str, Any] = {
        "dataset_version": "V1-F-100Base-R1",
        "dataset_npz": str(dataset_npz),
        "init_checkpoint": str(init_checkpoint),
        "demo_uid_collision_audit": collision,
        "retention_labeling_audit": retention_audit,
        "success_reference_isolation_audit": ref_audit,
        "pairwise_group_audit": pairwise_audit,
        "train_val_split_by_demo_uid": split_audit,
        "structural_loss_contributions": structural,
        "model_proxy_loss_contributions": model_proxy,
        "training_gate": {
            "checks": gate_checks,
            "training_allowed": training_allowed,
            "block_reasons": block_reasons,
        },
        "requirements": {
            "success_reference_ranking_focal_share_pct": 0.0,
            "demo_uid_collision_count": 0,
            "legacy_old_demo_4_retention_must_be_visible": True,
        },
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-F-100Base-R1 pre-train loss contribution gate")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_NPZ)
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_ALIGNED_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_LOSS_GATE_REPORT)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-model-proxy", action="store_true")
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"Dataset missing: {args.dataset}. Run build_v1f_100base_r1_dataset.py first.")

    report = run_loss_gate(
        dataset_npz=args.dataset,
        init_checkpoint=args.init_checkpoint,
        output_report=args.output,
        val_frac=args.val_frac,
        seed=args.seed,
        skip_model_proxy=args.skip_model_proxy,
    )
    gate = report["training_gate"]
    print(json.dumps({"output": str(args.output), "training_allowed": gate["training_allowed"]}, indent=2))
    if not gate["training_allowed"]:
        print("BLOCKED:", gate["block_reasons"], flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
