"""V1-F-100Base-R1：demo_uid namespace 隔离与路径常量。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from v1f_100base_utils import (  # noqa: E402
    ALIGNED_ORIGINAL_JSONL,
    DEFAULT_ALIGNED_MODEL,
    DEFAULT_CEM_REPORT,
    DEFAULT_FAILED_HDF5,
    OLD_DEMO_KEYS,
    OLD_STABLE_SOURCE_PREFIXES,
    NEW_FAILED_SOURCE_PREFIXES,
    SUCCESS_REFERENCE_SOURCES,
)

_V1F_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1F_DIR.parent.parent

SOURCE_LEGACY_OLD = "legacy_old"
SOURCE_NEW100_FAILED = "new100_failed"
SOURCE_SUCCESS_REF = "success_ref"

ALL_SOURCE_DATASETS = (SOURCE_LEGACY_OLD, SOURCE_NEW100_FAILED, SOURCE_SUCCESS_REF)

DEFAULT_100BASE_R1_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1f_100base_r1"
DEFAULT_100BASE_INPUT = _EXPERIMENT_DIR / "outputs" / "v1f_100base"
DEFAULT_DATASET_NPZ = DEFAULT_100BASE_R1_OUTPUT / "repair_parameter_dataset_v1f_100base_r1.npz"
DEFAULT_LOSS_GATE_REPORT = DEFAULT_100BASE_R1_OUTPUT / "pretrain_loss_contribution_report.json"
DEFAULT_TRAINED_MODEL = DEFAULT_100BASE_R1_OUTPUT / "trained_model" / "model_v1f_100base_r1.pt"
DEFAULT_EVAL_DIR = DEFAULT_100BASE_R1_OUTPUT / "evaluation"
DEFAULT_EVAL_REPORT = DEFAULT_EVAL_DIR / "quick_validation_report.json"
DEFAULT_CANDIDATE_MANIFEST = DEFAULT_EVAL_DIR / "validation_candidate_manifest.json"

# Reuse 100Base rollout artifacts unless overridden
DEFAULT_FAILED_ROLLOUT = DEFAULT_100BASE_INPUT / "failed_rollout_samples.jsonl"
DEFAULT_TARGETED_ROLLOUT = DEFAULT_100BASE_INPUT / "targeted_rollout_samples.jsonl"
DEFAULT_SUCCESS_REFERENCE = DEFAULT_100BASE_INPUT / "success_reference_samples.jsonl"
DEFAULT_REPAIRABILITY = DEFAULT_100BASE_INPUT / "repairability_audit" / "repairability_report.json"
DEFAULT_AUDIT_REPORT = DEFAULT_100BASE_INPUT / "audit" / "new_demo_audit_report.json"

RANKING_GATE_MIN_TOP20_OVERLAP = 0.8


def _is_old_stable_source(source: str) -> bool:
    return any(source.startswith(p) or source == p for p in OLD_STABLE_SOURCE_PREFIXES)


def _is_new_failed_source(source: str) -> bool:
    return any(source.startswith(p) for p in NEW_FAILED_SOURCE_PREFIXES)


def _is_success_reference_source(source: str) -> bool:
    return source in SUCCESS_REFERENCE_SOURCES or "success_reference" in source


def infer_source_dataset(meta: dict[str, Any]) -> str:
    if meta.get("is_success_reference") or _is_success_reference_source(str(meta.get("source", ""))):
        return SOURCE_SUCCESS_REF
    if meta.get("source_dataset"):
        return str(meta["source_dataset"])
    source = str(meta.get("source", ""))
    if _is_new_failed_source(source):
        return SOURCE_NEW100_FAILED
    if meta.get("is_old_demo") or _is_old_stable_source(source):
        return SOURCE_LEGACY_OLD
    return SOURCE_NEW100_FAILED


def make_demo_uid(source_dataset: str, demo_key: str) -> str:
    return f"{source_dataset}:{demo_key}"


def assign_r1_meta_fields(meta: dict[str, Any]) -> dict[str, Any]:
    """Ensure source_dataset, demo_key, demo_uid on meta."""
    out = dict(meta)
    demo_key = str(out.get("demo_key", "unknown"))
    source_dataset = infer_source_dataset(out)
    out["source_dataset"] = source_dataset
    out["demo_key"] = demo_key
    out["demo_uid"] = make_demo_uid(source_dataset, demo_key)
    if source_dataset == SOURCE_SUCCESS_REF:
        out["is_success_reference"] = True
        out["is_calibration_only"] = True
    return out


def legacy_old_retention_eligible(meta: dict[str, Any]) -> bool:
    return infer_source_dataset(meta) == SOURCE_LEGACY_OLD and str(meta.get("demo_key", "")) in OLD_DEMO_KEYS


def build_demo_uid_group_ids(meta_records: list[dict[str, Any]]) -> tuple[np.ndarray, dict[str, int]]:
    import numpy as np

    uid_to_id: dict[str, int] = {}
    group_ids = np.full(len(meta_records), -1, dtype=np.int64)
    for i, meta in enumerate(meta_records):
        uid = str(meta.get("demo_uid", make_demo_uid(infer_source_dataset(meta), meta.get("demo_key", ""))))
        if uid not in uid_to_id:
            uid_to_id[uid] = len(uid_to_id)
        group_ids[i] = uid_to_id[uid]
    return group_ids, uid_to_id


def split_indices_by_demo_uid(
    demo_group_id: np.ndarray,
    *,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """按 demo_uid (demo_group_id) 分层划分 train/val，同一 uid 不跨 split。"""
    import numpy as np

    rng = np.random.default_rng(seed)
    uid_to_indices: dict[int, list[int]] = {}
    for i, gid in enumerate(demo_group_id):
        uid_to_indices.setdefault(int(gid), []).append(i)
    uids = list(uid_to_indices.keys())
    rng.shuffle(uids)
    n_val_uids = max(1, int(round(len(uids) * val_frac))) if len(uids) >= 3 else 1
    val_uids = set(uids[:n_val_uids])
    train_idx: list[int] = []
    val_idx: list[int] = []
    for gid, idxs in uid_to_indices.items():
        if gid in val_uids:
            val_idx.extend(idxs)
        else:
            train_idx.extend(idxs)
    if not train_idx:
        train_idx, val_idx = val_idx, train_idx
    return np.array(train_idx, dtype=np.int64), np.array(val_idx, dtype=np.int64)


def audit_demo_uid_collisions(meta_records: list[dict[str, Any]], demo_group_id: np.ndarray) -> dict[str, Any]:
    """检查 legacy_old:demo_4 与 new100_failed:demo_4 是否共享 group id。"""
    uid_to_gid: dict[str, set[int]] = {}
    for i, meta in enumerate(meta_records):
        uid = str(meta.get("demo_uid", ""))
        uid_to_gid.setdefault(uid, set()).add(int(demo_group_id[i]))
    collisions = [uid for uid, gids in uid_to_gid.items() if len(gids) > 1]
    # cross-namespace same demo_key collision
    by_key: dict[str, set[str]] = {}
    for meta in meta_records:
        dk = str(meta.get("demo_key", ""))
        by_key.setdefault(dk, set()).add(str(meta.get("source_dataset", "")))
    mixed_keys = {dk: sorted(sds) for dk, sds in by_key.items() if len(sds) > 1 and SOURCE_LEGACY_OLD in sds and SOURCE_NEW100_FAILED in sds}
    same_gid_cross = []
    gid_to_uids: dict[int, set[str]] = {}
    for i, meta in enumerate(meta_records):
        gid = int(demo_group_id[i])
        gid_to_uids.setdefault(gid, set()).add(str(meta.get("demo_uid", "")))
    for gid, uids in gid_to_uids.items():
        if len(uids) > 1:
            same_gid_cross.append({"demo_group_id": gid, "demo_uids": sorted(uids)})
    return {
        "demo_uid_collision_count": len(collisions),
        "demo_group_id_cross_namespace_count": len(same_gid_cross),
        "demo_key_namespace_mixing": mixed_keys,
        "passed": len(collisions) == 0 and len(same_gid_cross) == 0,
    }
