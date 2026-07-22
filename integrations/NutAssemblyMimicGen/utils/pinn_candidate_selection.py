from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _demo_keys(data_group: h5py.Group) -> list[str]:
    return sorted(k for k in data_group.keys() if k.startswith("demo_"))


def _pos_from_mat(mat: np.ndarray) -> np.ndarray:
    return np.asarray(mat[:3, 3], dtype=np.float64)


def _tilt_from_mat(mat: np.ndarray) -> float:
    z_axis = np.asarray(mat[:3, :3], dtype=np.float64)[:, 2]
    return float(np.arccos(np.clip(z_axis[2], -1.0, 1.0)))


def compute_demo_quality_metrics(demo_grp: h5py.Group) -> dict[str, Any]:
    """Score a MimicGen demo for PINN repair candidacy using object_poses when available."""
    metrics: dict[str, Any] = {
        "final_xy_error": None,
        "final_height_error": None,
        "max_xy_error_during_insert": None,
        "nut_to_peg_xy_error_at_insert_start": None,
        "nut_tilt_error": None,
        "action_smoothness_score": None,
        "insertion_stability_score": None,
    }

    dg = demo_grp.get("datagen_info")
    nut_key = None
    peg_key = None
    nut_mats = None
    peg_mats = None
    if dg is not None and "object_poses" in dg:
        op = dg["object_poses"]
        for nk in ("square_nut", "round_nut"):
            if nk in op:
                nut_key = nk
                nut_mats = np.asarray(op[nk], dtype=np.float64)
                break
        for pk in ("square_peg", "round_peg"):
            if pk in op:
                peg_key = pk
                peg_mats = np.asarray(op[pk], dtype=np.float64)
                break

    if nut_mats is not None and peg_mats is not None and len(nut_mats) > 0:
        nut_xy = nut_mats[:, :2, 3]
        peg_xy = peg_mats[:, :2, 3]
        nut_z = nut_mats[:, 2, 3]
        peg_z = peg_mats[:, 2, 3]
        xy_errors = np.linalg.norm(nut_xy - peg_xy, axis=1)
        metrics["final_xy_error"] = float(xy_errors[-1])
        metrics["final_height_error"] = float(nut_z[-1] - peg_z[-1])
        metrics["nut_tilt_error"] = float(_tilt_from_mat(nut_mats[-1]))

        insert_start = max(len(xy_errors) // 2, 1)
        if "subtask_term_signals" in dg:
            sts = dg["subtask_term_signals"]
            for sig_name in ("insert_square_nut", "insert_round_nut", "grasp_square_nut"):
                if sig_name in sts:
                    sig = np.asarray(sts[sig_name], dtype=np.float64).reshape(-1)
                    hits = np.where(sig > 0.5)[0]
                    if len(hits) > 0:
                        insert_start = int(hits[0])
                    break
        metrics["nut_to_peg_xy_error_at_insert_start"] = float(xy_errors[min(insert_start, len(xy_errors) - 1)])
        insert_slice = xy_errors[insert_start:]
        if len(insert_slice) > 0:
            metrics["max_xy_error_during_insert"] = float(np.max(insert_slice))
        tail = xy_errors[-min(20, len(xy_errors)) :]
        metrics["insertion_stability_score"] = float(1.0 / (1.0 + float(np.std(tail))))

    if "actions" in demo_grp:
        actions = np.asarray(demo_grp["actions"], dtype=np.float64)
        if len(actions) > 1:
            deltas = np.diff(actions, axis=0)
            metrics["action_smoothness_score"] = float(1.0 / (1.0 + float(np.mean(np.abs(deltas)))))

    meta_raw = demo_grp.attrs.get("benchmark_episode_metadata") or demo_grp.attrs.get("failure_type")
    if meta_raw and metrics["final_xy_error"] is None:
        try:
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else {}
            if isinstance(meta, dict):
                if meta.get("final_xy_error") is not None:
                    metrics["final_xy_error"] = float(meta["final_xy_error"])
                if meta.get("final_height_error") is not None:
                    metrics["final_height_error"] = float(meta["final_height_error"])
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    failure_type = str(demo_grp.attrs.get("failure_type", "") or "")
    if failure_type in {"alignment_failed", "insertion_failed"}:
        metrics.setdefault("_failure_boost", 1.0)

    return metrics


def _composite_repair_score(metrics: dict[str, Any]) -> float:
    score = 0.0
    xy = metrics.get("final_xy_error")
    if xy is not None:
        score += float(xy) * 1000.0
    insert_xy = metrics.get("nut_to_peg_xy_error_at_insert_start")
    if insert_xy is not None:
        score += float(insert_xy) * 800.0
    max_insert = metrics.get("max_xy_error_during_insert")
    if max_insert is not None:
        score += float(max_insert) * 600.0
    stability = metrics.get("insertion_stability_score")
    if stability is not None:
        score += max(0.0, 1.0 - float(stability)) * 200.0
    if metrics.get("_failure_boost"):
        score += 150.0
    return score


def is_high_error_candidate(
    metrics: dict[str, Any],
    *,
    xy_threshold: float = 0.02,
    insert_start_threshold: float = 0.025,
    stability_threshold: float = 0.35,
) -> bool:
    final_xy = metrics.get("final_xy_error")
    if final_xy is not None and float(final_xy) > xy_threshold:
        return True
    insert_xy = metrics.get("nut_to_peg_xy_error_at_insert_start")
    if insert_xy is not None and float(insert_xy) > insert_start_threshold:
        return True
    stability = metrics.get("insertion_stability_score")
    if stability is not None and float(stability) < stability_threshold:
        return True
    return bool(metrics.get("_failure_boost"))


def select_high_error_candidates(
    hdf5_path: Path,
    *,
    max_candidates: int,
    xy_threshold: float = 0.02,
    insert_start_threshold: float = 0.025,
) -> list[dict[str, Any]]:
    if not hdf5_path.is_file():
        return []
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    with h5py.File(hdf5_path, "r") as f:
        data = f.get("data")
        if data is None:
            return []
        for demo_key in _demo_keys(data):
            grp = data[demo_key]
            metrics = compute_demo_quality_metrics(grp)
            if not is_high_error_candidate(
                metrics,
                xy_threshold=xy_threshold,
                insert_start_threshold=insert_start_threshold,
            ):
                continue
            ranked.append((_composite_repair_score(metrics), demo_key, metrics))
    ranked.sort(key=lambda item: item[0], reverse=True)
    out: list[dict[str, Any]] = []
    for score, demo_key, metrics in ranked[:max_candidates]:
        out.append(
            {
                "candidateId": f"high_error_{demo_key}",
                "demoKey": demo_key,
                "candidateSource": "high_error_generated_demos",
                "score": score,
                "qualityMetrics": metrics,
            }
        )
    return out


def _copy_demo_group(src_grp: h5py.Group, dst_parent: h5py.Group, demo_name: str) -> h5py.Group:
    dst = dst_parent.create_group(demo_name)

    def _copy_item(name: str, obj: h5py.Dataset | h5py.Group) -> None:
        if isinstance(obj, h5py.Dataset):
            dst.create_dataset(name, data=obj[()], compression="gzip")
        else:
            sub = dst.create_group(name)
            for sub_name, sub_obj in obj.items():
                _copy_item(sub_name, sub_obj)

    for key, item in src_grp.items():
        _copy_item(key, item)
    for attr_key, attr_val in src_grp.attrs.items():
        dst.attrs[attr_key] = attr_val
    return dst


def _apply_xy_perturbation(demo_grp: h5py.Group, *, xy_offset: np.ndarray, start_frac: float = 0.55) -> None:
    dg = demo_grp.get("datagen_info")
    if dg is None or "object_poses" not in dg:
        return
    op = dg["object_poses"]
    nut_key = "square_nut" if "square_nut" in op else ("round_nut" if "round_nut" in op else None)
    if nut_key is None:
        return
    mats = np.asarray(op[nut_key], dtype=np.float64).copy()
    start = int(len(mats) * start_frac)
    for i in range(start, len(mats)):
        mats[i, 0, 3] += xy_offset[0]
        mats[i, 1, 3] += xy_offset[1]
    del op[nut_key]
    op.create_dataset(nut_key, data=mats, compression="gzip")


def create_synthetic_perturbation_candidates(
    hdf5_path: Path,
    candidates_dir: Path,
    *,
    max_count: int,
    seed: int,
    existing_demo_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    if not hdf5_path.is_file() or max_count <= 0:
        return []
    existing_demo_keys = existing_demo_keys or set()
    rng = np.random.default_rng(seed)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        data = f.get("data")
        if data is None:
            return []
        demo_keys = [k for k in _demo_keys(data) if k not in existing_demo_keys]
        if not demo_keys:
            demo_keys = _demo_keys(data)
        if not demo_keys:
            return []

        ranked: list[tuple[float, str]] = []
        for demo_key in demo_keys:
            metrics = compute_demo_quality_metrics(data[demo_key])
            xy = metrics.get("final_xy_error")
            ranked.append((float(xy) if xy is not None else 999.0, demo_key))
        ranked.sort(key=lambda item: item[0])

        out: list[dict[str, Any]] = []
        offsets = [(0.03, 0.0), (0.0, 0.03), (-0.025, 0.015), (0.02, -0.02), (0.015, 0.025)]
        for idx in range(min(max_count, len(ranked), len(offsets))):
            _, parent_key = ranked[idx % len(ranked)]
            parent_grp = data[parent_key]
            candidate_id = f"synthetic_{parent_key}_{idx + 1}"
            out_path = candidates_dir / f"{candidate_id}.hdf5"
            xy_offset = np.asarray(offsets[idx % len(offsets)], dtype=np.float64)

            with h5py.File(out_path, "w") as out_f:
                out_data = out_f.create_group("data")
                if "env_args" in data.attrs:
                    out_data.attrs["env_args"] = data.attrs["env_args"]
                demo_grp = _copy_demo_group(parent_grp, out_data, "demo_0")
                _apply_xy_perturbation(demo_grp, xy_offset=xy_offset)
                demo_grp.attrs["candidate_source"] = "synthetic_perturbation_candidates"
                demo_grp.attrs["repair_parent"] = parent_key
                demo_grp.attrs["synthetic_xy_offset"] = json.dumps(xy_offset.tolist())
                out_data.attrs["total"] = 1

            meta = {
                "candidateId": candidate_id,
                "demoKey": parent_key,
                "parentDemoKey": parent_key,
                "candidateSource": "synthetic_perturbation_candidates",
                "syntheticPerturbation": {"xy_offset": xy_offset.tolist(), "start_frac": 0.55},
                "candidateHdf5": str(out_path),
            }
            (candidates_dir / f"{candidate_id}.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            out.append(meta)
    return out


def select_pinn_candidates(
    hdf5_path: Path,
    *,
    config: dict[str, Any],
    candidates_dir: Path,
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    max_candidates = int(config.get("maxCandidates") or 5)
    candidate_sources = config.get("candidateSource") or [
        "high_error_generated_demos",
        "synthetic_perturbation_candidates",
    ]
    xy_threshold = float(config.get("xyErrorThreshold") or 0.02)
    insert_start_threshold = float(config.get("insertStartErrorThreshold") or 0.025)

    candidates: list[dict[str, Any]] = []
    modes_used: list[str] = []

    if "high_error_generated_demos" in candidate_sources or "mimicgen_failed_trials" in candidate_sources:
        high_error = select_high_error_candidates(
            hdf5_path,
            max_candidates=max_candidates,
            xy_threshold=xy_threshold,
            insert_start_threshold=insert_start_threshold,
        )
        candidates.extend(high_error)
        if high_error:
            modes_used.append("high_error_generated_demos")

    if len(candidates) < max_candidates and "synthetic_perturbation_candidates" in candidate_sources:
        synthetic = create_synthetic_perturbation_candidates(
            hdf5_path,
            candidates_dir,
            max_count=max_candidates - len(candidates),
            seed=seed,
            existing_demo_keys={c.get("demoKey", "") for c in candidates},
        )
        candidates.extend(synthetic)
        if synthetic:
            modes_used.append("synthetic_perturbation_candidates")

    if not modes_used:
        modes_used = list(candidate_sources)

    meta = {
        "candidateMode": modes_used,
        "candidateCount": len(candidates[:max_candidates]),
        "qualityScores": [
            {"candidateId": c.get("candidateId"), "score": c.get("score"), "metrics": c.get("qualityMetrics")}
            for c in candidates[:max_candidates]
            if c.get("qualityMetrics")
        ],
    }
    (candidates_dir / "candidates.json").write_text(
        json.dumps(candidates[:max_candidates], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return candidates[:max_candidates], meta
