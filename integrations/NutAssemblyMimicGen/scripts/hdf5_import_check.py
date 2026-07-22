#!/usr/bin/env python3
"""P0: MimicGen nut_assembly HDF5 下载探测 + 只读 inspector + 平台 metadata 兼容性检查。"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_INTEGRATION_ROOT = _SCRIPT_DIR.parent
_REPO_ROOT = _INTEGRATION_ROOT.parents[1]
_DATA_ROOT = Path(os.environ.get("EAI_DATA_ROOT") or (_REPO_ROOT / "eai-data")).expanduser()
_MIMICGEN_REPO = _REPO_ROOT / "third_party" / "mimicgen"
_DEFAULT_DATASET_DIR = _DATA_ROOT / "runs" / "nut_assembly" / "datasets"
_DEFAULT_OUT = _DATA_ROOT / "runs" / "nut_assembly" / "debug" / "hdf5_inspect_result.json"
_SCHEMA_GAP_OUT = _DATA_ROOT / "runs" / "nut_assembly" / "debug" / "schema_gap_report.md"

CORE_TASK_KEY = "nut_assembly_d0"
SOURCE_TASK_KEY = "nut_assembly"


def _resolve_mimicgen_repo() -> Path | None:
    for candidate in (_MIMICGEN_REPO,):
        if (candidate / "mimicgen" / "__init__.py").is_file():
            return candidate
    return None


def _prepend_paths() -> None:
    mg_repo = _resolve_mimicgen_repo()
    if mg_repo is not None:
        mg_parent = str(mg_repo)
        if mg_parent not in sys.path:
            sys.path.insert(0, mg_parent)
    backend = str(_REPO_ROOT / "backend")
    if backend not in sys.path:
        sys.path.insert(0, backend)


def _registry_info() -> dict[str, Any]:
    _prepend_paths()
    try:
        import mimicgen

        registry = getattr(mimicgen, "DATASET_REGISTRY", {})
        hf_repo = getattr(mimicgen, "HF_REPO_ID", None)
        return {
            "mimicgen_version": getattr(mimicgen, "__version__", None),
            "hf_repo_id": hf_repo,
            "source_registered": SOURCE_TASK_KEY in (registry.get("source") or {}),
            "core_registered": CORE_TASK_KEY in (registry.get("core") or {}),
            "source_entry": (registry.get("source") or {}).get(SOURCE_TASK_KEY),
            "core_entry": (registry.get("core") or {}).get(CORE_TASK_KEY),
        }
    except Exception as exc:
        return {"error": str(exc), "traceback": traceback.format_exc()}


def _download_dataset(
    *,
    dataset_type: str,
    task_key: str,
    download_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    _prepend_paths()
    from mimicgen import DATASET_REGISTRY, HF_REPO_ID

    info: dict[str, Any] = {
        "dataset_type": dataset_type,
        "task_key": task_key,
        "download_dir": str(download_dir),
        "hf_repo_id": HF_REPO_ID,
        "dry_run": dry_run,
        "downloaded": False,
        "local_path": None,
        "error": None,
    }
    try:
        entry = DATASET_REGISTRY[dataset_type][task_key]
        rel_url = entry["url"]
        target = download_dir / dataset_type / Path(rel_url).name
        info["relative_url"] = rel_url
        info["expected_local_path"] = str(target)
        if target.is_file():
            info["downloaded"] = True
            info["local_path"] = str(target)
            info["note"] = "file already exists"
            return info
        if dry_run:
            info["note"] = "dry_run only"
            return info
        download_dir.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            info["error"] = f"huggingface_hub missing: {exc}"
            return info
        cache_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=rel_url,
            repo_type="dataset",
        )
        import shutil

        shutil.copy2(cache_path, target)
        if target.is_file():
            info["downloaded"] = True
            info["local_path"] = str(target)
        else:
            info["error"] = f"download finished but file missing: {target}"
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["traceback"] = traceback.format_exc()
    return info


def _demo_keys(data_grp: Any) -> list[str]:
    return sorted(k for k in data_grp.keys() if str(k).startswith("demo_"))


def _inspect_hdf5(path: Path) -> dict[str, Any]:
    import h5py

    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "demo_count": 0,
        "demo_lengths": {},
        "has_actions": False,
        "has_obs": False,
        "obs_keys_sample": [],
        "has_rewards": False,
        "has_dones": False,
        "has_states": False,
        "has_datagen_info": False,
        "has_object_poses_in_datagen_info": False,
        "object_pose_keys": [],
        "has_env_args": False,
        "env_args_preview": None,
        "platform_metadata_readable": False,
        "platform_dataset_row": None,
        "platform_inspection": None,
        "errors": [],
    }
    if not path.is_file():
        result["errors"].append("file not found")
        return result

    try:
        with h5py.File(path, "r") as handle:
            data = handle.get("data")
            if data is None:
                result["errors"].append("missing /data group")
                return result

            demos = _demo_keys(data)
            result["demo_count"] = len(demos)
            for demo in demos[:20]:
                grp = data[demo]
                result["demo_lengths"][demo] = int(grp.attrs.get("num_samples", len(grp.get("actions", []))))

            if demos:
                d0 = data[demos[0]]
                result["has_actions"] = "actions" in d0
                obs_grp = d0.get("obs")
                result["has_obs"] = obs_grp is not None
                if obs_grp is not None:
                    result["obs_keys_sample"] = sorted(list(obs_grp.keys()))[:30]
                result["has_rewards"] = "rewards" in d0
                result["has_dones"] = "dones" in d0
                result["has_states"] = "states" in d0

                dg = d0.get("datagen_info")
                result["has_datagen_info"] = dg is not None
                if dg is not None and "object_poses" in dg:
                    result["has_object_poses_in_datagen_info"] = True
                    op = dg["object_poses"]
                    if hasattr(op, "keys"):
                        result["object_pose_keys"] = sorted(list(op.keys()))

            for attr in ("env_args", "env_info"):
                if attr in data.attrs:
                    result["has_env_args"] = True
                    raw = data.attrs[attr]
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    try:
                        parsed = json.loads(raw) if isinstance(raw, str) else raw
                        result["env_args_preview"] = {
                            "env_name": parsed.get("env_name") if isinstance(parsed, dict) else None,
                            "type": parsed.get("type") if isinstance(parsed, dict) else None,
                        }
                    except Exception:
                        result["env_args_preview"] = {"raw_type": type(raw).__name__}
                    break
    except Exception as exc:
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["traceback"] = traceback.format_exc()
        return result

    try:
        from app.services.hdf5_platform_metadata import build_dataset_row_from_hdf5, read_hdf5_data_attrs
        from app.services.adapter_layer.hdf5_inspector import inspect_hdf5

        attrs = read_hdf5_data_attrs(path)
        result["platform_metadata_readable"] = bool(attrs)
        result["platform_data_attrs_keys"] = sorted(attrs.keys()) if attrs else []
        inspection = inspect_hdf5(path)
        result["platform_inspection"] = {
            "episode_count": inspection.episode_count,
            "action_dim": inspection.action_dim,
            "observation_keys": inspection.observation_keys[:20],
            "has_reward": inspection.has_reward,
            "has_done": inspection.has_done,
            "joint_action_available": inspection.joint_action_available,
            "warnings": inspection.warnings,
        }
        row = build_dataset_row_from_hdf5(path, manifest={"taskTemplateId": "nut_assembly_single_arm", "taskType": "nut_assembly"})
        result["platform_dataset_row"] = row
    except Exception as exc:
        result["errors"].append(f"platform_metadata: {type(exc).__name__}: {exc}")

    return result


def _write_schema_gap_report(inspect_payload: dict[str, Any], out_path: Path) -> None:
    hdf5 = inspect_payload.get("hdf5_inspection") or {}
    platform = hdf5.get("platform_dataset_row") or {}
    lines = [
        "# NutAssembly HDF5 Schema Gap Report (P0)",
        "",
        "对比对象：MimicGen `nut_assembly_d0.hdf5` vs 平台线缆穿杆 HDF5 schema（`hdf5_platform_schema.py`）。",
        "",
        "## 1. Demo 组织",
        "",
        "| 维度 | MimicGen / robomimic | 平台 cable_threading |",
        "|------|----------------------|----------------------|",
        f"| demo 数量 | {hdf5.get('demo_count', '—')} | 由 job 写入，结构相同 `/data/demo_*` |",
        "| 分组 | `/data/demo_N/actions,obs,rewards,dones[,states,datagen_info]` | 同上 + 平台 attrs |",
        "",
        "## 2. 观测 / 动作",
        "",
        f"- MimicGen obs keys（样例）: `{', '.join(hdf5.get('obs_keys_sample') or []) or '—'}`",
        f"- 平台期望 obs schema: `cable_threading_joint_obs_v1` / `cable_threading_eef_obs_v1`（NutAssembly 尚未定义）",
        f"- MimicGen has_actions: `{hdf5.get('has_actions')}`",
        f"- 平台 joint_actions 派生: `{ (hdf5.get('platform_inspection') or {}).get('joint_action_available') }`",
        "",
        "## 3. rewards / dones / success 语义",
        "",
        f"- MimicGen rewards: `{hdf5.get('has_rewards')}`",
        f"- MimicGen dones: `{hdf5.get('has_dones')}`",
        "- MimicGen 成功语义: episode 级 `success` attr 或仿真 `_check_success()`（非 `final_success`）",
        "- 平台 cable: `final_success` / `ever_success` + `failure_reason` 字符串",
        "",
        "## 4. env_args",
        "",
        f"- MimicGen env_args: `{hdf5.get('has_env_args')}` preview={json.dumps(hdf5.get('env_args_preview'), ensure_ascii=False)}",
        "- 平台 cable HDF5: `env_args` JSON + `taskTemplateId`/`taskType`/`observation_schema` 等 attrs",
        "",
        "## 5. datagen_info / object_poses",
        "",
        f"- datagen_info 存在: `{hdf5.get('has_datagen_info')}`",
        f"- object_poses keys: `{', '.join(hdf5.get('object_pose_keys') or []) or '—'}`",
        "- 平台 cable HDF5: **不写入** datagen_info；PINN repair 实验单独读取 MimicGen 字段",
        "",
        "## 6. 平台 benchmark_episode_metadata",
        "",
        "- 平台写入: `demo.attrs['benchmark_episode_metadata']`（cable 专用 summary）",
        "- MimicGen: 通常不存在该字段",
        "",
        "## 7. 平台导入兼容性（P0 结论）",
        "",
    ]

    can_import = bool(hdf5.get("platform_metadata_readable")) and hdf5.get("has_actions")
    lines.append(f"- **基础 HDF5 可读**: `{hdf5.get('platform_metadata_readable')}`")
    lines.append(f"- **可被 build_dataset_row_from_hdf5 解析**: `{bool(platform)}`")
    if platform:
        lines.append(f"- 默认推断 taskTemplateId: `{platform.get('taskTemplateId')}`（manifest 覆盖前可能落回 cable 默认）")
        lines.append(f"- 默认推断 taskType: `{platform.get('taskType')}`")
    lines.append("")
    if can_import:
        lines.append("**结论**: 文件结构为 robomimic 兼容 HDF5，平台读取器可打开并枚举 demo；但缺少 NutAssembly 专用 schema / taskTemplate 标记，**不能直接当作 cable_threading 数据集使用**。")
    else:
        lines.append("**结论**: 当前环境未能完整读取 HDF5 或文件缺失，需先完成下载与环境依赖安装。")

    lines.extend(
        [
            "",
            "## 8. P1 最小修改建议",
            "",
            "1. 定义 `nut_assembly_*_v1` observation/action/success schema（或 robomimic 原生映射）。",
            "2. `build_dataset_row_from_hdf5` / import service 增加 `nut_assembly` taskType 分支，避免默认回落到 cable。",
            "3. 导入时保留 `datagen_info/object_poses` 供 repair 使用。",
            "4. 评测指标复用 `metric_success_rate_v1`，failure_reason 映射 grasp/insert/timeout。",
            "",
        ]
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    download_dir: Path,
    skip_download: bool,
    dry_run_download: bool,
) -> dict[str, Any]:
    registry = _registry_info()
    downloads: dict[str, Any] = {}
    if not skip_download:
        downloads["core"] = _download_dataset(
            dataset_type="core",
            task_key=CORE_TASK_KEY,
            download_dir=download_dir,
            dry_run=dry_run_download,
        )
        downloads["source"] = _download_dataset(
            dataset_type="source",
            task_key=SOURCE_TASK_KEY,
            download_dir=download_dir,
            dry_run=dry_run_download,
        )

    candidate_paths: list[Path] = []
    for key in ("core", "source"):
        entry = downloads.get(key) or {}
        if entry.get("local_path"):
            candidate_paths.append(Path(entry["local_path"]))
        expected = entry.get("expected_local_path")
        if expected:
            candidate_paths.append(Path(expected))

    # 常见本地缓存位置
    candidate_paths.extend(
        [
            _REPO_ROOT / "mnt" / "data" / "demo.hdf5",
            _REPO_ROOT / "mnt" / "data" / "demo_failed.hdf5",
            download_dir / "core" / "core" / "nut_assembly_d0.hdf5",
            download_dir / "core" / "nut_assembly_d0.hdf5",
            _REPO_ROOT / "third_party" / "mimicgen" / "datasets" / "core" / "nut_assembly_d0.hdf5",
        ]
    )

    hdf5_path = next((p for p in candidate_paths if p.is_file()), None)
    inspection = _inspect_hdf5(hdf5_path) if hdf5_path else {"errors": ["no local nut_assembly_d0.hdf5 found"]}
    if hdf5_path and "nut_assembly_d0" not in hdf5_path.name:
        inspection["note"] = "using local proxy HDF5 (not official nut_assembly_d0 download)"

    payload = {
        "ok": bool(hdf5_path and hdf5_path.is_file() and not inspection.get("errors")),
        "registry": registry,
        "downloads": downloads,
        "hdf5_path": str(hdf5_path) if hdf5_path else None,
        "hdf5_inspection": inspection,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="NutAssembly HDF5 import check (P0)")
    parser.add_argument("--download-dir", type=str, default=str(_DEFAULT_DATASET_DIR))
    parser.add_argument("--output", type=str, default=str(_DEFAULT_OUT))
    parser.add_argument("--schema-gap-output", type=str, default=str(_SCHEMA_GAP_OUT))
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--dry-run-download", action="store_true")
    args = parser.parse_args()

    payload = run(
        download_dir=Path(args.download_dir),
        skip_download=args.skip_download,
        dry_run_download=args.dry_run_download,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    _write_schema_gap_report(payload, Path(args.schema_gap_output))

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"wrote: {out_path}")
    print(f"wrote: {args.schema_gap_output}")
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
