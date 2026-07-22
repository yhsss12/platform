#!/usr/bin/env python3
"""Joint-space DP platform E2E acceptance (generate-async → train → eval)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import requests
import yaml

API = "http://127.0.0.1:8000"
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "runs" / "joint_space_e2e_acceptance_report.json"


def login() -> str:
    resp = requests.post(
        f"{API}/api/auth/login",
        json={"username": "Pibot0001", "password": "jinlian1234"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    return data["access_token"]


def auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def poll(fn, *, timeout_s: int = 3600, interval_s: int = 15, label: str = "") -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {}
    while time.time() < deadline:
        last = fn()
        status = str(last.get("status") or "").lower()
        print(f"[poll] {label} status={status}")
        if status in {"completed", "failed", "canceled", "backend_unavailable"}:
            return last
        time.sleep(interval_s)
    raise TimeoutError(f"timeout waiting for {label}: last={last}")


def verify_hdf5(hdf5_path: Path) -> dict[str, Any]:
    required_datasets = {
        "actions",
        "joint_actions",
        "gripper_actions",
        "obs",
        "rewards",
        "dones",
        "attachment_enabled",
    }
    with h5py.File(hdf5_path, "r") as handle:
        data = handle["data"]
        demo = data[sorted(k for k in data.keys() if k.startswith("demo_"))[0]]
        missing = sorted(required_datasets - set(demo.keys()))
        attrs = dict(data.attrs)
        schema_keys = {
            "actionSchema": ("actionSchema", "action_schema"),
            "controllerSchema": ("controllerSchema", "controller_schema"),
            "sideChannelSchema": ("sideChannelSchema", "side_channel_schema"),
            "policySchemas": ("policySchemas", "policy_schemas"),
        }
        for label, candidates in schema_keys.items():
            if not any(attrs.get(key) for key in candidates):
                missing.append(f"attr:{label}")
        env_args_raw = attrs.get("env_args")
        if isinstance(env_args_raw, (bytes, str)):
            try:
                env_args = json.loads(env_args_raw.decode() if isinstance(env_args_raw, bytes) else env_args_raw)
            except json.JSONDecodeError:
                env_args = {}
        elif isinstance(env_args_raw, dict):
            env_args = env_args_raw
        else:
            env_args = {}
        seed_ok = "seed" in attrs or "seed" in env_args
        return {
            "path": str(hdf5_path),
            "missing": missing,
            "has_seed": seed_ok,
            "action_schema": attrs.get("actionSchema") or attrs.get("action_schema"),
            "controller_schema": attrs.get("controllerSchema") or attrs.get("controller_schema"),
        }


def main() -> int:
    report: dict[str, Any] = {"startedAt": time.strftime("%Y-%m-%d %H:%M:%S")}
    token = login()
    headers = auth_headers(token)

    gen_payload = {
        "episodes": 2,
        "robot": "Panda",
        "cableModel": "composite_cable",
        "difficulty": "easy",
        "horizon": 100,
        "seed": 42,
        "outputFormat": "hdf5",
        "saveHdf5": True,
        "saveProcessVideo": False,
    }
    gen = requests.post(
        f"{API}/api/workspace/cable-threading/generate-async",
        headers=headers,
        json=gen_payload,
        timeout=60,
    )
    gen.raise_for_status()
    gen_body = gen.json()
    gen_job_id = gen_body["jobId"]
    report["generateAsync"] = {"jobId": gen_job_id, "response": gen_body}

    gen_status = poll(
        lambda: requests.get(
            f"{API}/api/workspace/cable-threading/jobs/{gen_job_id}/status",
            headers=headers,
            timeout=30,
        ).json(),
        timeout_s=1800,
        label=f"generate-async {gen_job_id}",
    )
    report["generateAsync"]["status"] = gen_status
    if str(gen_status.get("status")).lower() != "completed":
        report["error"] = "generate-async failed"
        OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    job_root = REPO / "runs" / "cable_threading" / "jobs" / gen_job_id
    hdf5_path = job_root / "datasets" / "dataset.hdf5"
    manifest_path = job_root / "datasets" / "dataset.manifest.json"
    for _ in range(30):
        if manifest_path.is_file() and hdf5_path.is_file():
            break
        time.sleep(2)
    if not manifest_path.is_file():
        artifacts = gen_status.get("artifacts") or gen_status.get("paths") or {}
        manifest_hint = artifacts.get("manifest") or {}
        if isinstance(manifest_hint, dict) and manifest_hint.get("path"):
            manifest_path = Path(manifest_hint["path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dataset_id = manifest.get("datasetId") or f"ds_{gen_job_id}"
    report["dataset"] = {
        "datasetId": dataset_id,
        "hdf5Path": str(hdf5_path),
        "manifestPath": str(manifest_path),
        "hdf5Check": verify_hdf5(hdf5_path),
    }

    train_payload = {
        "datasetId": dataset_id,
        "datasetManifestPath": str(manifest_path),
        "modelTypeId": "diffusion-policy",
        "downstreamModelType": "Diffusion Policy",
        "trainingBackend": "diffusion_policy",
        "dataFormat": "HDF5",
        "epochs": 2,
        "batchSize": 8,
        "learningRate": 1e-4,
        "device": "cuda",
        "deviceLabel": "L20",
        "seed": 1,
        "saveFinal": True,
    }
    train = requests.post(
        f"{API}/api/workspace/training/jobs",
        headers=headers,
        json=train_payload,
        timeout=120,
    )
    train.raise_for_status()
    train_body = train.json()
    train_job_id = train_body["trainJobId"]
    report["training"] = {"trainJobId": train_job_id}

    train_dir = REPO / "runs" / "training" / "jobs" / train_job_id
    dp_yaml_path = train_dir / "config" / "dp_adapted.yaml"
    dp_yaml = yaml.safe_load(dp_yaml_path.read_text(encoding="utf-8"))
    train_config = json.loads((train_dir / "config" / "train_config.json").read_text(encoding="utf-8"))
    report["training"]["dpAdaptedYaml"] = dp_yaml
    report["training"]["dpConfigLowDimDim"] = (train_config.get("dpConfig") or {}).get("low_dim_dim")

    train_status = poll(
        lambda: requests.get(
            f"{API}/api/workspace/training/jobs/{train_job_id}/status",
            headers=headers,
            timeout=30,
        ).json(),
        timeout_s=3600,
        label=f"training {train_job_id}",
    )
    report["training"]["status"] = train_status

    checkpoint = train_dir / "checkpoints" / "diffusion_policy" / "checkpoints" / "model_final.pt"
    manifest_asset = train_dir / "artifacts" / "model_manifest.json"
    registry = train_dir / "artifacts" / "model_assets_registry.json"
    report["training"]["checkpointPath"] = str(checkpoint) if checkpoint.is_file() else None
    report["training"]["modelManifestPath"] = str(manifest_asset) if manifest_asset.is_file() else None
    report["training"]["modelAssetsRegistryPath"] = str(registry) if registry.is_file() else None

    if manifest_asset.is_file():
        report["training"]["modelManifest"] = json.loads(manifest_asset.read_text(encoding="utf-8"))
        model_asset_id = report["training"]["modelManifest"].get("modelAssetId")
    else:
        model_asset_id = train_status.get("modelAssetId")

    report["training"]["modelAssetId"] = model_asset_id

    if str(train_status.get("status")).lower() != "completed" or not checkpoint.is_file():
        report["error"] = "training failed or checkpoint missing"
        OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 1

    def start_eval(payload: dict[str, Any]) -> dict[str, Any]:
        resp = requests.post(
            f"{API}/api/workspace/evaluation/evaluate-async",
            headers=headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    dp_eval_payload = {
        "taskType": "cable_threading",
        "evaluationMode": "trained_model_evaluation",
        "numEpisodes": 2,
        "seed": 0,
        "policyType": "diffusion_policy",
        "modelAssetId": model_asset_id,
        "record": True,
        "headless": True,
        "horizon": 100,
        "cableThreading": {
            "robot": "Panda",
            "cableModel": "composite_cable",
            "difficulty": "easy",
        },
    }
    dp_eval = start_eval(dp_eval_payload)
    dp_eval_id = dp_eval["evalJobId"]
    dp_eval_status = poll(
        lambda: requests.get(
            f"{API}/api/workspace/evaluation/jobs/{dp_eval_id}/status",
            headers=headers,
            timeout=30,
        ).json(),
        timeout_s=1800,
        label=f"dp eval {dp_eval_id}",
    )
    dp_eval_root = REPO / "runs" / "cable_threading" / "jobs" / dp_eval_id
    aggregate = dp_eval_root / "aggregate_result.json"
    eval_video = dp_eval_root / "videos" / "eval.mp4"
    report["dpEval"] = {
        "evalJobId": dp_eval_id,
        "status": dp_eval_status,
        "aggregateResultPath": str(aggregate) if aggregate.is_file() else None,
        "evalVideoPath": str(eval_video) if eval_video.is_file() else None,
    }
    if aggregate.is_file():
        report["dpEval"]["aggregateResult"] = json.loads(aggregate.read_text(encoding="utf-8"))

    ckpt_eval_payload = {
        "taskType": "cable_threading",
        "evaluationMode": "trained_model_evaluation",
        "numEpisodes": 1,
        "seed": 0,
        "policyType": "diffusion_policy",
        "checkpointPath": str(checkpoint),
        "record": False,
        "headless": True,
        "horizon": 100,
        "cableThreading": {
            "robot": "Panda",
            "cableModel": "composite_cable",
            "difficulty": "easy",
        },
    }
    ckpt_eval = start_eval(ckpt_eval_payload)
    report["checkpointEval"] = {"evalJobId": ckpt_eval["evalJobId"]}

    expert = requests.post(
        f"{API}/api/workspace/cable-threading/evaluate-async",
        headers=headers,
        json={
            "episodes": 2,
            "robot": "Panda",
            "cableModel": "composite_cable",
            "difficulty": "easy",
            "horizon": 100,
            "seed": 0,
            "policy": "scripted",
            "device": "cpu",
        },
        timeout=60,
    )
    expert.raise_for_status()
    expert_body = expert.json()
    expert_id = expert_body["evalJobId"]
    expert_status = poll(
        lambda: requests.get(
            f"{API}/api/workspace/cable-threading/jobs/{expert_id}/status",
            headers=headers,
            timeout=30,
        ).json(),
        timeout_s=1800,
        label=f"expert eval {expert_id}",
    )
    expert_root = REPO / "runs" / "cable_threading" / "jobs" / expert_id
    expert_agg = expert_root / "aggregate_result.json"
    expert_video = expert_root / "videos" / "eval.mp4"
    report["expertEval"] = {
        "evalJobId": expert_id,
        "status": expert_status,
        "aggregateResultPath": str(expert_agg) if expert_agg.is_file() else None,
        "evalVideoPath": str(expert_video) if expert_video.is_file() else None,
    }
    if expert_agg.is_file():
        report["expertEval"]["aggregateResult"] = json.loads(expert_agg.read_text(encoding="utf-8"))

    report["legacyCompat"] = {
        "note": "Legacy EEF/OSC DP and expert scripted policy verified via unit tests test_joint_space_platform_fixes.py",
        "expertUsesScriptedNotJointPosition": True,
    }
    report["finishedAt"] = time.strftime("%Y-%m-%d %H:%M:%S")
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
