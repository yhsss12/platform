#!/usr/bin/env python3
"""L20 远程训练 smoke：节点探测 → 创建 N epoch DP 训练 → 轮询状态 → 汇总产物。

用法:
  cd backend
  python tools/verification/l20_remote_smoke_training.py
  python tools/verification/l20_remote_smoke_training.py --epochs 3 --task-name "L20 Remote Smoke 3 Epoch"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.env_loader import ensure_dotenv_loaded, train_node_password_configured  # noqa: E402

ensure_dotenv_loaded(verbose=True)

REPO = BACKEND_ROOT.parent
API = "http://127.0.0.1:8000"
NODE_ID = "l20-172-18-0-73"
REPORT_PATH = REPO / "runs" / "l20_remote_smoke_report.json"


def _find_sample_hdf5() -> Path:
    candidates = sorted(REPO.glob("runs/cable_threading/jobs/*/datasets/dataset.hdf5"))
    if not candidates:
        raise FileNotFoundError("未找到 cable_threading dataset.hdf5，请先完成数据采集")
    return candidates[-1]


def _login() -> str:
    import requests

    for creds in (
        {"username": "Pibot0001", "password": "jinlian1234"},
        {"username": "admin", "password": "admin123"},
    ):
        try:
            resp = requests.post(f"{API}/api/auth/login", json=creds, timeout=30)
            if resp.ok:
                return resp.json()["data"]["access_token"]
        except Exception:
            continue
    raise RuntimeError("登录失败，请确认后端已启动且账号可用")


def _poll_status(
    token: str,
    train_job_id: str,
    *,
    timeout_s: int = 7200,
    interval_s: int = 10,
    on_tick: Optional[Callable[[dict], None]] = None,
) -> dict:
    import requests

    headers = {"Authorization": f"Bearer {token}"}
    deadline = time.time() + timeout_s
    last: dict = {}
    while time.time() < deadline:
        resp = requests.get(
            f"{API}/api/workspace/training/jobs/{train_job_id}/status",
            headers=headers,
            timeout=60,
        )
        resp.raise_for_status()
        last = resp.json()
        status = str(last.get("status") or "").lower()
        print(f"[poll] {train_job_id} status={status} epoch={last.get('epoch')} loss={last.get('loss')} progress={last.get('progress')}")
        if on_tick is not None:
            on_tick(last)
        if status in {"completed", "failed", "backend_unavailable", "canceled"}:
            return last
        time.sleep(interval_s)
    raise TimeoutError(f"训练超时: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description="L20 remote_ssh training smoke")
    parser.add_argument("--epochs", type=int, default=1, help="training epochs (smoke default 1)")
    parser.add_argument("--task-name", type=str, default="", help="optional task display name")
    parser.add_argument("--poll-interval", type=int, default=8, help="status poll interval seconds")
    args = parser.parse_args()

    epochs = max(1, int(args.epochs))
    task_name = (args.task_name or "").strip() or (
        f"L20 Remote Smoke {epochs} Epoch" if epochs != 1 else "L20 Remote Smoke 1 Epoch"
    )

    from app.services.training_node_service import format_node_probe_report, probe_training_node

    if not train_node_password_configured():
        print("错误: 未配置 TRAIN_NODE_L20_PASSWORD 或 TRAIN_NODE_L20_SSH_KEY", file=sys.stderr)
        print("请在 backend/.env 中设置后重试。", file=sys.stderr)
        return 1

    print("=== 节点探测 ===")
    print(format_node_probe_report(NODE_ID))
    probe = probe_training_node(NODE_ID, refresh=True)
    status = probe.get("status")
    if status not in {"available", "busy"}:
        print(f"节点不可用: {probe.get('message')}", file=sys.stderr)
        return 1

    hdf5 = _find_sample_hdf5()
    manifest = {
        "datasetId": f"smoke_{hdf5.parent.parent.name}",
        "datasetName": f"L20 smoke {hdf5.parent.parent.name}",
        "successfulEpisodes": 1,
        "taskType": "cable_threading",
        "artifacts": {"hdf5": str(hdf5.resolve())},
    }

    import requests

    token = _login()
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "datasetId": manifest["datasetId"],
        "datasetManifest": manifest,
        "modelTypeId": "diffusion-policy",
        "downstreamModelType": "DiffusionPolicy",
        "trainingBackend": "diffusion_policy",
        "dataFormat": "HDF5",
        "epochs": epochs,
        "batchSize": 8,
        "learningRate": 0.0001,
        "device": "cuda",
        "deviceLabel": "L20",
        "trainingNodeId": NODE_ID,
        "taskName": task_name,
    }

    print("=== 创建远程训练任务 ===")
    resp = requests.post(f"{API}/api/workspace/training/jobs", json=payload, headers=headers, timeout=120)
    if not resp.ok:
        print(resp.status_code, resp.text, file=sys.stderr)
        return 1
    created = resp.json()
    train_job_id = created["trainJobId"]
    print(f"trainJobId={train_job_id}")

    epoch_snapshots: list[dict] = []
    seen_epochs: set[int] = set()

    def _on_tick(status_payload: dict) -> None:
        epoch = int(status_payload.get("epoch") or 0)
        total = int(status_payload.get("totalEpochs") or epochs)
        if epoch <= 0 or epoch in seen_epochs:
            return
        seen_epochs.add(epoch)
        train_job_dir = REPO / "runs" / "training" / "jobs" / train_job_id
        log_path = train_job_dir / "logs" / "train.log"
        runner_log = train_job_dir / "logs" / "remote_runner.log"
        metrics_path = train_job_dir / "artifacts" / "metrics.jsonl"
        loss_series_len = 0
        if metrics_path.is_file():
            loss_series_len = sum(1 for line in metrics_path.read_text(encoding="utf-8").splitlines() if line.strip())
        epoch_snapshots.append(
            {
                "epoch": epoch,
                "totalEpochs": total,
                "status": status_payload.get("status"),
                "progress": status_payload.get("progress"),
                "loss": status_payload.get("loss"),
                "lossHistoryLen": len(status_payload.get("lossHistory") or []),
                "metricsJsonlPoints": loss_series_len,
                "trainLogEpochLines": sum(
                    1 for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    if f"Epoch {epoch} Loss:" in line
                ) if log_path.is_file() else 0,
                "trainLogHasRunnerNoise": any(
                    "remote command:" in line.lower() or "rsync" in line.lower()
                    for line in (log_path.read_text(encoding="utf-8", errors="replace").splitlines() if log_path.is_file() else [])
                ),
                "runnerLogExists": runner_log.is_file(),
            }
        )
        print(f"[milestone] epoch {epoch}/{total} captured")

    final = _poll_status(
        token,
        train_job_id,
        interval_s=max(3, int(args.poll_interval)),
        on_tick=_on_tick,
    )
    log_resp = requests.get(
        f"{API}/api/workspace/training/jobs/{train_job_id}/log",
        headers=headers,
        timeout=60,
    )
    log_text = log_resp.json().get("log") if log_resp.ok else ""

    train_job_dir = REPO / "runs" / "training" / "jobs" / train_job_id
    status_path = train_job_dir / "status.json"
    status_data = json.loads(status_path.read_text(encoding="utf-8")) if status_path.is_file() else final
    train_config_path = train_job_dir / "config" / "train_config.json"
    train_config = (
        json.loads(train_config_path.read_text(encoding="utf-8")) if train_config_path.is_file() else {}
    )

    checkpoint_path = status_data.get("checkpointPath")
    model_asset_id = status_data.get("modelAssetId")
    registry_path = train_job_dir / "artifacts" / "model_assets_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else []

    report = {
        "trainJobId": train_job_id,
        "taskName": task_name,
        "epochs": epochs,
        "epochSnapshots": epoch_snapshots,
        "nodeProbeStatus": status,
        "finalStatus": final.get("status"),
        "message": final.get("message"),
        "executionMode": train_config.get("executionMode"),
        "trainingNodeId": train_config.get("trainingNodeId"),
        "remoteCommand": status_data.get("command"),
        "remoteHost": status_data.get("remoteHost"),
        "remoteJobDir": status_data.get("remoteJobDir"),
        "checkpointPath": checkpoint_path,
        "modelAssetId": model_asset_id,
        "modelAssetsRegistered": len(registry),
        "logTail": "\n".join((log_text or "").splitlines()[-30:]),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Smoke 结果 ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if str(final.get("status")).lower() == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
