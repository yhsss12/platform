#!/usr/bin/env python3
"""轮询 remote_ssh 训练任务：拉取日志/metrics 并记录 epoch 里程碑。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.env_loader import ensure_dotenv_loaded

ensure_dotenv_loaded()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_job_id")
    parser.add_argument("--interval", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    from app.services.training_metrics import normalized_training_metrics
    from app.services.training_remote_runner import reconcile_remote_training_job_runtime
    from app.services.training_service import _read_json, read_training_job_log
    from app.services.workspace_runtime_paths import resolve_training_job_root

    job_id = args.train_job_id.strip()
    train_job_dir = resolve_training_job_root(job_id)
    if train_job_dir is None:
        print(json.dumps({"ok": False, "error": "job not found"}))
        return 1

    deadline = time.time() + args.timeout
    milestones: list[dict] = []
    seen_epochs: set[int] = set()

    while time.time() < deadline:
        result = reconcile_remote_training_job_runtime(job_id)
        status = _read_json(train_job_dir / "status.json")
        metrics = normalized_training_metrics(train_job_dir, status)
        epoch = int(metrics.get("epoch") or status.get("epoch") or 0)
        total = int(metrics.get("totalEpochs") or status.get("totalEpochs") or 0)
        loss_series = metrics.get("lossSeries") or []
        log_text = read_training_job_log(job_id)

        if epoch > 0 and epoch not in seen_epochs:
            seen_epochs.add(epoch)
            milestones.append(
                {
                    "epoch": epoch,
                    "totalEpochs": total,
                    "status": status.get("status"),
                    "progress": metrics.get("progress"),
                    "loss": metrics.get("loss"),
                    "lossSeriesLen": len(loss_series),
                    "lossSeries": loss_series,
                }
            )
            print(
                f"[milestone] epoch={epoch}/{total} status={status.get('status')} "
                f"progress={metrics.get('progress')} lossSeries={len(loss_series)}",
                flush=True,
            )

        token = str(status.get("status") or "").lower()
        if token == "completed" or (
            result.get("status") == "completed" and epoch >= total > 0
        ):
            payload = {
                "ok": True,
                "trainJobId": job_id,
                "status": "completed",
                "milestones": milestones,
                "finalStatus": status,
                "metrics": metrics,
                "reconcile": result,
                "logTail": "\n".join(log_text.splitlines()[-20:]),
            }
            out = train_job_dir / "artifacts" / "smoke_monitor_report.json"
            out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 0
        if token == "failed":
            print(json.dumps({"ok": False, "status": status, "reconcile": result}, ensure_ascii=False))
            return 1

        print(
            f"[poll] status={status.get('status')} epoch={epoch}/{total} "
            f"progress={metrics.get('progress')} remote={result.get('message')}",
            flush=True,
        )
        time.sleep(max(5, args.interval))

    print(json.dumps({"ok": False, "error": "timeout", "milestones": milestones}, ensure_ascii=False))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
