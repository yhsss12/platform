"""远程 SSH 训练 runner：同步 job 产物、远端执行、回拉日志与 status。"""

from __future__ import annotations

import logging
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from app.services.training_node_service import TrainingNodeConfig
from app.services.training_node_ssh import TrainingNodeSSHClient, TrainingNodeSSHError

logger = logging.getLogger(__name__)

REMOTE_BUSY_WAIT_SEC = 120
REMOTE_BUSY_POLL_SEC = 10
REMOTE_POLL_INTERVAL_SEC = 2
REMOTE_SSH_RETRY_SEC = 5
REMOTE_MAX_CONSECUTIVE_SSH_ERRORS = 60


def remote_training_job_dir(node: TrainingNodeConfig, train_job_id: str) -> str:
    """Resolve remote data path while preserving nodes on the legacy layout."""
    remote_data_root = node.data_root.rstrip("/")
    if remote_data_root:
        return f"{remote_data_root}/runs/training/jobs/{train_job_id}"
    return f"{node.workdir.rstrip('/')}/runs/training/jobs/{train_job_id}"


def execute_remote_training_job(
    *,
    train_job_id: str,
    train_job_dir: Path,
    manifest: dict[str, Any],
    train_config: dict[str, Any],
    node: TrainingNodeConfig,
    resolved_backend: str,
    hdf5_path: Path | list[Path] | None,
    total_epochs: int,
    build_train_command: Callable[..., list[str]],
    resolve_device: Callable[[str], str],
    project_root: Path,
    cable_working_dir: Path,
    dual_arm_working_dir: Path,
    update_status: Callable[[Path, dict[str, Any]], dict[str, Any]],
    register_model_manifest: Callable[..., Any],
    sync_workspace_job: Callable[[str], Any],
    register_running_proc: Callable[[Any], None],
    unregister_running_proc: Callable[[], None],
    running_procs: dict[str, Any],
    finalize_training_job_sync: Callable[[str], Any],
    find_checkpoint: Callable[[Path], Optional[Path]],
    backend_framework_meta: Callable[[str], tuple[str, str]],
) -> None:
    from app.services.checkpoint_registry import parse_save_policy, register_checkpoint_assets
    from app.services.training_metrics import (
        append_metrics_point,
        parse_training_logs,
        sync_metrics_from_logs,
    )
    from app.services.training_node_service import probe_training_node, _missing_workdir_message

    train_log_path = train_job_dir / "logs" / "train.log"
    runner_log_path = train_job_dir / "logs" / "remote_runner.log"
    checkpoints_dir = train_job_dir / "checkpoints"
    backend_out = checkpoints_dir / resolved_backend
    backend_out.mkdir(parents=True, exist_ok=True)

    remote_root = node.workdir.rstrip("/")
    remote_job_dir = remote_training_job_dir(node, train_job_id)
    remote_log = f"{remote_job_dir}/logs/train.log"
    remote_status = f"{remote_job_dir}/status.json"
    remote_pid_file = f"{remote_job_dir}/artifacts/remote_train.pid"

    def _read_json(path: Path) -> dict[str, Any]:
        import json

        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _runner_log(message: str) -> None:
        runner_log_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with runner_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] {message}\n")

    def _fail(message: str) -> None:
        update_status(train_job_dir, {"status": "failed", "message": message, "progress": 0.0})
        _runner_log(f"FAILED: {message}")

    probe = probe_training_node(node.node_id, refresh=True)
    if probe.get("status") == "misconfigured":
        _fail(probe.get("message") or _missing_workdir_message(node.workdir))
        return
    if probe.get("status") == "unreachable":
        _fail(probe.get("message") or "训练节点不可连接")
        return

    if probe.get("status") == "busy":
        update_status(
            train_job_dir,
            {
                "status": "queued",
                "message": f"等待 GPU 空闲：{probe.get('message')}",
                "progress": 0.0,
            },
        )
        waited = 0
        while waited < REMOTE_BUSY_WAIT_SEC:
            time.sleep(REMOTE_BUSY_POLL_SEC)
            waited += REMOTE_BUSY_POLL_SEC
            probe = probe_training_node(node.node_id, refresh=True)
            if probe.get("status") in {"unreachable", "misconfigured"}:
                _fail(probe.get("message") or "训练节点不可用")
                return
            if probe.get("status") == "available":
                break
        if probe.get("status") == "busy":
            _fail(probe.get("message") or "训练节点 GPU 忙碌，请稍后重试")
            return

    hdf5_paths = hdf5_path if isinstance(hdf5_path, list) else ([hdf5_path] if hdf5_path else [])
    remote_hdf5_paths: list[Path] = []

    train_log_path.parent.mkdir(parents=True, exist_ok=True)
    if not train_log_path.is_file():
        train_log_path.write_text("", encoding="utf-8")
    _runner_log(f"remote job init trainJobId={train_job_id} node={node.node_id}")

    try:
        with TrainingNodeSSHClient(node) as client:
            client.run(
                f"mkdir -p {shlex.quote(remote_job_dir)}/config "
                f"{shlex.quote(remote_job_dir)}/logs "
                f"{shlex.quote(remote_job_dir)}/artifacts "
                f"{shlex.quote(remote_job_dir)}/checkpoints",
                timeout=30,
            )

            _runner_log("uploading config/artifacts to remote")
            _sftp_sync_dir(client, train_job_dir / "config", f"{remote_job_dir}/config", log=_runner_log)
            _sftp_sync_dir(
                client,
                train_job_dir / "artifacts",
                f"{remote_job_dir}/artifacts",
                skip={"remote_train.pid"},
                log=_runner_log,
            )

            for idx, local_hdf5 in enumerate(hdf5_paths):
                if local_hdf5 is None or not Path(local_hdf5).is_file():
                    continue
                remote_name = "dataset.hdf5" if len(hdf5_paths) == 1 else f"dataset_{idx}.hdf5"
                remote_hdf5 = f"{remote_job_dir}/artifacts/{remote_name}"
                _runner_log(f"uploading dataset {local_hdf5.name} -> {remote_hdf5}")
                _sftp_upload_file(client, Path(local_hdf5), remote_hdf5)
                remote_hdf5_paths.append(Path(remote_hdf5))

            remote_manifest = dict(manifest)
            if remote_hdf5_paths:
                artifacts = dict(remote_manifest.get("artifacts") or {})
                artifacts["hdf5"] = str(remote_hdf5_paths[0])
                if len(remote_hdf5_paths) > 1:
                    artifacts["hdf5Paths"] = [str(p) for p in remote_hdf5_paths]
                remote_manifest["artifacts"] = artifacts
                _sftp_write_json(client, f"{remote_job_dir}/artifacts/dataset_manifest.json", remote_manifest)

            remote_train_config = dict(train_config)
            remote_train_config["trainingNodeId"] = node.node_id
            remote_train_config["executionMode"] = "remote_ssh"
            remote_train_config["remoteHost"] = node.host
            remote_train_config["remoteJobDir"] = remote_job_dir
            if remote_hdf5_paths:
                remote_train_config["datasetHdf5Paths"] = [str(p) for p in remote_hdf5_paths]
            _sftp_write_json(client, f"{remote_job_dir}/config/train_config.json", remote_train_config)

            remote_backend_out = f"{remote_job_dir}/checkpoints/{resolved_backend}"
            client.run(f"mkdir -p {shlex.quote(remote_backend_out)}", timeout=15)

            local_cmd = build_train_command(
                backend=resolved_backend,
                hdf5_path=remote_hdf5_paths if len(remote_hdf5_paths) > 1 else (remote_hdf5_paths[0] if remote_hdf5_paths else Path("/dev/null")),
                out_dir=Path(remote_backend_out),
                train_config=remote_train_config,
            )

            # Remote nodes keep their own repository checkout, which can lag behind
            # the platform backend.  Always upload the concrete Python entry point
            # selected by this backend before launching so a job cannot silently run
            # stale training logic (the datasets/config are already job-scoped).
            if len(local_cmd) >= 2:
                local_entrypoint = Path(local_cmd[1])
                try:
                    entrypoint_relative = local_entrypoint.resolve().relative_to(project_root.resolve())
                except (OSError, ValueError):
                    entrypoint_relative = None
                if entrypoint_relative is not None and local_entrypoint.is_file():
                    remote_entrypoint = f"{remote_root}/{entrypoint_relative.as_posix()}"
                    _runner_log(
                        f"uploading training entrypoint {entrypoint_relative.as_posix()} -> {remote_entrypoint}"
                    )
                    _sftp_upload_file(client, local_entrypoint, remote_entrypoint)

            local_cmd = _remap_command_paths(
                local_cmd,
                project_root=project_root,
                cable_working_dir=cable_working_dir,
                dual_arm_working_dir=dual_arm_working_dir,
                remote_root=remote_root,
                node=node,
            )

            working_dir = (
                f"{remote_root}/integrations/DualArmCableManipulation"
                if resolved_backend == "torch_bc"
                else remote_root
                if resolved_backend == "pi0"
                else f"{remote_root}/integrations/CableThreadingMVP"
            )

            shell_cmd = " ".join(shlex.quote(part) for part in local_cmd)
            client.run(f"mkdir -p {shlex.quote(remote_job_dir)}/logs", timeout=10)
            launch = (
                f"cd {shlex.quote(working_dir)} && "
                f"nohup {shell_cmd} >> {shlex.quote(remote_log)} 2>&1 & echo $! > {shlex.quote(remote_pid_file)}"
            )
            _runner_log(f"launch remote command: {shell_cmd}")
            rc, out, err = client.run(launch, timeout=30)
            if rc != 0:
                _fail(f"远程训练进程启动失败: {err or out}")
                return

            rc, out, err = client.run(f"cat {shlex.quote(remote_pid_file)} 2>/dev/null", timeout=10)
            remote_pid = (out or "").strip()
            if not remote_pid.isdigit():
                _fail("远程训练进程 PID 获取失败")
                return
            _runner_log(f"remote pid recorded: {remote_pid}")

            update_status(
                train_job_dir,
                {
                    "status": "starting",
                    "message": f"远端训练进程已启动（{node.label}），等待首批训练日志",
                    "trainingBackendResolved": resolved_backend,
                    "command": local_cmd,
                    "trainingNodeId": node.node_id,
                    "executionMode": "remote_ssh",
                    "remoteHost": node.host,
                    "remoteJobDir": remote_job_dir,
                    "remotePid": int(remote_pid),
                    "processPid": None,
                },
            )

            class _RemoteProc:
                def __init__(self) -> None:
                    self.returncode: Optional[int] = None

                def poll(self) -> Optional[int]:
                    if _remote_training_process_active(client, train_job_id, remote_pid_file):
                        return None
                    self.returncode = _infer_remote_exit_code(client, remote_log)
                    return self.returncode

            remote_proc = _RemoteProc()
            register_running_proc(remote_proc)
            running_procs[train_job_id] = remote_proc

            framework_label, model_type = backend_framework_meta(resolved_backend)
            sync_counter = 0
            log_offset = 0
            consecutive_ssh_errors = 0

            try:
                while True:
                    try:
                        active = _remote_training_process_active(client, train_job_id, remote_pid_file)
                        log_offset = _incremental_sync_remote_file(
                            client, remote_log, train_log_path, log_offset
                        )
                        _incremental_sync_remote_file(
                            client, remote_status, train_job_dir / "status.json", 0, overwrite=True
                        )
                        epoch, loss = parse_training_logs(train_job_dir, total_epochs)
                        if epoch > 0 and resolved_backend not in {"act", "pi0"}:
                            append_metrics_point(train_job_dir, epoch=epoch, loss=loss)
                        status_snapshot = _read_json(train_job_dir / "status.json")
                        sync_metrics_from_logs(
                            train_job_dir,
                            status_snapshot or {"totalEpochs": total_epochs, "epoch": epoch},
                        )
                        progress = min(0.99, epoch / total_epochs) if total_epochs > 0 else 0.0
                        from app.services.training_job_status import training_activity_detected

                        has_activity = epoch > 0 or loss is not None or training_activity_detected(train_job_dir)
                        update_status(
                            train_job_dir,
                            {
                                "status": "running" if has_activity else "starting",
                                "epoch": epoch,
                                "totalEpochs": total_epochs,
                                "progress": progress if has_activity else 0.0,
                                "loss": loss,
                                "trainingNodeId": node.node_id,
                                "executionMode": "remote_ssh",
                                "message": (
                                    f"远程训练进行中（{node.label}）"
                                    if has_activity
                                    else f"远端训练进程运行中（{node.label}），等待首批训练日志"
                                ),
                            },
                        )
                        if sync_counter % 4 == 0:
                            status_data = _read_json(train_job_dir / "status.json")
                            try:
                                register_checkpoint_assets(
                                    train_job_dir=train_job_dir,
                                    train_job_id=train_job_id,
                                    manifest=manifest,
                                    train_config=train_config,
                                    status=status_data,
                                    resolved_backend=resolved_backend,
                                    framework_label=framework_label,
                                    model_type=model_type,
                                    register_final=False,
                                )
                            except Exception as reg_exc:
                                _runner_log(f"checkpoint registry skipped during poll: {reg_exc}")
                        sync_counter += 1
                        consecutive_ssh_errors = 0

                        if not active:
                            remote_proc.returncode = _infer_remote_exit_code(client, remote_log)
                            break

                        time.sleep(REMOTE_POLL_INTERVAL_SEC)
                    except TrainingNodeSSHError as exc:
                        consecutive_ssh_errors += 1
                        _runner_log(f"SSH poll error ({consecutive_ssh_errors}): {exc}")
                        if _local_checkpoint_ready(backend_out, checkpoints_dir, find_checkpoint):
                            _runner_log("local checkpoint detected during SSH outage; treating as completed")
                            remote_proc.returncode = 0
                            break
                        if consecutive_ssh_errors >= REMOTE_MAX_CONSECUTIVE_SSH_ERRORS:
                            if _try_recover_remote_completion(
                                node,
                                train_job_id,
                                remote_job_dir,
                                remote_log,
                                train_log_path,
                                train_job_dir / "status.json",
                                backend_out,
                                checkpoints_dir,
                                find_checkpoint,
                                _runner_log,
                            ):
                                remote_proc.returncode = 0
                                break
                            _fail(f"远程 SSH 连接中断且无法确认训练结果: {exc}")
                            return
                        time.sleep(REMOTE_SSH_RETRY_SEC)
                        client.close()
                        client._connect()

                _finalize_remote_training_job(
                    client=client,
                    train_job_id=train_job_id,
                    train_job_dir=train_job_dir,
                    manifest=manifest,
                    train_config=train_config,
                    node=node,
                    resolved_backend=resolved_backend,
                    framework_label=framework_label,
                    model_type=model_type,
                    total_epochs=total_epochs,
                    remote_job_dir=remote_job_dir,
                    remote_log=remote_log,
                    remote_status=remote_status,
                    train_log_path=train_log_path,
                    checkpoints_dir=checkpoints_dir,
                    backend_out=backend_out,
                    return_code=remote_proc.returncode if remote_proc.returncode is not None else 1,
                    update_status=update_status,
                    find_checkpoint=find_checkpoint,
                    register_checkpoint_assets=register_checkpoint_assets,
                    finalize_training_job_sync=finalize_training_job_sync,
                    runner_log=_runner_log,
                )
            finally:
                unregister_running_proc()
                running_procs.pop(train_job_id, None)
    except TrainingNodeSSHError as exc:
        message = str(exc).strip() or "SSH 连接失败"
        from app.services.training_job_status import training_activity_detected

        has_activity = training_activity_detected(train_job_dir)
        if _remote_training_still_active(node, train_job_id, remote_job_dir):
            _runner_log(f"SSH disconnected while remote training still active: {message}")
            update_status(
                train_job_dir,
                {
                    "status": "running" if has_activity else "starting",
                    "message": (
                        f"远程训练进行中（{node.label}，本地 SSH 轮询中断，请稍后自动同步）"
                        if has_activity
                        else f"远端训练已启动（{node.label}），本地连接中断，正在后台同步状态"
                    ),
                    "trainingNodeId": node.node_id,
                    "executionMode": "remote_ssh",
                    "remoteHost": node.host,
                    "remoteJobDir": remote_job_dir,
                },
            )
            unregister_running_proc()
            running_procs.pop(train_job_id, None)
            return
        if _try_recover_remote_completion(
            node,
            train_job_id,
            remote_job_dir,
            remote_log,
            train_log_path,
            train_job_dir / "status.json",
            backend_out,
            checkpoints_dir,
            find_checkpoint,
            _runner_log,
        ):
            _runner_log("recovered completion after SSH failure")
            try:
                with TrainingNodeSSHClient(node) as client:
                    framework_label, model_type = backend_framework_meta(resolved_backend)
                    _finalize_remote_training_job(
                        client=client,
                        train_job_id=train_job_id,
                        train_job_dir=train_job_dir,
                        manifest=manifest,
                        train_config=train_config,
                        node=node,
                        resolved_backend=resolved_backend,
                        framework_label=framework_label,
                        model_type=model_type,
                        total_epochs=total_epochs,
                        remote_job_dir=remote_job_dir,
                        remote_log=remote_log,
                        remote_status=remote_status,
                        train_log_path=train_log_path,
                        checkpoints_dir=checkpoints_dir,
                        backend_out=backend_out,
                        return_code=0,
                        update_status=update_status,
                        find_checkpoint=find_checkpoint,
                        register_checkpoint_assets=register_checkpoint_assets,
                        finalize_training_job_sync=finalize_training_job_sync,
                        runner_log=_runner_log,
                    )
                    return
            except Exception:
                logger.exception("remote training recovery finalize failed trainJobId=%s", train_job_id)
        _fail(f"远程 SSH 训练失败: {message}")
    except Exception as exc:
        logger.exception("remote training job failed trainJobId=%s", train_job_id)
        _fail(f"远程训练异常: {exc}")


def reconcile_remote_training_job_runtime(
    train_job_id: str,
    *,
    node: Optional[TrainingNodeConfig] = None,
) -> dict[str, Any]:
    """对已存在的 remote_ssh 任务拉取远端日志/checkpoint 并收敛状态（不写失败除非确认失败）。"""
    from app.services.training_job_sync_service import finalize_training_job_sync, sync_training_job_from_runtime
    from app.services.training_node_service import resolve_training_node
    from app.services.training_service import _find_checkpoint
    from app.services.training_service import _read_json, _update_status
    from app.services.workspace_runtime_paths import resolve_training_job_root
    from app.services.checkpoint_registry import register_checkpoint_assets
    from app.services.training_metrics import parse_training_logs, sync_metrics_from_logs

    job_id = (train_job_id or "").strip()
    train_job_dir = resolve_training_job_root(job_id)
    if train_job_dir is None:
        return {"ok": False, "message": "job directory not found"}

    status_data = _read_json(train_job_dir / "status.json")
    train_config = _read_json(train_job_dir / "config" / "train_config.json")
    execution_mode = str(
        status_data.get("executionMode")
        or train_config.get("executionMode")
        or ""
    ).lower()
    if execution_mode != "remote_ssh":
        sync_training_job_from_runtime(job_id)
        return {"ok": True, "message": "local sync only"}

    node_id = str(
        status_data.get("trainingNodeId")
        or train_config.get("trainingNodeId")
        or ""
    ).strip()
    cfg = node or (resolve_training_node(training_node_id=node_id) if node_id else None)
    if cfg is None:
        sync_training_job_from_runtime(job_id)
        return {"ok": False, "message": "training node config missing"}

    remote_job_dir = str(status_data.get("remoteJobDir") or "").strip()
    if not remote_job_dir:
        remote_job_dir = remote_training_job_dir(cfg, job_id)

    remote_log = f"{remote_job_dir}/logs/train.log"
    remote_status = f"{remote_job_dir}/status.json"
    train_log_path = train_job_dir / "logs" / "train.log"
    runner_log_path = train_job_dir / "logs" / "remote_runner.log"
    checkpoints_dir = train_job_dir / "checkpoints"
    manifest = _read_json(train_job_dir / "artifacts" / "dataset_manifest.json")
    resolved_backend = str(
        train_config.get("trainingBackend")
        or status_data.get("trainingBackendResolved")
        or status_data.get("trainingBackend")
        or "robomimic_bc"
    )
    backend_out = checkpoints_dir / resolved_backend
    total_epochs = int(status_data.get("totalEpochs") or train_config.get("epochs") or 0)

    def _runner_log(message: str) -> None:
        runner_log_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        with runner_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{stamp}] reconcile: {message}\n")

    try:
        with TrainingNodeSSHClient(cfg) as client:
            _incremental_sync_remote_file(client, remote_log, train_log_path, 0, overwrite=True)
            _incremental_sync_remote_file(
                client, remote_status, train_job_dir / "status.json", 0, overwrite=True
            )
            _sftp_download_dir(client, f"{remote_job_dir}/checkpoints", checkpoints_dir, log=_runner_log)

            active = _remote_training_process_active(client, job_id, f"{remote_job_dir}/artifacts/remote_train.pid")
            return_code = 0 if not active else None
            if not active:
                return_code = _infer_remote_exit_code(client, remote_log)

            epoch, loss = parse_training_logs(train_job_dir, total_epochs)
            sync_metrics_from_logs(
                train_job_dir,
                {"totalEpochs": total_epochs, "epoch": epoch, **status_data},
            )

            checkpoint = _find_checkpoint(backend_out) or _find_checkpoint(checkpoints_dir)
            if checkpoint is not None and not active and (return_code == 0 or checkpoint.is_file()):
                if resolved_backend == "diffusion_policy":
                    framework_label, model_type = "Diffusion Policy", "diffusion_policy"
                elif resolved_backend == "torch_bc":
                    framework_label, model_type = "BC (PyTorch)", "bc"
                else:
                    framework_label, model_type = "Robomimic BC", "bc"

                completion_status = {
                    **status_data,
                    "status": "completed",
                    "epoch": max(epoch, total_epochs) if total_epochs else epoch,
                    "totalEpochs": total_epochs,
                    "progress": 1.0,
                    "loss": loss,
                }
                assets = register_checkpoint_assets(
                    train_job_dir=train_job_dir,
                    train_job_id=job_id,
                    manifest=manifest,
                    train_config=train_config,
                    status=completion_status,
                    resolved_backend=resolved_backend,
                    framework_label=framework_label,
                    model_type=model_type,
                    register_final=True,
                )
                primary = next(
                    (item for item in assets if item.get("checkpointKind") == "final"),
                    assets[-1] if assets else None,
                )
                patch: dict[str, Any] = {
                    "status": "completed" if (return_code == 0 or checkpoint.is_file()) else status_data.get("status"),
                    "epoch": max(epoch, total_epochs) if total_epochs else epoch,
                    "totalEpochs": total_epochs,
                    "progress": 1.0,
                    "loss": loss,
                    "processPid": None,
                    "executionMode": "remote_ssh",
                    "trainingNodeId": node_id,
                    "message": f"远程训练已同步（{cfg.label}）",
                }
                if primary is not None:
                    patch["checkpointExists"] = True
                    patch["checkpointPath"] = str(primary.get("checkpointPath") or "")
                    patch["modelAssetId"] = primary.get("modelAssetId")
                _update_status(train_job_dir, patch)
                finalize_training_job_sync(job_id)
                _runner_log("completed reconcile")
                return {"ok": True, "message": "reconciled as completed", "status": "completed"}

            if active:
                from app.services.training_job_status import training_activity_detected

                has_activity = epoch > 0 or loss is not None or training_activity_detected(train_job_dir, status_data)
                progress = min(0.99, epoch / total_epochs) if total_epochs > 0 and has_activity else 0.0
                public_status = "running" if has_activity else "starting"
                _update_status(
                    train_job_dir,
                    {
                        "status": public_status,
                        "epoch": epoch,
                        "totalEpochs": total_epochs,
                        "progress": progress,
                        "loss": loss,
                        "executionMode": "remote_ssh",
                        "trainingNodeId": node_id,
                        "message": (
                            f"远程训练进行中（{cfg.label}）"
                            if has_activity
                            else f"远端进程运行中（{cfg.label}），等待首批训练日志"
                        ),
                    },
                )
                sync_training_job_from_runtime(job_id)
                return {"ok": True, "message": "still running on remote", "status": public_status}

            from app.services.training_job_status import training_activity_detected

            if not training_activity_detected(train_job_dir, status_data):
                fail_message = "远端训练进程已退出且未产生训练日志，可能启动失败"
                _update_status(
                    train_job_dir,
                    {
                        "status": "failed",
                        "message": fail_message,
                        "progress": 0.0,
                        "processPid": None,
                        "executionMode": "remote_ssh",
                        "trainingNodeId": node_id,
                    },
                )
                sync_training_job_from_runtime(job_id)
                _runner_log(f"marked failed: {fail_message}")
                return {"ok": True, "message": fail_message, "status": "failed"}

            sync_training_job_from_runtime(job_id)
            return {"ok": True, "message": "synced remote artifacts", "status": status_data.get("status")}
    except TrainingNodeSSHError as exc:
        if _local_checkpoint_ready(backend_out, checkpoints_dir, _find_checkpoint):
            sync_training_job_from_runtime(job_id)
            return {"ok": True, "message": "local checkpoint present; synced offline", "status": "completed"}
        return {"ok": False, "message": str(exc)}


def _finalize_remote_training_job(
    *,
    client: TrainingNodeSSHClient,
    train_job_id: str,
    train_job_dir: Path,
    manifest: dict[str, Any],
    train_config: dict[str, Any],
    node: TrainingNodeConfig,
    resolved_backend: str,
    framework_label: str,
    model_type: str,
    total_epochs: int,
    remote_job_dir: str,
    remote_log: str,
    remote_status: str,
    train_log_path: Path,
    checkpoints_dir: Path,
    backend_out: Path,
    return_code: int,
    update_status: Callable[[Path, dict[str, Any]], dict[str, Any]],
    find_checkpoint: Callable[[Path], Optional[Path]],
    register_checkpoint_assets: Callable[..., Any],
    finalize_training_job_sync: Callable[[str], Any],
    runner_log: Callable[[str], None],
) -> None:
    from app.services.checkpoint_registry import parse_save_policy
    from app.services.training_metrics import parse_training_logs, sync_metrics_from_logs

    def _read_json(path: Path) -> dict[str, Any]:
        import json

        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    _incremental_sync_remote_file(client, remote_log, train_log_path, 0, overwrite=True)
    _incremental_sync_remote_file(client, remote_status, train_job_dir / "status.json", 0, overwrite=True)
    runner_log("pulling remote checkpoints")
    _sftp_download_dir(client, f"{remote_job_dir}/checkpoints", checkpoints_dir, log=runner_log)

    epoch, loss = parse_training_logs(train_job_dir, total_epochs)
    status_snapshot = _read_json(train_job_dir / "status.json")
    sync_metrics_from_logs(
        train_job_dir,
        status_snapshot or {"totalEpochs": total_epochs, "epoch": epoch},
    )

    checkpoint = find_checkpoint(backend_out) or find_checkpoint(checkpoints_dir)
    if checkpoint is not None and parse_save_policy(train_config).get("saveFinal", True):
        suffix = checkpoint.suffix or ".pt"
        final_path = checkpoints_dir / f"model_final{suffix}"
        try:
            if not final_path.is_file():
                final_path.write_bytes(checkpoint.read_bytes())
        except OSError:
            pass

    effective_return = return_code
    if effective_return != 0 and checkpoint is not None and _local_checkpoint_ready(backend_out, checkpoints_dir, find_checkpoint):
        runner_log("checkpoint present despite non-zero exit; treating as success")
        effective_return = 0

    status_data = _read_json(train_job_dir / "status.json")
    completion_status = {
        **status_data,
        "status": "completed" if effective_return == 0 else status_data.get("status"),
        "epoch": max(epoch, total_epochs) if effective_return == 0 else epoch,
        "totalEpochs": total_epochs,
    }
    assets = register_checkpoint_assets(
        train_job_dir=train_job_dir,
        train_job_id=train_job_id,
        manifest=manifest,
        train_config=train_config,
        status=completion_status,
        resolved_backend=resolved_backend,
        framework_label=framework_label,
        model_type=model_type,
        register_final=(effective_return == 0),
    )
    primary = next(
        (item for item in assets if item.get("checkpointKind") == "final"),
        assets[-1] if assets else None,
    )

    if effective_return == 0 and primary is not None:
        update_status(
            train_job_dir,
            {
                "status": "completed",
                "epoch": max(epoch, total_epochs),
                "totalEpochs": total_epochs,
                "progress": 1.0,
                "loss": loss,
                "checkpointExists": True,
                "checkpointPath": str(primary.get("checkpointPath") or ""),
                "modelAssetId": primary.get("modelAssetId"),
                "message": f"远程训练完成（{node.label}），已登记 {len(assets)} 个模型资产",
                "processPid": None,
            },
        )
        finalize_training_job_sync(train_job_id)
        runner_log("training completed successfully")
        return

    message = (
        "远程训练失败，未找到有效 checkpoint"
        if effective_return == 0
        else f"远程训练进程退出，return code={effective_return}"
    )
    update_status(
        train_job_dir,
        {
            "status": "failed",
            "epoch": epoch,
            "totalEpochs": total_epochs,
            "progress": min(1.0, epoch / total_epochs) if total_epochs else 0.0,
            "loss": loss,
            "checkpointExists": bool(checkpoint),
            "message": message,
            "processPid": None,
        },
    )
    finalize_training_job_sync(train_job_id)
    runner_log(message)


def _remote_training_still_active(
    node: TrainingNodeConfig,
    train_job_id: str,
    remote_job_dir: str,
) -> bool:
    try:
        with TrainingNodeSSHClient(node) as client:
            return _remote_training_process_active(
                client,
                train_job_id,
                f"{remote_job_dir.rstrip('/')}/artifacts/remote_train.pid",
            )
    except TrainingNodeSSHError:
        return False


def _remote_training_process_active(
    client: TrainingNodeSSHClient,
    train_job_id: str,
    remote_pid_file: str,
) -> bool:
    train_scripts = ("train_dp.py", "train_bc.py", "run_openpi_train")
    quoted_job = shlex.quote(train_job_id)
    rc, out, _err = client.run(
        f"pgrep -af 'train_dp.py|train_bc.py|run_openpi_train' 2>/dev/null | grep -F {quoted_job} || true",
        timeout=15,
    )
    for line in (out or "").splitlines():
        lowered = line.lower()
        if train_job_id not in line:
            continue
        if "grep -f" in lowered or "pgrep -af" in lowered:
            continue
        if any(script in line for script in train_scripts):
            return True
    rc, out, _err = client.run(f"cat {shlex.quote(remote_pid_file)} 2>/dev/null", timeout=10)
    remote_pid = (out or "").strip()
    if remote_pid.isdigit():
        rc2, out2, _ = client.run(
            f"kill -0 {shlex.quote(remote_pid)} 2>/dev/null; echo $?",
            timeout=10,
        )
        if (out2 or "").strip().endswith("0"):
            return True
    return False


def _infer_remote_exit_code(client: TrainingNodeSSHClient, remote_log: str) -> int:
    rc, out, _err = client.run(
        f"grep -q 'Traceback' {shlex.quote(remote_log)} 2>/dev/null; echo trace=$?",
        timeout=15,
    )
    if "trace=0" in (out or ""):
        return 1
    return 0


def _local_checkpoint_ready(
    backend_out: Path,
    checkpoints_dir: Path,
    find_checkpoint: Callable[[Path], Optional[Path]],
) -> bool:
    checkpoint = find_checkpoint(backend_out) or find_checkpoint(checkpoints_dir)
    if checkpoint is None:
        return False
    try:
        return checkpoint.is_file() and checkpoint.stat().st_size > 0
    except OSError:
        return False


def _try_recover_remote_completion(
    node: TrainingNodeConfig,
    train_job_id: str,
    remote_job_dir: str,
    remote_log: str,
    train_log_path: Path,
    local_status_path: Path,
    backend_out: Path,
    checkpoints_dir: Path,
    find_checkpoint: Callable[[Path], Optional[Path]],
    runner_log: Callable[[str], None],
) -> bool:
    if _local_checkpoint_ready(backend_out, checkpoints_dir, find_checkpoint):
        return True
    try:
        with TrainingNodeSSHClient(node) as client:
            if _remote_training_process_active(client, train_job_id, f"{remote_job_dir}/artifacts/remote_train.pid"):
                return False
            _incremental_sync_remote_file(client, remote_log, train_log_path, 0, overwrite=True)
            _sftp_download_dir(client, f"{remote_job_dir}/checkpoints", checkpoints_dir, log=runner_log)
            return _local_checkpoint_ready(backend_out, checkpoints_dir, find_checkpoint)
    except TrainingNodeSSHError:
        return _local_checkpoint_ready(backend_out, checkpoints_dir, find_checkpoint)


def _incremental_sync_remote_file(
    client: TrainingNodeSSHClient,
    remote_path: str,
    local_path: Path,
    offset: int,
    *,
    overwrite: bool = False,
) -> int:
    ssh = client._connect()
    sftp = ssh.open_sftp()
    try:
        stat = sftp.stat(remote_path)
        size = int(stat.st_size)
        if overwrite:
            with sftp.open(remote_path, "rb") as remote_file:
                data = remote_file.read()
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            return size
        if size <= offset:
            return offset
        with sftp.open(remote_path, "rb") as remote_file:
            remote_file.seek(offset)
            chunk = remote_file.read()
        if chunk:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            mode = "ab" if offset > 0 and local_path.is_file() else "wb"
            with local_path.open(mode) as handle:
                handle.write(chunk)
        return size
    except OSError:
        return offset
    finally:
        sftp.close()


def _remap_command_paths(
    cmd: list[str],
    *,
    project_root: Path,
    cable_working_dir: Path,
    dual_arm_working_dir: Path,
    remote_root: str,
    node: TrainingNodeConfig,
) -> list[str]:
    mapping: list[tuple[Path, str]] = [
        (project_root.resolve(), remote_root),
        (cable_working_dir.resolve(), f"{remote_root}/integrations/CableThreadingMVP"),
        (dual_arm_working_dir.resolve(), f"{remote_root}/integrations/DualArmCableManipulation"),
    ]
    remapped: list[str] = []
    for part in cmd:
        text = part
        for local_root, remote_path in mapping:
            local_str = str(local_root)
            if text.startswith(local_str):
                text = remote_path + text[len(local_str) :]
                break
        remapped.append(text)

    if node.python_bin and remapped:
        remapped[0] = node.python_bin
    elif node.conda_bin and node.conda_env and remapped:
        remapped = [
            node.conda_bin,
            "run",
            "-n",
            node.conda_env,
            "python",
            *remapped[1:],
        ]
    return remapped


def _sftp_sync_dir(
    client: TrainingNodeSSHClient,
    local_dir: Path,
    remote_dir: str,
    *,
    skip: set[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> None:
    if not local_dir.is_dir():
        return
    if log:
        log(f"sync dir {local_dir} -> {remote_dir}")
    ssh = client._connect()
    sftp = ssh.open_sftp()
    skip = skip or set()
    try:
        for path in local_dir.rglob("*"):
            if path.is_dir():
                continue
            if path.name in skip:
                continue
            rel = path.relative_to(local_dir)
            remote_path = f"{remote_dir}/{rel.as_posix()}"
            _sftp_mkdirs(sftp, str(Path(remote_path).parent))
            sftp.put(str(path), remote_path)
    finally:
        sftp.close()


def _sftp_upload_file(client: TrainingNodeSSHClient, local_path: Path, remote_path: str) -> None:
    ssh = client._connect()
    sftp = ssh.open_sftp()
    try:
        _sftp_mkdirs(sftp, str(Path(remote_path).parent))
        sftp.put(str(local_path), remote_path)
    finally:
        sftp.close()


def _sftp_write_json(client: TrainingNodeSSHClient, remote_path: str, payload: dict[str, Any]) -> None:
    import json
    import tempfile

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".json") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        temp_path = handle.name
    try:
        _sftp_upload_file(client, Path(temp_path), remote_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)


def _sftp_download_dir(
    client: TrainingNodeSSHClient,
    remote_dir: str,
    local_dir: Path,
    *,
    log: Callable[[str], None] | None = None,
) -> None:
    if log:
        log(f"download remote dir {remote_dir} -> {local_dir}")
    ssh = client._connect()
    sftp = ssh.open_sftp()
    try:

        def _walk(remote: str, local: Path) -> None:
            local.mkdir(parents=True, exist_ok=True)
            for entry in sftp.listdir_attr(remote):
                name = entry.filename
                remote_path = f"{remote}/{name}"
                local_path = local / name
                if _sftp_is_dir(sftp, remote_path):
                    _walk(remote_path, local_path)
                else:
                    sftp.get(remote_path, str(local_path))

        _walk(remote_dir, local_dir)
    except OSError:
        pass
    finally:
        sftp.close()


def _sftp_mkdirs(sftp, remote_dir: str) -> None:
    parts = remote_dir.strip("/").split("/")
    current = ""
    for part in parts:
        current = f"{current}/{part}" if current else f"/{part}"
        try:
            sftp.stat(current)
        except OSError:
            sftp.mkdir(current)


def _sftp_is_dir(sftp, path: str) -> bool:
    import stat

    try:
        return stat.S_ISDIR(sftp.stat(path).st_mode)
    except OSError:
        return False
