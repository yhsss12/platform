from __future__ import annotations

import json
import subprocess
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

from scripts.prepare_source_dataset import prepare_source_dataset
from utils.job_status import (
    apply_important_stats,
    heartbeat_job_status,
    parse_important_stats,
    set_job_stage,
)
from utils.runtime_env import (
    build_mimicgen_subprocess_env,
    default_source_demo_path,
    resolve_mimicgen_root,
    resolve_mimicgen_python,
    resolve_source_demo_path,
)

_INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
_SQUARE_TEMPLATE = _INTEGRATION_ROOT / "configs" / "square_d0.json"
_NUT_ASSEMBLY_TEMPLATE = _INTEGRATION_ROOT / "configs" / "nut_assembly_d0.json"

_HEARTBEAT_INTERVAL_SEC = 5.0


def _emit_status(
    on_status: Callable[[dict[str, Any]], None] | None,
    status_base: dict[str, Any] | None,
    job_root: Path | None,
    **updates: Any,
) -> None:
    if on_status is not None:
        on_status(updates)
        return
    if status_base is not None and job_root is not None:
        stage = str(updates.pop("stage", "mimicgen_generate"))
        set_job_stage(job_root, status_base, stage=stage, **updates)


def _resolve_mimicgen_template(source_env_name: str) -> tuple[Path, str, str]:
    """Return (config_path, env_interface, config_name) for MimicGen datagen."""
    if source_env_name.startswith("NutAssembly"):
        return _NUT_ASSEMBLY_TEMPLATE, "MG_NutAssembly", "nut_assembly"
    return _SQUARE_TEMPLATE, "MG_Square", "square"


def try_mimicgen_datagen(
    *,
    job_root: Path,
    episodes: int,
    seed: int,
    source_env_name: str,
    source_demo_path: Path | None,
    render_video: bool,
    on_status: Callable[[dict[str, Any]], None] | None = None,
    status_base: dict[str, Any] | None = None,
) -> dict[str, Any]:
    mimicgen_root = resolve_mimicgen_root()
    if mimicgen_root is None:
        return {"ok": False, "reason": "mimicgen_import_failed", "error": "mimicgen package not found"}

    source_path, source_err = resolve_source_demo_path(str(source_demo_path) if source_demo_path else None)
    if source_err or source_path is None:
        return {
            "ok": False,
            "reason": "source_demo_missing",
            "error": source_err or "source demo not found",
            "sourceDemoPath": str(source_demo_path) if source_demo_path else None,
        }

    python_bin = resolve_mimicgen_python()
    if not python_bin.is_file():
        return {
            "ok": False,
            "reason": "mimicgen_env_not_ready",
            "error": f"nut-assembly-mvp python not found: {python_bin}",
            "sourceDemoPath": str(source_path),
        }

    env_name = source_env_name if source_env_name.endswith("_D0") else "Square_D0"
    template_path, env_interface, _config_name = _resolve_mimicgen_template(env_name)

    if not template_path.is_file():
        return {
            "ok": False,
            "reason": "mimicgen_import_failed",
            "error": f"config template missing: {template_path}",
        }

    _emit_status(
        on_status,
        status_base,
        job_root,
        stage="prepare_source",
        message="正在准备 source demo (datagen_info)...",
        sourceDemoPath=str(source_path),
        policyMode="mimicgen",
        generationMode="mimicgen_datagen",
        progress=5,
    )

    prepare_result = prepare_source_dataset(
        job_root=job_root,
        source_demo_path=source_path,
        env_interface=env_interface,
        env_interface_type="robosuite",
        python_bin=python_bin,
    )
    if not prepare_result.get("ok"):
        return {
            "ok": False,
            "reason": prepare_result.get("reason") or "prepare_source_failed",
            "error": prepare_result.get("error"),
            "traceback": prepare_result.get("traceback"),
            "prepareAttempt": prepare_result,
            "sourceDemoPath": str(source_path),
        }

    _emit_status(
        on_status,
        status_base,
        job_root,
        stage="prepare_source",
        message="source demo 准备完成",
        sourceDemoPath=str(source_path),
        hasDatagenInfo=prepare_result.get("hasDatagenInfo"),
        hasObjectPoses=prepare_result.get("hasObjectPoses"),
        progress=15,
    )

    prepared_path = Path(prepare_result["preparedPath"])

    config = json.loads(template_path.read_text(encoding="utf-8"))
    config = deepcopy(config)
    exp = config.setdefault("experiment", {})
    exp["seed"] = seed
    exp["render_video"] = render_video
    exp.setdefault("source", {})["dataset_path"] = str(prepared_path)
    exp.setdefault("source", {})["n"] = min(episodes, 10)
    gen = exp.setdefault("generation", {})
    gen["num_trials"] = episodes
    gen["path"] = str(job_root / "datasets" / "mimicgen_output")
    task = exp.setdefault("task", {})
    task["name"] = env_name
    task["robot"] = "Panda"
    task["gripper"] = "PandaGripper"
    task["interface"] = env_interface
    task["interface_type"] = "robosuite"

    runtime_config = job_root / "configs" / "mimicgen_nut_assembly_config.json"
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    script = mimicgen_root / "mimicgen" / "scripts" / "generate_dataset.py"
    log_path = job_root / "logs" / "mimicgen_attempt.log"
    cmd = [
        str(python_bin),
        str(script),
        "--config",
        str(runtime_config),
        "--auto-remove-exp",
    ]
    video_path = job_root / "videos" / "generate.mp4"
    if render_video:
        (job_root / "videos").mkdir(parents=True, exist_ok=True)
        cmd.extend(["--video_path", str(video_path)])

    env = build_mimicgen_subprocess_env(mimicgen_root=mimicgen_root)

    _emit_status(
        on_status,
        status_base,
        job_root,
        stage="mimicgen_generate",
        message="MimicGen 正在生成 demonstrations...",
        progress=20,
    )

    combined = ""
    return_code = 1
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(mimicgen_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        last_heartbeat = time.monotonic()
        last_log_size = 0
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            while True:
                line = proc.stdout.readline()
                if line:
                    log_file.write(line)
                    log_file.flush()
                    combined += line
                elif proc.poll() is not None:
                    break
                now = time.monotonic()
                if now - last_heartbeat >= _HEARTBEAT_INTERVAL_SEC:
                    last_heartbeat = now
                    updates: dict[str, Any] = {
                        "stage": "mimicgen_generate",
                        "message": "MimicGen 正在生成 demonstrations...",
                    }
                    current_size = log_path.stat().st_size if log_path.is_file() else 0
                    if current_size > last_log_size:
                        last_log_size = current_size
                    stats = parse_important_stats(job_root)
                    if stats and status_base is not None:
                        apply_important_stats(status_base, stats)
                        updates["episodesGenerated"] = status_base.get("episodesGenerated")
                        updates["datagenFailedTrials"] = status_base.get("datagenFailedTrials")
                        updates["progress"] = status_base.get("progress")
                    if on_status is not None:
                        on_status(updates)
                    elif status_base is not None:
                        heartbeat_job_status(job_root, status_base, **updates)
                    if not line:
                        time.sleep(0.2)
        return_code = proc.wait(timeout=30)
    except subprocess.TimeoutExpired as exc:
        combined = (exc.stdout or "") + "\n" + (exc.stderr or "")
        log_path.write_text(combined, encoding="utf-8")
        return {
            "ok": False,
            "reason": "mimicgen_datagen_failed",
            "error": "mimicgen datagen timeout",
            "traceback": combined,
            "prepareAttempt": prepare_result,
            "configPath": str(runtime_config),
            "sourceDemoPath": str(source_path),
            "preparedSourcePath": str(prepared_path),
        }
    except Exception:
        tb = traceback.format_exc()
        log_path.write_text(tb, encoding="utf-8")
        return {
            "ok": False,
            "reason": "mimicgen_datagen_failed",
            "error": tb.splitlines()[-1] if tb else "mimicgen datagen failed",
            "traceback": tb,
            "prepareAttempt": prepare_result,
            "configPath": str(runtime_config),
            "sourceDemoPath": str(source_path),
            "preparedSourcePath": str(prepared_path),
        }

    if not combined and log_path.is_file():
        combined = log_path.read_text(encoding="utf-8", errors="replace")

    if return_code != 0:
        err_hint = combined.strip().splitlines()[-1] if combined.strip() else f"exit {return_code}"
        reason = "mimicgen_datagen_failed"
        if "No module named" in combined:
            reason = "mimicgen_import_failed"
        elif "single_arm_env" in combined or "not registered" in combined.lower():
            reason = "env_registration_failed"
        return {
            "ok": False,
            "reason": reason,
            "error": err_hint,
            "traceback": combined,
            "logPath": str(log_path),
            "prepareAttempt": prepare_result,
            "configPath": str(runtime_config),
            "sourceDemoPath": str(source_path),
            "preparedSourcePath": str(prepared_path),
        }

    output_dir = Path(gen["path"])
    hdf5_candidates = [p for p in output_dir.rglob("*.hdf5") if "failed" not in p.name.lower()]
    if not hdf5_candidates:
        hdf5_candidates = list(output_dir.rglob("demo.hdf5"))
    if not hdf5_candidates:
        hdf5_candidates = list(output_dir.rglob("*.hdf5"))
    if not hdf5_candidates:
        return {
            "ok": False,
            "reason": "hdf5_write_failed",
            "error": "mimicgen finished without hdf5 output",
            "traceback": combined,
            "logPath": str(log_path),
            "prepareAttempt": prepare_result,
            "configPath": str(runtime_config),
        }

    stats_path = output_dir / "demo" / "important_stats.json"
    if not stats_path.is_file():
        stats_path = next(output_dir.rglob("important_stats.json"), None)
    mimicgen_stats: dict[str, Any] = {}
    if stats_path and stats_path.is_file():
        try:
            mimicgen_stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            mimicgen_stats = {}

    result: dict[str, Any] = {
        "ok": True,
        "hdf5Path": str(hdf5_candidates[0]),
        "logPath": str(log_path),
        "outputDir": str(output_dir),
        "configPath": str(runtime_config),
        "sourceDemoPath": str(source_path),
        "preparedSourcePath": str(prepared_path),
        "sourceEnvName": env_name,
        "runtimeEnvName": env_name,
        "prepareAttempt": prepare_result,
        "mimicgenStats": mimicgen_stats,
        "numSuccess": mimicgen_stats.get("num_success"),
        "numAttempts": mimicgen_stats.get("num_attempts"),
        "numFailures": mimicgen_stats.get("num_failures"),
    }
    if render_video and video_path.is_file():
        result["videoPath"] = str(video_path)
    return result
