"""Isaac Lab 运行时轻量探测（不启动 Isaac Sim 大进程）。"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status

from app.core.config import settings
from app.services.isaac_lab.generate_cli import DEFAULT_MIMIC_TASK_ID, DEFAULT_SCRIPTED_EXPERT_TASK_ID
from app.services.isaac_lab.paths import (
    read_isaaclab_version,
    resolve_isaaclab_root,
    resolve_isaaclab_sh,
    resolve_scripted_expert_platform_script,
    resolve_stack_cube_default_seed,
)

RUNTIME_NOT_CONFIGURED_MSG = (
    "Isaac Lab runtime is not configured. "
    "Please set ISAACLAB_ROOT and configure an Isaac Sim runtime node."
)

STACK_CUBE_ISSUE_MISSING_ROOT = "missing_isaaclab_root"
STACK_CUBE_ISSUE_RUNTIME_DISABLED = "runtime_disabled"
STACK_CUBE_ISSUE_MISSING_DEFAULT_SEED = "missing_default_seed"
STACK_CUBE_ISSUE_GPU_UNAVAILABLE = "gpu_unavailable"
STACK_CUBE_ISSUE_TASK_NOT_REGISTERED = "task_not_registered"
SCRIPTED_EXPERT_ISSUE_SCRIPT_MISSING = "scripted_expert_script_missing"


def check_isaaclab_root() -> tuple[bool, Path | None, list[str]]:
    issues: list[str] = []
    root = resolve_isaaclab_root()
    if root is None:
        issues.append("ISAACLAB_ROOT is not configured")
        return False, None, issues
    if not root.is_dir():
        issues.append(f"ISAACLAB_ROOT path does not exist: {root}")
        return False, root, issues
    return True, root, issues


def check_isaaclab_sh(root: Path | None) -> tuple[bool, Path | None, list[str]]:
    issues: list[str] = []
    sh = resolve_isaaclab_sh(root)
    if sh is None:
        if root is None:
            issues.append("isaaclab.sh cannot be resolved without ISAACLAB_ROOT")
        else:
            issues.append(f"isaaclab.sh not found under {root}")
        return False, None, issues
    return True, sh, issues


def check_gpu_available() -> tuple[bool, list[str]]:
    issues: list[str] = []
    if shutil.which("nvidia-smi") is None:
        issues.append("nvidia-smi not found in PATH")
        return False, issues
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0 or not (proc.stdout or "").strip():
            issues.append("nvidia-smi returned no GPU information")
            return False, issues
        return True, issues
    except (OSError, subprocess.TimeoutExpired) as exc:
        issues.append(f"nvidia-smi check failed: {exc}")
        return False, issues


def check_task_registered(root: Path | None, task_id: str | None = None) -> tuple[bool, list[str]]:
    issues: list[str] = []
    task_name = (task_id or settings.ISAACLAB_DEFAULT_TASK or "").strip()
    if root is None:
        issues.append("Cannot verify task registration without ISAACLAB_ROOT")
        return False, issues
    tasks_dir = root / "source" / "isaaclab_tasks"
    mimic_dir = root / "source" / "isaaclab_mimic"
    scan_dirs = [tasks_dir]
    if mimic_dir.is_dir():
        scan_dirs.append(mimic_dir)
    if not tasks_dir.is_dir() and not mimic_dir.is_dir():
        issues.append(f"isaaclab_tasks source directory missing: {tasks_dir}")
        return False, issues
    if not task_name:
        issues.append("Default task id is empty")
        return False, issues
    try:
        for scan_root in scan_dirs:
            for py_file in scan_root.rglob("*.py"):
                try:
                    content = py_file.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if task_name in content:
                    return True, issues
    except OSError as exc:
        issues.append(f"Failed to scan task registry files: {exc}")
        return False, issues
    issues.append(f"Task id not found in isaaclab_tasks sources: {task_name}")
    return False, issues


def is_runtime_configured() -> bool:
    root_ok, root, _ = check_isaaclab_root()
    sh_ok, _, _ = check_isaaclab_sh(root)
    return root_ok and sh_ok


def assert_runtime_configured_for_commands() -> None:
    """Smoke test / CLI 最低门槛：ROOT + isaaclab.sh + RUNTIME_ENABLED。"""
    if not bool(settings.ISAACLAB_RUNTIME_ENABLED):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=RUNTIME_NOT_CONFIGURED_MSG,
        )
    root_ok, root, root_issues = check_isaaclab_root()
    sh_ok, _, sh_issues = check_isaaclab_sh(root)
    if not root_ok or not sh_ok:
        detail = "; ".join([*root_issues, *sh_issues]) or RUNTIME_NOT_CONFIGURED_MSG
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=detail,
        )


def check_runtime() -> dict[str, Any]:
    """供 Adapter 调用的运行时检查结果（严格 available）。"""
    status_payload = get_runtime_status()
    return {
        "ok": bool(status_payload.get("available")),
        "message": None if status_payload.get("available") else RUNTIME_NOT_CONFIGURED_MSG,
        "status": status_payload,
    }


def _collect_stack_cube_issue_codes(
    *,
    enabled: bool,
    root_ok: bool,
    sh_ok: bool,
    mimic_task_ok: bool,
    default_seed_available: bool,
    gpu_ok: bool,
) -> list[str]:
    codes: list[str] = []
    if not enabled:
        codes.append(STACK_CUBE_ISSUE_RUNTIME_DISABLED)
    if not root_ok or not sh_ok:
        codes.append(STACK_CUBE_ISSUE_MISSING_ROOT)
    if root_ok and not mimic_task_ok:
        codes.append(STACK_CUBE_ISSUE_TASK_NOT_REGISTERED)
    if not default_seed_available:
        codes.append(STACK_CUBE_ISSUE_MISSING_DEFAULT_SEED)
    if enabled and root_ok and not gpu_ok:
        codes.append(STACK_CUBE_ISSUE_GPU_UNAVAILABLE)
    return codes


def _collect_scripted_expert_issue_codes(
    *,
    enabled: bool,
    root_ok: bool,
    sh_ok: bool,
    stack_task_ok: bool,
    gpu_ok: bool,
    script_available: bool,
) -> list[str]:
    codes: list[str] = []
    if not enabled:
        codes.append(STACK_CUBE_ISSUE_RUNTIME_DISABLED)
    if not root_ok or not sh_ok:
        codes.append(STACK_CUBE_ISSUE_MISSING_ROOT)
    if root_ok and not stack_task_ok:
        codes.append(STACK_CUBE_ISSUE_TASK_NOT_REGISTERED)
    if enabled and root_ok and not gpu_ok:
        codes.append(STACK_CUBE_ISSUE_GPU_UNAVAILABLE)
    if not script_available:
        codes.append(SCRIPTED_EXPERT_ISSUE_SCRIPT_MISSING)
    return codes


def get_runtime_status() -> dict[str, Any]:
    enabled = bool(settings.ISAACLAB_RUNTIME_ENABLED)
    root_ok, root, root_issues = check_isaaclab_root()
    sh_ok, sh_path, sh_issues = check_isaaclab_sh(root)
    gpu_ok, gpu_issues = check_gpu_available()
    task_ok, task_issues = check_task_registered(root, settings.ISAACLAB_DEFAULT_TASK)
    mimic_task_ok, mimic_task_issues = check_task_registered(root, DEFAULT_MIMIC_TASK_ID)
    stack_task_ok, stack_task_issues = check_task_registered(root, DEFAULT_SCRIPTED_EXPERT_TASK_ID)
    default_seed_path, default_seed_available = resolve_stack_cube_default_seed()
    _, scripted_expert_script_available = resolve_scripted_expert_platform_script()

    issues: list[str] = []
    if not enabled:
        issues.append("ISAACLAB_RUNTIME_ENABLED is false")
    issues.extend(root_issues)
    issues.extend(sh_issues)
    if root_ok:
        if not task_ok:
            issues.extend(task_issues)
    if enabled and root_ok:
        issues.extend(gpu_issues)

    version = read_isaaclab_version(root)
    tasks_present = root is not None and (root / "source" / "isaaclab_tasks").is_dir()
    configured = root_ok and sh_ok and tasks_present

    available = configured and enabled and task_ok and gpu_ok

    stack_cube_issue_codes = _collect_stack_cube_issue_codes(
        enabled=enabled,
        root_ok=root_ok,
        sh_ok=sh_ok,
        mimic_task_ok=mimic_task_ok,
        default_seed_available=default_seed_available,
        gpu_ok=gpu_ok,
    )
    stack_cube_generation_ready = len(stack_cube_issue_codes) == 0

    scripted_expert_issue_codes = _collect_scripted_expert_issue_codes(
        enabled=enabled,
        root_ok=root_ok,
        sh_ok=sh_ok,
        stack_task_ok=stack_task_ok,
        gpu_ok=gpu_ok,
        script_available=scripted_expert_script_available,
    )
    scripted_expert_available = scripted_expert_script_available
    scripted_expert_ready = len(scripted_expert_issue_codes) == 0

    return {
        "enabled": enabled,
        "configured": configured,
        "available": available,
        "runtimeMode": settings.ISAACLAB_RUNTIME_MODE,
        "isaacLabRoot": str(root) if root else None,
        "isaacLabSh": str(sh_path) if sh_path else None,
        "isaacLabPython": settings.ISAACLAB_PYTHON,
        "isaacLabVersion": version if root_ok else None,
        "defaultTask": settings.ISAACLAB_DEFAULT_TASK,
        "gpuAvailable": gpu_ok if root_ok else False,
        "taskRegistered": task_ok if root_ok else False,
        "mimicTaskRegistered": mimic_task_ok if root_ok else False,
        "outputRoot": settings.ISAACLAB_OUTPUT_ROOT,
        "defaultSeedFile": str(default_seed_path) if str(default_seed_path) else None,
        "defaultSeedAvailable": default_seed_available,
        "stackCubeGenerationReady": stack_cube_generation_ready,
        "scriptedExpertAvailable": scripted_expert_available,
        "scriptedExpertReady": scripted_expert_ready,
        "scriptedExpertIssueCodes": scripted_expert_issue_codes,
        "issues": issues,
        "stackCubeIssueCodes": stack_cube_issue_codes,
    }
