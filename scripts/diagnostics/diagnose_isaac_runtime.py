#!/usr/bin/env python3
"""Diagnose Isaac Lab / Isaac Sim runtime and FrankaPickPlace availability."""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROBE_SCRIPT = PROJECT_ROOT / "scripts" / "isaac_franka_controller_probe.py"
RUNTIME_PROBE_SCRIPT = PROJECT_ROOT / "scripts" / "isaac_runtime_import_probe.py"


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass
class Runner:
    kind: str
    command: list[str]
    cwd: str | None = None
    env: dict[str, str] | None = None
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label or self.kind,
            "command": self.command,
            "cwd": self.cwd,
        }


def _conda_env_from_python(python_path: Path) -> tuple[str | None, str | None]:
    if python_path.name != "python" or not python_path.is_file():
        return None, None
    conda_prefix = python_path.parent.parent
    if (conda_prefix / "conda-meta").is_dir():
        return str(conda_prefix), conda_prefix.name
    return None, None


def _runner_env_for_python(python_path: Path, base: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(base or os.environ)
    env.setdefault("TERM", "xterm")
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    conda_prefix, conda_env = _conda_env_from_python(python_path)
    if conda_prefix:
        env["CONDA_PREFIX"] = conda_prefix
        if conda_env:
            env["CONDA_DEFAULT_ENV"] = conda_env
    else:
        env.pop("CONDA_PREFIX", None)
        env.pop("CONDA_DEFAULT_ENV", None)
    return env


def resolve_isaaclab_root() -> Path | None:
    raw = os.environ.get("ISAACLAB_ROOT", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path if path.is_dir() else None


def discover_runners() -> list[Runner]:
    runners: list[Runner] = []
    seen: set[tuple[str, ...]] = set()

    def add_runner(runner: Runner) -> None:
        key = tuple(runner.command)
        if key in seen:
            return
        seen.add(key)
        runners.append(runner)

    isaaclab_root = resolve_isaaclab_root()
    isaaclab_sh = None
    if isaaclab_root is not None:
        candidate = isaaclab_root / "isaaclab.sh"
        if candidate.is_file():
            isaaclab_sh = candidate

    for search in (
        PROJECT_ROOT / "IsaacLab" / "isaaclab.sh",
        PROJECT_ROOT.parent / "IsaacLab" / "isaaclab.sh",
    ):
        if isaaclab_sh is None and search.is_file():
            isaaclab_sh = search
            isaaclab_root = search.parent

    if isaaclab_sh is None:
        for match in glob.glob("/home/*/IsaacLab/isaaclab.sh"):
            path = Path(match)
            if path.is_file():
                isaaclab_sh = path
                isaaclab_root = path.parent
                break

    # Priority A: ISAACLAB_PYTHON direct python (preferred when explicitly configured)
    for env_name in ("ISAACLAB_PYTHON", "ISAACSIM_PYTHON"):
        raw = os.environ.get(env_name, "").strip()
        if not raw:
            continue
        py = Path(raw).expanduser()
        if py.is_file():
            add_runner(
                Runner(
                    kind="python",
                    label=f"{env_name} ({py})",
                    command=[str(py)],
                    env=_runner_env_for_python(py),
                )
            )

    # Priority B: ISAACLAB_ROOT + isaaclab.sh
    if isaaclab_sh is not None and isaaclab_root is not None:
        env = _runner_env_for_python(Path(os.environ.get("ISAACLAB_PYTHON", "")))
        isaaclab_python = os.environ.get("ISAACLAB_PYTHON", "").strip()
        if isaaclab_python:
            py = Path(isaaclab_python).expanduser()
            env = _runner_env_for_python(py, env)
        add_runner(
            Runner(
                kind="isaaclab_sh",
                label=f"isaaclab.sh ({isaaclab_sh})",
                command=[str(isaaclab_sh), "-p"],
                cwd=str(isaaclab_root),
                env=env,
            )
        )

    # Priority C: ISAACSIM_ROOT/python.sh
    raw_root = os.environ.get("ISAACSIM_ROOT", "").strip()
    if raw_root:
        pysh = Path(raw_root).expanduser() / "python.sh"
        if pysh.is_file():
            add_runner(
                Runner(
                    kind="isaacsim_python_sh",
                    label=f"ISAACSIM_ROOT/python.sh ({pysh})",
                    command=[str(pysh)],
                    env={"TERM": "xterm", **os.environ},
                )
            )

    # Priority D: common search paths
    for pattern in (
        "/home/*/.local/share/ov/pkg/isaac-sim-*/python.sh",
        "/home/*/isaacsim/python.sh",
        "/opt/nvidia/isaac-sim/python.sh",
    ):
        for match in glob.glob(pattern):
            pysh = Path(match)
            if pysh.is_file():
                add_runner(
                    Runner(
                        kind="isaacsim_python_sh",
                        label=f"discovered python.sh ({pysh})",
                        command=[str(pysh)],
                        env={"TERM": "xterm", **os.environ},
                    )
                )

    return runners


def run_with_runner(runner: Runner, script_path: Path, *args: str, timeout: int = 180) -> tuple[int, str, str]:
    cmd = [*runner.command, str(script_path), *args]
    env = dict(os.environ)
    if runner.env:
        env.update(runner.env)
    try:
        proc = subprocess.run(
            cmd,
            cwd=runner.cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return 124, stdout, stderr or "timeout"


def probe_runtime_imports(runner: Runner) -> dict[str, Any]:
    if not RUNTIME_PROBE_SCRIPT.is_file():
        return {"error": f"missing probe script: {RUNTIME_PROBE_SCRIPT}"}
    code, stdout, stderr = run_with_runner(runner, RUNTIME_PROBE_SCRIPT, timeout=120)
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                payload["runner"] = runner.to_dict()
                payload["stderr_tail"] = stderr.strip()[-500:] if stderr else ""
                payload["exit_code"] = code
                return payload
            except json.JSONDecodeError:
                continue
    return {
        "runner": runner.to_dict(),
        "exit_code": code,
        "stdout": stdout[-1000:],
        "stderr": stderr[-1000:],
        "error": "runtime probe produced no JSON",
    }


def probe_franka_controller(runner: Runner) -> dict[str, Any]:
    if not PROBE_SCRIPT.is_file():
        return {"error": f"missing probe script: {PROBE_SCRIPT}"}
    output_file = PROJECT_ROOT / "runs" / ".isaac_franka_controller_probe.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.is_file():
        output_file.unlink()
    code, stdout, stderr = run_with_runner(runner, PROBE_SCRIPT, str(output_file), timeout=300)
    if output_file.is_file():
        try:
            payload = json.loads(output_file.read_text(encoding="utf-8"))
            payload["runner"] = runner.to_dict()
            payload["stderr_tail"] = stderr.strip()[-500:] if stderr else ""
            payload["exit_code"] = code
            return payload
        except json.JSONDecodeError:
            pass
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                payload["runner"] = runner.to_dict()
                payload["stderr_tail"] = stderr.strip()[-500:] if stderr else ""
                payload["exit_code"] = code
                return payload
            except json.JSONDecodeError:
                continue
    return {
        "runner": runner.to_dict(),
        "exit_code": code,
        "stdout": stdout[-1000:],
        "stderr": stderr[-1000:],
        "error": "controller probe produced no JSON",
    }


def search_franka_pick_place_files() -> list[str]:
    patterns = [
        "**/isaacsim/robot/manipulators/examples/franka/**/pick_place.py",
        "**/isaacsim/robot/experimental/manipulators/examples/franka/**",
        "**/omni/isaac/examples/**/franka/**",
    ]
    hits: list[str] = []
    search_roots = []
    for env_name in ("ISAACLAB_PYTHON", "ISAACSIM_PYTHON"):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            py = Path(raw).expanduser()
            site = py.parent.parent / "lib"
            if site.is_dir():
                search_roots.append(site)
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix:
        search_roots.append(Path(conda_prefix) / "lib")
    for root in search_roots:
        for pattern in patterns:
            for match in root.glob(pattern):
                text = str(match)
                if "FrankaPickPlace" in match.read_text(encoding="utf-8", errors="replace") or "pick_place" in text:
                    hits.append(text)
    return sorted(set(hits))[:20]


def diagnose(*, probe_controller: bool = True) -> dict[str, Any]:
    _load_dotenv(PROJECT_ROOT / ".env")
    runners = discover_runners()
    isaaclab_root = resolve_isaaclab_root()

    report: dict[str, Any] = {
        "python_executable": sys.executable,
        "ISAACLAB_ROOT": os.environ.get("ISAACLAB_ROOT"),
        "ISAACSIM_ROOT": os.environ.get("ISAACSIM_ROOT"),
        "ISAACSIM_PYTHON": os.environ.get("ISAACSIM_PYTHON"),
        "ISAACLAB_PYTHON": os.environ.get("ISAACLAB_PYTHON"),
        "isaaclab_root_resolved": str(isaaclab_root) if isaaclab_root else None,
        "runners": [runner.to_dict() for runner in runners],
        "runtime_probe": None,
        "controller_probe": None,
        "franka_pick_place_files": search_franka_pick_place_files(),
        "isaac_lab_runtime_available": False,
        "isaac_sim_runtime_available": False,
        "simulation_app_available": False,
        "can_import_franka_pick_place": False,
        "detected_isaac_version": None,
        "recommended_runner": None,
        "diagnosis": "no_runner_found",
        "skip_reason": "No Isaac Lab / Isaac Sim runner could be resolved.",
    }

    if not runners:
        return report

    runtime_result = None
    controller_result = None
    recommended: Runner | None = None

    for runner in runners:
        runtime_result = probe_runtime_imports(runner)
        can_isaaclab = bool(runtime_result.get("can_import_isaaclab"))
        can_isaacsim = bool(runtime_result.get("can_import_isaacsim"))
        can_sim_app = bool(runtime_result.get("can_import_simulation_app"))
        if can_isaaclab or can_isaacsim or can_sim_app:
            recommended = runner
            report["runtime_probe"] = runtime_result
            report["isaac_lab_runtime_available"] = can_isaaclab
            report["isaac_sim_runtime_available"] = can_isaacsim
            report["simulation_app_available"] = can_sim_app
            report["detected_isaac_version"] = runtime_result.get("detected_isaac_version")
            break

    if recommended is None:
        report["runtime_probe"] = runtime_result
        report["diagnosis"] = "runtime_not_detected"
        report["skip_reason"] = "Isaac Lab / Isaac Sim runtime not detected in any resolved runner."
        return report

    report["recommended_runner"] = recommended.to_dict()

    if probe_controller:
        controller_result = probe_franka_controller(recommended)
        report["controller_probe"] = controller_result
        report["can_import_franka_pick_place"] = bool(controller_result.get("controller_available"))
        if controller_result.get("controller_import_path"):
            report["franka_pick_place_import_path"] = controller_result.get("controller_import_path")

    if report["isaac_lab_runtime_available"] and not report["can_import_franka_pick_place"]:
        report["diagnosis"] = "controller_unavailable"
        report["skip_reason"] = (
            "Isaac Lab runtime detected, but NVIDIA Isaac Sim official FrankaPickPlace controller "
            "is not importable in this environment."
        )
    elif report["can_import_franka_pick_place"]:
        report["diagnosis"] = "ready"
        report["skip_reason"] = None
    elif report["isaac_sim_runtime_available"] and not report["can_import_franka_pick_place"]:
        report["diagnosis"] = "controller_unavailable"
        report["skip_reason"] = (
            "Isaac Sim runtime detected, but NVIDIA Isaac Sim official FrankaPickPlace controller "
            "is not importable in this environment."
        )
    else:
        report["diagnosis"] = "runtime_not_detected"
        report["skip_reason"] = "Isaac Lab / Isaac Sim runtime not detected in any resolved runner."

    return report


def run_script_with_runner(runner: Runner, script_path: Path, *args: str, timeout: int = 3600) -> tuple[int, str, str]:
    return run_with_runner(runner, script_path, *args, timeout=timeout)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose Isaac Lab / Isaac Sim / FrankaPickPlace runtime")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-controller-probe", action="store_true")
    parser.add_argument("--exec", dest="exec_script", default=None, help="Run a Python script with the recommended runner")
    args, exec_args = parser.parse_known_args()

    report = diagnose(probe_controller=not args.no_controller_probe)

    if args.exec_script:
        if report.get("diagnosis") != "ready" or not report.get("recommended_runner"):
            print(json.dumps({"error": report.get("skip_reason"), "diagnosis": report.get("diagnosis")}, ensure_ascii=False))
            return 3 if report.get("diagnosis") == "controller_unavailable" else 2
        runner = Runner(
            kind=report["recommended_runner"]["kind"],
            command=report["recommended_runner"]["command"],
            cwd=report["recommended_runner"].get("cwd"),
            label=report["recommended_runner"].get("label", ""),
        )
        exec_args = list(exec_args)
        if exec_args and exec_args[0] == "--":
            exec_args = exec_args[1:]
        code, stdout, stderr = run_script_with_runner(runner, Path(args.exec_script), *exec_args)
        if stdout:
            print(stdout, end="" if stdout.endswith("\n") else "\n")
        if stderr:
            print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n")
        return code

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    if report.get("diagnosis") == "ready":
        return 0
    if report.get("diagnosis") == "controller_unavailable":
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
