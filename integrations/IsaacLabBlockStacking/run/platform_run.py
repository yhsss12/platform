#!/usr/bin/env python3
"""Platform unified entry for Isaac Lab Franka Stack Cube task package."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

TASK_ID = "task_isaaclab_franka_stack_cube_v1"
TASK_NAME = "Isaac Lab Franka Stack Cube"
BACKEND = "Isaac Lab / Isaac Sim"
MIMIC_TASK = "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0"
RECORD_TASK = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
ISAAC_LAB_REQUIRED_MSG = (
    "Isaac Lab environment is required to run this task.\n"
    "Please run this entry inside an Isaac Lab / Isaac Sim configured environment."
)


def task_package_root() -> Path:
    return Path(__file__).resolve().parent.parent


def project_root() -> Path:
    return task_package_root().parent.parent


def run_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_isaaclab_root() -> Path | None:
    raw = (os.environ.get("ISAACLAB_ROOT") or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (project_root() / path).resolve()
    return path if path.is_dir() else None


def resolve_isaaclab_sh(root: Path | None = None) -> Path | None:
    root = root if root is not None else resolve_isaaclab_root()
    if root is None:
        return None
    candidate = root / "isaaclab.sh"
    return candidate if candidate.is_file() else None


def isaac_lab_available() -> bool:
    return resolve_isaaclab_sh() is not None


def require_isaac_lab() -> tuple[Path, Path]:
    root = resolve_isaaclab_root()
    sh = resolve_isaaclab_sh(root)
    if root is None or sh is None:
        print(ISAAC_LAB_REQUIRED_MSG, file=sys.stderr)
        raise SystemExit(1)
    return root, sh


def _datasets_dir(output_dir: Path) -> Path:
    path = output_dir / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_device(device: str | None) -> str:
    raw = (device or os.environ.get("ISAACLAB_DEVICE") or "cuda:0").strip()
    return raw or "cuda:0"


def _utc_now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


PHASE_LABELS: dict[str, str] = {
    "starting_isaac_lab": "正在启动 Isaac Lab 运行时",
    "annotating_mimic": "正在标注 Mimic 种子演示",
    "generating_mimic": "正在生成 Mimic 数据集",
    "failed": "失败",
}


def _default_progress_message(phase: str) -> str:
    if phase == "starting_isaac_lab":
        return "正在启动 Isaac Lab 运行时，首次加载可能需要 1–3 分钟"
    if phase == "annotating_mimic":
        return "正在标注 Mimic 种子演示，首次启动 Isaac Lab 可能需要 1–3 分钟"
    if phase == "generating_mimic":
        return "正在生成 Mimic 数据集，请稍候…"
    return PHASE_LABELS.get(phase, phase)


def _write_job_phase(
    output_dir: Path,
    phase: str,
    *,
    progress: int | None = None,
    progress_message: str | None = None,
) -> None:
    output_dir = Path(output_dir).resolve()
    message = progress_message or _default_progress_message(phase)
    label = PHASE_LABELS.get(phase, phase)
    now = _utc_now_iso()
    for rel in ("status.json", "live/status.json"):
        path = output_dir / rel
        current: dict = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    current = loaded
            except (OSError, json.JSONDecodeError):
                current = {}
        phase_timings = dict(current.get("phaseTimings") or {})
        if current.get("phase") != phase:
            phase_timings[phase] = {"startedAt": now}
        payload = {
            **current,
            "phase": phase,
            "phaseLabel": label,
            "phaseStartedAt": phase_timings.get(phase, {}).get("startedAt", now),
            "phaseUpdatedAt": now,
            "phaseTimings": phase_timings,
            "progressMessage": message,
            "message": message,
            "status": "running",
        }
        if progress is not None:
            payload["progress"] = progress
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _ensure_local_seed_hdf5(output_dir: Path, seed_path: Path) -> Path:
    """Copy external/default seed HDF5 into job datasets/ when needed for annotate."""
    datasets = _datasets_dir(output_dir)
    local_seed = datasets / "stack_cube_seed.hdf5"
    if local_seed.is_file():
        return local_seed
    if seed_path.is_file() and seed_path.resolve() != local_seed.resolve():
        shutil.copy2(seed_path, local_seed)
    return local_seed if local_seed.is_file() else seed_path


def _resolve_seed_input(output_dir: Path) -> Path | None:
    datasets = _datasets_dir(output_dir)
    for name in ("stack_cube_seed_annotated.hdf5", "stack_cube_seed.hdf5"):
        candidate = datasets / name
        if candidate.is_file():
            return candidate

    env_seed = (os.environ.get("ISAACLAB_STACK_CUBE_DEFAULT_SEED") or "").strip()
    if env_seed:
        seed_path = Path(env_seed).expanduser()
        if not seed_path.is_absolute():
            seed_path = (project_root() / seed_path).resolve()
        if seed_path.is_file():
            return seed_path
    return None


def _build_isaaclab_subprocess_env() -> dict[str, str]:
    """Mirror backend IsaacLabCliRunner env so isaaclab.sh picks the Isaac conda env."""
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env["TERM"] = "xterm"
    python_path = (os.environ.get("ISAACLAB_PYTHON") or "").strip()
    if python_path:
        env["ISAACLAB_PYTHON"] = python_path
        python = Path(python_path)
        if python.name == "python" and python.is_file():
            conda_prefix = python.parent.parent
            if (conda_prefix / "conda-meta").is_dir():
                env["CONDA_PREFIX"] = str(conda_prefix)
                env["CONDA_DEFAULT_ENV"] = conda_prefix.name
        elif python.name == "python.sh" and python.is_file():
            env.pop("CONDA_PREFIX", None)
            env.pop("CONDA_DEFAULT_ENV", None)
    return env


def _headless_args(headless: str) -> list[str]:
    return ["--headless"] if str(headless).lower() == "true" else []


def _run_isaaclab_script(
    sh: Path,
    root: Path,
    script_relative: str,
    *args: str,
    log_path: Path | None = None,
    headless: str = "true",
) -> int:
    cmd = [str(sh), "-p", script_relative, *_headless_args(headless), *args]
    env = _build_isaaclab_subprocess_env()
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"[platform_run] command={' '.join(cmd)}\n")
        with open(log_path, "a", encoding="utf-8") as handle:
            proc = subprocess.run(
                cmd,
                cwd=str(root),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            return proc.returncode
    proc = subprocess.run(cmd, cwd=str(root), env=env)
    return proc.returncode


def mode_check(_: argparse.Namespace) -> int:
    payload = {
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "backend": BACKEND,
        "status": "registered",
        "requires_isaac_lab": True,
        "can_generate_data": True,
        "data_format": ["HDF5", "Zarr"],
        "task_package_path": str(task_package_root().relative_to(project_root())),
        "isaac_lab_available": isaac_lab_available(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def mode_record_seed(args: argparse.Namespace) -> int:
    root, sh = require_isaac_lab()
    output_dir = Path(args.output_dir).resolve()
    datasets = _datasets_dir(output_dir)
    seed_path = datasets / "stack_cube_seed.hdf5"
    log_path = output_dir / "logs" / "platform_run.log"
    rc = _run_isaaclab_script(
        sh,
        root,
        "scripts/tools/record_demos.py",
        "--task",
        RECORD_TASK,
        "--device",
        "cpu",
        "--teleop_device",
        "keyboard",
        "--dataset_file",
        str(seed_path),
        "--num_demos",
        str(args.num_demos),
        log_path=log_path,
        headless=args.headless,
    )
    if rc != 0:
        print(f"record_seed failed with exit code {rc}", file=sys.stderr)
    return rc


def mode_annotate_seed(args: argparse.Namespace) -> int:
    root, sh = require_isaac_lab()
    output_dir = Path(args.output_dir).resolve()
    device = _resolve_device(getattr(args, "device", None))
    _write_job_phase(output_dir, "annotating_mimic", progress=20)
    datasets = _datasets_dir(output_dir)
    input_file = datasets / "stack_cube_seed.hdf5"
    if not input_file.is_file():
        resolved = _resolve_seed_input(output_dir)
        if resolved is None:
            print(f"Missing seed file: {input_file}", file=sys.stderr)
            return 1
        input_file = _ensure_local_seed_hdf5(output_dir, resolved)
    output_file = datasets / "stack_cube_seed_annotated.hdf5"
    if not input_file.is_file():
        print(f"Missing seed file: {input_file}", file=sys.stderr)
        return 1
    log_path = output_dir / "logs" / "platform_run.log"
    return _run_isaaclab_script(
        sh,
        root,
        "scripts/imitation_learning/isaaclab_mimic/annotate_demos.py",
        "--task",
        MIMIC_TASK,
        "--device",
        device,
        "--auto",
        "--input_file",
        str(input_file),
        "--output_file",
        str(output_file),
        log_path=log_path,
        headless=args.headless,
    )


def mode_generate_mimic(args: argparse.Namespace) -> int:
    root, sh = require_isaac_lab()
    output_dir = Path(args.output_dir).resolve()
    device = _resolve_device(getattr(args, "device", None))
    datasets = _datasets_dir(output_dir)
    log_path = output_dir / "logs" / "platform_run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _write_job_phase(output_dir, "starting_isaac_lab", progress=12)

    seed_input = _resolve_seed_input(output_dir)
    if seed_input is None:
        print(
            "Seed demonstration is required for Mimic generation. "
            "Run --mode record_seed and --mode annotate_seed first, "
            "or set ISAACLAB_STACK_CUBE_DEFAULT_SEED.",
            file=sys.stderr,
        )
        return 1

    annotated = datasets / "stack_cube_seed_annotated.hdf5"
    if annotated.is_file():
        seed_input = annotated
    else:
        _ensure_local_seed_hdf5(output_dir, seed_input)
        print("[platform_run] annotating seed demonstrations before mimic generation", file=sys.stderr)
        rc = mode_annotate_seed(args)
        if rc != 0:
            _write_job_phase(
                output_dir,
                "failed",
                progress=100,
                progress_message=f"annotate_seed failed with exit code {rc}",
            )
            return rc
        seed_input = annotated
        if not seed_input.is_file():
            print(f"Missing annotated seed file: {seed_input}", file=sys.stderr)
            _write_job_phase(
                output_dir,
                "failed",
                progress=100,
                progress_message=f"Missing annotated seed file: {seed_input}",
            )
            return 1

    _write_job_phase(output_dir, "generating_mimic", progress=35)
    output_file = datasets / "dataset.hdf5"
    generation_trials = max(1, int(args.num_demos))
    num_envs = min(generation_trials, 10)

    rc = _run_isaaclab_script(
        sh,
        root,
        "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py",
        "--task",
        MIMIC_TASK,
        "--device",
        device,
        "--num_envs",
        str(num_envs),
        "--generation_num_trials",
        str(generation_trials),
        "--input_file",
        str(seed_input),
        "--output_file",
        str(output_file),
        log_path=log_path,
        headless=args.headless,
    )
    if rc != 0:
        print(f"generate_mimic failed with exit code {rc}", file=sys.stderr)
        _write_job_phase(
            output_dir,
            "failed",
            progress=100,
            progress_message=f"generate_mimic failed with exit code {rc}",
        )
        return rc

    if output_file.is_file():
        generated_copy = datasets / "stack_cube_generated.hdf5"
        if generated_copy.resolve() != output_file.resolve():
            shutil.copy2(output_file, generated_copy)
    return 0


def mode_convert_zarr(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    datasets = _datasets_dir(output_dir)
    converter = run_dir() / "convert_isaac_hdf5_to_zarr.py"
    if not converter.is_file():
        print(f"Missing converter script: {converter}", file=sys.stderr)
        return 1

    input_hdf5 = datasets / "dataset.hdf5"
    if not input_hdf5.is_file():
        input_hdf5 = datasets / "stack_cube_generated.hdf5"
    if not input_hdf5.is_file():
        print(f"No HDF5 dataset found under {datasets}", file=sys.stderr)
        return 1

    zarr_out = datasets / "dataset.zarr"
    cmd = [
        sys.executable,
        str(converter),
        "--input",
        str(input_hdf5),
        "--output",
        str(zarr_out),
    ]
    proc = subprocess.run(cmd)
    return proc.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Platform entry for Isaac Lab Franka Stack Cube task.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=("record_seed", "annotate_seed", "generate_mimic", "convert_zarr", "check"),
        help="Execution mode.",
    )
    parser.add_argument("--task-id", default=TASK_ID, help="Platform task ID.")
    parser.add_argument(
        "--output-dir",
        default=str(project_root() / "runs" / "data_generation" / TASK_ID),
        help="Job output directory.",
    )
    parser.add_argument("--headless", default="true", choices=("true", "false"), help="Headless simulation flag.")
    parser.add_argument("--num-demos", type=int, default=10, help="Number of demonstrations to generate/record.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (reserved for downstream scripts).")
    parser.add_argument(
        "--device",
        default=_resolve_device(None),
        help="Isaac Lab simulation device (default: cuda:0 or ISAACLAB_DEVICE).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.task_id != TASK_ID:
        print(f"Unsupported task-id: {args.task_id}", file=sys.stderr)
        return 1

    handlers = {
        "check": mode_check,
        "record_seed": mode_record_seed,
        "annotate_seed": mode_annotate_seed,
        "generate_mimic": mode_generate_mimic,
        "convert_zarr": mode_convert_zarr,
    }
    return handlers[args.mode](args)


if __name__ == "__main__":
    raise SystemExit(main())
