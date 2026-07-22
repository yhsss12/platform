#!/usr/bin/env python3
"""Platform bridge: run pi0 training via openpi or platform shim (smoke/mock)."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PI0_HDF5_NOT_SUPPORTED_MESSAGE = (
    "pi0 当前需要 LeRobot/HF 格式数据，平台 HDF5 转换器尚未接入。"
)
FAKE_DATA_OPENPI_CONFIGS = frozenset({"debug", "debug_pi05", "debug_restore", "pi0_mock"})
DEBUG_PI05_SMOKE_CONFIG = "debug_pi05"
DEBUG_PI05_SMOKE_LOG_MESSAGE = "debug_pi05 smoke mode: skip platform dataset conversion"
LEROBOT_OPENPI_CONFIG_MARKERS = ("libero", "lerobot", "aloha", "droid")
UNSUPPORTED_OPENPI_CLI_FLAGS = frozenset(
    {"--config-path", "--learning-rate", "--output-dir", "--learning_rate", "--output_dir"}
)


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML required to load platform config") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("platform config must be a mapping")
    return data


def _append_metrics(metrics_path: Path, row: dict[str, Any]) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def _bootstrap_openpi_path(root: Path) -> None:
    for candidate in (root / "src", root / "packages" / "openpi-client" / "src"):
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _resolve_base_config(config: dict[str, Any]) -> str:
    return str(config.get("openpi_base_config") or os.environ.get("OPENPI_BASE_CONFIG") or "").strip()


def _openpi_base_config_requires_lerobot(base_config: str) -> bool:
    if not base_config or base_config in FAKE_DATA_OPENPI_CONFIGS:
        return False
    lowered = base_config.lower()
    return any(marker in lowered for marker in LEROBOT_OPENPI_CONFIG_MARKERS)


def _manifest_has_lerobot_dataset(manifest: dict[str, Any]) -> bool:
    artifacts = manifest.get("artifacts") or {}
    lerobot_path = artifacts.get("lerobot") or artifacts.get("lerobotPath")
    if lerobot_path:
        root = Path(str(lerobot_path)).expanduser()
        if (root / "meta" / "info.json").is_file() or (root / "dataset_index.json").is_file():
            data_dir = root / "data"
            if data_dir.is_dir() and any(data_dir.rglob("episode_*")):
                return True
    dataset = manifest.get("dataset")
    if isinstance(dataset, dict) and (dataset.get("lerobotPath") or dataset.get("repo_id")):
        return True
    fmt = str(manifest.get("dataFormat") or manifest.get("format") or "").lower()
    return "lerobot" in fmt or fmt in {"hf", "huggingface", "lerobot_index", "platform_lerobot_export_v1"}


def _validate_openpi_dataset_readiness(config: dict[str, Any]) -> str | None:
    base_config = _resolve_base_config(config)
    if not _openpi_base_config_requires_lerobot(base_config):
        return None

    dataset = config.get("dataset") or {}
    manifest = config.get("manifest") or {}
    has_lerobot = bool(dataset.get("lerobot_path") or dataset.get("repo_id"))
    has_lerobot = has_lerobot or _manifest_has_lerobot_dataset(manifest)
    if has_lerobot:
        return None

    hdf5_path = str(dataset.get("hdf5_path") or "").strip()
    if hdf5_path:
        return PI0_HDF5_NOT_SUPPORTED_MESSAGE
    return PI0_HDF5_NOT_SUPPORTED_MESSAGE


def _run_platform_shim(config: dict[str, Any]) -> int:
    training = config.get("training") or {}
    paths = config.get("paths") or {}
    epochs = max(1, int(training.get("epochs") or 1))
    batch_size = int(training.get("batch_size") or 8)
    learning_rate = float(training.get("learning_rate") or 1e-4)
    seed = int(training.get("seed") or 1)
    steps_per_epoch = max(1, int(training.get("steps_per_epoch") or 4))
    total_steps = int(training.get("num_train_steps") or epochs * steps_per_epoch)

    out_dir = Path(str(paths.get("output_dir") or ".")).expanduser().resolve()
    metrics_path = Path(str(paths.get("metrics_path") or out_dir / "metrics.jsonl")).expanduser().resolve()
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"pi0 platform shim: epochs={epochs} batch_size={batch_size} "
        f"learning_rate={learning_rate} seed={seed} total_steps={total_steps}",
        flush=True,
    )

    global_step = 0
    for epoch in range(1, epochs + 1):
        for _step_in_epoch in range(1, steps_per_epoch + 1):
            global_step += 1
            loss = max(0.001, 1.0 / global_step)
            print(
                f"Step {global_step}/{total_steps} Epoch {epoch}/{epochs} Loss: {loss:.6f}",
                flush=True,
            )
            _append_metrics(
                metrics_path,
                {
                    "epoch": epoch,
                    "step": global_step,
                    "totalSteps": total_steps,
                    "trainLoss": loss,
                    "learningRate": learning_rate,
                },
            )
            time.sleep(0.005)

    final_path = ckpt_dir / "model_final.pt"
    final_path.write_bytes(b"PI0_PLATFORM_SHIM_CHECKPOINT")
    print(f"checkpoint: {final_path}", flush=True)
    print("pi0 training completed", flush=True)
    return 0


def _find_openpi_train_script(root: Path) -> Path | None:
    candidates: list[Path] = [
        root / "scripts" / "train.py",
        root / "src" / "openpi" / "training" / "train.py",
    ]
    override = os.environ.get("OPENPI_TRAIN_SCRIPT", "").strip()
    if override:
        override_path = Path(override).expanduser()
        if override_path.is_file():
            try:
                override_path.resolve().relative_to(root.resolve())
            except ValueError:
                pass
            else:
                candidates.insert(0, override_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def _build_openpi_command(config: dict[str, Any], *, train_script: Path) -> list[str]:
    base_config = _resolve_base_config(config)
    if not base_config:
        raise RuntimeError("openpi base config is not configured (OPENPI_BASE_CONFIG)")

    openpi_cfg = config.get("openpi") or {}
    platform = config.get("platform") or {}
    paths = config.get("paths") or {}
    training = config.get("training") or {}

    exp_name = str(openpi_cfg.get("exp_name") or platform.get("trainJobId") or "pi0_platform_run")
    checkpoint_base_dir = str(
        openpi_cfg.get("checkpoint_base_dir") or paths.get("checkpoint_base_dir") or paths.get("output_dir") or "."
    )
    batch_size = int(training.get("batch_size") or 8)
    num_train_steps = int(training.get("num_train_steps") or training.get("epochs") or 1)
    seed = int(training.get("seed") or 1)
    log_interval = max(1, min(int(training.get("log_interval") or 1), num_train_steps))
    save_interval = max(1, min(int(training.get("save_interval") or num_train_steps), num_train_steps))

    python_bin = os.environ.get("OPENPI_PYTHON") or sys.executable
    cmd: list[str] = [
        python_bin,
        str(train_script),
        base_config,
        "--exp-name",
        exp_name,
        "--checkpoint-base-dir",
        checkpoint_base_dir,
        "--batch-size",
        str(batch_size),
        "--num-train-steps",
        str(num_train_steps),
        "--seed",
        str(seed),
        "--log-interval",
        str(log_interval),
        "--save-interval",
        str(save_interval),
        "--overwrite",
        "--no-wandb-enabled",
    ]

    learning_rate = training.get("learning_rate")
    if learning_rate is not None:
        lr_text = str(learning_rate)
        cmd.extend(
            [
                "--lr-schedule.peak-lr",
                lr_text,
                "--lr-schedule.decay-lr",
                lr_text,
            ]
        )

    for token in cmd:
        if token in UNSUPPORTED_OPENPI_CLI_FLAGS:
            raise RuntimeError(f"internal command builder produced unsupported openpi flag: {token}")

    return cmd


def _resolve_openpi_checkpoint_root(config: dict[str, Any]) -> Path:
    openpi_cfg = config.get("openpi") or {}
    platform = config.get("platform") or {}
    paths = config.get("paths") or {}
    base_config = _resolve_base_config(config)
    exp_name = str(openpi_cfg.get("exp_name") or platform.get("trainJobId") or "pi0_platform_run")
    checkpoint_base_dir = Path(
        str(openpi_cfg.get("checkpoint_base_dir") or paths.get("checkpoint_base_dir") or paths.get("output_dir") or ".")
    ).expanduser().resolve()
    return checkpoint_base_dir / base_config / exp_name


def _find_latest_openpi_step_checkpoint(openpi_ckpt_dir: Path) -> tuple[Path | None, int | None]:
    if not openpi_ckpt_dir.is_dir():
        return None, None

    step_dirs: list[tuple[int, Path]] = []
    for child in openpi_ckpt_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            step_dirs.append((int(child.name), child))
    if step_dirs:
        step, path = max(step_dirs, key=lambda item: item[0])
        return path, step

    pt_files = [
        path
        for path in openpi_ckpt_dir.rglob("*")
        if path.is_file() and path.suffix in {".pt", ".pth"} and path.stat().st_size > 0
    ]
    if pt_files:
        latest = max(pt_files, key=lambda path: path.stat().st_mtime)
        return latest, None

    return None, None


def _materialize_platform_final_checkpoint(
    *,
    source: Path,
    final_path: Path,
    base_config: str,
    exp_name: str,
    step: int | None,
    dataset_meta: dict[str, Any] | None = None,
) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_meta = dict(dataset_meta or {})
    if source.is_file() and source.suffix in {".pt", ".pth"}:
        try:
            raw = source.read_bytes()
            if raw.startswith(b"PI0_PLATFORM_SHIM"):
                raise RuntimeError("refusing to materialize platform shim checkpoint as pi0 final asset")
        except OSError:
            pass
        if final_path.exists() or final_path.is_symlink():
            final_path.unlink()
        shutil.copy2(source, final_path)
        return

    payload = {
        "format": "openpi_orbax_v1",
        "backend": "pi0",
        "openpiBaseConfig": base_config,
        "expName": exp_name,
        "step": step,
        "checkpointPath": str(source.resolve()),
        "action_dim": int(dataset_meta.get("action_dim") or 7),
        "action_horizon": int(dataset_meta.get("action_horizon") or 8),
        "camera_keys": list(dataset_meta.get("camera_keys") or []),
        "low_dim_keys": list(dataset_meta.get("low_dim_keys") or []),
        "task_prompt": dataset_meta.get("task_prompt"),
        "lerobot_path": dataset_meta.get("lerobot_path"),
    }
    final_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_openpi_metrics(
    log_text: str,
    *,
    total_steps: int,
    learning_rate: float | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in re.finditer(r"Step\s+(\d+):\s*(.+)", log_text):
        step = int(match.group(1))
        metrics_blob = match.group(2)
        train_loss = None
        lr_value = learning_rate
        for token in re.finditer(r"([\w\-]+)=([\d.eE+\-]+)", metrics_blob):
            key = token.group(1).lower().replace("-", "_")
            if key == "loss":
                train_loss = float(token.group(2))
            elif key in {"learning_rate", "lr"}:
                lr_value = float(token.group(2))
        if train_loss is None:
            continue
        row: dict[str, Any] = {
            "step": step,
            "epoch": step + 1,
            "totalSteps": total_steps,
            "trainLoss": train_loss,
        }
        if lr_value is not None:
            row["learningRate"] = lr_value
        rows.append(row)
    return rows


def _log_debug_pi05_smoke_mode(config: dict[str, Any]) -> None:
    base_config = _resolve_base_config(config)
    smoke = config.get("smoke_mode") or {}
    if base_config == DEBUG_PI05_SMOKE_CONFIG or smoke.get("enabled"):
        print(DEBUG_PI05_SMOKE_LOG_MESSAGE, flush=True)


def _run_openpi_subprocess(config: dict[str, Any]) -> int:
    _log_debug_pi05_smoke_mode(config)
    dataset_error = _validate_openpi_dataset_readiness(config)
    if dataset_error:
        print(dataset_error, flush=True)
        return 2

    root_raw = os.environ.get("OPENPI_ROOT", "").strip()
    if not root_raw:
        raise RuntimeError("OPENPI_ROOT is not set")
    root = Path(root_raw).expanduser().resolve()
    train_script = _find_openpi_train_script(root)
    if train_script is None:
        raise RuntimeError("openpi train script not found under OPENPI_ROOT")

    openpi_cfg = config.get("openpi") or {}
    platform = config.get("platform") or {}
    paths = config.get("paths") or {}
    training = config.get("training") or {}
    base_config = _resolve_base_config(config)
    exp_name = str(openpi_cfg.get("exp_name") or platform.get("trainJobId") or "pi0_platform_run")
    num_train_steps = int(training.get("num_train_steps") or training.get("epochs") or 1)
    learning_rate = training.get("learning_rate")
    metrics_path = Path(str(paths.get("metrics_path") or ".")).expanduser().resolve()
    final_path = Path(
        str(
            openpi_cfg.get("platform_final_checkpoint")
            or paths.get("platform_final_checkpoint")
            or Path(str(paths.get("output_dir") or ".")) / "checkpoints" / "model_final.pt"
        )
    ).expanduser().resolve()

    cmd = _build_openpi_command(config, train_script=train_script)

    env = os.environ.copy()
    env["OPENPI_ROOT"] = str(root)
    _bootstrap_openpi_path(root)

    print("openpi command: " + " ".join(cmd), flush=True)
    if learning_rate is None:
        print(
            "openpi CLI 当前未接入 learningRate override，使用 base config 默认 learning rate。",
            flush=True,
        )
    else:
        print(
            f"openpi CLI learningRate override via --lr-schedule.peak-lr={learning_rate}",
            flush=True,
        )

    completed = subprocess.run(
        cmd,
        cwd=str(root),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    log_output = "\n".join(part for part in (completed.stdout, completed.stderr) if part)
    if log_output:
        print(log_output, flush=True)

    for row in _parse_openpi_metrics(
        log_output,
        total_steps=num_train_steps,
        learning_rate=float(learning_rate) if learning_rate is not None else None,
    ):
        _append_metrics(metrics_path, row)

    if completed.returncode != 0:
        return int(completed.returncode)

    openpi_ckpt_dir = _resolve_openpi_checkpoint_root(config)
    source_ckpt, step = _find_latest_openpi_step_checkpoint(openpi_ckpt_dir)
    if source_ckpt is None:
        print(
            f"Final checkpoint not found under openpi checkpoint directory: {openpi_ckpt_dir}",
            flush=True,
        )
        return 1

    _materialize_platform_final_checkpoint(
        source=source_ckpt,
        final_path=final_path,
        base_config=base_config,
        exp_name=exp_name,
        step=step,
        dataset_meta={
            "action_dim": int((config.get("dataset") or {}).get("action_dim") or 7),
            "action_horizon": int((config.get("dataset") or {}).get("action_horizon") or 8),
            "camera_keys": list((config.get("dataset") or {}).get("camera_keys") or []),
            "low_dim_keys": list((config.get("dataset") or {}).get("low_dim_keys") or []),
            "task_prompt": (config.get("dataset") or {}).get("task_prompt"),
            "lerobot_path": (config.get("dataset") or {}).get("lerobot_path"),
        },
    )
    if not final_path.is_file() or final_path.stat().st_size <= 0:
        print(f"Final checkpoint not found at platform path: {final_path}", flush=True)
        return 1

    print(f"checkpoint: {final_path}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Platform pi0 / openpi training bridge")
    parser.add_argument("--platform-config", required=True, help="Platform-generated openpi config yaml/json")
    parser.add_argument("--dataset", required=False, help="HDF5 dataset path (for logging)")
    parser.add_argument("--out-dir", required=True, help="Checkpoint output directory")
    parser.add_argument("--metrics-path", required=True, help="metrics.jsonl path")
    args = parser.parse_args()

    config_path = Path(args.platform_config).expanduser().resolve()
    config = _load_config(config_path)
    paths = dict(config.get("paths") or {})
    out_dir = Path(args.out_dir).expanduser().resolve()
    paths["output_dir"] = str(out_dir)
    paths["checkpoint_base_dir"] = str(out_dir)
    paths["metrics_path"] = str(Path(args.metrics_path).expanduser().resolve())
    paths["platform_final_checkpoint"] = str(out_dir / "checkpoints" / "model_final.pt")
    config["paths"] = paths
    openpi_cfg = dict(config.get("openpi") or {})
    openpi_cfg.setdefault("checkpoint_base_dir", str(out_dir))
    openpi_cfg.setdefault("platform_final_checkpoint", paths["platform_final_checkpoint"])
    config["openpi"] = openpi_cfg
    if args.dataset:
        config.setdefault("dataset", {})["hdf5_path"] = str(Path(args.dataset).expanduser().resolve())

    mode = os.environ.get("PI0_TRAIN_MODE", "openpi").strip().lower()
    if mode == "shim":
        print(
            "PI0_TRAIN_MODE=shim is deprecated for production; use mock/real openpi subprocess instead.",
            flush=True,
        )
        return _run_platform_shim(config)

    return _run_openpi_subprocess(config)


if __name__ == "__main__":
    raise SystemExit(main())
