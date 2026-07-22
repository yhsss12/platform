"""训练指标归一化：metrics.jsonl + 日志解析 + status.json。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

EPOCH_LOG_PATTERN = re.compile(r"(?<!Validation )Epoch\s+(\d+)\s+Loss:", re.IGNORECASE)
TRAIN_EPOCH_LOG_PATTERN = re.compile(r"Train\s+Epoch\s+(\d+)", re.IGNORECASE)
VALIDATION_EPOCH_LOG_PATTERN = re.compile(
    r"Validation\s+Epoch\s+(\d+)\s+Loss:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)
LOSS_LOG_PATTERN = re.compile(r"Loss:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
JSON_LOSS_LOG_PATTERN = re.compile(r'"Loss":\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')

METRICS_JSONL = "metrics.jsonl"

TRAINING_LOG_NOISE_PATTERNS = (
    re.compile(r"^\s*remote command:", re.IGNORECASE),
    re.compile(r"^\s*remote pid:", re.IGNORECASE),
    re.compile(r"^\s*\d+%\|"),
    re.compile(r"^\s*\[\d+/\d+\]"),
    re.compile(r"^\s*reconcile:", re.IGNORECASE),
    re.compile(r"^\s*\[.*UTC\]\s*(sync dir|uploading|download remote|launch remote|FAILED:)", re.IGNORECASE),
)


def sanitize_training_log_for_display(text: str) -> str:
    """过滤 runner / 传输进度噪声，保留训练过程日志。"""
    kept: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue
        if any(pattern.search(line) for pattern in TRAINING_LOG_NOISE_PATTERNS):
            continue
        kept.append(line)
    return "\n".join(kept).strip("\n")


def resolve_training_log_for_display(train_job_dir: Path) -> Path:
    """详情页展示用日志：remote_ssh 优先 logs/train.log。"""
    primary = train_job_dir / "logs" / "train.log"
    status_path = train_job_dir / "status.json"
    execution_mode = ""
    if status_path.is_file():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                execution_mode = str(payload.get("executionMode") or "").lower()
        except (OSError, json.JSONDecodeError):
            pass
    if execution_mode == "remote_ssh" and primary.is_file():
        return primary
    return resolve_training_metrics_log_path(train_job_dir)


def collect_training_log_paths(train_job_dir: Path) -> list[Path]:
    """收集可用于指标解析的训练日志（主日志 + robomimic 嵌套 log.txt）。"""
    paths: list[Path] = []
    primary = train_job_dir / "logs" / "train.log"
    stdout = train_job_dir / "logs" / "stdout.log"
    if primary.is_file():
        paths.append(primary)
    elif stdout.is_file():
        paths.append(stdout)
    paths.extend(sorted(train_job_dir.glob("checkpoints/**/logs/log.txt")))

    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _train_epoch_count(log_path: Path) -> int:
    if not log_path.is_file():
        return 0
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return len(TRAIN_EPOCH_LOG_PATTERN.findall(text))


def resolve_training_metrics_log_path(train_job_dir: Path) -> Path:
    """选取 Train Epoch 条目最多的日志，用于展示与指标解析。"""
    candidates = collect_training_log_paths(train_job_dir)
    if not candidates:
        return train_job_dir / "logs" / "train.log"

    def sort_key(path: Path) -> tuple[int, int]:
        epoch_count = _train_epoch_count(path)
        try:
            size = path.stat().st_size if path.is_file() else 0
        except OSError:
            size = 0
        return epoch_count, size

    return max(candidates, key=sort_key)


def _resolve_training_log_path(train_job_dir: Path) -> Path:
    return resolve_training_metrics_log_path(train_job_dir)


def _parse_training_log(log_path: Path, total_epochs: int) -> tuple[int, Optional[float]]:
    if not log_path.is_file():
        return 0, None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0, None

    epoch = 0
    loss: Optional[float] = None
    for line in text.splitlines():
        train_epoch_match = TRAIN_EPOCH_LOG_PATTERN.search(line)
        if train_epoch_match:
            try:
                epoch = max(epoch, int(train_epoch_match.group(1)))
            except ValueError:
                pass
        epoch_match = EPOCH_LOG_PATTERN.search(line)
        if epoch_match:
            try:
                epoch = max(epoch, int(epoch_match.group(1)))
            except ValueError:
                pass
        json_loss_match = JSON_LOSS_LOG_PATTERN.search(line)
        if json_loss_match:
            try:
                loss = float(json_loss_match.group(1))
            except ValueError:
                pass
        loss_match = LOSS_LOG_PATTERN.search(line)
        if loss_match:
            try:
                loss = float(loss_match.group(1))
            except ValueError:
                pass
    epoch = min(epoch, total_epochs) if total_epochs > 0 else epoch
    return epoch, loss


def parse_training_logs(train_job_dir: Path, total_epochs: int) -> tuple[int, Optional[float]]:
    """从所有候选日志中取最大 epoch 与最新 loss。"""
    epoch = 0
    loss: Optional[float] = None
    for path in collect_training_log_paths(train_job_dir):
        parsed_epoch, parsed_loss = _parse_training_log(path, total_epochs)
        if parsed_epoch > epoch:
            epoch = parsed_epoch
            loss = parsed_loss
        elif parsed_epoch == epoch and parsed_loss is not None:
            loss = parsed_loss
    return epoch, loss


def parse_pi0_metrics_from_jsonl(train_job_dir: Path, total_epochs: int) -> tuple[int, Optional[float]]:
    """Read latest epoch/loss from pi0 openpi metrics.jsonl written by run_openpi_train."""
    path = metrics_jsonl_path(train_job_dir)
    if not path.is_file():
        return 0, None
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError):
        return 0, None
    if not rows:
        return 0, None
    last = rows[-1]
    epoch = int(last.get("epoch") or 0)
    if epoch <= 0:
        global_step = int(last.get("globalStep") or last.get("step") or 0)
        steps_per_epoch = max(1, int(last.get("stepsPerEpoch") or 1))
        if global_step > 0:
            epoch = min(total_epochs, max(1, (global_step + steps_per_epoch - 1) // steps_per_epoch))
    loss_val = last.get("trainLoss")
    if loss_val is None:
        loss_val = last.get("loss")
    loss = float(loss_val) if loss_val is not None else None
    return epoch, loss


def metrics_jsonl_path(train_job_dir: Path) -> Path:
    return train_job_dir / "artifacts" / METRICS_JSONL


def append_metrics_point(
    train_job_dir: Path,
    *,
    epoch: int,
    loss: Optional[float] = None,
    train_loss: Optional[float] = None,
    valid_loss: Optional[float] = None,
) -> None:
    if epoch <= 0:
        return
    existing = read_metrics_history(train_job_dir)
    for row in existing:
        if int(row.get("epoch") or 0) != int(epoch):
            continue
        if row.get("trainLoss") is not None and train_loss is None and valid_loss is None and loss is not None:
            return
    path = metrics_jsonl_path(train_job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {"epoch": int(epoch)}
    if train_loss is not None:
        payload["trainLoss"] = train_loss
    if valid_loss is not None:
        payload["validLoss"] = valid_loss
    if loss is not None and train_loss is None:
        payload["loss"] = loss
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def rewrite_metrics_history(train_job_dir: Path, points: list[dict[str, Any]]) -> None:
    """用完整序列覆盖 metrics.jsonl（训练完成或指标补齐时）。"""
    path = metrics_jsonl_path(train_job_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for row in sorted(points, key=lambda item: int(item.get("epoch") or 0)):
        epoch = int(row.get("epoch") or 0)
        if epoch <= 0:
            continue
        payload: dict[str, Any] = {"epoch": epoch}
        train_loss = row.get("trainLoss", row.get("loss"))
        if train_loss is not None:
            payload["trainLoss"] = train_loss
        if row.get("validLoss") is not None:
            payload["validLoss"] = row["validLoss"]
        lines.append(json.dumps(payload, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_metrics_history(train_job_dir: Path) -> list[dict[str, Any]]:
    path = metrics_jsonl_path(train_job_dir)
    if not path.is_file():
        return []
    points: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                points.append(row)
    except (OSError, json.JSONDecodeError):
        return []
    return points


def _extract_loss_value(line: str) -> Optional[float]:
    json_loss_match = JSON_LOSS_LOG_PATTERN.search(line)
    if json_loss_match:
        try:
            return float(json_loss_match.group(1))
        except ValueError:
            pass
    loss_match = LOSS_LOG_PATTERN.search(line)
    if loss_match:
        try:
            return float(loss_match.group(1))
        except ValueError:
            pass
    return None


def _row_train_loss(row: dict[str, Any]) -> Optional[float]:
    for key in ("trainLoss", "train_loss", "loss"):
        if row.get(key) is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                pass
    return None


def _row_valid_loss(row: dict[str, Any]) -> Optional[float]:
    for key in ("validLoss", "valid_loss", "valLoss", "validationLoss", "Validation Loss"):
        if row.get(key) is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                pass
    return None


def _row_richness(row: dict[str, Any]) -> int:
    score = 0
    if _row_train_loss(row) is not None:
        score += 2
    if _row_valid_loss(row) is not None:
        score += 1
    if row.get("totalEpochs") is not None:
        score += 1
    return score


def _dedupe_loss_series(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for row in points:
        epoch = int(row.get("epoch") or 0)
        if epoch <= 0:
            continue
        existing = merged.get(epoch, {"epoch": epoch})
        train_loss = _row_train_loss(row)
        valid_loss = _row_valid_loss(row)
        merged[epoch] = {
            "epoch": epoch,
            "trainLoss": train_loss if train_loss is not None else existing.get("trainLoss"),
            "validLoss": valid_loss if valid_loss is not None else existing.get("validLoss"),
        }
    return [merged[key] for key in sorted(merged.keys())]


def _series_from_log(log_path: Path, total_epochs: int) -> list[dict[str, Any]]:
    if not log_path.is_file():
        return []
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    points: list[dict[str, Any]] = []
    current_epoch = 0
    mode: Optional[str] = None

    for line in text.splitlines():
        valid_epoch_match = VALIDATION_EPOCH_LOG_PATTERN.search(line)
        if valid_epoch_match:
            try:
                current_epoch = int(valid_epoch_match.group(1))
                loss_value = float(valid_epoch_match.group(2))
            except ValueError:
                continue
            capped_epoch = min(current_epoch, total_epochs) if total_epochs > 0 else current_epoch
            points.append({"epoch": capped_epoch, "validLoss": loss_value})
            mode = None
            continue

        train_epoch_match = TRAIN_EPOCH_LOG_PATTERN.search(line)
        if train_epoch_match:
            try:
                current_epoch = int(train_epoch_match.group(1))
                mode = "train"
            except ValueError:
                pass
            loss_value = _extract_loss_value(line)
            if loss_value is not None and current_epoch > 0:
                capped_epoch = min(current_epoch, total_epochs) if total_epochs > 0 else current_epoch
                points.append({"epoch": capped_epoch, "trainLoss": loss_value})
                mode = None
            continue

        epoch_match = EPOCH_LOG_PATTERN.search(line)
        if epoch_match:
            try:
                current_epoch = max(current_epoch, int(epoch_match.group(1)))
                mode = "train"
            except ValueError:
                pass
            loss_value = _extract_loss_value(line)
            if loss_value is not None and current_epoch > 0:
                capped_epoch = min(current_epoch, total_epochs) if total_epochs > 0 else current_epoch
                points.append({"epoch": capped_epoch, "trainLoss": loss_value})
                mode = None
            continue

        if current_epoch <= 0 or mode is None:
            continue

        loss_value = _extract_loss_value(line)
        if loss_value is None:
            continue

        capped_epoch = min(current_epoch, total_epochs) if total_epochs > 0 else current_epoch
        row: dict[str, Any] = {"epoch": capped_epoch}
        if mode == "valid":
            row["validLoss"] = loss_value
        else:
            row["trainLoss"] = loss_value
        points.append(row)
        mode = None

    return points


def _series_from_training_logs(train_job_dir: Path, total_epochs: int) -> list[dict[str, Any]]:
    sources = [
        _series_from_log(path, total_epochs)
        for path in collect_training_log_paths(train_job_dir)
    ]
    return _merge_points(*sources)


def _merge_points(*sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[int, dict[str, Any]] = {}
    for source in sources:
        for row in source:
            epoch = int(row.get("epoch") or 0)
            if epoch <= 0:
                continue
            existing = merged.get(epoch, {"epoch": epoch})
            train_loss = _row_train_loss(row)
            valid_loss = _row_valid_loss(row)
            merged[epoch] = {
                "epoch": epoch,
                "trainLoss": train_loss if train_loss is not None else existing.get("trainLoss"),
                "validLoss": valid_loss if valid_loss is not None else existing.get("validLoss"),
            }
    return _dedupe_loss_series([merged[key] for key in sorted(merged.keys())])


def sync_metrics_from_logs(train_job_dir: Path, status: dict[str, Any]) -> list[dict[str, Any]]:
    """从全部训练日志重建 metrics.jsonl，返回合并后的 loss 序列。"""
    total_epochs = int(status.get("totalEpochs") or 0)
    loss_series = _series_from_training_logs(
        train_job_dir,
        total_epochs if total_epochs > 0 else max(1, int(status.get("epoch") or 1)),
    )
    if loss_series:
        rewrite_metrics_history(train_job_dir, loss_series)
    return loss_series


def normalized_training_metrics(
    train_job_dir: Path,
    status: dict[str, Any],
) -> dict[str, Any]:
    total_epochs = int(status.get("totalEpochs") or 0)
    history = read_metrics_history(train_job_dir)
    log_series = _series_from_training_logs(
        train_job_dir,
        total_epochs if total_epochs > 0 else max(1, int(status.get("epoch") or 1)),
    )
    loss_series = _merge_points(history, log_series)

    status_epoch = int(status.get("epoch") or 0)
    series_max = max((int(row["epoch"]) for row in loss_series), default=0)

    job_status = str(status.get("status") or "").lower()
    if job_status in {"completed", "failed", "canceled"} and (
        not loss_series or (total_epochs > 0 and series_max < min(status_epoch, total_epochs))
    ):
        loss_series = sync_metrics_from_logs(train_job_dir, status)
        series_max = max((int(row["epoch"]) for row in loss_series), default=0)

    current_epoch = max(status_epoch, series_max)

    if not loss_series:
        parsed_epoch, parsed_loss = parse_training_logs(
            train_job_dir,
            max(total_epochs, 1),
        )
        current_epoch = max(current_epoch, parsed_epoch)
        if parsed_epoch > 0 and parsed_loss is not None:
            loss_series = [{"epoch": parsed_epoch, "trainLoss": parsed_loss}]

    current_loss: Optional[float] = None
    for row in reversed(loss_series):
        if int(row.get("epoch") or 0) == current_epoch:
            value = row.get("trainLoss")
            if value is None:
                value = row.get("loss")
            if value is not None:
                current_loss = float(value)
                break
    if current_loss is None and status.get("loss") is not None:
        try:
            current_loss = float(status["loss"])
        except (TypeError, ValueError):
            current_loss = None

    best_loss: Optional[float] = None
    for row in loss_series:
        for key in ("trainLoss", "validLoss", "loss"):
            value = row.get(key)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if best_loss is None or numeric < best_loss:
                best_loss = numeric

    job_status = str(status.get("status") or "").lower()
    from app.services.training_job_status import compute_training_progress_fraction, normalize_training_status_token

    progress = compute_training_progress_fraction(
        current_epoch,
        total_epochs,
        job_status,
    )

    final_loss: Optional[float] = None
    if normalize_training_status_token(job_status) in {"completed", "success", "succeeded", "finished", "done"}:
        if loss_series:
            last_row = max(loss_series, key=lambda item: int(item.get("epoch") or 0))
            for key in ("trainLoss", "loss", "validLoss"):
                if last_row.get(key) is not None:
                    try:
                        final_loss = float(last_row[key])
                        break
                    except (TypeError, ValueError):
                        continue

    return {
        "epoch": current_epoch,
        "totalEpochs": total_epochs,
        "loss": current_loss,
        "progress": progress,
        "lossSeries": loss_series,
        "bestLoss": best_loss,
        "finalLoss": final_loss,
    }
