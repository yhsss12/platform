from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from app.core.config import settings
from app.services.experiment_config import get_experiment_config_service

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_LOG_DIR = _PROJECT_ROOT / "logs" / "experiment"
logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return value


class ExperimentLogger:
    def __init__(self, log_dir: Path = _LOG_DIR) -> None:
        self._log_dir = log_dir
        self._lock = threading.Lock()

    def current_log_path(self, ts: datetime | None = None) -> Path:
        dt = ts or datetime.now()
        return self._log_dir / f"{dt.strftime('%Y-%m-%d')}.jsonl"

    def log_event(self, *, role: str, event: str, **fields: Any) -> Dict[str, Any]:
        if not bool(getattr(settings, "EXPERIMENT_ENABLED", False)):
            event_id = str(fields.pop("event_id", "") or uuid.uuid4())
            body: Dict[str, Any] = {
                "event_id": event_id,
                "ts": str(fields.pop("ts", "") or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
                "role": role,
                "event": event,
                "disabled": True,
            }
            for key, value in fields.items():
                if value is None:
                    continue
                body[key] = _json_safe(value)
            return body

        config = get_experiment_config_service().load().experiment_method
        now = datetime.now(timezone.utc)
        event_id = str(fields.pop("event_id", "") or uuid.uuid4())
        body: Dict[str, Any] = {
            "event_id": event_id,
            "ts": str(fields.pop("ts", "") or now.isoformat().replace("+00:00", "Z")),
            "role": role,
            "event": event,
            "method": fields.pop("method", None) or config.method_code,
            "experiment_method_name": fields.pop("experiment_method_name", None) or config.name,
        }
        for key, value in fields.items():
            if value is None:
                continue
            body[key] = _json_safe(value)

        path = self.current_log_path(now)
        line = json.dumps(body, ensure_ascii=False, sort_keys=True)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                with path.open("a", encoding="utf-8") as fp:
                    fp.write(line)
                    fp.write("\n")
        except Exception as exc:
            logger.warning("experiment log write failed: %s", exc)
        return body


_experiment_logger = ExperimentLogger()


def get_experiment_logger() -> ExperimentLogger:
    return _experiment_logger


def log_experiment_event(*, role: str, event: str, **fields: Any) -> Dict[str, Any]:
    return get_experiment_logger().log_event(role=role, event=event, **fields)
