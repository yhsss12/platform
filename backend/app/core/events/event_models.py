"""平台事件模型定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class EventType(str, Enum):
    TRAINING_STARTED = "TRAINING_STARTED"
    TRAINING_COMPLETED = "TRAINING_COMPLETED"
    CHECKPOINT_CREATED = "CHECKPOINT_CREATED"
    DATASET_INGESTED = "DATASET_INGESTED"
    EVAL_STARTED = "EVAL_STARTED"
    EVAL_COMPLETED = "EVAL_COMPLETED"
    ARTIFACT_UPLOADED = "ARTIFACT_UPLOADED"


@dataclass
class PlatformEvent:
    event_id: str
    event_type: str
    job_id: str
    timestamp: datetime
    payload: dict[str, Any]
    source: str

    @classmethod
    def create(
        cls,
        event_type: EventType | str,
        job_id: str,
        *,
        payload: dict[str, Any] | None = None,
        source: str = "platform",
        event_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> PlatformEvent:
        if isinstance(event_type, EventType):
            type_text = event_type.value
        else:
            type_text = str(event_type or "").strip()
        return cls(
            event_id=event_id or uuid4().hex,
            event_type=type_text,
            job_id=(job_id or "").strip(),
            timestamp=timestamp or datetime.now(timezone.utc),
            payload=dict(payload or {}),
            source=source,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "job_id": self.job_id,
            "timestamp": self.timestamp.isoformat(),
            "payload": self.payload,
            "source": self.source,
        }
