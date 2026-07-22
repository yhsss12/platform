"""非阻塞事件发射器：写入 event_bus + platform_events 表。"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from app.core.events.event_bus import get_event_bus
from app.core.events.event_models import EventType, PlatformEvent

logger = logging.getLogger(__name__)


def _persist_event(event: PlatformEvent) -> None:
    try:
        from app.core.database import SessionLocal
        from app.models.platform_event import PlatformEventRecord

        with SessionLocal() as db:
            existing = db.query(PlatformEventRecord).filter(PlatformEventRecord.event_id == event.event_id).one_or_none()
            if existing is not None:
                return
            db.add(
                PlatformEventRecord(
                    event_id=event.event_id,
                    event_type=event.event_type,
                    job_id=event.job_id,
                    timestamp=event.timestamp,
                    payload=event.payload,
                    source=event.source,
                )
            )
            db.commit()
    except Exception as exc:
        logger.debug("platform event persist skipped event_id=%s: %s", event.event_id, exc)


def emit_event(
    event_type: EventType | str,
    job_id: str,
    *,
    payload: Optional[dict[str, Any]] = None,
    source: str = "platform",
    async_dispatch: bool = True,
) -> PlatformEvent:
    """发射事件；默认异步，不阻塞调用方。"""
    event = PlatformEvent.create(event_type, job_id, payload=payload, source=source)

    def _dispatch() -> None:
        get_event_bus().publish(event)
        _persist_event(event)

    if async_dispatch:
        threading.Thread(target=_dispatch, name=f"evt-{event.event_type[:12]}", daemon=True).start()
    else:
        _dispatch()
    return event
