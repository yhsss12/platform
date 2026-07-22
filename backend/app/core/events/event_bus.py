"""内存事件总线：支持 async dispatch、订阅与 replay。"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Callable, Deque, Optional

from app.core.events.event_models import PlatformEvent

logger = logging.getLogger(__name__)

EventHandler = Callable[[PlatformEvent], None]

_MAX_BUFFER = 5000


class EventBus:
    """线程安全 in-memory 事件总线。"""

    def __init__(self, *, buffer_size: int = _MAX_BUFFER) -> None:
        self._handlers: list[EventHandler] = []
        self._lock = threading.Lock()
        self._buffer: Deque[PlatformEvent] = deque(maxlen=buffer_size)

    def subscribe(self, handler: EventHandler) -> None:
        with self._lock:
            if handler not in self._handlers:
                self._handlers.append(handler)

    def publish(self, event: PlatformEvent) -> None:
        with self._lock:
            self._buffer.append(event)
            handlers = list(self._handlers)
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.warning("event handler failed type=%s job_id=%s: %s", event.event_type, event.job_id, exc)

    def replay(self, events: list[PlatformEvent]) -> int:
        count = 0
        for event in events:
            self.publish(event)
            count += 1
        return count

    def recent(self, *, limit: int = 100) -> list[PlatformEvent]:
        with self._lock:
            return list(self._buffer)[-limit:]


_bus: Optional[EventBus] = None
_bus_lock = threading.Lock()


def get_event_bus() -> EventBus:
    global _bus
    if _bus is None:
        with _bus_lock:
            if _bus is None:
                _bus = EventBus()
    return _bus
