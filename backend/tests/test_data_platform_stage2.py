"""Data Platform Stage II 单元测试。"""

from __future__ import annotations

import threading
import time

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.events.event_bus import EventBus
from app.core.events.event_models import EventType, PlatformEvent
from app.db.base import Base
from app.models.artifact_lineage import ArtifactLineage
from app.services.lineage_service import record_lineage, REL_DATASET_USED_BY


@pytest.fixture()
def stage2_env(tmp_path, monkeypatch):
    from sqlalchemy.dialects.postgresql import JSONB
    from sqlalchemy.ext.compiler import compiles
    from sqlalchemy import BigInteger

    @compiles(JSONB, "sqlite")
    def _jsonb_sqlite(_type, _compiler, **_kw):
        return "JSON"

    @compiles(BigInteger, "sqlite")
    def _bigint_sqlite(_type, _compiler, **_kw):
        return "INTEGER"

    engine = create_engine("sqlite:///:memory:")
    import app.models.artifact_lineage  # noqa: F401
    import app.models.platform_event  # noqa: F401

    Base.metadata.create_all(engine)
    TestSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr("app.services.lineage_service.SessionLocal", TestSession)
    return TestSession


def test_event_bus_publish():
    bus = EventBus()
    received: list[PlatformEvent] = []
    bus.subscribe(received.append)
    event = PlatformEvent.create(EventType.TRAINING_STARTED, "train_test", payload={"status": "running"})
    bus.publish(event)
    assert len(received) == 1
    assert received[0].event_type == EventType.TRAINING_STARTED.value


def test_lineage_idempotent(stage2_env):
    TestSession = stage2_env
    record_lineage(parent_id="ds_1", child_id="train_1", relation_type=REL_DATASET_USED_BY, job_id="train_1")
    record_lineage(parent_id="ds_1", child_id="train_1", relation_type=REL_DATASET_USED_BY, job_id="train_1")
    with TestSession() as db:
        count = db.query(ArtifactLineage).count()
    assert count == 1


def test_platform_event_to_dict():
    event = PlatformEvent.create(EventType.EVAL_COMPLETED, "eval_1", payload={"successRate": 0.9})
    data = event.to_dict()
    assert data["event_type"] == "EVAL_COMPLETED"
    assert data["job_id"] == "eval_1"
    assert data["payload"]["successRate"] == 0.9


def test_emit_event_async(monkeypatch):
    from app.core.events import event_emitter as emitter_mod

    called: list[str] = []

    def _fake_persist(event):
        called.append(event.event_id)

    monkeypatch.setattr(emitter_mod, "_persist_event", _fake_persist)
    monkeypatch.setattr(emitter_mod, "get_event_bus", lambda: EventBus())

    event = emitter_mod.emit_event(EventType.ARTIFACT_UPLOADED, "train_1", payload={"uri": "minio://b/k"})
    time.sleep(0.2)
    assert event.event_id
    assert called
