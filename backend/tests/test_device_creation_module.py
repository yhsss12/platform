from __future__ import annotations

import datetime as dt
import sys
import types

import pytest
from fastapi import HTTPException
from starlette.requests import Request


class _FakeAsyncSession:
    def __init__(self):
        self.added = []
        self.commits = 0
        self.refreshes = 0

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        self.commits += 1

    async def refresh(self, obj):
        self.refreshes += 1


class _User:
    def __init__(self, user_id: str, role: str):
        self.id = user_id
        self.role = role


class _DeviceObj:
    def __init__(self, **kw):
        self.id = kw.get("id", 1)
        self.name = kw.get("name", "robot-001")
        self.vendor = kw.get("vendor")
        self.model = kw.get("model")
        self.device_type = kw.get("device_type", "ROS2")
        self.created_at = kw.get("created_at", dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))
        self.updated_at = kw.get("updated_at", dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc))
        self.hardware_uuid = kw.get("hardware_uuid")
        self.hostname = kw.get("hostname")
        self.agent_ip = kw.get("agent_ip")
        self.agent_port = kw.get("agent_port")
        self.agent_status = kw.get("agent_status")
        self.camera_list_json = kw.get("camera_list_json")
        self.collect_script_compress = kw.get("collect_script_compress")
        self.collect_script_raw = kw.get("collect_script_raw")
        self.team_id = kw.get("team_id")
        self.ros2_config = kw.get("ros2_config")
        self.launch_config = kw.get("launch_config")
        self.test_results = kw.get("test_results", [])


def _install_fake_core_database_module() -> None:
    if "app.core.database" in sys.modules:
        return
    m = types.ModuleType("app.core.database")

    def _get_db():
        raise RuntimeError("sync db not available in this test environment")

    m.get_db = _get_db
    sys.modules["app.core.database"] = m

    deps = types.ModuleType("app.core.deps")

    async def _require_admin_async():
        raise RuntimeError("not used in tests")

    async def _get_current_user():
        raise RuntimeError("not used in tests")

    deps.require_admin_async = _require_admin_async
    deps.get_current_user = _get_current_user
    sys.modules["app.core.deps"] = deps

    audit = types.ModuleType("app.services.audit_service")

    def _log_audit_safe(**kwargs):
        return None

    audit.log_audit_safe = _log_audit_safe
    sys.modules["app.services.audit_service"] = audit


@pytest.mark.asyncio
async def test_create_device_requires_agent_id(monkeypatch):
    _install_fake_core_database_module()
    import app.api.routes_devices as routes
    from app.schemas.device import DeviceCreate

    db = _FakeAsyncSession()

    async def _fake_list_team_ids_accessible_by_user(db, user_id: str):
        return ["t1"]

    monkeypatch.setattr("app.core.roles.is_super_admin", lambda r: False)
    monkeypatch.setattr("app.crud.team.list_team_ids_accessible_by_user", _fake_list_team_ids_accessible_by_user)

    body = DeviceCreate(name=" robot-001 ", device_type="ROS2")
    with pytest.raises(HTTPException) as ei:
        await routes.create_new_device(
            request=Request({"type": "http", "headers": [], "client": ("127.0.0.1", 1234)}),
            device=body,
            db=db,
            current_user=_User("u1", "TEAM_ADMIN"),
        )
    assert ei.value.status_code == 400


@pytest.mark.asyncio
async def test_create_device_conflict_when_hardware_uuid_owned_by_other_team(monkeypatch):
    _install_fake_core_database_module()
    import app.api.routes_devices as routes
    from app.schemas.device import DeviceCreate

    db = _FakeAsyncSession()

    async def _fake_list_team_ids_accessible_by_user(db, user_id: str):
        return ["t1"]

    async def _fake_get_device_by_hardware_uuid(db, hw: str):
        return _DeviceObj(id=1, hardware_uuid=hw, team_id="t2")

    class _Agent:
        def __init__(self):
            self.agent_id = "hw-1"
            self.name = "a1"
            self.host = "127.0.0.1"
            self.port = 9100
            self.runtime_status = "ONLINE_IDLE"

    monkeypatch.setattr("app.core.roles.is_super_admin", lambda r: False)
    monkeypatch.setattr("app.crud.team.list_team_ids_accessible_by_user", _fake_list_team_ids_accessible_by_user)
    monkeypatch.setattr(routes, "get_device_by_hardware_uuid", _fake_get_device_by_hardware_uuid)
    monkeypatch.setattr(routes.agent_registry, "get_by_id", lambda aid: _Agent() if aid == "hw-1" else None)
    monkeypatch.setattr(routes.agent_tunnel_manager, "get_last_seen_ts", lambda aid: 1.0)

    async def _has_connection(aid: str) -> bool:
        return True

    monkeypatch.setattr(routes.agent_tunnel_manager, "has_connection", _has_connection)
    monkeypatch.setattr(routes.time, "time", lambda: 2.0)

    body = DeviceCreate(name="robot-001", device_type="ROS2", hardware_uuid="hw-1")
    with pytest.raises(HTTPException) as ei:
        await routes.create_new_device(
            request=Request({"type": "http", "headers": [], "client": ("127.0.0.1", 1234)}),
            device=body,
            db=db,
            current_user=_User("u1", "TEAM_ADMIN"),
        )
    assert ei.value.status_code == 403


@pytest.mark.asyncio
async def test_create_device_binds_team_on_first_claim(monkeypatch):
    _install_fake_core_database_module()
    import app.api.routes_devices as routes
    from app.schemas.device import DeviceCreate

    db = _FakeAsyncSession()

    async def _fake_list_team_ids_accessible_by_user(db, user_id: str):
        return ["t1"]

    existing = _DeviceObj(id=7, hardware_uuid="hw-7", team_id=None)

    async def _fake_get_device_by_hardware_uuid(db, hw: str):
        return existing

    async def _fake_get_team_by_id(db, team_id: str):
        class _T:
            name = "team1"

        return _T()

    class _Agent:
        def __init__(self):
            self.agent_id = "hw-7"
            self.name = "a7"
            self.host = "127.0.0.1"
            self.port = 9100
            self.runtime_status = "ONLINE_IDLE"

    monkeypatch.setattr("app.core.roles.is_super_admin", lambda r: False)
    monkeypatch.setattr("app.crud.team.list_team_ids_accessible_by_user", _fake_list_team_ids_accessible_by_user)
    monkeypatch.setattr("app.crud.team.get_team_by_id", _fake_get_team_by_id)
    monkeypatch.setattr(routes, "get_device_by_hardware_uuid", _fake_get_device_by_hardware_uuid)
    monkeypatch.setattr(routes.agent_registry, "get_by_id", lambda aid: _Agent() if aid == "hw-7" else None)
    monkeypatch.setattr(routes.agent_tunnel_manager, "get_last_seen_ts", lambda aid: 1.0)

    async def _has_connection(aid: str) -> bool:
        return True

    monkeypatch.setattr(routes.agent_tunnel_manager, "has_connection", _has_connection)
    monkeypatch.setattr(routes.time, "time", lambda: 2.0)

    body = DeviceCreate(name="robot-001", device_type="ROS2", hardware_uuid="hw-7")
    res = await routes.create_new_device(
        request=Request({"type": "http", "headers": [], "client": ("127.0.0.1", 1234)}),
        device=body,
        db=db,
        current_user=_User("u1", "TEAM_ADMIN"),
    )

    assert res.ok is True
    assert existing.team_id == "t1"
    assert db.commits >= 1
