import asyncio

import pytest


class _FakeResult:
    def __init__(self, rowcount: int):
        self.rowcount = rowcount


class _SharedAssetState:
    def __init__(self, sync_status: str):
        self.lock = asyncio.Lock()
        self.sync_status = sync_status


class _FakeAsyncSession:
    def __init__(self, state: _SharedAssetState, tracked_asset=None):
        self._state = state
        self._tracked_asset = tracked_asset

    async def execute(self, stmt):
        async with self._state.lock:
            status = (self._state.sync_status or "").strip().lower()
            if status in {"syncing", "synced"}:
                return _FakeResult(0)
            self._state.sync_status = "syncing"
            if self._tracked_asset is not None:
                self._tracked_asset.sync_status = "syncing"
            return _FakeResult(1)

    async def commit(self):
        if self._tracked_asset is not None:
            self._state.sync_status = (self._tracked_asset.sync_status or "").strip().lower()

    async def rollback(self):
        return None

    async def refresh(self, obj):
        obj.sync_status = self._state.sync_status


class _Asset:
    def __init__(self, asset_id: int):
        self.id = asset_id
        self.sync_status = "unsynced"
        self.sync_error = None
        self.file_path = "/tmp/a.mcap"
        self.meta = None
        self.project_id = "p1"
        self.project_name = "p1"
        self.device_id = None


@pytest.mark.asyncio
async def test_try_mark_asset_syncing_concurrent_only_one_acquires():
    from app.crud.data_asset import try_mark_asset_syncing

    shared = _SharedAssetState(sync_status="unsynced")
    s1 = _FakeAsyncSession(shared)
    s2 = _FakeAsyncSession(shared)

    r1, r2 = await asyncio.gather(
        try_mark_asset_syncing(s1, 1),
        try_mark_asset_syncing(s2, 1),
    )
    assert sorted([r1, r2]) == [False, True]


@pytest.mark.asyncio
async def test_concurrent_sync_flow_blocks_duplicate_worker():
    from app.crud.data_asset import try_mark_asset_syncing

    shared = _SharedAssetState(sync_status="unsynced")
    asset = _Asset(asset_id=1)

    async def worker(session: _FakeAsyncSession):
        acquired = await try_mark_asset_syncing(session, asset.id)
        if not acquired:
            return "blocked"
        await asyncio.sleep(0.05)
        asset.sync_status = "synced"
        await session.commit()
        return "synced"

    s1 = _FakeAsyncSession(shared, tracked_asset=asset)
    s2 = _FakeAsyncSession(shared, tracked_asset=asset)
    r1, r2 = await asyncio.gather(worker(s1), worker(s2))
    assert sorted([r1, r2]) == ["blocked", "synced"]
    assert (shared.sync_status or "").strip().lower() == "synced"
