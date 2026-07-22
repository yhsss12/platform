"""collect_disk_reconcile：FS_LIST mtime 与 since_ms 时区对齐。"""
from __future__ import annotations

from app.services.collect_disk_reconcile import (
    _episode_dir_name_mtime_ts,
    _fs_list_item_mtime_ts,
)


def test_fs_list_mtime_z_matches_since_ms_utc():
    """Agent 返回 UTC Z 时，mtime 应与浏览器 since_ms(UTC) 可比。"""
    since_ms = 1779090106683  # 2026-05-18T07:41:46.683Z
    slack_sec = 45.0
    since_sec = since_ms / 1000.0 - slack_sec

    # 修复后 Agent：episode 目录 mtime 为 UTC Z（与 st_mtime 一致）
    mtime_z = "2026-05-18T07:41:53Z"
    ts = _fs_list_item_mtime_ts(mtime_z)
    assert ts >= since_sec


def test_fs_list_mtime_naive_local_misaligned_with_since_ms():
    """旧 Agent 无时区本地墙钟字符串，在 UTC 容器内会被误解析为 UTC 而偏小。"""
    since_ms = 1779090106683
    since_sec = since_ms / 1000.0 - 45.0

    # 旧格式：EDT 墙钟 03:41:53 无时区 → 容器按 UTC 解析 → 比 since 早约 4h
    mtime_naive_edt_wall = "2026-05-18T03:41:53"
    ts = _fs_list_item_mtime_ts(mtime_naive_edt_wall)
    assert ts < since_sec


def test_episode_dir_name_ts_matches_since_when_mtime_missing():
    since_ms = 1779090106683
    since_sec = since_ms / 1000.0 - 45.0
    # 目录名墙钟按 UTC 解析（与修复后 Agent mtime Z 一致）
    ts = _episode_dir_name_mtime_ts("episode_27_20260518_074153")
    assert ts >= since_sec
