"""
单文件 upload-init 入参门槛（与 routes_data_assets.data_assets_upload_init 中 single_file 分支一致）。
不连数据库，用于证明：filename+size_bytes 或 items[1] 均可通过门槛，且不会落入「单文件请提供…」分支。
"""

from __future__ import annotations

from pathlib import Path


def _single_file_gate(body: dict) -> tuple[bool, str]:
    """返回 (通过门槛, 错误文案)。"""
    mode = (body.get("upload_mode") or "single_file").strip().lower()
    if mode != "single_file":
        return False, "not single_file"
    items_one = list(body.get("items") or [])
    if len(items_one) == 1:
        o = items_one[0]
        fn_src = Path((o.get("relative_path") or "").replace("\\", "/")).name
        try:
            sz = int(o.get("size_bytes"))
        except (TypeError, ValueError):
            return False, "size_bytes 无效"
        ct = o.get("content_type")
    else:
        fn_src = body.get("filename") or ""
        try:
            sz = int(body["size_bytes"]) if body.get("size_bytes") is not None else 0
        except (TypeError, ValueError):
            return False, "size_bytes 无效"
        ct = body.get("content_type")
    if not fn_src or sz <= 0:
        return False, "单文件请提供 filename+size_bytes 或 items[1]"
    _ = ct  # 与路由一致：此处不校验 content_type
    return True, ""


def test_gate_top_level_filename_only():
    ok, err = _single_file_gate(
        {
            "upload_mode": "single_file",
            "project_id": "p1",
            "filename": "episode_0.hdf5",
            "size_bytes": 1024,
        }
    )
    assert ok and not err


def test_gate_items_one_relative_path_basename():
    ok, err = _single_file_gate(
        {
            "upload_mode": "single_file",
            "project_id": "p1",
            "items": [
                {
                    "client_file_id": "cid-1",
                    "relative_path": "data/episode_1.hdf5",
                    "size_bytes": 2048,
                }
            ],
        }
    )
    assert ok and not err


def test_gate_items_one_matches_top_level_redundant():
    ok, err = _single_file_gate(
        {
            "upload_mode": "single_file",
            "project_id": "p1",
            "filename": "x.hdf5",
            "size_bytes": 100,
            "items": [
                {
                    "client_file_id": "cid",
                    "relative_path": "x.hdf5",
                    "size_bytes": 100,
                }
            ],
        }
    )
    assert ok and not err


def test_gate_rejects_empty_filename_and_zero_size():
    ok, err = _single_file_gate(
        {"upload_mode": "single_file", "project_id": "p1", "filename": "", "size_bytes": 0}
    )
    assert not ok
    assert "单文件请提供" in err


def test_gate_rejects_items_one_zero_size():
    ok, err = _single_file_gate(
        {
            "upload_mode": "single_file",
            "project_id": "p1",
            "items": [{"client_file_id": "c", "relative_path": "a.hdf5", "size_bytes": 0}],
        }
    )
    assert not ok
