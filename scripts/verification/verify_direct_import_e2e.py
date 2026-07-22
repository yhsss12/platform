#!/usr/bin/env python3
"""
真实导入链路冒烟：单文件 / 多文件 / LeRobot 目录 / 双子目录+混合 multi_file。
依赖：后端 8000、MinIO、PostgreSQL、至少一个项目。
账号：默认 admin/admin123；可通过环境变量 EAI_E2E_USERNAME / EAI_E2E_PASSWORD（或 E2E_USERNAME / E2E_PASSWORD）覆盖。
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Any, Dict, List, Tuple

import httpx

BASE = "http://127.0.0.1:8000/api"
TIMEOUT = 120.0

E2E_USER = os.environ.get("EAI_E2E_USERNAME", os.environ.get("E2E_USERNAME", "admin"))
E2E_PASS = os.environ.get("EAI_E2E_PASSWORD", os.environ.get("E2E_PASSWORD", "admin123"))


def login(client: httpx.Client) -> str:
    r = client.post(
        f"{BASE}/auth/login",
        json={"username": E2E_USER, "password": E2E_PASS},
    )
    r.raise_for_status()
    j = r.json()
    assert j.get("ok"), j
    return j["data"]["access_token"]


def headers(token: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def first_project_id(client: httpx.Client, token: str) -> str:
    r = client.get(f"{BASE}/projects", headers=headers(token))
    r.raise_for_status()
    j = r.json()
    assert j.get("ok"), j
    data = j.get("data") or {}
    items = data.get("projects") or data.get("items") or []
    assert items, "需要至少一个项目"
    pid = items[0].get("id")
    assert pid, items[0]
    return str(pid)


def put_presigned(url: str, body: bytes, hdrs: Dict[str, str]) -> None:
    h = {k: v for k, v in hdrs.items() if v}
    r = httpx.put(url, content=body, headers=h, timeout=TIMEOUT)
    if not (200 <= r.status_code < 300):
        raise RuntimeError(f"PUT failed HTTP {r.status_code}: {r.text[:300]}")


def upload_init(client: httpx.Client, token: str, body: Dict[str, Any]) -> Dict[str, Any]:
    r = client.post(f"{BASE}/data-assets/upload-init", headers=headers(token), json=body)
    j = r.json()
    if not j.get("ok"):
        raise RuntimeError(f"upload-init: {j.get('error')}")
    return j["data"]


def upload_complete(client: httpx.Client, token: str, body: Dict[str, Any]) -> Dict[str, Any]:
    r = client.post(f"{BASE}/data-assets/upload-complete", headers=headers(token), json=body)
    j = r.json()
    if not j.get("ok"):
        raise RuntimeError(f"upload-complete: {j.get('error')}")
    return j.get("data") or {}


def get_asset(client: httpx.Client, token: str, aid: int) -> Dict[str, Any]:
    r = client.get(f"{BASE}/data-assets/by-id/{aid}", headers=headers(token))
    j = r.json()
    assert j.get("ok"), j
    return j["data"]


def delete_assets(client: httpx.Client, token: str, ids: List[int]) -> None:
    r = client.post(
        f"{BASE}/data-assets/delete-batch",
        headers=headers(token),
        json={"asset_ids": ids, "delete_file": False},
    )
    j = r.json()
    assert j.get("ok"), j


def run_single(client: httpx.Client, token: str, project_id: str) -> int:
    """单文件：仅顶层 filename+size_bytes（与旧版前端一致，后端仍支持）。"""
    name = f"e2e_{uuid.uuid4().hex[:8]}.mcap"
    body = b"MCAP_E2E_PLACEHOLDER_BYTES"
    d = upload_init(
        client,
        token,
        {
            "upload_mode": "single_file",
            "project_id": project_id,
            "filename": name,
            "size_bytes": len(body),
        },
    )
    it = d["upload_items"][0]
    put_presigned(it["upload_url"], body, it.get("headers") or {})
    out = upload_complete(
        client,
        token,
        {"upload_session_id": d["upload_session_id"], "size_bytes": len(body)},
    )
    aid = out["asset"]["id"]
    row = get_asset(client, token, aid)
    assert row.get("filename") == name
    return aid


def run_single_hdf5_items_one_only(client: httpx.Client, token: str, project_id: str) -> int:
    """单文件：仅 items[1]（无顶层 filename），走后端 len(items)==1 分支。"""
    name = f"e2e_{uuid.uuid4().hex[:8]}.hdf5"
    raw = b"HDF5_E2E_PLACEHOLDER" * 4
    d = upload_init(
        client,
        token,
        {
            "upload_mode": "single_file",
            "project_id": project_id,
            "items": [
                {
                    "client_file_id": uuid.uuid4().hex,
                    "relative_path": f"subdir/{name}",
                    "size_bytes": len(raw),
                }
            ],
        },
    )
    it = d["upload_items"][0]
    put_presigned(it["upload_url"], raw, it.get("headers") or {})
    out = upload_complete(
        client,
        token,
        {"upload_session_id": d["upload_session_id"], "size_bytes": len(raw)},
    )
    aid = out["asset"]["id"]
    row = get_asset(client, token, aid)
    assert name in (row.get("filename") or ""), row
    return aid


def run_single_hdf5_top_level_plus_items(client: httpx.Client, token: str, project_id: str) -> int:
    """单文件：顶层 filename+size_bytes 且 items 长度 1（与当前前端 initDirectUpload 对齐）。"""
    name = f"e2e_{uuid.uuid4().hex[:8]}.hdf5"
    raw = b"HDF5_TOP_ITEMS" * 5
    cid = uuid.uuid4().hex
    init_body = {
        "upload_mode": "single_file",
        "project_id": project_id,
        "filename": name,
        "size_bytes": len(raw),
        "items": [
            {
                "client_file_id": cid,
                "relative_path": name,
                "size_bytes": len(raw),
            }
        ],
    }
    d = upload_init(client, token, init_body)
    it = d["upload_items"][0]
    put_presigned(it["upload_url"], raw, it.get("headers") or {})
    out = upload_complete(
        client,
        token,
        {"upload_session_id": d["upload_session_id"], "size_bytes": len(raw)},
    )
    aid = out["asset"]["id"]
    row = get_asset(client, token, aid)
    assert row.get("filename") == name
    return aid


def run_multi(client: httpx.Client, token: str, project_id: str) -> List[int]:
    files = [
        (f"a_{uuid.uuid4().hex[:6]}.mcap", b"A" * 64),
        (f"b_{uuid.uuid4().hex[:6]}.mcap", b"B" * 128),
    ]
    items = [
        {
            "client_file_id": uuid.uuid4().hex,
            "relative_path": fn,
            "size_bytes": len(data),
        }
        for fn, data in files
    ]
    d = upload_init(
        client,
        token,
        {"upload_mode": "multi_file", "project_id": project_id, "items": items},
    )
    for i, it in enumerate(d["upload_items"]):
        put_presigned(it["upload_url"], files[i][1], it.get("headers") or {})
    out = upload_complete(client, token, {"upload_session_id": d["upload_session_id"]})
    assets = out.get("assets") or []
    assert len(assets) == 2, out
    return [int(a["id"]) for a in assets]


def run_directory_lerobot(client: httpx.Client, token: str, project_id: str, root: str) -> int:
    paths = [
        (f"{root}/data/chunk.bin", b"D" * 20),
        (f"{root}/meta/info.json", b'{"x":1}'),
        (f"{root}/videos/clip.mp4", b"V" * 15),
    ]
    items = [
        {
            "client_file_id": uuid.uuid4().hex,
            "relative_path": rel,
            "size_bytes": len(blob),
        }
        for rel, blob in paths
    ]
    d = upload_init(
        client,
        token,
        {
            "upload_mode": "directory",
            "project_id": project_id,
            "items": items,
            "root_dir_name": root,
        },
    )
    for i, it in enumerate(d["upload_items"]):
        put_presigned(it["upload_url"], paths[i][1], it.get("headers") or {})
    total = sum(len(b) for _, b in paths)
    manifest = {
        "root_dir_name": root,
        "paths": [
            {"relative_path": d["upload_items"][i]["relative_path"], "size_bytes": len(paths[i][1])}
            for i in range(len(paths))
        ],
        "total_files": len(paths),
        "total_size_bytes": total,
    }
    out = upload_complete(
        client,
        token,
        {"upload_session_id": d["upload_session_id"], "manifest": manifest},
    )
    aid = out["asset"]["id"]
    row = get_asset(client, token, aid)
    assert root in (row.get("filename") or "") or root in str(row.get("file_path") or "")
    return aid


def run_mixed_two_lr_plus_multi(
    client: httpx.Client, token: str, project_id: str
) -> Tuple[List[int], str]:
    """两个 LeRobot 子目录各一条 directory + 根上两个 mcap 一条 multi_file。"""
    ids: List[int] = []
    for sub in ("dsA", "dsB"):
        ids.append(run_directory_lerobot(client, token, project_id, sub))
    files = [
        (f"root1_{uuid.uuid4().hex[:6]}.mcap", b"R" * 32),
        (f"root2_{uuid.uuid4().hex[:6]}.hdf5", b"H" * 48),
    ]
    items = [
        {
            "client_file_id": uuid.uuid4().hex,
            "relative_path": fn,
            "size_bytes": len(data),
        }
        for fn, data in files
    ]
    d = upload_init(
        client,
        token,
        {"upload_mode": "multi_file", "project_id": project_id, "items": items},
    )
    for i, it in enumerate(d["upload_items"]):
        put_presigned(it["upload_url"], files[i][1], it.get("headers") or {})
    out = upload_complete(client, token, {"upload_session_id": d["upload_session_id"]})
    for a in out.get("assets") or []:
        ids.append(int(a["id"]))
    return ids, "2x directory + 1x multi_file(2)"


def main() -> int:
    results: List[Tuple[str, str]] = []
    created: List[int] = []
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            token = login(client)
            pid = first_project_id(client, token)

            print("project_id:", pid)

            aid = run_single(client, token, pid)
            created.append(aid)
            results.append(("1a 单文件 mcap（仅顶层 filename）", "OK"))

            aid_items = run_single_hdf5_items_one_only(client, token, pid)
            created.append(aid_items)
            results.append(("1b 单文件 hdf5（仅 items[1]）", "OK"))

            aid_both = run_single_hdf5_top_level_plus_items(client, token, pid)
            created.append(aid_both)
            results.append(("1c 单文件 hdf5（顶层+items[1]，前端默认）", "OK"))

            mids = run_multi(client, token, pid)
            created.extend(mids)
            results.append(("2 多文件 2xmcap", "OK"))

            root = f"lr_{uuid.uuid4().hex[:8]}"
            did = run_directory_lerobot(client, token, pid, root)
            created.append(did)
            results.append(("3 LeRobot 目录 data/meta/videos", "OK"))

            mix_ids, _ = run_mixed_two_lr_plus_multi(client, token, pid)
            created.extend(mix_ids)
            results.append(("4 双子目录+混合 mcap/hdf5 multi", "OK"))

            # 导出：仅测单条 mcap（与批量格式一致）
            r = client.post(
                f"{BASE}/data-assets/export",
                headers=headers(token),
                json={"asset_ids": [created[0]], "target": "local"},
            )
            if r.status_code != 200:
                raise RuntimeError(f"export zip HTTP {r.status_code}: {r.text[:400]}")
            ct = (r.headers.get("content-type") or "").lower()
            if "zip" not in ct and "octet-stream" not in ct:
                raise RuntimeError(f"export 非 zip 响应: content-type={ct!r}")
            if len(r.content) < 100:
                raise RuntimeError("export zip 过小，可能异常")
            results.append(("导出 zip（单 mcap）", "OK"))

            delete_assets(client, token, created)
            results.append(("删除本批资产", "OK"))

    except Exception as e:
        print("FAIL:", e, file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1

    for name, st in results:
        print(f"[{st}] {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
