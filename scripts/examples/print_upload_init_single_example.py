#!/usr/bin/env python3
"""打印单文件 upload-init 示例 JSON，便于 curl 与文档对照。"""
from __future__ import annotations

import json
import uuid

project_id = "<你的项目 UUID>"
filename = "episode_0.hdf5"
size_bytes = 123456
cid = uuid.uuid4().hex

browser_like = {
    "upload_mode": "single_file",
    "project_id": project_id,
    "filename": filename,
    "size_bytes": size_bytes,
    "items": [
        {
            "client_file_id": cid,
            "relative_path": filename,
            "size_bytes": size_bytes,
            "content_type": None,
        }
    ],
}
print("=== 当前前端 initDirectUpload（单文件）典型请求体 ===")
print(json.dumps(browser_like, indent=2, ensure_ascii=False))
