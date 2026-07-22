"""
数据资产 meta.storage 合并：供导入、登记、同步、转换等共用。

导出链路优先读 meta.storage.minio_path；backend_local_path 仅为服务端暂存/溯源，非长期主存储 URI。
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional


def merge_storage_meta(meta_json: Optional[str], local_path: str, minio_path: str) -> Optional[str]:
    """浅拷贝顶层 meta，合并 storage 子树，仅覆盖 backend_local_path / minio_path。"""
    base: Dict[str, Any] = {}
    if meta_json:
        try:
            parsed = json.loads(meta_json)
            if isinstance(parsed, dict):
                base = dict(parsed)
        except Exception:
            base = {}
    prev_storage = base.get("storage")
    if isinstance(prev_storage, dict):
        storage_merged: Dict[str, Any] = dict(prev_storage)
    else:
        storage_merged = {}
    storage_merged["backend_local_path"] = local_path
    storage_merged["minio_path"] = minio_path
    base["storage"] = storage_merged
    try:
        return json.dumps(base, ensure_ascii=False)
    except Exception:
        return meta_json
