"""
单元测试：导入链路中的 meta 合并（不依赖 PostgreSQL / MinIO）。

边界说明：
- 仅覆盖 merge_storage_meta（app.services.storage_meta_merge）的合并行为。
- POST /api/data-assets/import 全链路需集成环境或手工按 tests/IMPORT_VERIFICATION.md 验证。
"""

import json

import pytest

from app.services.storage_meta_merge import merge_storage_meta


def test_merge_storage_preserves_top_level_and_storage_extras():
    raw = json.dumps({"episode": 1, "storage": {"custom": "keep", "note": "x"}})
    out = merge_storage_meta(raw, "/data/assets/p/a.mcap", "minio://bucket/proj/p/a.mcap")
    d = json.loads(out)
    assert d["episode"] == 1
    assert d["storage"]["custom"] == "keep"
    assert d["storage"]["note"] == "x"
    assert d["storage"]["backend_local_path"] == "/data/assets/p/a.mcap"
    assert d["storage"]["minio_path"] == "minio://bucket/proj/p/a.mcap"


def test_merge_storage_from_empty_meta():
    out = merge_storage_meta(None, "/local/x", "minio://b/k")
    d = json.loads(out)
    assert d["storage"]["backend_local_path"] == "/local/x"
    assert d["storage"]["minio_path"] == "minio://b/k"


def test_merge_storage_replaces_non_dict_storage():
    raw = json.dumps({"storage": "broken"})
    out = merge_storage_meta(raw, "/l", "minio://b/k")
    d = json.loads(out)
    assert d["storage"]["minio_path"] == "minio://b/k"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
