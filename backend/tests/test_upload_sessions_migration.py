"""
迁移 007_upload_sessions：revision 链与建表脚本存在性（不连库）。
upload-complete 行为以 docs/minio-direct-upload.md 与 routes_data_assets 为准。
"""

from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
MIGRATION = BACKEND_ROOT / "alembic/versions/007_upload_sessions_table.py"


def test_upload_sessions_migration_file_defines_revision_chain():
    assert MIGRATION.is_file(), f"missing {MIGRATION}"
    text = MIGRATION.read_text(encoding="utf-8")
    assert 'revision: str = "007_upload_sessions"' in text
    assert 'down_revision: Union[str, None] = "006_audit_bigserial"' in text
    assert "upload_sessions" in text
    assert "op.create_table" in text


def test_upload_sessions_migration_idempotent_upgrade_guard():
    """upgrade 内若表已存在则跳过，兼容历史 create_all。"""
    text = MIGRATION.read_text(encoding="utf-8")
    assert "get_table_names()" in text
    assert "upload_sessions" in text


MIGRATION_P2 = BACKEND_ROOT / "alembic/versions/008_upload_sessions_phase2.py"


def test_upload_sessions_phase2_migration_exists():
    assert MIGRATION_P2.is_file()
    t = MIGRATION_P2.read_text(encoding="utf-8")
    assert 'revision: str = "008_upload_sessions_p2"' in t
    assert 'down_revision: Union[str, None] = "007_upload_sessions"' in t
    assert "upload_mode" in t
    assert "items_json" in t
