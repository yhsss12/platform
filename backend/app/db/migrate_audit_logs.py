"""
主库 audit_logs：启动时补齐缺列。

SQLAlchemy create_all 不会修改已存在表结构，若历史库缺 role 等列会导致列表查询报错。
逻辑与 alembic/versions/005_audit_logs_align_columns.py 对齐（PostgreSQL）。
"""
from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection


def ensure_audit_logs_columns(connection: Connection) -> None:
    if connection.dialect.name != "postgresql":
        return
    insp = inspect(connection)
    if not insp.has_table("audit_logs"):
        return

    existing = {c["name"] for c in insp.get_columns("audit_logs")}

    def exec_sql(sql: str) -> None:
        connection.execute(text(sql))

    if "role" not in existing:
        exec_sql("ALTER TABLE audit_logs ADD COLUMN role VARCHAR(50)")
        existing.add("role")
    if "project_name" not in existing:
        exec_sql("ALTER TABLE audit_logs ADD COLUMN project_name VARCHAR(200)")
        existing.add("project_name")
    if "project_id" not in existing:
        exec_sql("ALTER TABLE audit_logs ADD COLUMN project_id VARCHAR(64)")
        existing.add("project_id")
    if "resource_type" not in existing:
        exec_sql("ALTER TABLE audit_logs ADD COLUMN resource_type VARCHAR(100)")
        existing.add("resource_type")
    if "resource_id" not in existing:
        exec_sql("ALTER TABLE audit_logs ADD COLUMN resource_id VARCHAR(100)")
        existing.add("resource_id")
    if "resource_name" not in existing:
        exec_sql("ALTER TABLE audit_logs ADD COLUMN resource_name VARCHAR(255)")
        existing.add("resource_name")
    if "error_message" not in existing:
        exec_sql("ALTER TABLE audit_logs ADD COLUMN error_message TEXT")
        existing.add("error_message")
    if "detail_json" not in existing:
        exec_sql("ALTER TABLE audit_logs ADD COLUMN detail_json JSONB")
        existing.add("detail_json")
    if "user_agent" not in existing:
        exec_sql("ALTER TABLE audit_logs ADD COLUMN user_agent TEXT")
        existing.add("user_agent")
    if "ip" not in existing:
        exec_sql("ALTER TABLE audit_logs ADD COLUMN ip VARCHAR(64)")
        existing.add("ip")
    if "result" not in existing:
        exec_sql(
            "ALTER TABLE audit_logs ADD COLUMN result VARCHAR(20) NOT NULL DEFAULT 'SUCCESS'"
        )
        existing.add("result")
    if "action_type" not in existing:
        exec_sql(
            "ALTER TABLE audit_logs ADD COLUMN action_type VARCHAR(100) "
            "NOT NULL DEFAULT 'UNKNOWN'"
        )
        if "action" in existing:
            exec_sql("UPDATE audit_logs SET action_type = action")
        existing.add("action_type")
    if "action_label" not in existing:
        exec_sql(
            "ALTER TABLE audit_logs ADD COLUMN action_label VARCHAR(200) "
            "NOT NULL DEFAULT '-'"
        )
        if "detail" in existing and "action" in existing:
            exec_sql(
                "UPDATE audit_logs SET action_label = LEFT("
                "COALESCE(NULLIF(TRIM(COALESCE(detail, '')), ''), "
                "NULLIF(TRIM(COALESCE(action, '')), ''), '-'), 200)"
            )
        elif "action" in existing:
            exec_sql(
                "UPDATE audit_logs SET action_label = LEFT("
                "COALESCE(NULLIF(TRIM(COALESCE(action, '')), ''), '-'), 200)"
            )
        elif "detail" in existing:
            exec_sql(
                "UPDATE audit_logs SET action_label = LEFT("
                "COALESCE(NULLIF(TRIM(COALESCE(detail, '')), ''), '-'), 200)"
            )
        existing.add("action_label")

    if "team_id" not in existing:
        exec_sql("ALTER TABLE audit_logs ADD COLUMN team_id VARCHAR(128)")
        existing.add("team_id")
    if "team_id" in existing:
        exec_sql(
            "CREATE INDEX IF NOT EXISTS ix_audit_logs_team_id ON audit_logs (team_id)"
        )
    if "project_id" in existing:
        exec_sql(
            "CREATE INDEX IF NOT EXISTS ix_audit_logs_project_id ON audit_logs (project_id)"
        )
    if "action_type" in existing:
        exec_sql(
            "CREATE INDEX IF NOT EXISTS ix_audit_logs_action_type ON audit_logs (action_type)"
        )
    if "result" in existing:
        exec_sql("CREATE INDEX IF NOT EXISTS ix_audit_logs_result ON audit_logs (result)")
