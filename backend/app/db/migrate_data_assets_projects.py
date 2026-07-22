"""
数据资产库 projects：启动时补齐缺列。

SQLAlchemy create_all 不会修改已存在表结构；历史库若缺 team_id 会导致项目列表查询失败。
逻辑与 alembic/versions/009_teams_team_admins_projects_team_id.py 中 projects.team_id 部分对齐。
"""
from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection


def ensure_projects_team_id_column(connection: Connection) -> None:
    if connection.dialect.name != "postgresql":
        return
    insp = inspect(connection)
    if not insp.has_table("projects"):
        return
    cols = {c["name"] for c in insp.get_columns("projects")}
    if "team_id" not in cols:
        connection.execute(
            text("ALTER TABLE projects ADD COLUMN team_id VARCHAR(128)")
        )
    connection.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_projects_team_id ON projects (team_id)"
        )
    )
