from sqlalchemy import text


async def ensure_auth_sessions_schema(conn) -> None:
    """
    幂等补齐 auth_sessions 的 schema。
    说明：
    - SQLAlchemy create_all 不会自动 ALTER 已存在列
    - 这里把 session_id 从 VARCHAR(36) 扩到 VARCHAR(128)，避免前端异常值导致插入失败
    """
    try:
        await conn.execute(text("ALTER TABLE auth_sessions ALTER COLUMN session_id TYPE VARCHAR(128)"))
    except Exception:
        # 表不存在或已是更大类型等情况都可忽略（create_all 负责建表）
        pass

