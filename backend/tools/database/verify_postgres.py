#!/usr/bin/env python3
"""
验证 PostgreSQL 连接及表与数据是否就绪。
用法：在 backend 目录下执行
  python tools/database/verify_postgres.py
或设置环境变量后执行：
  DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/eai_ide python tools/database/verify_postgres.py
"""
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# 加载 .env
try:
    from dotenv import load_dotenv
    for p in (BACKEND_DIR / ".env", BACKEND_DIR.parent / ".env"):
        if p.exists():
            load_dotenv(p)
            break
except Exception:
    pass


def main():
    from sqlalchemy import create_engine, text
    url = os.getenv("DATABASE_URL", "")
    if not url or not str(url).strip().lower().startswith("postgresql"):
        print("❌ 请设置 DATABASE_URL 为 PostgreSQL 连接串，例如：")
        print("   export DATABASE_URL=postgresql+asyncpg://admin:密码@172.18.0.93:5432/eai_ide")
        return 1
    sync_url = url.replace("postgresql+asyncpg", "postgresql", 1)
    engine = create_engine(sync_url)
    try:
        with engine.connect() as conn:
            r = conn.execute(text("SELECT version()"))
            version = r.scalar()
            print(f"✅ 连接成功: {version[:60]}...")
            r = conn.execute(text("""
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """))
            tables = [row[0] for row in r]
            print(f"✅ 表数量: {len(tables)}")
            for t in ("users", "devices", "data_assets", "label_tasks", "hdf5_datasets"):
                if t in tables:
                    c = conn.execute(text(f'SELECT COUNT(*) FROM "{t}"')).scalar()
                    print(f"   - {t}: {c} 行")
                else:
                    print(f"   - {t}: (表不存在)")
        return 0
    except Exception as e:
        print(f"❌ 连接或查询失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
