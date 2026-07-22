#!/usr/bin/env python3
"""将 file_size_bytes / mcap_size_bytes 等列从 INTEGER 改为 BIGINT，避免大文件溢出。"""
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))
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
        print("DATABASE_URL 未设置或非 PostgreSQL，跳过 ALTER。")
        return 0
    sync_url = url.replace("postgresql+asyncpg", "postgresql", 1)
    engine = create_engine(sync_url)
    alters = [
        ("data_assets", "file_size_bytes"),
        ("hdf5_datasets", "file_size_bytes"),
        ("collection_jobs", "mcap_size_bytes"),
        ("jobs", "mcap_size_bytes"),
    ]
    with engine.connect() as conn:
        for table, col in alters:
            try:
                conn.execute(text(f'ALTER TABLE "{table}" ALTER COLUMN "{col}" TYPE BIGINT USING "{col}"::BIGINT'))
                conn.commit()
                print(f"  {table}.{col} -> BIGINT")
            except Exception as e:
                if "type of column" in str(e).lower() or "already" in str(e).lower():
                    print(f"  {table}.{col} 已是 BIGINT 或不存在，跳过")
                else:
                    print(f"  {table}.{col}: {e}")
    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
