#!/usr/bin/env python3
"""
用与业务相同的插入方式验证各表：对每个表执行一次 INSERT 后 rollback，检查类型/约束错误。
依赖：先运行 alter_bigint_columns.py 将 file_size_bytes 等改为 BIGINT。
"""
import asyncio
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


async def run():
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.ext.asyncio import async_sessionmaker

    url = os.getenv("DATABASE_URL", "")
    if not url or not str(url).strip().lower().startswith("postgresql"):
        print("❌ 请设置 DATABASE_URL 为 PostgreSQL")
        return 1
    engine = create_async_engine(url)
    async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    errors = []

    # 1) data_assets：大 file_size_bytes 必须用 BIGINT
    async with async_session() as session:
        try:
            await session.execute(text("""
                INSERT INTO data_assets (dataset_id, code, filename, format, source, file_path, file_size_bytes, parse_status)
                VALUES ('DS_VERIFY', 'v', 'verify.h5', 'hdf5', 'import', '/tmp/verify.h5', 11606953456, '未解析')
            """))
            await session.rollback()
            print("✅ data_assets INSERT (file_size_bytes > 2G) -> rollback")
        except Exception as e:
            await session.rollback()
            errors.append(("data_assets", str(e)))
            print(f"❌ data_assets: {e}")

    # 2) hdf5_datasets
    async with async_session() as session:
        try:
            await session.execute(text("""
                INSERT INTO hdf5_datasets (name, file_size_bytes, format, storage_type, storage_uri, qc_status, label_status, assign_status)
                VALUES ('verify', 11606953456, 'HDF5', 'local', '/tmp/verify_hdf5_' || gen_random_uuid(), 'pending', 'unlabeled', 'unassigned')
            """))
            await session.rollback()
            print("✅ hdf5_datasets INSERT (file_size_bytes > 2G) -> rollback")
        except Exception as e:
            await session.rollback()
            errors.append(("hdf5_datasets", str(e)))
            print(f"❌ hdf5_datasets: {e}")

    # 3) users
    async with async_session() as session:
        try:
            await session.execute(text("""
                INSERT INTO users (id, username, password_hash, role, is_active)
                VALUES ('00000000-0000-0000-0000-000000000001', 'verify_user', 'hash', 'user', true)
            """))
            await session.rollback()
            print("✅ users INSERT -> rollback")
        except Exception as e:
            await session.rollback()
            errors.append(("users", str(e)))
            print(f"❌ users: {e}")

    # 4) devices
    async with async_session() as session:
        try:
            await session.execute(text("""
                INSERT INTO devices (name, device_type) VALUES ('verify_device', 'robot')
            """))
            await session.rollback()
            print("✅ devices INSERT -> rollback")
        except Exception as e:
            await session.rollback()
            errors.append(("devices", str(e)))
            print(f"❌ devices: {e}")

    # 5) projects
    async with async_session() as session:
        try:
            await session.execute(text("""
                INSERT INTO projects (id, name, status) VALUES ('p_verify', 'Verify Project', 'active')
            """))
            await session.rollback()
            print("✅ projects INSERT -> rollback")
        except Exception as e:
            await session.rollback()
            errors.append(("projects", str(e)))
            print(f"❌ projects: {e}")

    # 6) label_tasks
    async with async_session() as session:
        try:
            await session.execute(text("""
                INSERT INTO label_tasks (task_id, name, dataset_path, completed, verified)
                VALUES ('verify_task', 'Verify', '/tmp', false, false)
            """))
            await session.rollback()
            print("✅ label_tasks INSERT -> rollback")
        except Exception as e:
            await session.rollback()
            errors.append(("label_tasks", str(e)))
            print(f"❌ label_tasks: {e}")

    # 7) tasks (main)
    async with async_session() as session:
        try:
            await session.execute(text("""
                INSERT INTO tasks (id, name, status, created_at, updated_at)
                VALUES (gen_random_uuid(), 'Verify Task', 'PENDING', '2020-01-01T00:00:00', '2020-01-01T00:00:00')
            """))
            await session.rollback()
            print("✅ tasks INSERT -> rollback")
        except Exception as e:
            await session.rollback()
            errors.append(("tasks", str(e)))
            print(f"❌ tasks: {e}")

    # 8) jobs (mcap_size_bytes 可能大)
    async with async_session() as session:
        try:
            r = await session.execute(text("SELECT id FROM tasks LIMIT 1"))
            row = r.fetchone()
            if row:
                tid = row[0]
                await session.execute(text("""
                    INSERT INTO jobs (id, task_id, job_number, status, progress, created_at, updated_at, mcap_size_bytes)
                    VALUES (gen_random_uuid(), :tid, 0, 'PENDING', 0, '2020-01-01T00:00:00', '2020-01-01T00:00:00', 11606953456)
                """), {"tid": tid})
            await session.rollback()
            print("✅ jobs INSERT (mcap_size_bytes > 2G) -> rollback")
        except Exception as e:
            await session.rollback()
            errors.append(("jobs", str(e)))
            print(f"❌ jobs: {e}")

    # 9) providers
    async with async_session() as session:
        try:
            await session.execute(text("""
                INSERT INTO providers (name, code, type) VALUES ('Verify', 'verify', 'openai_compatible')
            """))
            await session.rollback()
            print("✅ providers INSERT -> rollback")
        except Exception as e:
            await session.rollback()
            errors.append(("providers", str(e)))
            print(f"❌ providers: {e}")

    if errors:
        print(f"\n❌ 共 {len(errors)} 个表插入验证失败")
        return 1
    print("\n✅ 所有表插入验证通过")
    return 0


def main():
    return asyncio.run(run())


if __name__ == "__main__":
    sys.exit(main())
