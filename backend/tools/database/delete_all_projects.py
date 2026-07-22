#!/usr/bin/env python3
"""
删除数据库中的全部项目（级联删任务/成员/数据资产记录，并尝试删除对应 MinIO bucket）。
用法（在 backend 目录下）：
  python tools/database/delete_all_projects.py
  python tools/database/delete_all_projects.py --yes
需设置 DATABASE_URL；MinIO 不可用时仍会删库记录。
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

try:
    from dotenv import load_dotenv

    for p in (BACKEND_DIR / ".env", BACKEND_DIR.parent / ".env"):
        if p.exists():
            load_dotenv(p)
            break
except Exception:
    pass


async def _run(*, yes: bool) -> int:
    url = os.getenv("DATABASE_URL", "")
    if not url or "postgresql" not in str(url).lower():
        print("❌ 请设置 DATABASE_URL 为 PostgreSQL 连接串（asyncpg）。")
        return 1

    from app.db.data_assets_session import DataAssetsSessionLocal
    from app.crud.project import list_projects, delete_project_with_cascade
    from app.services.minio_service import remove_project_bucket
    from app.services.minio_service import MinioConfigError, MinioBucketError

    async with DataAssetsSessionLocal() as db:
        items, _total = await list_projects(db)
        if not items:
            print("当前没有项目。")
            return 0
        print(f"即将删除 {len(items)} 个项目：")
        for p in items:
            print(f"  - {p.id}  {p.name!r}")

    if not yes:
        s = input("输入 YES 确认删除全部项目: ").strip()
        if s != "YES":
            print("已取消。")
            return 2

    deleted = 0
    errors: list[str] = []
    async with DataAssetsSessionLocal() as db:
        items, total = await list_projects(db)
        for p in items:
            pid = str(p.id).strip()
            pname = (p.name or "").strip() or pid
            try:
                remove_project_bucket(pname, force=True)
            except MinioConfigError as e:
                errors.append(f"MinIO 配置跳过 bucket「{pname}」: {e}")
            except MinioBucketError as e:
                errors.append(f"MinIO 删除 bucket「{pname}」: {e}")
            ok = await delete_project_with_cascade(db, pid)
            if ok:
                deleted += 1
                print(f"✓ 已删除项目 {pid} ({pname})")
            else:
                errors.append(f"数据库删除失败: {pid}")

    for msg in errors:
        print(f"⚠ {msg}")
    print(f"完成：成功删除 {deleted} / {total} 个项目。")
    return 0 if deleted == total else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="删除全部项目（危险操作）")
    parser.add_argument("--yes", action="store_true", help="不询问直接删除")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_run(yes=args.yes)))


if __name__ == "__main__":
    main()
