#!/usr/bin/env python3
"""
初始化用户脚本
幂等确保平台超级管理员存在（与启动时 get_or_create_admin_user 一致：账号 Pibot0001，展示名默认 Pibot）
"""
import asyncio
import sys
from pathlib import Path

# 添加项目路径
backend_dir = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(backend_dir))

from app.db.session import AsyncSessionLocal
from app.crud.user import get_or_create_admin_user


async def main():
    """确保平台超级管理员账号"""
    async with AsyncSessionLocal() as db:
        try:
            admin = await get_or_create_admin_user(db)
            print(f"✅ 用户初始化成功！")
            print(f"   登录账号: {admin.account_id}")
            print(f"   展示名: {admin.username}")
            print(f"   角色: {admin.role}")
            print(f"   默认密码: jinlian1234（请登录后及时修改；数据库仅存哈希）")
        except Exception as e:
            print(f"❌ 用户初始化失败: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())















