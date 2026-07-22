"""
初始化设备管理数据库表
"""
import asyncio
from app.db.session import engine
from app.db.base import Base
from app.models.device import Device, ROS2Config, DeviceTestResult, DeviceLaunchConfig  # noqa: F401


async def init_device_tables():
    """创建设备管理相关的数据库表"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


if __name__ == "__main__":
    asyncio.run(init_device_tables())
    print("设备管理表初始化完成")
























