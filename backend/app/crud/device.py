"""
设备管理 CRUD 操作
"""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload
from typing import List, Optional
import json
from app.models.device import Device, ROS2Config, DeviceTestResult, DeviceLaunchConfig
from app.schemas.device import DeviceCreate, DeviceUpdate, ROS2ConfigCreate, ROS2ConfigUpdate, DeviceTestResultBase, DeviceLaunchConfigCreate, DeviceLaunchConfigUpdate


async def get_devices(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 100
) -> tuple[List[Device], int]:
    """获取设备列表"""
    # 获取总数
    count_result = await db.execute(select(func.count()).select_from(Device))
    total = count_result.scalar() or 0

    # 获取设备列表（包含关联数据）
    query = select(Device).options(
        selectinload(Device.ros2_config),
        selectinload(Device.launch_config),
        selectinload(Device.test_results)
    ).order_by(Device.updated_at.desc()).offset(skip).limit(limit)

    result = await db.execute(query)
    devices = list(result.scalars().all())

    return devices, total


async def get_device_by_id(db: AsyncSession, device_id: int) -> Optional[Device]:
    """根据 ID 获取设备"""
    query = select(Device).options(
        selectinload(Device.ros2_config),
        selectinload(Device.launch_config),
        selectinload(Device.test_results)
    ).where(Device.id == device_id)

    result = await db.execute(query)
    return result.scalar_one_or_none()


async def create_device(db: AsyncSession, device: DeviceCreate) -> Device:
    """创建设备"""
    from sqlalchemy.orm import selectinload
    
    # 创建设备记录
    db_device = Device(
        name=device.name,
        vendor=device.vendor,
        model=device.model,
        device_type=device.device_type,
        hardware_uuid=device.hardware_uuid,
        team_id=getattr(device, "team_id", None),
        hostname=device.hostname,
        agent_ip=device.agent_ip,
        agent_port=device.agent_port,
        agent_status=device.agent_status,
        camera_list_json=json.dumps(device.camera_list) if device.camera_list is not None else None,
        collect_script_compress=(device.collect_script_compress or None),
        collect_script_raw=(device.collect_script_raw or None),
    )
    db.add(db_device)
    await db.flush()  # 获取 device.id

    # 如果提供了 ROS2 配置，创建配置记录
    if device.ros2_config:
        db_ros2_config = ROS2Config(
            device_id=db_device.id,
            mode=device.ros2_config.mode,
            local_bind_ip=device.ros2_config.local_bind_ip,
            domain_id=device.ros2_config.domain_id,
            discovery_protocol=device.ros2_config.discovery_protocol,
            initial_announcements_count=device.ros2_config.initial_announcements_count,
            initial_announcements_period_sec=device.ros2_config.initial_announcements_period_sec,
            peer_ips_json=json.dumps(device.ros2_config.peer_ips) if device.ros2_config.peer_ips else None
        )
        db.add(db_ros2_config)

    # 如果提供了启动配置，创建配置记录
    if device.launch_config:
        db_launch_config = DeviceLaunchConfig(
            device_id=db_device.id,
            script_path=device.launch_config.script_path,
            script_args=device.launch_config.script_args,
            stop_script_path=getattr(device.launch_config, "stop_script_path", None),
            stop_script_args=getattr(device.launch_config, "stop_script_args", None),
            env_vars_json=json.dumps(device.launch_config.env_vars) if device.launch_config.env_vars else None
        )
        db.add(db_launch_config)

    await db.commit()
    
    # 重新查询设备并加载关联的ros2_config，避免懒加载问题
    from sqlalchemy import select
    result = await db.execute(
        select(Device)
        .options(selectinload(Device.ros2_config), selectinload(Device.launch_config))
        .where(Device.id == db_device.id)
    )
    db_device = result.scalar_one()
    
    return db_device


async def update_device(
    db: AsyncSession,
    device_id: int,
    device_update: DeviceUpdate
) -> Optional[Device]:
    """更新设备"""
    db_device = await get_device_by_id(db, device_id)
    if db_device is None:
        return None

    # 更新设备基本信息
    update_data = device_update.model_dump(exclude_unset=True, exclude={"ros2_config", "launch_config"})
    camera_list_provided = "camera_list" in update_data
    if camera_list_provided:
        camera_list_value = update_data.pop("camera_list", None)
    for field, value in update_data.items():
        setattr(db_device, field, value)
    if camera_list_provided:
        db_device.camera_list_json = json.dumps(camera_list_value) if camera_list_value is not None else None

    # 更新 ROS2 配置
    if device_update.ros2_config is not None:
        if db_device.ros2_config:
            # 更新现有配置
            config_update = device_update.ros2_config.model_dump(exclude_unset=True)
            if "peer_ips" in config_update and config_update["peer_ips"] is not None:
                config_update["peer_ips_json"] = json.dumps(config_update.pop("peer_ips"))
            for field, value in config_update.items():
                if field != "peer_ips":
                    setattr(db_device.ros2_config, field, value)
        else:
            # 创建新配置
            ros2_config_data = device_update.ros2_config.model_dump()
            db_ros2_config = ROS2Config(
                device_id=db_device.id,
                mode=ros2_config_data.get("mode", "fastdds_tailscale_peer"),
                local_bind_ip=ros2_config_data.get("local_bind_ip"),
                domain_id=ros2_config_data.get("domain_id", 0),
                discovery_protocol=ros2_config_data.get("discovery_protocol", "SIMPLE"),
                initial_announcements_count=ros2_config_data.get("initial_announcements_count", 5),
                initial_announcements_period_sec=ros2_config_data.get("initial_announcements_period_sec", 1),
                peer_ips_json=json.dumps(ros2_config_data.get("peer_ips", [])) if ros2_config_data.get("peer_ips") else None
            )
            db.add(db_ros2_config)

    # 更新启动配置
    if device_update.launch_config is not None:
        if db_device.launch_config:
            # 更新现有配置
            launch_update = device_update.launch_config.model_dump(exclude_unset=True)
            if "env_vars" in launch_update and launch_update["env_vars"] is not None:
                launch_update["env_vars_json"] = json.dumps(launch_update.pop("env_vars"))
            
            for field, value in launch_update.items():
                setattr(db_device.launch_config, field, value)
        else:
            # 创建新配置
            launch_config_data = device_update.launch_config.model_dump()
            db_launch_config = DeviceLaunchConfig(
                device_id=db_device.id,
                script_path=launch_config_data.get("script_path"),
                script_args=launch_config_data.get("script_args"),
                stop_script_path=launch_config_data.get("stop_script_path"),
                stop_script_args=launch_config_data.get("stop_script_args"),
                env_vars_json=json.dumps(launch_config_data.get("env_vars", {})) if launch_config_data.get("env_vars") else None
            )
            db.add(db_launch_config)

    await db.commit()
    await db.refresh(db_device)
    return db_device


async def delete_device(db: AsyncSession, device_id: int) -> bool:
    """删除设备"""
    db_device = await get_device_by_id(db, device_id)
    if db_device is None:
        return False

    await db.delete(db_device)
    await db.commit()
    return True


async def create_test_result(
    db: AsyncSession,
    device_id: int,
    test_result: DeviceTestResultBase
) -> DeviceTestResult:
    """创建设备测试结果"""
    db_test_result = DeviceTestResult(
        device_id=device_id,
        status=test_result.status,
        node_count=test_result.node_count,
        nodes_sample_json=json.dumps(test_result.nodes_sample) if test_result.nodes_sample else None,
        topic_count=test_result.topic_count,
        topics_sample_json=json.dumps(test_result.topics_sample) if test_result.topics_sample else None,
        error_type=test_result.error_type,
        error_message=test_result.error_message
    )
    db.add(db_test_result)
    await db.commit()
    await db.refresh(db_test_result)
    return db_test_result


async def get_latest_test_result(
    db: AsyncSession,
    device_id: int
) -> Optional[DeviceTestResult]:
    """获取设备最新的测试结果"""
    query = select(DeviceTestResult).where(
        DeviceTestResult.device_id == device_id
    ).order_by(DeviceTestResult.tested_at.desc()).limit(1)

    result = await db.execute(query)
    return result.scalar_one_or_none()


async def get_device_by_hardware_uuid(
    db: AsyncSession,
    hardware_uuid: str,
) -> Optional[Device]:
    """根据采集端硬件 UUID 获取设备（用于“设备主动添加”去重/更新）"""
    if not hardware_uuid:
        return None

    query = select(Device).options(
        selectinload(Device.ros2_config),
        selectinload(Device.launch_config),
        selectinload(Device.test_results),
    ).where(Device.hardware_uuid == hardware_uuid)

    result = await db.execute(query)
    return result.scalar_one_or_none()

