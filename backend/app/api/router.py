import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from uuid import UUID
from app.api import (
    routes_auth,
    routes_tasks,
    routes_jobs,
    routes_runs,
    routes_datasets,
    routes_label,
    routes_fs,
    routes_hdf5_datasets,
    routes_data_assets,
    routes_devices,
    routes_script,
    routes_conversion,
    routes_llm,
    routes_projects,
    routes_teams,
    routes_agents,
    routes_agent_tunnel,
    routes_webrtc,
    routes_agent_installer,
    routes_experiment,
)
from app.realtime.job_ws import manager
from app.routes import users as legacy_users_routes, audit as legacy_audit_routes

logger = logging.getLogger(__name__)

# 实时流依赖 ROS cv_bridge，可能在不含 ROS 或 NumPy 不兼容时失败，改为可选加载
try:
    from app.api import routes_stream
    _routes_stream = routes_stream
except Exception as e:
    logger.warning("stream 路由未加载（依赖 ROS cv_bridge）: %s", e)
    _routes_stream = None

api_router = APIRouter()

# 认证路由
api_router.include_router(
    routes_auth.router,
    prefix="/auth",
    tags=["auth"],
)

# 任务路由
api_router.include_router(
    routes_tasks.router,
    prefix="/tasks",
    tags=["tasks"],
)

# 作业路由
api_router.include_router(
    routes_jobs.router,
    prefix="/jobs",
    tags=["jobs"],
)

# 运行路由
api_router.include_router(
    routes_runs.router,
    prefix="/runs",
    tags=["runs"],
)

# 数据集路由
api_router.include_router(
    routes_datasets.router,
    prefix="/datasets",
    tags=["datasets"],
)

# 标注路由
api_router.include_router(
    routes_label.router,
    prefix="/label",
    tags=["label"],
)

# 大模型厂商/模型配置（标注页「模型选择与管理」弹窗）
api_router.include_router(
    routes_llm.router,
    prefix="/label/llm",
    tags=["label-llm"],
)

# 文件系统路由
api_router.include_router(
    routes_fs.router,
    prefix="/fs",
    tags=["filesystem"],
)

# HDF5 数据集路由
api_router.include_router(
    routes_hdf5_datasets.router,
    prefix="/hdf5-datasets",
    tags=["hdf5-datasets"],
)

# 数据资产路由（文件目录由 DATA_ASSETS_ROOT/平台 assets 根配置，元数据 PostgreSQL）
api_router.include_router(
    routes_data_assets.router,
    prefix="/data-assets",
    tags=["data-assets"],
)

# 项目路由（projects 表，PostgreSQL）
api_router.include_router(
    routes_projects.router,
    prefix="/projects",
    tags=["projects"],
)

# 团队路由（teams / team_admins，与 projects 同库）
api_router.include_router(
    routes_teams.router,
    prefix="/teams",
    tags=["teams"],
)

# 设备管理路由
api_router.include_router(
    routes_devices.router,
    prefix="/devices",
    tags=["devices"],
)

# Agent 管理路由（采集端 Agent 注册/心跳）
api_router.include_router(
    routes_agents.router,
    prefix="/agents",
    tags=["agents"],
)

# 用户管理（GET /users：SUPER_ADMIN 全量；团队 ADMIN 仅辖区；分页见 routes/users）
api_router.include_router(
    legacy_users_routes.router,
    prefix="",
)

# 审计日志（SUPER_ADMIN 全量；ADMIN 按团队范围见 audit_log_scope）
api_router.include_router(
    legacy_audit_routes.router,
    prefix="",
)

# 实时流路由（可选，需 ROS cv_bridge）
if _routes_stream is not None:
    api_router.include_router(
        _routes_stream.router,
        prefix="/stream",
        tags=["stream"]
    )


# 脚本执行路由
api_router.include_router(
    routes_script.router,
    prefix="/script",
    tags=["script"]
)

# WebRTC 信令路由（转发到采集端 Agent）
api_router.include_router(
    routes_webrtc.router,
    prefix="/webrtc",
    tags=["webrtc"]
)

# Agent WS tunnel（Control/Log）
api_router.include_router(
    routes_agent_tunnel.router,
    prefix="/agent",
    tags=["agent-tunnel"],
)

api_router.include_router(
    routes_agent_installer.router,
    prefix="/agent",
    tags=["agent-installer"],
)

# 数据转换路由
api_router.include_router(
    routes_conversion.router,
    prefix="/conversion",
    tags=["conversion"]
)

# 实验配置与埋点
api_router.include_router(
    routes_experiment.router,
    prefix="/experiment",
    tags=["experiment"],
)

# Workspace benchmark / 训练 / 评测 / 资源索引（数据中心、训练中心、评测中心、资源中心依赖）
from app.api import (
    routes_workspace_cable_threading,
    routes_workspace_dual_arm_cable,
    routes_workspace_evaluation,
    routes_workspace_isaaclab_franka_stack_cube,
    routes_workspace_isaacsim_franka_pick_place,
    routes_workspace_jobs,
    routes_workspace_nut_assembly,
    routes_workspace_resources,
    routes_workspace_sam3d_assets,
    routes_workspace_training,
)
from app.api.workspace import datasets as workspace_datasets_routes
from app.api.workspace import isaac_lab as workspace_isaac_lab_routes
from app.api.workspace import model_assets as workspace_model_assets_routes
from app.api.workspace import model_types as workspace_model_types_routes
from app.api.workspace import task_templates as workspace_task_templates_routes

api_router.include_router(routes_workspace_jobs.router, prefix="/workspace", tags=["workspace-jobs"])
api_router.include_router(workspace_datasets_routes.router, prefix="/workspace", tags=["workspace-datasets"])
api_router.include_router(routes_workspace_resources.router, prefix="/workspace", tags=["workspace-resources"])
api_router.include_router(workspace_model_assets_routes.router, prefix="/workspace", tags=["workspace-model-assets"])
api_router.include_router(workspace_model_types_routes.router, prefix="/workspace", tags=["workspace-model-types"])
api_router.include_router(workspace_task_templates_routes.router, prefix="/workspace", tags=["workspace-task-templates"])
api_router.include_router(workspace_isaac_lab_routes.router, prefix="/workspace", tags=["workspace-isaac-lab"])

api_router.include_router(
    routes_workspace_training.router,
    prefix="/workspace/training",
    tags=["workspace-training"],
)
api_router.include_router(
    routes_workspace_evaluation.router,
    prefix="/workspace/evaluation",
    tags=["workspace-evaluation"],
)
api_router.include_router(
    routes_workspace_cable_threading.router,
    prefix="/workspace/cable-threading",
    tags=["workspace-cable-threading"],
)
api_router.include_router(
    routes_workspace_dual_arm_cable.router,
    prefix="/workspace/dual-arm-cable",
    tags=["workspace-dual-arm-cable"],
)
api_router.include_router(
    routes_workspace_nut_assembly.router,
    prefix="/workspace/nut-assembly",
    tags=["workspace-nut-assembly"],
)
api_router.include_router(
    routes_workspace_isaacsim_franka_pick_place.router,
    prefix="/workspace/isaacsim-franka-pick-place",
    tags=["workspace-isaacsim-franka-pick-place"],
)
api_router.include_router(
    routes_workspace_isaaclab_franka_stack_cube.router,
    prefix="/workspace/isaaclab-franka-stack-cube",
    tags=["workspace-isaaclab-franka-stack-cube"],
)
api_router.include_router(
    routes_workspace_sam3d_assets.router,
    prefix="/workspace/asset-pipeline",
    tags=["workspace-asset-pipeline"],
)
