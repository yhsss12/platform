"""
数据资产表（PostgreSQL 统一库）
"""
from sqlalchemy import Column, Integer, BigInteger, String, DateTime, Text, Index, Boolean, Float, ForeignKey, JSON, text
from sqlalchemy.sql import func

# 使用独立 Base，与主库 models 分离
from sqlalchemy.orm import declarative_base
Base = declarative_base()


class DataAsset(Base):
    """数据资产表"""
    __tablename__ = "data_assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    dataset_id = Column(String(32), unique=True, nullable=True, index=True, comment="数据专属唯一标识 DS000001，标注/转换/导出均以此为存在依据")
    code = Column(String(32), nullable=False, comment="展示编号 0001/0002...")
    filename = Column(String(512), nullable=False, comment="文件名")
    format = Column(String(32), nullable=False, comment="hdf5 | mcap | lerobot")
    # 来源：import（导入）| collect（采集）| label（标注）| convert（转换）
    source = Column(String(32), nullable=False, default="import", comment="来源：import/collect/label/convert")
    project_id = Column(String(128), nullable=True, comment="所属项目 id")
    project_name = Column(String(256), nullable=True, comment="所属项目名称（冗余）")
    file_path = Column(
        String(1024),
        nullable=False,
        comment="本地绝对路径或采集同步后的 minio:// URI；标准导入在 meta.storage 中另有 minio_path 供导出",
    )
    file_size_bytes = Column(BigInteger, nullable=False, default=0, comment="文件大小（字节），大文件可超 int32")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, comment="导入时间（UTC）")
    # 最近一次真实业务动作时间（导入 / 采集 / 标注任务 / 转换 / 导出等）
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False, comment="最近一次业务更新时间（UTC）")
    meta = Column(Text, nullable=True, comment="JSON 格式元数据")
    parse_status = Column(String(32), nullable=False, default="未解析", comment="未解析|解析中|成功|失败")
    error_msg = Column(Text, nullable=True, comment="解析失败原因")
    # 同步状态：unsynced（未同步）| syncing（同步中）| synced（已同步）| failed（同步失败）
    sync_status = Column(String(32), nullable=False, default="synced", comment="同步状态：unsynced/syncing/synced/failed")
    sync_error = Column(Text, nullable=True, comment="同步失败原因")
    instruction_text = Column(Text, nullable=True, comment="标注信息，与 instruction.json 同步")
    # 以任务划分展示：来源为采集/标注/转换时对应的任务名称
    label_task_name = Column(String(256), nullable=True, comment="关联的标注任务名称")
    collect_task_name = Column(String(256), nullable=True, comment="关联的采集任务名称")
    conversion_task_name = Column(String(256), nullable=True, comment="关联的转换任务名称")
    # 采集来源：平台 devices 表主键（与前端设备ID一致），用于区分不同采集端数据
    device_id = Column(String(64), nullable=True, comment="采集设备 platform id（devices.id）")
    operator_name = Column(String(256), nullable=True, comment="资产真实操作者账号名（历史字段）")
    minio_path = Column(String(1024), nullable=True, comment="MinIO 对象 URI（minio://bucket/key）")

    __table_args__ = (
        Index("idx_data_assets_dataset_id", "dataset_id"),
        Index("idx_data_assets_project", "project_id"),
        Index("idx_data_assets_format", "format"),
        Index("idx_data_assets_created", "created_at"),
        Index("idx_data_assets_updated", "updated_at"),
        Index("idx_data_assets_file_path", "file_path"),
        Index("idx_data_assets_device_id", "device_id"),
    )


class ConversionBatchJob(Base):
    """转换批量任务（父任务），一条父任务对应多条 conversion_jobs 子任务。"""

    __tablename__ = "conversion_batch_jobs"

    batch_id = Column(String(64), primary_key=True, comment="父任务 ID（UUID）")
    task_name = Column(String(256), nullable=True, comment="批量任务名称（展示）")
    source_format = Column(String(32), nullable=True, comment="源格式，如 MCAP")
    target_format = Column(String(64), nullable=True, comment="目标格式 HDF5/LeRobot")
    project_id = Column(String(128), nullable=True, index=True, comment="项目 ID")
    project_name = Column(String(256), nullable=True, comment="项目名称（冗余）")
    creator_id = Column(String(64), nullable=True, index=True, comment="创建人 users.id")

    total_count = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)
    failed_count = Column(Integer, nullable=False, default=0, comment="失败 + 取消（用于与 total 分解一致）")
    canceled_count = Column(Integer, nullable=False, default=0, server_default=text("0"), comment="子任务 canceled 条数")
    running_count = Column(Integer, nullable=False, default=0)
    pending_count = Column(Integer, nullable=False, default=0)
    progress_percent = Column(Float, nullable=False, default=0.0)
    overall_status = Column(String(32), nullable=False, default="PENDING", comment="PENDING|RUNNING|SUCCESS|PARTIAL_SUCCESS|FAILED|CANCELED")

    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_conversion_batch_jobs_project", "project_id"),
        Index("idx_conversion_batch_jobs_created", "created_at"),
        Index("idx_conversion_batch_jobs_updated", "updated_at"),
    )


class ConversionJobAsset(Base):
    """转换任务表（PostgreSQL，与数据资产同库）"""

    __tablename__ = "conversion_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_id = Column(String(64), nullable=True, comment="父批量任务 ID（conversion_batch_jobs.batch_id），旧数据为空")
    job_id = Column(String(64), unique=True, nullable=False, index=True, comment="转换任务 ID（UUID）")
    short_code = Column(String(32), nullable=True, comment="短码")
    task_no = Column(String(32), nullable=True, comment="展示编号 0001/0002...")
    task_name = Column(String(256), nullable=True, comment="转换任务名称（展示用）")

    input_dataset_id = Column(String(64), nullable=True, index=True, comment="输入资产 ID（data_assets.id）")
    input_asset_name = Column(String(512), nullable=True, comment="输入资产文件名")
    input_file_path = Column(String(1024), nullable=True, comment="输入文件路径（冗余）")

    project_id = Column(String(128), nullable=True, index=True, comment="项目 ID")
    project_name = Column(String(256), nullable=True, comment="项目名称（冗余）")
    device_name = Column(String(256), nullable=True, comment="设备名称（冗余）")

    output_format = Column(String(64), nullable=True, comment="HDF5/LeRobot")
    file_format = Column(String(32), nullable=True, comment="输入文件格式（MCAP）")
    output_location = Column(String(64), nullable=True, comment="local/cloud")
    output_file_name = Column(String(512), nullable=True, comment="输出文件名（或 repo_id）")
    output_path = Column(String(1024), nullable=True, comment="输出路径（目录或 repo_id）")

    status = Column(String(32), nullable=False, default="queued", comment="queued/running/succeeded/failed/canceled")
    progress_percent = Column(Float, nullable=False, default=0, comment="0-100")
    current_stage = Column(String(64), nullable=True, comment="当前阶段")
    stages_json = Column(Text, nullable=True, comment="JSON stages")
    logs_json = Column(Text, nullable=True, comment="JSON logs")

    artifact_ready = Column(Boolean, nullable=False, default=False, comment="产物是否就绪")
    error_message = Column(Text, nullable=True, comment="错误信息")
    operator_name = Column(String(256), nullable=True, comment="触发转换任务的操作人账号名")

    created_at = Column(DateTime, server_default=func.now(), nullable=False, comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")

    __table_args__ = (
        Index("idx_conversion_jobs_job_id", "job_id"),
        Index("idx_conversion_jobs_project", "project_id"),
        Index("idx_conversion_jobs_input_dataset_id", "input_dataset_id"),
        Index("idx_conversion_jobs_batch_id", "batch_id"),
        Index("idx_conversion_jobs_created", "created_at"),
        Index("idx_conversion_jobs_updated", "updated_at"),
    )


class CollectionTaskAsset(Base):
    __tablename__ = "collection_tasks"

    id = Column(String(64), primary_key=True)
    name = Column(String(256), nullable=False, index=True)
    description = Column(Text, nullable=True)
    status = Column(String(32), nullable=False, default="DRAFT")
    project_id = Column(String(128), nullable=True, index=True)
    project_name = Column(String(256), nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_collection_tasks_project", "project_id"),
        Index("idx_collection_tasks_updated", "updated_at"),
    )


class CollectionJobAsset(Base):
    __tablename__ = "collection_jobs"

    id = Column(String(64), primary_key=True)
    task_id = Column(String(64), nullable=False, index=True)
    job_number = Column(Integer, nullable=False, default=0)
    status = Column(String(32), nullable=False, default="PENDING")
    operator_name = Column(String(256), nullable=True)
    project_id = Column(String(128), nullable=True, index=True)
    project_name = Column(String(256), nullable=True)
    mcap_path = Column(String(1024), nullable=True)
    mcap_size_bytes = Column(BigInteger, nullable=True, comment="MCAP 文件大小（字节）")
    # 采集脚本经日志 EAI_VALIDATION_REPORT_JSON 输出的校验报告（JSON 文本），供质检页跨标签/刷新读取
    validation_report_json = Column(Text, nullable=True)
    duration_sec = Column(Integer, nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    progress = Column(Integer, nullable=False, default=0)
    collection_quantity = Column(Integer, nullable=True, default=0)
    completed_count = Column(Integer, nullable=True, default=0)
    # 领取作业时选中的平台设备 devices.id（与任务上旧 deviceId 脱钩，避免与采集端 DEVICES 配置数字混淆）
    device_id = Column(String(64), nullable=True, comment="平台设备主键 devices.id")
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_collection_jobs_task", "task_id"),
        Index("idx_collection_jobs_project", "project_id"),
        Index("idx_collection_jobs_updated", "updated_at"),
    )


class SyncBatchJob(Base):
    """批量同步任务（持久化，PostgreSQL）。"""

    __tablename__ = "sync_batch_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), unique=True, nullable=False, index=True)
    user_id = Column(String(64), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="queued")  # queued|running|succeeded|failed|canceled
    agent_id_query = Column(String(128), nullable=True)
    total = Column(Integer, nullable=False, default=0)
    succeeded = Column(Integer, nullable=False, default=0)
    failed = Column(Integer, nullable=False, default=0)
    progress_percent = Column(Float, nullable=False, default=0.0)
    current_step = Column(String(512), nullable=True, default="")
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    __table_args__ = (Index("idx_sync_batch_jobs_user", "user_id"), Index("idx_sync_batch_jobs_created", "created_at"))


class SyncBatchJobItem(Base):
    """批量同步子项。"""

    __tablename__ = "sync_batch_job_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(64), ForeignKey("sync_batch_jobs.job_id", ondelete="CASCADE"), nullable=False, index=True)
    asset_id = Column(Integer, nullable=False, index=True)
    sort_order = Column(Integer, nullable=False, default=0)
    status = Column(String(32), nullable=False, default="pending")  # pending|running|succeeded|failed|skipped
    error_message = Column(Text, nullable=True)
    minio_path = Column(String(1024), nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    __table_args__ = (Index("idx_sync_batch_items_job_order", "job_id", "sort_order"),)


class DataAssetUploadSession(Base):
    """浏览器直传 MinIO 上传会话（Phase 1 单文件 / Phase 2 多文件与目录）。"""

    __tablename__ = "upload_sessions"

    id = Column(String(64), primary_key=True, comment="upload_session_id（UUID hex 等）")
    user_id = Column(String(64), nullable=False, index=True)
    project_id = Column(String(128), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="presigned", comment="presigned|completed|failed|expired")
    bucket = Column(String(256), nullable=False)
    object_key = Column(String(1024), nullable=False)
    filename = Column(String(512), nullable=False)
    size_bytes = Column(BigInteger, nullable=False, default=0)
    content_type = Column(String(256), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    upload_mode = Column(String(32), nullable=False, default="single_file", comment="single_file|multi_file|directory")
    items_json = Column(Text, nullable=True, comment="JSON：预签名白名单 client_file_id/relative_path/object_key/size_bytes")
    manifest_json = Column(Text, nullable=True)
    expected_count = Column(Integer, nullable=True)
    expected_total_size = Column(BigInteger, nullable=True)
    root_dir_name = Column(String(512), nullable=True)
    asset_name = Column(String(512), nullable=True)
    result_payload_json = Column(Text, nullable=True, comment="completed 幂等返回缓存")

    __table_args__ = (Index("idx_upload_sessions_user_project", "user_id", "project_id"),)


class TaskJob(Base):
    """统一任务状态表（dispatcher/worker 共享状态源）。"""

    __tablename__ = "task_jobs"

    id = Column(String(64), primary_key=True)  # task_id
    rq_job_id = Column(String(64), nullable=True, index=True)
    task_type = Column(String(32), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="pending", index=True)  # pending|queued|running|success|failed
    user_id = Column(String(64), nullable=True, index=True)
    queue_name = Column(String(64), nullable=True, index=True)
    payload = Column(JSON, nullable=False)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_task_jobs_user_created", "user_id", "created_at"),
        Index("idx_task_jobs_type_status", "task_type", "status"),
    )
