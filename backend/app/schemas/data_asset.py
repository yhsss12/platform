"""
数据资产 Schema
"""
from pydantic import BaseModel, Field, computed_field, field_serializer
from typing import Optional
from datetime import datetime, timezone

_LOCAL_TZ = datetime.now().astimezone().tzinfo or timezone.utc


class DataAssetBase(BaseModel):
    filename: str
    format: str  # hdf5 | mcap | lerobot
    # 来源：import（导入）| collect（采集）| label（标注）| convert（转换）
    source: str = "import"
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    file_path: str
    file_size_bytes: int = 0
    meta: Optional[str] = None
    parse_status: str = "未解析"
    error_msg: Optional[str] = None
    sync_status: str = "synced"
    sync_error: Optional[str] = None
    # 采集来源：平台 devices.id，便于列表区分采集端
    device_id: Optional[str] = None
    # 最近一次业务动作时间；创建时可为空，由数据库默认值补充
    updated_at: Optional[datetime] = None
    # 资产创建/导入/登记时的真实操作者（历史字段）
    operator_name: Optional[str] = None
    # 采集/标注/转换来源的任务名称（列表与详情展示）
    collect_task_name: Optional[str] = None


class DataAssetCreate(DataAssetBase):
    code: str


class DataAssetUpdate(BaseModel):
    parse_status: Optional[str] = None
    error_msg: Optional[str] = None
    meta: Optional[str] = None
    sync_status: Optional[str] = None
    sync_error: Optional[str] = None
    device_id: Optional[str] = None


def _datetime_to_iso_utc(dt: datetime) -> str:
    """序列化为 ISO 字符串并带 Z，naive 时间按服务端本地时区解释后转 UTC。"""
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_LOCAL_TZ)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _datetime_to_epoch_ms(dt: datetime) -> int:
    """统一输出 UTC 毫秒时间戳（13 位）。naive 按服务端本地时区解释。"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=_LOCAL_TZ)
    dt_utc = dt.astimezone(timezone.utc)
    return int(dt_utc.timestamp() * 1000)


class DataAssetResponse(DataAssetBase):
    id: int
    dataset_id: Optional[str] = None  # 数据专属唯一标识 DS000001，标注/转换/导出均以此为存在依据
    code: str
    created_at: datetime
    updated_at: datetime
    instruction_text: Optional[str] = None  # 标注信息，与 instruction.json 同步
    label_task_name: Optional[str] = None
    collect_task_name: Optional[str] = None
    conversion_task_name: Optional[str] = None
    sync_status: str = "synced"
    sync_error: Optional[str] = None
    # 操作人（账号名）；历史字段，不应回落为当前查看者
    operator_name: Optional[str] = None
    # 对象存储地址 minio://...（由 file_path / meta 推导，供前端展示；与未同步前的本地 file_path 可并存）
    warehouse_uri: Optional[str] = None
    # 采集资产：相对作业 workspace 的 episode 路径（如 2026-05-11/episode_0_xxx）；仅列表开启对账时有值
    collect_episode_rel_path: Optional[str] = None
    # 采集资产：当前采集端磁盘是否仍存在该 episode 目录；None 表示未校验或无法关联 Agent
    collect_episode_on_device: Optional[bool] = None

    @field_serializer("created_at", "updated_at")
    def serialize_datetime_utc(self, dt: datetime) -> str:
        return _datetime_to_iso_utc(dt)

    @computed_field(return_type=int)
    @property
    def created_at_ms(self) -> int:
        return _datetime_to_epoch_ms(self.created_at)

    @computed_field(return_type=int)
    @property
    def updated_at_ms(self) -> int:
        return _datetime_to_epoch_ms(self.updated_at)

    class Config:
        from_attributes = True


class DataAssetQueryParams(BaseModel):
    keyword: Optional[str] = None
    project: Optional[str] = None
    format: Optional[str] = None
    source: Optional[str] = None
    task_id: Optional[str] = None
    task_name: Optional[str] = None
    # 入库/创建日期 YYYY-MM-DD（UTC 日界，含起止当日）
    created_from: Optional[str] = None
    created_to: Optional[str] = None
    page: int = 1
    page_size: int = 20


class DataAssetListResponse(BaseModel):
    items: list[DataAssetResponse]
    total: int
    page: int
    page_size: int


class ImportResultItem(BaseModel):
    name: str
    id: Optional[int] = None
    reason: Optional[str] = None


class LocalFileItem(BaseModel):
    name: str
    path: str  # 相对平台数据资产根目录的路径
    is_dir: bool
    size: Optional[int] = None


class RegisterAssetRequest(BaseModel):
    project_id: str
    project_name: Optional[str] = None
    type: str  # "file" | "dir"
    path: str  # 绝对路径


class DeleteAssetsBatchBody(BaseModel):
    asset_ids: list[int]
    delete_file: bool = False
    delete_remote: bool = False
    delete_cloud: bool = True


class ExportRequest(BaseModel):
    """导出数据资产。不传 output_path 时由服务端写入临时 zip，通过 /export/download 下载；传 output_path 时仍写入白名单目录（兼容旧调用）。"""
    asset_ids: list[int]
    target: str = "local"  # "local" | "cloud"，当前仅支持 local
    output_path: Optional[str] = None  # 可选；为空则浏览器 zip 下载模式
    compression_mode: Optional[str] = None  # 可选：store | deflated


class DeleteExportResultRequest(BaseModel):
    """删除导出任务产物。仅允许删除白名单内的路径。"""
    job_id: str


class SyncBatchCreateBody(BaseModel):
    """批量同步到 MinIO（异步任务，见 POST /data-assets/sync/batch）。"""
    asset_ids: list[int]
    agent_id: str | None = None


class ReparseFromMinioBatchBody(BaseModel):
    """对已同步到 MinIO 的资产重新解析 meta（修复历史「等待落盘后再解析」占位）。"""
    limit: int = Field(200, ge=1, le=2000, description="扫描条数上限（不含 asset_ids 模式）")
    project_id: Optional[str] = Field(None, description="可选，仅处理该项目")
    stale_only: bool = Field(
        True,
        description="True：仅 error_msg 含「等待落盘」；False：当前用户可见范围内全部 minio+mcap/hdf5（仅超级管理员）",
    )
    asset_ids: Optional[list[int]] = Field(
        None,
        description="指定资产 ID（最多 300）；指定时不受 stale_only 过滤，仍受项目可见性限制",
    )


class DirectUploadInitFileItem(BaseModel):
    client_file_id: str
    relative_path: str
    size_bytes: int
    content_type: str | None = None


class DirectUploadInitBody(BaseModel):
    """直传：申请预签名 PUT URL（single_file | multi_file | directory）。"""
    upload_mode: str = "single_file"
    project_id: str
    filename: str | None = None
    size_bytes: int | None = None
    content_type: str | None = None
    items: list[DirectUploadInitFileItem] | None = None
    root_dir_name: str | None = None


class DirectUploadManifestPathEntry(BaseModel):
    relative_path: str
    size_bytes: int


class DirectUploadDirectoryManifestIn(BaseModel):
    root_dir_name: str
    paths: list[DirectUploadManifestPathEntry]
    total_files: int
    total_size_bytes: int


class DirectUploadCompleteBody(BaseModel):
    """直传完成：single 传 size_bytes；directory 传 manifest；multi 仅 session。"""
    upload_session_id: str
    size_bytes: int | None = None
    manifest: DirectUploadDirectoryManifestIn | None = None


class DirectUploadInitItemOut(BaseModel):
    client_file_id: str
    relative_path: str
    object_key: str
    upload_url: str
    method: str = "PUT"
    headers: dict = Field(default_factory=dict)


class DirectUploadInitData(BaseModel):
    upload_session_id: str
    bucket: str
    expires_at: str
    upload_mode: str = "single_file"
    upload_items: list[DirectUploadInitItemOut] = Field(default_factory=list)
    object_key: str | None = None
    upload_url: str | None = None
    method: str | None = "PUT"
    headers: dict = Field(default_factory=dict)
