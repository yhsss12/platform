"""
自动标注服务
处理异步标注任务
"""
import asyncio
import uuid
from typing import Optional, Dict, Any
from app.services.hdf5_service import HDF5Service
from app.services.task_config_service import TaskConfigService
import json
import os
from app.core.config import settings
from app.services.dispatcher import dispatch_task
from app.services.task_job_store import is_cancelled
from app.services.episode_storage import EpisodeStorage
from app.services.storage_resolver import EpisodeResolveError
from app.services.data_asset_path_resolver import minio_uri_from_fields


def resolve_annotation_executor_file_path(task_params: Dict[str, Any]) -> str:
    """
    供 RQ worker 使用：API 进程解析出的 file_path（如 .minio_view_cache 下）在 worker 容器内常不存在。
    本机路径可读则直接返回；否则用 MinIO warehouse_path 拉取；再否则按 label task 的 episodes_index 在 worker 上重新 resolve。
    """
    from app.services.data_asset_path_resolver import resolve_read_local_from_warehouse_uri

    p = task_params or {}
    fp = (p.get("file_path") or "").strip()
    wh = (p.get("annotation_warehouse_path") or "").strip()
    episode_id = (p.get("annotation_episode_id") or "").strip()
    label_task_id = (p.get("annotation_label_task_id") or "").strip()

    if fp and os.path.isfile(fp):
        return fp

    if wh.startswith("minio://"):
        return resolve_read_local_from_warehouse_uri(wh)

    if label_task_id and episode_id:
        base_dir = os.getenv("HDF5_DATA_DIR", "/tmp/hdf5_data")
        svc = TaskConfigService(base_dir=base_dir)
        ep = svc.find_episode_by_id(label_task_id, episode_id)
        if ep:
            return EpisodeStorage(ep).resolve_local_path()

    if fp.startswith("minio://"):
        return resolve_read_local_from_warehouse_uri(fp)

    if fp:
        raise FileNotFoundError(fp)

    raise ValueError("annotation task missing file_path")


class AnnotationService:
    """自动标注服务"""
    
    def __init__(self, hdf5_service: HDF5Service):
        self.hdf5_service = hdf5_service
        self.jobs: Dict[str, Dict] = {}  # job_id -> job_info
        # 初始化 TaskConfigService
        base_dir = os.getenv("HDF5_DATA_DIR", "/tmp/hdf5_data")
        self.task_config_service = TaskConfigService(base_dir=base_dir)
    
    async def generate_description_async(
        self,
        episode_id: str,
        camera_name: Optional[str] = None,
        task_id: Optional[str] = None,
        user_id: Optional[str] = None,
        model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ) -> str:
        """
        异步生成任务描述
        
        Args:
            episode_id: episode ID
            camera_name: 相机名称
            task_id: 任务 ID（可选）
            model: 模型 ID（可选，覆盖环境变量）
            openai_api_key: API Key（可选）
            openai_base_url: API 地址（可选）
            
        Returns:
            str: job_id
        """
        job_id = str(uuid.uuid4())
        
        # 查找 episode
        if task_id:
            # 从 task_config_service 查找
            episode_info = self.task_config_service.find_episode_by_id(task_id, episode_id)
            if not episode_info:
                raise ValueError(f"Episode {episode_id} not found in task {task_id}")
            try:
                hdf5_path = EpisodeStorage(episode_info).resolve_local_path()
            except EpisodeResolveError as e:
                raise ValueError(str(e))
            annotation_wh = (episode_info.get("warehouse_path") or "").strip()
            ann_label_tid = (task_id or "").strip()
            ep_fp = (episode_info.get("file_path") or "").strip()
            meta_raw = episode_info.get("meta")
            if isinstance(meta_raw, dict):
                ep_meta_json = json.dumps(meta_raw, ensure_ascii=False)
            elif isinstance(meta_raw, str):
                ep_meta_json = meta_raw.strip() or None
            else:
                ep_meta_json = None
            annotation_asset_minio_uri = minio_uri_from_fields(ep_fp or None, ep_meta_json) or ""
        else:
            # 兼容旧逻辑：从默认 data_dir 查找
            episode = self.hdf5_service._find_episode_by_id(episode_id)
            if not episode:
                raise ValueError(f"Episode {episode_id} not found")
            hdf5_path = episode["path"]
            annotation_wh = (episode.get("warehouse_path") or "").strip()
            ann_label_tid = ""
            ep_fp = (episode.get("file_path") or "").strip()
            meta_raw = episode.get("meta")
            if isinstance(meta_raw, dict):
                ep_meta_json = json.dumps(meta_raw, ensure_ascii=False)
            elif isinstance(meta_raw, str):
                ep_meta_json = meta_raw.strip() or None
            else:
                ep_meta_json = None
            annotation_asset_minio_uri = minio_uri_from_fields(ep_fp or None, ep_meta_json) or ""

        # 初始化任务状态
        self.jobs[job_id] = {
            "status": "running",
            "progress": 0,
            "episode_id": episode_id,
            "camera_name": camera_name,
            "task_id": task_id,
            "result": None,
            "error": None,
        }
        
        # 在后台执行（统一走 dispatcher）
        dispatch_task({
            "type": "annotation",
            "task_id": job_id,
            "user_id": user_id,
            "job_id": job_id,
            "file_path": hdf5_path,
            "annotation_episode_id": str(episode_id),
            "annotation_label_task_id": ann_label_tid,
            "annotation_warehouse_path": annotation_wh,
            "annotation_asset_minio_uri": annotation_asset_minio_uri,
            "camera_name": camera_name,
            "model": model,
            "openai_api_key": openai_api_key,
            "openai_base_url": openai_base_url,
        })
        
        return job_id
    
    async def _run_annotation(
        self,
        job_id: str,
        file_path: str,
        camera_name: Optional[str],
        model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ):
        """执行标注任务。支持 HDF5（camera_name）和 MCAP（camera_name 为完整 topic 时作为 image_topic）。"""
        # worker 进程与 API 进程是不同对象，self.jobs 可能为空。
        # 这里仅用于避免 KeyError；最终 UI 状态以 task_jobs 持久层为准。
        if job_id not in self.jobs:
            self.jobs[job_id] = {
                "status": "running",
                "progress": 0,
                "episode_id": None,
                "camera_name": camera_name,
                "task_id": None,
                "result": None,
                "error": None,
            }

        if is_cancelled(job_id):
            print(f"[Cancel] Task {job_id} cancelled")
            self.jobs[job_id]["status"] = "cancelled"
            return None
        try:
            # 更新进度
            self.jobs[job_id]["progress"] = 10
            self.jobs[job_id]["status"] = "running"

            is_mcap = file_path.lower().endswith(".mcap")
            loop = asyncio.get_event_loop()
            if is_mcap:
                description = await loop.run_in_executor(
                    None,
                    lambda: self.hdf5_service.generate_task_description(
                        file_path,
                        image_topic=camera_name,
                        model=model,
                        api_key=openai_api_key,
                        base_url=openai_base_url,
                    ),
                )
            else:
                description = await loop.run_in_executor(
                    None,
                    lambda: self.hdf5_service.generate_task_description(
                        file_path,
                        camera_name=camera_name,
                        model=model,
                        api_key=openai_api_key,
                        base_url=openai_base_url,
                    ),
                )

            if is_cancelled(job_id):
                print(f"[Cancel] Task {job_id} cancelled")
                self.jobs[job_id]["status"] = "cancelled"
                return None
            
            if description:
                self.jobs[job_id]["progress"] = 100
                self.jobs[job_id]["status"] = "completed"
                self.jobs[job_id]["result"] = description
                return description
            else:
                self.jobs[job_id]["status"] = "failed"
                # 检查是否是环境变量问题
                api_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    self.jobs[job_id]["error"] = "OPENAI_API_KEY 环境变量未设置。请在 backend/.env 文件中配置环境变量。"
                else:
                    self.jobs[job_id]["error"] = "Failed to generate description"
                # 让 worker 把任务置为 failed，并记录 error。
                raise RuntimeError(self.jobs[job_id]["error"])
                
        except Exception as e:
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            raise
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """获取任务状态"""
        return self.jobs.get(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """取消任务"""
        if job_id in self.jobs:
            if self.jobs[job_id]["status"] == "running":
                self.jobs[job_id]["status"] = "cancelled"
                return True
        return False


