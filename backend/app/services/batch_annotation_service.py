"""
批量标注服务
整合 annotation_page.py 的批量标注功能，支持 HDF5 和 MCAP。
"""
import os
import json
from typing import List, Dict, Optional
from app.core.instruction_path import get_instruction_path_for_data_path
from app.services.hdf5_service import HDF5Service
from app.services.task_config_service import TaskConfigService
from app.services.mcap_service import MCAPService
from app.services.data_asset_path_resolver import (
    evict_minio_view_cache,
    local_cache_path_for_minio_uri,
    resolve_read_local_from_warehouse_uri,
)
from app.services.episode_storage import EpisodeStorage
from app.services.storage_resolver import EpisodeResolveError


def _is_mcap(path: str) -> bool:
    return path.lower().endswith(".mcap")


class BatchAnnotationService:
    """批量标注服务"""
    
    def __init__(self, hdf5_service: HDF5Service):
        self.hdf5_service = hdf5_service
        self.mcap_service = MCAPService()
    
    def perform_batch_annotation(
        self,
        dataset_paths: List[str],
        camera_name: Optional[str] = None,
        fallback_first_camera: bool = True,
    ) -> List[Dict]:
        """
        批量自动标注（整合 annotation_page.py 的逻辑）
        
        Args:
            dataset_paths: HDF5 文件路径列表
            camera_name: 相机名称，如果为 None 则使用第一个相机
            fallback_first_camera: 如果指定相机不存在，是否回退到第一个相机
            
        Returns:
            List[Dict]: 结果列表，每个元素包含 episode_data / dataset_path / output_path / camera_used
        """
        from app.services.episode_storage import clear_episode_cache

        results: List[Dict] = []
        try:
            for dataset_path in dataset_paths:
                logical_path = dataset_path
                try:
                    read_path = dataset_path
                    if (dataset_path or "").strip().startswith("minio://"):
                        read_path = resolve_read_local_from_warehouse_uri(dataset_path.strip())

                    if not os.path.exists(read_path):
                        results.append({"dataset_path": logical_path, "error": "File not found"})
                        continue

                    is_mcap = _is_mcap(read_path)
                    if is_mcap:
                        cameras = self.mcap_service.list_cameras(read_path)
                        if not cameras:
                            results.append({"dataset_path": logical_path, "error": "No cameras/topics found in MCAP"})
                            continue
                        if camera_name and camera_name in cameras:
                            selected_camera = camera_name
                        else:
                            selected_camera = cameras[0]
                        task_description = self.hdf5_service.generate_task_description(
                            read_path,
                            image_topic=selected_camera,
                        )
                    else:
                        import h5py

                        with h5py.File(read_path, "r") as f:
                            cameras = self.hdf5_service.list_cameras(f)
                            if not cameras:
                                results.append({"dataset_path": logical_path, "error": "No cameras found"})
                                continue
                        if camera_name and camera_name in cameras:
                            selected_camera = camera_name
                        else:
                            selected_camera = cameras[0]
                        task_description = self.hdf5_service.generate_task_description(
                            read_path,
                            camera_name=selected_camera,
                        )

                    if not task_description:
                        results.append({"dataset_path": logical_path, "error": "Failed to generate description"})
                        continue

                    episode_index = self.hdf5_service._extract_episode_id(os.path.basename(read_path))
                    out_path = get_instruction_path_for_data_path(read_path)

                    existing_instructions = []
                    if os.path.exists(out_path):
                        try:
                            with open(out_path, "r", encoding="utf-8") as f:
                                content = f.read().strip()
                                if content:
                                    data = json.loads(content)
                                    if isinstance(data, dict) and "instructions" in data:
                                        existing_instructions = data["instructions"]
                        except Exception:
                            pass

                    episode_idx = int(episode_index) if episode_index.isdigit() else 0
                    while len(existing_instructions) <= episode_idx:
                        existing_instructions.append("")
                    existing_instructions[episode_idx] = task_description

                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump({"instructions": existing_instructions}, f, ensure_ascii=False, indent=2)

                    results.append(
                        {
                            "dataset_path": logical_path,
                            "output_path": out_path,
                            "camera_used": selected_camera,
                            "episode_data": {
                                "episode_index": episode_idx,
                                "tasks": [task_description],
                                "length": len(task_description),
                            },
                        }
                    )
                except Exception as e:
                    results.append({"dataset_path": logical_path, "error": str(e)})
                finally:
                    if (dataset_path or "").strip().startswith("minio://"):
                        evict_minio_view_cache(dataset_path.strip())
            return results
        finally:
            # batch：每个 task 执行完清空（任务级生命周期）
            clear_episode_cache()

    def perform_batch_annotation_by_task(
        self,
        task_id: str,
        task_config_service: TaskConfigService,
        camera_name: Optional[str] = None,
        fallback_first_camera: bool = True,
        model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
    ) -> List[Dict]:
        """
        按任务批量自动标注：从任务 episodes_index 取路径，逐条生成并写入该任务专属的 instructions.json。
        """
        from app.services.episode_storage import clear_episode_cache

        episodes = task_config_service.load_episodes_index(task_id)
        if not episodes:
            return []
        if not isinstance(episodes, list):
            episodes = list(episodes) if episodes else []

        results: List[Dict] = []
        instructions = task_config_service.load_task_instructions(task_id)
        if not isinstance(instructions, list):
            instructions = []

        try:
            for idx, ep in enumerate(episodes):
                if not isinstance(ep, dict):
                    results.append({"episode_id": str(idx), "error": "Invalid episode format"})
                    continue

                storage = EpisodeStorage(ep)
                wh = storage.get_storage_key()
                episode_id = ep.get("episode_id")
                to_evict: Optional[str] = wh if wh.startswith("minio://") else None

                try:
                    abs_path = storage.resolve_local_path()
                    if not abs_path or not os.path.isfile(abs_path):
                        results.append({"episode_id": episode_id, "error": f"File not found: {abs_path}"})
                        continue

                    is_mcap = _is_mcap(abs_path)
                    if is_mcap:
                        cameras = self.mcap_service.list_cameras(abs_path)
                        if not cameras:
                            results.append({"episode_id": episode_id, "error": "No cameras/topics in MCAP"})
                            continue
                        selected_camera = camera_name if (camera_name and camera_name in cameras) else cameras[0]
                        task_description = self.hdf5_service.generate_task_description(
                            abs_path,
                            image_topic=selected_camera,
                            model=model,
                            api_key=openai_api_key,
                            base_url=openai_base_url,
                        )
                    else:
                        import h5py

                        with h5py.File(abs_path, "r") as f:
                            cameras = self.hdf5_service.list_cameras(f)
                            if not cameras:
                                results.append({"episode_id": episode_id, "error": "No cameras found"})
                                continue
                        selected_camera = camera_name if (camera_name and camera_name in cameras) else cameras[0]
                        task_description = self.hdf5_service.generate_task_description(
                            abs_path,
                            camera_name=selected_camera,
                            model=model,
                            api_key=openai_api_key,
                            base_url=openai_base_url,
                        )

                    if not task_description:
                        results.append({"episode_id": episode_id, "error": "Failed to generate description"})
                        continue

                    while len(instructions) <= idx:
                        instructions.append("")
                    instructions[idx] = task_description
                    task_config_service.save_task_instructions(task_id, instructions)

                    out_path = get_instruction_path_for_data_path(abs_path)
                    out_dir = abs_path if os.path.isdir(abs_path) else os.path.dirname(abs_path)

                    def _ep_sidecar_dir(e: dict) -> str:
                        es = EpisodeStorage(e)
                        w = es.get_storage_key()
                        if w.startswith("minio://"):
                            try:
                                lf = local_cache_path_for_minio_uri(w)
                                return lf if w.endswith("/") else os.path.dirname(lf)
                            except Exception:
                                return ""
                        try:
                            ap = es.resolve_local_path()
                            return ap if os.path.isdir(ap) else os.path.dirname(ap)
                        except Exception:
                            return ""

                    same_dir = [(i, e) for i, e in enumerate(episodes) if _ep_sidecar_dir(e) == out_dir]
                    same_dir.sort(key=lambda x: EpisodeStorage(x[1]).get_storage_key())
                    dir_idx = next((i for i, (_, e) in enumerate(same_dir) if e.get("episode_id") == episode_id), 0)

                    existing = []
                    if os.path.exists(out_path):
                        try:
                            with open(out_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                existing = data.get("instructions") or []
                        except Exception:
                            pass
                    while len(existing) <= dir_idx:
                        existing.append("")
                    existing[dir_idx] = task_description
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump({"instructions": existing}, f, ensure_ascii=False, indent=2)

                    results.append(
                        {
                            "episode_id": episode_id,
                            "path": abs_path,
                            "abs_path": abs_path,
                            "warehouse_path": wh if wh.startswith("minio://") else "",
                            "output_path": task_config_service.get_instructions_path(task_id),
                            "camera_used": selected_camera,
                            "instruction": task_description,
                        }
                    )
                except Exception as e:
                    results.append({"episode_id": episode_id, "error": str(e)})
                finally:
                    if to_evict:
                        evict_minio_view_cache(to_evict)

            return results
        finally:
            # batch：每个 task 执行完清空（任务级生命周期）
            clear_episode_cache()

