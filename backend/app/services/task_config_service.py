"""
任务配置服务
用于管理任务的配置文件（task.json），不依赖数据库
"""
import os
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime


class TaskConfigService:
    """任务配置服务"""
    
    def __init__(self, base_dir: Optional[str] = None):
        """
        初始化服务
        
        Args:
            base_dir: 基础目录，如果为 None 则从环境变量读取
        """
        self.base_dir = base_dir or os.getenv("HDF5_DATA_DIR", "/tmp/hdf5_data")
        self.tasks_dir = os.path.join(self.base_dir, "tasks")
        os.makedirs(self.tasks_dir, exist_ok=True)
    
    def get_task_config_path(self, task_id: str) -> str:
        """获取任务配置文件路径"""
        task_dir = os.path.join(self.tasks_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)
        return os.path.join(task_dir, "task.json")
    
    def get_episodes_index_path(self, task_id: str) -> str:
        """获取 episodes 索引文件路径"""
        task_dir = os.path.join(self.tasks_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)
        return os.path.join(task_dir, "episodes_index.json")
    
    def save_task_config(self, task_id: str, config: Dict[str, Any]) -> None:
        """
        保存任务配置到 task.json
        
        Args:
            task_id: 任务 ID
            config: 配置字典，应包含 task_id, name, dataset_path, device_type, created_at 等
        """
        config_path = self.get_task_config_path(task_id)
        
        # 确保包含必要字段
        config.setdefault("task_id", task_id)
        if "created_at" not in config:
            config["created_at"] = datetime.now().isoformat()
        
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    
    def load_task_config(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        从 task.json 加载任务配置
        
        Args:
            task_id: 任务 ID
            
        Returns:
            配置字典，如果文件不存在则返回 None
        """
        config_path = self.get_task_config_path(task_id)
        
        if not os.path.exists(config_path):
            return None
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading task config for {task_id}: {e}")
            return None
    
    def get_dataset_path(self, task_id: str) -> Optional[str]:
        """
        获取任务的 dataset_path
        
        Args:
            task_id: 任务 ID
            
        Returns:
            dataset_path，如果不存在则返回 None
        """
        config = self.load_task_config(task_id)
        if not config:
            return None
        return config.get("dataset_path")
    
    def validate_dataset_path(self, dataset_path: str) -> bool:
        """
        验证 dataset_path 是否在允许的根目录下
        
        Args:
            dataset_path: 数据集路径（可以是文件或目录）
            
        Returns:
            是否有效
        """
        # 允许的根目录白名单（可配置）
        home_dir = os.path.expanduser("~")
        extra_roots = (os.getenv("DATA_PATH_ALLOWED_ROOTS") or "").strip().split(":")
        extra_roots = [r for r in extra_roots if r]
        allowed_roots = [
            "/data/hdf5",
            "/app/data/hdf5",
            os.getenv("HDF5_DATA_DIR", "/tmp/hdf5_data"),
            home_dir,  # 用户主目录，如 /home/ubuntu
        ] + extra_roots
        
        # 规范化路径
        abs_path = os.path.abspath(dataset_path)
        
        # 如果是文件路径，检查其父目录或文件本身
        # 如果是目录路径，检查目录本身
        if os.path.isfile(abs_path):
            # 对于文件，检查文件本身或其父目录是否在允许的根目录下
            parent_dir = os.path.dirname(abs_path)
            # 也允许项目目录下的文件
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            if abs_path.startswith(project_root) or parent_dir.startswith(project_root):
                return True
        else:
            # 对于目录，检查目录本身
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
            if abs_path.startswith(project_root):
                return True
        
        # 检查是否在允许的根目录下
        for root in allowed_roots:
            root_abs = os.path.abspath(root)
            if abs_path.startswith(root_abs):
                return True
        
        return False
    
    def save_episodes_index(self, task_id: str, episodes: List[Dict[str, Any]]) -> None:
        """
        保存 episodes 索引到 episodes_index.json
        
        Args:
            task_id: 任务 ID
            episodes: episode 列表，每个包含 episode_id, filename, abs_path, mtime, size_bytes 等
        """
        index_path = self.get_episodes_index_path(task_id)
        
        index_data = {
            "task_id": task_id,
            "updated_at": datetime.now().isoformat(),
            "episodes": episodes
        }
        
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
    
    def load_episodes_index(self, task_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        从 episodes_index.json 加载 episodes 索引
        
        Args:
            task_id: 任务 ID
            
        Returns:
            episode 列表，如果文件不存在则返回 None
        """
        index_path = self.get_episodes_index_path(task_id)
        
        if not os.path.exists(index_path):
            return None
        
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("episodes", [])
        except Exception as e:
            print(f"Error loading episodes index for {task_id}: {e}")
            return None
    
    def find_episode_by_id(self, task_id: str, episode_id: str) -> Optional[Dict[str, Any]]:
        """
        根据 episode_id 查找 episode 信息
        
        Args:
            task_id: 任务 ID
            episode_id: episode ID
            
        Returns:
            episode 信息，如果不存在则返回 None
        """
        episodes = self.load_episodes_index(task_id)
        if not episodes:
            return None
        
        for episode in episodes:
            if episode.get("episode_id") == episode_id:
                return episode
        
        return None

    def get_instructions_path(self, task_id: str) -> str:
        """任务专属标注结果文件路径（按任务隔离，同一数据集多任务互不干扰）"""
        task_dir = os.path.join(self.tasks_dir, task_id)
        os.makedirs(task_dir, exist_ok=True)
        return os.path.join(task_dir, "instructions.json")

    def load_task_instructions(self, task_id: str) -> List[str]:
        """加载该任务的标注结果列表（与 episodes 顺序一致），无则返回空列表"""
        path = self.get_instructions_path(task_id)
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("instructions") or []
        except Exception:
            return []

    def save_task_instructions(self, task_id: str, instructions: List[str]) -> None:
        """保存该任务的标注结果列表"""
        path = self.get_instructions_path(task_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"instructions": instructions}, f, ensure_ascii=False, indent=2)


