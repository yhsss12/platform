from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class EvaluationJob:
    """平台级评测 job 句柄。"""

    eval_job_id: str
    task_type: str
    evaluation_mode: str
    job_root: Path
    status: str = "queued"
    source_job_ids: list[str] = field(default_factory=list)


@dataclass
class EvaluationStatus:
    """统一评测状态（API 响应体）。"""

    eval_job_id: str
    task_type: str
    evaluation_mode: str
    status: str
    phase: Optional[str] = None
    progress: Optional[float] = None
    current_episode: Optional[int] = None
    total_episodes: Optional[int] = None
    message: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    updated_at: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evalJobId": self.eval_job_id,
            "taskType": self.task_type,
            "evaluationMode": self.evaluation_mode,
            "status": self.status,
            "phase": self.phase,
            "progress": self.progress,
            "currentEpisode": self.current_episode,
            "totalEpisodes": self.total_episodes,
            "message": self.message,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "updatedAt": self.updated_at,
        }


class BaseEvaluationAdapter(ABC):
    """任务评测适配器基类。"""

    task_type: str
    supported_modes: list[str]

    @abstractmethod
    def validate_request(self, request: Any) -> None:
        """校验 taskType + evaluationMode + 任务专有参数。"""

    @abstractmethod
    def start_async(
        self,
        request: Any,
        *,
        eval_job_id: str,
        job_root: Path,
    ) -> EvaluationJob:
        """启动异步评测，返回平台 eval job。"""

    @abstractmethod
    def get_status(self, eval_job_id: str, job_root: Path) -> EvaluationStatus:
        """读取统一状态。"""

    @abstractmethod
    def get_result(self, eval_job_id: str, job_root: Path) -> dict[str, Any]:
        """返回 aggregate result；未完成时返回部分结果或空 dict。"""

    @abstractmethod
    def get_log(self, eval_job_id: str, job_root: Path) -> str:
        """返回 eval.log tail。"""

    @abstractmethod
    def get_video_path(
        self,
        eval_job_id: str,
        job_root: Path,
        *,
        episode: Optional[int] = None,
    ) -> Optional[Path]:
        """返回评测视频路径；多 episode 时 episode 指定索引。"""

    def supports_mode(self, evaluation_mode: str) -> bool:
        return evaluation_mode in self.supported_modes


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
