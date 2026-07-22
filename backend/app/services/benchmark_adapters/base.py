from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.schemas.evaluation import EvaluateAsyncRequest
from app.services.evaluation.evaluation_request_resolver import NormalizedEvaluateRequest

EVAL_JOB_ID_PATTERN = re.compile(r"^eval_\d{8}_\d{6}_[a-f0-9]{4}$")
_CT_EVAL_JOB_SUFFIX = r"(?:\d{8}_\d{6}_[a-f0-9]{4}|[a-z0-9_]+)"
CT_EVAL_JOB_ID_PATTERN = re.compile(rf"^ct_eval_{_CT_EVAL_JOB_SUFFIX}$")


def is_ct_eval_job_id(eval_job_id: str) -> bool:
    return CT_EVAL_JOB_ID_PATTERN.match((eval_job_id or "").strip()) is not None


def is_platform_eval_job_id(eval_job_id: str) -> bool:
    return EVAL_JOB_ID_PATTERN.match((eval_job_id or "").strip()) is not None


@dataclass
class BenchmarkCapabilities:
    task_template_id: str
    task_type: str
    task_family: str
    simulator_type: str
    supported_evaluation_modes: list[str]
    supported_policy_types: list[str] = field(default_factory=list)
    supports_checkpoint: bool = False
    supports_policy_evaluation: bool = False
    supports_episode_stability: bool = False
    supports_train_model_evaluation: bool = False
    supports_video: bool = False
    result_artifact: Optional[str] = None
    description: str = ""
    extensions: dict[str, Any] = field(default_factory=dict)

    def to_api_dict(self) -> dict[str, Any]:
        payload = {
            "taskType": self.task_type,
            "taskTemplateId": self.task_template_id,
            "supportedModes": list(self.supported_evaluation_modes),
            "supportedPolicyTypes": list(self.supported_policy_types),
            "supportsCheckpoint": self.supports_checkpoint,
            "supportsPolicyEvaluation": self.supports_policy_evaluation,
            "supportsEpisodeStability": self.supports_episode_stability,
            "supportsTrainModelEvaluation": self.supports_train_model_evaluation,
            "supportsVideo": self.supports_video,
            "resultArtifact": self.result_artifact,
            "description": self.description,
        }
        if self.extensions:
            payload.update(self.extensions)
        return payload


class BenchmarkTaskAdapter(ABC):
    """标准任务模板评测适配器（仅抽象评测能力）。"""

    task_template_id: str
    task_type: str
    task_family: str
    simulator_type: str
    supported_evaluation_modes: list[str]

    @abstractmethod
    def get_capabilities(self) -> BenchmarkCapabilities:
        """返回该任务模板支持的评测能力。"""

    @abstractmethod
    def normalize_request(self, request: EvaluateAsyncRequest) -> NormalizedEvaluateRequest:
        """校验并归一化评测请求。"""

    @abstractmethod
    def start_evaluation(
        self,
        request: EvaluateAsyncRequest,
        normalized: NormalizedEvaluateRequest,
    ) -> dict[str, Any]:
        """启动评测，返回统一 EvaluateAsyncResponse 结构 dict。"""

    @abstractmethod
    def get_status(self, eval_job_id: str) -> dict[str, Any]:
        """读取评测状态（EvaluationJobStatusResponse 结构）。"""

    @abstractmethod
    def get_result(self, eval_job_id: str) -> dict[str, Any]:
        """读取评测结果。"""

    @abstractmethod
    def get_log(self, eval_job_id: str) -> str:
        """读取评测日志 tail。"""

    @abstractmethod
    def get_video(self, eval_job_id: str, episode_id: Optional[int] = None) -> Optional[Path]:
        """返回评测视频路径。"""

    def recognizes_eval_job_id(self, eval_job_id: str) -> bool:
        return False
