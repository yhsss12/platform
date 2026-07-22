"""V1-A / V1-B / V1-C：最小 PyTorch Residual Energy Model（非 PINN 最终版）。"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from residual_dataset import FAILURE_TYPES, FEATURE_NAMES_V1B, FEATURE_NAMES_V1C, OUTCOME_TYPES

ENERGY_WEIGHTS = torch.tensor([3.0, 3.0, 2.0, 2.0, 0.2], dtype=torch.float32)


class ResidualEnergyModel(nn.Module):
    """MLP：物理特征 → 归一化 energy 分量 + success / failure_type / outcome / grasp / lift。"""

    def __init__(
        self,
        input_dim: int | None = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        num_failure_types: int = len(FAILURE_TYPES),
        num_outcome_types: int = len(OUTCOME_TYPES),
        activation: str = "gelu",
        predict_outcome: bool = True,
        predict_grasp_lift: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim or len(FEATURE_NAMES_V1B)
        self.hidden_dim = hidden_dim
        self.num_failure_types = num_failure_types
        self.num_outcome_types = num_outcome_types
        self.predict_outcome = predict_outcome
        self.predict_grasp_lift = predict_grasp_lift

        act_cls = nn.GELU if activation == "gelu" else nn.ReLU
        layers: list[nn.Module] = []
        in_dim = self.input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        self.head_components = nn.Linear(hidden_dim, 5)
        self.head_total = nn.Linear(hidden_dim, 1)
        self.head_success = nn.Linear(hidden_dim, 1)
        self.head_failure = nn.Linear(hidden_dim, num_failure_types)
        self.head_outcome = nn.Linear(hidden_dim, num_outcome_types) if predict_outcome else None
        self.head_grasp_success = nn.Linear(hidden_dim, 1) if predict_grasp_lift else None
        self.head_lift_success = nn.Linear(hidden_dim, 1) if predict_grasp_lift else None

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.backbone(features)
        components = torch.nn.functional.softplus(self.head_components(h))
        total_direct = torch.nn.functional.softplus(self.head_total(h)).squeeze(-1)
        success_logit = self.head_success(h).squeeze(-1)
        failure_logits = self.head_failure(h)
        total_consistent = torch.sum(components * ENERGY_WEIGHTS.to(components.device), dim=-1)
        out: dict[str, torch.Tensor] = {
            "E_components": components,
            "E_total_direct": total_direct,
            "E_total_consistent": total_consistent,
            "E_total": total_direct,
            "success_logit": success_logit,
            "failure_type_logits": failure_logits,
        }
        if self.head_outcome is not None:
            out["outcome_logits"] = self.head_outcome(h)
        if self.head_grasp_success is not None:
            out["grasp_success_logit"] = self.head_grasp_success(h).squeeze(-1)
        if self.head_lift_success is not None:
            out["lift_success_logit"] = self.head_lift_success(h).squeeze(-1)
        return out

    def predict_dict(self, features: torch.Tensor) -> dict[str, Any]:
        self.eval()
        with torch.no_grad():
            out = self.forward(features)
        return {k: v.cpu().numpy() for k, v in out.items()}
