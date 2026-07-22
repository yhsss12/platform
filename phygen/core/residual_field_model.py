from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


PINN_BETA = 8.0
PINN_SUCCESS_ENERGY_THRESHOLD = 0.25
PINN_FAILURE_ENERGY_FLOOR = 0.12
PINN_SUCCESS_CORRECTION_SCALE = 0.25


@dataclass(frozen=True)
class FeatureLayout:
    context_dim: int
    theta_disc_dim: int
    theta_cont_dim: int
    boundary_dim: int = 1

    @property
    def theta_disc_start(self) -> int:
        return self.context_dim

    @property
    def theta_disc_end(self) -> int:
        return self.theta_disc_start + self.theta_disc_dim

    @property
    def theta_cont_start(self) -> int:
        return self.theta_disc_end

    @property
    def theta_cont_end(self) -> int:
        return self.theta_cont_start + self.theta_cont_dim

    @property
    def boundary_index(self) -> int:
        return self.theta_cont_end

    @property
    def input_dim(self) -> int:
        return self.context_dim + self.theta_disc_dim + self.theta_cont_dim + self.boundary_dim


class RepairParameterResidualFieldPINNProtocol(Protocol):
    def __call__(self, inp: Any) -> dict[str, Any]: ...


class RepairParameterResidualFieldPINN:
    """Factory wrapper for the torch model; avoids importing torch at module import time."""

    @staticmethod
    def build(layout: FeatureLayout, component_dim: int):
        import torch
        import torch.nn as nn

        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input_dim = layout.input_dim
                self.context_dim = layout.context_dim
                self.theta_disc_dim = layout.theta_disc_dim
                self.theta_cont_dim = layout.theta_cont_dim
                self.boundary_dim = layout.boundary_dim
                self.model_family = "repair_parameter_residual_field_pinn"

                self.discrete_encoder = nn.Sequential(
                    nn.Linear(self.theta_disc_dim, 32),
                    nn.SiLU(),
                    nn.LayerNorm(32),
                    nn.Linear(32, 32),
                    nn.SiLU(),
                )
                self.backbone = nn.Sequential(
                    nn.Linear(self.context_dim + 32 + self.theta_cont_dim + self.boundary_dim, 128),
                    nn.SiLU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 96),
                    nn.SiLU(),
                    nn.LayerNorm(96),
                )
                self.head_value = nn.Linear(96, 1)
                self.head_success_correction = nn.Linear(96, 1)
                self.head_components = nn.Linear(96, component_dim)

            def split(self, inp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                context = inp[:, : layout.theta_disc_start]
                theta_disc = inp[:, layout.theta_disc_start : layout.theta_disc_end]
                theta_cont = inp[:, layout.theta_cont_start : layout.theta_cont_end]
                boundary = inp[:, layout.boundary_index : layout.boundary_index + 1]
                return context, theta_disc, theta_cont, boundary

            def forward(self, inp: torch.Tensor) -> dict[str, torch.Tensor]:
                context, theta_disc, theta_cont, boundary = self.split(inp)
                theta_disc_emb = self.discrete_encoder(theta_disc)
                h = self.backbone(torch.cat([context, theta_disc_emb, theta_cont, boundary], dim=1))
                value = torch.sigmoid(self.head_value(h))
                correction = PINN_SUCCESS_CORRECTION_SCALE * torch.tanh(self.head_success_correction(h))
                success_logit = PINN_BETA * (PINN_SUCCESS_ENERGY_THRESHOLD - value) + correction
                components = torch.sigmoid(self.head_components(h))
                return {
                    "energy_success": torch.cat([value, success_logit], dim=1),
                    "components": components,
                    "residual_value": value,
                    "success_correction": correction,
                }

        return _Model()


def build_mlp_selector(input_dim: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.SiLU(),
        nn.LayerNorm(128),
        nn.Linear(128, 96),
        nn.SiLU(),
        nn.LayerNorm(96),
        nn.Linear(96, 2),
    )
