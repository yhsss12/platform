from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        emb = math.log(10000) / max(half - 1, 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ConditionalUnet1D(nn.Module):
    """Minimal 1D UNet for action-sequence diffusion."""

    def __init__(self, action_dim: int, horizon: int, cond_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.Mish(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        in_dim = action_dim + cond_dim + hidden_dim
        self.down1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, hidden_dim))
        self.down2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, hidden_dim))
        self.mid = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, hidden_dim))
        self.up1 = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, hidden_dim))
        self.up2 = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Mish(), nn.Linear(hidden_dim, action_dim))

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, global_cond: torch.Tensor) -> torch.Tensor:
        # sample: (B, T, A)
        b, t, a = sample.shape
        time_emb = self.time_mlp(timestep)
        cond = torch.cat([global_cond, time_emb], dim=-1)
        cond = cond[:, None, :].expand(b, t, -1)
        x = torch.cat([sample, cond], dim=-1)

        h1 = self.down1(x)
        h2 = self.down2(h1)
        mid = self.mid(h2)
        u1 = self.up1(torch.cat([mid, h2], dim=-1))
        out = self.up2(torch.cat([u1, h1], dim=-1))
        return out


class TinyVisionEncoder(nn.Module):
    """轻量 CNN，用于 smoke / CPU debug，无需下载预训练权重。"""

    def __init__(self, num_cameras: int, out_dim: int = 64) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64 * num_cameras, out_dim)
        self.num_cameras = num_cameras

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, t, n_cam, c, h, w = images.shape
        flat = images.reshape(b * t * n_cam, c, h, w)
        feats = self.cnn(flat).flatten(1)
        feats = feats.view(b, t, n_cam, -1).reshape(b, t, -1)
        return self.proj(feats)


class ResNetVisionEncoder(nn.Module):
    def __init__(self, num_cameras: int, out_dim: int = 128) -> None:
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        modules = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.proj = nn.Linear(512 * num_cameras, out_dim)
        self.num_cameras = num_cameras

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, t, n_cam, c, h, w = images.shape
        flat = images.reshape(b * t * n_cam, c, h, w)
        feats = self.backbone(flat).flatten(1)
        feats = feats.view(b, t, n_cam, -1).reshape(b, t, -1)
        return self.proj(feats)


def build_vision_encoder(name: str, num_cameras: int, out_dim: int) -> nn.Module:
    if name == "tiny_cnn":
        return TinyVisionEncoder(num_cameras, out_dim=out_dim)
    if name == "resnet18":
        return ResNetVisionEncoder(num_cameras, out_dim=out_dim)
    raise ValueError(f"unsupported vision_encoder: {name}")


class ConditionalDiffusionPolicy(nn.Module):
    def __init__(
        self,
        *,
        action_dim: int,
        horizon: int,
        low_dim_dim: int,
        n_obs_steps: int,
        num_cameras: int,
        image_size: int,
        num_diffusion_steps: int = 20,
        vision_encoder: str = "resnet18",
        vision_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.num_diffusion_steps = num_diffusion_steps
        self.vision_encoder_name = vision_encoder

        self.vision = build_vision_encoder(vision_encoder, num_cameras, out_dim=vision_dim)
        self.low_dim_proj = nn.Linear(low_dim_dim, vision_dim)
        cond_dim = vision_dim * 2
        self.unet = ConditionalUnet1D(action_dim, horizon, cond_dim, hidden_dim=hidden_dim)

        betas = torch.linspace(1e-4, 0.02, num_diffusion_steps)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))

    def encode_obs(self, images: torch.Tensor, low_dim: torch.Tensor) -> torch.Tensor:
        vis = self.vision(images).mean(dim=1)
        low = self.low_dim_proj(low_dim).mean(dim=1)
        return torch.cat([vis, low], dim=-1)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        actions = batch["actions"]
        cond = self.encode_obs(batch["images"], batch["low_dim"])
        b = actions.shape[0]
        t = torch.randint(0, self.num_diffusion_steps, (b,), device=actions.device)
        noise = torch.randn_like(actions)
        sqrt_alpha = self.sqrt_alpha_cumprod[t][:, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[t][:, None, None]
        noisy = sqrt_alpha * actions + sqrt_one_minus * noise
        pred = self.unet(noisy, t, cond)
        return torch.nn.functional.mse_loss(pred, noise)

    @torch.no_grad()
    def predict_actions(self, batch: dict[str, torch.Tensor], num_inference_steps: int | None = None) -> torch.Tensor:
        steps = num_inference_steps or self.num_diffusion_steps
        cond = self.encode_obs(batch["images"], batch["low_dim"])
        b = batch["images"].shape[0]
        sample = torch.randn(b, self.horizon, self.action_dim, device=batch["images"].device)
        for step in reversed(range(steps)):
            t = torch.full((b,), step, device=sample.device, dtype=torch.long)
            pred_noise = self.unet(sample, t, cond)
            alpha = self.alphas[step]
            alpha_bar = self.alpha_cumprod[step]
            beta = self.betas[step]
            if step > 0:
                noise = torch.randn_like(sample)
            else:
                noise = torch.zeros_like(sample)
            sample = (1.0 / torch.sqrt(alpha)) * (
                sample - (beta / torch.sqrt(1.0 - alpha_bar)) * pred_noise
            ) + torch.sqrt(beta) * noise
        return sample
