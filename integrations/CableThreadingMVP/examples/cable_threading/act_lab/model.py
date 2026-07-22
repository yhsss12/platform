from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyVisionEncoder(nn.Module):
    def __init__(self, num_cameras: int, out_dim: int = 128) -> None:
        super().__init__()
        self.num_cameras = num_cameras
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(64 * max(num_cameras, 1), out_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, N, C, H, W)
        if images.numel() == 0:
            batch = images.shape[0]
            return images.new_zeros((batch, self.proj.out_features))
        b, n_cam, c, h, w = images.shape
        flat = images.reshape(b * n_cam, c, h, w)
        feats = self.cnn(flat).flatten(1)
        feats = feats.view(b, n_cam, -1).reshape(b, -1)
        return self.proj(feats)


class ActPolicy(nn.Module):
    """Image + proprio ACT-style CVAE policy predicting action chunks."""

    def __init__(
        self,
        *,
        action_dim: int,
        chunk_size: int,
        state_dim: int,
        num_cameras: int,
        hidden_dim: int = 512,
        latent_dim: int = 32,
        kl_weight: float = 10.0,
        enc_layers: int = 4,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.kl_weight = kl_weight
        self.latent_dim = latent_dim

        vision_dim = hidden_dim // 2
        self.vision = TinyVisionEncoder(num_cameras, out_dim=vision_dim)
        self.proprio = nn.Sequential(
            nn.Linear(max(state_dim, 1), hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        fused_dim = vision_dim + hidden_dim // 2
        self.fuse = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(enc_layers, 1))
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, latent_dim)
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )

    def encode(self, images: torch.Tensor, proprio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        vision_feat = self.vision(images)
        proprio_feat = self.proprio(proprio)
        fused = self.fuse(torch.cat([vision_feat, proprio_feat], dim=-1))
        token = fused.unsqueeze(1)
        encoded = self.encoder(token).squeeze(1)
        return self.latent_mu(encoded), self.latent_logvar(encoded)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mu, logvar = self.encode(batch["images"], batch["proprio"])
        z = self.reparameterize(mu, logvar)
        encoded = self.encoder(
            self.fuse(
                torch.cat(
                    [
                        self.vision(batch["images"]),
                        self.proprio(batch["proprio"]),
                    ],
                    dim=-1,
                )
            ).unsqueeze(1)
        ).squeeze(1)
        pred = self.action_head(torch.cat([encoded, z], dim=-1))
        pred = pred.view(-1, self.chunk_size, self.action_dim)
        target = batch["actions"]
        mask = (1.0 - batch["is_pad"]).unsqueeze(-1)
        recon = F.mse_loss(pred * mask, target * mask, reduction="sum") / mask.sum().clamp(min=1.0)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + self.kl_weight * kl
