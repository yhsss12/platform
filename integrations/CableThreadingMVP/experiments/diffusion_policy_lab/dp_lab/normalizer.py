from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LinearNormalizer:
    """Per-dimension linear map to approximately [-1, 1]."""

    scale: np.ndarray
    offset: np.ndarray

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.scale + self.offset

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.offset) / np.clip(self.scale, 1e-8, None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale.tolist(),
            "offset": self.offset.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LinearNormalizer":
        return cls(
            scale=np.asarray(payload["scale"], dtype=np.float32),
            offset=np.asarray(payload["offset"], dtype=np.float32),
        )

    @classmethod
    def fit(cls, data: np.ndarray, output_min: float = -1.0, output_max: float = 1.0) -> "LinearNormalizer":
        arr = np.asarray(data, dtype=np.float64)
        data_min = arr.min(axis=0)
        data_max = arr.max(axis=0)
        range_ = np.clip(data_max - data_min, 1e-6, None)
        scale = (output_max - output_min) / range_
        offset = output_min - scale * data_min
        return cls(scale=scale.astype(np.float32), offset=offset.astype(np.float32))


@dataclass
class DatasetStats:
    action: LinearNormalizer
    low_dim: LinearNormalizer

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.to_dict(),
            "low_dim": self.low_dim.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetStats":
        return cls(
            action=LinearNormalizer.from_dict(payload["action"]),
            low_dim=LinearNormalizer.from_dict(payload["low_dim"]),
        )
