"""Minimal openpi.training.config stub for pi0 runner probe tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MockTrainConfig:
    name: str


_CONFIGS: dict[str, MockTrainConfig] = {
    "pi0_mock": MockTrainConfig(name="pi0_mock"),
    "pi05_libero": MockTrainConfig(name="pi05_libero"),
}


def get_config(name: str) -> MockTrainConfig:
    if name not in _CONFIGS:
        raise KeyError(name)
    return _CONFIGS[name]
