"""Minimal ACT (Action Chunking Transformer) training lab for platform integration."""

from .config import ActLabConfig
from .trainer import train_act_policy

__all__ = ["ActLabConfig", "train_act_policy"]
