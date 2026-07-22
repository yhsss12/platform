"""PhyGen: task-agnostic residual-field PINN with TaskAdapter extensions."""

from phygen.adapters.registry import get_adapter
from phygen.core.trainer import PhyGenTrainer

__all__ = ["get_adapter", "PhyGenTrainer"]
