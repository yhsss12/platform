from __future__ import annotations

from phygen.adapters.base_adapter import BasePhyGenAdapter
from phygen.adapters.mimicgen.coffee_preparation_adapter import CoffeePreparationAdapter
from phygen.adapters.mimicgen.stack_three_adapter import StackThreeAdapter

ADAPTER_REGISTRY: dict[str, type[BasePhyGenAdapter]] = {
    "coffee_preparation": CoffeePreparationAdapter,
    "stack_three": StackThreeAdapter,
}


def get_adapter(task: str) -> BasePhyGenAdapter:
    if task not in ADAPTER_REGISTRY:
        known = ", ".join(sorted(ADAPTER_REGISTRY))
        raise ValueError(f"Unsupported task '{task}'. Available adapters: {known}")
    return ADAPTER_REGISTRY[task]()
