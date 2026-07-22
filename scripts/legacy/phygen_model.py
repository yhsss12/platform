#!/usr/bin/env python3
"""Legacy PhyGen CLI shim; prefer ``python scripts/train_phygen.py``."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from phygen.adapters.base_adapter import (  # noqa: E402
    BasePhyGenAdapter,
    TaskSpec,
    _external_load_failed_contexts,
    _external_make_candidate,
    demo_sort_key,
    load_records,
    load_success_demo_keys,
)
from phygen.adapters.mimicgen.stack_three_adapter import StackThreeAdapter  # noqa: E402
from phygen.adapters.registry import ADAPTER_REGISTRY, get_adapter  # noqa: E402
from phygen.core.residual_field_model import (  # noqa: E402
    FeatureLayout,
    RepairParameterResidualFieldPINN,
    build_mlp_selector,
)
from phygen.core.selector import (  # noqa: E402
    attach_selector_scores,
    build_candidate_plan,
    offline_selector_report,
    predict,
    predict_with_details,
    unique_union,
)
from phygen.core.trainer import PhyGenTrainer, TrainingResult  # noqa: E402

__all__ = [
    "ADAPTER_REGISTRY",
    "BasePhyGenAdapter",
    "FeatureLayout",
    "PhyGenTrainer",
    "RepairParameterResidualFieldPINN",
    "StackThreeAdapter",
    "TaskSpec",
    "TrainingResult",
    "attach_selector_scores",
    "build_candidate_plan",
    "build_mlp_selector",
    "demo_sort_key",
    "get_adapter",
    "load_records",
    "load_success_demo_keys",
    "offline_selector_report",
    "predict",
    "predict_with_details",
    "unique_union",
]


def main() -> None:
    from scripts.train_phygen import main as train_main

    train_main()


if __name__ == "__main__":
    main()
