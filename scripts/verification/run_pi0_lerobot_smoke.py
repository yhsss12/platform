#!/usr/bin/env python3
"""Phase D: pi0 LeRobot joint-space training smoke (standalone, no platform DB)."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
sys.path.insert(0, str(BACKEND_ROOT))

from app.services.pi0_lerobot_smoke_runner import (
    DEFAULT_SMOKE_DATASET,
    assess_pi0_lerobot_training_capability,
    run_pi0_lerobot_training_smoke,
)
from app.services.policy_schema_resolver import PI0_JOINT_SPACE_ENABLED


def main() -> int:
    dataset = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else DEFAULT_SMOKE_DATASET
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "runs" / "pi0_lerobot_smoke" / f"smoke_{ts}"

    print("=== Phase D: pi0 LeRobot Training Smoke ===")
    print("dataset:", dataset)
    print("output_dir:", output_dir)
    print("PI0_JOINT_SPACE_ENABLED:", PI0_JOINT_SPACE_ENABLED)

    result = run_pi0_lerobot_training_smoke(
        dataset_path=dataset,
        output_dir=output_dir,
        epochs=1,
        batch_size=2,
        max_steps=10,
    )
    capability = assess_pi0_lerobot_training_capability(
        dataset_path=dataset,
        smoke_success=result.get("status") == "completed",
    )

    print("\n=== smoke result ===")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n=== capability ===")
    print(json.dumps(capability, ensure_ascii=False, indent=2))

    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
