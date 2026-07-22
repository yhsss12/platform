#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB_ROOT))

from dp_lab.config import DpLabConfig
from dp_lab.dataset import inspect_hdf5


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect cable threading HDF5 for DP lab")
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    info = inspect_hdf5(Path(args.dataset))
    cfg = DpLabConfig()
    required_images = set(cfg.image_keys)
    present_images = set(info["obs_shapes"].keys()) & required_images
    missing = required_images - present_images

    print(json.dumps(info, ensure_ascii=False, indent=2))
    print("\nDP readiness:")
    print(f"  required image keys: {sorted(required_images)}")
    print(f"  present image keys:  {sorted(present_images)}")
    if missing:
        print(f"  MISSING: {sorted(missing)}")
        return 1
    print("  status: OK for DP lab")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
