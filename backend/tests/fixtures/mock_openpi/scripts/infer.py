#!/usr/bin/env python3
"""Mock openpi inference for platform pi0 eval smoke tests."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--obs-json", required=True, help="JSON file with observation dict")
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()

    obs = json.loads(Path(args.obs_json).read_text(encoding="utf-8"))
    # deterministic small action from proprio if present
    action = np.zeros((args.action_dim,), dtype=np.float32)
    for key, value in obs.items():
        if "pos" in key.lower() or "eef" in key.lower():
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size > 0:
                action[: min(args.action_dim, arr.size)] = np.tanh(arr[: args.action_dim]) * 0.05
                break

    chunk = [action.tolist() for _ in range(max(1, args.horizon))]
    print(json.dumps({"actions": chunk}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
