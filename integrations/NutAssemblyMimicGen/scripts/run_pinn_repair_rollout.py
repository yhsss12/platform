#!/usr/bin/env python3
"""Run a single PINN repair rollout in cable-threading-mvp subprocess."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

_INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
if str(_INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION_ROOT))

from utils.robosuite_rollout import rollout_episodes


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="NutAssembly PINN repair rollout worker")
    parser.add_argument("--env-name", default="NutAssembly_D0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--extra-xy-bias", default="")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    extra_bias = None
    if args.extra_xy_bias.strip():
        extra_bias = np.asarray(json.loads(args.extra_xy_bias), dtype=np.float64)

    output_json = Path(args.output_json)
    try:
        rollout_result = rollout_episodes(
            env_name=args.env_name,
            episodes=1,
            seed=args.seed,
            horizon=args.horizon,
            render_video=False,
            video_path=None,
            debug_log_path=None,
            extra_xy_bias=extra_bias,
        )
        episodes = rollout_result.get("episodes") or []
        if not episodes:
            payload = {"ok": False, "error": "empty_rollout"}
        else:
            ep = episodes[0]
            metadata = ep.get("metadata") or {}
            payload = _jsonify(
                {
                    "ok": True,
                    "episode": ep,
                    "metadata": metadata,
                    "runtimeEnvName": rollout_result.get("runtimeEnvName"),
                }
            )
        output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0 if payload.get("ok") else 1
    except Exception as exc:
        output_json.write_text(
            json.dumps({"ok": False, "error": str(exc), "traceback": traceback.format_exc()}, indent=2),
            encoding="utf-8",
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
