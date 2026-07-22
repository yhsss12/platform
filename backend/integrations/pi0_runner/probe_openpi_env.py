#!/usr/bin/env python3
"""Verify openpi import and training config access (run with OPENPI_PYTHON)."""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_openpi_path() -> Path:
    root_raw = os.environ.get("OPENPI_ROOT", "").strip()
    if not root_raw:
        raise SystemExit("OPENPI_ROOT is not set")
    root = Path(root_raw).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"OPENPI_ROOT does not exist: {root}")

    candidates = [
        root / "src",
        root / "packages" / "openpi-client" / "src",
    ]
    for candidate in candidates:
        if candidate.is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    return root


def main() -> int:
    root = _bootstrap_openpi_path()
    try:
        from openpi.training import config as training_config  # noqa: WPS433
    except Exception as exc:  # pragma: no cover - exercised via mock in tests
        raise SystemExit(f"failed to import openpi.training.config: {exc}") from exc

    if not hasattr(training_config, "get_config"):
        raise SystemExit("openpi.training.config missing get_config")

    configs: list[str] = []
    if hasattr(training_config, "_CONFIGS") and isinstance(training_config._CONFIGS, dict):
        configs = list(training_config._CONFIGS.keys())
    elif hasattr(training_config, "get_config"):
        base_name = os.environ.get("OPENPI_BASE_CONFIG", "").strip()
        if base_name:
            training_config.get_config(base_name)

    if hasattr(training_config, "_CONFIGS") and isinstance(training_config._CONFIGS, dict) and not configs:
        raise SystemExit("openpi training configs registry is empty")

    print(f"OPENPI_PROBE_OK root={root} configs={len(configs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
