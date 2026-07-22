#!/usr/bin/env python3
"""探测训练节点健康状态。

用法:
  cd backend
  python tools/diagnostics/check_training_node.py l20-172-18-0-73
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.env_loader import ensure_dotenv_loaded  # noqa: E402

ensure_dotenv_loaded(verbose=True)

from app.services.training_node_service import format_node_probe_report, probe_training_node  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    node_id = args[0] if args else "l20-172-18-0-73"
    print(format_node_probe_report(node_id))
    probe = probe_training_node(node_id, refresh=True)
    status = probe.get("status")
    if status in {"available", "placeholder"}:
        return 0
    if status == "busy":
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
