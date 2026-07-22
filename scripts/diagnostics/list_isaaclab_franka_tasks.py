#!/usr/bin/env python3
"""List Franka, Panda, stack, lift, and pick tasks from the Isaac Lab registry."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ISAACLAB_ROOT = Path(__file__).resolve().parent.parent / "third_party" / "IsaacLab"
SCRIPTS_DIR = PROJECT_ROOT / "backend" / "integrations" / "isaac_lab" / "scripts"
KEYWORDS = re.compile(r"franka|panda|stack|lift|pick", re.I)
TASK_ID_RE = re.compile(r'id="(Isaac-[^"]+)"')
ENTRY_RE = re.compile(r'entry_point\s*=\s*f?"([^"]+)"')
ENV_CFG_RE = re.compile(r'env_cfg_entry_point\s*:\s*f?"([^"]+)"')

SCRIPT_HINTS: dict[str, list[str]] = {}
for script in SCRIPTS_DIR.glob("*.py"):
    try:
        text = script.read_text(encoding="utf-8", errors="replace")
    except OSError:
        continue
    for match in re.finditer(r"Isaac-[A-Za-z0-9_-]+", text):
        SCRIPT_HINTS.setdefault(match.group(0), []).append(script.name)


def _task_type(task_id: str) -> str:
    lowered = task_id.lower()
    if "stack" in lowered:
        return "stacking"
    if "lift" in lowered:
        return "lifting"
    if "pick" in lowered or "place" in lowered:
        return "pick_place"
    if "reach" in lowered:
        return "reach"
    return "manipulation"


def _robot(task_id: str) -> str:
    lowered = task_id.lower()
    if "franka" in lowered or "panda" in lowered:
        return "Franka Panda"
    if "ur10" in lowered:
        return "UR10"
    if "galbot" in lowered:
        return "Galbot"
    if "gr1" in lowered:
        return "GR1T2"
    if "openarm" in lowered:
        return "OpenArm"
    return "unknown"


def _scan_source_registry(isaaclab_root: Path) -> list[dict]:
    roots = [
        isaaclab_root / "source" / "isaaclab_tasks",
        isaaclab_root / "source" / "isaaclab_mimic",
    ]
    rows: dict[str, dict] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for init_file in root.rglob("__init__.py"):
            try:
                text = init_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for task_id in TASK_ID_RE.findall(text):
                if not KEYWORDS.search(task_id):
                    continue
                entry = ENTRY_RE.search(text) or ENV_CFG_RE.search(text)
                scripts = sorted(set(SCRIPT_HINTS.get(task_id, [])))
                lowered_scripts = " ".join(scripts).lower()
                rows[task_id] = {
                    "task_id": task_id,
                    "env_type": entry.group(1) if entry else None,
                    "robot": _robot(task_id),
                    "task_type": _task_type(task_id),
                    "supports_render": True,
                    "has_mimic_demo": "mimic" in task_id.lower() or "mimic" in lowered_scripts,
                    "has_teleop_demo": "teleop" in lowered_scripts or "record_demos" in lowered_scripts,
                    "has_expert_policy": "expert_policy" in lowered_scripts or "scripted_expert" in lowered_scripts,
                    "has_policy_eval": "policy_eval" in lowered_scripts,
                    "has_replay": "replay" in lowered_scripts,
                    "local_scripts": scripts,
                    "can_generate_trajectory": bool(scripts) or "mimic" in task_id.lower(),
                    "can_generate_video": bool(scripts),
                    "can_generate_metrics": bool(scripts) or "mimic" in task_id.lower(),
                }
    return sorted(rows.values(), key=lambda row: row["task_id"])


def main() -> int:
    parser = argparse.ArgumentParser(description="List local Isaac Lab Franka-related tasks")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--isaaclab-root", default=str(ISAACLAB_ROOT))
    args = parser.parse_args()

    isaaclab_root = Path(args.isaaclab_root).expanduser()
    tasks = _scan_source_registry(isaaclab_root)
    payload = {
        "isaaclab_root": str(isaaclab_root),
        "count": len(tasks),
        "recommended_phase1_task_id": "Isaac-Stack-Cube-Franka-IK-Rel-v0",
        "recommended_mimic_task_id": "Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0",
        "tasks": tasks,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
