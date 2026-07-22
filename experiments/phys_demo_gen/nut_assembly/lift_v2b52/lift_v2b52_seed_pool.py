"""V2-B5.2 CEM seed pool from B5.1 top candidates。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lift_v2b52_refiner import LiftV2B52Params, lift_v2b52_from_b51


def _nut_z(rec: dict[str, Any]) -> float:
    return float(rec.get("nut_z_lift_delta", rec.get("nut_lift_phase_delta", 0.0)))


def load_b51_records(jsonl_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def build_seed_pool(
    records: list[dict[str, Any]],
    *,
    top_k: int = 20,
) -> tuple[list[LiftV2B52Params], dict[str, Any]]:
    search_records = [r for r in records if r.get("lift_v2b51_params") or r.get("lift_v2b52_params")]
    seeds: list[LiftV2B52Params] = []
    seen: set[str] = set()
    meta: dict[str, Any] = {"categories": {}, "seed_sources": []}

    def _add_category(name: str, ranked: list[dict[str, Any]]) -> None:
        added: list[int] = []
        for rec in ranked[:top_k]:
            raw = rec.get("lift_v2b52_params") or rec.get("lift_v2b51_params")
            if not raw:
                continue
            key = json.dumps(raw, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            params = lift_v2b52_from_b51(raw)
            seeds.append(params)
            idx = rec.get("search_index")
            added.append(int(idx) if idx is not None else -1)
            meta["seed_sources"].append(
                {
                    "category": name,
                    "search_index": rec.get("search_index"),
                    "nut_z_lift_delta": _nut_z(rec),
                    "bilateral_contact_steps": rec.get("bilateral_contact_steps"),
                    "right_finger_contact_count": rec.get("right_finger_contact_count"),
                }
            )
        meta["categories"][name] = {"requested": top_k, "added": len(added), "indices": added}

    _add_category("nut_z_lift_delta", sorted(search_records, key=_nut_z, reverse=True))
    _add_category(
        "bilateral_contact_steps",
        sorted(search_records, key=lambda r: int(r.get("bilateral_contact_steps", 0)), reverse=True),
    )
    _add_category(
        "nut_eef_coupling_ratio",
        sorted(search_records, key=lambda r: float(r.get("nut_eef_coupling_ratio", -999)), reverse=True),
    )
    _add_category("low_nut_xy_slip", sorted(search_records, key=lambda r: float(r.get("nut_xy_slip", 999))))

    meta["unique_seed_count"] = len(seeds)
    return seeds, meta
