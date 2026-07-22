"""V2-B5.3 seed pool: contact-rich + weak-lift + transport + cross hybrids."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from lift_v2b53_objective import DEMO3_BASELINE_PEG_XY, nut_z_delta, weak_lift_positive
from lift_v2b53_refiner import LiftV2B53Params, lift_v2b53_from_b52, lift_v2b53_params_from_dict

TOP_K = 20


def _nut_z(rec: dict[str, Any]) -> float:
    return nut_z_delta(rec)


def _min_peg(rec: dict[str, Any]) -> float:
    return float(rec.get("min_nut_peg_xy", rec.get("final_nut_peg_xy", DEMO3_BASELINE_PEG_XY)))


def _params_dict(rec: dict[str, Any]) -> dict[str, Any] | None:
    raw = rec.get("lift_v2b53_params") or rec.get("lift_v2b52_params") or rec.get("lift_v2b51_params")
    return dict(raw) if raw else None


def load_rollout_records(*paths: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def _blend_params(a: dict[str, Any], b: dict[str, Any], rng: random.Random, alpha: float | None = None) -> LiftV2B53Params:
    t = alpha if alpha is not None else rng.uniform(0.35, 0.65)
    merged: dict[str, Any] = {}
    keys = set(a.keys()) | set(b.keys())
    for key in keys:
        va, vb = a.get(key), b.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if isinstance(va, int) and isinstance(vb, int):
                merged[key] = int(round((1 - t) * va + t * vb))
            else:
                merged[key] = float((1 - t) * float(va) + t * float(vb))
        elif isinstance(va, str):
            merged[key] = va if rng.random() < 0.5 else vb
        elif va is not None:
            merged[key] = va
        else:
            merged[key] = vb
    return lift_v2b53_params_from_dict(merged)


def build_seed_pool(
    records: list[dict[str, Any]],
    *,
    top_k: int = TOP_K,
    rng: random.Random | None = None,
    baseline_peg_xy: float = DEMO3_BASELINE_PEG_XY,
) -> tuple[list[LiftV2B53Params], dict[str, Any]]:
    rng = rng or random.Random(42)
    valid = [r for r in records if _params_dict(r)]
    seeds: list[LiftV2B53Params] = []
    seen: set[str] = set()
    meta: dict[str, Any] = {"categories": {}, "seed_sources": [], "cross_hybrids": []}

    def _add_params(params: LiftV2B53Params, *, category: str, rec: dict[str, Any] | None = None) -> bool:
        key = json.dumps(params.to_dict(), sort_keys=True)
        if key in seen:
            return False
        seen.add(key)
        seeds.append(params)
        meta["seed_sources"].append(
            {
                "category": category,
                "cem_index": rec.get("cem_index") if rec else None,
                "search_index": rec.get("search_index") if rec else None,
                "nut_z_lift_delta": _nut_z(rec) if rec else None,
                "min_nut_peg_xy": _min_peg(rec) if rec else None,
                "bilateral_contact_steps": rec.get("bilateral_contact_steps") if rec else None,
                "right_finger_contact_count": rec.get("right_finger_contact_count") if rec else None,
            }
        )
        return True

    def _add_category(name: str, ranked: list[dict[str, Any]]) -> None:
        added = 0
        for rec in ranked[:top_k]:
            raw = _params_dict(rec)
            if raw and _add_params(lift_v2b53_from_b52(raw), category=name, rec=rec):
                added += 1
        meta["categories"][name] = {"requested": top_k, "added": added}

    contact_rich = [
        r
        for r in valid
        if int(r.get("right_finger_contact_count", 0)) > 0 and int(r.get("bilateral_contact_steps", 0)) > 0
    ]
    contact_rich.sort(
        key=lambda r: (
            int(r.get("bilateral_contact_steps", 0)),
            int(r.get("right_finger_contact_count", 0)),
        ),
        reverse=True,
    )
    _add_category("contact_rich", contact_rich)

    weak_lift = [r for r in valid if weak_lift_positive(r)]
    weak_lift.sort(key=_nut_z, reverse=True)
    _add_category("weak_lift_positive", weak_lift)

    transport_improved = [
        r
        for r in valid
        if _min_peg(r) < baseline_peg_xy * 0.99 or float(r.get("final_nut_peg_xy", baseline_peg_xy)) < baseline_peg_xy * 0.99
    ]
    transport_improved.sort(key=_min_peg)
    _add_category("transport_improved", transport_improved)

    low_peg = sorted(valid, key=lambda r: min(_min_peg(r), float(r.get("final_nut_peg_xy", baseline_peg_xy))))
    _add_category("low_nut_peg_xy", low_peg)

    cross_added = 0
    contact_top = contact_rich[: min(8, len(contact_rich))]
    weak_top = weak_lift[: min(8, len(weak_lift))]
    for c_rec in contact_top:
        for w_rec in weak_top:
            c_raw, w_raw = _params_dict(c_rec), _params_dict(w_rec)
            if not c_raw or not w_raw:
                continue
            hybrid = _blend_params(c_raw, w_raw, rng)
            if _add_params(hybrid, category="contact_weak_cross", rec=c_rec):
                cross_added += 1
                meta["cross_hybrids"].append(
                    {
                        "contact_cem_index": c_rec.get("cem_index"),
                        "weak_cem_index": w_rec.get("cem_index"),
                        "contact_bilateral": c_rec.get("bilateral_contact_steps"),
                        "weak_nut_z_lift_delta": _nut_z(w_rec),
                    }
                )
            if cross_added >= top_k:
                break
        if cross_added >= top_k:
            break
    meta["categories"]["contact_weak_cross"] = {"requested": top_k, "added": cross_added}
    meta["unique_seed_count"] = len(seeds)
    meta["baseline_peg_xy_m"] = baseline_peg_xy
    return seeds, meta
