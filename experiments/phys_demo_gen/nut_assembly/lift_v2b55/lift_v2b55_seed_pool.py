"""V2-B5.5 seed pool from B5.4 results."""
from __future__ import annotations

import ast
import csv
import json
import random
from pathlib import Path
from typing import Any

from lift_v2b55_objective import has_bilateral_contact, min_peg_xy, nut_z_delta, peg_xy, weak_lift_positive
from lift_v2b55_refiner import LiftV2B55Params, lift_v2b55_from_prior, lift_v2b55_params_from_dict

TOP_K = 25


def _params_dict(rec: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("lift_v2b55_params", "lift_v2b54_params", "lift_v2b54_params_json", "lift_v2b55_params_json"):
        raw = rec.get(key)
        if not raw:
            continue
        if isinstance(raw, dict):
            return dict(raw)
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                try:
                    return dict(ast.literal_eval(raw))
                except (SyntaxError, ValueError):
                    continue
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_csv(path: Path, limit: int = 50) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed = _params_dict(row)
            if parsed:
                row["lift_v2b54_params"] = parsed
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def _blend(a: dict, b: dict, rng: random.Random) -> LiftV2B55Params:
    t = rng.uniform(0.35, 0.65)
    merged: dict[str, Any] = {}
    for key in set(a) | set(b):
        va, vb = a.get(key), b.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            merged[key] = int(round((1 - t) * va + t * vb)) if isinstance(va, int) and isinstance(vb, int) else float((1 - t) * va + t * vb)
        elif isinstance(va, str):
            merged[key] = va if rng.random() < 0.5 else vb
        elif va is not None:
            merged[key] = va
        else:
            merged[key] = vb
    return lift_v2b55_params_from_dict(merged)


def build_seed_pool(
    *, b54_jsonl: Path, b54_csv: Path, top_k: int = TOP_K, rng: random.Random | None = None
) -> tuple[list[LiftV2B55Params], dict[str, Any]]:
    rng = rng or random.Random(45)
    recs = load_jsonl(b54_jsonl)
    csv_recs = load_csv(b54_csv, limit=top_k * 2)
    seeds: list[LiftV2B55Params] = []
    seen: set[str] = set()
    meta: dict[str, Any] = {"categories": {}, "seed_sources": []}

    def add(params: LiftV2B55Params, cat: str, rec: dict | None = None) -> bool:
        key = json.dumps(params.to_dict(), sort_keys=True)
        if key in seen:
            return False
        seen.add(key)
        seeds.append(params)
        meta["seed_sources"].append({"category": cat, "nut_z_lift_delta": nut_z_delta(rec) if rec else None})
        return True

    def add_cat(name: str, ranked: list[dict[str, Any]]) -> None:
        n = 0
        for rec in ranked[:top_k]:
            raw = _params_dict(rec)
            if raw and add(lift_v2b55_from_prior(raw), name, rec):
                n += 1
        meta["categories"][name] = {"requested": top_k, "added": n}

    weak = [r for r in recs if weak_lift_positive(r)]
    add_cat("b54_weak_lift_positive", sorted(weak, key=nut_z_delta, reverse=True))

    joint = [r for r in recs if weak_lift_positive(r) and has_bilateral_contact(r) and peg_xy(r) < 0.33]
    add_cat("b54_best_joint", sorted(joint, key=lambda r: (nut_z_delta(r), -peg_xy(r)), reverse=True))

    contact = [r for r in recs if has_bilateral_contact(r)]
    add_cat("b54_best_contact", sorted(contact, key=lambda r: int(r.get("bilateral_contact_steps", 0)), reverse=True))

    transport = sorted(recs + csv_recs, key=lambda r: min_peg_xy(r))
    add_cat("b54_best_transport", transport)

    cross = 0
    for w in sorted(weak, key=nut_z_delta, reverse=True)[:8]:
        for t in transport[:8]:
            wr, tr = _params_dict(w), _params_dict(t)
            if wr and tr and add(_blend(wr, tr, rng), "weak_transport_cross", w):
                cross += 1
            if cross >= top_k:
                break
        if cross >= top_k:
            break
    meta["categories"]["weak_transport_cross"] = {"requested": top_k, "added": cross}
    meta["unique_seed_count"] = len(seeds)
    return seeds, meta
