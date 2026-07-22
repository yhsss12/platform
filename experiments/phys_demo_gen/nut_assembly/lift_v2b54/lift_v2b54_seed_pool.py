"""V2-B5.4 seed pool from B5.1/B5.2/B5.3 + cross hybrids."""
from __future__ import annotations

import ast
import csv
import json
import random
from pathlib import Path
from typing import Any

from lift_v2b54_objective import (
    DEMO3_BASELINE_PEG_XY,
    has_bilateral_contact,
    min_peg_xy,
    nut_z_delta,
    transport_improved,
    weak_lift_positive,
)
from lift_v2b54_refiner import LiftV2B54Params, lift_v2b54_from_prior, lift_v2b54_params_from_dict

TOP_K = 20


def _params_dict(rec: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("lift_v2b54_params", "lift_v2b53_params", "lift_v2b52_params", "lift_v2b51_params", "lift_v2b53_params_json"):
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


def load_rollout_records(*paths: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def load_csv_records(csv_path: Path, *, limit: int = 40) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("lift_v2b53_params_json"):
                try:
                    row["lift_v2b53_params"] = json.loads(row["lift_v2b53_params_json"])
                except json.JSONDecodeError:
                    pass
            elif row.get("lift_v2b52_params"):
                parsed = _params_dict(row)
                if parsed:
                    row["lift_v2b53_params"] = parsed
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def _blend(a: dict[str, Any], b: dict[str, Any], rng: random.Random, t: float | None = None) -> LiftV2B54Params:
    alpha = t if t is not None else rng.uniform(0.35, 0.65)
    merged: dict[str, Any] = {}
    for key in set(a.keys()) | set(b.keys()):
        va, vb = a.get(key), b.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            if isinstance(va, int) and isinstance(vb, int):
                merged[key] = int(round((1 - alpha) * va + alpha * vb))
            else:
                merged[key] = float((1 - alpha) * float(va) + alpha * float(vb))
        elif isinstance(va, str):
            merged[key] = va if rng.random() < 0.5 else vb
        elif va is not None:
            merged[key] = va
        else:
            merged[key] = vb
    return lift_v2b54_params_from_dict(merged)


def build_seed_pool(
    *,
    b51_jsonl: Path,
    b52_jsonl: Path,
    b53_jsonl: Path,
    b53_csv: Path,
    top_k: int = TOP_K,
    rng: random.Random | None = None,
) -> tuple[list[LiftV2B54Params], dict[str, Any]]:
    rng = rng or random.Random(44)
    all_recs = load_rollout_records(b51_jsonl, b52_jsonl, b53_jsonl)
    csv_recs = load_csv_records(b53_csv, limit=top_k * 2)
    seeds: list[LiftV2B54Params] = []
    seen: set[str] = set()
    meta: dict[str, Any] = {"categories": {}, "seed_sources": [], "cross_hybrids": []}

    def _add(params: LiftV2B54Params, *, category: str, rec: dict[str, Any] | None = None) -> bool:
        key = json.dumps(params.to_dict(), sort_keys=True)
        if key in seen:
            return False
        seen.add(key)
        seeds.append(params)
        meta["seed_sources"].append(
            {
                "category": category,
                "cem_index": rec.get("cem_index") if rec else None,
                "nut_z_lift_delta": nut_z_delta(rec) if rec else None,
                "min_nut_peg_xy": rec.get("min_nut_peg_xy") if rec else None,
                "final_nut_peg_xy": rec.get("final_nut_peg_xy") if rec else None,
            }
        )
        return True

    def _add_cat(name: str, ranked: list[dict[str, Any]]) -> None:
        added = 0
        for rec in ranked[:top_k]:
            raw = _params_dict(rec)
            if raw and _add(lift_v2b54_from_prior(raw), category=name, rec=rec):
                added += 1
        meta["categories"][name] = {"requested": top_k, "added": added}

    b53_recs = [r for r in all_recs if str(r.get("rollout_kind", "")).startswith("lift_v2b53")]
    _add_cat(
        "b53_best_transport",
        sorted(
            csv_recs or b53_recs,
            key=lambda r: float(r.get("min_nut_peg_xy", r.get("final_nut_peg_xy", 999))),
        ),
    )
    _add_cat(
        "b53_contact_gated",
        sorted(
            b53_recs,
            key=lambda r: (
                int(r.get("bilateral_contact_steps", 0)),
                float(r.get("transport_lift_score", r.get("lift_preserving_score", -999))),
            ),
            reverse=True,
        ),
    )
    weak_all = [r for r in all_recs if weak_lift_positive(r)]
    _add_cat("weak_lift_positive", sorted(weak_all, key=nut_z_delta, reverse=True))

    contact_transport = [
        r for r in all_recs if has_bilateral_contact(r) and transport_improved(r)
    ]
    _add_cat(
        "contact_rich_transport_improved",
        sorted(contact_transport, key=lambda r: (nut_z_delta(r), -min_peg_xy(r)), reverse=True),
    )

    cross_added = 0
    weak_top = sorted(weak_all, key=nut_z_delta, reverse=True)[:8]
    transport_top = sorted(contact_transport, key=lambda r: min_peg_xy(r))[:8]
    for w_rec in weak_top:
        for t_rec in transport_top:
            w_raw, t_raw = _params_dict(w_rec), _params_dict(t_rec)
            if not w_raw or not t_raw:
                continue
            if _add(_blend(w_raw, t_raw, rng), category="weak_transport_cross", rec=w_rec):
                cross_added += 1
                meta["cross_hybrids"].append(
                    {
                        "weak_nut_z": nut_z_delta(w_rec),
                        "transport_min_peg": t_rec.get("min_nut_peg_xy"),
                    }
                )
            if cross_added >= top_k:
                break
        if cross_added >= top_k:
            break
    meta["categories"]["weak_transport_cross"] = {"requested": top_k, "added": cross_added}
    meta["unique_seed_count"] = len(seeds)
    meta["baseline_peg_xy_m"] = DEMO3_BASELINE_PEG_XY
    return seeds, meta
