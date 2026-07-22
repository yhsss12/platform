#!/usr/bin/env python3
"""Convert experiment event logs into a metrics CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.experiment_metrics import compute_run_metrics, group_events_by_run, read_jsonl_events


def main() -> int:
    parser = argparse.ArgumentParser(description="Process experiment JSONL logs into metrics.csv")
    parser.add_argument("--log-dir", required=True, help="Directory containing JSONL experiment logs")
    parser.add_argument("--out", required=True, help="Output CSV path")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    out_path = Path(args.out)
    events = read_jsonl_events(log_dir)
    grouped = group_events_by_run(events)
    rows = [compute_run_metrics(items) for _, items in sorted(grouped.items())]
    rows = [row for row in rows if row]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        print(f"[experiment] no runs found in {log_dir}")
        return 0

    fieldnames = list(rows[0].keys())
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[experiment] wrote {len(rows)} rows -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
