#!/usr/bin/env python3
"""Split PhyGen feedback jsonl into train/test sets."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_feedback(
    rows: list[dict[str, Any]],
    *,
    split_by: str,
    train_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not rows:
        raise ValueError("No feedback rows to split")

    rng = random.Random(seed)
    if split_by == "sweep_id":
        keys = sorted({str(r.get("sweep_id", "unknown")) for r in rows})
    elif split_by == "source_demo_key":
        keys = sorted({str(r.get("source_demo_key", r.get("demo_key", "unknown"))) for r in rows})
    else:
        raise ValueError(f"Unsupported split_by: {split_by}")

    rng.shuffle(keys)
    n_train = max(1, int(round(len(keys) * train_ratio)))
    if n_train >= len(keys) and len(keys) > 1:
        n_train = len(keys) - 1
    train_keys = set(keys[:n_train])
    test_keys = set(keys[n_train:])

    train_rows = []
    test_rows = []
    for row in rows:
        if split_by == "sweep_id":
            key = str(row.get("sweep_id", "unknown"))
        else:
            key = str(row.get("source_demo_key", row.get("demo_key", "unknown")))
        if key in train_keys:
            train_rows.append(row)
        elif key in test_keys:
            test_rows.append(row)
        else:
            train_rows.append(row)

    summary = {
        "split_by": split_by,
        "train_ratio": train_ratio,
        "seed": seed,
        "train_keys": sorted(train_keys),
        "test_keys": sorted(test_keys),
        "num_train": len(train_rows),
        "num_test": len(test_rows),
        "train_success": sum(1 for r in train_rows if r.get("success")),
        "train_failure": sum(1 for r in train_rows if not r.get("success")),
        "test_success": sum(1 for r in test_rows if r.get("success")),
        "test_failure": sum(1 for r in test_rows if not r.get("success")),
    }
    return train_rows, test_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--test-output", required=True)
    parser.add_argument("--split-by", choices=["sweep_id", "source_demo_key"], default="sweep_id")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=9701)
    parser.add_argument("--summary-output", default=None)
    args = parser.parse_args()

    rows = _load_jsonl((ROOT / args.input).resolve())
    train_rows, test_rows, summary = split_feedback(
        rows,
        split_by=args.split_by,
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    _write_jsonl((ROOT / args.train_output).resolve(), train_rows)
    _write_jsonl((ROOT / args.test_output).resolve(), test_rows)

    summary_path = Path(args.summary_output) if args.summary_output else (ROOT / args.train_output).resolve().parent / "split_summary.json"
    if not summary_path.is_absolute():
        summary_path = (ROOT / summary_path).resolve()
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
