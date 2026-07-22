#!/usr/bin/env python3
"""V1-D 方法验证：多 seed PINN 训练稳定性（LODO 指标均值/方差）。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_pinn_group_split import run_group_splits
from group_split_utils import load_enriched_meta, summarize_generalization_risk
from residual_dataset import load_npz_dataset

_V1_DIR = Path(__file__).resolve().parent
_EXPERIMENT_DIR = _V1_DIR.parent
DEFAULT_DATASET = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_v1c" / "training_dataset.npz"
DEFAULT_OUTPUT = _EXPERIMENT_DIR / "outputs" / "v1_residual_model_pinn"

LODO_METRICS = [
    "leave_one_demo_out_avg_pearson",
    "leave_one_demo_out_avg_success_accuracy",
    "leave_one_demo_out_avg_failure_type_accuracy",
    "leave_one_demo_out_avg_grasp_success_accuracy",
    "leave_one_demo_out_avg_lift_success_accuracy",
]


def _split_metric_means(split_results: list[dict[str, Any]]) -> dict[str, float | None]:
    lodo = [r for r in split_results if r.get("split_type") == "leave_one_demo_out"]
    if not lodo:
        return {
            "leave_one_demo_out_avg_pearson": None,
            "leave_one_demo_out_avg_success_accuracy": None,
            "leave_one_demo_out_avg_failure_type_accuracy": None,
            "leave_one_demo_out_avg_grasp_success_accuracy": None,
            "leave_one_demo_out_avg_lift_success_accuracy": None,
        }
    return {
        "leave_one_demo_out_avg_pearson": float(
            np.nanmean([r.get("pearson_E_total") or np.nan for r in lodo])
        ),
        "leave_one_demo_out_avg_success_accuracy": float(
            np.nanmean([r.get("success_classification_accuracy") or np.nan for r in lodo])
        ),
        "leave_one_demo_out_avg_failure_type_accuracy": float(
            np.nanmean([r.get("failure_type_accuracy") or np.nan for r in lodo])
        ),
        "leave_one_demo_out_avg_grasp_success_accuracy": float(
            np.nanmean([r.get("grasp_success_accuracy") or np.nan for r in lodo])
        ),
        "leave_one_demo_out_avg_lift_success_accuracy": float(
            np.nanmean([r.get("lift_success_accuracy") or np.nan for r in lodo])
        ),
    }


def _aggregate(values: list[float | None]) -> dict[str, float | None]:
    clean = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not clean:
        return {"mean": None, "variance": None, "std": None, "n": 0}
    arr = np.array(clean, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "variance": float(np.var(arr)),
        "std": float(np.std(arr)),
        "n": int(len(arr)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V1-D PINN multi-seed LODO stability check")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_npz_dataset(args.dataset)
    meta_records = load_enriched_meta(args.dataset)

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"PINN stability seed={seed}")
        split_results, _, group_extra = run_group_splits(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            bundle=bundle,
            meta_records=meta_records,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            hidden_dim=128,
            num_layers=3,
            dropout=0.1,
            val_frac=args.val_frac,
            seed=seed,
        )
        generalization = summarize_generalization_risk(split_results, None)
        metric_means = _split_metric_means(split_results)
        per_seed.append(
            {
                "seed": seed,
                "epochs": args.epochs,
                "split_results": split_results,
                "generalization": generalization,
                **metric_means,
            }
        )

    aggregate: dict[str, Any] = {}
    for key in LODO_METRICS:
        aggregate[key] = _aggregate([row.get(key) for row in per_seed])

    report = {
        "task": "V1-D_PINN_stability_check",
        "model_version": "V1-D_PINN_style_residual_energy_mlp",
        "dataset": str(args.dataset),
        "seeds": seeds,
        "epochs_per_split": args.epochs,
        "evaluation_protocol": "leave_one_demo_out_group_split_per_seed",
        "per_seed_results": [
            {
                "seed": row["seed"],
                "leave_one_demo_out_avg_pearson": row["leave_one_demo_out_avg_pearson"],
                "leave_one_demo_out_avg_success_accuracy": row["leave_one_demo_out_avg_success_accuracy"],
                "leave_one_demo_out_avg_failure_type_accuracy": row["leave_one_demo_out_avg_failure_type_accuracy"],
                "leave_one_demo_out_avg_grasp_success_accuracy": row["leave_one_demo_out_avg_grasp_success_accuracy"],
                "leave_one_demo_out_avg_lift_success_accuracy": row["leave_one_demo_out_avg_lift_success_accuracy"],
                "generalization_risk": row["generalization"].get("generalization_risk"),
            }
            for row in per_seed
        ],
        "aggregate_across_seeds": aggregate,
        "notes": [
            "Method validation only; not a cross-task generalization claim.",
            "Each seed retrains fresh LODO PINN models (5 demos × 1 training per split).",
            "Prefer aggregate mean/variance over single-seed random split metrics.",
        ],
    }

    out_path = args.output_dir / "stability_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "aggregate": aggregate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
