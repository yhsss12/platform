"""V1-C.5：Group split / leave-one-demo-out utilities."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import pearsonr, spearmanr

from residual_dataset import (
    DEMO_KEYS,
    FAILURE_TYPES,
    OUTCOME_TYPES,
    enrich_meta_record,
    load_npz_dataset,
)

COARSE_HOLDOUT_MODES = [
    "insertion_failed",
    "transport_failed",
    "grasp_lift_failed",
]


def _safe_corr(fn, x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) < 1e-8 or np.std(y) < 1e-8:
        return None
    value, _ = fn(x, y)
    return float(value)


def top_k_success_rate(pred_total: np.ndarray, success: np.ndarray, k: int) -> float:
    if len(pred_total) == 0:
        return 0.0
    k = min(k, len(pred_total))
    order = np.argsort(pred_total)
    picked = success[order[:k]]
    return float(np.any(picked > 0.5))


def top_k_refined_success_rate(pred_total: np.ndarray, refined_success: np.ndarray, k: int) -> float:
    if len(pred_total) == 0:
        return 0.0
    k = min(k, len(pred_total))
    order = np.argsort(pred_total)
    picked = refined_success[order[:k]]
    return float(np.any(picked > 0.5))


def load_enriched_meta(dataset_path) -> list[dict[str, Any]]:
    bundle = load_npz_dataset(dataset_path)
    records = bundle["meta"].get("meta_records", [])
    return [enrich_meta_record(r) for r in records]


def build_leave_one_demo_out_splits(meta_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    splits: list[dict[str, Any]] = []
    n = len(meta_records)
    all_idx = np.arange(n)
    for demo in DEMO_KEYS:
        test_mask = np.array([r.get("source_demo") == demo for r in meta_records], dtype=bool)
        test_idx = all_idx[test_mask]
        train_idx = all_idx[~test_mask]
        if len(test_idx) == 0:
            continue
        splits.append(
            {
                "split_id": f"leave_one_demo_out__{demo}",
                "split_type": "leave_one_demo_out",
                "test_demo": demo,
                "train_idx": train_idx,
                "test_idx": test_idx,
            }
        )
    return splits


def build_failure_mode_holdout_splits(meta_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    splits: list[dict[str, Any]] = []
    n = len(meta_records)
    all_idx = np.arange(n)

    holdouts = [
        ("insertion_failed", lambda r: r.get("source_failure_mode") == "insertion_failed"),
        ("transport_failed", lambda r: r.get("source_failure_mode") == "transport_failed"),
        (
            "grasp_lift_failed",
            lambda r: r.get("source_failure_mode") in ("grasp_failed", "lift_failed"),
        ),
    ]
    for holdout_name, predicate in holdouts:
        test_mask = np.array([predicate(r) for r in meta_records], dtype=bool)
        test_idx = all_idx[test_mask]
        train_idx = all_idx[~test_mask]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        splits.append(
            {
                "split_id": f"failure_mode_holdout__{holdout_name}",
                "split_type": "failure_mode_holdout",
                "holdout_failure_mode": holdout_name,
                "train_idx": train_idx,
                "test_idx": test_idx,
            }
        )
    return splits


def split_train_val(train_idx: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffled = train_idx.copy()
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, shuffled
    val_size = max(1, int(round(len(shuffled) * val_frac)))
    val_size = min(val_size, len(shuffled) - 1)
    val_idx = shuffled[:val_size]
    inner_train = shuffled[val_size:]
    return inner_train, val_idx


def confusion_matrix_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    label_to_i = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        ti = label_to_i.get(str(t))
        pi = label_to_i.get(str(p))
        if ti is not None and pi is not None:
            matrix[ti][pi] += 1
    return {"labels": labels, "matrix": matrix}


def evaluate_predictions_on_indices(
    *,
    bundle: dict[str, Any],
    meta_records: list[dict[str, Any]],
    test_idx: np.ndarray,
    pred_total: np.ndarray,
    pred_success_prob: np.ndarray,
    pred_failure_idx: np.ndarray,
    pred_outcome_idx: np.ndarray | None,
    pred_grasp_prob: np.ndarray | None,
    pred_lift_prob: np.ndarray | None,
    split_info: dict[str, Any],
) -> dict[str, Any]:
    target_total = bundle["target_E_total"][test_idx]
    success_flag = bundle["success_flag"][test_idx]
    failure_type_idx = bundle["failure_type_idx"][test_idx]
    outcome_idx = bundle["outcome_idx"][test_idx] if "outcome_idx" in bundle else None
    refined_success_flag = (
        bundle["refined_success_flag"][test_idx] if "refined_success_flag" in bundle else success_flag
    )
    grasp_success_flag = bundle.get("grasp_success_flag")
    lift_success_flag = bundle.get("lift_success_flag")
    if grasp_success_flag is not None:
        grasp_success_flag = grasp_success_flag[test_idx]
    if lift_success_flag is not None:
        lift_success_flag = lift_success_flag[test_idx]

    pred_total_t = pred_total[test_idx]
    pred_success_t = pred_success_prob[test_idx]
    pred_failure_t = pred_failure_idx[test_idx]
    pred_outcome_t = pred_outcome_idx[test_idx] if pred_outcome_idx is not None else None
    pred_grasp_t = pred_grasp_prob[test_idx] if pred_grasp_prob is not None else None
    pred_lift_t = pred_lift_prob[test_idx] if pred_lift_prob is not None else None

    mae = float(np.mean(np.abs(pred_total_t - target_total)))
    rmse = float(np.sqrt(np.mean((pred_total_t - target_total) ** 2)))
    success_acc = float(np.mean((pred_success_t >= 0.5) == (success_flag >= 0.5)))
    failure_acc = float(np.mean(pred_failure_t == failure_type_idx))

    outcome_acc = None
    if pred_outcome_t is not None and outcome_idx is not None:
        outcome_acc = float(np.mean(pred_outcome_t == outcome_idx))

    grasp_success_acc = None
    lift_success_acc = None
    if pred_grasp_t is not None and grasp_success_flag is not None:
        grasp_success_acc = float(np.mean((pred_grasp_t >= 0.5) == (grasp_success_flag >= 0.5)))
    if pred_lift_t is not None and lift_success_flag is not None:
        lift_success_acc = float(np.mean((pred_lift_t >= 0.5) == (lift_success_flag >= 0.5)))

    target_failure_names = [FAILURE_TYPES[int(i)] for i in failure_type_idx]
    pred_failure_names = [FAILURE_TYPES[int(i)] for i in pred_failure_t]
    failure_labels = sorted(set(target_failure_names) | set(pred_failure_names))

    target_outcome_names = (
        [OUTCOME_TYPES[int(i)] for i in outcome_idx] if outcome_idx is not None else []
    )
    pred_outcome_names = (
        [OUTCOME_TYPES[int(i)] for i in pred_outcome_t] if pred_outcome_t is not None else []
    )
    outcome_labels = sorted(set(target_outcome_names) | set(pred_outcome_names))

    special_focus = _special_focus_checks(
        meta_records=meta_records,
        test_idx=test_idx,
        pred_total_t=pred_total_t,
        pred_success_t=pred_success_t,
        pred_failure_names=pred_failure_names,
        target_failure_names=target_failure_names,
        pred_outcome_names=pred_outcome_names,
        target_outcome_names=target_outcome_names,
        pred_grasp_t=pred_grasp_t,
        pred_lift_t=pred_lift_t,
        grasp_success_flag=grasp_success_flag,
        lift_success_flag=lift_success_flag,
        success_flag=success_flag,
        split_info=split_info,
    )

    return {
        "split_id": split_info["split_id"],
        "split_type": split_info.get("split_type"),
        "test_demo": split_info.get("test_demo"),
        "holdout_failure_mode": split_info.get("holdout_failure_mode"),
        "train_size": int(len(split_info["train_idx"])),
        "test_size": int(len(test_idx)),
        "E_total_mae": mae,
        "E_total_rmse": rmse,
        "pearson_E_total": _safe_corr(pearsonr, pred_total_t, target_total),
        "spearman_E_total": _safe_corr(spearmanr, pred_total_t, target_total),
        "success_classification_accuracy": success_acc,
        "failure_type_accuracy": failure_acc,
        "outcome_classification_accuracy": outcome_acc,
        "grasp_success_accuracy": grasp_success_acc,
        "lift_success_accuracy": lift_success_acc,
        "top_k_low_energy_contains_success": {
            "top_1": top_k_success_rate(pred_total_t, success_flag, 1),
            "top_3": top_k_success_rate(pred_total_t, success_flag, 3),
            "top_5": top_k_success_rate(pred_total_t, success_flag, 5),
        },
        "top_k_low_energy_contains_refined_success": {
            "top_1": top_k_refined_success_rate(pred_total_t, refined_success_flag, 1),
            "top_3": top_k_refined_success_rate(pred_total_t, refined_success_flag, 3),
            "top_5": top_k_refined_success_rate(pred_total_t, refined_success_flag, 5),
        },
        "confusion_matrix_failure_type": confusion_matrix_dict(
            target_failure_names, pred_failure_names, failure_labels
        ),
        "confusion_matrix_outcome": confusion_matrix_dict(
            target_outcome_names, pred_outcome_names, outcome_labels
        )
        if outcome_labels
        else {"labels": [], "matrix": []},
        "special_focus": special_focus,
    }


def _special_focus_checks(
    *,
    meta_records,
    test_idx,
    pred_total_t,
    pred_success_t,
    pred_failure_names,
    target_failure_names,
    pred_outcome_names,
    target_outcome_names,
    pred_grasp_t,
    pred_lift_t,
    grasp_success_flag,
    lift_success_flag,
    success_flag,
    split_info,
) -> dict[str, Any]:
    focus: dict[str, Any] = {}
    test_demo = split_info.get("test_demo")

    if test_demo == "demo_3":
        lift_mask = [target_failure_names[i] == "lift_failed" for i in range(len(test_idx))]
        focus["demo_3_lift_failed"] = {
            "count": int(sum(lift_mask)),
            "failure_type_accuracy_on_lift_failed": (
                float(np.mean([pred_failure_names[i] == "lift_failed" for i, m in enumerate(lift_mask) if m]))
                if any(lift_mask)
                else None
            ),
            "grasp_improved_but_failed_outcome_recall": (
                float(
                    np.mean(
                        [
                            pred_outcome_names[i] == "grasp_improved_but_failed"
                            for i, m in enumerate(
                                [target_outcome_names[j] == "grasp_improved_but_failed" for j in range(len(test_idx))]
                            )
                            if m
                        ]
                    )
                )
                if any(t == "grasp_improved_but_failed" for t in target_outcome_names)
                else None
            ),
            "v2_b5_priority": True,
            "notes": [
                "demo_3 lift_failed recognition is a priority target for V2-B5 if unstable under group split.",
            ],
        }
        if pred_lift_t is not None and lift_success_flag is not None:
            focus["demo_3_lift_failed"]["lift_success_accuracy"] = float(
                np.mean((pred_lift_t >= 0.5) == (lift_success_flag >= 0.5))
            )

    if test_demo == "demo_2":
        refined_mask = [target_outcome_names[i] == "refined_success" for i in range(len(test_idx))]
        focus["demo_2_grasp_refined_success"] = {
            "count_refined_success": int(sum(refined_mask)),
            "outcome_accuracy_on_refined_success": (
                float(
                    np.mean(
                        [pred_outcome_names[i] == "refined_success" for i, m in enumerate(refined_mask) if m]
                    )
                )
                if any(refined_mask)
                else None
            ),
            "low_energy_refined_success": (
                float(np.min(pred_total_t[[i for i, m in enumerate(refined_mask) if m]]))
                if any(refined_mask)
                else None
            ),
        }
        if pred_grasp_t is not None and grasp_success_flag is not None:
            focus["demo_2_grasp_refined_success"]["grasp_success_accuracy"] = float(
                np.mean((pred_grasp_t >= 0.5) == (grasp_success_flag >= 0.5))
            )

    if test_demo == "demo_4":
        insertion_mask = [
            target_failure_names[i] in ("insertion_failed", "success") for i in range(len(test_idx))
        ]
        refined_mask = [target_outcome_names[i] == "refined_success" for i in range(len(test_idx))]
        focus["demo_4_insertion_refined_success"] = {
            "count_insertion_or_success": int(sum(insertion_mask)),
            "count_refined_success": int(sum(refined_mask)),
            "success_accuracy": float(np.mean((pred_success_t >= 0.5) == (success_flag >= 0.5))),
            "refined_success_outcome_recall": (
                float(
                    np.mean(
                        [pred_outcome_names[i] == "refined_success" for i, m in enumerate(refined_mask) if m]
                    )
                )
                if any(refined_mask)
                else None
            ),
        }

    holdout = split_info.get("holdout_failure_mode")
    if holdout == "grasp_lift_failed":
        focus["grasp_lift_holdout"] = {
            "v2_b5_priority": True,
            "notes": ["Grasp/lift failure-mode holdout tests generalization beyond demo-specific search."],
        }

    return focus


def build_prediction_rows(
    *,
    split_id: str,
    test_idx: np.ndarray,
    meta_records: list[dict[str, Any]],
    bundle: dict[str, Any],
    pred_total: np.ndarray,
    pred_success_prob: np.ndarray,
    pred_failure_idx: np.ndarray,
    pred_outcome_idx: np.ndarray | None,
    pred_grasp_prob: np.ndarray | None,
    pred_lift_prob: np.ndarray | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local_i, global_i in enumerate(test_idx):
        meta = meta_records[int(global_i)]
        row = {
            "split_id": split_id,
            "sample_idx": int(global_i),
            "source_demo": meta.get("source_demo"),
            "source_failure_mode": meta.get("source_failure_mode"),
            "sample_source": meta.get("sample_source"),
            "target_E_total": float(bundle["target_E_total"][global_i]),
            "pred_E_total": float(pred_total[global_i]),
            "target_success": int(bundle["success_flag"][global_i]),
            "pred_success_prob": float(pred_success_prob[global_i]),
            "target_failure_type": meta.get("failure_type"),
            "pred_failure_type": FAILURE_TYPES[int(pred_failure_idx[global_i])],
            "target_outcome": meta.get("outcome"),
            "pred_outcome": OUTCOME_TYPES[int(pred_outcome_idx[global_i])] if pred_outcome_idx is not None else "",
        }
        if pred_grasp_prob is not None and "grasp_success_flag" in bundle:
            row["target_grasp_success"] = int(bundle["grasp_success_flag"][global_i])
            row["pred_grasp_success_prob"] = float(pred_grasp_prob[global_i])
        if pred_lift_prob is not None and "lift_success_flag" in bundle:
            row["target_lift_success"] = int(bundle["lift_success_flag"][global_i])
            row["pred_lift_success_prob"] = float(pred_lift_prob[global_i])
        rows.append(row)
    return rows


def summarize_generalization_risk(
    split_results: list[dict[str, Any]],
    random_baseline: dict[str, Any] | None,
) -> dict[str, Any]:
    if not split_results:
        return {"generalization_risk": "unknown", "notes": ["No split results available."]}

    lodo = [r for r in split_results if r.get("split_type") == "leave_one_demo_out"]
    avg_pearson = float(np.nanmean([r.get("pearson_E_total") or np.nan for r in lodo]))
    avg_success = float(np.nanmean([r.get("success_classification_accuracy") or np.nan for r in lodo]))
    random_pearson = (random_baseline or {}).get("pearson_E_total")
    random_success = (random_baseline or {}).get("success_classification_accuracy")

    notes: list[str] = []
    risk = "moderate"
    if random_pearson is not None and avg_pearson < random_pearson - 0.05:
        notes.append(
            f"Leave-one-demo-out Pearson ({avg_pearson:.3f}) is materially below random split ({random_pearson:.3f})."
        )
        risk = "high"
    if random_success is not None and avg_success < random_success - 0.1:
        notes.append(
            f"Leave-one-demo-out success accuracy ({avg_success:.3f}) drops vs random split ({random_success:.3f})."
        )
        risk = "high"

    demo_3_splits = [r for r in lodo if r.get("test_demo") == "demo_3"]
    if demo_3_splits:
        d3 = demo_3_splits[0]
        lift_focus = (d3.get("special_focus") or {}).get("demo_3_lift_failed", {})
        lift_acc = lift_focus.get("failure_type_accuracy_on_lift_failed")
        if lift_acc is not None and lift_acc < 0.5:
            notes.append("demo_3 / lift_failed recognition is unstable under leave-one-demo-out.")
            notes.append("Mark V2-B5 (demo_3 lift_failed refinement) as priority before PINA/PINN formalization.")
            risk = "high"

    if not notes:
        notes.append("Group split metrics are reasonably aligned with random split; still prefer group split for generalization claims.")

    return {
        "generalization_risk": risk,
        "leave_one_demo_out_avg_pearson": avg_pearson if not np.isnan(avg_pearson) else None,
        "leave_one_demo_out_avg_success_accuracy": avg_success if not np.isnan(avg_success) else None,
        "random_split_pearson": random_pearson,
        "random_split_success_accuracy": random_success,
        "v2_b5_priority": any(
            (r.get("special_focus") or {}).get("demo_3_lift_failed", {}).get("v2_b5_priority")
            for r in lodo
        ),
        "notes": notes,
    }
