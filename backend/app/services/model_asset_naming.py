"""模型资产用户可见命名（注册与列表展示统一来源）。"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional

from app.services.dataset_naming import (
    is_canonical_dataset_display_name,
    normalize_dataset_display_name,
    task_display_name,
)

INTERNAL_ID_PATTERN = re.compile(
    r"^(?:model|train)_[0-9]{8}_[0-9]{6}(?:_[0-9a-f]{4})?$",
    re.IGNORECASE,
)
JOB_ID_IN_NAME_PATTERN = re.compile(
    r"(?:^|_)(?:ct_gen|dac_gen|isaac_import|isaac_gen|isaac_ds)_[0-9]{8}_[0-9]{6}",
    re.IGNORECASE,
)
SNAKE_CASE_INTERNAL_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
TASK_TEMPLATE_ID_PATTERN = re.compile(r"^task_[a-z0-9_]+$")
DATE_TIME_SUFFIX_PATTERN = re.compile(
    r"^\d{4}/\d{2}/\d{2}\s+\d{2}:\d{2}$|^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?$"
)

INTERNAL_NAME_MARKERS = (
    "isaac_block_stacking",
    "isaac_stack_bc_smoke",
    "task_cable_threading_v1",
    "task_dual_arm_cable_manipulation_v1",
    "task_isaac_block_stacking_v1",
    "generated_dataset",
    "dataset.hdf5",
)


def format_model_recipe_label(
    *,
    framework: str | None = None,
    model_type: str | None = None,
    training_backend: str | None = None,
) -> str:
    fw = (framework or "").strip()
    mt = (model_type or "").strip()
    backend = (training_backend or "").strip()

    legacy = {
        "Isaac Robomimic BC": "Robomimic BC",
        "isaac_robomimic_bc": "Robomimic BC",
        "robomimic_bc": "Robomimic BC",
        "robomimic": "Robomimic BC",
        "Robomimic": "Robomimic BC",
        "torch_bc": "BC (PyTorch)",
        "BC (torch)": "BC (PyTorch)",
        "diffusion_policy": "Diffusion Policy",
        "Diffusion Policy": "Diffusion Policy",
    }
    for candidate in (fw, backend, mt):
        if candidate in legacy:
            return legacy[candidate]

    lowered_fw = fw.lower()
    lowered_backend = backend.lower()
    lowered_mt = mt.lower()

    if lowered_mt == "diffusion_policy" or lowered_fw == "diffusion_policy":
        return "Diffusion Policy"
    if lowered_mt == "bc" or lowered_fw == "bc":
        if "torch" in lowered_fw or lowered_backend == "torch_bc":
            return "BC (PyTorch)"
        if (
            "robomimic" in lowered_fw
            or lowered_backend in {"robomimic_bc", "isaac_robomimic_bc"}
            or "robomimic" in lowered_backend
        ):
            return "Robomimic BC"
        return "Robomimic BC"

    if "robomimic" in lowered_fw or lowered_backend in {"robomimic_bc", "isaac_robomimic_bc"}:
        return "Robomimic BC"
    if lowered_backend == "torch_bc" or "torch" in lowered_fw:
        return "BC (PyTorch)"

    return fw or backend or mt or "Unknown"


def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if re.match(r"^\d{8}T", raw):
        raw = f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}{raw[8:]}"
    if " " in raw and "T" not in raw:
        raw = raw.replace(" ", "T", 1)
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt
        return dt
    except ValueError:
        return None


def format_created_at_for_asset_name(created_at: Any) -> str:
    dt = _parse_datetime(created_at)
    if dt is None:
        return "—"
    if dt.tzinfo is None:
        return f"{dt.year:04d}/{dt.month:02d}/{dt.day:02d} {dt.hour:02d}:{dt.minute:02d}"
    local = dt.astimezone()
    return f"{local.year:04d}/{local.month:02d}/{local.day:02d} {local.hour:02d}:{local.minute:02d}"


def _contains_cjk(value: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in value)


def is_internal_context_label(value: str | None) -> bool:
    if not value or not value.strip():
        return True
    text = value.strip()
    lowered = text.lower()
    if INTERNAL_ID_PATTERN.match(text):
        return True
    if JOB_ID_IN_NAME_PATTERN.search(text):
        return True
    if text.startswith(("train_", "model_")):
        return True
    if TASK_TEMPLATE_ID_PATTERN.match(text):
        return True
    for marker in INTERNAL_NAME_MARKERS:
        if marker in lowered or marker in text:
            return True
    if text.startswith("isaac_stack") or text.startswith("isaac_block"):
        return True
    if SNAKE_CASE_INTERNAL_PATTERN.match(text) and not _contains_cjk(text):
        return True
    return False


def is_internal_model_asset_name(name: str | None) -> bool:
    if not name or not name.strip():
        return True
    text = name.strip()
    if is_internal_context_label(text):
        return True
    parts = [part.strip() for part in text.split("·") if part.strip()]
    if not parts:
        return True
    if len(parts) == 1:
        return is_internal_context_label(parts[0])
    if INTERNAL_ID_PATTERN.match(parts[0]) or parts[0].startswith("model_"):
        return True
    if parts[0].startswith("train_"):
        return True
    if len(parts) >= 2 and is_internal_context_label(parts[0]):
        return True
    if len(parts) >= 2 and (parts[-1] in {"bc", "diffusion_policy"} or parts[-1].lower() == "bc"):
        return True
    return False


def is_friendly_model_asset_display_name(name: str | None) -> bool:
    if not name or not name.strip():
        return False
    if is_internal_model_asset_name(name):
        return False
    if is_checkpoint_asset_display_name(name):
        return True
    parts = [part.strip() for part in name.split("·") if part.strip()]
    if len(parts) < 3:
        return False
    if is_internal_context_label(parts[0]):
        return False
    recipe = parts[1]
    if recipe.lower() in {"bc", "diffusion_policy", "unknown"}:
        return False
    return bool(DATE_TIME_SUFFIX_PATTERN.match(parts[-1]) or parts[-1] == "—")


def _normalize_context_label(
    *,
    training_task_name: str | None = None,
    dataset_name: str | None = None,
    dataset_id: str | None = None,
    task_template_id: str | None = None,
    task_type: str | None = None,
) -> str:
    task_name = (training_task_name or "").strip()
    if task_name and not is_internal_context_label(task_name):
        return task_name

    dataset = (dataset_name or "").strip()
    if dataset:
        normalized = normalize_dataset_display_name(
            display_name=dataset,
            name=dataset,
            task_type=task_type,
            source_job_id=dataset_id,
        )
        if normalized and normalized not in {"未知数据集", "未知任务数据"} and not is_internal_context_label(normalized):
            return normalized
        if is_canonical_dataset_display_name(dataset):
            return dataset
        if not is_internal_context_label(dataset):
            return dataset

    template = (task_template_id or "").strip()
    if template:
        label = task_display_name(template)
        if label and not is_internal_context_label(label):
            return f"{label}数据" if not label.endswith("数据") else label

    task = (task_type or "").strip()
    if task:
        label = task_display_name(task)
        if label and not is_internal_context_label(label):
            return f"{label}数据" if not label.endswith("数据") else label

    return "未命名模型资产"


def resolve_model_asset_context_label(
    *,
    training_task_name: str | None = None,
    dataset_name: str | None = None,
    dataset_id: str | None = None,
    task_template_id: str | None = None,
    task_type: str | None = None,
) -> str:
    return _normalize_context_label(
        training_task_name=training_task_name,
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        task_template_id=task_template_id,
        task_type=task_type,
    )


def build_model_asset_display_name(
    *,
    training_task_name: str | None = None,
    dataset_name: str | None = None,
    dataset_id: str | None = None,
    task_template_id: str | None = None,
    task_type: str | None = None,
    framework: str | None = None,
    model_type: str | None = None,
    training_backend: str | None = None,
    created_at: Any = None,
) -> str:
    context = _normalize_context_label(
        training_task_name=training_task_name,
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        task_template_id=task_template_id,
        task_type=task_type,
    )
    recipe = format_model_recipe_label(
        framework=framework,
        model_type=model_type,
        training_backend=training_backend,
    )
    created_label = format_created_at_for_asset_name(created_at)
    return f"{context} · {recipe} · {created_label}"


def build_checkpoint_asset_display_name(
    *,
    context_label: str,
    kind: str,
    epoch: int | None = None,
    metric_name: str | None = None,
) -> str:
    context = _strip_checkpoint_kind_suffix(context_label or "")
    normalized_kind = (kind or "").strip().lower()
    if normalized_kind == "final":
        if is_checkpoint_asset_display_name(context):
            return context
        return f"{context} · Final"
    if normalized_kind == "best":
        metric = (metric_name or "Loss").strip() or "Loss"
        suffix = f"Best {metric}"
        if context.endswith(f"· {suffix}"):
            return context
        return f"{context} · {suffix}"
    if normalized_kind == "epoch" and epoch is not None:
        suffix = f"Epoch {epoch}"
        if context.endswith(f"· {suffix}"):
            return context
        return f"{context} · {suffix}"
    if normalized_kind == "step" and epoch is not None:
        suffix = f"Step {epoch}"
        if context.endswith(f"· {suffix}"):
            return context
        return f"{context} · {suffix}"
    return context


def _strip_checkpoint_kind_suffix(label: str) -> str:
    text = (label or "").strip() or "未命名模型资产"
    parts = [part.strip() for part in text.split("·") if part.strip()]
    if len(parts) >= 2 and (
        parts[-1] == "Final"
        or parts[-1].startswith("Best ")
        or parts[-1].startswith("Epoch ")
        or parts[-1].startswith("Step ")
    ):
        return " · ".join(parts[:-1]).strip() or text
    return text


def is_checkpoint_asset_display_name(name: str | None) -> bool:
    if not name or not name.strip():
        return False
    parts = [part.strip() for part in name.split("·") if part.strip()]
    if len(parts) != 2:
        return False
    suffix = parts[1]
    if suffix in {"Final", "Best"}:
        return True
    if suffix.startswith("Best "):
        return True
    if suffix.startswith("Epoch ") or suffix.startswith("Step "):
        return True
    return False


def resolve_model_asset_display_name(
    *,
    stored_name: str | None = None,
    display_name: str | None = None,
    training_task_name: str | None = None,
    dataset_name: str | None = None,
    dataset_id: str | None = None,
    task_template_id: str | None = None,
    task_type: str | None = None,
    framework: str | None = None,
    model_type: str | None = None,
    training_backend: str | None = None,
    created_at: Any = None,
) -> str:
    explicit = (display_name or "").strip()
    if explicit and not is_internal_model_asset_name(explicit):
        if (
            is_friendly_model_asset_display_name(explicit)
            or is_checkpoint_asset_display_name(explicit)
            or not is_internal_context_label(explicit)
        ):
            return explicit

    stored = (stored_name or "").strip()
    if stored and (is_friendly_model_asset_display_name(stored) or is_checkpoint_asset_display_name(stored)):
        return stored
    if stored and not is_internal_model_asset_name(stored) and "·" not in stored:
        return stored

    return build_model_asset_display_name(
        training_task_name=training_task_name,
        dataset_name=dataset_name,
        dataset_id=dataset_id,
        task_template_id=task_template_id,
        task_type=task_type,
        framework=framework,
        model_type=model_type,
        training_backend=training_backend,
        created_at=created_at,
    )
