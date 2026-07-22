from __future__ import annotations

from app.services.model_asset_naming import (
    build_model_asset_display_name,
    format_model_recipe_label,
    is_friendly_model_asset_display_name,
    is_internal_model_asset_name,
    resolve_model_asset_display_name,
)


def test_format_model_recipe_label_bc_robomimic():
    assert format_model_recipe_label(framework="bc", model_type="bc", training_backend="robomimic_bc") == "Robomimic BC"
    assert format_model_recipe_label(framework="Isaac Robomimic BC", model_type="bc") == "Robomimic BC"
    assert format_model_recipe_label(framework="torch_bc", model_type="bc") == "BC (PyTorch)"


def test_build_model_asset_display_name_prefers_training_task():
    name = build_model_asset_display_name(
        training_task_name="物块堆叠 BC 训练",
        dataset_name="物块堆叠数据_20260617_1058",
        framework="Robomimic BC",
        model_type="bc",
        training_backend="isaac_robomimic_bc",
        created_at="2026-06-17 13:32:00",
    )
    assert name.startswith("物块堆叠 BC 训练 · Robomimic BC · 2026/06/17")


def test_build_model_asset_display_name_falls_back_to_dataset():
    name = build_model_asset_display_name(
        training_task_name="isaac_block_stacking",
        dataset_name="物块堆叠数据_20260617_1058",
        framework="Robomimic BC",
        model_type="bc",
        training_backend="isaac_robomimic_bc",
        created_at="2026-06-17 13:32:00",
    )
    assert name.startswith("物块堆叠数据_20260617_1058 · Robomimic BC ·")


def test_resolve_historical_internal_name():
    resolved = resolve_model_asset_display_name(
        stored_name="model_20260616_181729_96ce · task_cable_threading_v1 · ct_gen_20260616_132844_5f81 · bc",
        dataset_name="线缆穿杆数据_20260616_1328",
        framework="robomimic_bc",
        model_type="bc",
        training_backend="robomimic_bc",
        created_at="2026-06-16 18:17:29",
    )
    assert "task_cable_threading_v1" not in resolved
    assert "ct_gen_" not in resolved
    assert "Robomimic BC" in resolved
    assert is_friendly_model_asset_display_name(resolved)


def test_internal_name_detection():
    assert is_internal_model_asset_name("isaac_block_stacking")
    assert is_internal_model_asset_name("isaac_stack_bc_smoke2")
    assert is_internal_model_asset_name("model_20260617_131759_3d1b · 物块堆叠数据 · bc")
    assert not is_internal_model_asset_name("我的自定义模型 v2")


def test_checkpoint_asset_display_name_avoids_duplicate_final():
    from app.services.model_asset_naming import build_checkpoint_asset_display_name

    context = "线缆穿杆数据_20260625_0929 · Final"
    assert build_checkpoint_asset_display_name(context_label=context, kind="final") == (
        "线缆穿杆数据_20260625_0929 · Final"
    )
