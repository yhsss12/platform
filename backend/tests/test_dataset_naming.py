from __future__ import annotations

from datetime import datetime, timezone

from app.services.dataset_naming import (
    build_dataset_display_name,
    is_canonical_dataset_display_name,
    normalize_dataset_display_name,
    task_display_name,
)


def test_task_display_name_mapping():
    assert task_display_name("cable_threading") == "线缆穿杆"
    assert task_display_name("dual_arm_cable_manipulation") == "线缆整理"
    assert task_display_name("block_stacking") == "物块堆叠"


def test_build_dataset_display_name_format():
    created = datetime(2026, 6, 17, 12, 36, tzinfo=timezone.utc)
    name = build_dataset_display_name(
        task_type="cable_threading",
        created_at=created,
        source_job_id="ct_gen_20260617_123600_abcd",
    )
    assert name == "线缆穿杆数据_20260617_1236"
    assert is_canonical_dataset_display_name(name)


def test_build_dataset_display_name_duplicate_suffix():
    created = datetime(2026, 6, 17, 12, 36, tzinfo=timezone.utc)
    name = build_dataset_display_name(
        task_type="cable_threading",
        created_at=created,
        dataset_index=2,
    )
    assert name == "线缆穿杆数据_20260617_1236_02"


def test_normalize_prefers_canonical_display_name():
    assert (
        normalize_dataset_display_name(
            task_type="cable_threading",
            display_name="线缆穿杆数据_20260617_1236",
            name="task_cable_threading_v1 · ct_gen_20260617_123600_abcd",
        )
        == "线缆穿杆数据_20260617_1236"
    )


def test_normalize_derives_from_task_type_and_job_id():
    name = normalize_dataset_display_name(
        task_type="cable_threading",
        name="task_cable_threading_v1 · ct_gen_20260617_141035_7299",
        source_job_id="ct_gen_20260617_141035_7299",
    )
    assert name == "线缆穿杆数据_20260617_1410"
    assert is_canonical_dataset_display_name(name)


def test_normalize_legacy_isaac_name():
    name = normalize_dataset_display_name(
        task_type="block_stacking",
        name="Isaac Stack Cube Dataset",
        created_at="2026-06-17T13:05:00+00:00",
        source_job_id="isaac_gen_20260617_130500_abcd",
    )
    assert name == "物块堆叠数据_20260617_1305"
