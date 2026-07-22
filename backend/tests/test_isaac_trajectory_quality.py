from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from integrations.isaac_lab.trajectory_quality import (
    analyze_demo_actions,
    analyze_hdf5_dataset,
    classify_quality_display,
    write_trajectory_quality_report,
)


def _write_demo(path: Path, demo_key: str, actions: np.ndarray) -> None:
    with h5py.File(path, "a") if path.is_file() else h5py.File(path, "w") as f:
        grp = f.require_group("data").create_group(demo_key)
        grp.create_dataset("actions", data=actions)


def test_analyze_demo_flags_arm_spike(tmp_path: Path):
    t = 20
    actions = np.zeros((t, 7), dtype=np.float32)
    actions[:, :3] = 0.01
    actions[10, 0] = 1.0
    actions[9, 0] = -1.0
    q = analyze_demo_actions(actions, demo_key="demo_0")
    assert q.max_arm_delta > 0.5
    assert q.low_quality
    assert any("arm_action_delta" in w for w in q.warnings)


def test_gripper_switch_does_not_dominate_arm_delta(tmp_path: Path):
    t = 10
    actions = np.zeros((t, 7), dtype=np.float32)
    actions[5, 6] = 1.0
    actions[6, 6] = -1.0
    q = analyze_demo_actions(actions, demo_key="demo_0")
    assert q.max_arm_delta < 0.1
    assert q.gripper_switch_count == 1


def test_classify_episode_length_only_warning():
    display = classify_quality_display(
        quality_status="warning",
        quality_warnings=["demo_0:episode_length_high:551"],
    )
    assert display["qualityDisplayTier"] == "usable_long_episode"
    assert display["qualityDisplaySeverity"] == "mild"
    assert "episode 较长" in display["qualityDisplayLabel"]


def test_classify_motion_anomaly_warning():
    display = classify_quality_display(
        quality_status="warning",
        quality_warnings=["demo_1:arm_action_delta_high:0.539", "demo_1:wrist_flip_steps:1"],
        suspicious_wrist_flip_count=1,
    )
    assert display["qualityDisplayTier"] == "motion_warning"
    assert display["qualityDisplaySeverity"] == "motion"
    assert display["qualityDisplayLabel"] == "存在警告"


def test_write_report_for_mimic_job_if_present():
    job_h5 = Path(
        "/home/ubuntu/project/eai-idev2.1/runs/isaac_lab/jobs/"
        "isaac_gen_20260617_161852_cf89/artifacts/dataset.hdf5"
    )
    if not job_h5.is_file():
        pytest.skip("mimic_auto sample job missing")
    report = analyze_hdf5_dataset(job_h5, generation_mode="mimic_auto")
    assert report["demoCount"] == 10
    assert report["qualityStatus"] in {"passed", "warning", "failed"}
    assert "meanActionDelta" in report
    out = job_h5.parent / "trajectory_quality_report.json"
    write_trajectory_quality_report(job_h5, out, generation_mode="mimic_auto")
    assert out.is_file()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["demoCount"] == 10
