from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from app.services.evaluation.report_export.export_options import ExportOptions
from app.services.evaluation.report_export.report_data import build_evaluation_report_data
from app.services.evaluation.report_export.service import export_evaluation_report


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _find_job(prefix: str) -> str | None:
    for root in (
        PROJECT_ROOT / "runs" / "evaluations" / "jobs",
        PROJECT_ROOT / "runs" / "cable_threading" / "jobs",
    ):
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir(), reverse=True):
            if path.is_dir() and path.name.startswith(prefix):
                return path.name
    return None


@pytest.mark.parametrize(
    "job_prefix",
    ["ct_eval_", "eval_", "isaac_eval_"],
)
def test_build_evaluation_report_data_smoke(job_prefix: str) -> None:
    job_id = _find_job(job_prefix)
    if job_id is None:
        pytest.skip(f"no runtime job found for prefix {job_prefix}")
    data = build_evaluation_report_data(job_id)
    assert data["reportMeta"]["jobId"] == job_id
    assert isinstance(data.get("basicInfo"), dict)
    assert isinstance(data.get("metricResults"), dict)
    assert isinstance(data.get("selectedMetricIds"), list)
    assert isinstance(data.get("episodeResults"), list)
    assert isinstance(data.get("runtimeFiles"), list)


def test_metric_results_only_selected_ids() -> None:
    job_id = _find_job("ct_eval_")
    if job_id is None:
        pytest.skip("no ct_eval job")
    data = build_evaluation_report_data(job_id)
    selected = data.get("selectedMetricIds") or []
    metric_results = data.get("metricResults") or {}
    if selected:
        assert set(metric_results.keys()).issubset(set(selected))


def test_old_eval_job_legacy_notice_or_basic_info() -> None:
    job_id = _find_job("eval_")
    if job_id is None:
        pytest.skip("no eval job")
    data = build_evaluation_report_data(job_id)
    assert data["basicInfo"]["jobId"] == job_id
    if not data.get("selectedMetricIds"):
        assert data.get("legacyNotice") or len(data.get("metricResults") or {}) <= 1


@pytest.mark.parametrize(
    "fmt,expected_name",
    [
        ("json", "report.json"),
        ("markdown", "report.md"),
        ("xlsx", "report.xlsx"),
        ("latex", "report.tex"),
        ("csv", "report_csv.zip"),
    ],
)
def test_export_formats(fmt: str, expected_name: str) -> None:
    job_id = _find_job("ct_eval_") or _find_job("eval_")
    if job_id is None:
        pytest.skip("no evaluation job")
    options = ExportOptions(format=fmt)  # type: ignore[arg-type]
    output_path, media_type, filename = export_evaluation_report(job_id, options)
    assert output_path.is_file()
    assert output_path.name == expected_name
    assert media_type
    assert filename


def test_export_zip_bundle_without_html() -> None:
    job_id = _find_job("ct_eval_") or _find_job("eval_")
    if job_id is None:
        pytest.skip("no evaluation job")
    output_path, _, _ = export_evaluation_report(job_id, ExportOptions(format="zip"))
    assert output_path.is_file()
    with zipfile.ZipFile(output_path) as archive:
        names = set(archive.namelist())
    assert "report.json" in names
    assert not any(name.endswith(".html") or name == "report.html" for name in names)


def test_reject_html_export_format() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        export_evaluation_report("eval_20260624_170036_ee2e", ExportOptions(format="html"))  # type: ignore[arg-type]
    assert exc.value.status_code == 400
    detail = exc.value.detail
    assert isinstance(detail, dict)
    assert detail.get("error") == "不支持 HTML 导出"


def test_missing_job_returns_structured_not_found() -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        build_evaluation_report_data("eval_20990101_000000_ffff")
    assert exc.value.status_code == 404
    detail = exc.value.detail
    assert isinstance(detail, dict)
    assert detail.get("error") == "未找到评测任务"
    assert detail.get("hint")


def test_export_json_contains_metric_results_structure() -> None:
    job_id = _find_job("ct_eval_")
    if job_id is None:
        pytest.skip("no ct_eval job")
    output_path, _, _ = export_evaluation_report(job_id, ExportOptions(format="json"))
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["reportMeta"]["jobId"] == job_id
    for entry in (payload.get("metricResults") or {}).values():
        assert "metricId" in entry
        assert "available" in entry


def test_report_excludes_wall_time_metrics() -> None:
    job_id = _find_job("eval_") or _find_job("ct_eval_")
    if job_id is None:
        pytest.skip("no evaluation job")
    data = build_evaluation_report_data(job_id)
    metric_results = data.get("metricResults") or {}
    assert "metric_runtime_max_runtime_sec_v1" not in metric_results
    assert "metric_episode_stability_v1" not in metric_results
    for row in data.get("episodeResults") or []:
        assert "runtimeSec" not in row
        assert "videoDurationSec" not in row


def test_dual_arm_episode_sim_time_from_trajectory() -> None:
    job_id = "eval_20260626_103509_1b68"
    job_root = PROJECT_ROOT / "runs" / "evaluations" / "jobs" / job_id
    if not job_root.is_dir():
        pytest.skip("eval job not found")
    data = build_evaluation_report_data(job_id)
    episodes = data.get("episodeResults") or []
    assert episodes
    first = episodes[0]
    assert first.get("stepCount") not in {None, "-", ""}
    assert first.get("simTimeSec") not in {None, "-", ""}
    mean_sim = (data.get("metricResults") or {}).get("metric_runtime_mean_sim_time_sec_v1")
    assert mean_sim is not None
    assert mean_sim.get("available") is True
    assert "34.2" in str(mean_sim.get("formattedValue") or mean_sim.get("value"))
