import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.experiment_metrics import compute_run_metrics


def test_compute_run_metrics_with_preview_and_platform_samples():
    events = [
        {"run_id": "r1", "scenario_id": "E1-CON", "method": "P", "event": "command_sent", "command_id": "c1", "cmd": "COLLECT_START", "ts_ms": 100.0},
        {"run_id": "r1", "scenario_id": "E1-CON", "method": "P", "event": "ack_received", "command_id": "c1", "cmd": "COLLECT_START", "ts_ms": 160.0},
        {"run_id": "r1", "scenario_id": "E1-CON", "method": "P", "event": "result_received", "command_id": "c1", "cmd": "COLLECT_START", "ts_ms": 220.0, "success": True},
        {"run_id": "r1", "scenario_id": "E1-CON", "method": "P", "event": "preview_request", "preview_request_ts_ms": 300.0},
        {"run_id": "r1", "scenario_id": "E1-CON", "method": "P", "event": "first_frame", "first_frame_ts_ms": 480.0},
        {"run_id": "r1", "scenario_id": "E1-CON", "method": "P", "event": "primary_preview_fail", "primary_preview_fail_ts_ms": 1000.0},
        {"run_id": "r1", "scenario_id": "E1-CON", "method": "P", "event": "primary_to_fallback_switch", "switch_ts_ms": 1180.0},
        {
            "run_id": "r1",
            "scenario_id": "E1-CON",
            "method": "P",
            "event": "preview_end",
            "preview_fps": 24.5,
            "preview_rtt_ms": 88.0,
            "preview_freeze_count": 1,
            "preview_freeze_total_ms": 180.0,
            "preview_availability": 0.97,
        },
        {"run_id": "r1", "scenario_id": "E1-CON", "method": "P", "event": "platform_resource_sample", "platform_cpu_percent": 10.0, "platform_rss_bytes": 1000.0, "relay_cpu_percent": 8.0, "relay_rss_bytes": 800.0},
        {"run_id": "r1", "scenario_id": "E1-CON", "method": "P", "event": "platform_resource_sample", "platform_cpu_percent": 14.0, "platform_rss_bytes": 1200.0, "relay_cpu_percent": 9.0, "relay_rss_bytes": 900.0},
    ]

    row = compute_run_metrics(events)

    assert row["ack_latency_ms_median"] == 60.0
    assert row["result_latency_ms_median"] == 120.0
    assert row["command_completion_reliability"] == 1.0
    assert row["first_frame_latency_ms_median"] == 180.0
    assert row["recovery_time_ms_median"] == 180.0
    assert row["preview_fps"] == 24.5
    assert row["preview_rtt_ms"] == 88.0
    assert row["platform_cpu_percent_avg"] == 12.0
    assert row["relay_cpu_percent_max"] == 9.0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
