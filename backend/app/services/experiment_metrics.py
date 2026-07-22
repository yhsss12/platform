from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def percentile(values: Iterable[float], p: float) -> Optional[float]:
    seq = sorted(v for v in values if v is not None)
    if not seq:
        return None
    if len(seq) == 1:
        return float(seq[0])
    rank = max(0.0, min(1.0, float(p))) * (len(seq) - 1)
    lo = int(rank)
    hi = min(len(seq) - 1, lo + 1)
    if lo == hi:
        return float(seq[lo])
    frac = rank - lo
    return float(seq[lo] + (seq[hi] - seq[lo]) * frac)


def read_jsonl_events(log_dir: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not log_dir.exists():
        return events
    for path in sorted(log_dir.glob("*.jsonl")):
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                events.append(row)
    return events


def group_events_by_run(events: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for event in events:
        run_id = str(event.get("run_id") or "").strip()
        if not run_id:
            continue
        grouped[run_id].append(event)
    for items in grouped.values():
        items.sort(key=lambda item: str(item.get("ts") or ""))
    return dict(grouped)


def compute_run_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not events:
        return {}

    sent_by_command: Dict[str, Dict[str, Any]] = {}
    ack_latencies: List[float] = []
    result_latencies: List[float] = []
    preview_fail_ts: List[float] = []
    recovery_times: List[float] = []
    preview_end = None
    samples: List[Dict[str, Any]] = []
    first_preview_request_ts: Optional[float] = None
    first_frame_ts: Optional[float] = None
    result_success = 0
    sent_total = 0

    for event in events:
        ts = _to_float(event.get("ts_ms")) or _to_float(event.get("client_ts_ms"))
        if ts is None:
            payload = event.get("payload")
            if isinstance(payload, dict):
                ts = _to_float(payload.get("ts_ms"))
        event_name = str(event.get("event") or "")
        command_id = str(event.get("command_id") or "").strip()

        if event_name == "command_sent" and command_id and ts is not None:
            sent_by_command[command_id] = {"ts": ts, "cmd": event.get("cmd")}
            sent_total += 1
        elif event_name == "ack_received" and command_id and ts is not None:
            sent = sent_by_command.get(command_id)
            if sent:
                ack_latencies.append(max(0.0, ts - float(sent["ts"])))
        elif event_name == "result_received" and command_id and ts is not None:
            sent = sent_by_command.get(command_id)
            if sent:
                result_latencies.append(max(0.0, ts - float(sent["ts"])))
            if bool(event.get("success", True)):
                result_success += 1
        elif event_name == "preview_request":
            tsv = ts or _to_float(event.get("preview_request_ts_ms"))
            if tsv is not None and (first_preview_request_ts is None or tsv < first_preview_request_ts):
                first_preview_request_ts = tsv
        elif event_name == "first_frame":
            tsv = ts or _to_float(event.get("first_frame_ts_ms"))
            if tsv is not None and (first_frame_ts is None or tsv < first_frame_ts):
                first_frame_ts = tsv
        elif event_name == "primary_preview_fail":
            tsv = ts or _to_float(event.get("primary_preview_fail_ts_ms"))
            if tsv is not None:
                preview_fail_ts.append(tsv)
        elif event_name == "primary_to_fallback_switch":
            tsv = ts or _to_float(event.get("switch_ts_ms"))
            if tsv is not None and preview_fail_ts:
                fail_ts = preview_fail_ts.pop(0)
                recovery_times.append(max(0.0, tsv - fail_ts))
        elif event_name == "preview_end":
            preview_end = event
        elif event_name == "platform_resource_sample":
            samples.append(event)

    cpu_vals = [_to_float(item.get("platform_cpu_percent")) for item in samples]
    relay_cpu_vals = [_to_float(item.get("relay_cpu_percent")) for item in samples]
    mem_vals = [_to_float(item.get("platform_rss_bytes") or item.get("platform_mem_bytes")) for item in samples]
    relay_mem_vals = [_to_float(item.get("relay_rss_bytes") or item.get("relay_mem_bytes")) for item in samples]

    preview_fps = _to_float((preview_end or {}).get("preview_fps"))
    preview_rtt = _to_float((preview_end or {}).get("preview_rtt_ms") or (preview_end or {}).get("preview_rtt"))
    preview_freeze_count = int(_to_float((preview_end or {}).get("preview_freeze_count")) or 0)
    preview_freeze_total_ms = _to_float((preview_end or {}).get("preview_freeze_total_ms"))
    preview_availability = _to_float((preview_end or {}).get("preview_availability"))

    row: Dict[str, Any] = {
        "run_id": events[0].get("run_id"),
        "scenario_id": events[0].get("scenario_id"),
        "method": events[0].get("method"),
        "experiment_method_name": events[0].get("experiment_method_name"),
        "task_id": events[0].get("task_id"),
        "job_id": events[0].get("job_id"),
        "device_id": events[0].get("device_id"),
        "ack_latency_ms_median": median(ack_latencies) if ack_latencies else None,
        "ack_latency_ms_p95": percentile(ack_latencies, 0.95),
        "result_latency_ms_median": median(result_latencies) if result_latencies else None,
        "result_latency_ms_p95": percentile(result_latencies, 0.95),
        "command_completion_reliability": (result_success / sent_total) if sent_total else None,
        "first_frame_latency_ms_median": (
            max(0.0, first_frame_ts - first_preview_request_ts)
            if first_preview_request_ts is not None and first_frame_ts is not None
            else None
        ),
        "first_frame_latency_ms_p95": (
            max(0.0, first_frame_ts - first_preview_request_ts)
            if first_preview_request_ts is not None and first_frame_ts is not None
            else None
        ),
        "recovery_time_ms_median": median(recovery_times) if recovery_times else None,
        "recovery_time_ms_p95": percentile(recovery_times, 0.95),
        "preview_fps": preview_fps,
        "preview_rtt_ms": preview_rtt,
        "preview_freeze_count": preview_freeze_count,
        "preview_freeze_total_ms": preview_freeze_total_ms,
        "preview_availability": preview_availability,
        "platform_cpu_percent_avg": (sum(v for v in cpu_vals if v is not None) / len([v for v in cpu_vals if v is not None])) if any(v is not None for v in cpu_vals) else None,
        "platform_cpu_percent_max": max((v for v in cpu_vals if v is not None), default=None),
        "platform_rss_bytes_avg": (sum(v for v in mem_vals if v is not None) / len([v for v in mem_vals if v is not None])) if any(v is not None for v in mem_vals) else None,
        "platform_rss_bytes_max": max((v for v in mem_vals if v is not None), default=None),
        "relay_cpu_percent_avg": (sum(v for v in relay_cpu_vals if v is not None) / len([v for v in relay_cpu_vals if v is not None])) if any(v is not None for v in relay_cpu_vals) else None,
        "relay_cpu_percent_max": max((v for v in relay_cpu_vals if v is not None), default=None),
        "relay_rss_bytes_avg": (sum(v for v in relay_mem_vals if v is not None) / len([v for v in relay_mem_vals if v is not None])) if any(v is not None for v in relay_mem_vals) else None,
        "relay_rss_bytes_max": max((v for v in relay_mem_vals if v is not None), default=None),
    }
    return row

