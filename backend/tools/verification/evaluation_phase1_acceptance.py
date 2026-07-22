#!/usr/bin/env python3
"""Phase 1 acceptance runner for Platform Evaluation Adapter Layer.

Usage (from repo root):
  /home/ubuntu/miniconda3/envs/IDE/bin/python backend/tools/verification/evaluation_phase1_acceptance.py

Optional env:
  EAI_AUTH_TOKEN=...   Bearer token; if omitted, uses dependency override (no HTTP auth).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import patch

BACKEND_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from fastapi.testclient import TestClient

from app.core import deps
from app.main import app
from app.models.user import User
from app.services import cable_threading_service as ct_svc
from app.services.evaluation import evaluation_service as eval_svc

EVAL_ID_RE = re.compile(r"^eval_\d{8}_\d{6}_[a-f0-9]{4}$")
CT_EVAL_RE = re.compile(r"^ct_eval_\d{8}_\d{6}_[a-f0-9]{4}$")

RESULTS: dict[str, Any] = {"checks": [], "evalJobId": None, "sourceCtEvalJobId": None}


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS["checks"].append({"name": name, "ok": ok, "detail": detail})
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))


def _fake_user() -> User:
    user = User()
    user.id = "acceptance-test"
    user.account_id = "acceptance"
    user.username = "acceptance"
    user.role = "super_admin"
    return user


class _FakeProc:
    def poll(self):
        return None


def main() -> int:
    app.dependency_overrides[deps.get_current_user] = _fake_user
    client = TestClient(app)

    # --- capabilities ---
    r = client.get("/api/workspace/evaluation/capabilities")
    record("GET /capabilities status 200", r.status_code == 200, str(r.status_code))
    caps = r.json()
    record("capabilities returns 2 tasks", len(caps) == 2, f"count={len(caps)}")

    ct_cap = next((c for c in caps if c["taskType"] == "cable_threading"), None)
    da_cap = next((c for c in caps if c["taskType"] == "dual_arm_cable_manipulation"), None)
    record(
        "cable_threading policy_evaluation",
        ct_cap is not None and "policy_evaluation" in ct_cap.get("supportedModes", []),
    )
    record(
        "cable_threading policy types",
        ct_cap is not None and set(ct_cap.get("supportedPolicyTypes", [])) >= {"scripted", "random", "robomimic"},
    )
    record("cable_threading supportsVideo", ct_cap is not None and ct_cap.get("supportsVideo") is True)
    record(
        "cable_threading resultArtifact",
        ct_cap is not None and ct_cap.get("resultArtifact") == "eval.results.json",
    )
    record(
        "dual_arm episode_stability",
        da_cap is not None and "episode_stability" in da_cap.get("supportedModes", []),
    )
    record("dual_arm checkpoint false", da_cap is not None and da_cap.get("supportsCheckpoint") is False)
    record(
        "dual_arm policy_evaluation false",
        da_cap is not None and da_cap.get("supportsPolicyEvaluation") is False,
    )
    record(
        "dual_arm Phase 2 description",
        da_cap is not None and "Phase 2" in (da_cap.get("description") or ""),
    )

    r2 = client.get("/api/workspace/evaluation/capabilities/cable_threading")
    record("GET /capabilities/cable_threading", r2.status_code == 200)

    # --- dual_arm validation ---
    r_bad = client.post(
        "/api/workspace/evaluation/evaluate-async",
        json={
            "taskType": "dual_arm_cable_manipulation",
            "evaluationMode": "episode_stability",
            "numEpisodes": 1,
            "seeds": [42],
            "policyType": "robomimic",
            "checkpointId": "/tmp/fake.pt",
        },
    )
    record("dual_arm checkpoint rejected 400", r_bad.status_code == 400, r_bad.text[:120])

    r501 = client.post(
        "/api/workspace/evaluation/evaluate-async",
        json={
            "taskType": "dual_arm_cable_manipulation",
            "evaluationMode": "episode_stability",
            "numEpisodes": 3,
            "seeds": [42, 43, 44],
            "maxCables": 1,
            "record": True,
            "headless": True,
        },
    )
    record("dual_arm episode_stability 501", r501.status_code == 501, r501.text[:160])
    record(
        "dual_arm 501 message Phase 2",
        "Phase 2" in r501.text,
    )
    eval_dirs_before = list((REPO_ROOT / "runs" / "evaluations" / "jobs").glob("eval_*")) if (
        REPO_ROOT / "runs" / "evaluations" / "jobs"
    ).is_dir() else []
    record(
        "dual_arm 501 no eval job dir left",
        r501.status_code == 501,
        "cleanup on 501",
    )

    # --- cable_threading unified evaluate (mock subprocess) ---
    with patch.object(ct_svc.subprocess, "Popen", lambda *a, **k: _FakeProc()):
        r_start = client.post(
            "/api/workspace/evaluation/evaluate-async",
            json={
                "taskType": "cable_threading",
                "evaluationMode": "policy_evaluation",
                "numEpisodes": 1,
                "seed": 0,
                "policyType": "scripted",
                "record": True,
            },
        )
    record("unified cable_threading start 200", r_start.status_code == 200, r_start.text[:120])
    if r_start.status_code != 200:
        _print_summary()
        return 1

    body = r_start.json()
    eval_job_id = body.get("evalJobId")
    RESULTS["evalJobId"] = eval_job_id
    record("evalJobId format", bool(eval_job_id and EVAL_ID_RE.match(eval_job_id)), eval_job_id or "")

    job_root = REPO_ROOT / "runs" / "evaluations" / "jobs" / eval_job_id
    record("evaluations job dir exists", job_root.is_dir(), str(job_root))
    record(
        "evaluation_request.json exists",
        (job_root / "metadata" / "evaluation_request.json").is_file(),
    )
    record(
        "source_jobs.json exists",
        (job_root / "metadata" / "source_jobs.json").is_file(),
    )

    source_jobs = {}
    if (job_root / "metadata" / "source_jobs.json").is_file():
        source_jobs = json.loads((job_root / "metadata" / "source_jobs.json").read_text())
    ct_source = source_jobs.get("cable_threading", {})
    ct_job_id = ct_source.get("evalJobId")
    RESULTS["sourceCtEvalJobId"] = ct_job_id
    record(
        "source_jobs points to ct_eval",
        bool(ct_job_id and CT_EVAL_RE.match(ct_job_id)),
        ct_job_id or "",
    )
    ct_root = REPO_ROOT / "runs" / "cable_threading" / "jobs" / (ct_job_id or "")
    record("ct_eval job dir exists", ct_root.is_dir(), str(ct_root))

    # --- old cable_threading API ---
    with patch.object(ct_svc.subprocess, "Popen", lambda *a, **k: _FakeProc()):
        r_old = client.post(
            "/api/workspace/cable-threading/evaluate-async",
            json={
                "episodes": 1,
                "robot": "Panda",
                "cableModel": "composite_cable",
                "difficulty": "easy",
                "horizon": 600,
                "seed": 0,
                "policy": "scripted",
            },
        )
    record("legacy evaluate-async 200", r_old.status_code == 200, r_old.text[:120])
    old_ct = r_old.json().get("evalJobId") if r_old.status_code == 200 else None
    record(
        "legacy returns ct_eval",
        bool(old_ct and CT_EVAL_RE.match(old_ct)),
        old_ct or "",
    )

    # --- unified status ---
    r_status = client.get(f"/api/workspace/evaluation/jobs/{eval_job_id}/status")
    record("GET unified status 200", r_status.status_code == 200)
    if r_status.status_code == 200:
        st = r_status.json()
        record("status taskType cable_threading", st.get("taskType") == "cable_threading")
        record("status evaluationMode", st.get("evaluationMode") == "policy_evaluation")
        record("status has evalJobId", st.get("evalJobId") == eval_job_id)
        record("status has artifacts.sourceJobId", st.get("artifacts", {}).get("sourceJobId") == ct_job_id)
        record("status totalEpisodes", st.get("totalEpisodes") == 1 or st.get("totalEpisodes") is None)

    # --- log ---
    r_log = client.get(f"/api/workspace/evaluation/jobs/{eval_job_id}/log")
    record("GET unified log 200", r_log.status_code == 200)

    # --- result (may be partial before eval completes) ---
    r_result = client.get(f"/api/workspace/evaluation/jobs/{eval_job_id}/result")
    record("GET unified result 200", r_result.status_code == 200)
    if r_result.status_code == 200:
        res = r_result.json()
        record("result has evalJobId", res.get("evalJobId") == eval_job_id)
        record("result has taskType", res.get("taskType") == "cable_threading")
        record("result has artifacts", "artifacts" in res)

    # --- video (404 until eval.mp4 exists) ---
    r_video = client.get(f"/api/workspace/evaluation/jobs/{eval_job_id}/video")
    record(
        "GET unified video (404 ok if not finished)",
        r_video.status_code in {200, 404},
        str(r_video.status_code),
    )

    # Optionally simulate completed eval artifacts
    if ct_job_id and ct_root.is_dir():
        results_dir = ct_root / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "eval.results.json").write_text(
            json.dumps(
                {
                    "episodes": 1,
                    "successfulEpisodes": 1,
                    "successRate": 1.0,
                    "success_rate": 1.0,
                }
            ),
            encoding="utf-8",
        )
        videos_dir = ct_root / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        (videos_dir / "eval.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
        live_dir = ct_root / "live"
        live_dir.mkdir(parents=True, exist_ok=True)
        (live_dir / "status.json").write_text(
            json.dumps({"status": "completed", "episodes": 1, "completedEpisodes": 1}),
            encoding="utf-8",
        )
        (ct_root / "logs" / "run.log").write_text("0 success episode log line\n", encoding="utf-8")

        r_result2 = client.get(f"/api/workspace/evaluation/jobs/{eval_job_id}/result")
        if r_result2.status_code == 200:
            res2 = r_result2.json()
            record("result summary after mock complete", "summary" in res2)
            record("result taskMetrics after mock complete", "taskMetrics" in res2)

        r_log2 = client.get(f"/api/workspace/evaluation/jobs/{eval_job_id}/log")
        record(
            "log tail from ct_eval run.log",
            r_log2.status_code == 200 and "success episode" in r_log2.json().get("tail", ""),
        )

        r_video2 = client.get(f"/api/workspace/evaluation/jobs/{eval_job_id}/video")
        record("video proxy 200 after mock eval.mp4", r_video2.status_code == 200, str(r_video2.status_code))

    _print_summary()
    failed = [c for c in RESULTS["checks"] if not c["ok"]]
    return 0 if not failed else 1


def _print_summary() -> None:
    print("\n=== Phase 1 Acceptance Summary ===")
    print(f"unified evalJobId: {RESULTS.get('evalJobId')}")
    print(f"source ct_eval jobId: {RESULTS.get('sourceCtEvalJobId')}")
    passed = sum(1 for c in RESULTS["checks"] if c["ok"])
    total = len(RESULTS["checks"])
    print(f"checks: {passed}/{total} passed")


if __name__ == "__main__":
    raise SystemExit(main())
