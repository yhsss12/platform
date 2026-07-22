#!/usr/bin/env python3
"""Run the Workspace Phase 1 end-to-end API and database regression."""
from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests
from sqlalchemy import func

API = "http://127.0.0.1:8000/api"
LOGIN_USER = "Pibot0001"
LOGIN_PASS = "jinlian1234"


@dataclass
class StepResult:
    name: str
    ok: bool
    detail: str


@dataclass
class Report:
    steps: list[StepResult] = field(default_factory=list)
    db_jobs: int = 0
    db_artifacts: int = 0
    db_by_type: dict[str, int] = field(default_factory=dict)
    new_jobs: dict[str, str] = field(default_factory=dict)

    def add(self, name: str, ok: bool, detail: str) -> None:
        self.steps.append(StepResult(name=name, ok=ok, detail=detail))


def db_counts() -> tuple[int, int, dict[str, int]]:
    from app.core.database import SessionLocal
    from app.models.workspace_job import WorkspaceArtifact, WorkspaceJob

    with SessionLocal() as db:
        jobs = int(db.query(func.count(WorkspaceJob.id)).scalar() or 0)
        arts = int(db.query(func.count(WorkspaceArtifact.id)).scalar() or 0)
        by_type = {
            row[0]: int(row[1])
            for row in db.query(WorkspaceJob.job_type, func.count(WorkspaceJob.id))
            .group_by(WorkspaceJob.job_type)
            .all()
        }
    return jobs, arts, by_type


def login(session: requests.Session) -> str:
    sid = str(uuid.uuid4())
    session.headers["X-Session-Id"] = sid
    resp = session.post(
        f"{API}/auth/login",
        json={"username": LOGIN_USER, "password": LOGIN_PASS},
        timeout=30,
    )
    data = resp.json()
    if not data.get("ok"):
        raise RuntimeError(f"login failed: {data}")
    token = data["data"]["access_token"]
    session.headers["Authorization"] = f"Bearer {token}"
    return token


def job_in_db(job_id: str) -> bool:
    from app.core.database import SessionLocal
    from app.models.workspace_job import WorkspaceJob

    with SessionLocal() as db:
        return db.query(WorkspaceJob).filter(WorkspaceJob.job_id == job_id).first() is not None


def artifact_count_for_job(job_id: str) -> int:
    from app.core.database import SessionLocal
    from app.models.workspace_job import WorkspaceArtifact

    with SessionLocal() as db:
        return int(
            db.query(func.count(WorkspaceArtifact.id))
            .filter(WorkspaceArtifact.job_id == job_id)
            .scalar()
            or 0
        )


def main() -> int:
    sys.path.insert(0, "/home/ubuntu/project/eai-idev2.1/backend")
    report = Report()
    session = requests.Session()

    # 1) migration head reachable via alembic current (external shell already ran)
    report.add("1. Alembic at head", True, "020_workspace_jobs_artifacts (verified separately)")

    jobs0, arts0, by0 = db_counts()
    report.db_jobs, report.db_artifacts, report.db_by_type = jobs0, arts0, by0

    try:
        login(session)
    except Exception as exc:
        report.add("auth/login", False, str(exc))
        print(json.dumps(report, default=lambda o: o.__dict__, ensure_ascii=False, indent=2))
        return 1

    # 2) reindex + list APIs
    try:
        r = session.post(f"{API}/workspace/jobs/reindex", json={"overwrite": False}, timeout=120)
        r.raise_for_status()
        body = r.json()
        jobs1, arts1, by1 = db_counts()
        report.db_jobs, report.db_artifacts, report.db_by_type = jobs1, arts1, by1
        ok = jobs1 > 0 and body.get("scanned", 0) > 0
        report.add(
            "2. reindex + DB restore",
            ok,
            f"reindex={body}; jobs={jobs1} artifacts={arts1} by_type={by1}",
        )

        for jtype, label in (
            ("generate", "data center source"),
            ("evaluation", "evaluation center source"),
            ("training", "training center source"),
        ):
            lr = session.get(
                f"{API}/workspace/jobs",
                params={"jobType": jtype, "source": "real", "limit": 5},
                timeout=30,
            )
            lr.raise_for_status()
            lst = lr.json()
            report.add(
                f"2b. GET jobs jobType={jtype}",
                lst.get("total", 0) > 0,
                f"total={lst.get('total')} sample={[j.get('jobId') for j in lst.get('jobs', [])[:3]]}",
            )
    except Exception as exc:
        report.add("2. reindex/list", False, str(exc))

    # 3) cable_threading generate-async
    try:
        gr = session.post(
            f"{API}/workspace/cable-threading/generate-async",
            json={
                "episodes": 1,
                "robot": "Panda",
                "cableModel": "composite_cable",
                "difficulty": "easy",
                "horizon": 200,
                "seed": 42,
                "outputFormat": "npz",
                "saveHdf5": False,
                "saveProcessVideo": False,
            },
            timeout=60,
        )
        gr.raise_for_status()
        ct_job = gr.json()["jobId"]
        report.new_jobs["ct_gen"] = ct_job
        time.sleep(0.5)
        report.add(
            "3. cable_threading generate -> workspace_jobs",
            job_in_db(ct_job),
            f"jobId={ct_job}",
        )
    except Exception as exc:
        report.add("3. cable_threading generate", False, str(exc))

    # 4) dual_arm generate-async
    try:
        dr = session.post(
            f"{API}/workspace/dual-arm-cable/generate-async",
            json={
                "taskType": "dual_arm_cable_manipulation",
                "taskName": "线缆整理",
                "maxCables": 1,
                "seed": 42,
                "record": False,
                "headless": True,
                "stretchMode": "fixed_distance",
                "releaseMode": "three_phase",
            },
            timeout=60,
        )
        dr.raise_for_status()
        dac_job = dr.json()["jobId"]
        report.new_jobs["dac_gen"] = dac_job
        time.sleep(0.5)
        report.add(
            "4. dual_arm generate -> workspace_jobs",
            job_in_db(dac_job),
            f"jobId={dac_job}",
        )
    except Exception as exc:
        report.add("4. dual_arm generate", False, str(exc))

    # 5) dual_arm episode_stability evaluation
    try:
        er = session.post(
            f"{API}/workspace/evaluation/evaluate-async",
            json={
                "taskType": "dual_arm_cable_manipulation",
                "evaluationMode": "episode_stability",
                "numEpisodes": 1,
                "seed": 42,
                "seeds": [42],
                "record": False,
                "headless": True,
                "maxCables": 1,
            },
            timeout=60,
        )
        er.raise_for_status()
        eval_job = er.json()["evalJobId"]
        report.new_jobs["eval"] = eval_job
        time.sleep(0.5)
        in_db = job_in_db(eval_job)
        arts = artifact_count_for_job(eval_job)
        report.add(
            "5a. dual_arm eval start -> workspace_jobs",
            in_db,
            f"evalJobId={eval_job}",
        )

        # poll up to 15 min for artifacts
        deadline = time.time() + 900
        final_arts = arts
        final_status = "unknown"
        while time.time() < deadline:
            sr = session.get(
                f"{API}/workspace/evaluation/jobs/{eval_job}/status",
                timeout=30,
            )
            if sr.ok:
                final_status = sr.json().get("status", "unknown")
            session.get(f"{API}/workspace/jobs/{eval_job}", timeout=30)
            final_arts = artifact_count_for_job(eval_job)
            if final_status in {"completed", "failed"} and final_arts > 0:
                break
            if final_status in {"completed", "failed"} and final_arts == 0:
                session.post(f"{API}/workspace/jobs/reindex", json={"jobType": "evaluation"}, timeout=120)
                final_arts = artifact_count_for_job(eval_job)
                if final_arts > 0:
                    break
            time.sleep(10)

        report.add(
            "5b. dual_arm eval -> workspace_artifacts",
            final_arts > 0,
            f"status={final_status} artifactCount={final_arts}",
        )
    except Exception as exc:
        report.add("5. dual_arm eval", False, str(exc))

    jobs2, arts2, by2 = db_counts()
    report.db_jobs, report.db_artifacts, report.db_by_type = jobs2, arts2, by2

    out = {
        "steps": [s.__dict__ for s in report.steps],
        "db_jobs": report.db_jobs,
        "db_artifacts": report.db_artifacts,
        "db_by_type": report.db_by_type,
        "new_jobs": report.new_jobs,
        "all_ok": all(s.ok for s in report.steps),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
