from __future__ import annotations

import argparse
import json
import shutil
import sys
import traceback
from datetime import datetime
from pathlib import Path

_INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
if str(_INTEGRATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_INTEGRATION_ROOT))

from scripts.try_mimicgen_datagen import try_mimicgen_datagen
from utils.hdf5_inspector import inspect_hdf5_dataset
from utils.hdf5_writer import save_rollout_hdf5
from utils.job_status import (
    mark_job_failed,
    set_job_stage,
    update_job_status,
)
from utils.pinn_repair_pipeline import run_pinn_repair_pipeline
from utils.rollout_subprocess import run_rollout_fallback_subprocess
from utils.robosuite_rollout import resolve_runtime_env, rollout_episodes
from utils.runtime_env import (
    bootstrap_mimicgen_worker_runtime,
    resolve_source_demo_path,
)
from utils.source_demo_provenance import audit_source_demo


def _update_status(job_root: Path, payload: dict) -> None:
    update_job_status(job_root, payload)


def _load_physics_enhancement_config(path: str | None) -> dict:
    if not path:
        return {"enabled": False}
    cfg_path = Path(path)
    if not cfg_path.is_file():
        return {"enabled": False}
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {"enabled": False}
    except (OSError, json.JSONDecodeError):
        return {"enabled": False}


def _build_summary(
    *,
    job_root: Path,
    args: argparse.Namespace,
    generation_mode: str,
    runtime_env: str,
    rollout_result: dict | None,
    hdf5_info: dict,
    mimic_result: dict,
    video_status: str,
    video_error: str | None,
    source_demo_path: str | None,
    source_demo_audit: dict | None = None,
    pinn_summary: dict | None = None,
    physics_enhancement: dict | None = None,
) -> dict:
    is_mimicgen = generation_mode == "mimicgen_datagen"
    rr = rollout_result or {}

    if is_mimicgen:
        success_eps = hdf5_info.get("successEpisodes")
        success_rate = None
        policy_mode = "mimicgen"
        episodes_gen = int(hdf5_info.get("demoCount") or mimic_result.get("numSuccess") or 0)
        failed_eps = max(args.episodes - episodes_gen, 0)
        datagen_failed_trials = max(args.episodes - episodes_gen, 0)
        datagen_success_rate = round(episodes_gen / max(args.episodes, 1), 4)
        success_status = "datagen_success_count" if episodes_gen > 0 else "not_evaluated"
    else:
        datagen_success_rate = None
        success_eps = int(
            rr.get("successEpisodes")
            or rr.get("validEpisodes")
            or hdf5_info.get("successEpisodes")
            or 0
        )
        failed_eps = int(rr.get("failedEpisodes") or max(args.episodes - success_eps, 0))
        episodes_gen = int(rr.get("episodesGenerated") or args.episodes)
        success_rate = float(rr.get("successRate") or (success_eps / max(episodes_gen, 1)))
        policy_mode = str(rr.get("policyMode") or "partial_scripted")
        datagen_failed_trials = None
        success_status = "evaluated"

    summary: dict = {
        "jobId": job_root.name,
        "taskTemplateId": args.task_template_id,
        "generationMode": generation_mode,
        "policyMode": policy_mode,
        "sourceEnvName": args.env_name,
        "runtimeEnvName": runtime_env,
        "sourceDemoPath": source_demo_path,
        "sourceDemoOrigin": (source_demo_audit or {}).get("sourceDemoOrigin"),
        "sourceDemoOriginReason": (source_demo_audit or {}).get("sourceDemoOriginReason"),
        "sourceDemoMd5": (source_demo_audit or {}).get("md5"),
        "sourceDemoHash": (source_demo_audit or {}).get("sourceDemoHash") or (source_demo_audit or {}).get("md5"),
        "sourceDemoEnvName": (source_demo_audit or {}).get("envName"),
        "episodesRequested": args.episodes,
        "episodesGenerated": episodes_gen,
        "validEpisodes": hdf5_info.get("validEpisodes"),
        "successEpisodes": success_eps if success_eps is not None else (0 if not is_mimicgen else None),
        "failedEpisodes": failed_eps,
        "successRate": round(success_rate, 4) if success_rate is not None else None,
        "successStatus": success_status if is_mimicgen else "evaluated",
        "datagenFailedTrials": datagen_failed_trials,
        "datagenSuccessRate": datagen_success_rate if is_mimicgen else None,
        "datasetFormat": "robomimic_hdf5",
        "mimicgenStats": mimic_result.get("mimicgenStats") if is_mimicgen else None,
        "validForTrainingEpisodes": int(
            rr.get("validForTrainingEpisodes") or hdf5_info.get("validForTrainingEpisodes") or (success_eps or 0)
        )
        if not is_mimicgen
        else hdf5_info.get("validForTrainingEpisodes"),
        "graspSuccessEpisodes": int(rr.get("graspSuccessEpisodes") or 0) if not is_mimicgen else None,
        "liftSuccessEpisodes": int(rr.get("liftSuccessEpisodes") or 0) if not is_mimicgen else None,
        "alignmentSuccessEpisodes": int(rr.get("alignmentSuccessEpisodes") or 0) if not is_mimicgen else None,
        "insertionSuccessEpisodes": int(rr.get("insertionSuccessEpisodes") or 0) if not is_mimicgen else None,
        "averageGraspAttempts": rr.get("averageGraspAttempts") if not is_mimicgen else None,
        "averageFinalXYError": rr.get("averageFinalXYError") if not is_mimicgen else None,
        "averageFinalHeightError": rr.get("averageFinalHeightError") if not is_mimicgen else None,
        "hasStageStatistics": bool(rr.get("hasStageStatistics")) if not is_mimicgen else False,
        "averageSteps": hdf5_info.get("totalSteps", 0) // max(episodes_gen, 1),
        "demoCount": hdf5_info.get("demoCount", episodes_gen),
        "totalSteps": hdf5_info.get("totalSteps"),
        "failureDistribution": rr.get("failureDistribution") or {},
        "hasEpisodeMetadata": bool(hdf5_info.get("hasEpisodeMetadata")),
        "hasDatagenInfo": bool(hdf5_info.get("hasDatagenInfo")),
        "hasObjectPoses": bool(hdf5_info.get("hasObjectPoses")),
        "objectPoseKeys": hdf5_info.get("objectPoseKeys") or [],
        "datasetPath": "datasets/nut_assembly_generated.hdf5",
        "videoPath": "videos/generate.mp4" if args.render_video else None,
        "videoStatus": video_status,
        "videoError": video_error,
        "mimicgenAttempt": mimic_result,
        "mimicgenDatagenFailed": not mimic_result.get("ok", False),
        "fallbackToRobosuiteRollout": generation_mode == "robosuite_rollout",
        "completedAt": datetime.now().isoformat(timespec="seconds"),
    }

    enhancement = physics_enhancement or {}
    pinn = pinn_summary or {}
    if enhancement.get("enabled"):
        raw_count = int(pinn.get("rawDemoCount") or episodes_gen)
        repaired_count = int(pinn.get("repairedDemoCount") or 0)
        final_count = int(pinn.get("finalDemoCount") or (episodes_gen + repaired_count))
        summary.update(
            {
                "physicsEnhancementEnabled": True,
                "enhancementMode": "pinn_repair",
                "pinnModelId": pinn.get("modelId") or enhancement.get("modelId"),
                "pinnBackend": pinn.get("pinnBackend"),
                "modelLoaded": pinn.get("modelLoaded"),
                "modelPath": pinn.get("modelPath"),
                "pipelineVersion": pinn.get("pipelineVersion"),
                "candidateMode": pinn.get("candidateMode"),
                "mimicgenGeneratedDemos": raw_count,
                "pinnCandidateCount": pinn.get("candidateCount", 0),
                "pinnRepairAttempted": pinn.get("repairAttempted", 0),
                "pinnRepairSucceeded": pinn.get("repairSucceeded", 0),
                "pinnValidationSucceeded": pinn.get("validationSucceeded", 0),
                "rawDemoCount": raw_count,
                "repairedDemoCount": repaired_count,
                "finalDemoCount": final_count,
                "episodesGenerated": final_count,
                "demoCount": final_count,
                "pinnRepairValidationRate": round(
                    (pinn.get("validationSucceeded") or 0) / max(pinn.get("candidateCount") or 1, 1),
                    4,
                ),
                "enhancementGain": repaired_count,
                "enhancementStatus": pinn.get("enhancementStatus"),
            }
        )
    else:
        summary["physicsEnhancementEnabled"] = False

    if (
        generation_mode == "robosuite_rollout"
        and mimic_result
        and not mimic_result.get("ok")
        and not mimic_result.get("skipped")
    ):
        summary["fallbackFrom"] = "mimicgen_datagen"
        summary["fallbackReason"] = mimic_result.get("reason") or "mimicgen_datagen_failed"

    return summary


def run_generation(args: argparse.Namespace) -> int:
    job_root = Path(args.job_root).resolve()
    job_root.mkdir(parents=True, exist_ok=True)
    (job_root / "logs").mkdir(parents=True, exist_ok=True)
    (job_root / "datasets").mkdir(parents=True, exist_ok=True)
    (job_root / "results").mkdir(parents=True, exist_ok=True)
    (job_root / "live").mkdir(parents=True, exist_ok=True)
    (job_root / "intermediate").mkdir(parents=True, exist_ok=True)
    video_path = job_root / "videos" / "generate.mp4"
    if args.render_video:
        (job_root / "videos").mkdir(parents=True, exist_ok=True)

    log_path = job_root / "logs" / "generate.log"
    pinn_log_path = job_root / "logs" / "pinn_repair.log"
    hdf5_out = job_root / "datasets" / "nut_assembly_generated.hdf5"
    raw_hdf5_out = job_root / "datasets" / "nut_assembly_mimicgen_raw.hdf5"
    manifest_path = job_root / "manifest.json"
    summary_path = job_root / "results" / "generation_summary.json"
    physics_enhancement = _load_physics_enhancement_config(getattr(args, "physics_enhancement_config", None))

    log_lines: list[str] = ["=== NutAssembly data generation worker ==="]
    # MimicGen requires upstream robosuite, while the explicit expert rollout
    # intentionally uses the vendored robosuite that contains NutAssemblySquare.
    # Do not apply the mutually exclusive MimicGen runtime check to rollout jobs.
    if args.generation_mode == "mimicgen_datagen":
        try:
            log_lines.extend(bootstrap_mimicgen_worker_runtime())
        except RuntimeError as exc:
            log_lines.append(str(exc))
            log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
            mark_job_failed(
                job_root,
                {
                    "status": "failed",
                    "jobId": job_root.name,
                    "generationModePreference": args.generation_mode,
                },
                error=str(exc),
                failure_reason="wrong_robosuite_source",
                traceback_text="\n".join(log_lines),
            )
            return 1
    else:
        log_lines.append("runtime_check=robosuite_rollout")

    runtime_env, fallback_reason = resolve_runtime_env(args.env_name)
    prefer_mimicgen = args.generation_mode != "robosuite_rollout"
    source_demo_resolved = args.source_demo_path
    mimicgen_error: str | None = None
    fallback_error: str | None = None
    fallback_attempted = False
    fallback_succeeded = False

    resolved_source, source_err = resolve_source_demo_path(
        args.source_demo_path,
        selection=getattr(args, "source_demo_selection", None),
    )
    if source_err and prefer_mimicgen:
        log_lines.append(source_err)
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        mark_job_failed(
            job_root,
            {
                "status": "running",
                "jobId": job_root.name,
                "generationMode": "mimicgen_datagen",
                "generationModePreference": args.generation_mode,
                "policyMode": "mimicgen",
            },
            error=source_err,
            failure_reason="source_demo_missing",
            traceback_text=source_err,
        )
        return 1
    if resolved_source is not None:
        source_demo_resolved = str(resolved_source)

    user_provided_source = bool(
        (args.source_demo_path and str(args.source_demo_path).strip())
        or (getattr(args, "source_demo_selection", None) == "custom")
    )
    source_demo_audit: dict = {}
    if resolved_source is not None:
        source_demo_audit = audit_source_demo(resolved_source, user_provided=user_provided_source)

    status_payload = {
        "status": "running",
        "stage": "queued",
        "jobType": "generate",
        "taskType": "nut_assembly",
        "taskTemplateId": args.task_template_id,
        "jobId": job_root.name,
        "episode": 0,
        "episodes": args.episodes,
        "episodesRequested": args.episodes,
        "episodesGenerated": 0,
        "datagenFailedTrials": 0,
        "progress": 0,
        "step": 0,
        "horizon": args.horizon,
        "seed": args.seed,
        "sourceEnvName": args.env_name,
        "runtimeEnvName": runtime_env,
        "envFallbackReason": fallback_reason or None,
        "generationMode": "mimicgen_datagen" if prefer_mimicgen else "robosuite_rollout",
        "generationModePreference": args.generation_mode,
        "policyMode": "mimicgen" if prefer_mimicgen else "partial_scripted",
        "sourceDemoPath": str(source_demo_resolved) if source_demo_resolved else None,
        "sourceDemoOrigin": source_demo_audit.get("sourceDemoOrigin"),
        "sourceDemoOriginReason": source_demo_audit.get("sourceDemoOriginReason"),
        "successfulEpisodes": 0,
        "failedEpisodes": 0,
        "successRate": None,
        "failureDistribution": {},
        "videoStatus": "pending" if args.render_video else None,
        "message": "NutAssembly 数据生成任务已启动",
        "startedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
        "error": None,
        "failureReason": None,
    }
    _update_status(job_root, status_payload)

    generation_mode = "mimicgen_datagen" if prefer_mimicgen else "robosuite_rollout"
    mimic_result: dict = {"ok": False, "skipped": not prefer_mimicgen}
    rollout_result: dict | None = None
    hdf5_info: dict = {}
    pinn_summary: dict | None = None
    video_status = "skipped" if not args.render_video else "pending"
    video_error: str | None = None

    try:
        source_demo = resolved_source

        if prefer_mimicgen:
            def on_mimicgen_status(update: dict) -> None:
                status_payload.update(update)
                _update_status(job_root, status_payload)

            mimic_result = try_mimicgen_datagen(
                job_root=job_root,
                episodes=args.episodes,
                seed=args.seed,
                source_env_name=args.env_name,
                source_demo_path=source_demo,
                render_video=args.render_video,
                on_status=on_mimicgen_status,
                status_base=status_payload,
            )
            source_demo_resolved = (
                mimic_result.get("sourceDemoPath")
                or source_demo_resolved
                or str(resolved_source or "")
            )

            if mimic_result.get("ok"):
                import shutil as _shutil

                _shutil.copy2(mimic_result["hdf5Path"], raw_hdf5_out)
                generation_mode = "mimicgen_datagen"
                runtime_env = mimic_result.get("runtimeEnvName") or args.env_name
                hdf5_info = inspect_hdf5_dataset(raw_hdf5_out)

                if physics_enhancement.get("enabled") and generation_mode == "mimicgen_datagen":
                    pinn_log_lines: list[str] = ["=== PINN repair pipeline ==="]

                    def _on_pinn_status(update: dict) -> None:
                        status_payload.update(update)
                        _update_status(job_root, status_payload)

                    try:
                        pinn_summary = run_pinn_repair_pipeline(
                            job_root,
                            raw_hdf5=raw_hdf5_out,
                            final_hdf5=hdf5_out,
                            config=physics_enhancement,
                            env_name=runtime_env,
                            horizon=args.horizon,
                            seed=args.seed,
                            generation_mode=generation_mode,
                            policy_mode="mimicgen",
                            on_status=_on_pinn_status,
                            log_lines=pinn_log_lines,
                        )
                        hdf5_info = inspect_hdf5_dataset(hdf5_out)
                        log_lines.extend(pinn_log_lines)
                        pinn_log_path.write_text("\n".join(pinn_log_lines) + "\n", encoding="utf-8")
                        status_payload["physicsEnhancementEnabled"] = True
                        status_payload["pinnBackend"] = pinn_summary.get("pinnBackend")
                        status_payload["modelLoaded"] = pinn_summary.get("modelLoaded")
                        status_payload["pinnCandidateCount"] = pinn_summary.get("candidateCount", 0)
                        status_payload["pinnRepairAttempted"] = pinn_summary.get("repairAttempted", 0)
                        status_payload["pinnValidationSucceeded"] = pinn_summary.get("validationSucceeded", 0)
                        status_payload["enhancementStatus"] = pinn_summary.get("enhancementStatus")
                        status_payload["finalDemoCount"] = pinn_summary.get("finalDemoCount")
                    except Exception as pinn_exc:
                        import shutil as _shutil2

                        _shutil2.copy2(raw_hdf5_out, hdf5_out)
                        pinn_log_lines.append(f"pinn_repair_failed={pinn_exc}")
                        pinn_log_path.write_text("\n".join(pinn_log_lines) + "\n", encoding="utf-8")
                        log_lines.extend(pinn_log_lines)
                        log_lines.append(f"pinn_repair_error={pinn_exc}")
                        pinn_summary = {
                            "validationSucceeded": 0,
                            "repairedDemoCount": 0,
                            "rawDemoCount": hdf5_info.get("demoCount", 0),
                            "finalDemoCount": hdf5_info.get("demoCount", 0),
                            "enhancementStatus": "completed_no_repair_success",
                            "error": str(pinn_exc),
                        }
                else:
                    _shutil.copy2(raw_hdf5_out, hdf5_out)

                log_lines.append("mimicgen_datagen=success")
                if mimic_result.get("videoPath") and Path(mimic_result["videoPath"]).is_file():
                    if not video_path.is_file():
                        shutil.copy2(mimic_result["videoPath"], video_path)
                    video_status = "available"
                    status_payload["generateVideoExists"] = True
                    status_payload["generateVideo"] = str(video_path)
                elif args.render_video:
                    video_status = "failed" if not video_path.is_file() else "available"
            else:
                generation_mode = "robosuite_rollout"
                mimicgen_error = str(mimic_result.get("error") or mimic_result.get("reason") or "mimicgen_failed")
                log_lines.extend(
                    [
                        "mimicgen_datagen_failed",
                        f"mimicgen_reason={mimic_result.get('reason')}",
                        f"mimicgen_error={mimicgen_error}",
                        "fallback_to_robosuite_rollout",
                    ]
                )
                status_payload["mimicgenError"] = mimicgen_error
                status_payload["mimicgenFallbackReason"] = mimic_result.get("reason")
                status_payload["mimicgenFallbackError"] = mimic_result.get("error")
                status_payload["fallbackFrom"] = "mimicgen_datagen"
                status_payload["fallbackReason"] = mimic_result.get("reason") or "mimicgen_datagen_failed"
                status_payload["fallbackAttempted"] = True
                fallback_attempted = True
        else:
            log_lines.append("generation_mode_preference=robosuite_rollout (skip mimicgen)")

        if generation_mode == "robosuite_rollout":
            use_fallback_subprocess = prefer_mimicgen and not mimic_result.get("ok")
            if use_fallback_subprocess:
                fb = run_rollout_fallback_subprocess(
                    job_root=job_root,
                    env_name=args.env_name,
                    episodes=args.episodes,
                    seed=args.seed,
                    horizon=args.horizon,
                    render_video=args.render_video,
                )
                fallback_attempted = True
                status_payload["fallbackAttempted"] = True
                if fb.get("ok"):
                    fallback_succeeded = True
                    status_payload["fallbackSucceeded"] = True
                    rollout_result = fb.get("rolloutResult") or {}
                    hdf5_info = fb.get("hdf5Info") or {}
                    log_lines.append("fallback_rollout=success")
                else:
                    fallback_succeeded = False
                    fallback_error = str(fb.get("error") or "rollout_fallback_failed")
                    status_payload["fallbackSucceeded"] = False
                    status_payload["fallbackError"] = fallback_error
                    log_lines.extend(
                        [
                            "fallback_rollout=failed",
                            f"fallback_error={fallback_error}",
                            str(fb.get("traceback") or ""),
                        ]
                    )
                    raise RuntimeError(f"rollout_fallback_failed: {fallback_error}")
            else:
                def on_progress(progress: dict) -> None:
                    status_payload.update(progress)
                    _update_status(job_root, status_payload)

                rollout_result = rollout_episodes(
                    env_name=args.env_name,
                    episodes=args.episodes,
                    seed=args.seed,
                    horizon=args.horizon,
                    render_video=args.render_video,
                    video_path=video_path if args.render_video else None,
                    debug_log_path=job_root / "logs" / "stage_debug.jsonl",
                    on_progress=on_progress,
                )
                hdf5_info = save_rollout_hdf5(
                    hdf5_out,
                    rollout_result["episodes"],
                    env_name=rollout_result["runtimeEnvName"],
                    generation_mode=generation_mode,
                    policy_mode=str(rollout_result.get("policyMode") or "partial_scripted"),
                )

            if rollout_result:
                runtime_env = rollout_result.get("runtimeEnvName") or runtime_env
                status_payload["successfulEpisodes"] = rollout_result.get("successEpisodes", 0)
                status_payload["failedEpisodes"] = rollout_result.get("failedEpisodes", 0)
                status_payload["successRate"] = rollout_result.get("successRate")
                status_payload["failureDistribution"] = rollout_result.get("failureDistribution")
                status_payload["policyMode"] = rollout_result.get("policyMode")
                status_payload["runtimeEnvName"] = runtime_env
                if rollout_result.get("fallbackReason"):
                    status_payload["envFallbackReason"] = rollout_result["fallbackReason"]

                if args.render_video:
                    vr = rollout_result.get("videoResult") or {}
                    if vr.get("ok"):
                        video_status = "available"
                        status_payload["generateVideoExists"] = True
                        status_payload["generateVideo"] = str(video_path)
                    else:
                        video_status = "failed"
                        video_error = str(vr.get("error") or "video_render_failed")
                        status_payload["videoStatus"] = video_status
                        status_payload["videoError"] = video_error

        if not hdf5_out.is_file():
            raise RuntimeError("hdf5_write_failed: output file missing")

        set_job_stage(job_root, status_payload, stage="write_summary", progress=95)
        _update_status(job_root, status_payload)

        if generation_mode == "mimicgen_datagen" and not hdf5_info:
            hdf5_info = inspect_hdf5_dataset(hdf5_out)

        policy_mode = "mimicgen" if generation_mode == "mimicgen_datagen" else str(
            (rollout_result or {}).get("policyMode") or "partial_scripted"
        )

        summary = _build_summary(
            job_root=job_root,
            args=args,
            generation_mode=generation_mode,
            runtime_env=runtime_env,
            rollout_result=rollout_result,
            hdf5_info=hdf5_info,
            mimic_result=mimic_result,
            video_status=video_status,
            video_error=video_error,
            source_demo_path=source_demo_resolved,
            source_demo_audit=source_demo_audit,
            pinn_summary=pinn_summary,
            physics_enhancement=physics_enhancement,
        )
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        set_job_stage(job_root, status_payload, stage="write_manifest", progress=98)
        _update_status(job_root, status_payload)

        manifest = {
            "jobId": job_root.name,
            "taskTemplateId": args.task_template_id,
            "taskName": "螺母装配",
            "taskType": "nut_assembly",
            "source": "mimicgen_robosuite",
            "generationMode": generation_mode,
            "policyMode": policy_mode,
            "sourceDemoPath": source_demo_resolved,
            "sourceDemoOrigin": summary.get("sourceDemoOrigin"),
            "sourceDemoOriginReason": summary.get("sourceDemoOriginReason"),
            "sourceDemoMd5": summary.get("sourceDemoMd5"),
            "sourceDemoHash": summary.get("sourceDemoHash"),
            "sourceDemoEnvName": summary.get("sourceDemoEnvName"),
            "sourceEnvName": args.env_name,
            "runtimeEnvName": runtime_env,
            "episodesRequested": args.episodes,
            "episodesGenerated": summary["episodesGenerated"],
            "successEpisodes": summary.get("successEpisodes"),
            "failedEpisodes": summary.get("failedEpisodes"),
            "successRate": summary.get("successRate"),
            "successStatus": summary.get("successStatus"),
            "failureDistribution": summary.get("failureDistribution"),
            "validForTrainingEpisodes": summary.get("validForTrainingEpisodes"),
            "graspSuccessEpisodes": summary.get("graspSuccessEpisodes"),
            "liftSuccessEpisodes": summary.get("liftSuccessEpisodes"),
            "alignmentSuccessEpisodes": summary.get("alignmentSuccessEpisodes"),
            "insertionSuccessEpisodes": summary.get("insertionSuccessEpisodes"),
            "averageGraspAttempts": summary.get("averageGraspAttempts"),
            "hasStageStatistics": summary.get("hasStageStatistics"),
            "validEpisodes": summary.get("validEpisodes") or summary.get("successEpisodes"),
            "datasetPath": "datasets/nut_assembly_generated.hdf5",
            "hdf5Path": str(hdf5_out.relative_to(job_root)),
            "simulatorBackend": "mujoco",
            "datasetFormat": "robomimic_hdf5",
            "demoCount": summary["demoCount"],
            "totalSteps": summary["totalSteps"],
            "hasEpisodeMetadata": summary.get("hasEpisodeMetadata"),
            "hasDatagenInfo": summary.get("hasDatagenInfo"),
            "hasObjectPoses": summary.get("hasObjectPoses"),
            "objectPoseKeys": summary.get("objectPoseKeys") or [],
            "datagenFailedTrials": summary.get("datagenFailedTrials"),
            "datagenSuccessRate": summary.get("datagenSuccessRate"),
            "mimicgenStats": summary.get("mimicgenStats"),
            "physicsEnhancementEnabled": summary.get("physicsEnhancementEnabled"),
            "enhancementMode": summary.get("enhancementMode"),
            "pinnModelId": summary.get("pinnModelId"),
            "mimicgenGeneratedDemos": summary.get("mimicgenGeneratedDemos"),
            "rawDemoCount": summary.get("rawDemoCount"),
            "repairedDemoCount": summary.get("repairedDemoCount"),
            "finalDemoCount": summary.get("finalDemoCount"),
            "pinnRepairValidationRate": summary.get("pinnRepairValidationRate"),
            "enhancementGain": summary.get("enhancementGain"),
            "enhancementStatus": summary.get("enhancementStatus"),
            "videoPath": summary.get("videoPath"),
            "videoStatus": video_status,
            "status": "success",
            "outputName": args.output_name,
        }
        if summary.get("fallbackFrom"):
            manifest["fallbackFrom"] = summary["fallbackFrom"]
            manifest["fallbackReason"] = summary.get("fallbackReason")

        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        status_payload.update(
            {
                "status": "completed",
                "stage": "completed",
                "message": "数据生成已完成",
                "progress": 100,
                "generationMode": generation_mode,
                "policyMode": policy_mode,
                "hasDatagenInfo": summary.get("hasDatagenInfo"),
                "hasObjectPoses": summary.get("hasObjectPoses"),
                "objectPoseKeys": summary.get("objectPoseKeys"),
                "sourceDemoPath": source_demo_resolved,
                "sourceDemoOrigin": summary.get("sourceDemoOrigin"),
                "sourceDemoOriginReason": summary.get("sourceDemoOriginReason"),
                "episodesGenerated": summary.get("finalDemoCount") or summary.get("episodesGenerated"),
                "datagenFailedTrials": summary.get("datagenFailedTrials"),
                "finalDemoCount": summary.get("finalDemoCount"),
                "pinnValidationSucceeded": summary.get("pinnValidationSucceeded"),
                "physicsEnhancementEnabled": summary.get("physicsEnhancementEnabled"),
                "savedHdf5": str(hdf5_out),
                "savedManifest": str(manifest_path),
                "savedSummary": str(summary_path),
                "datasetReady": True,
                "videoStatus": video_status,
            }
        )
        if video_status == "available":
            status_payload["generateVideoExists"] = True
            status_payload["generateVideo"] = str(video_path)
        _update_status(job_root, status_payload)

        log_lines.extend(
            [
                f"generation_mode={generation_mode}",
                f"policy_mode={policy_mode}",
                f"demo_count={summary.get('demoCount')}",
                f"hdf5={hdf5_out}",
            ]
        )
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"generation_mode: {generation_mode}")
        print(f"policy_mode: {policy_mode}")
        print(f"demo_count: {summary.get('demoCount')}")
        print(f"saved_hdf5: {hdf5_out}")
        print(f"saved_manifest: {manifest_path}")
        print(f"saved_summary: {summary_path}")
        return 0
    except Exception as exc:
        tb = traceback.format_exc()
        log_lines.append(f"error={exc}")
        log_lines.append(tb)
        log_path.write_text("\n".join(log_lines), encoding="utf-8")
        status_payload.update(
            {
                "fallbackAttempted": fallback_attempted,
                "fallbackSucceeded": fallback_succeeded if fallback_attempted else None,
                "mimicgenError": mimicgen_error,
                "fallbackError": fallback_error or (str(exc) if fallback_attempted else None),
            }
        )
        failure_reason = "rollout_fallback_failed" if fallback_attempted else "robosuite_runtime_failed"
        if "wrong_robosuite_source" in str(exc):
            failure_reason = "wrong_robosuite_source"
        mark_job_failed(
            job_root,
            status_payload,
            error=str(exc),
            failure_reason=failure_reason,
            traceback_text=tb,
        )
        print(f"error: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="NutAssembly data generation worker")
    parser.add_argument("--job-root", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--env-name", default="Square_D0")
    parser.add_argument("--task-template-id", default="nut_assembly_single_arm")
    parser.add_argument("--output-name", default="nut_assembly_dataset")
    parser.add_argument("--source-demo-path", default=None)
    parser.add_argument(
        "--source-demo-selection",
        default=None,
        choices=["official", "local", "custom", "auto"],
        help="Source demo selection: official registry file, local debug demo, custom path, or auto",
    )
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument(
        "--generation-mode",
        default="mimicgen_datagen",
        choices=["mimicgen_datagen", "robosuite_rollout"],
        help="Preferred generation mode; mimicgen_datagen may fallback to robosuite_rollout on failure",
    )
    parser.add_argument(
        "--physics-enhancement-config",
        default=None,
        help="Path to physicsEnhancement JSON config for optional PINN repair stage",
    )
    return run_generation(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
