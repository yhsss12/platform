from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from utils.episode_eval import (
    RolloutPhase,
    build_episode_metadata,
    build_stage_debug_record,
    check_episode_success,
    classify_failure_type,
    compute_pose_errors,
    is_out_of_bounds,
    verify_success_stable,
    _eef_pos,
    _is_grasping,
    _nut_body_pos,
    _nut_grasp_pos,
    _peg_body_pos,
)
from utils.scripted_policy import ScriptedPolicyState, compute_scripted_action, policy_mode_label
from utils.video_writer import write_video

_INTEGRATION_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _INTEGRATION_ROOT.parents[1]
_CABLE_MVP = _REPO_ROOT / "integrations" / "CableThreadingMVP"

_DEBUG_LOG_INTERVAL = 15


def _ensure_robosuite_path() -> None:
    cable_str = str(_CABLE_MVP)
    if cable_str not in sys.path:
        sys.path.insert(0, cable_str)


def resolve_runtime_env(source_env_name: str) -> tuple[str, str]:
    preferred = (source_env_name or "Square_D0").strip()
    if preferred in {"NutAssemblySquare", "NutAssembly", "NutAssemblyRound"}:
        return preferred, ""
    if preferred in {"Square_D0", "NutAssembly_D0"}:
        return "NutAssemblySquare", f"{preferred} not registered; fallback NutAssemblySquare"
    return "NutAssemblySquare", f"unknown env {preferred}; fallback NutAssemblySquare"


def _body_pose_4x4(sim, body_name: str) -> np.ndarray:
    import robosuite.utils.transform_utils as T

    body_id = sim.model.body_name2id(body_name)
    pos = np.array(sim.data.body_xpos[body_id], dtype=np.float64)
    rot = np.array(sim.data.body_xmat[body_id], dtype=np.float64).reshape(3, 3)
    return T.make_pose(pos, rot).astype(np.float32)


def _collect_object_poses(env) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    sim = env.sim
    if hasattr(env, "nut_to_id"):
        for nut_key in env.nut_to_id:
            body = f"{nut_key}_nut"
            try:
                sim.model.body_name2id(body)
                poses[nut_key] = _body_pose_4x4(sim, body)
            except Exception:
                pass
            try:
                poses.setdefault(nut_key, _body_pose_4x4(sim, f"{nut_key.capitalize()}Nut"))
            except Exception:
                pass
    for nut in getattr(env, "nuts", []):
        try:
            key = nut.name.replace("Nut", "").lower()
            body_name = str(getattr(nut, "root_body", "") or nut.name)
            poses[key] = _body_pose_4x4(sim, body_name)
        except Exception:
            pass
    for peg_key, peg_body in (("square", "peg1"), ("round", "peg2")):
        try:
            sim.model.body_name2id(peg_body)
            poses[f"{peg_key}_peg"] = _body_pose_4x4(sim, peg_body)
        except Exception:
            pass
    return poses


def _render_frame(env, *, width: int = 640, height: int = 480) -> np.ndarray | None:
    try:
        img = env.sim.render(camera_name="agentview", width=width, height=height)
        return np.flipud(np.asarray(img))
    except Exception:
        return None


def _stage_success_flag(state: ScriptedPolicyState) -> bool:
    phase = state.phase
    if phase == RolloutPhase.APPROACH_NUT:
        return state.phase_step > 0
    if phase == RolloutPhase.DESCEND_TO_GRASP:
        return state.phase_step > 0
    if phase == RolloutPhase.CLOSE_GRIPPER:
        return state.grasp_stable_count > 0 or state.grasp_success
    if phase == RolloutPhase.SETTLE_AFTER_GRASP:
        return state.grasp_success
    if phase == RolloutPhase.LIFT_NUT:
        return state.lift_success
    if phase == RolloutPhase.MOVE_TO_PEG:
        return state.phase_step > 10
    if phase == RolloutPhase.ALIGN_OVER_PEG:
        return state.alignment_success
    if phase == RolloutPhase.DESCEND_INSERT:
        return state.insertion_success
    if phase == RolloutPhase.RELEASE:
        return state.insertion_success
    if phase == RolloutPhase.VERIFY_SUCCESS:
        return state.insertion_success
    return False


def _append_stage_debug(
    debug_log: list[dict[str, Any]],
    *,
    episode: int,
    step: int,
    state: ScriptedPolicyState,
    env,
    failure_type: str | None = None,
) -> None:
    nut = _nut_grasp_pos(env, state.nut_name)
    nut_body = _nut_body_pos(env, state.nut_name)
    peg = _peg_body_pos(env, state.peg_id)
    eef = _eef_pos(env)
    errors = compute_pose_errors(env, nut_name=state.nut_name)
    grasping = _is_grasping(env, state.nut_name)
    record = build_stage_debug_record(
        episode=episode,
        stage=state.phase.name.lower(),
        nut_pos=(nut_body if nut_body is not None else nut).round(4).tolist()
        if (nut_body is not None or nut is not None)
        else None,
        peg_pos=peg.round(4).tolist() if peg is not None else None,
        eef_pos=eef.round(4).tolist(),
        xy_error=errors.get("final_xy_error") if peg is not None else errors.get("xy_error"),
        height_error=errors.get("height_error"),
        tilt_error=errors.get("tilt_error"),
        gripper_state="closed" if grasping else "open",
        stage_success=_stage_success_flag(state),
        grasp_attempts=state.grasp_attempts,
        failure_type=failure_type,
        extra={"step": step, "phase_step": state.phase_step},
    )
    debug_log.append(record)


def rollout_episodes(
    *,
    env_name: str,
    episodes: int,
    seed: int,
    horizon: int,
    render_video: bool = False,
    video_path: Path | None = None,
    debug_log_path: Path | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    extra_xy_bias: np.ndarray | None = None,
) -> dict[str, Any]:
    _ensure_robosuite_path()
    import robosuite

    rng = np.random.default_rng(seed)
    runtime_env, fallback_reason = resolve_runtime_env(env_name)
    generation_mode = "robosuite_rollout"
    policy_mode = "partial_scripted"

    env = robosuite.make(
        env_name=runtime_env,
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=render_video,
        use_camera_obs=False,
        control_freq=20,
        horizon=horizon,
    )

    collected: list[dict[str, Any]] = []
    success_count = 0
    failed_count = 0
    failure_counter: Counter[str] = Counter()
    grasp_success_count = 0
    lift_success_count = 0
    alignment_success_count = 0
    insertion_success_count = 0
    valid_for_training_count = 0
    grasp_attempts_total = 0
    final_xy_errors: list[float] = []
    final_height_errors: list[float] = []
    stage_debug_log: list[dict[str, Any]] = []
    video_frames: list[np.ndarray] = []
    video_result: dict[str, Any] = {"ok": False, "error": "not_requested"}

    try:
        for ep_idx in range(episodes):
            ep_seed = int(rng.integers(0, 2**31 - 1))
            obs = env.reset()
            policy_state = ScriptedPolicyState()
            if extra_xy_bias is not None:
                policy_state.extra_xy_bias = np.asarray(extra_xy_bias, dtype=np.float64)[:2]

            obs_history: dict[str, list[np.ndarray]] = {k: [] for k in obs}
            actions: list[np.ndarray] = []
            states: list[np.ndarray] = []
            object_pose_history: dict[str, list[np.ndarray]] = {}

            timed_out = True
            oob = False
            success_flag = False
            failure_type = "timeout"
            last_logged_phase = policy_state.phase

            for step in range(horizon):
                for key, value in obs.items():
                    obs_history[key].append(np.asarray(value, dtype=np.float32))
                try:
                    flat_state = env.sim.get_state().flatten()
                    states.append(np.asarray(flat_state, dtype=np.float32))
                except Exception:
                    pass

                poses = _collect_object_poses(env)
                for obj_key, mat in poses.items():
                    object_pose_history.setdefault(obj_key, []).append(mat)

                if render_video and (step % 5 == 0 or step == 0):
                    frame = _render_frame(env)
                    if frame is not None:
                        video_frames.append(frame)

                if step == 0 or policy_state.phase != last_logged_phase or step % _DEBUG_LOG_INTERVAL == 0:
                    _append_stage_debug(stage_debug_log, episode=ep_idx, step=step, state=policy_state, env=env)
                    last_logged_phase = policy_state.phase

                oob = is_out_of_bounds(env)
                if oob:
                    failure_type = "out_of_bounds"
                    timed_out = False
                    action = compute_scripted_action(env, policy_state)
                    actions.append(action)
                    obs, _reward, done, _info = env.step(action)
                    break

                if policy_state.done and not success_flag:
                    timed_out = False
                    break

                action = compute_scripted_action(env, policy_state)
                actions.append(action)
                obs, _reward, done, info = env.step(action)

                env_success, _method = check_episode_success(env)
                if policy_state.phase >= RolloutPhase.RELEASE and env_success:
                    success_flag = True
                    failure_type = "success"
                    timed_out = False
                    policy_state.insertion_success = True
                    break

                if done:
                    timed_out = False
                    env_success, _method = check_episode_success(env)
                    if env_success:
                        success_flag = True
                        failure_type = "success"
                        policy_state.insertion_success = True
                    break
            else:
                timed_out = True

            if not success_flag:
                env_success, _method = check_episode_success(env)
                if env_success:
                    success_flag = True
                    failure_type = "success"
                    policy_state.insertion_success = True
                    timed_out = False

            pose_errors = compute_pose_errors(env, nut_name=policy_state.nut_name)
            final_xy = pose_errors.get("final_xy_error")
            final_height = pose_errors.get("final_height_error")
            if isinstance(final_xy, float):
                final_xy_errors.append(final_xy)
            if isinstance(final_height, float):
                final_height_errors.append(final_height)

            if not success_flag:
                failure_type = classify_failure_type(
                    success=False,
                    max_phase=policy_state.max_phase,
                    grasp_success=policy_state.grasp_success,
                    lift_success=policy_state.lift_success,
                    alignment_success=policy_state.alignment_success,
                    insertion_attempted=policy_state.max_phase >= RolloutPhase.DESCEND_INSERT,
                    timed_out=timed_out,
                    out_of_bounds=oob,
                )

            if policy_state.grasp_success:
                grasp_success_count += 1
            if policy_state.lift_success:
                lift_success_count += 1
            if policy_state.alignment_success:
                alignment_success_count += 1
            if policy_state.insertion_success or success_flag:
                insertion_success_count += 1

            grasp_attempts_total += policy_state.grasp_attempts

            if success_flag:
                success_count += 1
                valid_for_training_count += 1
            else:
                failed_count += 1
            failure_counter[failure_type] += 1

            _append_stage_debug(
                stage_debug_log,
                episode=ep_idx,
                step=len(actions),
                state=policy_state,
                env=env,
                failure_type=failure_type,
            )

            ep_policy_mode = policy_mode_label(policy_state, used_scripted=True)
            metadata = build_episode_metadata(
                success_flag=success_flag,
                failure_type=failure_type,
                episode_steps=len(actions),
                env_name=runtime_env,
                generation_mode=generation_mode,
                policy_mode=ep_policy_mode,
                seed=ep_seed,
                episode_index=ep_idx,
                grasp_attempts=policy_state.grasp_attempts,
                valid_for_training=success_flag,
                final_xy_error=final_xy if isinstance(final_xy, float) else None,
                final_height_error=final_height if isinstance(final_height, float) else None,
                grasp_success=policy_state.grasp_success,
                lift_success=policy_state.lift_success,
                alignment_success=policy_state.alignment_success,
                insertion_success=policy_state.insertion_success or success_flag,
                max_phase=policy_state.max_phase.name,
            )

            datagen_info: dict[str, Any] = {"object_poses": {}}
            for obj_key, mats in object_pose_history.items():
                datagen_info["object_poses"][obj_key] = np.stack(mats, axis=0)

            collected.append(
                {
                    "obs": {k: np.stack(v, axis=0) for k, v in obs_history.items()},
                    "actions": np.stack(actions, axis=0) if actions else np.zeros((0, 7), dtype=np.float32),
                    "states": np.stack(states, axis=0) if states else None,
                    "datagen_info": datagen_info,
                    "success": success_flag,
                    "metadata": metadata,
                    "max_phase": int(policy_state.max_phase),
                }
            )

            if on_progress:
                on_progress(
                    {
                        "episode": ep_idx + 1,
                        "episodes": episodes,
                        "step": len(actions),
                        "successfulEpisodes": success_count,
                        "failedEpisodes": failed_count,
                        "validForTrainingEpisodes": valid_for_training_count,
                        "graspSuccessEpisodes": grasp_success_count,
                        "liftSuccessEpisodes": lift_success_count,
                        "insertionSuccessEpisodes": insertion_success_count,
                        "averageGraspAttempts": grasp_attempts_total / max(ep_idx + 1, 1),
                        "policyMode": policy_mode,
                        "generationMode": generation_mode,
                    }
                )
    finally:
        env.close()

    if debug_log_path is not None:
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_log_path.open("w", encoding="utf-8") as fh:
            for row in stage_debug_log:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    if render_video and video_path is not None:
        video_result = write_video(video_frames, video_path, fps=20)

    episodes_gen = len(collected)
    failure_distribution = {k: int(v) for k, v in failure_counter.items() if k != "success"}

    avg_xy = round(float(np.mean(final_xy_errors)), 4) if final_xy_errors else None
    avg_height = round(float(np.mean(final_height_errors)), 4) if final_height_errors else None

    return {
        "episodes": collected,
        "runtimeEnvName": runtime_env,
        "sourceEnvName": env_name,
        "fallbackReason": fallback_reason,
        "validEpisodes": success_count,
        "validForTrainingEpisodes": valid_for_training_count,
        "failedEpisodes": failed_count,
        "successEpisodes": success_count,
        "graspSuccessEpisodes": grasp_success_count,
        "liftSuccessEpisodes": lift_success_count,
        "alignmentSuccessEpisodes": alignment_success_count,
        "insertionSuccessEpisodes": insertion_success_count,
        "episodesGenerated": episodes_gen,
        "generationMode": generation_mode,
        "policyMode": policy_mode,
        "failureDistribution": failure_distribution,
        "successRate": success_count / max(episodes_gen, 1),
        "averageGraspAttempts": round(grasp_attempts_total / max(episodes_gen, 1), 3),
        "averageFinalXYError": avg_xy,
        "averageFinalHeightError": avg_height,
        "hasStageStatistics": True,
        "videoResult": video_result,
    }
