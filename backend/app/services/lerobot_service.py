import sys
import os
import logging
import asyncio
import traceback
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from app.services.asset_registration_service import upsert_converted_asset, DataAssetsSyncSessionLocal
from app.services.task_job_store import is_cancelled
from app.models.data_asset import ConversionJobAsset
from app.services.conversion_batch_service import recompute_conversion_batch_stats
import re
from mcap.reader import make_reader
from app.services.mcap_converter import _scan_topics_and_create_config

# Configure logging
logger = logging.getLogger(__name__)

# Add scripts/relman to path
# Use relative path: backend/app/services -> ../../../scripts/relman
RELMAN_PATH = Path(__file__).resolve().parents[3] / "scripts" / "relman"
if str(RELMAN_PATH) not in sys.path:
    sys.path.append(str(RELMAN_PATH))

# Import from scripts/relman
try:
    import flexible_mcap_to_hdf5
    import mcap_to_lerobot
except ImportError as e:
    logger.error(f"Failed to import relman scripts: {e}")
    # Fallback or re-raise depending on strictness
    raise

# Alias classes
TopicConfig = flexible_mcap_to_hdf5.TopicConfig
AlignmentConfig = flexible_mcap_to_hdf5.AlignmentConfig
LeRobotConfig = mcap_to_lerobot.LeRobotConfig


def _resolve_lerobot_output_dir(output_repo_id: str, dataset: Optional[Any] = None) -> Optional[str]:
    """
    将 LeRobot repo_id 解析为本地真实目录路径。
    优先使用 dataset 对象中的路径信息，兜底使用 HF_LEROBOT_HOME/repo_id。
    """
    # 1) 优先从 dataset 对象提取已存在目录
    if dataset is not None:
        for attr in ("repo_path", "path", "root", "dataset_path"):
            try:
                val = getattr(dataset, attr, None)
            except Exception:
                val = None
            if isinstance(val, (str, Path)):
                p = Path(val).expanduser().resolve()
                if p.exists() and p.is_dir():
                    return str(p)

    # 2) 兜底：HF_LEROBOT_HOME / repo_id
    try:
        home = getattr(mcap_to_lerobot, "HF_LEROBOT_HOME", None)
        if home:
            p = (Path(home) / str(output_repo_id or "").strip()).expanduser().resolve()
            if p.exists() and p.is_dir():
                return str(p)
    except Exception:
        pass
    return None

def convert_mcap_to_lerobot(
    mcap_path: str,
    output_repo_id: str,
    topic_configs: List[TopicConfig],
    alignment_config: AlignmentConfig,
    lerobot_config: LeRobotConfig,
    dataset: Optional[Any] = None,
) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Wrapper around mcap_to_lerobot.convert_mcap_to_lerobot
    第三项为失败时的可读原因（成功为 None）。
    """
    return mcap_to_lerobot.convert_mcap_to_lerobot(
        mcap_path=mcap_path,
        output_repo_id=output_repo_id,
        topic_configs=topic_configs,
        alignment_config=alignment_config,
        lerobot_config=lerobot_config,
        dataset=dataset
    )

async def convert_mcap_to_lerobot_task(
    job_id: str,
    mcap_path: str,
    output_repo_id: str,
    config: Dict[str, Any],
    jobs_store: Dict[str, Any]
):
    """
    Background task to convert MCAP to LeRobot dataset.
    """
    try:
        _update_job_status(jobs_store, job_id, "running", progress=10)
        if job_id in jobs_store:
            job0 = jobs_store[job_id]
            try:
                job0.currentStage = "Parse"
                job0.logs.append(f"[{datetime.now().isoformat()}] LeRobot 转换初始化...")
                job0.updatedAt = datetime.now().isoformat()
                _persist_conversion_job_to_assets_db(job0)
            except Exception:
                pass
        
        actual_topics = _scan_mcap_topic_names(mcap_path)

        # Parse configuration
        topic_configs = []
        raw_topics_cfg = config.get("topics", [])
        if isinstance(raw_topics_cfg, list) and raw_topics_cfg:
            for tc in raw_topics_cfg:
                topic_configs.append(TopicConfig(**tc))
            
        ac_dict = config.get("alignment", {})
        alignment_config = AlignmentConfig(**ac_dict)
        
        lr_dict = config.get("lerobot", {})
        if output_repo_id:
            lr_dict["repo_id"] = output_repo_id
        lerobot_config = LeRobotConfig(**lr_dict)

        if not topic_configs:
            # 优先从真实 MCAP 话题自动构造 TopicConfig，避免默认配置与数据源话题不一致导致 0 数据。
            discovered = _scan_topics_and_create_config(mcap_path)
            if discovered:
                topic_configs = [TopicConfig(**tc) for tc in discovered]
            else:
                required_topics = _extract_required_topics_from_lerobot_config(lerobot_config)
                topic_configs = _build_default_topic_configs(required_topics)

        _attach_topic_patterns_for_topic_mismatch(topic_configs, actual_topics, job_id, jobs_store)
        _adapt_lerobot_config_for_available_topics(lerobot_config, topic_configs, actual_topics, job_id, jobs_store)

        logger.info(f"Starting LeRobot conversion for job {job_id}, input: {mcap_path}")

        # Run conversion in executor
        progress_stop = asyncio.Event()

        async def _heartbeat_progress() -> None:
            # LeRobot 转换没有细粒度回调：用心跳推进避免前端长期停在 10%
            # 最多推进到 90%，最终状态仍由成功/失败分支覆盖（95/100 或 failed）。
            while not progress_stop.is_set():
                await asyncio.sleep(2.0)
                if progress_stop.is_set():
                    break
                job_hb = jobs_store.get(job_id)
                if job_hb is None:
                    continue
                try:
                    cur = int(getattr(job_hb, "progressPercent", 0) or 0)
                    if cur < 12:
                        nxt = 12
                    elif cur < 90:
                        nxt = min(90, cur + 3)
                    else:
                        nxt = cur
                    if nxt != cur:
                        job_hb.progressPercent = nxt
                        job_hb.currentStage = "Convert"
                        job_hb.updatedAt = datetime.now().isoformat()
                        _persist_conversion_job_to_assets_db(job_hb)
                except Exception:
                    continue

        hb_task = asyncio.create_task(_heartbeat_progress())
        loop = asyncio.get_running_loop()
        try:
            success, dataset, conv_err = await loop.run_in_executor(
                None,
                lambda: convert_mcap_to_lerobot(
                    mcap_path=mcap_path,
                    output_repo_id=output_repo_id,
                    topic_configs=topic_configs,
                    alignment_config=alignment_config,
                    lerobot_config=lerobot_config,
                    dataset=None 
                )
            )
        finally:
            progress_stop.set()
            try:
                await hb_task
            except Exception:
                pass

        if is_cancelled(job_id):
            _update_job_status(jobs_store, job_id, "canceled", error="Task cancelled")
            return
        
        if success:
            _update_job_status(jobs_store, job_id, "running", progress=95)
            if job_id in jobs_store:
                job = jobs_store[job_id]
                job.logs.append(f"[{datetime.now().isoformat()}] Converting finished, registering output asset...")
                job.updatedAt = datetime.now().isoformat()
                # LeRobot 转换产物是目录：需传入本地真实目录供统一“上传 MinIO + 入库”链路处理
                output_dir = _resolve_lerobot_output_dir(output_repo_id, dataset=dataset)
                if not output_dir:
                    raise RuntimeError(
                        f"LeRobot output directory not found for repo_id={output_repo_id}"
                    )
                if is_cancelled(job_id):
                    _update_job_status(jobs_store, job_id, "canceled", error="Task cancelled")
                    return
                upsert_converted_asset(job, mcap_path, output_dir)
                # 上传 MinIO 并登记资产后，清理本地目录，避免本地残留。
                try:
                    if os.path.isdir(output_dir):
                        shutil.rmtree(output_dir, ignore_errors=False)
                    elif os.path.isfile(output_dir):
                        os.remove(output_dir)
                except FileNotFoundError:
                    pass
                except Exception as cleanup_err:
                    logger.warning(f"Cleanup local LeRobot output failed: {output_dir}, error={cleanup_err}")
            _update_job_status(jobs_store, job_id, "succeeded", progress=100)
            logger.info(f"Job {job_id} succeeded")
        else:
            detail = (conv_err or "").strip() or "转换未成功但未返回具体原因，请查看服务端日志"
            _update_job_status(jobs_store, job_id, "failed", error=detail)

    except Exception as e:
        logger.error(f"LeRobot conversion failed: {e}")
        traceback.print_exc()
        _update_job_status(jobs_store, job_id, "failed", error=str(e))

def _update_job_status(store, job_id, status, progress=None, error=None):
    if job_id in store:
        job = store[job_id]
        job.status = status
        if progress is not None:
            job.progressPercent = progress
        if error:
            job.errorMessage = error
            job.logs.append(f"Error: {error}")
        job.updatedAt = datetime.now().isoformat()
        _persist_conversion_job_to_assets_db(job)

def _persist_conversion_job_to_assets_db(job: Any) -> None:
    session = DataAssetsSyncSessionLocal()
    try:
        job_id = str(getattr(job, "jobId", "") or "")
        if not job_id:
            return
        rec = session.query(ConversionJobAsset).filter(ConversionJobAsset.job_id == job_id).one_or_none()
        if rec is None:
            rec = ConversionJobAsset(job_id=job_id)
        rec.status = str(getattr(job, "status", "") or rec.status or "queued")
        rec.progress_percent = float(getattr(job, "progressPercent", 0) or 0)
        rec.current_stage = getattr(job, "currentStage", None)
        rec.artifact_ready = bool(getattr(job, "artifactReady", False))
        rec.error_message = getattr(job, "errorMessage", None)
        rec.operator_name = (
            (getattr(job, "operatorName", None) or getattr(job, "operator_name", None) or "").strip() or None
        )
        try:
            logs = getattr(job, "logs", None) or []
            rec.logs_json = json.dumps(list(logs), ensure_ascii=False)
        except Exception:
            rec.logs_json = rec.logs_json
        try:
            stages = getattr(job, "stages", None) or []
            rec.stages_json = json.dumps(
                [s.model_dump() if hasattr(s, "model_dump") else dict(s) for s in stages],
                ensure_ascii=False,
            )
        except Exception:
            rec.stages_json = rec.stages_json
        try:
            rec.updated_at = datetime.fromisoformat(getattr(job, "updatedAt", "").replace("Z", "+00:00"))
        except Exception:
            pass
        session.add(rec)
        session.commit()
        bid = getattr(rec, "batch_id", None)
        if bid:
            recompute_conversion_batch_stats(str(bid))
    except Exception:
        session.rollback()
    finally:
        session.close()

def _scan_mcap_topic_names(mcap_path: str) -> set:
    try:
        with open(mcap_path, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            if summary and getattr(summary, "channels", None):
                return {ch.topic for ch in summary.channels.values() if getattr(ch, "topic", None)}
            topics = set()
            for _, channel, _ in reader.iter_messages():
                if getattr(channel, "topic", None):
                    topics.add(channel.topic)
            return topics
    except Exception:
        return set()

def _extract_required_topics_from_lerobot_config(lerobot_config: Any) -> List[str]:
    topics: List[str] = []
    try:
        for src in getattr(lerobot_config, "state_sources", []) or []:
            t = src.get("topic") if isinstance(src, dict) else None
            if t:
                topics.append(str(t))
    except Exception:
        pass
    try:
        cam_map = getattr(lerobot_config, "camera_mapping", {}) or {}
        if isinstance(cam_map, dict):
            for t in cam_map.values():
                if t:
                    topics.append(str(t))
    except Exception:
        pass
    seen = set()
    ordered: List[str] = []
    for t in topics:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    return ordered

def _build_default_topic_configs(required_topics: List[str]) -> List[TopicConfig]:
    configs: List[TopicConfig] = []
    for topic in required_topics:
        t = str(topic)
        if t.endswith("/joint_states"):
            configs.append(
                TopicConfig(
                    topic_name=t,
                    message_type="sensor_msgs/msg/JointState",
                    hdf5_path=t,
                    data_type="float32",
                    description="auto: joint_states",
                    custom_processor="joint_state",
                    custom_params={"joint_count": -1},
                )
            )
        elif "image" in t:
            if t.endswith("/compressed") or "/compressed" in t:
                configs.append(
                    TopicConfig(
                        topic_name=t,
                        message_type="sensor_msgs/msg/CompressedImage",
                        hdf5_path=t,
                        data_type="uint8",
                        description="auto: compressed_image",
                        custom_processor="compressed_image",
                        custom_params={},
                    )
                )
            else:
                configs.append(
                    TopicConfig(
                        topic_name=t,
                        message_type="sensor_msgs/msg/Image",
                        hdf5_path=t,
                        data_type="uint8",
                        description="auto: image",
                        custom_processor="image",
                        custom_params={},
                    )
                )
        else:
            tl = t.lower()
            # 末端六维力（Sixforce）
            if "get_force_data_result" in tl:
                configs.append(
                    TopicConfig(
                        topic_name=t,
                        message_type="rm_ros_interfaces/msg/Sixforce",
                        hdf5_path=t,
                        data_type="float64",
                        description="auto: sixforce",
                        custom_processor="sixforce",
                        custom_params={},
                    )
                )
            # 处理后的六维力（Float64MultiArray: data=[Fx,Fy,Fz,Mx,My,Mz]）
            elif "float64multiarray" in tl or ("processed" in tl and "force" in tl):
                configs.append(
                    TopicConfig(
                        topic_name=t,
                        message_type="std_msgs/msg/Float64MultiArray",
                        hdf5_path=t,
                        data_type="float64",
                        description="auto: float64_multiarray(force)",
                        custom_processor="float64_multiarray",
                        custom_params={},
                    )
                )
            else:
                configs.append(
                    TopicConfig(
                        topic_name=t,
                        message_type="std_msgs/msg/Float32",
                        hdf5_path=t,
                        data_type="float32",
                        description="auto: scalar",
                        custom_processor="float32",
                        custom_params={},
                    )
                )
    return configs

def _attach_topic_patterns_for_topic_mismatch(topic_configs: List[TopicConfig], actual_topics: set, job_id: str, jobs_store: Dict[str, Any]):
    if not actual_topics:
        return
    for cfg in topic_configs:
        canonical = getattr(cfg, "topic_name", None)
        if not canonical:
            continue
        if canonical in actual_topics:
            continue
        matched = _guess_topic_match(canonical, actual_topics)
        if not matched:
            continue
        existing = getattr(cfg, "topic_patterns", None)
        patterns: List[str] = []
        if isinstance(existing, list):
            patterns.extend([str(p) for p in existing if p])
        patterns.extend([str(matched), str(canonical)])
        deduped: List[str] = []
        seen = set()
        for p in patterns:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        cfg.topic_patterns = deduped
        job = jobs_store.get(job_id)
        if job:
            job.logs.append(f"[{datetime.now().isoformat()}] 话题映射: {matched} -> {canonical}")
            job.updatedAt = datetime.now().isoformat()

def _guess_topic_match(canonical: str, actual_topics: set) -> Optional[str]:
    c = canonical.lower()
    candidates = list(actual_topics)
    if canonical.endswith("/joint_states"):
        candidates = [t for t in candidates if str(t).endswith("/joint_states")]
        if "/left/" in c:
            candidates = [t for t in candidates if "left" in str(t).lower()]
        elif "/right/" in c:
            candidates = [t for t in candidates if "right" in str(t).lower()]
    elif "gripper" in c:
        candidates = [t for t in candidates if "gripper" in str(t).lower()]
        if "left" in c:
            candidates = [t for t in candidates if "left" in str(t).lower()]
        elif "right" in c:
            candidates = [t for t in candidates if "right" in str(t).lower()]
    elif "camera" in c and "image" in c:
        m = re.search(r"camera(\d+)", c)
        if m:
            idx = m.group(1)
            candidates = [t for t in candidates if f"camera{idx}" in str(t).lower()]
        if "color" in c:
            candidates = [t for t in candidates if "color" in str(t).lower() or "rgb" in str(t).lower()]
        if "depth" in c:
            candidates = [t for t in candidates if "depth" in str(t).lower()]
        candidates = [t for t in candidates if "image" in str(t).lower()]

    if not candidates:
        return None

    tokens = [s for s in canonical.split("/") if s]
    def score(topic: str) -> int:
        lt = str(topic).lower()
        return sum(1 for tok in tokens if tok.lower() in lt)
    candidates.sort(key=lambda t: (score(t), -len(str(t))), reverse=True)
    return str(candidates[0])


def _adapt_lerobot_config_for_available_topics(
    lerobot_config: Any,
    topic_configs: List[TopicConfig],
    actual_topics: set,
    job_id: str,
    jobs_store: Dict[str, Any],
) -> None:
    """
    根据实际可用话题自适配 LeRobot 配置，提升跨设备/跨项目兼容性：
    - state_sources: 缺失时自动回退到实际 joint/gripper 话题
    - camera_mapping: 若配置的话题不存在，自动挑选可用 color 图像话题
    """

    def _log(msg: str) -> None:
        job = jobs_store.get(job_id)
        if job:
            job.logs.append(f"[{datetime.now().isoformat()}] {msg}")
            job.updatedAt = datetime.now().isoformat()

    topic_names = [str(getattr(tc, "topic_name", "") or "") for tc in topic_configs]
    topic_name_set = {t for t in topic_names if t}

    def _candidate_joint_topics() -> List[str]:
        out = [t for t in topic_names if t.endswith("/joint_states")]
        if not out:
            out = [t for t in topic_names if "joint_states" in t]
        return list(dict.fromkeys(out))

    def _candidate_gripper_topics() -> List[str]:
        out = [t for t in topic_names if "gripper" in t and "state" in t]
        if not out:
            out = [t for t in topic_names if "gripper" in t]
        return list(dict.fromkeys(out))

    def _candidate_color_topics() -> List[str]:
        out = []
        for t in topic_names:
            tl = t.lower()
            if "image" not in tl:
                continue
            if "depth" in tl:
                continue
            if "color" in tl or "rgb" in tl:
                out.append(t)
        return list(dict.fromkeys(out))

    def _topic_kind(topic: str) -> str:
        tl = str(topic or "").lower()
        if "joint_states" in tl:
            return "joint"
        if "gripper" in tl:
            return "gripper"
        if "force" in tl or "get_force_data_result" in tl or "sixforce" in tl:
            return "force"
        return "other"

    # 1) state_sources 适配
    srcs = getattr(lerobot_config, "state_sources", None) or []
    adapted_state_sources: List[Dict[str, str]] = []
    for src in srcs:
        if not isinstance(src, dict):
            continue
        canonical = str(src.get("topic") or "").strip()
        if not canonical:
            continue
        if canonical in topic_name_set:
            adapted_state_sources.append(dict(src))
            continue
        matched = _guess_topic_match(canonical, actual_topics) or _guess_topic_match(canonical, topic_name_set)
        if matched:
            # 防止语义错配：joint/gripper/force 必须同类映射
            ck = _topic_kind(canonical)
            mk = _topic_kind(matched)
            if ck in ("joint", "gripper", "force") and ck != mk:
                continue
            s2 = dict(src)
            s2["topic"] = matched
            adapted_state_sources.append(s2)
            _log(f"LeRobot state_sources 话题自适配: {canonical} -> {matched}")

    if not adapted_state_sources:
        fallback_sources: List[Dict[str, str]] = []
        for t in _candidate_joint_topics():
            fallback_sources.append({"topic": t, "field": "data"})
        for t in _candidate_gripper_topics():
            fallback_sources.append({"topic": t, "field": "data"})
        if fallback_sources:
            lerobot_config.state_sources = fallback_sources
            _log(f"LeRobot state_sources 自动回退为 {len(fallback_sources)} 个可用话题")
    else:
        lerobot_config.state_sources = adapted_state_sources

    # 2) camera_mapping 适配
    cam_map = getattr(lerobot_config, "camera_mapping", None) or {}
    adapted_cam_map: Dict[str, str] = {}
    if isinstance(cam_map, dict):
        for cam_name, canonical in cam_map.items():
            key = str(cam_name or "").strip()
            ctp = str(canonical or "").strip()
            if not key or not ctp:
                continue
            if ctp in topic_name_set:
                adapted_cam_map[key] = ctp
                continue
            matched = _guess_topic_match(ctp, actual_topics) or _guess_topic_match(ctp, topic_name_set)
            if matched:
                adapted_cam_map[key] = matched
                _log(f"LeRobot camera_mapping 话题自适配: {ctp} -> {matched}")

    if not adapted_cam_map:
        colors = _candidate_color_topics()
        if colors:
            generated: Dict[str, str] = {}
            for i, t in enumerate(colors[:3], start=1):
                generated[f"cam_extra_{i}"] = t
            lerobot_config.camera_mapping = generated
            _log(f"LeRobot camera_mapping 自动回退为 {len(generated)} 路彩色图像话题")
    else:
        lerobot_config.camera_mapping = adapted_cam_map
