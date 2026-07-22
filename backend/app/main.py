"""
FastAPI 应用入口
"""

# 在导入 app.* 其他模块前，最早期加载 .env，让 os.getenv 在整个进程中可用
try:
    from app.core.env_loader import ensure_dotenv_loaded

    _loaded_env_files = ensure_dotenv_loaded(verbose=True)
    if not _loaded_env_files:
        import os

        has_runtime = any(
            os.getenv(k)
            for k in (
                "DATABASE_URL",
                "JWT_SECRET",
                "SECRET_KEY",
                "REDIS_HOST",
            )
        )
        if has_runtime:
            print("ℹ 未挂载 .env 文件，使用容器/进程已注入的环境变量")
        else:
            from app.core.env_loader import backend_root, project_root

            print("⚠ 警告: 未找到 .env 且缺少关键环境变量，请检查 DATABASE_URL / JWT_SECRET 等")
            print(f"   查找位置: {project_root() / '.env'} 或 {backend_root() / '.env'}")
except Exception as e:  # pragma: no cover
    # 不阻塞启动，只打印提示
    print(f"⚠ 警告: 自动加载 .env 失败: {e}")

# 尽早挂载运行日志落盘（全量 root + uvicorn 汇总），失败不阻塞启动
try:
    from app.core.runtime_logging import configure_runtime_file_logging

    configure_runtime_file_logging()
except Exception as e:  # pragma: no cover
    print(f"⚠ 运行日志落盘初始化失败（不影响启动）: {e}")

from fastapi import FastAPI, Request, Response, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, inspect
from pathlib import Path
from app.api.router import api_router
from app.services.queue_backpressure import QueueBackpressureError
from app.db.session import AsyncSessionLocal
from app.crud.user import get_or_create_admin_user
from app.schemas.common import ApiResponse
import traceback
import asyncio
import logging
import os
import signal
import subprocess
import sys


class _IgnoreExperimentEvent404Filter(logging.Filter):
    """静默无害的 /api/experiment/event 404 access 日志。"""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        if '"/api/experiment/event' in msg and " 404 " in msg:
            return False
        return True


_uvicorn_access_logger = logging.getLogger("uvicorn.access")
_uvicorn_access_logger.addFilter(_IgnoreExperimentEvent404Filter())


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _rq_worker_running_for_queue(queue: str) -> bool:
    """检测是否已有监听该队列的 start_worker 进程（按队列独立判断，避免个别 worker 挂掉后整体被误判为已就绪）。"""
    q = (queue or "").strip()
    if not q:
        return False
    try:
        pat = f"start_worker.py --queues {q}"
        r = subprocess.run(
            ["pgrep", "-f", pat],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return r.returncode == 0
    except Exception:
        return False


def _start_queue_workers_if_needed() -> list[subprocess.Popen]:
    if not _env_bool("USE_QUEUE", False):
        return []
    if not _env_bool("AUTO_START_WORKERS", True):
        return []
    # main.py 位于 backend/app/，这里显式区分：backend_dir=backend/，project_root=项目根
    backend_dir = Path(__file__).resolve().parents[1]
    project_root = backend_dir.parent
    worker_entry = backend_dir / "worker" / "start_worker.py"
    if not worker_entry.exists():
        print(f"⚠ 自动启动队列 worker 失败：未找到 {worker_entry}")
        return []

    queue_names = [
        os.getenv("QUEUE_ANNOTATION", "gpu_queue").strip() or "gpu_queue",
        os.getenv("QUEUE_CONVERSION", "cpu_queue").strip() or "cpu_queue",
        os.getenv("QUEUE_COLLECT", "collect_queue").strip() or "collect_queue",
        os.getenv("QUEUE_BATCH", "io_queue").strip() or "io_queue",
    ]
    seen = set()
    deduped = []
    for q in queue_names:
        if q in seen:
            continue
        seen.add(q)
        deduped.append(q)

    if all(_rq_worker_running_for_queue(q) for q in deduped):
        print(f"ℹ️ RQ 各队列 worker 已在运行（{', '.join(deduped)}），跳过自动拉起")
        return []

    procs: list[subprocess.Popen] = []
    log_dir = project_root / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    for q in deduped:
        if _rq_worker_running_for_queue(q):
            print(f"ℹ️ 队列 {q} 已有 worker，跳过启动")
            continue
        try:
            log_path = log_dir / f"rq-worker-{q}.log"
            out_f = open(log_path, "a", encoding="utf-8", errors="ignore")
            p = subprocess.Popen(
                [sys.executable, str(worker_entry), "--queues", q],
                cwd=str(backend_dir),
                stdout=out_f,
                stderr=out_f,
            )
            procs.append(p)
            print(f"✅ 已自动启动队列 worker: queue={q}, pid={p.pid}, log={log_path}")
        except Exception as e:
            print(f"⚠ 自动启动队列 worker 失败（queue={q}）: {e}")
    return procs


async def _stop_queue_workers(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        try:
            if p.poll() is not None:
                continue
            p.send_signal(signal.SIGTERM)
        except Exception:
            continue
    await asyncio.sleep(0.2)
    for p in procs:
        try:
            if p.poll() is None:
                p.kill()
        except Exception:
            pass


class EpisodeStorageClearASGIMiddleware:
    """
    请求结束后清理 episode 解析缓存（与原先 @app.middleware("http") 语义一致）。

    不使用 BaseHTTPMiddleware（@app.middleware("http") 的实现），避免与部分路由组合时
    Starlette 抛出 RuntimeError('No response returned.')，例如 POST /api/webrtc/offer。
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        finally:
            try:
                from app.services.episode_storage import clear_episode_cache

                clear_episode_cache()
            except Exception:
                pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    cache_cleanup_stop_event = asyncio.Event()
    cache_cleanup_task: asyncio.Task | None = None
    db_backup_stop_event = asyncio.Event()
    db_backup_task: asyncio.Task | None = None
    telemetry_stop_event = asyncio.Event()
    telemetry_task: asyncio.Task | None = None
    worker_processes: list[subprocess.Popen] = []

    try:
        from app.core.config import settings
        from app.services.public_endpoint_resolver import get_public_endpoint_resolver

        get_public_endpoint_resolver(default_base_url=str(getattr(settings, "PUBLIC_BASE_URL", "") or "")).refresh()
    except Exception:
        pass
    # 启动时：初始化 PostgreSQL 业务表（Jobs/Tasks/Run/Dataset/HDF5/设备，同一 metadata）
    # 生产/联调环境请优先执行: alembic upgrade head
    from app.db.session import engine
    from app.db.base import Base as MainOrmBase
    import app.models.job
    import app.models.task
    import app.models.run
    import app.models.dataset
    import app.models.hdf5_dataset
    import app.models.workspace_job  # noqa: F401
    import app.models.workspace_index  # noqa: F401
    try:
        import app.models.device  # noqa: F401
    except ImportError:
        pass

    try:
        async with engine.begin() as conn:
            await conn.run_sync(MainOrmBase.metadata.create_all)
            def migrate_jobs_schema(connection):
                import json
                inspector = inspect(connection)
                if "tasks" in inspector.get_table_names():
                    columns = [c["name"] for c in inspector.get_columns("tasks")]
                    if "project_id" not in columns:
                        connection.execute(text("ALTER TABLE tasks ADD COLUMN project_id VARCHAR"))
                    if "project_name" not in columns:
                        connection.execute(text("ALTER TABLE tasks ADD COLUMN project_name VARCHAR"))
                    try:
                        rows = connection.execute(
                            text("SELECT id, description FROM tasks WHERE project_id IS NULL AND description IS NOT NULL")
                        ).fetchall()
                        for tid, desc in rows:
                            try:
                                cfg = json.loads(desc) if desc else None
                            except Exception:
                                cfg = None
                            if not isinstance(cfg, dict):
                                continue
                            pid = (cfg.get("projectId") or cfg.get("project_id") or "").strip()
                            pname = (cfg.get("projectName") or cfg.get("project_name") or "").strip()
                            if pid:
                                connection.execute(
                                    text("UPDATE tasks SET project_id = :pid, project_name = :pname WHERE id = :tid"),
                                    {"pid": pid, "pname": (pname or None), "tid": str(tid)},
                                )
                    except Exception:
                        pass
                if "jobs" in inspector.get_table_names():
                    columns = [c["name"] for c in inspector.get_columns("jobs")]
                    if "collection_quantity" not in columns:
                        connection.execute(text("ALTER TABLE jobs ADD COLUMN collection_quantity INTEGER DEFAULT 0"))
                    if "completed_count" not in columns:
                        connection.execute(text("ALTER TABLE jobs ADD COLUMN completed_count INTEGER DEFAULT 0"))
                    if "project_id" not in columns:
                        connection.execute(text("ALTER TABLE jobs ADD COLUMN project_id VARCHAR"))
                    if "project_name" not in columns:
                        connection.execute(text("ALTER TABLE jobs ADD COLUMN project_name VARCHAR"))
                    try:
                        rows = connection.execute(
                            text(
                                "SELECT jobs.id, tasks.project_id, tasks.project_name "
                                "FROM jobs JOIN tasks ON jobs.task_id = tasks.id "
                                "WHERE (jobs.project_id IS NULL OR jobs.project_id = '') AND tasks.project_id IS NOT NULL"
                            )
                        ).fetchall()
                        for jid, pid, pname in rows:
                            connection.execute(
                                text("UPDATE jobs SET project_id = :pid, project_name = :pname WHERE id = :jid"),
                                {"pid": pid, "pname": pname, "jid": str(jid)},
                            )
                    except Exception:
                        pass
            def migrate_hdf5_devices_schema(connection):
                inspector = inspect(connection)
                if "hdf5_datasets" in inspector.get_table_names():
                    columns = [c["name"] for c in inspector.get_columns("hdf5_datasets")]
                    if "source" not in columns:
                        print("🔄 正在迁移数据库: 添加 hdf5_datasets.source 字段...")
                        connection.execute(text("ALTER TABLE hdf5_datasets ADD COLUMN source VARCHAR DEFAULT 'local'"))
                    if "uploader" not in columns:
                        print("🔄 正在迁移数据库: 添加 hdf5_datasets.uploader 字段...")
                        connection.execute(text("ALTER TABLE hdf5_datasets ADD COLUMN uploader VARCHAR"))
                if "device_launch_configs" in inspector.get_table_names():
                    columns = [c["name"] for c in inspector.get_columns("device_launch_configs")]
                    if "stop_script_path" not in columns:
                        print("🔄 正在迁移数据库: 添加 device_launch_configs.stop_script_path 字段...")
                        connection.execute(text("ALTER TABLE device_launch_configs ADD COLUMN stop_script_path VARCHAR"))
                    if "stop_script_args" not in columns:
                        print("🔄 正在迁移数据库: 添加 device_launch_configs.stop_script_args 字段...")
                        connection.execute(text("ALTER TABLE device_launch_configs ADD COLUMN stop_script_args VARCHAR"))
                if "devices" in inspector.get_table_names():
                    columns = [c["name"] for c in inspector.get_columns("devices")]
                    if "hardware_uuid" not in columns:
                        print("🔄 正在迁移数据库: 添加 devices.hardware_uuid 字段...")
                        connection.execute(text("ALTER TABLE devices ADD COLUMN hardware_uuid VARCHAR"))
                    if "hostname" not in columns:
                        print("🔄 正在迁移数据库: 添加 devices.hostname 字段...")
                        connection.execute(text("ALTER TABLE devices ADD COLUMN hostname VARCHAR"))
                    if "agent_ip" not in columns:
                        print("🔄 正在迁移数据库: 添加 devices.agent_ip 字段...")
                        connection.execute(text("ALTER TABLE devices ADD COLUMN agent_ip VARCHAR"))
                    if "agent_port" not in columns:
                        print("🔄 正在迁移数据库: 添加 devices.agent_port 字段...")
                        connection.execute(text("ALTER TABLE devices ADD COLUMN agent_port INTEGER"))
                    if "agent_status" not in columns:
                        print("🔄 正在迁移数据库: 添加 devices.agent_status 字段...")
                        connection.execute(text("ALTER TABLE devices ADD COLUMN agent_status VARCHAR"))
                    if "camera_list_json" not in columns:
                        print("🔄 正在迁移数据库: 添加 devices.camera_list_json 字段...")
                        connection.execute(text("ALTER TABLE devices ADD COLUMN camera_list_json TEXT"))
                    if "collect_script_compress" not in columns:
                        print("🔄 正在迁移数据库: 添加 devices.collect_script_compress 字段...")
                        connection.execute(text("ALTER TABLE devices ADD COLUMN collect_script_compress VARCHAR(1024)"))
                    if "collect_script_raw" not in columns:
                        print("🔄 正在迁移数据库: 添加 devices.collect_script_raw 字段...")
                        connection.execute(text("ALTER TABLE devices ADD COLUMN collect_script_raw VARCHAR(1024)"))
                    if "team_id" not in columns:
                        print("🔄 正在迁移数据库: 添加 devices.team_id 字段...")
                        connection.execute(text("ALTER TABLE devices ADD COLUMN team_id VARCHAR(128)"))

            await conn.run_sync(migrate_jobs_schema)
            await conn.run_sync(migrate_hdf5_devices_schema)
        print("✅ PostgreSQL 业务表已初始化（Jobs/Tasks/Run/Dataset/HDF5/设备）")
        try:
            import app.models.model_type_definition  # noqa: F401
            from app.core.database import SessionLocal
            from app.services.model_type_service import ensure_default_model_types

            with SessionLocal() as db:
                ensure_default_model_types(db)
            print("✅ 默认模型类型已就绪（robomimic-bc / act / diffusion-policy / pi0）")
        except Exception as seed_exc:
            print(f"⚠ 默认模型类型 seed 失败: {seed_exc}")

        async def _warm_training_capabilities_cache() -> None:
            try:
                from app.services.model_type_service import schedule_model_type_readiness_refresh

                schedule_model_type_readiness_refresh()
            except Exception as warm_exc:
                print(f"ℹ️ 训练能力 probe 预热跳过: {warm_exc}")

        asyncio.create_task(_warm_training_capabilities_cache())
    except (ConnectionRefusedError, Exception) as e:
        if "Connection refused" in str(e) or "could not connect" in str(e).lower() or isinstance(e, ConnectionRefusedError):
            raise RuntimeError(
                "无法连接 PostgreSQL。请确认：\n"
                "  1) PostgreSQL 已启动（如 sudo systemctl start postgresql）\n"
                "  2) .env 中 DATABASE_URL 的主机、端口、用户名、密码正确"
            ) from e
        raise

    # 数据资产库（PostgreSQL 统一库）
    try:
        from app.db.data_assets_session import data_assets_engine
        from app.models.data_asset import Base as DataAssetBase
        import app.models.label_task_asset  # noqa: F401 标注任务表（与数据资产同库）
        import app.models.project_asset  # noqa: F401 项目表（与数据资产同库）
        import app.models.team  # noqa: F401 团队表（与数据资产同库）
        from app.db.migrate_data_assets_projects import ensure_projects_team_id_column

        print("📂 数据资产库（标注任务同库）: PostgreSQL (统一库)")
        async with data_assets_engine.begin() as conn:
            await conn.run_sync(DataAssetBase.metadata.create_all)
            await conn.run_sync(ensure_projects_team_id_column)

            # 设备归属团队回填（best-effort）
            # - 仅回填 devices.team_id 为空的设备
            # - 从 data_assets / collection_jobs 的 device_id -> project_id，再由 projects.team_id 推断
            # - 若同一 device 出现在多个团队，取出现次数最多的团队（ROW_NUMBER 按 cnt 排序）
            # - 幂等：已有 team_id 的记录不会被覆盖
            try:
                backfill_sql = """
                WITH candidates AS (
                    SELECT
                        x.device_id AS device_id_str,
                        p.team_id AS team_id,
                        COUNT(*) AS cnt
                    FROM (
                        SELECT da.device_id, da.project_id
                        FROM data_assets da
                        WHERE da.device_id IS NOT NULL
                          AND da.device_id <> ''
                          AND da.project_id IS NOT NULL
                          AND da.project_id <> ''
                        UNION ALL
                        SELECT cj.device_id, cj.project_id
                        FROM collection_jobs cj
                        WHERE cj.device_id IS NOT NULL
                          AND cj.device_id <> ''
                          AND cj.project_id IS NOT NULL
                          AND cj.project_id <> ''
                    ) x
                    JOIN devices d
                      ON d.id::text = x.device_id
                    JOIN projects p
                      ON p.id = x.project_id
                    WHERE (d.team_id IS NULL OR d.team_id = '')
                      AND p.team_id IS NOT NULL
                      AND p.team_id <> ''
                    GROUP BY x.device_id, p.team_id
                ),
                ranked AS (
                    SELECT
                        device_id_str,
                        team_id,
                        ROW_NUMBER() OVER (
                            PARTITION BY device_id_str
                            ORDER BY cnt DESC, team_id ASC
                        ) AS rn
                    FROM candidates
                )
                UPDATE devices d
                SET team_id = ranked.team_id
                FROM ranked
                WHERE d.id::text = ranked.device_id_str
                  AND ranked.rn = 1
                  AND (d.team_id IS NULL OR d.team_id = '');
                """
                res = await conn.execute(text(backfill_sql))
                # asyncpg 在 UPDATE 时可能返回 -1，兼容兜底日志
                updated = getattr(res, "rowcount", None)
                print(f"🔄 回填 devices.team_id 完成（updated={updated if updated is not None else 'unknown'}）")
            except Exception as e:
                print(f"⚠ 回填 devices.team_id 失败（已跳过）: {e}")
        print("✅ 数据资产库已初始化（PostgreSQL，projects.team_id 已按需补齐）")
    except Exception as e:
        print(f"⚠ 数据资产库初始化失败: {e}")

    # 主库（auth）：确保 users / refresh_tokens 等表存在
    try:
        import app.models.user  # noqa: F401
        import app.models.refresh_token  # noqa: F401
        import app.models.auth_session  # noqa: F401
        import app.models.audit_log  # noqa: F401
        import app.models.account_counter  # noqa: F401 账号流水计数表
        from app.models import Base as ModelsBase
        from app.db.session import engine as main_engine
        from app.db.migrate_audit_logs import ensure_audit_logs_columns
        from app.db.migrate_auth_sessions import ensure_auth_sessions_schema

        async with main_engine.begin() as conn:
            await conn.run_sync(ModelsBase.metadata.create_all)
            await conn.run_sync(ensure_audit_logs_columns)
            await ensure_auth_sessions_schema(conn)
        print("✅ 主库表已初始化（users + refresh_tokens + auth_sessions + audit_logs 缺列已补齐）")
    except Exception as e:
        print(f"⚠ 主库表初始化跳过: {e}")
    
    # 启动时：幂等确保平台超级管理员存在（登录账号 Pibot0001，展示名默认 Pibot；密码 jinlian1234，仅哈希入库）
    try:
        async with AsyncSessionLocal() as db:
            await get_or_create_admin_user(db)
        print("✅ 平台超级管理员已就绪（账号 Pibot0001，展示名 Pibot / jinlian1234）")
    except Exception as e:
        print(f"⚠ 初始化平台超级管理员失败: {e}")

    # 启动时清理磁盘缓存（MinIO 视图缓存 / project 缓存目录，见 CLEAR_CACHE_ON_STARTUP*）
    try:
        from app.services.cache_cleanup_service import run_startup_cache_cleanup_from_env

        startup_cache_result = await asyncio.to_thread(run_startup_cache_cleanup_from_env)
        if startup_cache_result.get("ran"):
            print(f"✅ 启动时已清理缓存: scope={startup_cache_result.get('scope')} data={startup_cache_result.get('data_cache')} project={startup_cache_result.get('project_cache')}")
        elif startup_cache_result.get("skipped"):
            print(f"ℹ️ 启动时跳过缓存清理（{startup_cache_result.get('skipped')}）")
    except Exception as e:
        print(f"⚠ 启动时缓存清理失败（已跳过）: {e}")

    # 启动 ROS2 相机流管理器（无 rclpy 时自动跳过）
    try:
        from app.services.ros2_camera_stream import stream_manager
        if stream_manager.start():
            print("✅ ROS2 相机流管理器已启动")
        else:
            print("ℹ️ ROS2 相机流已跳过（未安装 rclpy）")
    except Exception as e:
        print(f"⚠ ROS2 相机流管理器启动失败: {e}")

    # 启动补偿：恢复未完成的批量同步任务（仅重调度 queued/running）
    try:
        from app.services.sync_batch_service import recover_sync_batch_jobs_on_startup
        recovered = await recover_sync_batch_jobs_on_startup()
        if recovered > 0:
            print(f"✅ 批量同步任务恢复完成，共重调度 {recovered} 个任务")
        else:
            print("ℹ️ 无需恢复的批量同步任务")
    except Exception as e:
        print(f"⚠ 批量同步任务恢复失败: {e}")

    # 启动时回填训练/评测索引（workspace_jobs 已有记录或 runtime 目录存在时幂等同步）
    try:
        from app.services.training_job_sync_service import reindex_runtime_jobs

        reindex_result = await asyncio.to_thread(
            reindex_runtime_jobs,
            dry_run=False,
            overwrite=False,
        )
        synced_training = int(reindex_result.get("syncedTrainingJobs") or 0)
        synced_eval = int(reindex_result.get("syncedEvalJobs") or 0)
        if synced_training or synced_eval:
            print(
                f"✅ runtime 索引回填完成: training={synced_training} eval={synced_eval} "
                f"jobs={reindex_result.get('insertedJobs', 0)}+{reindex_result.get('updatedJobs', 0)}"
            )
        else:
            print("ℹ️ runtime 索引回填：无新增训练/评测任务")
    except Exception as e:
        print(f"⚠ runtime 索引回填失败（已跳过）: {e}")

    try:
        worker_processes = _start_queue_workers_if_needed()
    except Exception as e:
        print(f"⚠ 自动启动队列 workers 失败: {e}")

    # 启动后台自动缓存清理（默认关闭，开启需设置 CACHE_AUTO_CLEANUP_ENABLED=true）
    try:
        from app.services.cache_cleanup_service import periodic_cache_cleanup_loop

        cache_cleanup_task = asyncio.create_task(periodic_cache_cleanup_loop(cache_cleanup_stop_event))
    except Exception as e:
        print(f"⚠ 自动缓存清理任务启动失败: {e}")
    try:
        from app.services.db_backup_service import periodic_db_backup_loop

        db_backup_task = asyncio.create_task(periodic_db_backup_loop(db_backup_stop_event))
    except Exception as e:
        print(f"⚠ 数据库自动备份任务启动失败: {e}")
    try:
        from app.services.telemetry_file_logger import telemetry_periodic_sampler_loop

        telemetry_task = asyncio.create_task(telemetry_periodic_sampler_loop(telemetry_stop_event))
    except Exception as e:
        print(f"⚠ 遥测采样任务启动失败: {e}")

    yield
    
    # 关闭时清理（如果需要）
    try:
        cache_cleanup_stop_event.set()
        if cache_cleanup_task is not None:
            await asyncio.wait_for(cache_cleanup_task, timeout=3)
    except Exception:
        try:
            if cache_cleanup_task is not None:
                cache_cleanup_task.cancel()
        except Exception:
            pass
    try:
        db_backup_stop_event.set()
        if db_backup_task is not None:
            await asyncio.wait_for(db_backup_task, timeout=3)
    except Exception:
        try:
            if db_backup_task is not None:
                db_backup_task.cancel()
        except Exception:
            pass
    try:
        telemetry_stop_event.set()
        if telemetry_task is not None:
            await asyncio.wait_for(telemetry_task, timeout=3)
    except Exception:
        try:
            if telemetry_task is not None:
                telemetry_task.cancel()
        except Exception:
            pass
    try:
        await _stop_queue_workers(worker_processes)
    except Exception:
        pass


app = FastAPI(
    title="EAI Data Platform API",
    description="一湃智能数据平台后端 API",
    version="1.0.0",
    lifespan=lifespan
)


@app.on_event("startup")
async def _warm_public_endpoint_cache():
    try:
        from app.services.public_endpoint_resolver import public_endpoint_resolver

        public_endpoint_resolver.refresh(force=True)
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning("public_endpoint: warmup failed: %s", e)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# 必须在 CORS 之后注册，使其位于栈外侧，保证整次请求（含 CORS 与路由）结束后再清理缓存
app.add_middleware(EpisodeStorageClearASGIMiddleware)

# 注册路由
app.include_router(api_router, prefix="/api")


@app.api_route("/static/bin/agent_linux_x64.tar.gz", methods=["GET", "HEAD"])
async def agent_linux_x64_tarball(request: Request):
    from app.core.config import settings
    from app.services.agent_package_manager import resolve_linux_x86_64_tarball_path

    try:
        p = resolve_linux_x86_64_tarball_path(
            override_path=getattr(settings, "AGENT_LINUX_X64_TARBALL_PATH", None),
        )
    except FileNotFoundError as e:
        hint = (
            "未找到 Agent 离线包。请在部署机执行：cd backend/agent_packages && "
            "./build_agent_bundle.sh <与 manifest.json latest 一致的版本> linux x86_64，"
            "将输出的 sha256 写入 manifest.json，或将已构建的 .tar.gz 绝对路径写入环境变量 "
            "AGENT_LINUX_X64_TARBALL_PATH 后重启后端。详见 docs/agent-bundling.md。"
        )
        return PlainTextResponse(f"{hint}\n{e}", status_code=404)
    except ValueError as e:
        return PlainTextResponse(f"agent 安装包配置错误：{e}", status_code=500)
    except Exception as e:
        return PlainTextResponse(f"internal error: {e}", status_code=500)

    if request.method == "HEAD":
        st = p.stat()
        return Response(
            status_code=200,
            headers={"Content-Type": "application/gzip", "Content-Length": str(st.st_size)},
        )
    return StreamingResponse(p.open("rb"), media_type="application/gzip")

# 全局异常处理器 - 处理所有 HTTPException（包括 404）
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """HTTP 异常处理器，确保返回 JSON 格式"""
    if isinstance(exc.detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
            media_type="application/json",
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "ok": False,
            "error": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
        },
        media_type="application/json"
    )

@app.exception_handler(StarletteHTTPException)
async def starlette_http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Starlette HTTP 异常处理器（处理 404 等）"""
    if isinstance(exc.detail, dict):
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.detail,
            media_type="application/json",
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "ok": False,
            "error": exc.detail if isinstance(exc.detail, str) else str(exc.detail),
        },
        media_type="application/json"
    )


@app.exception_handler(QueueBackpressureError)
async def queue_backpressure_handler(request: Request, exc: QueueBackpressureError):
    """RQ 队列积压超限，拒绝入队（429）"""
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={
            "ok": False,
            "error": str(exc),
            "queue": exc.queue_name,
            "depth": exc.depth,
            "limit": exc.limit,
        },
        media_type="application/json",
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理器，确保所有错误都返回 JSON 格式"""
    import traceback
    error_detail = str(exc)[:500]
    traceback.print_exc()  # 打印到控制台用于调试
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "ok": False,
            "error": f"服务器内部错误：{error_detail}",
        },
        media_type="application/json"
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """请求验证异常处理器"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "ok": False,
            "error": "请求参数验证失败",
            "detail": str(exc)
        },
        media_type="application/json"
    )


@app.get("/health", response_model=ApiResponse)
async def health_check():
    """健康检查"""
    return ApiResponse(ok=True, data={"status": "healthy"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
