from pathlib import Path
from pydantic import AliasChoices, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    APP_NAME: str = "EAI Data Platform Backend"
    API_PREFIX: str = "/api"

    JWT_SECRET: str = Field(
        default="dev-secret-change-me",
        validation_alias=AliasChoices("JWT_SECRET", "SECRET_KEY"),
    )
    JWT_ALG: str = Field(
        default="HS256",
        validation_alias=AliasChoices("JWT_ALG", "ALGORITHM"),
    )
    ACCESS_TOKEN_EXPIRE_MIN: int = Field(
        default=15,
        validation_alias=AliasChoices(
            "ACCESS_TOKEN_EXPIRE_MIN",
            "ACCESS_TOKEN_EXPIRE_MINUTES",
        ),
    )
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # 数据库：仅支持 PostgreSQL（已从 SQLite 完全迁移）
    # 异步 ORM：app.db.session、app.db.data_assets_session 等均指向同一 PostgreSQL
    DATABASE_URL: str = "postgresql+asyncpg://admin:CHANGE_ME@172.18.0.93:5432/eai_ide"

    # checkpoint 归档 MinIO（None=随 MINIO_ENDPOINT 自动启用；False=强制关闭）
    CHECKPOINT_ARCHIVE_ENABLED: bool | None = None
    CHECKPOINT_ARCHIVE_BUCKET: str = "eai-checkpoints"

    # 统一产物异步上传（dataset / eval / checkpoint → MinIO）
    ARTIFACT_UPLOAD_ENABLED: bool | None = None
    WORKSPACE_ARTIFACT_BUCKET: str = "eai-workspace-artifacts"
    ARTIFACT_UPLOAD_INTERMEDIATE_CHECKPOINTS: bool = False
    ARTIFACT_UPLOAD_WORKER_INTERVAL_SEC: int = 120
    ARTIFACT_UPLOAD_SCAN_LIMIT: int = 20

    @property
    def sync_database_url(self) -> str:
        """同步驱动 URL（psycopg2），供 app.core.database 等同步代码使用。"""
        u = self.DATABASE_URL
        if "asyncpg" in u:
            u = u.replace("postgresql+asyncpg", "postgresql", 1)
        return u


    # OpenAI 兼容网关配置（用于自动标注）
    OPENAI_BASE_URL: str | None = None
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str | None = None


    # Cookie & CORS 配置
    COOKIE_SECURE: bool = False  # 开发环境 False，生产环境 True
    COOKIE_SAMESITE: str = "Lax"
    COOKIE_DOMAIN: str | None = None
    FRONTEND_ORIGIN: str = "http://localhost:3000"
    # 后端对外可访问基址（供采集端 Agent 回连平台注册/心跳/隧道）
    # 示例：http://172.18.0.114:8000
    PUBLIC_BASE_URL: str = "http://127.0.0.1:8000"
    # 实验子系统总开关：关闭后 /api/experiment 不可用，实验埋点与采样均静默。
    EXPERIMENT_ENABLED: bool = False

    # 平台遥测 JSONL（隧道命令 / WebRTC 转发 / MJPEG 计数器 / 本机资源）：与 uvicorn 访问日志分离；默认保留 2 天。
    TELEMETRY_FILE_LOG_ENABLED: bool = True
    TELEMETRY_FILE_LOG_DIR: str = "logs/telemetry"
    TELEMETRY_FILE_RETENTION_DAYS: int = 2
    TELEMETRY_SAMPLE_INTERVAL_SEC: int = 30

    # MinIO 配置（项目创建时自动创建同名 bucket）
    # MINIO_ENDPOINT：服务端（API/worker/本机脚本）访问 MinIO 的 host:port，compose host 网络常用 127.0.0.1:9000
    # MINIO_PUBLIC_ENDPOINT：浏览器直传预签名 URL；远程采集端经隧道同步写入平台 MinIO 时也优先使用该地址（须为采集机可达的局域网 IP 或域名，勿填 127.0.0.1）；不填则回退 MINIO_ENDPOINT
    MINIO_ENDPOINT: str | None = None
    MINIO_PUBLIC_ENDPOINT: str | None = None
    MINIO_ACCESS_KEY: str | None = None
    MINIO_SECRET_KEY: str | None = None
    MINIO_SECURE: bool = False

    # 批量同步任务：并发上限（与 PostgreSQL 持久化任务配合，见 POST /data-assets/sync/batch）
    SYNC_BATCH_MAX_CONCURRENT_GLOBAL: int = 4
    SYNC_BATCH_MAX_CONCURRENT_PER_AGENT: int = 1

    # Agent WebSocket 隧道：若设置非空，则连接 query 必须带 `token=` 且一致（与采集端 EAI_AGENT_TUNNEL_TOKEN 对齐）
    AGENT_TUNNEL_TOKEN: str | None = None
    # 可选：按 agent_id 绑定密钥的 JSON，如 {"agent-1":"secretA","agent-2":"secretB"}（文档 §4.2）
    # 若设置非空 JSON，则仅允许 map 中出现的 agent_id 建连，且 token 必须与该 agent 条目一致。
    AGENT_TUNNEL_TOKEN_BY_AGENT_JSON: str | None = None
    # 超过该秒数未收到 HEARTBEAT/任意隧道文本则视为 stale（用于 /agents/online 等）
    AGENT_TUNNEL_OFFLINE_AFTER_SEC: float = 45.0
    # WebRTC SDP 信令优先走采集端 HTTP（/api/agent/webrtc/offer），避免占用隧道队列导致 COLLECT_* / FS_LIST 拥塞
    WEBRTC_OFFER_PREFER_HTTP: bool = True

    # Agent 安装包签名校验：Ed25519 公钥列表（base64），如 ["<pk1_b64>","<pk2_b64>"]
    AGENT_INSTALL_PUBKEYS_JSON: str | None = None
    # 可选：覆盖 Linux x86_64 离线包路径（绝对或相对路径均可）。不配则使用 agent_packages/manifest.json 中的 path + sha256 校验。
    AGENT_LINUX_X64_TARBALL_PATH: str | None = None

    # Agent 一键安装 linux.sh：生成 /opt/eai-agent/run-agent.sh + systemd（默认 venv + ROS；可启用 Conda）
    AGENT_INSTALL_USE_CONDA: bool = False
    AGENT_INSTALL_CONDA_SH: str = ""
    AGENT_INSTALL_CONDA_ENV: str = ""
    AGENT_INSTALL_ROS_SETUP: str = "/opt/ros/humble/setup.bash"
    AGENT_INSTALL_ROS_WS_SETUP: str = ""
    # systemd User=。__SUDO_USER__ 表示与执行「sudo bash 安装脚本」的登录用户一致（推荐，非 root 跑服务）
    AGENT_INSTALL_SERVICE_USER: str = "__SUDO_USER__"

    # Redis（RQ 任务队列）；与 .env 中 REDIS_* 一致
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str | None = None

    # Isaac Lab 外部运行时（subprocess，不在 FastAPI 进程内 import isaaclab）
    ISAACLAB_ROOT: str | None = None
    ISAACLAB_SH: str | None = None
    ISAACLAB_PYTHON: str | None = None
    ISAACLAB_DEFAULT_TASK: str = "Isaac-Stack-Cube-Franka-IK-Rel-v0"
    ISAACLAB_RUNTIME_ENABLED: bool = False
    ISAACLAB_RUNTIME_MODE: str = "external_subprocess"
    ISAACLAB_OUTPUT_ROOT: str = "runs/isaac_lab/jobs"
    ISAACLAB_SMOKE_TEST_TIMEOUT: int = 900
    ISAACLAB_REPLAY_TIMEOUT: int = 1800
    ISAACLAB_GENERATE_TIMEOUT: int = 7200
    ISAACLAB_TRAIN_TIMEOUT: int = 7200
    ISAACLAB_EVAL_TIMEOUT: int = 3600
    # Empty means the platform-managed seed under EAI_DATA_ROOT. An explicit
    # path still overrides this for deployments with an external seed library.
    ISAACLAB_STACK_CUBE_DEFAULT_SEED: str = ""

    # SAM3 + SAM3D Objects 图像重建流水线（subprocess，双 conda 环境）
    ASSET3DRECONS_ROOT: str = "/home/ubuntu/project/asset3Drecons"
    SAM3_ROOT: str = "/home/ubuntu/project/asset3Drecons/sam3"
    SAM3_PYTHON: str = "/home/ubuntu/miniconda3/envs/sam3/bin/python"
    SAM3D_OBJECTS_ROOT: str = "/home/ubuntu/project/asset3Drecons/sam-3d-objects"
    SAM3D_OBJECTS_PYTHON: str = "/home/ubuntu/miniconda3/envs/sam3d-objects/bin/python"
    SAM3D_PIPELINE_ENABLED: bool = True
    SAM3D_OUTPUT_ROOT: str = "runs/asset_pipeline/jobs"
    SAM3D_OFFLINE_MODE: bool = True
    SAM3D_DINOV2_REPO: str = "/home/ubuntu/.cache/torch/hub/facebookresearch_dinov2_main"
    SAM3D_DINOV2_MODEL: str = "dinov2_vitl14_reg"
    SAM3D_MOGE_MODEL_PATH: str = (
        "/home/ubuntu/.cache/huggingface/hub/models--Ruicheng--moge-vitl"
    )
    SAM3D_TORCH_HOME: str = "/home/ubuntu/.cache/torch"
    SAM3D_HF_HOME: str = "/home/ubuntu/.cache/huggingface"
    SAM3D_RECONSTRUCT_TIMEOUT_SECONDS: int = 1800
    SAM3D_HF_ENDPOINT: str = "https://hf-mirror.com"
    SAM3D_GIT_GITHUB_SSH_REWRITE: bool = True

    # MuJoCo 离屏渲染（使用独立 conda 环境，不影响 SAM3D Python）
    MUJOCO_RENDER_PYTHON: str = "/home/ubuntu/miniconda3/envs/cable/bin/python"
    MUJOCO_RENDER_GL: str = "egl"
    MUJOCO_RENDER_WIDTH: int = 960
    MUJOCO_RENDER_HEIGHT: int = 720
    MUJOCO_RENDER_ENABLED: bool = True

    model_config = SettingsConfigDict(
        env_file=(
            BASE_DIR / ".env",
            BASE_DIR.parent / ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
        # 确保环境变量不会覆盖默认值，除非明确设置
        case_sensitive=False,
    )


settings = Settings()
