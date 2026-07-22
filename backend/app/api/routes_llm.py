"""
大模型厂商/模型配置 API（数据存 PostgreSQL）
供标注页「模型选择与管理」弹窗：providers、models、user_providers、user_models 的增删改查与验证。
"""
import logging
from typing import Optional, List, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.common import ApiResponse
from app.db.data_assets_session import get_data_assets_db
from app.core.deps import get_current_user
from app.models.user import User
from app.core.roles import CanonicalUserRole, is_super_admin, normalize_role
from app.core.label_conversion_access import scoped_project_ids_for_platform_tasks

logger = logging.getLogger(__name__)
router = APIRouter()

# 标注页未登录时使用的默认用户 ID（与 user_providers / user_models 对应）
DEFAULT_USER_ID = 0


_PROVIDER_CODE_ALIASES = {
    "openai-compatible": "openai",
    "openai_compatible": "openai",
}

_DEFAULT_PROJECT_PROVIDERS = [
    {"provider_id": 1001, "provider_code": "openai", "provider_name": "OpenAI", "model_name": "gpt-5.4"},
    {"provider_id": 1002, "provider_code": "gemini", "provider_name": "Google Gemini", "model_name": "gemini-2.5-flash"},
    {"provider_id": 1003, "provider_code": "deepseek", "provider_name": "DeepSeek", "model_name": "deepseek-chat"},
    {"provider_id": 1004, "provider_code": "qwen", "provider_name": "通义千问", "model_name": "qwen-plus"},
]

_PROVIDER_DEFAULT_NAMES = {
    "openai": "OpenAI",
    "gemini": "Google Gemini",
    "deepseek": "DeepSeek",
    "qwen": "通义千问",
}


# ---------- Schemas ----------
class ProviderOut(BaseModel):
    id: int
    name: str
    code: str
    type: str
    base_url: Optional[str]
    api_key_prefix: Optional[str]
    supports_stream: bool
    supports_functions: bool
    logo: Optional[str]
    is_active: bool
    sort_order: int
    is_enabled: bool = False
    has_api_key: bool = False
    is_verified: bool = False


class ModelOut(BaseModel):
    id: int
    provider_id: int
    name: str
    display_name: Optional[str]
    context_length: Optional[int]
    max_output_tokens: Optional[int]
    is_default: bool
    is_active: bool
    sort_order: int
    is_selected: bool = False


class ProviderDetailOut(ProviderOut):
    api_base: Optional[str] = None  # 用户自定义 API 地址（不返回 api_key）
    models: List[ModelOut] = []


class UserProviderUpdate(BaseModel):
    provider_id: int
    project_id: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    is_enabled: Optional[bool] = None


class VerifyRequest(BaseModel):
    provider_id: int
    project_id: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None


class UserModelsUpdate(BaseModel):
    provider_id: int
    project_id: str
    model_ids: List[int]


class ModelCreate(BaseModel):
    project_id: str
    provider_id: int
    name: str
    display_name: Optional[str] = None
    context_length: Optional[int] = None
    max_output_tokens: Optional[int] = None


class ModelUpdate(BaseModel):
    name: Optional[str] = None
    display_name: Optional[str] = None


def _row_to_provider(row: Any, user_enabled: bool = False, has_key: bool = False, is_verified: bool = False) -> dict:
    return {
        "id": row.id,
        "name": row.name,
        "code": row.code,
        "type": row.type,
        "base_url": row.base_url,
        "api_key_prefix": row.api_key_prefix,
        "supports_stream": bool(row.supports_stream),
        "supports_functions": bool(row.supports_functions),
        "logo": row.logo,
        "is_active": bool(row.is_active),
        "sort_order": row.sort_order or 0,
        "is_enabled": user_enabled,
        "has_api_key": has_key,
        "is_verified": is_verified,
    }


def _row_to_model(row: Any, is_selected: bool = False) -> dict:
    return {
        "id": row.id,
        "provider_id": row.provider_id,
        "name": row.name,
        "display_name": row.display_name,
        "context_length": row.context_length,
        "max_output_tokens": row.max_output_tokens,
        "is_default": bool(row.is_default),
        "is_active": bool(row.is_active),
        "sort_order": row.sort_order or 0,
        "is_selected": is_selected,
    }


async def _ensure_project_scope_columns(db: AsyncSession) -> None:
    """为 llm 配置表补齐 project_id，支持按项目绑定配置。

    新库若未跑旧版迁移，可能不存在 user_providers / user_models。直接 ALTER 会让 PostgreSQL
    将当前事务标为失败；仅 Python try/except 无法恢复会话，后续 CREATE project_llm_bindings
    会报 asyncpg InFailedSQLTransactionError。故先 to_regclass 判断，且每条 DDL 用 savepoint 隔离。
    """
    has_up = (
        await db.execute(text("SELECT to_regclass('public.user_providers') IS NOT NULL"))
    ).scalar()
    has_um = (
        await db.execute(text("SELECT to_regclass('public.user_models') IS NOT NULL"))
    ).scalar()
    if bool(has_up):
        try:
            async with db.begin_nested():
                await db.execute(
                    text("ALTER TABLE user_providers ADD COLUMN IF NOT EXISTS project_id VARCHAR")
                )
        except Exception:
            logger.debug("ALTER user_providers.project_id skipped", exc_info=True)
    if bool(has_um):
        try:
            async with db.begin_nested():
                await db.execute(
                    text("ALTER TABLE user_models ADD COLUMN IF NOT EXISTS project_id VARCHAR")
                )
        except Exception:
            logger.debug("ALTER user_models.project_id skipped", exc_info=True)


async def _ensure_project_llm_bindings_table(db: AsyncSession) -> None:
    """单表强绑定：项目+厂商+模型+密钥+地址+权限。"""
    await db.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS project_llm_bindings (
              id BIGSERIAL PRIMARY KEY,
              project_id VARCHAR NOT NULL,
              provider_id BIGINT NOT NULL,
              provider_code VARCHAR NOT NULL,
              provider_name VARCHAR NOT NULL,
              model_name VARCHAR NOT NULL,
              api_base TEXT,
              api_key TEXT,
              api_key_masked VARCHAR,
              editable_roles VARCHAR NOT NULL DEFAULT 'SUPER_ADMIN,ADMIN',
              is_enabled BOOLEAN NOT NULL DEFAULT TRUE,
              is_verified BOOLEAN NOT NULL DEFAULT FALSE,
              verified_at TIMESTAMP NULL,
              created_by VARCHAR NULL,
              updated_by VARCHAR NULL,
              created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
              UNIQUE(project_id, provider_id, model_name)
            )
            """
        )
    )
    await db.execute(
        text(
            "CREATE INDEX IF NOT EXISTS idx_project_llm_bindings_project_provider ON project_llm_bindings(project_id, provider_id)"
        )
    )


def _norm_provider_code(code: Optional[str]) -> str:
    c = (code or "").strip().lower()
    if not c:
        return c
    return _PROVIDER_CODE_ALIASES.get(c, c)


async def _ensure_default_project_providers(db: AsyncSession, scope_key: str, actor_user: User) -> None:
    """
    为 scope_key 补齐统一供应商模板，保证所有团队在 UI 上看到一致的供应商分组（图2样式）。
    scope_key 为 team:<team_id> 或 project:<project_id>。
    """
    proj = _norm_pid(scope_key)
    if not proj:
        return
    rows = (
        await db.execute(
            text(
                """
                SELECT DISTINCT provider_id, provider_code, provider_name
                FROM project_llm_bindings
                WHERE project_id = :proj
                """
            ),
            {"proj": proj},
        )
    ).fetchall()
    code_to_existing: dict[str, tuple[int, str]] = {}
    for r in rows:
        norm_code = _norm_provider_code(getattr(r, "provider_code", None))
        if not norm_code:
            continue
        if norm_code in code_to_existing:
            continue
        code_to_existing[norm_code] = (
            int(getattr(r, "provider_id", 0) or 0),
            str(getattr(r, "provider_name", "") or ""),
        )

    uid = str(getattr(actor_user, "id", "") or "")
    inserted = False
    for p in _DEFAULT_PROJECT_PROVIDERS:
        code = p["provider_code"]
        existing = code_to_existing.get(code)
        provider_id = existing[0] if existing and existing[0] > 0 else int(p["provider_id"])
        provider_name = existing[1] if existing and existing[1] else str(p["provider_name"])
        model_name = str(p["model_name"])
        has_row = (
            await db.execute(
                text(
                    """
                    SELECT 1
                    FROM project_llm_bindings
                    WHERE project_id = :proj AND provider_id = :pid AND model_name = :mname
                    LIMIT 1
                    """
                ),
                {"proj": proj, "pid": provider_id, "mname": model_name},
            )
        ).fetchone()
        if has_row:
            continue
        await db.execute(
            text(
                """
                INSERT INTO project_llm_bindings
                  (project_id, provider_id, provider_code, provider_name, model_name,
                   api_base, api_key, api_key_masked, editable_roles, is_enabled, is_verified, created_by, updated_by)
                VALUES
                  (:proj, :pid, :pcode, :pname, :mname,
                   '', '', NULL, 'SUPER_ADMIN,ADMIN', FALSE, FALSE, :uid, :uid)
                """
            ),
            {
                "proj": proj,
                "pid": provider_id,
                "pcode": code,
                "pname": provider_name,
                "mname": model_name,
                "uid": uid,
            },
        )
        inserted = True
    if inserted:
        await db.commit()


def _norm_pid(project_id: Optional[str]) -> str:
    return (project_id or "").strip()


async def _resolve_llm_scope_key(db: AsyncSession, project_id: str) -> str:
    """
    团队优先隔离：
    - 有 team_id: 使用 team:<team_id>（同团队共享一套配置）
    - 无 team_id: 回退 project:<project_id>
    """
    pid = _norm_pid(project_id)
    if not pid:
        return ""
    row = (
        await db.execute(
            text("SELECT team_id FROM projects WHERE id = :pid LIMIT 1"),
            {"pid": pid},
        )
    ).fetchone()
    tid = (getattr(row, "team_id", None) or "").strip() if row else ""
    if tid:
        return f"team:{tid}"
    return f"project:{pid}"


async def _assert_llm_scope_visible_or_403(db: AsyncSession, current_user: User, project_id: str) -> str:
    pid = _norm_pid(project_id)
    if not pid:
        raise HTTPException(status_code=400, detail="project_id required")
    scoped = await scoped_project_ids_for_platform_tasks(db, current_user)
    if scoped is not None and pid not in scoped:
        raise HTTPException(status_code=403, detail="无权限访问该项目配置")
    return pid


async def _assert_llm_scope_write_or_403(db: AsyncSession, current_user: User, project_id: str) -> str:
    pid = await _assert_llm_scope_visible_or_403(db, current_user, project_id)
    role = normalize_role(getattr(current_user, "role", None))
    if not (
        is_super_admin(role)
        or role in {CanonicalUserRole.ADMIN, CanonicalUserRole.OWNER}
    ):
        raise HTTPException(status_code=403, detail="仅团队管理员可配置项目模型")
    return pid


@router.get("/providers", response_model=ApiResponse)
async def list_providers(
    search: Optional[str] = Query(None, description="搜索厂商/模型名称"),
    project_id: Optional[str] = Query(None, description="项目 ID"),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """列出项目内已绑定厂商（完全来源 project_llm_bindings）。"""
    await _ensure_project_llm_bindings_table(db)
    pid = await _assert_llm_scope_visible_or_403(db, current_user, project_id or "")
    scope_key = await _resolve_llm_scope_key(db, pid)
    await _ensure_default_project_providers(db, scope_key, current_user)
    rows = (
        await db.execute(
            text(
                """
                SELECT
                  provider_id,
                  MAX(provider_name) AS provider_name,
                  MAX(provider_code) AS provider_code,
                  BOOL_OR(is_enabled) AS is_enabled,
                  MAX(CASE WHEN COALESCE(api_key,'') <> '' THEN 1 ELSE 0 END) AS has_key_i,
                  MAX(CASE WHEN is_verified THEN 1 ELSE 0 END) AS is_verified_i,
                  MAX(updated_at) AS latest_ts
                FROM project_llm_bindings
                WHERE project_id = :proj
                GROUP BY provider_id
                ORDER BY
                  CASE LOWER(MAX(provider_code))
                    WHEN 'openai' THEN 1
                    WHEN 'openai-compatible' THEN 1
                    WHEN 'openai_compatible' THEN 1
                    WHEN 'gemini' THEN 2
                    WHEN 'deepseek' THEN 3
                    WHEN 'qwen' THEN 4
                    ELSE 99
                  END ASC,
                  latest_ts DESC NULLS LAST,
                  provider_id ASC
                """
            ),
            {"proj": scope_key},
        )
    ).fetchall()
    out = []
    for r in rows:
        norm_code = _norm_provider_code(getattr(r, "provider_code", None))
        default_name = _PROVIDER_DEFAULT_NAMES.get(norm_code, "")
        item = {
            "id": int(r.provider_id),
            "name": default_name or (r.provider_name or "").strip() or f"Provider {r.provider_id}",
            "code": norm_code or f"provider_{r.provider_id}",
            "type": "openai-compatible",
            "base_url": None,
            "api_key_prefix": None,
            "supports_stream": False,
            "supports_functions": False,
            "logo": None,
            "is_active": True,
            "sort_order": 0,
            "is_enabled": bool(getattr(r, "is_enabled", False)),
            "has_api_key": bool((getattr(r, "has_key_i", 0) or 0) > 0),
            "is_verified": bool((getattr(r, "is_verified_i", 0) or 0) > 0),
        }
        if search and search.strip():
            kw = search.strip().lower()
            if kw not in (item["name"] or "").lower() and kw not in (item["code"] or "").lower():
                continue
        out.append(item)
    return ApiResponse(ok=True, data=out)


@router.get("/providers/{provider_id}", response_model=ApiResponse)
async def get_provider_detail(
    provider_id: int,
    project_id: Optional[str] = Query(None, description="项目 ID"),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """单个厂商详情 + 其下模型列表（完全来源 project_llm_bindings）。"""
    await _ensure_project_llm_bindings_table(db)
    pid = await _assert_llm_scope_visible_or_403(db, current_user, project_id or "")
    scope_key = await _resolve_llm_scope_key(db, pid)
    await _ensure_default_project_providers(db, scope_key, current_user)
    up = await db.execute(
        text(
            """
            SELECT
                   MAX(provider_name) AS provider_name,
                   MAX(provider_code) AS provider_code,
                   api_base, api_key_masked, editable_roles,
                   BOOL_OR(is_enabled) AS is_enabled,
                   MAX(CASE WHEN COALESCE(api_key,'') <> '' THEN 1 ELSE 0 END) AS has_key_i,
                   MAX(CASE WHEN is_verified THEN 1 ELSE 0 END) AS is_verified_i
            FROM project_llm_bindings
            WHERE project_id = :proj AND provider_id = :pid
            GROUP BY api_base, api_key_masked, editable_roles
            ORDER BY MAX(updated_at) DESC NULLS LAST
            LIMIT 1
            """
        ),
        {"pid": provider_id, "proj": scope_key},
    )
    up_row = up.fetchone()
    is_enabled = bool(getattr(up_row, "is_enabled", False)) if up_row else False
    has_key = bool((getattr(up_row, "has_key_i", 0) or 0) > 0) if up_row else False
    is_verified = bool((getattr(up_row, "is_verified_i", 0) or 0) > 0) if up_row else False
    api_base = getattr(up_row, "api_base", None) if up_row else None
    api_key_masked = getattr(up_row, "api_key_masked", None) if up_row else None
    if not up_row:
        raise HTTPException(status_code=404, detail="厂商不存在")
    raw_code = getattr(up_row, "provider_code", None)
    norm_code = _norm_provider_code(raw_code)
    display_name = _PROVIDER_DEFAULT_NAMES.get(norm_code) or (getattr(up_row, "provider_name", None) or f"Provider {provider_id}")

    detail = {
        "id": provider_id,
        "name": display_name,
        "code": norm_code or f"provider_{provider_id}",
        "type": "openai-compatible",
        "base_url": None,
        "api_key_prefix": None,
        "supports_stream": False,
        "supports_functions": False,
        "logo": None,
        "is_active": True,
        "sort_order": 0,
        "is_enabled": is_enabled,
        "has_api_key": has_key,
        "is_verified": is_verified,
        "api_base": api_base,
        "api_key_masked": api_key_masked,
    }
    models_rows = (
        await db.execute(
            text(
                """
                SELECT id, provider_id, model_name, is_enabled, updated_at
                FROM project_llm_bindings
                WHERE project_id = :proj AND provider_id = :pid
                ORDER BY updated_at DESC NULLS LAST, id ASC
                """
            ),
            {"proj": scope_key, "pid": provider_id},
        )
    ).fetchall()
    detail["models"] = [
        {
            "id": int(m.id),
            "provider_id": int(m.provider_id),
            "name": str(m.model_name or ""),
            "display_name": str(m.model_name or ""),
            "context_length": None,
            "max_output_tokens": None,
            "is_default": False,
            "is_active": True,
            "sort_order": 0,
            "is_selected": bool(m.is_enabled),
        }
        for m in models_rows
        if str(m.model_name or "").strip()
    ]
    return ApiResponse(ok=True, data=detail)


@router.get("/models", response_model=ApiResponse)
async def list_models(
    provider_id: int = Query(..., description="厂商 ID"),
    project_id: Optional[str] = Query(None, description="项目 ID"),
    search: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """列出某厂商下模型（完全来源 project_llm_bindings）。"""
    await _ensure_project_llm_bindings_table(db)
    pid = await _assert_llm_scope_visible_or_403(db, current_user, project_id or "")
    scope_key = await _resolve_llm_scope_key(db, pid)
    await _ensure_default_project_providers(db, scope_key, current_user)
    rows = (
        await db.execute(
            text(
                """
                SELECT id, provider_id, model_name, is_enabled, updated_at
                FROM project_llm_bindings
                WHERE project_id = :proj AND provider_id = :pid
                ORDER BY updated_at DESC NULLS LAST, id ASC
                """
            ),
            {"proj": scope_key, "pid": provider_id},
        )
    ).fetchall()
    out = []
    for r in rows:
        name = str(r.model_name or "").strip()
        if not name:
            continue
        item = {
            "id": int(r.id),
            "provider_id": int(r.provider_id),
            "name": name,
            "display_name": name,
            "context_length": None,
            "max_output_tokens": None,
            "is_default": False,
            "is_active": True,
            "sort_order": 0,
            "is_selected": bool(r.is_enabled),
        }
        if search and search.strip():
            kw = search.strip().lower()
            if kw not in name.lower():
                continue
        out.append(item)
    return ApiResponse(ok=True, data=out)


@router.patch("/user-models", response_model=ApiResponse)
async def update_user_models(
    body: UserModelsUpdate = Body(...),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """设置项目在某厂商下启用的模型列表（以 project_llm_bindings.id 作为模型标识）。"""
    provider_id = body.provider_id
    await _ensure_project_scope_columns(db)
    await _ensure_project_llm_bindings_table(db)
    pid = await _assert_llm_scope_write_or_403(db, current_user, body.project_id)
    scope_key = await _resolve_llm_scope_key(db, pid)
    await db.execute(
        text(
            "UPDATE project_llm_bindings SET is_enabled = FALSE, updated_at = CURRENT_TIMESTAMP "
            "WHERE project_id=:proj AND provider_id=:pid"
        ),
        {"pid": provider_id, "proj": scope_key},
    )
    await db.execute(
        text(
            "UPDATE project_llm_bindings SET is_enabled = TRUE, updated_at = CURRENT_TIMESTAMP "
            "WHERE project_id=:proj AND provider_id=:pid AND id = ANY(:ids)"
        ),
        {
            "proj": scope_key,
            "pid": provider_id,
            "ids": list(body.model_ids or []),
        },
    )
    await db.commit()
    return ApiResponse(ok=True, data={"updated": True, "selected_count": len(list(body.model_ids or []))})


@router.patch("/user-providers", response_model=ApiResponse)
async def update_user_provider(
    body: UserProviderUpdate = Body(...),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """更新项目对某厂商的配置：api_key、api_base、is_enabled（写入 project_llm_bindings）。"""
    provider_id = body.provider_id
    await _ensure_project_scope_columns(db)
    await _ensure_project_llm_bindings_table(db)
    pid = await _assert_llm_scope_write_or_403(db, current_user, body.project_id)
    scope_key = await _resolve_llm_scope_key(db, pid)
    rows = (
        await db.execute(
            text(
                "SELECT id, provider_code, provider_name, model_name, api_key, api_base, is_enabled "
                "FROM project_llm_bindings WHERE project_id=:proj AND provider_id=:pid "
                "ORDER BY updated_at DESC NULLS LAST"
            ),
            {"proj": scope_key, "pid": provider_id},
        )
    ).fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail="该项目下暂无该厂商绑定，请先添加模型")
    old_key = next(((r.api_key or "") for r in rows if getattr(r, "api_key", None)), "")
    old_base = next(((r.api_base or "") for r in rows if getattr(r, "api_base", None)), "")
    old_enabled = any(bool(getattr(r, "is_enabled", False)) for r in rows)
    api_key = body.api_key if body.api_key is not None else old_key
    api_base = body.api_base if body.api_base is not None else old_base
    is_enabled = bool(body.is_enabled) if body.is_enabled is not None else old_enabled
    masked = None
    k = (api_key or "").strip()
    if k:
        masked = k[:7] + "***" + k[-4:] if len(k) > 11 else "***"
    pcode = (rows[0].provider_code or f"provider_{provider_id}").strip()
    pname = (rows[0].provider_name or f"Provider {provider_id}").strip()
    # 项目内厂商单选：开启当前厂商时，先关闭其他厂商
    if is_enabled:
        await db.execute(
            text(
                "UPDATE project_llm_bindings SET is_enabled = FALSE, updated_at = CURRENT_TIMESTAMP "
                "WHERE project_id = :proj AND provider_id <> :pid"
            ),
            {"proj": scope_key, "pid": provider_id},
        )
    await db.execute(
        text(
            """
            UPDATE project_llm_bindings
            SET api_key=:api_key, api_key_masked=:masked, api_base=:api_base, is_enabled=:is_enabled,
                provider_code=:pcode, provider_name=:pname, updated_by=:uid, updated_at=CURRENT_TIMESTAMP
            WHERE project_id=:proj AND provider_id=:pid
            """
        ),
        {
            "proj": scope_key,
            "pid": provider_id,
            "api_key": api_key or "",
            "masked": masked,
            "api_base": api_base or "",
            "is_enabled": True if is_enabled else False,
            "pcode": pcode,
            "pname": pname,
            "uid": str(getattr(current_user, "id", "") or ""),
        },
    )
    await db.commit()
    return ApiResponse(ok=True, data={"updated": True})


@router.post("/verify", response_model=ApiResponse)
async def verify_provider(
    body: VerifyRequest = Body(...),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """验证厂商 API Key：请求 base_url/v1/models（或用户传入 api_base）。"""
    await _ensure_project_scope_columns(db)
    await _ensure_project_llm_bindings_table(db)
    pid = await _assert_llm_scope_write_or_403(db, current_user, body.project_id)
    scope_key = await _resolve_llm_scope_key(db, pid)
    row = (
        await db.execute(
            text(
                "SELECT api_base, api_key FROM project_llm_bindings "
                "WHERE project_id=:proj AND provider_id=:pid "
                "ORDER BY updated_at DESC NULLS LAST LIMIT 1"
            ),
            {"proj": scope_key, "pid": body.provider_id},
        )
    ).fetchone()
    base = (body.api_base or "").strip() or ((row.api_base or "").strip() if row else "")
    api_key = (body.api_key or "").strip() or ((row.api_key or "").strip() if row else "")
    if not base:
        return ApiResponse(ok=False, error="未配置 API 地址")
    if not api_key:
        return ApiResponse(ok=False, error="未配置 API 密钥")
    base = base.rstrip("/")
    url = f"{base}/v1/models"
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers={"Authorization": f"Bearer {api_key}"}, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status == 200:
                    masked = api_key[:7] + "***" + api_key[-4:] if len(api_key) > 11 else "***"
                    await db.execute(
                        text(
                            """
                            UPDATE project_llm_bindings
                            SET api_key=:api_key, api_key_masked=:masked, api_base=:api_base,
                                is_verified=TRUE, verified_at=CURRENT_TIMESTAMP, is_enabled=TRUE, updated_at=CURRENT_TIMESTAMP
                            WHERE project_id=:proj AND provider_id=:pid
                            """
                        ),
                        {"proj": scope_key, "pid": body.provider_id, "api_key": api_key, "masked": masked, "api_base": base},
                    )
                    await db.commit()
                    return ApiResponse(ok=True, data={"success": True})
                text_body = await resp.text()
                return ApiResponse(ok=False, error=f"验证失败: {resp.status} {text_body[:200]}")
    except Exception as e:
        return ApiResponse(ok=False, error=str(e)[:300])


@router.post("/models", response_model=ApiResponse)
async def create_model(
    body: ModelCreate = Body(...),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """为项目下指定厂商新增模型（写入 project_llm_bindings）。"""
    pid = await _assert_llm_scope_write_or_403(db, current_user, body.project_id)
    scope_key = await _resolve_llm_scope_key(db, pid)
    await _ensure_project_llm_bindings_table(db)
    base_row = (
        await db.execute(
            text(
                "SELECT provider_code, provider_name, api_base, api_key, api_key_masked, editable_roles "
                "FROM project_llm_bindings WHERE project_id=:proj AND provider_id=:pid "
                "ORDER BY updated_at DESC NULLS LAST LIMIT 1"
            ),
            {"proj": scope_key, "pid": body.provider_id},
        )
    ).fetchone()
    if not base_row:
        raise HTTPException(status_code=404, detail="该项目下暂无该厂商绑定")
    await db.execute(
        text(
            """
            INSERT INTO project_llm_bindings
              (project_id, provider_id, provider_code, provider_name, model_name, api_base, api_key, api_key_masked, editable_roles, is_enabled, created_by, updated_by)
            VALUES
              (:proj, :pid, :pcode, :pname, :mname, :api_base, :api_key, :api_key_masked, :editable_roles, TRUE, :uid, :uid)
            """
        ),
        {
            "proj": scope_key,
            "pid": body.provider_id,
            "pcode": base_row.provider_code or f"provider_{body.provider_id}",
            "pname": base_row.provider_name or f"Provider {body.provider_id}",
            "mname": (body.name or "").strip(),
            "api_base": base_row.api_base or "",
            "api_key": base_row.api_key or "",
            "api_key_masked": base_row.api_key_masked,
            "editable_roles": base_row.editable_roles or "SUPER_ADMIN,ADMIN",
            "uid": str(getattr(current_user, "id", "") or ""),
        },
    )
    await db.commit()
    return ApiResponse(ok=True, data={"created": True})


@router.post("/models/{model_id}/delete", response_model=ApiResponse)
async def delete_model(
    model_id: int,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
    project_id: str = Query(..., description="项目 ID"),
):
    """删除模型（删除 project_llm_bindings 行）。"""
    pid = await _assert_llm_scope_write_or_403(db, current_user, project_id)
    scope_key = await _resolve_llm_scope_key(db, pid)
    r = await db.execute(
        text("SELECT provider_id, model_name FROM project_llm_bindings WHERE id = :id AND project_id = :proj"),
        {"id": model_id, "proj": scope_key},
    )
    row = r.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="模型不存在")
    await db.execute(
        text("DELETE FROM project_llm_bindings WHERE id=:id AND project_id=:proj"),
        {"id": model_id, "proj": scope_key},
    )
    await db.commit()
    return ApiResponse(ok=True, data={"deleted": True})


@router.patch("/models/{model_id}", response_model=ApiResponse)
async def update_model(
    model_id: int,
    body: ModelUpdate = Body(...),
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
    project_id: str = Query(..., description="项目 ID"),
):
    """编辑模型名称（更新 project_llm_bindings.model_name）。"""
    pid = await _assert_llm_scope_write_or_403(db, current_user, project_id)
    scope_key = await _resolve_llm_scope_key(db, pid)
    r = await db.execute(
        text("SELECT id FROM project_llm_bindings WHERE id = :id AND project_id = :proj"),
        {"id": model_id, "proj": scope_key},
    )
    row = r.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="模型不存在")
    new_name = (body.name or body.display_name or "").strip()
    if not new_name:
        return ApiResponse(ok=True, data={"updated": False})
    await db.execute(
        text("UPDATE project_llm_bindings SET model_name=:new_name, updated_at=CURRENT_TIMESTAMP WHERE id=:id AND project_id=:proj"),
        {"id": model_id, "proj": scope_key, "new_name": new_name},
    )
    await db.commit()
    return ApiResponse(ok=True, data={"updated": True})
