from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from fastapi import BackgroundTasks, Request
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.constants import audit_actions as AA
from app.core.database import SessionLocal
from app.core.roles import normalize_role
from app.models import AuditLog, User

logger = logging.getLogger(__name__)

_TEAM_RESOURCE_TYPES = frozenset({"TEAM", "team"})


def _resolve_audit_team_id(
    db: Session,
    *,
    explicit_team_id: Optional[str],
    project_id: Optional[str],
    resource_type: Optional[str],
    resource_id: Optional[str],
    user_id: Optional[str],
) -> Optional[str]:
    """
    写入前解析 team_id：显式参数 > 团队类 resource > 项目 team_id >
    用户仅在 team_users∪team_admins 中恰好归属一个团队时推断。
    """
    ex = (explicit_team_id or "").strip() or None
    if ex:
        return ex
    rt = (resource_type or "").strip()
    rid = (resource_id or "").strip() or None
    if rt in _TEAM_RESOURCE_TYPES and rid:
        return rid
    pid = (project_id or "").strip() or None
    if pid:
        try:
            row = db.execute(
                text("SELECT team_id FROM projects WHERE id = :pid LIMIT 1"),
                {"pid": pid},
            ).fetchone()
            if row and row[0]:
                s = str(row[0]).strip()
                if s:
                    return s
        except Exception:
            logger.debug("audit team resolve from project_id failed", exc_info=True)
    uid = (user_id or "").strip() or None
    if uid:
        try:
            rows = db.execute(
                text(
                    """
                    SELECT DISTINCT team_id FROM (
                        SELECT team_id FROM team_users WHERE user_id = :uid
                        UNION
                        SELECT team_id FROM team_admins WHERE user_id = :uid
                    ) sub
                    """
                ),
                {"uid": uid},
            ).fetchall()
            if len(rows) == 1 and rows[0][0]:
                s = str(rows[0][0]).strip()
                if s:
                    return s
        except Exception:
            logger.debug("audit team resolve from user_id failed", exc_info=True)
    return None


def _role_str(user: Optional[User]) -> Optional[str]:
    if user is None:
        return None
    try:
        return normalize_role(user.role).value
    except Exception:
        r = getattr(user, "role", None)
        return getattr(r, "value", str(r)) if r is not None else None


def request_client_ip(request: Optional[Request]) -> Optional[str]:
    if not request or not request.client:
        return None
    return request.client.host


def request_user_agent(request: Optional[Request]) -> Optional[str]:
    if not request:
        return None
    return request.headers.get("user-agent")


def log_audit(
    db: Session,
    *,
    user: Optional[User] = None,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    role: Optional[str] = None,
    project_id: Optional[str] = None,
    project_name: Optional[str] = None,
    team_id: Optional[str] = None,
    action_type: str,
    action_label: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    resource_name: Optional[str] = None,
    result: str = "SUCCESS",
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
    detail_json: Optional[Mapping[str, Any]] = None,
    error_message: Optional[str] = None,
    request: Optional[Request] = None,
) -> Optional[AuditLog]:
    """
    写入审计日志（使用调用方提供的 Session，成功则 commit）。
    若需与主业务事务隔离，请使用 log_audit_safe。
    """
    uid = user_id or (str(user.id) if user else None)
    uname = username or (user.username if user else None)
    rrole = role if role is not None else _role_str(user)

    if ip is None and request is not None:
        ip = request_client_ip(request)
    if user_agent is None and request is not None:
        user_agent = request_user_agent(request)

    label = action_label or AA.ACTION_LABELS_ZH.get(action_type, action_type)

    resolved_team_id = _resolve_audit_team_id(
        db,
        explicit_team_id=team_id,
        project_id=project_id,
        resource_type=resource_type,
        resource_id=resource_id,
        user_id=uid,
    )

    row = AuditLog(
        user_id=uid,
        username=uname,
        role=rrole,
        project_id=(project_id or None),
        project_name=(project_name or None),
        team_id=resolved_team_id,
        action_type=action_type,
        action_label=label,
        resource_type=resource_type,
        resource_id=resource_id,
        resource_name=resource_name,
        result=result,
        ip=ip,
        user_agent=user_agent,
        detail_json=dict(detail_json) if detail_json is not None else None,
        error_message=error_message,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def log_audit_safe(
    *,
    user: Optional[User] = None,
    user_id: Optional[str] = None,
    username: Optional[str] = None,
    role: Optional[str] = None,
    project_id: Optional[str] = None,
    project_name: Optional[str] = None,
    team_id: Optional[str] = None,
    action_type: str,
    action_label: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    resource_name: Optional[str] = None,
    result: str = "SUCCESS",
    ip: Optional[str] = None,
    user_agent: Optional[str] = None,
    detail_json: Optional[Mapping[str, Any]] = None,
    error_message: Optional[str] = None,
    request: Optional[Request] = None,
) -> None:
    """独立 Session 写入；失败仅打日志，不影响主业务。"""
    db: Session | None = None
    try:
        uid = user_id or (str(user.id) if user else None)
        uname = username or (user.username if user else None)
        rrole = role if role is not None else _role_str(user)

        if ip is None and request is not None:
            ip = request_client_ip(request)
        if user_agent is None and request is not None:
            user_agent = request_user_agent(request)

        label = action_label or AA.ACTION_LABELS_ZH.get(action_type, action_type)

        db = SessionLocal()
        resolved_team_id = _resolve_audit_team_id(
            db,
            explicit_team_id=team_id,
            project_id=project_id,
            resource_type=resource_type,
            resource_id=resource_id,
            user_id=uid,
        )
        row = AuditLog(
            user_id=uid,
            username=uname,
            role=rrole,
            project_id=(project_id or None),
            project_name=(project_name or None),
            team_id=resolved_team_id,
            action_type=action_type,
            action_label=label,
            resource_type=resource_type,
            resource_id=resource_id,
            resource_name=resource_name,
            result=result,
            ip=ip,
            user_agent=user_agent,
            detail_json=dict(detail_json) if detail_json is not None else None,
            error_message=error_message,
        )
        db.add(row)
        db.commit()
    except Exception:
        logger.exception("audit log write failed action=%s", action_type)
        if db is not None:
            try:
                db.rollback()
            except Exception:
                pass
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def sync_resolve_user_for_audit(request: Optional[Request]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    从 Bearer access token 解析当前用户（同步 Session），用于无 Depends 的同步路由。
    返回 (user_id, username, role)，失败则 (None, None, None)。
    """
    if request is None:
        return None, None, None
    auth = request.headers.get("Authorization") or ""
    if not auth.startswith("Bearer "):
        return None, None, None
    token = auth.removeprefix("Bearer ").strip()
    db: Session | None = None
    try:
        from jose import JWTError

        from app.core.security import decode_token, require_token_type
        from app.services.user_service import get_user_by_account_id

        payload = decode_token(token)
        require_token_type(payload, "access")
        account_id = payload.get("sub")
        if not account_id:
            return None, None, None
        db = SessionLocal()
        u = get_user_by_account_id(db, account_id)
        if u is None or not getattr(u, "is_active", True):
            return None, None, None
        return str(u.id), u.username, _role_str(u)
    except (JWTError, Exception):
        return None, None, None
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def enqueue_audit_log(background_tasks: BackgroundTasks, **kwargs: Any) -> None:
    """异步路由：在响应返回后由 FastAPI 后台线程执行 log_audit_safe。"""
    u = kwargs.pop("user", None)
    if u is not None:
        kwargs.setdefault("user_id", str(getattr(u, "id", "") or ""))
        kwargs.setdefault("username", getattr(u, "username", None))
        if kwargs.get("role") is None:
            kwargs["role"] = _role_str(u)

    def _run() -> None:
        log_audit_safe(**kwargs)

    background_tasks.add_task(_run)


# 兼容旧调用方（逐步淘汰）
def log_action(
    db: Session,
    action: str,
    user: Optional[User] = None,
    username: Optional[str] = None,
    detail: Optional[str] = None,
    request: Optional[Request] = None,
) -> None:
    log_audit_safe(
        user=user,
        username=username,
        action_type=action,
        action_label=AA.ACTION_LABELS_ZH.get(action, action),
        detail_json={"legacy_detail": detail} if detail else None,
        request=request,
    )
