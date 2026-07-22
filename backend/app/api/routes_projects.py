"""
项目 API：CRUD 及项目下任务/数据列表（projects 表，PostgreSQL）
"""
from typing import Optional, List

from fastapi import APIRouter, BackgroundTasks, Depends, Query, HTTPException, Request

from app.db.data_assets_session import get_data_assets_db
from app.core.deps import get_current_user
from app.core.permissions import sees_all_projects_without_filter
from app.core.project_permissions import can_delete_project, can_edit_project, can_manage_project_members
from app.core.roles import CanonicalUserRole, is_super_admin, is_team_admin_role, normalize_role
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from app.models import User, UserRole
from app.models.team import TeamUser
from app.models.project_asset import Project
from app.models.label_task_asset import LabelTask
from app.models.data_asset import DataAsset
from app.crud.project import (
    list_projects,
    get_project_by_id,
    get_project_stats,
    get_projects_stats_batch,
    create_project,
    update_project,
    delete_project_with_cascade,
    _tags_from_db,
    get_visible_project_ids,
    is_project_visible_to_user,
    upsert_project_member,
    delete_project_member,
    list_project_member_ids,
    project_ids_with_membership_for_user,
    project_member_display_counts_batch,
    project_member_display_count_for_project,
)
from app.crud import team as team_crud
from app.schemas.project import (
    ProjectCreate,
    ProjectUpdate,
    ProjectResponse,
    ProjectListResponse,
)
from app.schemas.common import ApiResponse
from app.db.session import AsyncSessionLocal
from app.crud.user import get_user_by_id
from app.services.minio_service import (
    ensure_project_bucket,
    remove_project_bucket,
    MinioConfigError,
    MinioBucketError,
)
from app.constants import audit_actions as AA
from app.constants import audit_resources as AR
from app.services.audit_service import enqueue_audit_log
from app.services.user_provision import UserProvisionError, create_user_with_allocated_account_id
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

router = APIRouter()


def _project_to_response(p: Project) -> ProjectResponse:
    return ProjectResponse(
        id=p.id,
        name=p.name,
        description=p.description,
        tags=_tags_from_db(p.tags),
        status=p.status,
        owner_id=p.owner_id,
        team_id=getattr(p, "team_id", None),
        created_at=p.created_at,
        updated_at=p.updated_at,
    )


def _project_to_response_dict_with_viewer(
    p: Project,
    *,
    viewer_user_id: str,
    member_project_ids: set[str],
    member_count: int = 0,
) -> dict:
    """序列化项目并附带当前用户对 project_members 的归属（供「共享项目」等前端 Tab 使用）。"""
    uid = (viewer_user_id or "").strip()
    oid = (getattr(p, "owner_id", None) or "").strip()
    pid = str(p.id).strip()
    d = _project_to_response(p).model_dump()
    d["viewer_is_project_owner"] = bool(uid and oid == uid)
    d["viewer_is_project_member"] = pid in member_project_ids
    d["member_count"] = int(member_count)
    return d


@router.get("", response_model=ApiResponse)
async def list_projects_api(
    status: Optional[str] = Query(None, description="按状态筛选：进行中 | 已暂停 | 已归档"),
    with_stats: bool = Query(False, description="是否返回每个项目的任务数、数据数（按 project_id 统计）"),
    team_id: Optional[str] = Query(None, description="按所属团队过滤（projects.team_id）；与可见性交集"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """获取项目列表，按更新时间倒序。with_stats=true 时每个项目带 label_task_count、dataset_count。"""
    allowed_ids: Optional[List[str]] = None
    if not sees_all_projects_without_filter(current_user):
        allowed_ids = await get_visible_project_ids(db, user_id=str(current_user.id), include_owner_projects=True)
    team_filter = (team_id or "").strip() or None
    items, total = await list_projects(
        db, status=status, allowed_project_ids=allowed_ids, team_id=team_filter
    )
    viewer_uid = str(current_user.id)
    member_pids: set[str] = set()
    if items:
        pids = [p.id for p in items]
        member_pids = await project_ids_with_membership_for_user(
            db, user_id=viewer_uid, project_ids=pids
        )
        member_display_counts = await project_member_display_counts_batch(db, items)
    else:
        member_display_counts = {}
    data: dict = {
        "items": [
            _project_to_response_dict_with_viewer(
                p,
                viewer_user_id=viewer_uid,
                member_project_ids=member_pids,
                member_count=member_display_counts.get(str(p.id).strip(), 0),
            )
            for p in items
        ],
        "total": total,
    }
    if with_stats and items:
        project_ids = [p.id for p in items]
        stats_map = await get_projects_stats_batch(db, project_ids)
        data["stats"] = stats_map
    return ApiResponse(ok=True, data=data)


@router.get("/permissions-context", response_model=ApiResponse)
async def project_permissions_context(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """
    供前端收敛项目卡片/详情操作按钮。
    - SUPER_ADMIN：team_admin_team_ids 为 null（表示不按团队列表限制，由后端全权校验）。
    - ADMIN：返回其在 team_admins 中管辖的 team_id 列表（可能为空）。
    - 其他角色：空列表（前端以 ownerId / 角色组合判断即可）。
    """
    if is_super_admin(current_user.role):
        return ApiResponse(ok=True, data={"team_admin_team_ids": None})
    if is_team_admin_role(current_user.role):
        tids = await team_crud.list_team_ids_where_user_is_team_admin(db, str(current_user.id))
        return ApiResponse(ok=True, data={"team_admin_team_ids": tids})
    return ApiResponse(ok=True, data={"team_admin_team_ids": []})


@router.post("", response_model=ApiResponse)
async def create_project_api(
    body: ProjectCreate,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """创建项目（项目名称、描述、标签）。id 可不传，由后端生成 UUID。"""
    if not (is_super_admin(current_user.role) or is_team_admin_role(current_user.role)):
        raise HTTPException(status_code=403, detail="无权限创建项目")
    body.name = (body.name or "").strip()
    if not body.name:
        raise HTTPException(status_code=400, detail="项目名称不能为空")
    if body.tags and len(body.tags) > 20:
        body.tags = body.tags[:20]
    raw_team = getattr(body, "team_id", None)
    team_id_norm = (str(raw_team).strip() if raw_team is not None else "") or None

    # 团队管理员未传 team_id 时：若仅管辖一个团队，自动绑定该团队（与前端默认 teamId 一致）
    if (
        is_team_admin_role(current_user.role)
        and not is_super_admin(current_user.role)
        and not team_id_norm
    ):
        admin_teams = await team_crud.list_team_ids_where_user_is_team_admin(db, str(current_user.id))
        if len(admin_teams) == 1:
            team_id_norm = admin_teams[0]

    if team_id_norm:
        if not await team_crud.get_team_by_id(db, team_id_norm):
            raise HTTPException(status_code=400, detail="团队不存在")
        # 团队管理员：仅可在担任 team_admins 的团队下创建项目
        if is_team_admin_role(current_user.role) and not is_super_admin(current_user.role):
            if not await team_crud.get_team_admin_by_user(db, team_id_norm, str(current_user.id)):
                raise HTTPException(status_code=403, detail="仅可在您担任团队管理员的项目所属团队下创建项目")
        creator_id = str(current_user.id)
        in_users = await team_crud.get_team_user_by_user(db, team_id_norm, creator_id)
        is_admin_of_team = await team_crud.get_team_admin_by_user(db, team_id_norm, creator_id)
        # 超管不受限；团队管理员可为所辖团队创建而无需先写入 team_users；普通路径仍须为团队成员
        if not is_super_admin(current_user.role):
            if not (in_users or is_admin_of_team):
                raise HTTPException(
                    status_code=400,
                    detail="在已绑定团队下创建项目前，请先将当前账号加入该团队的成员列表（团队管理 → 管理成员）",
                )
        body = body.model_copy(update={"team_id": team_id_norm})
    elif raw_team is not None:
        body = body.model_copy(update={"team_id": None})

    # 团队管理员创建项目必须有所属团队（避免在无 team_id 的遗留路径越权创建）
    final_team_for_admin_check = (getattr(body, "team_id", None) or "").strip() or None
    if is_team_admin_role(current_user.role) and not is_super_admin(current_user.role):
        if not final_team_for_admin_check:
            raise HTTPException(status_code=400, detail="团队管理员创建项目时必须选择所属团队")

    # Owner：表字段 owner_id 为项目内负责人；未传时默认为当前登录用户（创建者即 Owner）
    owner_norm = (body.owner_id or "").strip() or str(current_user.id)
    body = body.model_copy(update={"owner_id": owner_norm})

    # 先创建项目记录，再创建同名 MinIO bucket；若 bucket 失败则回滚项目
    project = await create_project(db, body)
    try:
        ensure_project_bucket(project.name)
    except MinioConfigError as e:
        await delete_project_with_cascade(db, project.id)
        raise HTTPException(status_code=500, detail=str(e))
    except MinioBucketError as e:
        await delete_project_with_cascade(db, project.id)
        raise HTTPException(status_code=400, detail=str(e))
    # 创建者写入 project_members（表无角色列；列表接口中 Owner 仍由 projects.owner_id 推导，与成员表并存且不重复展示）
    await upsert_project_member(db, project_id=project.id, user_id=str(current_user.id))

    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=request,
        action_type=AA.CREATE_PROJECT,
        project_id=project.id,
        project_name=project.name,
        resource_type=AR.PROJECT,
        resource_id=project.id,
        resource_name=project.name,
    )
    mp = await project_ids_with_membership_for_user(
        db, user_id=str(current_user.id), project_ids=[project.id]
    )
    mc = await project_member_display_count_for_project(db, project.id)
    return ApiResponse(
        ok=True,
        data=_project_to_response_dict_with_viewer(
            project,
            viewer_user_id=str(current_user.id),
            member_project_ids=mp,
            member_count=mc,
        ),
    )


@router.get("/{project_id}", response_model=ApiResponse)
async def get_project_api(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """获取单个项目详情。"""
    if not is_super_admin(current_user.role):
        if not await is_project_visible_to_user(
            db, project_id=project_id, user_id=str(current_user.id), include_owner_projects=True
        ):
            raise HTTPException(status_code=404, detail="项目不存在")
    project = await get_project_by_id(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    mp = await project_ids_with_membership_for_user(
        db, user_id=str(current_user.id), project_ids=[project_id]
    )
    mc = await project_member_display_count_for_project(db, project_id)
    return ApiResponse(
        ok=True,
        data=_project_to_response_dict_with_viewer(
            project,
            viewer_user_id=str(current_user.id),
            member_project_ids=mp,
            member_count=mc,
        ),
    )


@router.get("/{project_id}/stats", response_model=ApiResponse)
async def get_project_stats_api(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """按项目 id 统计：返回该项目的标注任务数、数据资产数。表与表通过 project_id 关联。"""
    if not is_super_admin(current_user.role):
        if not await is_project_visible_to_user(
            db, project_id=project_id, user_id=str(current_user.id), include_owner_projects=True
        ):
            raise HTTPException(status_code=404, detail="项目不存在")
    project = await get_project_by_id(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    label_count, dataset_count = await get_project_stats(db, project_id)
    return ApiResponse(
        ok=True,
        data={"label_task_count": label_count, "dataset_count": dataset_count},
    )


@router.patch("/{project_id}", response_model=ApiResponse)
async def update_project_api(
    project_id: str,
    body: ProjectUpdate,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """更新项目（名称、描述、标签、状态等）。"""
    project_row = await get_project_by_id(db, project_id)
    if not project_row:
        raise HTTPException(status_code=404, detail="项目不存在")
    if not is_super_admin(current_user.role):
        if not await is_project_visible_to_user(
            db, project_id=project_id, user_id=str(current_user.id), include_owner_projects=True
        ):
            raise HTTPException(status_code=404, detail="项目不存在")
    if not await can_edit_project(db, current_user, project_row):
        raise HTTPException(status_code=403, detail="无权限编辑该项目")
    project = await update_project(db, project_id, body)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=request,
        action_type=AA.UPDATE_PROJECT,
        project_id=project.id,
        project_name=project.name,
        resource_type=AR.PROJECT,
        resource_id=project.id,
        resource_name=project.name,
    )
    mp = await project_ids_with_membership_for_user(
        db, user_id=str(current_user.id), project_ids=[project.id]
    )
    mc = await project_member_display_count_for_project(db, project.id)
    return ApiResponse(
        ok=True,
        data=_project_to_response_dict_with_viewer(
            project,
            viewer_user_id=str(current_user.id),
            member_project_ids=mp,
            member_count=mc,
        ),
    )


@router.delete("/{project_id}", response_model=ApiResponse)
async def delete_project_api(
    project_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """删除项目并级联删除：删除该项目下所有标注任务、所有数据资产记录，再删除项目。通过 project_id 建立表与表关联；磁盘上的数据文件不删除。"""
    project = await get_project_by_id(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    if not is_super_admin(current_user.role):
        if not await is_project_visible_to_user(
            db, project_id=project_id, user_id=str(current_user.id), include_owner_projects=True
        ):
            raise HTTPException(status_code=404, detail="项目不存在")
    if not await can_delete_project(db, current_user, project):
        raise HTTPException(status_code=403, detail="无权限删除该项目")
    pname = project.name
    team_id_for_audit = (getattr(project, "team_id", None) or "").strip() or None
    # 先删除 MinIO bucket（含清空对象），避免项目删库后遗留孤儿桶
    try:
        remove_project_bucket(pname, force=True)
    except MinioConfigError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except MinioBucketError as e:
        raise HTTPException(status_code=400, detail=str(e))
    ok = await delete_project_with_cascade(db, project_id)
    if not ok:
        raise HTTPException(status_code=404, detail="项目不存在")
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=request,
        action_type=AA.DELETE_PROJECT,
        project_id=project_id,
        project_name=pname,
        team_id=team_id_for_audit,
        resource_type=AR.PROJECT,
        resource_id=project_id,
        resource_name=pname or project_id,
    )
    return ApiResponse(ok=True, data={"deleted": project_id})


@router.get("/{project_id}/label-tasks", response_model=ApiResponse)
async def list_project_label_tasks(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """获取项目下的标注任务列表（label_tasks 表中 project_id 匹配）。"""
    if not is_super_admin(current_user.role):
        if not await is_project_visible_to_user(
            db, project_id=project_id, user_id=str(current_user.id), include_owner_projects=True
        ):
            raise HTTPException(status_code=404, detail="项目不存在")
    stmt = (
        select(LabelTask)
        .where(LabelTask.project_id == project_id)
        .order_by(LabelTask.created_at.desc())
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    def _row_to_dict(r: LabelTask) -> dict:
        def _iso(t):
            if t is None:
                return ""
            return t.isoformat() if hasattr(t, "isoformat") else str(t)
        return {
            "id": r.id,
            "task_id": r.task_id,
            "name": r.name,
            "dataset_path": r.dataset_path or "",
            "project_id": r.project_id,
            "created_at": _iso(r.created_at),
            "updated_at": _iso(r.updated_at),
        }

    return ApiResponse(ok=True, data={"items": [_row_to_dict(r) for r in rows], "total": len(rows)})


@router.get("/{project_id}/datasets", response_model=ApiResponse)
async def list_project_datasets(
    project_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """获取项目下的数据资产列表（data_assets 表中 project_id 或 project_name 匹配）。"""
    if not is_super_admin(current_user.role):
        if not await is_project_visible_to_user(
            db, project_id=project_id, user_id=str(current_user.id), include_owner_projects=True
        ):
            raise HTTPException(status_code=404, detail="项目不存在")
    count_stmt = (
        select(func.count())
        .select_from(DataAsset)
        .where(DataAsset.project_id == project_id)
    )
    total_r = await db.execute(count_stmt)
    total = total_r.scalar() or 0

    stmt = (
        select(DataAsset)
        .where(DataAsset.project_id == project_id)
        .order_by(DataAsset.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()

    def _asset_to_item(a: DataAsset) -> dict:
        def _iso(t):
            if t is None:
                return ""
            return t.isoformat() if hasattr(t, "isoformat") else str(t)
        return {
            "id": a.id,
            "dataset_id": a.dataset_id,
            "filename": a.filename,
            "format": a.format,
            "file_path": a.file_path,
            "project_id": a.project_id,
            "project_name": a.project_name,
            "created_at": _iso(a.created_at),
        }

    return ApiResponse(
        ok=True,
        data={
            "items": [_asset_to_item(a) for a in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
        },
    )


class UpsertProjectMemberRequest(BaseModel):
    user_id: str


class CreateProjectUserRequest(BaseModel):
    """在项目所属团队维度分配 account_id（teams.code+流水），并加入 team_users 与 project_members。"""

    username: str
    password: str


@router.post("/{project_id}/users", response_model=ApiResponse)
async def create_project_user(
    project_id: str,
    body: CreateProjectUserRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """
    在项目中新建平台用户（普通 USER 角色）：登录账号由后端按团队规则生成，展示名为 username。
    要求项目已绑定 team_id；调用方须具备 can_manage_project_members。
    """
    project = await get_project_by_id(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    if not is_super_admin(current_user.role):
        if not await is_project_visible_to_user(
            db, project_id=project_id, user_id=str(current_user.id), include_owner_projects=True
        ):
            raise HTTPException(status_code=404, detail="项目不存在")
    if not await can_manage_project_members(db, current_user, project):
        raise HTTPException(status_code=403, detail="无权限管理该项目成员")

    team_id = (getattr(project, "team_id", None) or "").strip()
    if not team_id:
        raise HTTPException(
            status_code=400,
            detail="项目未关联团队，无法在此创建用户；请先在项目设置中关联团队",
        )

    display_username = (body.username or "").strip()
    if not display_username:
        raise HTTPException(status_code=400, detail="用户名不能为空")
    if not (body.password or "").strip():
        raise HTTPException(status_code=400, detail="密码不能为空")

    creator = str(current_user.id)

    async with AsyncSessionLocal() as udb:
        try:
            user = await create_user_with_allocated_account_id(
                udb,
                db,
                display_username=display_username,
                password=body.password,
                role=UserRole.USER,
                team_id_for_account=team_id,
            )
        except UserProvisionError as e:
            key = str(e)
            detail = {
                "USERNAME_EMPTY": "请输入用户名",
                "PASSWORD_EMPTY": "密码不能为空",
                "TEAM_NOT_FOUND": "团队不存在",
                "TEAM_INACTIVE": "团队已停用",
                "ACCOUNT_ID_ALLOCATION_FAILED": "无法分配登录账号，请稍后重试",
                "USER_CREATE_CONFLICT": "创建用户失败（账号分配冲突等），请稍后重试",
            }.get(key, "创建用户失败")
            raise HTTPException(status_code=400, detail=detail)

    uid = str(user.id)
    try:
        tu = await team_crud.get_team_user_by_user(db, team_id, uid)
        if tu is None:
            db.add(
                TeamUser(
                    team_id=team_id,
                    user_id=uid,
                    created_by=creator,
                )
            )
        await upsert_project_member(db, project_id=project_id, user_id=uid)
        await db.commit()
    except IntegrityError:
        await db.rollback()
        async with AsyncSessionLocal() as cleanup_udb:
            await cleanup_udb.execute(delete(User).where(User.id == user.id))
            await cleanup_udb.commit()
        raise HTTPException(
            status_code=400,
            detail="添加团队成员或项目成员失败（可能因数据冲突），请稍后重试",
        )
    except Exception:
        await db.rollback()
        async with AsyncSessionLocal() as cleanup_udb:
            await cleanup_udb.execute(delete(User).where(User.id == user.id))
            await cleanup_udb.commit()
        raise HTTPException(status_code=500, detail="创建用户关联失败，请稍后重试")

    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=request,
        action_type=AA.CREATE_USER,
        project_id=project_id,
        project_name=getattr(project, "name", None),
        resource_type=AR.USER,
        resource_id=uid,
        resource_name=display_username,
        detail_json={
            "context": "project",
            "project_id": project_id,
            "team_id": team_id,
            "new_username": display_username,
            "new_account_id": user.account_id,
        },
    )
    return ApiResponse(
        ok=True,
        data={
            "user_id": uid,
            "account_id": user.account_id,
            "username": user.username,
        },
    )


@router.get("/{project_id}/members", response_model=ApiResponse)
async def list_project_members(
    project_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    if not is_super_admin(current_user.role):
        if not await is_project_visible_to_user(
            db, project_id=project_id, user_id=str(current_user.id), include_owner_projects=True
        ):
            raise HTTPException(status_code=404, detail="项目不存在")
    project = await get_project_by_id(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")

    member_ids = await list_project_member_ids(db, project_id=project_id)
    owner_id = (getattr(project, "owner_id", None) or "").strip()
    all_ids = []
    if owner_id:
        all_ids.append(owner_id)
    for uid in member_ids:
        if uid and uid not in all_ids:
            all_ids.append(uid)

    user_map: dict[str, str] = {}
    async with AsyncSessionLocal() as udb:
        for uid in all_ids:
            u = await get_user_by_id(udb, uid)
            if u is not None:
                user_map[uid] = str(getattr(u, "username", "") or "")

    items = []
    if owner_id:
        items.append(
            {
                "user_id": owner_id,
                "username": user_map.get(owner_id, "") or owner_id,
                "role": "Owner",
            }
        )
    role_by_uid: dict[str, CanonicalUserRole] = {}
    async with AsyncSessionLocal() as udb:
        for uid in member_ids:
            if not uid or uid == owner_id:
                continue
            u = await get_user_by_id(udb, uid)
            role_by_uid[uid] = normalize_role(getattr(u, "role", None)) if u else CanonicalUserRole.USER

    for uid in member_ids:
        if not uid or uid == owner_id:
            continue
        pr = role_by_uid.get(uid, CanonicalUserRole.USER)
        display_role = (
            "Admin"
            if pr
            in (
                CanonicalUserRole.SUPER_ADMIN,
                CanonicalUserRole.ADMIN,
                CanonicalUserRole.OWNER,
            )
            else "Member"
        )
        items.append(
            {
                "user_id": uid,
                "username": user_map.get(uid, "") or uid,
                "role": display_role,
            }
        )
    return ApiResponse(ok=True, data={"items": items, "total": len(items)})


@router.post("/{project_id}/members", response_model=ApiResponse)
async def add_project_member(
    project_id: str,
    body: UpsertProjectMemberRequest,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    project = await get_project_by_id(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    if not is_super_admin(current_user.role):
        if not await is_project_visible_to_user(
            db, project_id=project_id, user_id=str(current_user.id), include_owner_projects=True
        ):
            raise HTTPException(status_code=404, detail="项目不存在")
    if not await can_manage_project_members(db, current_user, project):
        raise HTTPException(status_code=403, detail="无权限管理该项目成员")

    uid = (body.user_id or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id 不能为空")

    member_username = ""
    async with AsyncSessionLocal() as udb:
        u = await get_user_by_id(udb, uid)
        if u is None or not getattr(u, "is_active", True):
            raise HTTPException(status_code=404, detail="用户不存在")
        member_username = str(getattr(u, "username", "") or "").strip() or uid

    team_id = (getattr(project, "team_id", None) or "").strip()
    if team_id:
        tu = await team_crud.get_team_user_by_user(db, team_id, uid)
        if tu is None:
            raise HTTPException(
                status_code=400,
                detail="用户须先加入该项目所属团队（team_users）后才能加入项目；团队管理员不会自动视为团队成员",
            )

    await upsert_project_member(db, project_id=project_id, user_id=uid)
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=request,
        action_type=AA.ADD_PROJECT_MEMBER,
        project_id=project_id,
        project_name=getattr(project, "name", None),
        resource_type=AR.PROJECT_MEMBER,
        resource_id=f"{project_id}:{uid}",
        resource_name=member_username,
        detail_json={"member_user_id": uid, "member_username": member_username},
    )
    return ApiResponse(ok=True, data={"added": True, "user_id": uid})


@router.delete("/{project_id}/members/{user_id}", response_model=ApiResponse)
async def remove_project_member(
    project_id: str,
    user_id: str,
    request: Request,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    project = await get_project_by_id(db, project_id)
    if not project:
        raise HTTPException(status_code=404, detail="项目不存在")
    if not is_super_admin(current_user.role):
        if not await is_project_visible_to_user(
            db, project_id=project_id, user_id=str(current_user.id), include_owner_projects=True
        ):
            raise HTTPException(status_code=404, detail="项目不存在")
    if not await can_manage_project_members(db, current_user, project):
        raise HTTPException(status_code=403, detail="无权限管理该项目成员")
    uid = (user_id or "").strip()
    if not uid:
        raise HTTPException(status_code=400, detail="user_id 不能为空")
    owner_id = (getattr(project, "owner_id", None) or "").strip()
    if owner_id and uid == owner_id:
        raise HTTPException(status_code=400, detail="不能移除项目负责人")

    deleted = await delete_project_member(db, project_id=project_id, user_id=uid)
    if not deleted:
        raise HTTPException(status_code=404, detail="成员不存在")
    member_username = ""
    async with AsyncSessionLocal() as udb:
        u = await get_user_by_id(udb, uid)
        if u is not None:
            member_username = str(getattr(u, "username", "") or "").strip() or uid
        else:
            member_username = uid
    enqueue_audit_log(
        background_tasks,
        user=current_user,
        request=request,
        action_type=AA.REMOVE_PROJECT_MEMBER,
        project_id=project_id,
        project_name=getattr(project, "name", None),
        resource_type=AR.PROJECT_MEMBER,
        resource_id=f"{project_id}:{uid}",
        resource_name=member_username,
        detail_json={"member_user_id": uid, "member_username": member_username},
    )
    return ApiResponse(ok=True, data={"removed": True, "user_id": uid})
