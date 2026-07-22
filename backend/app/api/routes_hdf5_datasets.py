"""
HDF5 数据集 API 路由
"""
import os
import h5py
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, Query, HTTPException, UploadFile, File, Form, Body, status
from fastapi.responses import StreamingResponse
import csv
import io
import shutil
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.data_assets_session import get_data_assets_db
from app.crud.data_asset import (
    create_asset,
    delete_asset,
    get_asset_by_file_path,
    get_asset_by_id,
    get_assets,
    next_code,
    update_asset,
)
from app.schemas.data_asset import DataAssetCreate, DataAssetQueryParams
from app.schemas.hdf5_dataset import (
    DatasetListResponse,
    HDF5DatasetResponse,
    ImportRequest,
    ImportResponse,
    BatchImportResponse,
    BatchImportResult,
)
from app.schemas.common import ApiResponse
from app.services.asset_meta_parser import parse_meta_for_asset
from app.core.deps import get_current_user, require_super_admin, require_super_admin_or_team_admin
from app.core.data_asset_access import (
    data_assets_allowed_project_ids,
    data_asset_visible_to_user,
    user_cannot_delete_data_asset,
    assert_may_write_project_for_data_asset_import,
)
from app.models.user import User

router = APIRouter()


def _normalize_source(src: Optional[str]) -> str:
    v = (src or "").strip().lower()
    if v in ("import", "local", "本地", ""):
        return "local"
    return v


def _normalize_format(fmt: Optional[str]) -> str:
    v = (fmt or "").strip().lower()
    if v in ("hdf5", "h5"):
        return "hdf5"
    if v in ("mcap",):
        return "mcap"
    if v in ("lerobot", "le robot"):
        return "lerobot"
    return v or "hdf5"


def _asset_to_dataset_response(asset) -> HDF5DatasetResponse:
    return HDF5DatasetResponse(
        id=int(asset.id),
        name=str(asset.filename),
        project=asset.project_id or asset.project_name,
        task=None,
        device=None,
        uploader=None,
        source=_normalize_source(getattr(asset, "source", None)),
        created_at=asset.created_at,
        file_size_bytes=int(asset.file_size_bytes or 0),
        duration_sec=None,
        format=(asset.format or "").upper() if asset.format else "HDF5",
        storage_type="local",
        storage_uri=str(asset.file_path),
        qc_status="pending",
        label_status="unlabeled",
        assign_status="unassigned",
        tags=None,
    )


@router.get("/health")
async def health_check():
    """健康检查"""
    return {"ok": True}


def validate_hdf5_file(file_path: str) -> tuple:
    """验证 HDF5 文件是否有效"""
    try:
        if not os.path.exists(file_path):
            return False, "文件不存在"
        
        if not os.path.isfile(file_path):
            return False, "不是文件"
        
        # 尝试打开 HDF5 文件
        with h5py.File(file_path, 'r') as f:
            # 可选：读取少量 group 信息
            groups = list(f.keys())[:5]  # 只读取前 5 个 group
            return True, ""
    except Exception as e:
        return False, str(e)[:100]  # 限制错误信息长度


def get_storage_dir() -> Path:
    """获取存储目录"""
    project_root = Path(__file__).parent.parent.parent.parent
    storage_dir = project_root / "backend" / "storage" / "hdf5"
    storage_dir.mkdir(parents=True, exist_ok=True)
    return storage_dir


@router.post("/import", response_model=ApiResponse)
async def import_file_multipart(
    files: List[UploadFile] = File(...),
    project: str = Form(...),
    tags: Optional[str] = Form(None),
    current_user: User = Depends(require_super_admin_or_team_admin),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """导入文件（multipart/form-data，支持单文件或批量）；与数据页主导入权限口径一致（超管/团队管理员账号）。"""
    if not files:
        return ApiResponse(
            ok=False,
            error="请至少上传一个文件"
        )
    p, werr = await assert_may_write_project_for_data_asset_import(db, current_user, project)
    if werr or not p:
        return ApiResponse(ok=False, error=werr or "项目校验失败")
    resolved_pid = str(p.id)
    resolved_pname = (p.name or resolved_pid).strip()

    storage_dir = get_storage_dir()
    results: List[BatchImportResult] = []
    
    for upload_file in files:
        filename = upload_file.filename or "unknown"
        
        # 检查文件扩展名
        if not filename.lower().endswith(('.hdf5', '.h5', '.mcap', '.zip')):
            results.append(BatchImportResult(
                filename=filename,
                success=False,
                message="只支持 .hdf5/.h5, .mcap, .zip 文件"
            ))
            continue
        
        try:
            # 保存文件到存储目录
            file_path = storage_dir / filename
            
            # 如果文件已存在，检查是否已入库
            if file_path.exists():
                existing = await get_asset_by_file_path(db, str(file_path))
                if existing:
                    results.append(BatchImportResult(
                        filename=filename,
                        success=False,
                        message="文件已存在，不能重复导入"
                    ))
                    continue
            
            # 保存上传的文件
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(upload_file.file, f)
            
            # 验证文件是否有效
            if filename.lower().endswith(('.hdf5', '.h5')):
                valid, reason = validate_hdf5_file(str(file_path))
                if not valid:
                    # 删除无效文件
                    if file_path.exists():
                        file_path.unlink()
                    results.append(BatchImportResult(
                        filename=filename,
                        success=False,
                        message=f"文件验证失败: {reason}"
                    ))
                    continue
            
            # 获取文件大小
            file_size_bytes = file_path.stat().st_size
            
            # 检查是否已存在（再次检查，防止并发）
            existing = await get_asset_by_file_path(db, str(file_path))
            if existing:
                # 如果已存在，删除刚保存的文件
                if file_path.exists():
                    file_path.unlink()
                results.append(BatchImportResult(
                    filename=filename,
                    success=False,
                    message="文件已存在，不能重复导入"
                ))
                continue
            
            # Determine format based on extension
            file_ext = Path(filename).suffix.lower()
            fmt = "MCAP" if file_ext == ".mcap" else "LeRobot" if file_ext == ".zip" else "HDF5"
            
            code = await next_code(db)
            create_data = DataAssetCreate(
                code=code,
                filename=filename,
                format=_normalize_format(fmt),
                source="import",
                project_id=resolved_pid,
                project_name=resolved_pname,
                file_path=str(file_path),
                file_size_bytes=file_size_bytes,
                meta=None,
                parse_status="解析中",
                error_msg=None,
                operator_name=(getattr(current_user, "username", None) or "").strip() or None,
            )
            asset = await create_asset(db, create_data)
            meta_json, parse_status, err_msg = parse_meta_for_asset(str(file_path), _normalize_format(fmt))
            await update_asset(
                db,
                asset.id,
                parse_status=parse_status,
                error_msg=err_msg,
                meta=meta_json,
            )
            results.append(BatchImportResult(
                filename=filename,
                success=True,
                message="导入成功",
                dataset_id=asset.id
            ))
            
        except Exception as e:
            # 如果保存失败，尝试删除文件
            file_path = storage_dir / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                except:
                    pass
            
            results.append(BatchImportResult(
                filename=filename,
                success=False,
                message=f"导入失败: {str(e)[:200]}"
            ))
    
    # 如果是单文件，返回单文件格式的响应（兼容旧接口）
    if len(files) == 1:
        result = results[0]
        if result.success:
            return ApiResponse(
                ok=True,
                data=ImportResponse(
                    success=True,
                    message=result.message or "导入成功",
                    dataset_id=result.dataset_id
                )
            )
        else:
            return ApiResponse(
                ok=False,
                error=result.message or "导入失败"
            )
    else:
        # 批量导入返回批量格式
        success_count = sum(1 for r in results if r.success)
        return ApiResponse(
            ok=True,
            data=BatchImportResponse(
                success=success_count > 0,
                message=f"导入完成：成功 {success_count}，失败 {len(results) - success_count}",
                results=results
            )
        )


@router.post("/import/path", response_model=ApiResponse)
async def import_file_by_path(
    request: ImportRequest,
    current_user: User = Depends(require_super_admin_or_team_admin),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """导入单个文件（服务器路径扫描/注册）；仅超管或团队管理员账号，避免未认证旁路入库。"""
    p, werr = await assert_may_write_project_for_data_asset_import(db, current_user, request.project_id)
    if werr or not p:
        return ApiResponse(ok=False, error=werr or "项目校验失败")
    resolved_pid = str(p.id)
    resolved_pname = (p.name or resolved_pid).strip()

    file_path = request.file_path.strip()
    
    # 如果输入的是相对路径或文件名，尝试解析为绝对路径
    if not os.path.isabs(file_path):
        # 尝试在当前工作目录查找
        current_dir = os.getcwd()
        # 尝试在项目根目录查找
        project_root = Path(__file__).parent.parent.parent.parent
        # 尝试在常见的数据目录查找
        possible_paths = [
            file_path,  # 原始路径
            os.path.join(current_dir, file_path),  # 当前目录
            os.path.join(project_root, file_path),  # 项目根目录
            os.path.join(project_root, "11", file_path),  # 项目下的 11 目录
        ]
        
        # 查找第一个存在的文件
        found_path = None
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(path):
                found_path = os.path.abspath(path)
                break
        
        if found_path:
            file_path = found_path
        else:
            # 如果都找不到，尝试递归搜索项目目录（限制深度和排除某些目录）
            max_depth = 3
            excluded_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', '.pids'}
            found = False
            for root, dirs, files in os.walk(project_root):
                # 限制搜索深度
                depth = root[len(str(project_root)):].count(os.sep)
                if depth > max_depth:
                    dirs[:] = []  # 不再深入
                    continue
                
                # 排除某些目录
                dirs[:] = [d for d in dirs if d not in excluded_dirs]
                
                if file_path in files:
                    file_path = os.path.abspath(os.path.join(root, file_path))
                    found = True
                    break
            
            if not found:
                return ApiResponse(
                    ok=False,
                    error=f"文件不存在: {request.file_path}。请确保输入完整路径，或文件在项目目录下。"
                )
    
    # 验证文件是否存在
    if not os.path.exists(file_path):
        return ApiResponse(
            ok=False,
            error=f"文件不存在: {file_path}"
        )
    
    if not os.path.isfile(file_path):
        return ApiResponse(
            ok=False,
            error=f"不是文件: {file_path}"
        )
    
    # 验证文件是否有效
    if file_path.lower().endswith(('.hdf5', '.h5')):
        valid, reason = validate_hdf5_file(file_path)
        if not valid:
            return ApiResponse(
                ok=False,
                error=f"文件验证失败: {reason}"
            )
    
    # 检查是否已存在
    existing = await get_asset_by_file_path(db, file_path)
    if existing:
        return ApiResponse(
            ok=False,
            error="文件已存在，不能重复导入"
        )
    
    try:
        # 获取文件信息
        file_stat = os.stat(file_path)
        file_size_bytes = file_stat.st_size
        
        # Determine format based on extension
        file_ext = Path(file_path).suffix.lower()
        fmt = "MCAP" if file_ext == ".mcap" else "LeRobot" if file_ext == ".zip" else "HDF5"
        
        code = await next_code(db)
        create_data = DataAssetCreate(
            code=code,
            filename=Path(file_path).name,
            format=_normalize_format(fmt),
            source="import",
            project_id=resolved_pid,
            project_name=resolved_pname,
            file_path=file_path,
            file_size_bytes=file_size_bytes,
            meta=None,
            parse_status="解析中",
            error_msg=None,
            operator_name=(getattr(current_user, "username", None) or "").strip() or None,
        )
        asset = await create_asset(db, create_data)
        meta_json, parse_status, err_msg = parse_meta_for_asset(file_path, _normalize_format(fmt))
        await update_asset(
            db,
            asset.id,
            parse_status=parse_status,
            error_msg=err_msg,
            meta=meta_json,
        )
        return ApiResponse(
            ok=True,
            data=ImportResponse(
                success=True,
                message="导入成功",
                dataset_id=asset.id
            )
        )
    except Exception as e:
        return ApiResponse(
            ok=False,
            error=f"导入失败: {str(e)[:200]}"
        )


@router.get("", response_model=ApiResponse)
async def list_hdf5_datasets(
    keyword: str = Query(None),
    project: str = Query(None),
    fmt: str = Query(None, alias="format", description="数据格式: hdf5 / mcap / lerobot"),
    id: int = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=5000, description="单页条数，项目详情等场景可传 2000"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """获取数据资产列表（支持 HDF5 / MCAP / LeRobot，筛选和分页）；范围与 /data-assets 一致。"""
    try:
        # 如果指定了 id，直接查询单个数据集
        if id:
            asset = await get_asset_by_id(db, id)
            if asset and await data_asset_visible_to_user(db, current_user, asset):
                return ApiResponse(
                    ok=True,
                    data=DatasetListResponse(
                        items=[_asset_to_dataset_response(asset)],
                        total=1,
                        page=1,
                        page_size=1,
                    )
                )
            return ApiResponse(
                ok=False,
                error="数据集不存在",
            )

        params = DataAssetQueryParams(
            keyword=keyword,
            project=project,
            format=_normalize_format(fmt) if fmt else None,
            page=page,
            page_size=page_size,
        )
        allowed = await data_assets_allowed_project_ids(db, current_user)
        assets, total = await get_assets(db, params, allowed_project_ids=allowed)

        return ApiResponse(
            ok=True,
            data=DatasetListResponse(
                items=[_asset_to_dataset_response(a) for a in assets],
                total=total,
                page=page,
                page_size=page_size,
            )
        )
    except Exception as e:
        import traceback
        error_detail = str(e)[:500]
        traceback.print_exc()  # 打印到控制台用于调试
        return ApiResponse(
            ok=False,
            error=f"获取数据集列表失败: {error_detail}"
        )


@router.get("/export")
async def export_datasets_csv(
    keyword: str = Query(None),
    project: str = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_data_assets_db),
):
    """导出数据集为 CSV（范围与列表一致）"""
    params = DataAssetQueryParams(
        keyword=keyword,
        project=project,
        page=1,
        page_size=10000,  # 导出时获取所有数据
    )

    allowed = await data_assets_allowed_project_ids(db, current_user)
    assets, _ = await get_assets(db, params, allowed_project_ids=allowed)
    
    # 创建 CSV
    output = io.StringIO()
    writer = csv.writer(output)
    
    _source_display = lambda s: {"local": "本地", "collect": "采集", "label": "标注", "convert": "转换"}.get(s or "", "本地")
    headers = [
        "id", "name", "project", "task", "device", "source",
        "created_at", "file_size_bytes", "duration_sec", "format",
        "storage_type", "storage_uri", "qc_status", "label_status",
        "assign_status"
    ]
    writer.writerow(headers)
    
    for asset in assets:
        dataset = _asset_to_dataset_response(asset)
        writer.writerow([
            dataset.id,
            dataset.name,
            dataset.project or "",
            dataset.task or "",
            dataset.device or "",
            _source_display(getattr(dataset, "source", None)),
            dataset.created_at.isoformat() if dataset.created_at else "",
            dataset.file_size_bytes,
            dataset.duration_sec or "",
            dataset.format,
            dataset.storage_type,
            dataset.storage_uri,
            dataset.qc_status,
            dataset.label_status,
            dataset.assign_status,
        ])
    
    output.seek(0)
    
    # 返回 CSV 文件
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="datasets.csv"'
        }
    )


@router.post("/migrate-project-binding", response_model=ApiResponse)
async def migrate_project_binding_api(
    body: dict = Body(..., description="mappings: [{ name: 项目名, id: 项目ID }, ...]"),
    _actor: User = Depends(require_super_admin),
    db: AsyncSession = Depends(get_data_assets_db),
):
    try:
        return ApiResponse(ok=True, data={"updated": 0})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ApiResponse(ok=False, error=str(e)[:300])


@router.delete("/{dataset_id}", response_model=ApiResponse)
async def delete_hdf5_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_data_assets_db),
    current_user: User = Depends(get_current_user),
):
    """删除 HDF5 数据集（只删除数据库记录，不删除文件）"""
    try:
        if user_cannot_delete_data_asset(current_user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="当前角色无权限删除数据集",
            )
        asset = await get_asset_by_id(db, dataset_id)
        if not asset or not await data_asset_visible_to_user(db, current_user, asset):
            return ApiResponse(
                ok=False,
                error="数据集不存在",
            )
        success = await delete_asset(db, dataset_id)
        if not success:
            return ApiResponse(
                ok=False,
                error="数据集不存在"
            )
        return ApiResponse(ok=True, data=None)
    except Exception as e:
        import traceback
        error_detail = str(e)[:500]
        traceback.print_exc()  # 打印到控制台用于调试
        return ApiResponse(
            ok=False,
            error=f"删除数据集失败: {error_detail}"
        )
