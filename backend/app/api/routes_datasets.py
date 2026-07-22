import asyncio

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from uuid import UUID
from typing import List, Optional
import os
import h5py
import shutil
from pathlib import Path
from datetime import datetime

from app.core.deps import get_current_user, require_super_admin_or_team_admin
from app.core.data_asset_access import assert_may_write_project_for_data_asset_import, user_cannot_delete_data_asset
from app.db.session import get_db
from app.db.data_assets_session import get_data_assets_db
from app.crud.dataset import (
    get_datasets,
    get_dataset,
    create_dataset,
    update_dataset,
    delete_dataset
)
from app.schemas.dataset import DatasetCreate, DatasetUpdate, DatasetResponse
from app.crud.data_asset import create_asset, get_asset_by_file_path, next_code as next_asset_code, update_asset
from app.schemas.data_asset import DataAssetCreate
from app.schemas.common import ApiResponse
from app.schemas.workspace_benchmark import DatasetListResponse, DatasetResponse
from app.models.user import User
from app.services.asset_meta_parser import parse_meta_for_asset
from app.services import workspace_dataset_service as workspace_dataset_svc

router = APIRouter()


@router.get("/list", response_model=DatasetListResponse)
async def list_platform_datasets_for_evaluation(
    _: User = Depends(get_current_user),
) -> DatasetListResponse:
    """平台已构建数据集列表（评测「离线数据集评测」下拉数据源）。"""
    rows = await asyncio.to_thread(workspace_dataset_svc.list_datasets)
    datasets = [DatasetResponse(**row) for row in rows]
    return DatasetListResponse(datasets=datasets, total=len(datasets))


@router.get("", response_model=ApiResponse)
async def list_datasets(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取数据集列表"""
    datasets = await get_datasets(db, skip=skip, limit=limit)
    return ApiResponse(
        ok=True,
        data=[DatasetResponse.model_validate(d) for d in datasets]
    )


@router.post("", response_model=ApiResponse)
async def create_new_dataset(
    dataset: DatasetCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建数据集"""
    db_dataset = await create_dataset(db, dataset)
    return ApiResponse(
        ok=True,
        data=DatasetResponse.model_validate(db_dataset)
    )


@router.get("/{dataset_id}", response_model=ApiResponse)
async def get_dataset_by_id(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """根据 ID 获取数据集"""
    db_dataset = await get_dataset(db, dataset_id)
    if db_dataset is None:
        return ApiResponse(
            ok=False,
            error="Dataset not found"
        )
    return ApiResponse(
        ok=True,
        data=DatasetResponse.model_validate(db_dataset)
    )


@router.patch("/{dataset_id}", response_model=ApiResponse)
async def update_dataset_by_id(
    dataset_id: UUID,
    dataset_update: DatasetUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新数据集"""
    db_dataset = await update_dataset(db, dataset_id, dataset_update)
    if db_dataset is None:
        return ApiResponse(
            ok=False,
            error="Dataset not found"
        )
    return ApiResponse(
        ok=True,
        data=DatasetResponse.model_validate(db_dataset)
    )


@router.delete("/{dataset_id}", response_model=ApiResponse)
async def delete_dataset_by_id(
    dataset_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除数据集（旧表 datasets；USER 禁止删除，与 data_assets 一致）"""
    if user_cannot_delete_data_asset(current_user):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="当前角色无权限删除数据集",
        )
    success = await delete_dataset(db, dataset_id)
    if not success:
        return ApiResponse(
            ok=False,
            error="Dataset not found"
        )
    return ApiResponse(ok=True, data=None)


def get_storage_dir() -> Path:
    """获取 HDF5 文件存储目录"""
    try:
        project_root = Path(__file__).parent.parent.parent.parent
        storage_dir = project_root / "backend" / "storage" / "hdf5"
        storage_dir.mkdir(parents=True, exist_ok=True)
        # 检查目录是否可写
        test_file = storage_dir / ".test_write"
        try:
            test_file.touch()
            test_file.unlink()
        except Exception:
            raise PermissionError(f"存储目录不可写: {storage_dir}")
        return storage_dir
    except Exception as e:
        raise RuntimeError(f"无法创建存储目录: {str(e)}")


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


def _sanitize_project_path(project: str) -> Path:
    """将项目名称转为单层安全目录名（仅取第一段，忽略原本的文件夹路径如 frank_data）。"""
    if not project or not project.strip():
        return Path(".")
    # 只取第一段作为项目目录，不在磁盘上保留多级（如 franka/frank_data -> 只用 franka）
    first = project.strip().replace(" ", "_").split("/")[0].strip()
    safe = "".join(c for c in first if c.isalnum() or c in "._-").strip()
    return Path(safe) if safe and safe != ".." else Path(".")


def _import_format_from_filename(filename: str) -> Optional[str]:
    """根据文件名返回入库格式：HDF5 / MCAP / LeRobot，不支持则返回 None。"""
    lower = filename.lower().strip()
    if lower.endswith(".hdf5") or lower.endswith(".h5"):
        return "HDF5"
    if lower.endswith(".mcap"):
        return "MCAP"
    if lower.endswith(".zip") and "lerobot" in lower:
        return "LeRobot"
    if lower.endswith(".zip"):
        return "LeRobot"  # 无 lerobot 关键字也视为 LeRobot
    return None


def generate_unique_filename(storage_dir: Path, original_filename: str, project_subdir: Optional[str] = None) -> Path:
    """按项目名称分子目录保存；若文件夹不存在则自动创建。同名冲突时追加时间戳。"""
    if project_subdir:
        rel = _sanitize_project_path(project_subdir)
        base_dir = (storage_dir / rel).resolve()
        # 确保路径仍在 storage_dir 下，防止路径遍历
        try:
            base_dir.resolve().relative_to(storage_dir.resolve())
        except ValueError:
            base_dir = storage_dir
        base_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_dir = storage_dir

    file_path = base_dir / original_filename

    if not file_path.exists():
        return file_path

    name_parts = original_filename.rsplit('.', 1)
    if len(name_parts) == 2:
        name, ext = name_parts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        new_filename = f"{name}_{timestamp}.{ext}"
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        new_filename = f"{original_filename}_{timestamp}"

    return base_dir / new_filename


@router.post("/import", response_model=ApiResponse)
async def import_datasets(
    files: List[UploadFile] = File(...),
    project: Optional[str] = Form(None),
    current_user: User = Depends(require_super_admin_or_team_admin),
    assets_db: AsyncSession = Depends(get_data_assets_db),
):
    """
    【旧链路 / 非数据页主导入】导入数据资产（multipart/form-data），支持 HDF5 / MCAP / LeRobot。

    数据管理页请使用 POST /api/data-assets/import（落盘 + MinIO + meta.storage.minio_path）。
    本接口将文件保存至 storage 目录且不入 MinIO，可能导致后续导出（依赖 MinIO）失败。
    
    说明：
    - 浏览器选择文件夹/多文件，只能拿到 File 对象；无法把"本地路径"当成服务器可访问路径。
    - 因此批量导入必须通过上传文件内容，让后端负责落盘并入库。
    - 单文件导入也统一走该接口，避免两套逻辑不一致。
    
    支持格式：
    - .hdf5 / .h5 → HDF5
    - .mcap → MCAP
    - .zip（含 lerobot 或任意 .zip）→ LeRobot
    
    行为：
    1) 校验：只接受上述后缀，其余拒绝并返回失败原因；仅对 HDF5 做 h5py 校验。
    2) 存储：将上传的文件保存到后端固定目录（backend/storage/hdf5/）
    3) 入库：将每个文件写入 data_assets 表（format 字段为 hdf5/mcap/lerobot）
    4) 返回结果：逐文件成功/失败列表
    """
    try:
        if not files:
            return ApiResponse(
                ok=False,
                error="请至少上传一个文件"
            )
        if not project or not str(project).strip():
            return ApiResponse(
                ok=False,
                error="请选择所属项目，否则数据不会在项目管理中展示"
            )
        p, werr = await assert_may_write_project_for_data_asset_import(assets_db, current_user, str(project).strip())
        if werr or not p:
            return ApiResponse(ok=False, error=werr or "项目校验失败")
        project_val_id = str(p.id)
        project_val_name = (p.name or project_val_id).strip()

        try:
            storage_dir = get_storage_dir()
        except Exception as e:
            return ApiResponse(
                ok=False,
                error=f"存储目录初始化失败：{str(e)[:200]}"
            )
        
        imported = []
        failed = []
        
        for upload_file in files:
            # 浏览器选文件夹时 filename 可能是相对路径（如 frank_data/episode_0.hdf5），只取最后一段作为文件名
            raw_name = upload_file.filename or "unknown"
            filename = raw_name.replace("\\", "/").split("/")[-1].strip() or "unknown"
            
            # 1. 校验文件扩展名并确定格式（HDF5 / MCAP / LeRobot）
            import_fmt = _import_format_from_filename(filename)
            if not import_fmt:
                failed.append({
                    "name": filename,
                    "reason": "不支持的文件类型：仅支持 .hdf5 / .h5、.mcap、.zip（LeRobot）"
                })
                continue
            
            file_path = None
            try:
                # 2. 按项目名称分子目录保存，目录不存在则自动创建
                file_path = generate_unique_filename(
                    storage_dir,
                    filename,
                    project_subdir=project_val_id,
                )
                
                # 3. 保存上传的文件
                try:
                    content = await upload_file.read()
                    with open(file_path, 'wb') as f:
                        f.write(content)
                except (OSError, IOError) as e:
                    failed.append({
                        "name": filename,
                        "reason": f"保存失败：{str(e)[:200]}"
                    })
                    continue
                
                # 4. 仅对 HDF5 做 h5py 校验；MCAP/LeRobot 只做存在性检查
                if import_fmt == "HDF5":
                    valid, reason = validate_hdf5_file(str(file_path))
                    if not valid:
                        if file_path.exists():
                            try:
                                file_path.unlink()
                            except Exception:
                                pass
                        failed.append({
                            "name": filename,
                            "reason": f"文件损坏：{reason}"
                        })
                        continue
                
                # 5. 获取文件大小
                file_size_bytes = file_path.stat().st_size
                
                # 6. 检查是否已存在（通过 file_path）
                existing = await get_asset_by_file_path(assets_db, str(file_path.absolute()))
                if existing:
                    if file_path.exists():
                        try:
                            file_path.unlink()
                        except:
                            pass
                    failed.append({
                        "name": filename,
                        "reason": "文件已存在，不能重复导入"
                    })
                    continue

                # 7. 入库（写入数据资产表 data_assets）
                try:
                    fmt_lower = import_fmt.lower()
                    if fmt_lower == "lerobot":
                        fmt_lower = "lerobot"
                    elif fmt_lower == "mcap":
                        fmt_lower = "mcap"
                    else:
                        fmt_lower = "hdf5"

                    code = await next_asset_code(assets_db)
                    create_data = DataAssetCreate(
                        code=code,
                        filename=file_path.name,
                        format=fmt_lower,
                        source="import",
                        project_id=project_val_id,
                        project_name=project_val_name,
                        file_path=str(file_path.absolute()),
                        file_size_bytes=file_size_bytes,
                        meta=None,
                        parse_status="解析中",
                        error_msg=None,
                        operator_name=(getattr(current_user, "username", None) or "").strip() or None,
                    )
                    asset = await create_asset(assets_db, create_data)

                    meta_json, parse_status, err_msg = parse_meta_for_asset(str(file_path.absolute()), fmt_lower)
                    await update_asset(
                        assets_db,
                        asset.id,
                        parse_status=parse_status,
                        error_msg=err_msg,
                        meta=meta_json,
                    )
                    imported.append({
                        "name": file_path.name,
                        "id": asset.id
                    })
                except Exception as e:
                    # 入库失败，删除已保存的文件
                    if file_path.exists():
                        try:
                            file_path.unlink()
                        except:
                            pass
                    failed.append({
                        "name": filename,
                        "reason": f"入库失败：{str(e)[:200]}"
                    })
                
            except Exception as e:
                # 其他异常
                if file_path and file_path.exists():
                    try:
                        file_path.unlink()
                    except:
                        pass
                failed.append({
                    "name": filename,
                    "reason": f"导入失败：{str(e)[:200]}"
                })
        
        return ApiResponse(
            ok=True,
            data={
                "imported": imported,
                "failed": failed
            }
        )
    except Exception as e:
        # 全局异常处理，确保返回 JSON 格式
        import traceback
        error_detail = str(e)[:500]
        traceback.print_exc()  # 打印到控制台用于调试
        return ApiResponse(
            ok=False,
            error=f"导入过程发生错误：{error_detail}"
        )
