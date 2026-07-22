"""
项目表（PostgreSQL，与数据资产、标注任务同库）
data_assets.project_id、label_tasks.project_id 关联本表 id。

文件名为 project_asset 表示与「数据资产域」同库同 Base，区别于主库 jobs 等模型。
若后续要整理目录，可考虑迁入专用包并统一命名，需同步修改全库 import。
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, Index, UniqueConstraint
from sqlalchemy.sql import func

from app.models.data_asset import Base


class Project(Base):
    """项目表：项目名称、描述、标签；用于聚合展示项目下任务与数据"""
    __tablename__ = "projects"

    id = Column(String(128), primary_key=True, comment="项目业务 ID，如 UUID")
    name = Column(String(256), nullable=False, comment="项目名称")
    description = Column(Text, nullable=True, comment="项目描述")
    tags = Column(Text, nullable=True, comment="标签，JSON 数组字符串，如 [\"tag1\",\"tag2\"]")
    status = Column(String(32), nullable=False, default="进行中", comment="进行中 | 已暂停 | 已归档")
    owner_id = Column(String(128), nullable=True, comment="创建者/负责人 ID")
    team_id = Column(String(128), nullable=True, comment="所属团队 ID（teams.id，可选）")
    created_at = Column(DateTime, server_default=func.now(), nullable=False, comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")

    __table_args__ = (
        Index("idx_projects_status", "status"),
        Index("idx_projects_updated", "updated_at"),
        Index("idx_projects_team_id", "team_id"),
    )


class ProjectMember(Base):
    """项目成员关系表：用于项目级可见性与权限过滤（与 projects/data_assets 同库）。"""

    __tablename__ = "project_members"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(128), nullable=False, comment="项目 ID")
    user_id = Column(String(36), nullable=False, comment="用户 ID（主库 users.id）")
    created_at = Column(DateTime, server_default=func.now(), nullable=False, comment="创建时间")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False, comment="更新时间")

    __table_args__ = (
        UniqueConstraint("project_id", "user_id", name="uq_project_members_project_user"),
        Index("idx_project_members_project", "project_id"),
        Index("idx_project_members_user", "user_id"),
        Index("idx_project_members_updated", "updated_at"),
    )
